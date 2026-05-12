from __future__ import annotations

import argparse
import csv
import os
import statistics
import timeit
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from cs336_systems.benchmark_transformer import (
    compute_loss,
    instantiate_transformer,
    make_batch,
)
from cs336_systems.ddp import (
    DDPBucketed,
    DDPIndividualParameters,
    DDPNaiveFlat,
    DDPNaiveIndividual,
)


class ArgsForLoss:
    def __init__(self, precision: str):
        self.precision = precision


def setup_process_group(rank: int, world_size: int, backend: str, master_port: str):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = master_port
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)


def cleanup_process_group():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def sync_if_needed(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def make_ddp_model(model: torch.nn.Module, implementation: str, bucket_size_mb: float | None):
    if implementation == "naive_individual":
        return DDPNaiveIndividual(model)

    if implementation == "naive_flat":
        return DDPNaiveFlat(model)

    if implementation == "overlap_individual":
        return DDPIndividualParameters(model)

    if implementation == "bucketed":
        return DDPBucketed(model, bucket_size_mb=bucket_size_mb)

    raise ValueError(f"Unknown DDP implementation: {implementation}")


def run_train_step(ddp_model, optimizer, x, y, device, precision: str):
    if hasattr(ddp_model, "train_batch_start"):
        ddp_model.train_batch_start()

    optimizer.zero_grad(set_to_none=True)

    loss_args = ArgsForLoss(precision=precision)
    loss = compute_loss(ddp_model, x, y, device, loss_args)
    loss.backward()

    sync_start = timeit.default_timer()
    ddp_model.finish_gradient_synchronization()
    sync_if_needed(device)
    sync_end = timeit.default_timer()

    optimizer.step()
    sync_if_needed(device)

    return float(loss.detach().cpu()), sync_end - sync_start


def worker(
    rank: int,
    world_size: int,
    backend: str,
    device_type: str,
    implementation: str,
    model_size: str,
    context_length: int,
    global_batch_size: int,
    vocab_size: int,
    precision: str,
    warmup_steps: int,
    measure_steps: int,
    bucket_size_mb: float | None,
    master_port: str,
    output: str,
):
    setup_process_group(rank, world_size, backend, master_port)

    if device_type == "cuda":
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device("cpu")

    torch.manual_seed(1234 + rank)

    if global_batch_size % world_size != 0:
        raise ValueError("global_batch_size must divide evenly by world_size.")

    local_batch_size = global_batch_size // world_size

    base_model = instantiate_transformer(
        model_size=model_size,
        context_length=context_length,
        vocab_size=vocab_size,
        device=device,
    )
    base_model.train()

    ddp_model = make_ddp_model(
        model=base_model,
        implementation=implementation,
        bucket_size_mb=bucket_size_mb,
    )

    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=1e-4)

    # Each rank gets its own local shard of a global batch.
    x, y = make_batch(
        batch_size=local_batch_size,
        context_length=context_length,
        vocab_size=vocab_size,
        device=device,
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # Warmup
    for _ in range(warmup_steps):
        run_train_step(ddp_model, optimizer, x, y, device, precision)
        sync_if_needed(device)

    step_times = []
    sync_times = []
    losses = []

    for _ in range(measure_steps):
        sync_if_needed(device)
        start = timeit.default_timer()

        loss_value, sync_seconds = run_train_step(
            ddp_model=ddp_model,
            optimizer=optimizer,
            x=x,
            y=y,
            device=device,
            precision=precision,
        )

        sync_if_needed(device)
        end = timeit.default_timer()

        step_times.append(end - start)
        sync_times.append(sync_seconds)
        losses.append(loss_value)

    peak_memory_mb = 0.0
    if device.type == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    local_row = {
        "rank": rank,
        "world_size": world_size,
        "backend": backend,
        "device": device_type,
        "implementation": implementation,
        "model_size": model_size,
        "context_length": context_length,
        "global_batch_size": global_batch_size,
        "local_batch_size": local_batch_size,
        "precision": precision,
        "bucket_size_mb": "" if bucket_size_mb is None else bucket_size_mb,
        "warmup_steps": warmup_steps,
        "measure_steps": measure_steps,
        "mean_step_seconds": statistics.mean(step_times),
        "std_step_seconds": statistics.stdev(step_times) if len(step_times) > 1 else 0.0,
        "mean_sync_seconds": statistics.mean(sync_times),
        "std_sync_seconds": statistics.stdev(sync_times) if len(sync_times) > 1 else 0.0,
        "sync_fraction": statistics.mean(sync_times) / statistics.mean(step_times),
        "peak_memory_mb": peak_memory_mb,
        "last_loss": losses[-1],
    }

    gathered = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, local_row)

    if rank == 0:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with out_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(gathered[0].keys()))
            writer.writeheader()
            writer.writerows(gathered)

        mean_step = statistics.mean(float(row["mean_step_seconds"]) for row in gathered)
        mean_sync = statistics.mean(float(row["mean_sync_seconds"]) for row in gathered)

        print(f"Wrote DDP benchmark to {out_path}")
        print(f"implementation={implementation}")
        print(f"mean_step_seconds_across_ranks={mean_step:.6f}")
        print(f"mean_sync_seconds_across_ranks={mean_sync:.6f}")
        print(f"sync_fraction={mean_sync / mean_step:.4f}")

    cleanup_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--implementation",
        choices=["naive_individual", "naive_flat", "overlap_individual", "bucketed"],
        required=True,
    )
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--backend", choices=["gloo", "nccl"], default="gloo")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--model-size", type=str, default="small")
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--global-batch-size", type=int, default=4)
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="fp32")
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measure-steps", type=int, default=10)
    parser.add_argument("--bucket-size-mb", type=float, default=None)
    parser.add_argument("--master-port", type=str, default="29601")
    parser.add_argument("--output", type=str, default="results/ddp_benchmark.csv")
    args = parser.parse_args()

    if args.backend == "nccl" and args.device != "cuda":
        raise ValueError("Use NCCL with --device cuda.")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but this PyTorch install does not have CUDA enabled. "
            "Run this on Koa/GCP, not on your Mac."
        )

    if args.device == "cuda" and torch.cuda.device_count() < args.world_size:
        raise RuntimeError(
            f"Requested world_size={args.world_size}, but only "
            f"{torch.cuda.device_count()} CUDA devices are available."
        )

    mp.spawn(
        worker,
        args=(
            args.world_size,
            args.backend,
            args.device,
            args.implementation,
            args.model_size,
            args.context_length,
            args.global_batch_size,
            args.vocab_size,
            args.precision,
            args.warmup_steps,
            args.measure_steps,
            args.bucket_size_mb,
            args.master_port,
            args.output,
        ),
        nprocs=args.world_size,
        join=True,
    )


if __name__ == "__main__":
    main()
