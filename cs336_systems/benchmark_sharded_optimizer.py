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
from cs336_systems.ddp import DDPBucketed
from cs336_systems.sharded_optimizer import ShardedOptimizer


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


def memory_allocated_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    sync_if_needed(device)
    return torch.cuda.memory_allocated(device) / (1024 ** 2)


def peak_memory_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    sync_if_needed(device)
    return torch.cuda.max_memory_allocated(device) / (1024 ** 2)


def make_optimizer(params, optimizer_kind: str, lr: float):
    if optimizer_kind == "unsharded":
        return torch.optim.AdamW(params, lr=lr)

    if optimizer_kind == "sharded":
        return ShardedOptimizer(params, torch.optim.AdamW, lr=lr)

    raise ValueError(f"Unknown optimizer kind: {optimizer_kind}")


def run_training_step(ddp_model, optimizer, x, y, device, precision: str):
    if hasattr(ddp_model, "train_batch_start"):
        ddp_model.train_batch_start()

    optimizer.zero_grad(set_to_none=True)

    loss_args = ArgsForLoss(precision=precision)
    loss = compute_loss(ddp_model, x, y, device, loss_args)
    loss.backward()

    ddp_model.finish_gradient_synchronization()

    optimizer.step()
    sync_if_needed(device)

    return float(loss.detach().cpu())


def worker(
    rank: int,
    world_size: int,
    backend: str,
    device_type: str,
    optimizer_kind: str,
    model_size: str,
    context_length: int,
    global_batch_size: int,
    vocab_size: int,
    precision: str,
    bucket_size_mb: float,
    lr: float,
    warmup_steps: int,
    measure_steps: int,
    master_port: str,
    output: str,
):
    setup_process_group(rank, world_size, backend, master_port)

    if device_type == "cuda":
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device("cpu")

    torch.manual_seed(2026 + rank)

    if global_batch_size % world_size != 0:
        raise ValueError("global_batch_size must divide evenly by world_size.")

    local_batch_size = global_batch_size // world_size

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    base_model = instantiate_transformer(
        model_size=model_size,
        context_length=context_length,
        vocab_size=vocab_size,
        device=device,
    )
    base_model.train()

    ddp_model = DDPBucketed(base_model, bucket_size_mb=bucket_size_mb)

    after_model_init_mb = memory_allocated_mb(device)

    optimizer = make_optimizer(
        ddp_model.parameters(),
        optimizer_kind=optimizer_kind,
        lr=lr,
    )

    x, y = make_batch(
        batch_size=local_batch_size,
        context_length=context_length,
        vocab_size=vocab_size,
        device=device,
    )

    # Memory accounting pass: capture memory before and after optimizer step.
    if hasattr(ddp_model, "train_batch_start"):
        ddp_model.train_batch_start()

    optimizer.zero_grad(set_to_none=True)

    loss_args = ArgsForLoss(precision=precision)
    loss = compute_loss(ddp_model, x, y, device, loss_args)
    loss.backward()
    ddp_model.finish_gradient_synchronization()
    sync_if_needed(device)

    before_optimizer_step_mb = memory_allocated_mb(device)

    optimizer.step()
    sync_if_needed(device)

    after_optimizer_step_mb = memory_allocated_mb(device)
    first_step_peak_mb = peak_memory_mb(device)

    # Timing pass.
    for _ in range(warmup_steps):
        run_training_step(ddp_model, optimizer, x, y, device, precision)

    step_times = []
    losses = []

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for _ in range(measure_steps):
        sync_if_needed(device)
        start = timeit.default_timer()

        loss_value = run_training_step(ddp_model, optimizer, x, y, device, precision)

        sync_if_needed(device)
        end = timeit.default_timer()

        step_times.append(end - start)
        losses.append(loss_value)

    timing_peak_mb = peak_memory_mb(device)

    row = {
        "rank": rank,
        "world_size": world_size,
        "backend": backend,
        "device": device_type,
        "optimizer_kind": optimizer_kind,
        "model_size": model_size,
        "context_length": context_length,
        "global_batch_size": global_batch_size,
        "local_batch_size": local_batch_size,
        "precision": precision,
        "bucket_size_mb": bucket_size_mb,
        "after_model_init_mb": after_model_init_mb,
        "before_optimizer_step_mb": before_optimizer_step_mb,
        "after_optimizer_step_mb": after_optimizer_step_mb,
        "first_step_peak_mb": first_step_peak_mb,
        "timing_peak_mb": timing_peak_mb,
        "warmup_steps": warmup_steps,
        "measure_steps": measure_steps,
        "mean_step_seconds": statistics.mean(step_times),
        "std_step_seconds": statistics.stdev(step_times) if len(step_times) > 1 else 0.0,
        "last_loss": losses[-1] if losses else "",
    }

    gathered = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, row)

    if rank == 0:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with out_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(gathered[0].keys()))
            writer.writeheader()
            writer.writerows(gathered)

        mean_step = statistics.mean(float(r["mean_step_seconds"]) for r in gathered)
        mean_after_init = statistics.mean(float(r["after_model_init_mb"]) for r in gathered)
        mean_before_step = statistics.mean(float(r["before_optimizer_step_mb"]) for r in gathered)
        mean_after_step = statistics.mean(float(r["after_optimizer_step_mb"]) for r in gathered)

        print(f"Wrote optimizer benchmark to {out_path}")
        print(f"optimizer_kind={optimizer_kind}")
        print(f"mean_step_seconds_across_ranks={mean_step:.6f}")
        print(f"mean_after_model_init_mb={mean_after_init:.2f}")
        print(f"mean_before_optimizer_step_mb={mean_before_step:.2f}")
        print(f"mean_after_optimizer_step_mb={mean_after_step:.2f}")

    cleanup_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer-kind", choices=["unsharded", "sharded"], required=True)
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--backend", choices=["gloo", "nccl"], default="gloo")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--model-size", type=str, default="small")
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--global-batch-size", type=int, default=4)
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="fp32")
    parser.add_argument("--bucket-size-mb", type=float, default=100.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measure-steps", type=int, default=10)
    parser.add_argument("--master-port", type=str, default="29651")
    parser.add_argument("--output", type=str, default="results/sharded_optimizer.csv")
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
            args.optimizer_kind,
            args.model_size,
            args.context_length,
            args.global_batch_size,
            args.vocab_size,
            args.precision,
            args.bucket_size_mb,
            args.lr,
            args.warmup_steps,
            args.measure_steps,
            args.master_port,
            args.output,
        ),
        nprocs=args.world_size,
        join=True,
    )


if __name__ == "__main__":
    main()
