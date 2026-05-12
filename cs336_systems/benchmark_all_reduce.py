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


def setup_process_group(rank: int, world_size: int, backend: str, master_port: str):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = master_port
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)


def cleanup_process_group():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def sync_if_needed(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def make_tensor(size_mb: int, device: torch.device) -> torch.Tensor:
    # float32 = 4 bytes per value
    num_elements = (size_mb * 1024 * 1024) // 4
    return torch.empty(num_elements, dtype=torch.float32, device=device)


def benchmark_worker(
    rank: int,
    world_size: int,
    backend: str,
    device_type: str,
    sizes_mb: list[int],
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

    local_rows = []

    for size_mb in sizes_mb:
        tensor = make_tensor(size_mb, device)

        # Warmup
        for _ in range(warmup_steps):
            tensor.fill_(rank + 1)
            sync_if_needed(device)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM, async_op=False)
            sync_if_needed(device)

        timings = []

        for _ in range(measure_steps):
            tensor.fill_(rank + 1)
            sync_if_needed(device)

            start = timeit.default_timer()
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM, async_op=False)
            sync_if_needed(device)
            end = timeit.default_timer()

            timings.append(end - start)

        row = {
            "rank": rank,
            "world_size": world_size,
            "backend": backend,
            "device": device_type,
            "size_mb": size_mb,
            "warmup_steps": warmup_steps,
            "measure_steps": measure_steps,
            "mean_seconds": statistics.mean(timings),
            "std_seconds": statistics.stdev(timings) if len(timings) > 1 else 0.0,
            "min_seconds": min(timings),
            "max_seconds": max(timings),
        }

        local_rows.append(row)

    gathered_rows = [None for _ in range(world_size)]
    dist.all_gather_object(gathered_rows, local_rows)

    if rank == 0:
        all_rows = []
        for rank_rows in gathered_rows:
            all_rows.extend(rank_rows)

        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with out_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)

        print(f"Wrote all-reduce results to {out_path}")

        # Print a simple summary averaged over ranks.
        print("\nSummary averaged over ranks:")
        grouped = {}
        for row in all_rows:
            key = (row["world_size"], row["backend"], row["device"], row["size_mb"])
            grouped.setdefault(key, []).append(float(row["mean_seconds"]))

        for key, vals in grouped.items():
            ws, be, dev, size = key
            print(
                f"world_size={ws}, backend={be}, device={dev}, "
                f"size={size}MB, mean_across_ranks={statistics.mean(vals):.6f}s"
            )

    cleanup_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--backend", choices=["gloo", "nccl"], default="gloo")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--sizes-mb", type=int, nargs="+", default=[1, 10, 100, 1024])
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measure-steps", type=int, default=10)
    parser.add_argument("--master-port", type=str, default="29531")
    parser.add_argument("--output", type=str, default="results/all_reduce.csv")
    args = parser.parse_args()

    if args.backend == "nccl" and args.device != "cuda":
        raise ValueError("NCCL should be used with --device cuda.")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but this PyTorch install does not have CUDA enabled. "
            "Run the CUDA/NCCL benchmark on Koa/GCP, not on your Mac."
        )

    if args.device == "cuda" and torch.cuda.device_count() < args.world_size:
        raise RuntimeError(
            f"Requested world_size={args.world_size}, but only "
            f"{torch.cuda.device_count()} CUDA devices are available."
        )

    mp.spawn(
        benchmark_worker,
        args=(
            args.world_size,
            args.backend,
            args.device,
            args.sizes_mb,
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
