from __future__ import annotations

import argparse
import csv
import math
import timeit
from pathlib import Path

import torch


def synchronize_if_needed(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def reset_memory_if_needed(device: torch.device):
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def get_memory_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return torch.cuda.memory_allocated() / (1024 ** 2)


def normal_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    q, k, v shape: (batch, seq_len, d_model)
    No multihead dimension for this assignment section.
    """
    d = q.shape[-1]
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d)
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v)


def make_inputs(batch_size: int, seq_len: int, d_model: int, device: torch.device, dtype: torch.dtype):
    q = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype, requires_grad=True)
    grad_out = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype)
    return q, k, v, grad_out


def time_forward(attn_fn, q, k, v, warmup_steps: int, measure_steps: int, device: torch.device):
    for _ in range(warmup_steps):
        out = attn_fn(q, k, v)
        synchronize_if_needed(device)

    times = []
    for _ in range(measure_steps):
        start = timeit.default_timer()
        out = attn_fn(q, k, v)
        synchronize_if_needed(device)
        end = timeit.default_timer()
        times.append(end - start)

    return sum(times) / len(times)


def time_backward(attn_fn, q, k, v, grad_out, warmup_steps: int, measure_steps: int, device: torch.device):
    memory_before_backward_mb = 0.0

    for _ in range(warmup_steps):
        if q.grad is not None:
            q.grad = None
        if k.grad is not None:
            k.grad = None
        if v.grad is not None:
            v.grad = None

        out = attn_fn(q, k, v)
        synchronize_if_needed(device)
        out.backward(grad_out)
        synchronize_if_needed(device)

    times = []
    for i in range(measure_steps):
        if q.grad is not None:
            q.grad = None
        if k.grad is not None:
            k.grad = None
        if v.grad is not None:
            v.grad = None

        out = attn_fn(q, k, v)
        synchronize_if_needed(device)

        if i == 0:
            memory_before_backward_mb = get_memory_mb(device)

        start = timeit.default_timer()
        out.backward(grad_out)
        synchronize_if_needed(device)
        end = timeit.default_timer()

        times.append(end - start)

    return sum(times) / len(times), memory_before_backward_mb


def benchmark_one(seq_len: int, d_model: int, compiled: bool, args, device: torch.device):
    dtype = torch.float32
    if args.dtype == "bf16":
        dtype = torch.bfloat16
    elif args.dtype == "fp16":
        dtype = torch.float16

    q, k, v, grad_out = make_inputs(
        batch_size=args.batch_size,
        seq_len=seq_len,
        d_model=d_model,
        device=device,
        dtype=dtype,
    )

    attn_fn = normal_attention
    impl_name = "compiled" if compiled else "eager"

    if compiled:
        attn_fn = torch.compile(normal_attention)

    reset_memory_if_needed(device)

    forward_seconds = time_forward(
        attn_fn=attn_fn,
        q=q,
        k=k,
        v=v,
        warmup_steps=args.warmup_steps,
        measure_steps=args.measure_steps,
        device=device,
    )

    backward_seconds, memory_before_backward_mb = time_backward(
        attn_fn=attn_fn,
        q=q,
        k=k,
        v=v,
        grad_out=grad_out,
        warmup_steps=args.warmup_steps,
        measure_steps=args.measure_steps,
        device=device,
    )

    peak_memory_mb = 0.0
    if device.type == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    return {
        "implementation": impl_name,
        "batch_size": args.batch_size,
        "seq_len": seq_len,
        "d_model": d_model,
        "dtype": args.dtype,
        "forward_seconds": forward_seconds,
        "backward_seconds": backward_seconds,
        "memory_before_backward_mb": memory_before_backward_mb,
        "peak_memory_mb": peak_memory_mb,
        "status": "ok",
        "error": "",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[256, 1024, 4096, 8192, 16384])
    parser.add_argument("--d-models", type=int, nargs="+", default=[16, 32, 64, 128])
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measure-steps", type=int, default=100)
    parser.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default="fp32")
    parser.add_argument("--include-compiled", action="store_true")
    parser.add_argument("--output", type=str, default="results/attention_benchmark.csv")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"device={device}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    rows = []
    implementations = [False]
    if args.include_compiled:
        implementations.append(True)

    for d_model in args.d_models:
        for seq_len in args.seq_lens:
            for compiled in implementations:
                label = "compiled" if compiled else "eager"
                print(f"Running {label}: seq_len={seq_len}, d_model={d_model}")

                try:
                    row = benchmark_one(seq_len, d_model, compiled, args, device)
                except torch.cuda.OutOfMemoryError as exc:
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    row = {
                        "implementation": label,
                        "batch_size": args.batch_size,
                        "seq_len": seq_len,
                        "d_model": d_model,
                        "dtype": args.dtype,
                        "forward_seconds": "",
                        "backward_seconds": "",
                        "memory_before_backward_mb": "",
                        "peak_memory_mb": "",
                        "status": "oom",
                        "error": str(exc).replace("\n", " ")[:300],
                    }
                except Exception as exc:
                    row = {
                        "implementation": label,
                        "batch_size": args.batch_size,
                        "seq_len": seq_len,
                        "d_model": d_model,
                        "dtype": args.dtype,
                        "forward_seconds": "",
                        "backward_seconds": "",
                        "memory_before_backward_mb": "",
                        "peak_memory_mb": "",
                        "status": "error",
                        "error": repr(exc).replace("\n", " ")[:300],
                    }

                rows.append(row)

                with open(args.output, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                    writer.writeheader()
                    writer.writerows(rows)

                print(row)

    print(f"Wrote results to {args.output}")


if __name__ == "__main__":
    main()
