from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import torch

from cs336_systems.flash_attention import FlashAttention2PyTorch, FlashAttention2Triton


def normal_attention(q, k, v, is_causal: bool):
    d = q.shape[-1]
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d)

    if is_causal:
        n_queries = q.shape[-2]
        n_keys = k.shape[-2]
        q_idx = torch.arange(n_queries, device=q.device)
        k_idx = torch.arange(n_keys, device=q.device)
        mask = q_idx[:, None] >= k_idx[None, :]
        scores = torch.where(mask, scores, torch.tensor(-1e6, device=q.device, dtype=scores.dtype))

    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v)


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def make_inputs(batch_size, seq_len, d_model, dtype, device):
    q = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype, requires_grad=True)
    grad = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype)
    return q, k, v, grad


def bench_ms(fn, warmup_ms=100, rep_ms=300):
    import triton

    return triton.testing.do_bench(fn, warmup=warmup_ms, rep=rep_ms)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[128, 256, 512, 1024])
    parser.add_argument("--d-models", type=int, nargs="+", default=[16, 32, 64, 128])
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--output", type=str, default="results/flash_attention_benchmark.csv")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but this PyTorch install does not have CUDA enabled. "
            "Run this on Koa/GCP, or use --device cpu for local testing."
        )

    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    rows = []

    implementations = [
        ("pytorch_regular", lambda q, k, v: normal_attention(q, k, v, args.causal)),
        ("flash_pytorch", lambda q, k, v: FlashAttention2PyTorch.apply(q, k, v, args.causal)),
    ]

    if device.type == "cuda":
        implementations.append(
            ("flash_triton", lambda q, k, v: FlashAttention2Triton.apply(q, k, v, args.causal))
        )

    for d_model in args.d_models:
        for seq_len in args.seq_lens:
            for name, fn in implementations:
                print(f"Running {name}: seq_len={seq_len}, d_model={d_model}, dtype={args.dtype}")

                try:
                    q, k, v, grad = make_inputs(args.batch_size, seq_len, d_model, dtype, device)

                    def forward_only():
                        out = fn(q, k, v)
                        sync(device)

                    def backward_only():
                        q.grad = None
                        k.grad = None
                        v.grad = None
                        out = fn(q, k, v)
                        sync(device)
                        out.backward(grad)
                        sync(device)

                    if device.type == "cuda":
                        fwd_ms = bench_ms(forward_only)
                        bwd_ms = bench_ms(backward_only)
                    else:
                        # CPU fallback timing for smoke testing only.
                        import timeit
                        fwd_ms = timeit.timeit(forward_only, number=1) * 1000
                        bwd_ms = timeit.timeit(backward_only, number=1) * 1000

                    row = {
                        "implementation": name,
                        "batch_size": args.batch_size,
                        "seq_len": seq_len,
                        "d_model": d_model,
                        "dtype": args.dtype,
                        "causal": args.causal,
                        "forward_ms": fwd_ms,
                        "backward_ms": bwd_ms,
                        "status": "ok",
                        "error": "",
                    }

                except torch.cuda.OutOfMemoryError as exc:
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    row = {
                        "implementation": name,
                        "batch_size": args.batch_size,
                        "seq_len": seq_len,
                        "d_model": d_model,
                        "dtype": args.dtype,
                        "causal": args.causal,
                        "forward_ms": "",
                        "backward_ms": "",
                        "status": "oom",
                        "error": str(exc).replace("\n", " ")[:300],
                    }

                except Exception as exc:
                    row = {
                        "implementation": name,
                        "batch_size": args.batch_size,
                        "seq_len": seq_len,
                        "d_model": d_model,
                        "dtype": args.dtype,
                        "causal": args.causal,
                        "forward_ms": "",
                        "backward_ms": "",
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
