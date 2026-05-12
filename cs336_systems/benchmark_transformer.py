from __future__ import annotations

import argparse
import contextlib
import inspect
import json
import statistics
import timeit
from pathlib import Path

import sys

# Make the local cs336-basics staff package importable.
REPO_ROOT = Path(__file__).resolve().parents[1]
CS336_BASICS_PATH = REPO_ROOT / "cs336-basics"
if CS336_BASICS_PATH.exists():
    sys.path.insert(0, str(CS336_BASICS_PATH))

import torch
import torch.nn.functional as F


MODEL_CONFIGS = {
    "small":  {"d_model": 768,  "d_ff": 3072,  "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096,  "num_layers": 24, "num_heads": 16},
    "large":  {"d_model": 1280, "d_ff": 5120,  "num_layers": 36, "num_heads": 20},
    "xl":     {"d_model": 1600, "d_ff": 6400,  "num_layers": 48, "num_heads": 25},
    "2.7B":   {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
}


def synchronize_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def get_autocast_context(device: torch.device, precision: str):
    if precision == "bf16":
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    if precision == "fp16":
        return torch.autocast(device_type=device.type, dtype=torch.float16)
    return contextlib.nullcontext()


def find_transformer_class():
    """
    The staff solution name can vary slightly, so this tries common class names.
    """
    import cs336_basics.model as model_mod

    candidate_names = [
        "BasicsTransformerLM",
        "TransformerLM",
        "TransformerLanguageModel",
        "Transformer",
    ]

    for name in candidate_names:
        if hasattr(model_mod, name):
            return getattr(model_mod, name)

    available = [name for name in dir(model_mod) if "Transformer" in name or "LM" in name]
    raise RuntimeError(
        "Could not find a Transformer model class in cs336_basics.model. "
        f"Possible names found: {available}"
    )


def instantiate_transformer(model_size: str, context_length: int, vocab_size: int, device: torch.device):
    cls = find_transformer_class()
    cfg = MODEL_CONFIGS[model_size]

    # Common constructor keyword names used in CS336-style TransformerLM code.
    kwargs_candidates = [
        {
            "vocab_size": vocab_size,
            "context_length": context_length,
            "d_model": cfg["d_model"],
            "num_layers": cfg["num_layers"],
            "num_heads": cfg["num_heads"],
            "d_ff": cfg["d_ff"],
            "rope_theta": 10000.0,
        },
        {
            "vocab_size": vocab_size,
            "context_length": context_length,
            "num_layers": cfg["num_layers"],
            "d_model": cfg["d_model"],
            "num_heads": cfg["num_heads"],
            "d_ff": cfg["d_ff"],
        },
        {
            "vocab_size": vocab_size,
            "context_length": context_length,
            "num_layers": cfg["num_layers"],
            "num_heads": cfg["num_heads"],
            "d_model": cfg["d_model"],
            "d_ff": cfg["d_ff"],
        },
    ]

    errors = []
    for kwargs in kwargs_candidates:
        try:
            model = cls(**kwargs)
            return model.to(device)
        except TypeError as exc:
            errors.append(str(exc))

    sig = inspect.signature(cls)
    raise RuntimeError(
        f"Could not instantiate {cls.__name__}. Signature: {sig}. Errors: {errors}"
    )


def make_batch(batch_size: int, context_length: int, vocab_size: int, device: torch.device):
    x = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
    y = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
    return x, y


def compute_loss(model, x, y, device: torch.device, precision: str):
    with get_autocast_context(device, precision):
        logits = model(x)
        if isinstance(logits, tuple):
            logits = logits[0]

        # Expected shape: (batch, seq_len, vocab_size)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            y.reshape(-1),
        )
    return loss


def run_one_step(model, x, y, optimizer, args, device: torch.device):
    if args.mode == "forward":
        with torch.no_grad():
            loss = compute_loss(model, x, y, device, args.precision)
        return float(loss.detach().cpu())

    if args.mode == "forward_backward":
        optimizer.zero_grad(set_to_none=True)
        loss = compute_loss(model, x, y, device, args.precision)
        loss.backward()
        return float(loss.detach().cpu())

    if args.mode == "train_step":
        optimizer.zero_grad(set_to_none=True)
        loss = compute_loss(model, x, y, device, args.precision)
        loss.backward()
        optimizer.step()
        return float(loss.detach().cpu())

    raise ValueError(f"Unknown mode: {args.mode}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-size", choices=list(MODEL_CONFIGS.keys()), default="small")
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measure-steps", type=int, default=10)
    parser.add_argument("--mode", choices=["forward", "forward_backward", "train_step"], default="forward")
    parser.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="fp32")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--memory-profile", action="store_true")
    parser.add_argument("--memory-output", type=str, default="memory_snapshot.pickle")
    parser.add_argument("--output-json", type=str, default="")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    torch.manual_seed(0)

    print(f"device={device}")
    print(f"model_size={args.model_size}, context_length={args.context_length}, mode={args.mode}, precision={args.precision}")

    model = instantiate_transformer(
        model_size=args.model_size,
        context_length=args.context_length,
        vocab_size=args.vocab_size,
        device=device,
    )
    model.train(args.mode != "forward")

    if args.compile:
        model = torch.compile(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    x, y = make_batch(args.batch_size, args.context_length, args.vocab_size, device)

    # Warmup
    for _ in range(args.warmup_steps):
        run_one_step(model, x, y, optimizer, args, device)
        synchronize_if_needed(device)

    if args.memory_profile:
        if device.type != "cuda":
            raise RuntimeError("Memory profiler snapshot requires CUDA.")
        torch.cuda.memory._record_memory_history(max_entries=1000000)

    times = []
    losses = []

    for _ in range(args.measure_steps):
        start = timeit.default_timer()
        loss_value = run_one_step(model, x, y, optimizer, args, device)
        synchronize_if_needed(device)
        end = timeit.default_timer()

        times.append(end - start)
        losses.append(loss_value)

    if args.memory_profile:
        torch.cuda.memory._dump_snapshot(args.memory_output)
        torch.cuda.memory._record_memory_history(enabled=None)
        print(f"Saved memory snapshot to {args.memory_output}")

    mean_time = statistics.mean(times)
    std_time = statistics.stdev(times) if len(times) > 1 else 0.0

    result = {
        "model_size": args.model_size,
        "context_length": args.context_length,
        "batch_size": args.batch_size,
        "vocab_size": args.vocab_size,
        "mode": args.mode,
        "precision": args.precision,
        "compile": args.compile,
        "device": str(device),
        "warmup_steps": args.warmup_steps,
        "measure_steps": args.measure_steps,
        "mean_seconds": mean_time,
        "std_seconds": std_time,
        "last_loss": losses[-1] if losses else None,
    }

    print(json.dumps(result, indent=2))

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2))
        print(f"Wrote result to {out_path}")


if __name__ == "__main__":
    main()
