from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors


def _dist_ready() -> bool:
    return dist.is_available() and dist.is_initialized()


def _world_size() -> int:
    return dist.get_world_size() if _dist_ready() else 1


def _broadcast_parameters_and_buffers(module: nn.Module) -> None:
    if not _dist_ready():
        return

    for param in module.parameters():
        dist.broadcast(param.data, src=0)

    for buffer in module.buffers():
        dist.broadcast(buffer.data, src=0)


class DDPNaiveIndividual(nn.Module):
    """
    Minimal DDP: wait until backward is fully done, then all-reduce
    each parameter gradient one-by-one.
    """

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self.world_size = _world_size()
        _broadcast_parameters_and_buffers(self.module)
        self.communication_time = 0.0

    def forward(self, *args: Any, **kwargs: Any):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self):
        if not _dist_ready():
            return

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        import time
        start = time.perf_counter()

        for param in self.module.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=False)
                param.grad.div_(self.world_size)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        self.communication_time += time.perf_counter() - start


class DDPNaiveFlat(nn.Module):
    """
    Minimal DDP with one flat all-reduce after backward.
    """

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self.world_size = _world_size()
        _broadcast_parameters_and_buffers(self.module)
        self.communication_time = 0.0

    def forward(self, *args: Any, **kwargs: Any):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self):
        if not _dist_ready():
            return

        grads = [p.grad for p in self.module.parameters() if p.grad is not None]
        if not grads:
            return

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        import time
        start = time.perf_counter()

        flat = _flatten_dense_tensors(grads)
        dist.all_reduce(flat, op=dist.ReduceOp.SUM, async_op=False)
        flat.div_(self.world_size)

        synced_grads = _unflatten_dense_tensors(flat, grads)
        for grad, synced in zip(grads, synced_grads):
            grad.copy_(synced)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        self.communication_time += time.perf_counter() - start


class DDPIndividualParameters(nn.Module):
    """
    DDP with async individual all-reduce hooks.
    Each parameter gradient is communicated as soon as it is ready.
    """

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self.world_size = _world_size()
        self.handles = []

        _broadcast_parameters_and_buffers(self.module)
        self._register_hooks()

    def _register_hooks(self):
        if not _dist_ready():
            return

        for param in self.module.parameters():
            if not param.requires_grad:
                continue

            def hook(_param_from_hook, p=param):
                if p.grad is None:
                    return

                handle = dist.all_reduce(
                    p.grad,
                    op=dist.ReduceOp.SUM,
                    async_op=True,
                )
                self.handles.append(handle)

            param.register_post_accumulate_grad_hook(hook)

    def forward(self, *args: Any, **kwargs: Any):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self):
        for handle in self.handles:
            handle.wait()

        if self.world_size > 1:
            for param in self.module.parameters():
                if param.grad is not None:
                    param.grad.div_(self.world_size)

        self.handles.clear()


class DDPBucketed(nn.Module):
    """
    Bucketed DDP.
    Gradients are grouped into buckets, and each bucket is all-reduced
    asynchronously once all gradients inside that bucket are ready.
    """

    def __init__(self, module: nn.Module, bucket_size_mb: float | None = None):
        super().__init__()
        self.module = module
        self.world_size = _world_size()
        self.bucket_size_mb = bucket_size_mb
        self.bucket_size_bytes = float("inf") if bucket_size_mb is None else bucket_size_mb * 1024 * 1024

        self.buckets: list[list[torch.nn.Parameter]] = []
        self.param_to_bucket: dict[int, int] = {}
        self.bucket_ready_counts: list[int] = []
        self.bucket_handles: dict[int, Any] = {}
        self.bucket_flats: dict[int, torch.Tensor] = {}

        _broadcast_parameters_and_buffers(self.module)
        self._build_buckets()
        self._register_hooks()

    def _build_buckets(self):
        params = [p for p in self.module.parameters() if p.requires_grad]

        current_bucket = []
        current_size = 0

        # Reverse order is recommended because gradients become ready
        # roughly in reverse parameter order during backward.
        for param in reversed(params):
            param_size = param.numel() * param.element_size()

            if current_bucket and current_size + param_size > self.bucket_size_bytes:
                self.buckets.append(current_bucket)
                current_bucket = []
                current_size = 0

            current_bucket.append(param)
            current_size += param_size

        if current_bucket:
            self.buckets.append(current_bucket)

        for bucket_idx, bucket in enumerate(self.buckets):
            for param in bucket:
                self.param_to_bucket[id(param)] = bucket_idx

        self.bucket_ready_counts = [0 for _ in self.buckets]

    def _register_hooks(self):
        if not _dist_ready():
            return

        for param in self.module.parameters():
            if not param.requires_grad:
                continue

            def hook(_param_from_hook, p=param):
                if p.grad is None:
                    return

                bucket_idx = self.param_to_bucket[id(p)]
                self.bucket_ready_counts[bucket_idx] += 1

                bucket = self.buckets[bucket_idx]
                if self.bucket_ready_counts[bucket_idx] == len(bucket):
                    grads = [bucket_param.grad for bucket_param in bucket]
                    flat = _flatten_dense_tensors(grads)
                    handle = dist.all_reduce(flat, op=dist.ReduceOp.SUM, async_op=True)

                    self.bucket_flats[bucket_idx] = flat
                    self.bucket_handles[bucket_idx] = handle

            param.register_post_accumulate_grad_hook(hook)

    def train_batch_start(self):
        self.bucket_ready_counts = [0 for _ in self.buckets]
        self.bucket_handles.clear()
        self.bucket_flats.clear()

    def forward(self, *args: Any, **kwargs: Any):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self):
        for bucket_idx, handle in self.bucket_handles.items():
            handle.wait()

            bucket = self.buckets[bucket_idx]
            grads = [param.grad for param in bucket]
            flat = self.bucket_flats[bucket_idx]

            flat.div_(self.world_size)

            synced_grads = _unflatten_dense_tensors(flat, grads)
            for grad, synced in zip(grads, synced_grads):
                grad.copy_(synced)

        self.bucket_handles.clear()
        self.bucket_flats.clear()
