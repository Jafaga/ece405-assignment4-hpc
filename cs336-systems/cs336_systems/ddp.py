from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn


class DDPIndividualParameters(nn.Module):
    """
    Simple Distributed Data Parallel wrapper.

    It broadcasts parameters from rank 0 during initialization, then registers
    backward hooks so each parameter gradient is asynchronously all-reduced
    as soon as that gradient is ready.
    """

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self.handles = []
        self.world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1

        self._broadcast_parameters()
        self._register_gradient_hooks()

    def _broadcast_parameters(self):
        if not (dist.is_available() and dist.is_initialized()):
            return

        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)

        # Broadcast buffers too, just in case the wrapped model has any.
        for buffer in self.module.buffers():
            dist.broadcast(buffer.data, src=0)

    def _register_gradient_hooks(self):
        if not (dist.is_available() and dist.is_initialized()):
            return

        for param in self.module.parameters():
            if not param.requires_grad:
                continue

            def hook(_unused_grad, p=param):
                if p.grad is None:
                    return
                handle = dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, async_op=True)
                self.handles.append(handle)

            # This hook runs after the gradient has been accumulated into param.grad.
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


class DDPBucketed(DDPIndividualParameters):
    """
    Temporary correctness version for bucketed DDP.

    For now, this reuses the individual-parameter synchronization logic so we can
    pass correctness tests first. Later, we can upgrade this to real bucketed
    communication for the performance/report section.
    """

    def __init__(self, module: nn.Module, bucket_size_mb: float | None = None):
        self.bucket_size_mb = bucket_size_mb
        super().__init__(module)

    def train_batch_start(self):
        # Placeholder for the later true bucketed implementation.
        self.handles.clear()
