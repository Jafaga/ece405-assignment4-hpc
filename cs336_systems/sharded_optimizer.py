from __future__ import annotations

from typing import Any, Type

import torch
import torch.distributed as dist


def _dist_ready() -> bool:
    return dist.is_available() and dist.is_initialized()


class ShardedOptimizer(torch.optim.Optimizer):
    """
    Simplified optimizer state sharding.

    Each rank owns approximately 1 / world_size of the model parameters.
    The local optimizer only stores optimizer state for that shard.
    After the local optimizer step, parameters are broadcast from owner ranks.
    """

    def __init__(self, params, optimizer_cls: Type[torch.optim.Optimizer], **kwargs: Any):
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = kwargs

        self.rank = dist.get_rank() if _dist_ready() else 0
        self.world_size = dist.get_world_size() if _dist_ready() else 1

        self._local_optimizer_ready = False
        self.local_optimizer = None
        self.param_owner: dict[int, int] = {}

        # Important: call superclass constructor.
        # This creates self.state and self.param_groups.
        super().__init__(params, defaults=kwargs)

        self._rebuild_local_optimizer()
        self._local_optimizer_ready = True

    def _rebuild_local_optimizer(self):
        local_param_groups = []
        global_param_index = 0
        self.param_owner = {}

        for group in self.param_groups:
            local_params = []

            for param in group["params"]:
                owner = global_param_index % self.world_size
                self.param_owner[id(param)] = owner

                if owner == self.rank:
                    local_params.append(param)

                global_param_index += 1

            if local_params:
                group_copy = {k: v for k, v in group.items() if k != "params"}
                group_copy["params"] = local_params
                local_param_groups.append(group_copy)

        if local_param_groups:
            self.local_optimizer = self.optimizer_cls(local_param_groups, **self.optimizer_kwargs)
        else:
            self.local_optimizer = None

    def add_param_group(self, param_group):
        super().add_param_group(param_group)

        # During superclass initialization, local optimizer is not ready yet.
        if getattr(self, "_local_optimizer_ready", False):
            self._rebuild_local_optimizer()

    def zero_grad(self, set_to_none: bool = True):
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue

                if set_to_none:
                    param.grad = None
                else:
                    param.grad.detach_()
                    param.grad.zero_()

    def step(self, closure=None, **kwargs):
        loss = None

        if self.local_optimizer is not None:
            loss = self.local_optimizer.step(closure=closure, **kwargs)

        if _dist_ready():
            for group in self.param_groups:
                for param in group["params"]:
                    owner = self.param_owner[id(param)]
                    dist.broadcast(param.data, src=owner)

        return loss
