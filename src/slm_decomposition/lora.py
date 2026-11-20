"""Float32 low-rank updates to selected projections of a frozen text decoder."""

from contextlib import contextmanager
import math

import torch
from torch import nn
from torch.nn import functional as F


class TaskLoRA(nn.Module):
    def __init__(self, base, rank, alpha):
        super().__init__()
        if rank < 1 or alpha <= 0:
            raise ValueError("rank and alpha must be positive")
        self.base = base.requires_grad_(False)
        self.a = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device,
                                          dtype=torch.float32))
        self.b = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device,
                                          dtype=torch.float32))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))
        self.scale = alpha / rank
        self.enabled = True

    def forward(self, inputs):
        output = self.base(inputs)
        if self.enabled:
            delta = F.linear(F.linear(inputs.float(), self.a), self.b)
            output = output + (delta * self.scale).to(output.dtype)
        return output


def attach(model, spec):
    """Install the shared adapter once. Keep registered names for checkpoint compatibility."""
    if any(isinstance(m, TaskLoRA) for m in model.lm.modules()):
        raise ValueError("The decoder already has an adapter")
    model.requires_grad_(False)
    projections = [(name, module) for name, module in model.lm.named_modules()
                   if isinstance(module, nn.Linear) and name.rsplit(".", 1)[-1] in spec["targets"]]
    if not projections:
        raise ValueError("The target projections were not found in the model")
    for name, module in projections:
        parent, _, key = name.rpartition(".")
        adapter = TaskLoRA(module, spec["rank"], spec["alpha"])
        setattr(model.lm.get_submodule(parent), key, adapter)
    return {
        "targets": [name for name, _ in projections],
        "rank": spec["rank"],
        "alpha": spec["alpha"],
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }


def adapter_state(model):
    return {name: p.detach().cpu().clone() for name, p in model.named_parameters() if p.requires_grad}


@torch.no_grad()
def restore(model, state):
    params = {name: p for name, p in model.named_parameters() if p.requires_grad}
    if params.keys() != state.keys() or any(p.shape != state[key].shape for key, p in params.items()):
        raise ValueError("Checkpoint projection names or shapes do not match this model")
    for key, p in params.items():
        p.copy_(state[key])


@contextmanager
def disabled(model):
    modules = [m for m in model.modules() if isinstance(m, TaskLoRA)]
    previous = [m.enabled for m in modules]
    try:
        for m in modules:
            m.enabled = False
        yield
    finally:
        for module, enabled in zip(modules, previous):
            module.enabled = enabled
