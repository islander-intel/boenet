# utils/optim.py
"""
optim.py

Optimizer and LR scheduler utilities for BFSNet experiments.

What you get
------------
1) Parameter grouping with correct weight decay behavior:
   - Decay on "weight" tensors of Linear/Conv/etc.
   - NO decay on biases and normalization layers (BatchNorm/LayerNorm/GroupNorm/InstanceNorm).
2) Optimizer builders:
   - AdamW (default betas/eps can be overridden)
   - SGD (momentum + Nesterov optional)
3) LR schedulers (epoch-wise by default):
   - "cosine": optional warmup_epochs (LinearLR) → CosineAnnealingLR
   - "step":   optional warmup_epochs (LinearLR) → StepLR
   - "none":   no scheduler
4) Logging helpers:
   - current_lr(optimizer): float
   - format_optimizer_summary(optimizer, sched_name, **sched_kwargs): one-line summary

Typical usage (in your trainer)
-------------------------------
from utils.optim import (
    build_param_groups, build_optimizer, build_scheduler,
    current_lr, format_optimizer_summary
)

param_groups = build_param_groups(model, weight_decay=args.weight_decay)
optimizer = build_optimizer(
    "adamw", param_groups,
    lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999), eps=1e-8
)
scheduler = build_scheduler(
    optimizer,
    sched_name=args.lr_sched,        # "cosine", "step", or "none"
    total_epochs=args.epochs,
    warmup_epochs=getattr(args, "warmup_epochs", 0),
    # cosine:
    cosine_eta_min=0.0,
    # step:
    step_size= max(1, args.epochs // 3),
    gamma=0.1,
)

print(format_optimizer_summary(
    optimizer, sched_name=args.lr_sched,
    total_epochs=args.epochs,
    warmup_epochs=getattr(args, "warmup_epochs", 0),
    step_size=max(1, args.epochs // 3), gamma=0.1, cosine_eta_min=0.0
))

for epoch in range(args.epochs):
    # ... train epoch ...
    print(f"[epoch {epoch+1}] lr={current_lr(optimizer):.6f}")
    if scheduler is not None:
        scheduler.step()  # epoch-wise stepping

Notes
-----
- This module is *framework-agnostic* to your training loop; it just returns
  PyTorch optimizer/scheduler instances.
- Schedulers here are designed to be stepped **once per epoch**. If you prefer
  per-iteration stepping, wrap them outside or extend with a batch scheduler.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from torch.optim import Optimizer


# --------------------------- Norm layer registry --------------------------- #

# Layers whose parameters should NOT receive weight decay.
_NORM_TYPES: Tuple[type, ...] = (
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.SyncBatchNorm,
    nn.LayerNorm,
    nn.GroupNorm,
    nn.InstanceNorm1d,
    nn.InstanceNorm2d,
    nn.InstanceNorm3d,
)
# Torch may not have RMSNorm in older versions; ignore if missing.
if hasattr(nn, "RMSNorm"):  # type: ignore[attr-defined]
    _NORM_TYPES = _NORM_TYPES + (nn.RMSNorm,)  # type: ignore


# ------------------------ Parameter grouping logic ------------------------ #

def build_param_groups(
    model: nn.Module,
    weight_decay: float = 0.0,
    *,
    norm_no_decay: bool = True,
    bias_no_decay: bool = True,
) -> List[Dict[str, object]]:
    """
    Split model parameters into decay / no_decay groups.

    Rules:
      - no_decay: biases (name endswith ".bias") and ANY parameter whose owning
                  module is a normalization layer (BatchNorm/LayerNorm/GroupNorm/...).
      - decay:    everything else.

    This matches common AdamW/SGD best practices, avoiding L2 on normalization stats
    and biases.

    Returns
    -------
    A list of param-group dicts suitable for torch.optim:
      [
        {"params": [...], "weight_decay": weight_decay},       # decay
        {"params": [...], "weight_decay": 0.0},                # no_decay
      ]
    """
    decay_params: List[torch.nn.Parameter] = []
    no_decay_params: List[torch.nn.Parameter] = []

    # Map module names to modules for quick lookup of owning module
    module_dict = dict(model.named_modules())

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Heuristic: biases to no_decay
        is_bias = name.endswith(".bias")

        # Find owning module (everything before last '.')
        parent_name = name.rsplit(".", 1)[0] if "." in name else ""
        parent_mod = module_dict.get(parent_name, None)

        is_norm = isinstance(parent_mod, _NORM_TYPES) if parent_mod is not None else False

        if (bias_no_decay and is_bias) or (norm_no_decay and is_norm):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    groups: List[Dict[str, object]] = []
    if decay_params:
        groups.append({"params": decay_params, "weight_decay": float(weight_decay)})
    if no_decay_params:
        groups.append({"params": no_decay_params, "weight_decay": 0.0})
    return groups


# ---------------------------- Optimizer builders --------------------------- #

def build_optimizer(
    name: str,
    param_groups: Union[Iterable[torch.nn.Parameter], List[Dict[str, object]]],
    *,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    # AdamW
    betas: Tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    # SGD
    momentum: float = 0.9,
    nesterov: bool = True,
) -> Optimizer:
    """
    Create an optimizer.

    Parameters
    ----------
    name : {"adamw","sgd"}
    param_groups : model.parameters() or output of build_param_groups()
    lr, weight_decay : learning rate, global weight decay (decay group only)
    betas, eps : AdamW hyperparameters
    momentum, nesterov : SGD hyperparameters

    Returns
    -------
    torch.optim.Optimizer
    """
    name = name.lower().strip()
    if name in ("adamw", "adamw_torch"):
        return torch.optim.AdamW(
            param_groups,
            lr=lr,
            weight_decay=weight_decay,  # per-group wd overrides this when provided
            betas=betas,
            eps=eps,
        )
    if name in ("sgd",):
        return torch.optim.SGD(
            param_groups,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,  # per-group wd overrides this when provided
            nesterov=bool(nesterov),
        )
    raise ValueError(f"Unknown optimizer '{name}'. Use 'adamw' or 'sgd'.")


# ------------------------------ LR schedulers ------------------------------ #

@dataclass
class SchedulerSpec:
    """Simple container describing the scheduler selection and knobs."""
    name: str = "none"            # "none" | "cosine" | "step"
    total_epochs: int = 0
    warmup_epochs: int = 0
    # cosine
    cosine_eta_min: float = 0.0
    # step
    step_size: int = 30
    gamma: float = 0.1


def build_scheduler(
    optimizer: Optimizer,
    *,
    sched_name: str = "none",
    total_epochs: int = 0,
    warmup_epochs: int = 0,
    # cosine
    cosine_eta_min: float = 0.0,
    # step
    step_size: int = 30,
    gamma: float = 0.1,
):
    """
    Build an **epoch-wise** LR scheduler (call `scheduler.step()` once per epoch).

    Schedulers:
      - "none":   returns None
      - "cosine": Linear warmup (warmup_epochs) → CosineAnnealingLR(T_max = total_epochs - warmup)
      - "step":   Linear warmup (warmup_epochs) → StepLR(step_size, gamma)

    Returns
    -------
    torch.optim.lr_scheduler._LRScheduler | None
    """
    sched_name = sched_name.lower().strip()
    if sched_name == "none":
        return None

    # Safety
    total_epochs = int(total_epochs)
    warmup_epochs = max(0, int(warmup_epochs))
    remain = max(1, total_epochs - warmup_epochs) if total_epochs > 0 else 1

    from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingLR, StepLR

    if warmup_epochs > 0:
        warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)
    else:
        warmup = None

    if sched_name == "cosine":
        main = CosineAnnealingLR(optimizer, T_max=remain, eta_min=float(cosine_eta_min))
    elif sched_name == "step":
        main = StepLR(optimizer, step_size=max(1, int(step_size)), gamma=float(gamma))
    else:
        raise ValueError(f"Unknown scheduler '{sched_name}'. Use 'none', 'cosine', or 'step'.")

    if warmup is None:
        return main
    else:
        return SequentialLR(optimizer, schedulers=[warmup, main], milestones=[warmup_epochs])


# ------------------------------ Logging helpers ---------------------------- #

def current_lr(optimizer: Optimizer, group: int = 0) -> float:
    """
    Return the current learning rate of the specified param group (default: group 0).
    If multiple groups exist, you can choose to log max/min across groups in your trainer.
    """
    if not optimizer.param_groups:
        return 0.0
    group = max(0, min(group, len(optimizer.param_groups) - 1))
    return float(optimizer.param_groups[group].get("lr", 0.0))


def format_optimizer_summary(
    optimizer: Optimizer,
    *,
    sched_name: str = "none",
    total_epochs: int = 0,
    warmup_epochs: int = 0,
    cosine_eta_min: float = 0.0,
    step_size: int = 30,
    gamma: float = 0.1,
) -> str:
    """
    Produce a compact, human-friendly one-liner describing the optimizer
    (type, lr, wd, betas/momentum) and the LR schedule.

    Example:
      "opt=AdamW lr=0.0030 wd=0.0005 betas=(0.9,0.999) eps=1e-08 | sched=cosine T=20 warmup=3 eta_min=0.0"
    """
    pg0 = optimizer.param_groups[0] if optimizer.param_groups else {}
    lr = pg0.get("lr", None)
    wd = pg0.get("weight_decay", None)

    if isinstance(optimizer, torch.optim.AdamW):
        betas = optimizer.defaults.get("betas", (0.9, 0.999))
        eps = optimizer.defaults.get("eps", 1e-8)
        opt_str = f"opt=AdamW lr={lr:.6f} wd={wd:.6f} betas=({betas[0]}, {betas[1]}) eps={eps:g}"
    elif isinstance(optimizer, torch.optim.SGD):
        momentum = optimizer.defaults.get("momentum", 0.0)
        nesterov = optimizer.defaults.get("nesterov", False)
        opt_str = f"opt=SGD lr={lr:.6f} wd={wd:.6f} momentum={momentum} nesterov={nesterov}"
    else:
        opt_str = f"opt={optimizer.__class__.__name__} lr={lr} wd={wd}"

    sched_name = sched_name.lower().strip()
    if sched_name == "none":
        sched_str = "sched=none"
    elif sched_name == "cosine":
        sched_str = f"sched=cosine T={max(0, total_epochs)} warmup={max(0, warmup_epochs)} eta_min={cosine_eta_min}"
    elif sched_name == "step":
        sched_str = f"sched=step T={max(0, total_epochs)} warmup={max(0, warmup_epochs)} step_size={step_size} gamma={gamma}"
    else:
        sched_str = f"sched={sched_name}"

    return f"{opt_str} | {sched_str}"


# -------------------------------- Self-test -------------------------------- #

if __name__ == "__main__":
    # Minimal smoke test (does not train)
    model = nn.Sequential(
        nn.Linear(8, 16),
        nn.ReLU(),
        nn.LayerNorm(16),
        nn.Linear(16, 4),
    )

    groups = build_param_groups(model, weight_decay=0.01)
    # Ensure we created two groups (decay / no_decay)
    assert len(groups) in (1, 2), "Expected 1-2 parameter groups."

    opt = build_optimizer("adamw", groups, lr=3e-3, weight_decay=0.01)
    sch = build_scheduler(opt, sched_name="cosine", total_epochs=10, warmup_epochs=2, cosine_eta_min=1e-5)

    print(format_optimizer_summary(opt, sched_name="cosine", total_epochs=10, warmup_epochs=2, cosine_eta_min=1e-5))
    print("lr@start =", current_lr(opt))
    for epoch in range(10):
        if sch is not None:
            sch.step()
        print(f"epoch {epoch+1:02d}: lr={current_lr(opt):.6f}")
