# utils/metrics.py
# -*- coding: utf-8 -*-
"""
metrics.py

Small helpers for BFS training logs, compute accounting, and early-epoch
"pipe alive?" gradient probes.

What you get
------------
1) Pretty-print & padding helpers
   - pretty_list(nums, decimals=3)
   - pad_to_match(a, b)

2) Trace aggregation across batches
   - aggregate_trace_sums(agg, trace)
   - average_trace(agg, denom)

3) Compute accounting
   - estimate_compute_per_example(trace, batch_size)

4) Batch-level debug of active masks
   - debug_print_active_masks(epoch_idx, batch_idx, trace, batch_size, note="")

5) Gradient-flow probes (first epochs / first batches)
   - GradProbeConfig: knobs for when/what to print
   - maybe_probe_gradients(model, logits, loss, epoch_idx, batch_idx, config=...)
     Prints:
       • batch loss
       • logits mean/std/min/max
       • ||∇root_fc.weight||_2 and ||∇output_fc.weight||_2 (or first/last Linear)
     If norms are ~0 or non-finite, prints a loud warning and returns should_stop=True
     so the caller can early-exit the run.

USAGE in your trainer (minimal)
-------------------------------
# After loss.backward() and BEFORE optimizer.step():
from utils.metrics import GradProbeConfig, maybe_probe_gradients

bad, msg = maybe_probe_gradients(
    model, outputs, loss,
    epoch_idx=epoch, batch_idx=batch_i,
    config=GradProbeConfig(enable_epochs=2, max_batches=2)
)
if bad:
    print(msg)
    # (optional) break or raise to stop the run early
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import math
import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
#                          Pretty / shape helpers                              #
# --------------------------------------------------------------------------- #

def pretty_list(nums: List[float] | List[int], decimals: int = 3) -> str:
    """Format a short vector nicely for logs."""
    if len(nums) == 0:
        return "[]"
    if isinstance(nums[0], int):
        return "[" + ", ".join(str(int(x)) for x in nums) + "]"
    fmt = f"{{:.{decimals}f}}"
    return "[" + ", ".join(fmt.format(float(x)) for x in nums) + "]"


def pad_to_match(a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pad the shorter 1D tensor with zeros so both have equal length.
    Keeps dtype/device, returns (a_padded, b_padded).
    """
    if a.numel() < b.numel():
        pad = torch.zeros(b.numel() - a.numel(), device=b.device, dtype=a.dtype)
        a = torch.cat([a, pad], dim=0)
    elif b.numel() < a.numel():
        pad = torch.zeros(a.numel() - b.numel(), device=a.device, dtype=b.dtype)
        b = torch.cat([b, pad], dim=0)
    return a, b


# --------------------------------------------------------------------------- #
#                         Trace aggregation across batches                     #
# --------------------------------------------------------------------------- #

def aggregate_trace_sums(agg: Dict[str, torch.Tensor], trace: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Sum per-batch trace vectors; pad to the longest observed length.

    Known keys (when present):
      - num_nodes_per_depth (<= max_depth+1)   [Long/Float]
      - spawn_counts_sum    (<= max_depth)     [Long/Float]
      - active_mask_sums    (<= max_depth+1)   [Float]
    """
    if not isinstance(trace, dict):
        return agg

    for key in ("num_nodes_per_depth", "spawn_counts_sum", "active_mask_sums"):
        if key not in trace:
            continue
        cur = trace[key]
        if not torch.is_tensor(cur):
            continue
        cur = cur.detach().to(torch.float32)
        if key not in agg:
            agg[key] = cur.clone()
        else:
            a, b = pad_to_match(agg[key], cur)
            agg[key] = a + b
    return agg


def average_trace(agg: Dict[str, torch.Tensor], denom: int) -> Dict[str, List[float]]:
    """Average summed traces by `denom` (e.g., number of batches)."""
    out: Dict[str, List[float]] = {}
    for k, v in agg.items():
        out[k] = (v / max(1, denom)).tolist()
    return out


# --------------------------------------------------------------------------- #
#                              Compute accounting                              #
# --------------------------------------------------------------------------- #

def estimate_compute_per_example(trace: Dict[str, torch.Tensor] | None, batch_size: int) -> float | None:
    """
    Heuristic "compute per example" estimate.

    Priority:
      1) If trace['spawn_counts_sum'] exists: sum(spawns)/B
      2) Else if trace['active_mask_sums'] exists: sum(active_parents)/B
      3) Otherwise: None (unavailable, e.g., dense baseline)
    """
    if not isinstance(trace, dict) or batch_size <= 0:
        return None
    if "spawn_counts_sum" in trace and torch.is_tensor(trace["spawn_counts_sum"]):
        val = trace["spawn_counts_sum"].detach().float().sum().item() / float(batch_size)
        return float(val)
    if "active_mask_sums" in trace and torch.is_tensor(trace["active_mask_sums"]):
        val = trace["active_mask_sums"].detach().float().sum().item() / float(batch_size)
        return float(val)
    return None


# --------------------------------------------------------------------------- #
#                           Early-epoch debug printing                         #
# --------------------------------------------------------------------------- #

def debug_print_active_masks(
    epoch_idx: int,
    batch_idx: int,
    trace: Dict[str, torch.Tensor] | None,
    batch_size: int,
    note: str = "",
) -> None:
    """
    Print raw active parents per depth for one batch. Useful in the first
    1–2 epochs to verify scale (≈ batch_size at depth 1 in warmup soft_full).

    Example output:
      [debug e=1 b=0] active_mask_sums: [64.000, 64.000, 128.000] (bs=64) warmup

    Safe to call when `trace` is None (no-op).
    """
    if not isinstance(trace, dict):
        return
    am = trace.get("active_mask_sums", None)
    if am is None or not torch.is_tensor(am):
        return
    vec = [float(x) for x in am.detach().cpu().tolist()]
    head = f"[debug e={epoch_idx} b={batch_idx}] active_mask_sums:"
    tail = f"(bs={batch_size})"
    if note:
        tail += f" {note}"
    print(f"{head} {pretty_list(vec)} {tail}")


# --------------------------------------------------------------------------- #
#                           Gradient-flow ("pipe") probes                      #
# --------------------------------------------------------------------------- #

@dataclass
class GradProbeConfig:
    """
    Knobs for the early-epoch gradient probe.

    enable_epochs : only probe for epochs 1..enable_epochs (inclusive)
    max_batches   : only probe for batches 0..max_batches-1 each enabled epoch
    warn_zero_tol : if grad norm < warn_zero_tol, treat as "nearly zero"
    scream_prefix : prefix added to severe warnings
    """
    enable_epochs: int = 2
    max_batches: int = 2
    warn_zero_tol: float = 1e-12
    scream_prefix: str = "[!!! PIPE ISSUE !!!]"


def _tensor_stats(x: torch.Tensor) -> Tuple[float, float, float, float]:
    x = x.detach()
    return float(x.mean()), float(x.std(unbiased=False)), float(x.min()), float(x.max())


def _safe_norm(t: Optional[torch.Tensor]) -> float | None:
    if t is None:
        return None
    if t.numel() == 0:
        return 0.0
    val = float(torch.linalg.norm(t.detach().float(), ord=2).item())
    if math.isnan(val) or math.isinf(val):
        return None
    return val


def _find_named_param(model: nn.Module, dotted_name: str) -> Optional[nn.Parameter]:
    """Return parameter by dotted path (e.g., 'root_fc.weight'), or None if missing."""
    cur = model
    parts = dotted_name.split(".")
    try:
        for p in parts[:-1]:
            cur = getattr(cur, p)
        return getattr(cur, parts[-1])
    except Exception:
        return None


def _pick_linear_weights(model: nn.Module) -> Tuple[Optional[nn.Parameter], Optional[nn.Parameter]]:
    """
    Prefer BFSNet's root/output weights; otherwise fall back to first/last Linear weights.
    """
    root_w = _find_named_param(model, "root_fc.weight")
    out_w  = _find_named_param(model, "output_fc.weight")
    if isinstance(root_w, torch.Tensor) and isinstance(out_w, torch.Tensor):
        return root_w, out_w

    first_w, last_w = None, None
    for m in model.modules():
        if isinstance(m, nn.Linear):
            if first_w is None:
                first_w = m.weight
            last_w = m.weight
    return first_w, last_w


def maybe_probe_gradients(
    model: nn.Module,
    logits: torch.Tensor,
    loss: torch.Tensor,
    epoch_idx: int,
    batch_idx: int,
    config: GradProbeConfig = GradProbeConfig(),
) -> Tuple[bool, str]:
    """
    Print per-batch sanity for the very first batches/epochs and return (should_stop, message).

    Expected call site:
      # forward -> loss -> loss.backward()
      bad, msg = maybe_probe_gradients(model, outputs, loss, epoch, batch_i)
      if bad:
          print(msg)
          # optional: break/return to stop early

    Returns
    -------
    should_stop : bool
        True if we detect non-finite or (near-)zero grads on key layers.
    message : str
        Human-friendly message describing the issue (empty if ok or probe skipped).
    """
    # Gate the probe: only early epochs/batches
    if epoch_idx < 1 or epoch_idx > config.enable_epochs:
        return False, ""
    if batch_idx < 0 or batch_idx >= config.max_batches:
        return False, ""

    # Basic loss / logits stats
    msg_lines: List[str] = []
    msg_lines.append(f"[probe e={epoch_idx} b={batch_idx}] loss={float(loss.detach().item()):.6f}")

    if torch.is_tensor(logits):
        lm, ls, lmin, lmax = _tensor_stats(logits)
        msg_lines.append(f"  logits: mean={lm:.4f} std={ls:.4f} min={lmin:.4f} max={lmax:.4f}")
    else:
        msg_lines.append("  logits: <non-tensor>")

    # Pick parameters to check
    p_root, p_out = _pick_linear_weights(model)
    names = []
    grads = []
    if p_root is not None:
        names.append("root_fc.weight" if p_root is _find_named_param(model, "root_fc.weight") else "first_linear.weight")
        grads.append(p_root.grad if hasattr(p_root, "grad") else None)
    if p_out is not None:
        names.append("output_fc.weight" if p_out is _find_named_param(model, "output_fc.weight") else "last_linear.weight")
        grads.append(p_out.grad if hasattr(p_out, "grad") else None)

    # If grads are None (likely called before backward), note it clearly
    if any(g is None for g in grads) or len(grads) == 0:
        msg_lines.append("  grad norms: (grads missing — call this AFTER loss.backward())")
        print("\n".join(msg_lines))
        return False, ""

    # Compute grad norms
    bad = False
    for name, g in zip(names, grads):
        gnorm = _safe_norm(g)
        if gnorm is None:
            msg_lines.append(f"  {name}: grad_norm=NaN/Inf")
            bad = True
        else:
            msg_lines.append(f"  {name}: grad_norm={gnorm:.6e}")
            if gnorm < config.warn_zero_tol:
                bad = True
                msg_lines.append(f"  {config.scream_prefix} {name} grad_norm≈0 (<{config.warn_zero_tol})")

    # Optional: also print parameter norms (helps spot dead init)
    if p_root is not None:
        pnorm = _safe_norm(p_root.data)
        msg_lines.append(f"  ||{names[0].split('.')[0]}||_2={pnorm:.6e}" if pnorm is not None else f"  {names[0]} param norm: NaN/Inf")
    if p_out is not None and len(names) > 1:
        pnorm = _safe_norm(p_out.data)
        msg_lines.append(f"  ||{names[1].split('.')[0]}||_2={pnorm:.6e}" if pnorm is not None else f"  {names[1]} param norm: NaN/Inf")

    # Scream loudly if bad
    if bad:
        msg_lines.append(f"{config.scream_prefix} Gradient flow suspect. "
                         f"Check optimizer wiring, loss call order, and device/dtype alignment.")

    print("\n".join(msg_lines))
    return bad, ("\n".join(msg_lines) if bad else "")
