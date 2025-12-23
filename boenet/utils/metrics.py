#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
boenet/utils/metrics.py (v1.0.0 - Language Model)

Small helpers for BoeNet training logs, compute accounting, and early-epoch
"pipe alive?" gradient probes.

Converted from utils/metrics.py (Vision) to boenet/utils/metrics.py (Language)
-------------------------------------------------------------------------------
Key Changes:
  - ADDED: compute_perplexity(cross_entropy_loss) -> float
  - UPDATED: _pick_linear_weights() to look for BoeNet layers (embed_proj, output_fc)
  - UPDATED: Layer name detection in maybe_probe_gradients() for language model
  - UNCHANGED: All other functionality (trace aggregation, debug printing, etc.)

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

4) Perplexity computation (NEW for language models)
   - compute_perplexity(cross_entropy_loss) -> float

5) Batch-level debug of active masks
   - debug_print_active_masks(epoch_idx, batch_idx, trace, batch_size, note="")

6) Gradient-flow probes (first epochs / first batches)
   - GradProbeConfig: knobs for when/what to print
   - maybe_probe_gradients(model, logits, loss, epoch_idx, batch_idx, config=...)
     Prints:
       - batch loss
       - logits mean/std/min/max
       - ||grad embed_proj.weight||_2 and ||grad output_fc.weight||_2 (BoeNet)
       - ||grad root_fc.weight||_2 and ||grad output_fc.weight||_2 (BFSNet fallback)
     If norms are ~0 or non-finite, prints a loud warning and returns should_stop=True
     so the caller can early-exit the run.

USAGE in your trainer (minimal)
-------------------------------
# After loss.backward() and BEFORE optimizer.step():
from boenet.utils.metrics import GradProbeConfig, maybe_probe_gradients, compute_perplexity

# Compute perplexity from cross-entropy loss
ppl = compute_perplexity(loss.item())
print(f"Perplexity: {ppl:.2f}")

# Gradient probe
bad, msg = maybe_probe_gradients(
    model, outputs, loss,
    epoch_idx=epoch, batch_idx=batch_i,
    config=GradProbeConfig(enable_epochs=2, max_batches=2)
)
if bad:
    print(msg)
    # (optional) break or raise to stop the run early

Author: BoeNet project (converted from BFSNet)
Version: 1.0.0
Date: 2025-12-22
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union

import math
import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
#                          Pretty / shape helpers                              #
# --------------------------------------------------------------------------- #

def pretty_list(nums: Union[List[float], List[int]], decimals: int = 3) -> str:
    """
    Format a short vector nicely for logs.
    
    Parameters
    ----------
    nums : List[float] or List[int]
        List of numbers to format.
    decimals : int
        Number of decimal places for floats.
        
    Returns
    -------
    str
        Formatted string representation.
        
    Examples
    --------
    >>> pretty_list([1.234, 5.678, 9.012])
    '[1.234, 5.678, 9.012]'
    >>> pretty_list([1, 2, 3])
    '[1, 2, 3]'
    """
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
    
    Parameters
    ----------
    a : torch.Tensor
        First tensor.
    b : torch.Tensor
        Second tensor.
        
    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        Both tensors padded to the same length.
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

def aggregate_trace_sums(
    agg: Dict[str, torch.Tensor], 
    trace: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """
    Sum per-batch trace vectors; pad to the longest observed length.

    Known keys (when present):
      - num_nodes_per_depth (<= max_depth+1)   [Long/Float]
      - spawn_counts_sum    (<= max_depth)     [Long/Float]
      - active_mask_sums    (<= max_depth+1)   [Float]
      
    Parameters
    ----------
    agg : Dict[str, torch.Tensor]
        Accumulated trace sums.
    trace : Dict[str, torch.Tensor]
        Current batch trace to add.
        
    Returns
    -------
    Dict[str, torch.Tensor]
        Updated accumulated trace sums.
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
    """
    Average summed traces by `denom` (e.g., number of batches).
    
    Parameters
    ----------
    agg : Dict[str, torch.Tensor]
        Accumulated trace sums.
    denom : int
        Denominator for averaging (e.g., number of batches).
        
    Returns
    -------
    Dict[str, List[float]]
        Averaged trace values as lists.
    """
    out: Dict[str, List[float]] = {}
    for k, v in agg.items():
        out[k] = (v / max(1, denom)).tolist()
    return out


# --------------------------------------------------------------------------- #
#                              Compute accounting                              #
# --------------------------------------------------------------------------- #

def estimate_compute_per_example(
    trace: Optional[Dict[str, torch.Tensor]], 
    batch_size: int
) -> Optional[float]:
    """
    Heuristic "compute per example" estimate.

    Priority:
      1) If trace['spawn_counts_sum'] exists: sum(spawns)/B
      2) Else if trace['active_mask_sums'] exists: sum(active_parents)/B
      3) Otherwise: None (unavailable, e.g., dense baseline)
      
    Parameters
    ----------
    trace : Optional[Dict[str, torch.Tensor]]
        Trace dictionary from model forward pass.
    batch_size : int
        Batch size for normalization.
        
    Returns
    -------
    Optional[float]
        Estimated compute per example, or None if unavailable.
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
#                      Perplexity Computation (NEW)                            #
# --------------------------------------------------------------------------- #

def compute_perplexity(cross_entropy_loss: float) -> float:
    """
    Compute perplexity from cross-entropy loss.
    
    Perplexity is defined as exp(cross_entropy_loss) and is a standard
    metric for language models. Lower perplexity indicates better performance.
    
    Parameters
    ----------
    cross_entropy_loss : float
        Cross-entropy loss value (natural log base).
        
    Returns
    -------
    float
        Perplexity value (exp(loss)).
        
    Notes
    -----
    - Random baseline perplexity = vocab_size (e.g., 256 for char-level)
    - Perfect model perplexity = 1.0
    - Typical trained char-level models: 2-20 perplexity
    - If loss is very large (>20), perplexity is clamped to avoid overflow
    
    Examples
    --------
    >>> compute_perplexity(2.0)  # exp(2.0) = 7.39
    7.3890560989306495
    >>> compute_perplexity(5.545)  # exp(5.545) = 256 (random baseline for char-level)
    256.0
    >>> compute_perplexity(0.0)  # exp(0) = 1.0 (perfect model)
    1.0
    """
    # Clamp to avoid overflow (exp(710) = inf for float64)
    # For practical purposes, perplexity > 1e8 is meaningless
    clamped_loss = min(cross_entropy_loss, 20.0)
    return math.exp(clamped_loss)


# --------------------------------------------------------------------------- #
#                           Early-epoch debug printing                         #
# --------------------------------------------------------------------------- #

def debug_print_active_masks(
    epoch_idx: int,
    batch_idx: int,
    trace: Optional[Dict[str, torch.Tensor]],
    batch_size: int,
    note: str = "",
) -> None:
    """
    Print raw active parents per depth for one batch. Useful in the first
    1-2 epochs to verify scale.

    Example output:
      [debug e=1 b=0] active_mask_sums: [64.000, 64.000, 128.000] (bs=64) warmup

    Safe to call when `trace` is None (no-op).
    
    Parameters
    ----------
    epoch_idx : int
        Current epoch index.
    batch_idx : int
        Current batch index.
    trace : Optional[Dict[str, torch.Tensor]]
        Trace dictionary from model forward pass.
    batch_size : int
        Batch size for reference.
    note : str
        Optional note to append to output.
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

    Attributes
    ----------
    enable_epochs : int
        Only probe for epochs 1..enable_epochs (inclusive).
    max_batches : int
        Only probe for batches 0..max_batches-1 each enabled epoch.
    warn_zero_tol : float
        If grad norm < warn_zero_tol, treat as "nearly zero".
    scream_prefix : str
        Prefix added to severe warnings.
    """
    enable_epochs: int = 2
    max_batches: int = 2
    warn_zero_tol: float = 1e-12
    scream_prefix: str = "[!!! PIPE ISSUE !!!]"


def _tensor_stats(x: torch.Tensor) -> Tuple[float, float, float, float]:
    """Compute mean, std, min, max for a tensor."""
    x = x.detach()
    return float(x.mean()), float(x.std(unbiased=False)), float(x.min()), float(x.max())


def _safe_norm(t: Optional[torch.Tensor]) -> Optional[float]:
    """Safely compute L2 norm, returning None if non-finite or empty."""
    if t is None:
        return None
    if t.numel() == 0:
        return 0.0
    val = float(torch.linalg.norm(t.detach().float(), ord=2).item())
    if math.isnan(val) or math.isinf(val):
        return None
    return val


def _find_named_param(model: nn.Module, dotted_name: str) -> Optional[nn.Parameter]:
    """Return parameter by dotted path (e.g., 'embed_proj.weight'), or None if missing."""
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
    Pick key linear layer weights for gradient checking.
    
    For BoeNet (language model):
      - Input projection: embed_proj.weight (projects embeddings to hidden_dim)
      - Output layer: output_fc.weight (projects to vocab_size)
    
    For BFSNet (vision model - backwards compatible):
      - Input layer: root_fc.weight (projects flattened image to hidden_dim)
      - Output layer: output_fc.weight (projects to num_classes)
    
    Falls back to first/last Linear weights if named layers not found.
    
    Returns
    -------
    Tuple[Optional[nn.Parameter], Optional[nn.Parameter]]
        (input_weight, output_weight) parameters for gradient checking.
    """
    # Try BoeNet language model layers first (embed_proj is the input projection)
    embed_proj_w = _find_named_param(model, "embed_proj.weight")
    output_fc_w = _find_named_param(model, "output_fc.weight")
    
    if isinstance(embed_proj_w, torch.Tensor) and isinstance(output_fc_w, torch.Tensor):
        return embed_proj_w, output_fc_w
    
    # Try BFSNet vision model layers (backwards compatibility)
    root_fc_w = _find_named_param(model, "root_fc.weight")
    if isinstance(root_fc_w, torch.Tensor) and isinstance(output_fc_w, torch.Tensor):
        return root_fc_w, output_fc_w
    
    # Fallback: find first and last Linear layer weights
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

    For BoeNet (language model), checks:
      - embed_proj.weight gradient (input projection)
      - output_fc.weight gradient (output projection)
      
    For BFSNet (vision model), checks:
      - root_fc.weight gradient (input layer)
      - output_fc.weight gradient (output layer)

    Parameters
    ----------
    model : nn.Module
        The model being trained.
    logits : torch.Tensor
        Output logits from forward pass.
    loss : torch.Tensor
        Computed loss value.
    epoch_idx : int
        Current epoch (1-indexed).
    batch_idx : int
        Current batch index (0-indexed).
    config : GradProbeConfig
        Configuration for probing behavior.

    Returns
    -------
    Tuple[bool, str]
        (should_stop, message)
        - should_stop: True if we detect non-finite or (near-)zero grads on key layers.
        - message: Human-friendly message describing the issue (empty if ok or probe skipped).
    """
    # Gate the probe: only early epochs/batches
    if epoch_idx < 1 or epoch_idx > config.enable_epochs:
        return False, ""
    if batch_idx < 0 or batch_idx >= config.max_batches:
        return False, ""

    # Basic loss / logits stats
    msg_lines: List[str] = []
    loss_val = float(loss.detach().item())
    msg_lines.append(f"[probe e={epoch_idx} b={batch_idx}] loss={loss_val:.6f}")

    if torch.is_tensor(logits):
        lm, ls, lmin, lmax = _tensor_stats(logits)
        msg_lines.append(f"  logits: mean={lm:.4f} std={ls:.4f} min={lmin:.4f} max={lmax:.4f}")
    else:
        msg_lines.append("  logits: <non-tensor>")

    # Pick parameters to check
    p_input, p_output = _pick_linear_weights(model)
    
    # Determine layer names based on what we found
    embed_proj_w = _find_named_param(model, "embed_proj.weight")
    root_fc_w = _find_named_param(model, "root_fc.weight")
    output_fc_w = _find_named_param(model, "output_fc.weight")
    
    names = []
    params = []
    grads = []
    
    if p_input is not None:
        # Determine the correct name for the input layer
        if p_input is embed_proj_w:
            name = "embed_proj.weight"
        elif p_input is root_fc_w:
            name = "root_fc.weight"
        else:
            name = "first_linear.weight"
        names.append(name)
        params.append(p_input)
        grads.append(p_input.grad if hasattr(p_input, "grad") else None)
        
    if p_output is not None:
        # Determine the correct name for the output layer
        if p_output is output_fc_w:
            name = "output_fc.weight"
        else:
            name = "last_linear.weight"
        names.append(name)
        params.append(p_output)
        grads.append(p_output.grad if hasattr(p_output, "grad") else None)

    # If grads are None (likely called before backward), note it clearly
    if any(g is None for g in grads) or len(grads) == 0:
        msg_lines.append("  grad norms: (grads missing - call this AFTER loss.backward())")
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
                msg_lines.append(f"  {config.scream_prefix} {name} grad_norm~=0 (<{config.warn_zero_tol})")

    # Also print parameter norms (helps spot dead init)
    for name, param in zip(names, params):
        pnorm = _safe_norm(param.data)
        layer_name = name.split('.')[0]
        if pnorm is not None:
            msg_lines.append(f"  ||{layer_name}||_2={pnorm:.6e}")
        else:
            msg_lines.append(f"  {layer_name} param norm: NaN/Inf")

    # Scream loudly if bad
    if bad:
        msg_lines.append(f"{config.scream_prefix} Gradient flow suspect. "
                         f"Check optimizer wiring, loss call order, and device/dtype alignment.")

    print("\n".join(msg_lines))
    return bad, ("\n".join(msg_lines) if bad else "")


# --------------------------------------------------------------------------- #
#                                 Self-test                                    #
# --------------------------------------------------------------------------- #

def _self_test():
    """Run basic self-tests for metrics module."""
    print("=" * 60)
    print("metrics.py self-test")
    print("=" * 60)
    
    # Test 1: compute_perplexity
    print("\n[Test 1] compute_perplexity")
    test_cases = [
        (0.0, 1.0),           # exp(0) = 1
        (1.0, math.e),        # exp(1) = e
        (2.0, math.exp(2)),   # exp(2)
        (5.545, 256.0),       # ln(256) = 5.545, so exp(5.545) = 256
    ]
    for loss, expected in test_cases:
        ppl = compute_perplexity(loss)
        print(f"  loss={loss:.3f} -> ppl={ppl:.2f} (expected ~{expected:.2f})")
        assert abs(ppl - expected) < 1.0, f"Perplexity mismatch for loss={loss}"
    print("  [PASS] compute_perplexity")
    
    # Test 2: pretty_list
    print("\n[Test 2] pretty_list")
    result = pretty_list([1.234, 5.678])
    print(f"  {result}")
    assert "1.234" in result and "5.678" in result
    result_int = pretty_list([1, 2, 3])
    print(f"  {result_int}")
    assert result_int == "[1, 2, 3]"
    print("  [PASS] pretty_list")
    
    # Test 3: pad_to_match
    print("\n[Test 3] pad_to_match")
    a = torch.tensor([1.0, 2.0])
    b = torch.tensor([3.0, 4.0, 5.0])
    a_pad, b_pad = pad_to_match(a, b)
    assert a_pad.numel() == b_pad.numel() == 3
    print(f"  a_pad={a_pad.tolist()}, b_pad={b_pad.tolist()}")
    print("  [PASS] pad_to_match")
    
    # Test 4: aggregate_trace_sums
    print("\n[Test 4] aggregate_trace_sums")
    agg = {}
    trace1 = {"spawn_counts_sum": torch.tensor([1.0, 2.0])}
    trace2 = {"spawn_counts_sum": torch.tensor([3.0, 4.0, 5.0])}
    agg = aggregate_trace_sums(agg, trace1)
    agg = aggregate_trace_sums(agg, trace2)
    print(f"  aggregated: {agg['spawn_counts_sum'].tolist()}")
    assert agg["spawn_counts_sum"].numel() == 3
    print("  [PASS] aggregate_trace_sums")
    
    # Test 5: average_trace
    print("\n[Test 5] average_trace")
    avg = average_trace(agg, denom=2)
    print(f"  averaged: {avg}")
    assert "spawn_counts_sum" in avg
    print("  [PASS] average_trace")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    _self_test()