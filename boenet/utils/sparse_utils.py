#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utils/sparse_utils.py

Lightweight helpers for sparse BFSNet execution.

Primary fix
-----------
`safe_index_add_` now uses **in-place** `Tensor.index_add_` so callers
do NOT need to capture a return value. A backward-compatible alias
`safe_index_add` is provided and also performs the in-place update.

Why this matters
----------------
If you call:
    safe_index_add(agg_sum, 0, act_idx, contrib)
and the helper returned a *new* tensor (out-of-place), but the caller
didn't reassign `agg_sum = ...`, the update was lost. Using `index_add_`
fixes this by mutating `agg_sum` in-place.

What's included
---------------
1) safe_index_select(x, dim, index)
2) safe_index_add_(dst, dim, index, src)   ← in-place (primary fix)
   safe_index_add(...)                      ← alias to the in-place version
3) build_batch_index(batch_size, num, device=None)
4) scatter_mean(src, index, dim, dim_size)  ← now uses in-place adds
5) ensure_nonempty(x, shape, device=None)

Design notes
------------
- Guards against empty indices (no-ops instead of crashes).
- Validates dtype/device/shapes early with helpful error messages.
- Automatically casts src to match dst dtype for mixed precision support.
- Pure PyTorch; CUDA-safe.
- Keeps behavior deterministic and explicit.

Float16/Mixed Precision Support (2025-09-17)
--------------------------------------------
- `safe_index_add_` now automatically casts `src` to match `dst` dtype
- This enables float16 (half precision) training without dtype mismatch errors
- The cast is performed only when necessary to minimize overhead

Author: BFS project
Updated: 2025-09-17
"""

from __future__ import annotations
from typing import Optional
import logging

import torch

# Module-level logger
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#                        Safe indexing and selection                          #
# --------------------------------------------------------------------------- #

def safe_index_select(x: torch.Tensor, dim: int, index: torch.Tensor) -> torch.Tensor:
    """
    Like torch.index_select but returns an empty tensor if `index` is empty.

    Args
    ----
    x     : Tensor [*, ..., *]
    dim   : int (selection dimension)
    index : LongTensor [M]

    Returns
    -------
    out   : Tensor with size(out, dim) == M
    """
    if not torch.is_tensor(x):
        raise TypeError("safe_index_select: x must be a torch.Tensor")
    if not torch.is_tensor(index):
        raise TypeError("safe_index_select: index must be a torch.Tensor")
    if index.numel() == 0:
        shape = list(x.shape)
        shape[dim] = 0
        return x.new_empty(shape)
    if index.dtype != torch.long:
        index = index.to(torch.long)
    if index.device != x.device:
        index = index.to(x.device)
    return torch.index_select(x, dim, index)


# --------------------------------------------------------------------------- #
#                      In-place indexed addition (PRIMARY)                    #
# --------------------------------------------------------------------------- #

def _validate_index_add_args(dst: torch.Tensor, dim: int, index: torch.Tensor, src: torch.Tensor) -> None:
    """
    Validate arguments for safe_index_add_ operation.
    
    Note: This function validates the arguments but does not modify them.
    The caller (safe_index_add_) handles dtype/device normalization before calling this.
    
    Parameters
    ----------
    dst : torch.Tensor
        Destination tensor.
    dim : int
        Dimension along which to add.
    index : torch.Tensor
        Index tensor (must be torch.long).
    src : torch.Tensor
        Source tensor to add.
        
    Raises
    ------
    TypeError
        If dst, src, or index are not tensors.
    ValueError
        If there are device, dtype, or shape mismatches.
    """
    if not torch.is_tensor(dst) or not torch.is_tensor(src):
        raise TypeError("safe_index_add_: dst and src must be tensors")
    if not torch.is_tensor(index):
        raise TypeError("safe_index_add_: index must be a tensor")
    if dst.device != src.device:
        raise ValueError(f"safe_index_add_: device mismatch dst={dst.device} vs src={src.device}")
    if dst.device != index.device:
        raise ValueError(f"safe_index_add_: index device {index.device} must match dst device {dst.device}")
    if index.dtype != torch.long:
        raise ValueError(f"safe_index_add_: index dtype must be torch.long, got {index.dtype}")
    # Note: dtype validation for dst vs src is NOT done here because we handle casting in safe_index_add_
    # shape check: src.size(dim) must equal index.numel(); other dims must match dst
    if src.dim() != dst.dim():
        raise ValueError(f"safe_index_add_: rank mismatch src.dim()={src.dim()} vs dst.dim()={dst.dim()}")
    if src.size(dim) != index.numel():
        raise ValueError(
            f"safe_index_add_: src.size(dim={dim})={src.size(dim)} must equal index.numel()={index.numel()}"
        )
    for d in range(dst.dim()):
        if d == dim:
            continue
        if src.size(d) != dst.size(d):
            raise ValueError(
                f"safe_index_add_: size mismatch on dim {d}: src={src.size(d)} vs dst={dst.size(d)}"
            )


def safe_index_add_(dst: torch.Tensor, dim: int, index: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    """
    In-place: dst.index_add_(dim, index, src), but **safe** on empty inputs.

    Args
    ----
    dst   : destination tensor [N, ...]  (WILL BE MODIFIED IN-PLACE)
    dim   : dimension along which to add
    index : LongTensor [M] indices into dst along `dim`
    src   : source tensor [M, ...] to be added into `dst` at `index`

    Returns
    -------
    dst   : the same (mutated) tensor, for convenience

    Notes
    -----
    - If `index` or `src` is empty, the function is a no-op.
    - Performs basic device/dtype/shape validation with clear errors.
    - Automatically casts `src` to match `dst` dtype for mixed precision support.
      This enables float16/half precision training without manual dtype management.
    
    Examples
    --------
    >>> dst = torch.zeros(4, 3, dtype=torch.float16, device='cuda')
    >>> src = torch.ones(2, 3, dtype=torch.float32, device='cuda')  # Different dtype
    >>> idx = torch.tensor([1, 3], device='cuda')
    >>> safe_index_add_(dst, 0, idx, src)  # Works! src is cast to float16
    """
    if index.numel() == 0 or src.numel() == 0:
        return dst  # nothing to add
    
    # Normalize index dtype/device *before* validation
    if index.dtype != torch.long:
        index = index.to(torch.long)
    if index.device != dst.device:
        index = index.to(dst.device)
    
    # Cast src to match dst dtype if necessary (for mixed precision support)
    if src.dtype != dst.dtype:
        logger.debug(f"safe_index_add_: casting src from {src.dtype} to {dst.dtype}")
        src = src.to(dst.dtype)
    
    _validate_index_add_args(dst, dim, index, src)
    dst.index_add_(dim, index, src)  # IN-PLACE
    return dst


# Backward-compatible alias (also in-place)
def safe_index_add(dst: torch.Tensor, dim: int, index: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    """
    Alias to `safe_index_add_` (in-place). Kept for existing call sites.
    
    This function automatically handles dtype mismatches between dst and src
    by casting src to match dst's dtype. This enables mixed precision training
    (e.g., float16) without manual dtype management.
    
    Parameters
    ----------
    dst : torch.Tensor
        Destination tensor (modified in-place).
    dim : int
        Dimension along which to add.
    index : torch.Tensor
        Index tensor.
    src : torch.Tensor
        Source tensor (will be cast to dst's dtype if necessary).
        
    Returns
    -------
    torch.Tensor
        The dst tensor (same object, modified in-place).
    """
    return safe_index_add_(dst, dim, index, src)


# --------------------------------------------------------------------------- #
#                          Batch index helpers                                #
# --------------------------------------------------------------------------- #

def build_batch_index(batch_size: int, num: int, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Build [B*num] batch indices aligned with [B, num, ...] flattening.

    Example
    -------
    B=2, num=3 → tensor([0,0,0, 1,1,1])
    
    Parameters
    ----------
    batch_size : int
        Batch size B.
    num : int
        Number of elements per batch item.
    device : torch.device, optional
        Device to place the tensor on.
        
    Returns
    -------
    torch.Tensor
        Long tensor of shape [B*num] with batch indices.
    """
    if batch_size <= 0 or num <= 0:
        return torch.empty(0, dtype=torch.long, device=device)
    return torch.arange(batch_size, device=device).repeat_interleave(num)


# --------------------------------------------------------------------------- #
#                           Safe scatter reductions                           #
# --------------------------------------------------------------------------- #

def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim: int, dim_size: int) -> torch.Tensor:
    """
    Scatter-mean: reduce `src` into an output of size [dim_size, ...] by index,
    computing the mean along `dim`.

    Args
    ----
    src      : Tensor [N, ...]
    index    : LongTensor [N] with 0 <= index[i] < dim_size
    dim      : int (reduction dimension in output)
    dim_size : int (size of the reduced dimension)

    Returns
    -------
    out : Tensor [dim_size, ...] with mean-reduced values.
    
    Notes
    -----
    - If src is empty, returns a zero tensor of appropriate shape.
    - Handles dtype/device normalization for index tensor.
    - Uses in-place operations for efficiency.
    """
    if not torch.is_tensor(src):
        raise TypeError("scatter_mean: src must be a tensor")
    if not torch.is_tensor(index):
        raise TypeError("scatter_mean: index must be a tensor")
    if src.numel() == 0:
        # fabricate an appropriately shaped zero tensor
        shape = list(src.shape)
        shape[dim] = dim_size
        return src.new_zeros(shape)

    if index.dtype != torch.long:
        index = index.to(torch.long)
    if index.device != src.device:
        index = index.to(src.device)

    # Prepare outputs
    out = src.new_zeros((dim_size,) + src.shape[1:])
    counts = src.new_zeros(dim_size, dtype=torch.long)

    # In-place reductions
    out.index_add_(dim, index, src)
    counts.index_add_(0, index, torch.ones_like(index, dtype=torch.long))

    # Avoid division by zero
    # Expand counts to broadcast across remaining dims
    while counts.dim() < out.dim():
        counts = counts.unsqueeze(-1)
    counts = counts.clamp_min(1)

    return out / counts


# --------------------------------------------------------------------------- #
#                         Empty-tensor safety utility                          #
# --------------------------------------------------------------------------- #

def ensure_nonempty(x: torch.Tensor, shape: torch.Size, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Ensure tensor is non-empty: if empty, replace with zeros of given shape.

    Useful when a frontier dies early and you still want to return a tensor
    with consistent shape.
    
    Parameters
    ----------
    x : torch.Tensor
        Input tensor to check.
    shape : torch.Size
        Shape to use if x is empty.
    device : torch.device, optional
        Device to place the replacement tensor on. Defaults to x.device.
        
    Returns
    -------
    torch.Tensor
        Original tensor if non-empty, or zero tensor of given shape.
    """
    if not torch.is_tensor(x):
        raise TypeError("ensure_nonempty: x must be a tensor")
    if x.numel() == 0:
        return torch.zeros(shape, device=(device or x.device), dtype=x.dtype)
    return x


# --------------------------------------------------------------------------- #
#                                   Self-test                                 #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    # Configure logging for self-test
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    # Minimal sanity tests you can run with: python utils/sparse_utils.py
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Running sparse_utils self-tests on {device}")
    print("=" * 60)

    # Test 1: in-place index_add
    print("\n[Test 1] In-place index_add")
    dst = torch.zeros(4, 3, device=device)
    src = torch.ones(2, 3, device=device)
    idx = torch.tensor([1, 3], device=device, dtype=torch.long)
    _ = safe_index_add(dst, 0, idx, src)   # in-place
    assert torch.allclose(dst[1], torch.ones(3, device=device))
    assert torch.allclose(dst[3], torch.ones(3, device=device))
    print("  ✓ safe_index_add in-place update works")

    # Test 2: empty index is a no-op
    print("\n[Test 2] Empty index no-op")
    dst2 = dst.clone()
    _ = safe_index_add(dst2, 0, torch.tensor([], device=device, dtype=torch.long), src[:0])
    assert torch.allclose(dst2, dst), "[fail] empty index should not change dst"
    print("  ✓ empty index/no-op guarded")

    # Test 3: scatter_mean
    print("\n[Test 3] Scatter mean")
    src3 = torch.tensor([[1.0, 2.0],
                         [3.0, 4.0],
                         [5.0, 6.0],
                         [7.0, 8.0]], device=device)
    idx3 = torch.tensor([0, 1, 0, 1], device=device, dtype=torch.long)
    out = scatter_mean(src3, idx3, dim=0, dim_size=2)
    # group 0: rows [0,2] mean → [(1+5)/2, (2+6)/2] = [3,4]
    # group 1: rows [1,3] mean → [(3+7)/2, (4+8)/2] = [5,6]
    expect = torch.tensor([[3.0, 4.0], [5.0, 6.0]], device=device)
    assert torch.allclose(out, expect), "[fail] scatter_mean incorrect"
    print("  ✓ scatter_mean works")

    # Test 4: dtype casting (float16 support)
    print("\n[Test 4] Dtype casting (mixed precision support)")
    if device.type == "cuda":
        dst_half = torch.zeros(4, 3, device=device, dtype=torch.float16)
        src_float = torch.ones(2, 3, device=device, dtype=torch.float32)
        idx_cast = torch.tensor([0, 2], device=device, dtype=torch.long)
        
        # This should NOT raise - src should be cast to float16
        safe_index_add(dst_half, 0, idx_cast, src_float)
        
        assert dst_half.dtype == torch.float16, "dst dtype should remain float16"
        assert torch.allclose(dst_half[0], torch.ones(3, device=device, dtype=torch.float16))
        assert torch.allclose(dst_half[2], torch.ones(3, device=device, dtype=torch.float16))
        print("  ✓ float32 -> float16 casting works")
        
        # Also test float16 -> float32 casting
        dst_float = torch.zeros(4, 3, device=device, dtype=torch.float32)
        src_half = torch.ones(2, 3, device=device, dtype=torch.float16)
        idx_cast2 = torch.tensor([1, 3], device=device, dtype=torch.long)
        
        safe_index_add(dst_float, 0, idx_cast2, src_half)
        
        assert dst_float.dtype == torch.float32, "dst dtype should remain float32"
        print("  ✓ float16 -> float32 casting works")
    else:
        print("  ⊘ Skipped (CUDA not available for float16 test)")

    # Test 5: build_batch_index
    print("\n[Test 5] Build batch index")
    batch_idx = build_batch_index(2, 3, device=device)
    expected_idx = torch.tensor([0, 0, 0, 1, 1, 1], device=device, dtype=torch.long)
    assert torch.equal(batch_idx, expected_idx), "[fail] build_batch_index incorrect"
    print("  ✓ build_batch_index works")

    # Test 6: ensure_nonempty
    print("\n[Test 6] Ensure nonempty")
    empty_tensor = torch.empty(0, 5, device=device)
    result = ensure_nonempty(empty_tensor, torch.Size([3, 5]), device=device)
    assert result.shape == (3, 5), "[fail] ensure_nonempty wrong shape"
    assert (result == 0).all(), "[fail] ensure_nonempty should be zeros"
    print("  ✓ ensure_nonempty works")

    print("\n" + "=" * 60)
    print("All sparse_utils self-tests passed!")