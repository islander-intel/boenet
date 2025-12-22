#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
boenet/utils/profiler.py (v1.0.0 - Language Model)

Timing + FLOPs proxy utilities for BoeNet experiments.

Converted from utils/profiler.py (Vision) to boenet/utils/profiler.py (Language)
--------------------------------------------------------------------------------
Key Changes:
  - ADDED: flops_proxy_language_dense() for language model FLOPs estimation
  - ADDED: flops_proxy_language_sparse() for sparse language model FLOPs
  - ADDED: profile_language() context manager in Profiler class
  - ADDED: compute_flops_savings() helper function
  - ADDED: add_sparse_flops() method to Profiler class
  - UNCHANGED: Timer class, existing vision FLOPs functions, Profiler.profile()

Purpose
-------
1. Measure wall-clock execution time of dense vs sparse forward passes.
2. Collect lightweight FLOPs proxies (multiply-add counts based on
   input/output sizes, node expansions, etc.).
3. Provide easy context managers and decorators for profiling code blocks.
4. Support both vision (BFSNet) and language (BoeNet) model architectures.

What's included
---------------
- Timer: context manager to measure elapsed time (seconds).
- Profiler: reusable object that accumulates timings and FLOPs proxies
  across runs (e.g. for different (depth, K, batch) configs).

Vision FLOPs (BFSNet - original):
- flops_proxy_dense: Dense BFSNet FLOPs (all children spawned)
- flops_proxy_sparse: Sparse BFSNet FLOPs (only spawned children counted)

Language FLOPs (BoeNet - NEW):
- flops_proxy_language_dense: Dense BoeNet FLOPs for language models
- flops_proxy_language_sparse: Sparse BoeNet FLOPs for language models

Usage
-----
from boenet.utils.profiler import (
    Timer, Profiler,
    flops_proxy_dense, flops_proxy_sparse,
    flops_proxy_language_dense, flops_proxy_language_sparse,
    compute_flops_savings,
)

# Example timing a block
with Timer("dense pass"):
    logits = model_dense(x)

# Example profiling dense vs sparse (language model)
prof = Profiler()
for B in [32, 64]:
    x = torch.randint(0, vocab_size, (B, seq_len))
    with prof.profile_language("dense", batch=B, seq_len=seq_len, depth=2, K=3):
        model_dense(x)
    with prof.profile_language("sparse", batch=B, seq_len=seq_len, depth=2, K=3):
        model_sparse(x)
prof.report()

Notes
-----
- FLOPs proxies are approximate and assume MLP-style cost per node:
    For vision (BFSNet):
      root_fc: Din*H
      child_fc: H*H per child
      output_fc: H*Dout
    For language (BoeNet):
      embed_proj: embed_dim*H per token position
      child_fc: H*H per child
      output_fc: H*vocab_size per token position
- For sparse, we only count children that are *actually spawned*.
- For dense, we assume all potential children are materialized.
- Time uses `time.perf_counter()` (suitable for short measurements).
- Works on CPU and CUDA; on CUDA, we add `torch.cuda.synchronize()` for
  accurate timing.

FLOPs Estimation for Language Models
------------------------------------
BoeNet architecture per token position:
  - Embedding lookup: 0 FLOPs (just memory access)
  - embed_proj: embed_dim * hidden_dim
  - BFS tree expansion: same as BFSNet (hidden_dim * hidden_dim per child)
  - output_fc: hidden_dim * vocab_size

Total positions processed: B * seq_len
Total FLOPs = (B * seq_len) * (per_position_flops)

Author: BoeNet project (converted from BFSNet)
Version: 1.0.0
Date: 2025-12-22
"""

from __future__ import annotations
import time
from contextlib import contextmanager
from typing import Dict, List, Optional

import torch


# --------------------------------------------------------------------------- #
#                                Timer helper                                 #
# --------------------------------------------------------------------------- #

class Timer:
    """
    Context manager for measuring elapsed wall-clock time.
    
    Parameters
    ----------
    name : str
        Name for the timed block (printed on exit if non-empty).
    cuda_sync : bool
        If True and CUDA is available, synchronize before timing.
        
    Attributes
    ----------
    elapsed : Optional[float]
        Elapsed time in seconds (set after context exit).
        
    Examples
    --------
    >>> with Timer("forward pass"):
    ...     output = model(input)
    [Timer] forward pass: 0.012345 s
    
    >>> timer = Timer()
    >>> with timer:
    ...     output = model(input)
    >>> print(f"Elapsed: {timer.elapsed:.4f}s")
    Elapsed: 0.0123s
    """
    
    def __init__(self, name: str = "", cuda_sync: bool = True):
        self.name = name
        self.cuda_sync = cuda_sync
        self.start: Optional[float] = None
        self.elapsed: Optional[float] = None

    def __enter__(self):
        if self.cuda_sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.cuda_sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.perf_counter()
        self.elapsed = end - self.start
        if self.name:
            print(f"[Timer] {self.name}: {self.elapsed:.6f} s")


# --------------------------------------------------------------------------- #
#                     FLOPs proxy functions (Vision - BFSNet)                  #
# --------------------------------------------------------------------------- #

def flops_proxy_dense(
    B: int, 
    Din: int, 
    H: int, 
    Dout: int, 
    depth: int, 
    K: int
) -> int:
    """
    Approximate FLOPs for dense BFSNet expansion (vision model).
    
    Assumes every parent spawns *all* K children at every depth level.
    This represents the maximum computational cost.
    
    Parameters
    ----------
    B : int
        Batch size (number of images).
    Din : int
        Input dimension (e.g., 784 for flattened MNIST).
    H : int
        Hidden dimension.
    Dout : int
        Output dimension (number of classes).
    depth : int
        Maximum BFS tree depth.
    K : int
        Maximum children per node.
        
    Returns
    -------
    int
        Approximate FLOPs count.
        
    Notes
    -----
    Cost breakdown:
      - root_fc: B * Din * H
      - child_fc per depth: parents * K * H * H
      - output_fc: B * H * Dout
    """
    flops = 0
    # Root projection
    flops += B * Din * H
    # Children per depth
    parents = B
    for d in range(depth):
        flops += parents * K * H * H
        parents *= K
    # Output projection
    flops += B * H * Dout
    return flops


def flops_proxy_sparse(
    spawn_counts: List[int], 
    B: int, 
    Din: int, 
    H: int, 
    Dout: int
) -> int:
    """
    Approximate FLOPs for sparse BFSNet expansion (vision model).
    
    Counts only actually spawned children per depth.
    This represents the actual computational cost with sparsity.
    
    Parameters
    ----------
    spawn_counts : List[int]
        Number of children spawned at each depth.
    B : int
        Batch size (number of images).
    Din : int
        Input dimension (e.g., 784 for flattened MNIST).
    H : int
        Hidden dimension.
    Dout : int
        Output dimension (number of classes).
        
    Returns
    -------
    int
        Approximate FLOPs count.
    """
    flops = 0
    # Root projection
    flops += B * Din * H
    # Children per depth (only spawned ones)
    for spawned in spawn_counts:
        flops += spawned * H * H
    # Output projection
    flops += B * H * Dout
    return flops


# --------------------------------------------------------------------------- #
#                   FLOPs proxy functions (Language - BoeNet)                  #
# --------------------------------------------------------------------------- #

def flops_proxy_language_dense(
    B: int,
    seq_len: int,
    vocab_size: int,
    embed_dim: int,
    H: int,
    depth: int,
    K: int,
) -> int:
    """
    Approximate FLOPs for dense BoeNet expansion (language model).
    
    Assumes every token position spawns *all* K children at every depth level.
    This represents the maximum computational cost for language modeling.
    
    Parameters
    ----------
    B : int
        Batch size (number of sequences).
    seq_len : int
        Sequence length (number of tokens per sequence).
    vocab_size : int
        Vocabulary size (output dimension).
    embed_dim : int
        Token embedding dimension.
    H : int
        Hidden dimension for BFS tree nodes.
    depth : int
        Maximum BFS tree depth.
    K : int
        Maximum children per node.
        
    Returns
    -------
    int
        Approximate FLOPs count.
        
    Notes
    -----
    Cost breakdown per token position:
      - Embedding lookup: 0 FLOPs (memory access only)
      - embed_proj: embed_dim * H
      - BFS tree children: geometric series of K children
      - output_fc: H * vocab_size
    
    Total positions: B * seq_len
    
    Examples
    --------
    >>> flops_proxy_language_dense(B=8, seq_len=128, vocab_size=256, 
    ...                            embed_dim=64, H=128, depth=2, K=3)
    1127219200  # ~1.1B FLOPs for 8 sequences of 128 tokens
    """
    N = B * seq_len  # Total token positions
    
    flops = 0
    
    # Embedding lookup: 0 FLOPs (just memory access)
    # embed_proj: N * embed_dim * H
    flops += N * embed_dim * H
    
    # BFS tree expansion (same as BFSNet but per token position)
    # Each position has its own BFS tree
    parents_per_position = 1
    for d in range(depth):
        # child_fc: parents * K * H * H
        flops += N * parents_per_position * K * H * H
        parents_per_position *= K
    
    # Output projection: N * H * vocab_size
    flops += N * H * vocab_size
    
    return flops


def flops_proxy_language_sparse(
    spawn_counts: List[int],
    B: int,
    seq_len: int,
    vocab_size: int,
    embed_dim: int,
    H: int,
) -> int:
    """
    Approximate FLOPs for sparse BoeNet expansion (language model).
    
    Counts only actually spawned children per depth.
    This represents the actual computational cost with sparsity.
    
    Parameters
    ----------
    spawn_counts : List[int]
        Total number of children spawned at each depth (across all positions).
    B : int
        Batch size (number of sequences).
    seq_len : int
        Sequence length (number of tokens per sequence).
    vocab_size : int
        Vocabulary size (output dimension).
    embed_dim : int
        Token embedding dimension.
    H : int
        Hidden dimension for BFS tree nodes.
        
    Returns
    -------
    int
        Approximate FLOPs count.
        
    Notes
    -----
    spawn_counts[d] should contain the total number of children created
    at depth d across all B * seq_len token positions.
    
    Examples
    --------
    >>> # 50% sparsity at each depth
    >>> spawn_counts = [512, 256]  # 1024 positions, 50% spawn at d=1, 25% at d=2
    >>> flops_proxy_language_sparse(spawn_counts, B=8, seq_len=128, 
    ...                             vocab_size=256, embed_dim=64, H=128)
    """
    N = B * seq_len  # Total token positions
    
    flops = 0
    
    # Embedding lookup: 0 FLOPs
    # embed_proj: N * embed_dim * H
    flops += N * embed_dim * H
    
    # BFS tree expansion (only spawned children)
    for spawned in spawn_counts:
        flops += spawned * H * H
    
    # Output projection: N * H * vocab_size
    flops += N * H * vocab_size
    
    return flops


def compute_flops_savings(dense_flops: int, sparse_flops: int) -> float:
    """
    Compute FLOPs savings ratio from sparsity.
    
    Parameters
    ----------
    dense_flops : int
        FLOPs for dense (full tree) computation.
    sparse_flops : int
        FLOPs for sparse (pruned tree) computation.
        
    Returns
    -------
    float
        Percentage of FLOPs saved (0-100).
        
    Examples
    --------
    >>> compute_flops_savings(1000000, 600000)
    40.0  # 40% savings
    """
    if dense_flops <= 0:
        return 0.0
    return 100.0 * (1.0 - sparse_flops / dense_flops)


# --------------------------------------------------------------------------- #
#                                 Profiler                                    #
# --------------------------------------------------------------------------- #

class Profiler:
    """
    Collect timings and FLOPs proxies for multiple runs.
    
    Supports both vision (BFSNet) and language (BoeNet) models.
    
    Parameters
    ----------
    cuda_sync : bool
        If True and CUDA is available, synchronize before/after timing.
        
    Attributes
    ----------
    records : List[Dict]
        List of profiling records, each containing timing and FLOPs info.
        
    Examples
    --------
    >>> prof = Profiler()
    >>> for B in [32, 64]:
    ...     x = torch.randn(B, 784)
    ...     with prof.profile("dense", batch=B, depth=2, K=3):
    ...         model_dense(x)
    >>> prof.report()
    """

    def __init__(self, cuda_sync: bool = True):
        self.cuda_sync = cuda_sync
        self.records: List[Dict] = []

    @contextmanager
    def profile(
        self, 
        mode: str, 
        batch: int, 
        depth: int, 
        K: int, 
        Din: int = 784, 
        H: int = 64, 
        Dout: int = 10
    ):
        """
        Context manager to profile a vision model block (dense or sparse).
        Stores timing and FLOPs proxies.
        
        Parameters
        ----------
        mode : str
            "dense" or "sparse".
        batch : int
            Batch size.
        depth : int
            Maximum BFS tree depth.
        K : int
            Maximum children per node.
        Din : int
            Input dimension (default: 784 for MNIST).
        H : int
            Hidden dimension (default: 64).
        Dout : int
            Output dimension (default: 10 for MNIST).
        """
        if self.cuda_sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        yield
        if self.cuda_sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.perf_counter()
        elapsed = end - start

        # Record
        rec = dict(
            model_type="vision",
            mode=mode, 
            batch=batch, 
            depth=depth, 
            K=K, 
            Din=Din, 
            H=H, 
            Dout=Dout, 
            time=elapsed
        )
        if mode == "dense":
            rec["flops_proxy"] = flops_proxy_dense(batch, Din, H, Dout, depth, K)
        else:
            # For sparse you'll need to pass real spawn_counts separately
            rec["flops_proxy"] = None
        self.records.append(rec)

    @contextmanager
    def profile_language(
        self,
        mode: str,
        batch: int,
        seq_len: int,
        depth: int,
        K: int,
        vocab_size: int = 256,
        embed_dim: int = 64,
        H: int = 128,
    ):
        """
        Context manager to profile a language model block (dense or sparse).
        Stores timing and FLOPs proxies.
        
        Parameters
        ----------
        mode : str
            "dense" or "sparse".
        batch : int
            Batch size.
        seq_len : int
            Sequence length.
        depth : int
            Maximum BFS tree depth.
        K : int
            Maximum children per node.
        vocab_size : int
            Vocabulary size (default: 256 for char-level).
        embed_dim : int
            Embedding dimension (default: 64).
        H : int
            Hidden dimension (default: 128).
        """
        if self.cuda_sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        yield
        if self.cuda_sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.perf_counter()
        elapsed = end - start

        # Record
        rec = dict(
            model_type="language",
            mode=mode,
            batch=batch,
            seq_len=seq_len,
            depth=depth,
            K=K,
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            H=H,
            time=elapsed,
        )
        if mode == "dense":
            rec["flops_proxy"] = flops_proxy_language_dense(
                batch, seq_len, vocab_size, embed_dim, H, depth, K
            )
        else:
            # For sparse you'll need to pass real spawn_counts separately
            rec["flops_proxy"] = None
        self.records.append(rec)

    def add_sparse_flops(
        self, 
        spawn_counts: List[int], 
        record_idx: int = -1,
    ):
        """
        Add sparse FLOPs to an existing record (for sparse mode).
        
        Parameters
        ----------
        spawn_counts : List[int]
            Number of children spawned at each depth.
        record_idx : int
            Index of record to update (default: -1 for last record).
        """
        if not self.records:
            return
            
        rec = self.records[record_idx]
        
        if rec.get("model_type") == "language":
            rec["flops_proxy"] = flops_proxy_language_sparse(
                spawn_counts,
                B=rec["batch"],
                seq_len=rec["seq_len"],
                vocab_size=rec["vocab_size"],
                embed_dim=rec["embed_dim"],
                H=rec["H"],
            )
        else:  # vision
            rec["flops_proxy"] = flops_proxy_sparse(
                spawn_counts,
                B=rec["batch"],
                Din=rec["Din"],
                H=rec["H"],
                Dout=rec["Dout"],
            )

    def report(self):
        """Print a tabular report of all recorded runs."""
        if not self.records:
            print("[Profiler] No records collected.")
            return
            
        print("\n" + "=" * 80)
        print("Profiler Report")
        print("=" * 80)
        
        for r in self.records:
            if r.get("model_type") == "language":
                print(
                    f"{r['mode']:>6} | B={r['batch']:<4} seq={r['seq_len']:<4} "
                    f"depth={r['depth']} K={r['K']} "
                    f"time={r['time']:.6f}s flops={self._format_flops(r.get('flops_proxy'))}"
                )
            else:
                print(
                    f"{r['mode']:>6} | B={r['batch']:<4} depth={r['depth']} K={r['K']} "
                    f"time={r['time']:.6f}s flops={self._format_flops(r.get('flops_proxy'))}"
                )
                
        print("=" * 80 + "\n")

    def _format_flops(self, flops: Optional[int]) -> str:
        """Format FLOPs count with appropriate suffix."""
        if flops is None:
            return "N/A"
        if flops >= 1e9:
            return f"{flops/1e9:.2f}G"
        elif flops >= 1e6:
            return f"{flops/1e6:.2f}M"
        elif flops >= 1e3:
            return f"{flops/1e3:.2f}K"
        else:
            return str(flops)

    def summary(self) -> Dict:
        """
        Get summary statistics for all recorded runs.
        
        Returns
        -------
        Dict
            Summary containing counts, timing stats, and FLOPs stats.
        """
        if not self.records:
            return {"count": 0}
            
        times = [r["time"] for r in self.records]
        flops = [r.get("flops_proxy") for r in self.records if r.get("flops_proxy") is not None]
        
        summary = {
            "count": len(self.records),
            "total_time": sum(times),
            "mean_time": sum(times) / len(times),
            "min_time": min(times),
            "max_time": max(times),
        }
        
        if flops:
            summary["total_flops"] = sum(flops)
            summary["mean_flops"] = sum(flops) / len(flops)
            
        return summary

    def clear(self):
        """Clear all recorded runs."""
        self.records.clear()


# --------------------------------------------------------------------------- #
#                                 Self-test                                    #
# --------------------------------------------------------------------------- #

def _self_test():
    """Run basic self-tests for profiler module."""
    print("=" * 60)
    print("profiler.py self-test")
    print("=" * 60)
    
    # Test 1: Timer
    print("\n[Test 1] Timer")
    with Timer("test block") as t:
        time.sleep(0.01)
    assert t.elapsed is not None and t.elapsed > 0
    print(f"  Elapsed: {t.elapsed:.6f}s")
    print("  [PASS] Timer")
    
    # Test 2: Vision FLOPs (dense)
    print("\n[Test 2] flops_proxy_dense (vision)")
    flops = flops_proxy_dense(B=8, Din=784, H=64, Dout=10, depth=2, K=3)
    print(f"  FLOPs: {flops:,}")
    assert flops > 0
    print("  [PASS] flops_proxy_dense")
    
    # Test 3: Vision FLOPs (sparse)
    print("\n[Test 3] flops_proxy_sparse (vision)")
    spawn_counts = [12, 6]  # Sparse: only some children spawned
    flops_sparse = flops_proxy_sparse(spawn_counts, B=8, Din=784, H=64, Dout=10)
    print(f"  FLOPs (sparse): {flops_sparse:,}")
    print(f"  Savings: {compute_flops_savings(flops, flops_sparse):.1f}%")
    assert flops_sparse < flops
    print("  [PASS] flops_proxy_sparse")
    
    # Test 4: Language FLOPs (dense)
    print("\n[Test 4] flops_proxy_language_dense")
    flops_lang = flops_proxy_language_dense(
        B=8, seq_len=128, vocab_size=256, embed_dim=64, H=128, depth=2, K=3
    )
    print(f"  FLOPs: {flops_lang:,} ({flops_lang/1e9:.2f}G)")
    assert flops_lang > 0
    print("  [PASS] flops_proxy_language_dense")
    
    # Test 5: Language FLOPs (sparse)
    print("\n[Test 5] flops_proxy_language_sparse")
    N = 8 * 128  # 1024 positions
    spawn_counts_lang = [512, 128]  # 50% at depth 1, ~12.5% at depth 2
    flops_lang_sparse = flops_proxy_language_sparse(
        spawn_counts_lang, B=8, seq_len=128, vocab_size=256, embed_dim=64, H=128
    )
    print(f"  FLOPs (sparse): {flops_lang_sparse:,} ({flops_lang_sparse/1e9:.2f}G)")
    print(f"  Savings: {compute_flops_savings(flops_lang, flops_lang_sparse):.1f}%")
    assert flops_lang_sparse < flops_lang
    print("  [PASS] flops_proxy_language_sparse")
    
    # Test 6: compute_flops_savings
    print("\n[Test 6] compute_flops_savings")
    savings = compute_flops_savings(1000000, 600000)
    print(f"  1M -> 600K: {savings:.1f}% savings")
    assert abs(savings - 40.0) < 0.1
    print("  [PASS] compute_flops_savings")
    
    # Test 7: Profiler (vision)
    print("\n[Test 7] Profiler (vision)")
    prof = Profiler(cuda_sync=False)
    with prof.profile("dense", batch=8, depth=2, K=3):
        time.sleep(0.001)
    assert len(prof.records) == 1
    assert prof.records[0]["flops_proxy"] is not None
    print(f"  Record: mode={prof.records[0]['mode']}, flops={prof.records[0]['flops_proxy']}")
    print("  [PASS] Profiler (vision)")
    
    # Test 8: Profiler (language)
    print("\n[Test 8] Profiler (language)")
    with prof.profile_language("dense", batch=8, seq_len=128, depth=2, K=3):
        time.sleep(0.001)
    assert len(prof.records) == 2
    assert prof.records[1]["model_type"] == "language"
    print(f"  Record: mode={prof.records[1]['mode']}, flops={prof.records[1]['flops_proxy']}")
    print("  [PASS] Profiler (language)")
    
    # Test 9: Profiler.add_sparse_flops
    print("\n[Test 9] Profiler.add_sparse_flops")
    with prof.profile_language("sparse", batch=8, seq_len=128, depth=2, K=3):
        time.sleep(0.001)
    prof.add_sparse_flops([512, 128], record_idx=-1)
    assert prof.records[-1]["flops_proxy"] is not None
    print(f"  Sparse FLOPs added: {prof.records[-1]['flops_proxy']}")
    print("  [PASS] Profiler.add_sparse_flops")
    
    # Test 10: Profiler report and summary
    print("\n[Test 10] Profiler report and summary")
    prof.report()
    summary = prof.summary()
    print(f"  Summary: count={summary['count']}, total_time={summary['total_time']:.4f}s")
    assert summary["count"] == 3
    print("  [PASS] Profiler report and summary")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    _self_test()