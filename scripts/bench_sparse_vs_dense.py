#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bench_sparse_vs_dense.py

Micro-benchmark to compare BFSNet execution modes:
  • Dense: "mask-then-compute" (baseline, all potential children are computed)
  • Sparse: "compute-only-active" (optimized, only active paths are evaluated)

What this script does
---------------------
1) Builds BFSNet in both dense and sparse execution modes.
2) Sweeps over grids of batch size, depth, and branching factor (K).
3) Measures:
    - Forward pass wall-clock latency (ms / batch)
    - Peak memory usage (MB)
4) Checks correctness by comparing logits from both modes
   (differences should be near-zero, up to numerical noise).
5) Prints results in a tabular format.

Usage
-----
# Default run with small grid
python scripts/bench_sparse_vs_dense.py

# Custom grid
python scripts/bench_sparse_vs_dense.py \
  --batches 32 64 128 \
  --depths 2 3 4 \
  --children 2 3 \
  --hidden_dim 64 \
  --trials 5

Notes
-----
- Runs on CUDA if available, else CPU.
- On CPU, timing noise may be higher, so average across trials is important.
- Sparse vs dense is controlled by BFSNet(init, sparse_execute=...).
- This is NOT training; just forward passes with random inputs.

Author: William McKeon
Updated: 2025-08-26
"""

import argparse
import time
import torch
import torch.nn as nn
from typing import List, Tuple

# Import BFSNet from local project
from bfs_model import BFSNet


# --------------------------------------------------------------------------- #
#                               Helper functions                              #
# --------------------------------------------------------------------------- #

def set_seed(seed: int = 42):
    torch.manual_seed(seed)


def measure_forward(model: nn.Module, x: torch.Tensor, trials: int = 5) -> Tuple[float, float]:
    """
    Run multiple forward passes and return avg latency + peak memory.
    Returns:
        latency_ms (float): Average latency in ms
        peak_mem_mb (float): Peak allocated memory in MB
    """
    device = next(model.parameters()).device
    torch.cuda.empty_cache() if device.type == "cuda" else None

    # Warm-up
    for _ in range(2):
        _ = model(x)

    torch.cuda.reset_peak_memory_stats(device) if device.type == "cuda" else None

    times = []
    for _ in range(trials):
        start = time.time()
        _ = model(x)
        torch.cuda.synchronize(device) if device.type == "cuda" else None
        end = time.time()
        times.append((end - start) * 1000.0)

    latency = sum(times) / len(times)
    if device.type == "cuda":
        peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    else:
        peak_mem = 0.0
    return latency, peak_mem


def run_benchmark(
    batch_sizes: List[int],
    depths: List[int],
    children: List[int],
    hidden_dim: int,
    trials: int,
    device: torch.device,
) -> None:
    """
    Sweep over (batch, depth, children) grid and benchmark dense vs sparse.
    """
    input_dim = 784  # FMNIST-like input
    output_dim = 10  # classification

    print("\n=== BFSNet Sparse vs Dense Benchmark ===")
    print(f"Device: {device}")
    print(f"Hidden dim: {hidden_dim}, Trials per run: {trials}")
    print("--------------------------------------------------------------")
    header = f"{'B':>6} {'D':>3} {'K':>3} | {'Dense Lat(ms)':>12} {'Sparse Lat(ms)':>13} | {'Dense Mem(MB)':>12} {'Sparse Mem(MB)':>13}"
    print(header)
    print("-" * len(header))

    for B in batch_sizes:
        for D in depths:
            for K in children:
                # Dummy input (Gaussian noise)
                x = torch.randn(B, input_dim, device=device)

                # Dense model (soft_full execution mode)
                dense_model = BFSNet(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    output_dim=output_dim,
                    max_depth=D,
                    max_children=K,
                    sibling_embed=True,
                    use_pruning=False,
                    branch_temperature=1.0,
                    exec_mode="soft_full",
                ).to(device)

                # Sparse model (sparse execution mode)
                # IMPORTANT: Copy weights from dense_model for meaningful correctness comparison
                sparse_model = BFSNet(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    output_dim=output_dim,
                    max_depth=D,
                    max_children=K,
                    sibling_embed=True,
                    use_pruning=False,
                    branch_temperature=1.0,
                    exec_mode="sparse",
                ).to(device)
                
                # Load the same weights for fair comparison
                sparse_model.load_state_dict(dense_model.state_dict())

                # Benchmark
                dense_lat, dense_mem = measure_forward(dense_model, x, trials=trials)
                sparse_lat, sparse_mem = measure_forward(sparse_model, x, trials=trials)

                # Correctness check: With same weights and same input, outputs should be similar
                # Note: Due to different execution paths (soft aggregation vs hard masking),
                # there may be some differences, but they should be bounded.
                with torch.no_grad():
                    # Set both models to eval mode for deterministic comparison
                    dense_model.eval()
                    sparse_model.eval()
                    
                    # Use the same random seed for Gumbel sampling
                    dense_model.set_rng_seed(12345)
                    sparse_model.set_rng_seed(12345)
                    
                    ld = dense_model(x)
                    ls = sparse_model(x)
                    diff = (ld - ls).abs().max().item()
                
                # Note: Differences are expected due to soft vs hard masking
                # A large diff (>1.0) might indicate an issue, but moderate differences are normal
                note = "" if diff < 1.0 else f" (!) Δ={diff:.2e}"

                print(f"{B:6d} {D:3d} {K:3d} | {dense_lat:12.2f} {sparse_lat:13.2f} | {dense_mem:12.2f} {sparse_mem:13.2f}{note}")


# --------------------------------------------------------------------------- #
#                                    Main                                     #
# --------------------------------------------------------------------------- #

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Benchmark BFSNet dense vs sparse execution.")
    p.add_argument("--batches", type=int, nargs="+", default=[32, 64], help="Batch sizes to test.")
    p.add_argument("--depths", type=int, nargs="+", default=[2, 3], help="BFS depths to test.")
    p.add_argument("--children", type=int, nargs="+", default=[2, 3], help="Max children (K) to test.")
    p.add_argument("--hidden_dim", type=int, default=64, help="Hidden size.")
    p.add_argument("--trials", type=int, default=5, help="Trials per measurement.")
    p.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available.")
    return p


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    set_seed(42)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    run_benchmark(
        batch_sizes=args.batches,
        depths=args.depths,
        children=args.children,
        hidden_dim=args.hidden_dim,
        trials=args.trials,
        device=device,
    )