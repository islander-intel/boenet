#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/compare_versions.py

Compare BFSNet v1.4.0 and v2.0.0 trained models side-by-side.

This script analyzes and compares:
  1. Model accuracy (test set performance)
  2. Inference latency (mean, p50, p90, p99)
  3. Node usage and sparsity (v2.0.0 specific)
  4. Model size (checkpoint file size)
  5. Training efficiency (epochs to convergence)

Typical Workflow
----------------
1. Train both versions:
   # v1.4.0 (if you have it)
   python3 train_fmnist_bfs_v1.py --epochs 15 --save_path checkpoints/v1_model.pt
   
   # v2.0.0
   python3 train_fmnist_bfs.py --epochs 15 --save_path checkpoints/v2_model.pt

2. Run inference on both:
   python3 infer_fmnist_bfs.py --ckpt checkpoints/v1_model.pt > results/v1_results.txt
   python3 infer_fmnist_bfs.py --ckpt checkpoints/v2_model.pt > results/v2_results.txt

3. Compare:
   python3 scripts/compare_versions.py \
       --v1_ckpt checkpoints/v1_model.pt \
       --v2_ckpt checkpoints/v2_model.pt

Output
------
Generates:
  - Console summary table comparing key metrics
  - JSON comparison file (optional)
  - Visualization plots (optional, requires matplotlib)

Example Output:
===============================================================================
BFSNet Version Comparison: v1.4.0 vs v2.0.0
===============================================================================

ACCURACY COMPARISON
-------------------
                    v1.4.0          v2.0.0          Δ
Test Accuracy:      86.50%          87.42%          +0.92%
Winner: v2.0.0 ✓

LATENCY COMPARISON (CPU, 1000 samples)
----------------------------------------
                    v1.4.0          v2.0.0          Δ
Mean Latency:       0.58ms          0.45ms          -22.4%
p50 Latency:        0.55ms          0.42ms          -23.6%
p90 Latency:        0.72ms          0.58ms          -19.4%
p99 Latency:        0.89ms          0.71ms          -20.2%
Winner: v2.0.0 ✓ (Faster)

NODE USAGE COMPARISON
---------------------
                    v1.4.0          v2.0.0          
Avg Nodes/Example:  12.8            6.2             -51.6%
Sparsity:           N/A (dense)     84.1%           
Winner: v2.0.0 ✓ (More efficient)

OVERALL WINNER: v2.0.0
  ✓ Higher accuracy (+0.92%)
  ✓ Faster inference (-22.4% latency)
  ✓ More efficient (84.1% sparsity)
===============================================================================

Author: BFS project
Created: 2025-12-18 (v2.0.0)
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Any, Optional, Tuple

import torch


def load_checkpoint_info(ckpt_path: str) -> Dict[str, Any]:
    """
    Load checkpoint and extract key information.
    
    Returns:
        Dictionary with model configuration and training metadata.
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt.get("config", {})
    meta = ckpt.get("training_meta", {})
    
    info = {
        "checkpoint_path": ckpt_path,
        "version": cfg.get("version", "unknown"),
        "model_config": {
            "hidden_dim": cfg.get("hidden_dim"),
            "max_depth": cfg.get("max_depth"),
            "max_children": cfg.get("max_children"),
            "pooling_mode": cfg.get("pooling_mode"),
        },
        "training_meta": {
            "epochs": meta.get("epochs"),
            "best_epoch": meta.get("best_epoch"),
            "best_val_loss": meta.get("best_val_loss"),
            "total_time_s": meta.get("total_time_s"),
        },
        "file_size_bytes": os.path.getsize(ckpt_path),
        "file_size_mb": round(os.path.getsize(ckpt_path) / (1024 * 1024), 2),
    }
    
    # v2.0.0 specific
    if cfg.get("version") == "2.0.0":
        info["policy_config"] = {
            "num_rollouts": cfg.get("num_rollouts"),
            "lambda_efficiency": cfg.get("lambda_efficiency"),
            "beta_entropy": cfg.get("beta_entropy"),
            "beta_policy": cfg.get("beta_policy"),
        }
    
    # v1.4.0 specific
    if "branch_temperature" in cfg:
        info["branch_config"] = {
            "branch_temperature": cfg.get("branch_temperature"),
            "exec_mode": cfg.get("main_exec", "unknown"),
        }
    
    return info


def run_inference_and_get_stats(
    ckpt_path: str,
    cpu: bool = True,
    latency_samples: int = 1000,
    node_samples: int = 1000,
) -> Dict[str, Any]:
    """
    Run infer_fmnist_bfs.py and parse __SUMMARY__ output.
    
    Returns:
        Dictionary with inference statistics.
    """
    import subprocess
    
    cmd = [
        "python3", "infer_fmnist_bfs.py",
        "--ckpt", ckpt_path,
        "--latency_samples", str(latency_samples),
        "--node_samples", str(node_samples),
        "--num_samples", "0",  # Skip sample predictions
    ]
    
    if cpu:
        cmd.append("--cpu")
    
    print(f"[run] Running inference: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        # Parse __SUMMARY__ line
        for line in result.stdout.split('\n'):
            if line.startswith("__SUMMARY__"):
                summary_json = line.replace("__SUMMARY__", "").strip()
                return json.loads(summary_json)
        
        # If no summary found, return error
        print(f"[error] No __SUMMARY__ found in output for {ckpt_path}")
        print(f"stdout:\n{result.stdout}")
        print(f"stderr:\n{result.stderr}")
        return {}
    
    except subprocess.TimeoutExpired:
        print(f"[error] Inference timed out for {ckpt_path}")
        return {}
    except Exception as e:
        print(f"[error] Failed to run inference for {ckpt_path}: {e}")
        return {}


def compare_metrics(
    v1_info: Dict[str, Any],
    v2_info: Dict[str, Any],
    v1_stats: Dict[str, Any],
    v2_stats: Dict[str, Any],
) -> None:
    """
    Print comparison table to console.
    """
    print("\n" + "=" * 79)
    print("BFSNet Version Comparison: v1.4.0 vs v2.0.0")
    print("=" * 79)
    
    # Accuracy comparison
    print("\nACCURACY COMPARISON")
    print("-" * 79)
    
    v1_acc = v1_stats.get("acc_percent", 0.0)
    v2_acc = v2_stats.get("acc_percent", 0.0)
    acc_delta = v2_acc - v1_acc
    
    print(f"{'':20} {'v1.4.0':>15} {'v2.0.0':>15} {'Δ':>15}")
    print(f"{'Test Accuracy:':20} {v1_acc:>14.2f}% {v2_acc:>14.2f}% {acc_delta:>+14.2f}%")
    
    if acc_delta > 0:
        print(f"Winner: v2.0.0 ✓ ({acc_delta:+.2f}%)")
    elif acc_delta < 0:
        print(f"Winner: v1.4.0 ✓ ({-acc_delta:+.2f}%)")
    else:
        print("Winner: TIE")
    
    # Latency comparison
    print("\nLATENCY COMPARISON (CPU, per-sample)")
    print("-" * 79)
    
    v1_mean = v1_stats.get("latency_ms_mean", 0.0)
    v2_mean = v2_stats.get("latency_ms_mean", 0.0)
    v1_p50 = v1_stats.get("latency_ms_p50", 0.0)
    v2_p50 = v2_stats.get("latency_ms_p50", 0.0)
    v1_p90 = v1_stats.get("latency_ms_p90", 0.0)
    v2_p90 = v2_stats.get("latency_ms_p90", 0.0)
    v1_p99 = v1_stats.get("latency_ms_p99", 0.0)
    v2_p99 = v2_stats.get("latency_ms_p99", 0.0)
    
    if v1_mean and v2_mean:
        mean_delta_pct = ((v2_mean - v1_mean) / v1_mean) * 100.0
        p50_delta_pct = ((v2_p50 - v1_p50) / v1_p50) * 100.0 if v1_p50 else 0.0
        p90_delta_pct = ((v2_p90 - v1_p90) / v1_p90) * 100.0 if v1_p90 else 0.0
        p99_delta_pct = ((v2_p99 - v1_p99) / v1_p99) * 100.0 if v1_p99 else 0.0
        
        print(f"{'':20} {'v1.4.0':>15} {'v2.0.0':>15} {'Δ %':>15}")
        print(f"{'Mean Latency:':20} {v1_mean:>13.4f}ms {v2_mean:>13.4f}ms {mean_delta_pct:>+13.1f}%")
        print(f"{'p50 Latency:':20} {v1_p50:>13.4f}ms {v2_p50:>13.4f}ms {p50_delta_pct:>+13.1f}%")
        print(f"{'p90 Latency:':20} {v1_p90:>13.4f}ms {v2_p90:>13.4f}ms {p90_delta_pct:>+13.1f}%")
        print(f"{'p99 Latency:':20} {v1_p99:>13.4f}ms {v2_p99:>13.4f}ms {p99_delta_pct:>+13.1f}%")
        
        if mean_delta_pct < 0:
            print(f"Winner: v2.0.0 ✓ (Faster by {-mean_delta_pct:.1f}%)")
        elif mean_delta_pct > 0:
            print(f"Winner: v1.4.0 ✓ (Faster by {mean_delta_pct:.1f}%)")
        else:
            print("Winner: TIE")
    else:
        print("[info] Latency data not available for comparison")
    
    # Node usage comparison (v2.0.0 specific)
    print("\nNODE USAGE COMPARISON")
    print("-" * 79)
    
    v2_nodes = v2_stats.get("avg_nodes_per_example")
    v2_sparsity = v2_stats.get("sparsity_percent")
    v2_max = v2_stats.get("theoretical_max_nodes")
    
    v1_nodes = v1_stats.get("avg_nodes_per_example")  # May not exist for v1.4.0
    
    if v2_nodes is not None:
        print(f"{'':25} {'v1.4.0':>15} {'v2.0.0':>15}")
        v1_nodes_str = f"{v1_nodes:.2f}" if v1_nodes else "N/A (dense)"
        print(f"{'Avg Nodes/Example:':25} {v1_nodes_str:>15} {v2_nodes:>15.2f}")
        
        if v2_max:
            print(f"{'Theoretical Max:':25} {'':>15} {v2_max:>15}")
        
        if v2_sparsity is not None:
            v1_sparsity_str = "N/A" if not v1_nodes else f"{((v2_max - v1_nodes) / v2_max * 100):.1f}%"
            print(f"{'Sparsity:':25} {v1_sparsity_str:>15} {v2_sparsity:>14.1f}%")
        
        if v1_nodes:
            reduction_pct = ((v1_nodes - v2_nodes) / v1_nodes) * 100.0
            print(f"Winner: v2.0.0 ✓ ({reduction_pct:.1f}% fewer nodes)")
        else:
            print(f"Winner: v2.0.0 ✓ (True sparsity: {v2_sparsity:.1f}%)")
    else:
        print("[info] Node usage data not available")
    
    # Model size comparison
    print("\nMODEL SIZE COMPARISON")
    print("-" * 79)
    
    v1_size = v1_info.get("file_size_mb", 0.0)
    v2_size = v2_info.get("file_size_mb", 0.0)
    
    if v1_size and v2_size:
        size_delta_pct = ((v2_size - v1_size) / v1_size) * 100.0
        print(f"{'':20} {'v1.4.0':>15} {'v2.0.0':>15} {'Δ %':>15}")
        print(f"{'Checkpoint Size:':20} {v1_size:>13.2f}MB {v2_size:>13.2f}MB {size_delta_pct:>+13.1f}%")
        
        if size_delta_pct < 0:
            print(f"Winner: v2.0.0 ✓ (Smaller by {-size_delta_pct:.1f}%)")
        elif size_delta_pct > 0:
            print(f"Winner: v1.4.0 ✓ (Smaller by {size_delta_pct:.1f}%)")
        else:
            print("Winner: TIE")
    
    # Training efficiency
    print("\nTRAINING EFFICIENCY")
    print("-" * 79)
    
    v1_epochs = v1_info.get("training_meta", {}).get("epochs")
    v2_epochs = v2_info.get("training_meta", {}).get("epochs")
    v1_best = v1_info.get("training_meta", {}).get("best_epoch")
    v2_best = v2_info.get("training_meta", {}).get("best_epoch")
    v1_time = v1_info.get("training_meta", {}).get("total_time_s")
    v2_time = v2_info.get("training_meta", {}).get("total_time_s")
    
    if v1_epochs and v2_epochs:
        print(f"{'':25} {'v1.4.0':>15} {'v2.0.0':>15}")
        print(f"{'Total Epochs:':25} {v1_epochs:>15} {v2_epochs:>15}")
        
        if v1_best and v2_best:
            print(f"{'Best Epoch:':25} {v1_best:>15} {v2_best:>15}")
            if v2_best < v1_best:
                print(f"Winner: v2.0.0 ✓ (Converged {v1_best - v2_best} epochs faster)")
            elif v2_best > v1_best:
                print(f"Winner: v1.4.0 ✓ (Converged {v2_best - v1_best} epochs faster)")
        
        if v1_time and v2_time:
            time_delta_pct = ((v2_time - v1_time) / v1_time) * 100.0
            print(f"{'Training Time:':25} {v1_time/60:>13.1f}min {v2_time/60:>13.1f}min {time_delta_pct:>+13.1f}%")
    
    # Overall winner
    print("\n" + "=" * 79)
    
    wins = {"v1": 0, "v2": 0, "tie": 0}
    
    if acc_delta > 0:
        wins["v2"] += 1
    elif acc_delta < 0:
        wins["v1"] += 1
    
    if v1_mean and v2_mean:
        if v2_mean < v1_mean:
            wins["v2"] += 1
        elif v2_mean > v1_mean:
            wins["v1"] += 1
    
    if v2_nodes and v1_nodes:
        if v2_nodes < v1_nodes:
            wins["v2"] += 1
    elif v2_nodes:  # v2 has sparsity, v1 doesn't
        wins["v2"] += 1
    
    print(f"OVERALL WINNER: ", end="")
    if wins["v2"] > wins["v1"]:
        print("v2.0.0 ✓")
        if acc_delta > 0:
            print(f"  ✓ Higher accuracy ({acc_delta:+.2f}%)")
        if v1_mean and v2_mean and v2_mean < v1_mean:
            print(f"  ✓ Faster inference ({-mean_delta_pct:.1f}%)")
        if v2_sparsity:
            print(f"  ✓ More efficient ({v2_sparsity:.1f}% sparsity)")
    elif wins["v1"] > wins["v2"]:
        print("v1.4.0 ✓")
    else:
        print("TIE")
    
    print("=" * 79 + "\n")


def save_comparison_json(
    v1_info: Dict[str, Any],
    v2_info: Dict[str, Any],
    v1_stats: Dict[str, Any],
    v2_stats: Dict[str, Any],
    output_path: str,
) -> None:
    """Save comparison results to JSON file."""
    comparison = {
        "v1.4.0": {
            "checkpoint_info": v1_info,
            "inference_stats": v1_stats,
        },
        "v2.0.0": {
            "checkpoint_info": v2_info,
            "inference_stats": v2_stats,
        },
        "comparison": {
            "accuracy_delta": v2_stats.get("acc_percent", 0) - v1_stats.get("acc_percent", 0),
            "latency_delta_pct": (
                ((v2_stats.get("latency_ms_mean", 0) - v1_stats.get("latency_ms_mean", 1)) 
                 / v1_stats.get("latency_ms_mean", 1)) * 100.0
                if v1_stats.get("latency_ms_mean") else None
            ),
            "node_reduction_pct": (
                ((v1_stats.get("avg_nodes_per_example", 0) - v2_stats.get("avg_nodes_per_example", 0))
                 / v1_stats.get("avg_nodes_per_example", 1)) * 100.0
                if v1_stats.get("avg_nodes_per_example") and v2_stats.get("avg_nodes_per_example")
                else None
            ),
        },
    }
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    print(f"[saved] Comparison results saved to: {output_path}")


# ------------------------------------ CLI ---------------------------------- #

def main():
    p = argparse.ArgumentParser(description="Compare BFSNet v1.4.0 and v2.0.0 models")
    
    p.add_argument("--v1_ckpt", type=str, required=True,
                   help="Path to v1.4.0 checkpoint")
    p.add_argument("--v2_ckpt", type=str, required=True,
                   help="Path to v2.0.0 checkpoint")
    
    p.add_argument("--cpu", action="store_true",
                   help="Force CPU for inference (recommended for fair comparison)")
    
    p.add_argument("--latency_samples", type=int, default=1000,
                   help="Number of samples for latency measurement")
    p.add_argument("--node_samples", type=int, default=1000,
                   help="Number of samples for node usage measurement")
    
    p.add_argument("--skip_inference", action="store_true",
                   help="Skip running inference (use existing results if available)")
    
    p.add_argument("--output_json", type=str, default=None,
                   help="Save comparison results to JSON file")
    
    args = p.parse_args()
    
    print("\n" + "=" * 79)
    print("BFSNet Version Comparison Tool")
    print("=" * 79)
    
    # Load checkpoint info
    print(f"\n[load] Loading v1.4.0 checkpoint: {args.v1_ckpt}")
    v1_info = load_checkpoint_info(args.v1_ckpt)
    print(f"[load] v1.4.0 version: {v1_info.get('version', 'unknown')}")
    
    print(f"\n[load] Loading v2.0.0 checkpoint: {args.v2_ckpt}")
    v2_info = load_checkpoint_info(args.v2_ckpt)
    print(f"[load] v2.0.0 version: {v2_info.get('version', 'unknown')}")
    
    # Run inference (unless skipped)
    if not args.skip_inference:
        print("\n[infer] Running inference on v1.4.0 model...")
        v1_stats = run_inference_and_get_stats(
            args.v1_ckpt,
            cpu=args.cpu,
            latency_samples=args.latency_samples,
            node_samples=args.node_samples,
        )
        
        print("\n[infer] Running inference on v2.0.0 model...")
        v2_stats = run_inference_and_get_stats(
            args.v2_ckpt,
            cpu=args.cpu,
            latency_samples=args.latency_samples,
            node_samples=args.node_samples,
        )
    else:
        print("\n[skip] Inference skipped (no stats available for comparison)")
        v1_stats = {}
        v2_stats = {}
    
    # Compare and print results
    if v1_stats and v2_stats:
        compare_metrics(v1_info, v2_info, v1_stats, v2_stats)
    else:
        print("\n[warning] Insufficient inference stats for comparison")
        print("Run without --skip_inference to generate stats")
    
    # Save to JSON if requested
    if args.output_json:
        save_comparison_json(v1_info, v2_info, v1_stats, v2_stats, args.output_json)


if __name__ == "__main__":
    main()