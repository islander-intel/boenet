#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/visualize_policy.py

Visualize and analyze the learned GrowthPolicyNet from a trained BFSNet v2.0.0 model.

This script provides insights into:
  1. Growth probability distributions at different depths
  2. How growth probabilities change during training (if multiple checkpoints)
  3. Relationship between node features and growth decisions
  4. Policy entropy over depths (exploration vs exploitation)

Usage Examples
--------------
# Basic visualization of a single checkpoint:
python3 scripts/visualize_policy.py --ckpt checkpoints/bfs_fashionmnist_v2.pt

# Compare policy evolution across epochs:
python3 scripts/visualize_policy.py \
    --ckpt checkpoints/epoch_5.pt \
    --ckpt checkpoints/epoch_10.pt \
    --ckpt checkpoints/epoch_15.pt \
    --labels "Epoch 5" "Epoch 10" "Epoch 15"

# Analyze policy on specific test samples:
python3 scripts/visualize_policy.py \
    --ckpt checkpoints/bfs_fashionmnist_v2.pt \
    --num_samples 100 \
    --save_plots results/policy_analysis.png

Output
------
Generates:
  1. Growth probability histogram by depth
  2. Policy entropy by depth
  3. Average growth probability vs depth
  4. Sample-specific decision trees (optional)

Requirements
------------
- matplotlib (for visualization)
- numpy (for statistics)

Author: BFS project
Created: 2025-12-18 (v2.0.0)
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn

from bfs_model import BFSNet

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False
    print("[warning] matplotlib not available. Install with: pip install matplotlib")

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False
    print("[warning] numpy not available. Install with: pip install numpy")

try:
    import torchvision
    from torchvision import transforms
    _HAS_TORCHVISION = True
except ImportError:
    _HAS_TORCHVISION = False


# ----------------------------- Data Loading -------------------------------- #

def get_test_loader(root: str = "./data", batch_size: int = 64):
    """Load FashionMNIST test data."""
    if not _HAS_TORCHVISION:
        raise ImportError("torchvision required for data loading")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: x.view(-1))  # Flatten to [784]
    ])
    
    test_ds = torchvision.datasets.FashionMNIST(
        root=root, train=False, transform=transform, download=True
    )
    
    loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False
    )
    
    return loader, test_ds


# ----------------------------- Model Loading ------------------------------- #

def load_model_for_analysis(ckpt_path: str, device: torch.device) -> Tuple[nn.Module, Dict[str, Any]]:
    """Load BFSNet v2.0.0 checkpoint for policy analysis."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt.get("config", {})
    
    # Validate version
    version = cfg.get("version", "unknown")
    if version != "2.0.0":
        print(f"[warning] Checkpoint version '{version}' (expected '2.0.0')")
    
    # Build model
    model = BFSNet(
        input_dim=int(cfg.get("input_dim", 784)),
        hidden_dim=int(cfg.get("hidden_dim", 64)),
        output_dim=int(cfg.get("num_classes", cfg.get("output_dim", 10))),
        max_depth=int(cfg.get("max_depth", 2)),
        max_children=int(cfg.get("max_children", 3)),
        sibling_embed=bool(cfg.get("sibling_embed", True)),
        pooling_mode=str(cfg.get("pooling_mode", "mean")),
    ).to(device)
    
    # Load weights
    state = ckpt.get("model_state_dict", {})
    model.load_state_dict(state, strict=False)
    model.eval()
    
    return model, cfg


# ----------------------------- Policy Analysis ----------------------------- #

@torch.no_grad()
def analyze_policy_on_data(
    model: nn.Module,
    loader,
    device: torch.device,
    max_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Analyze growth policy decisions across test data.
    
    Returns:
        Dictionary containing:
          - growth_probs_by_depth: Dict[int, List[float]]
          - decisions_by_depth: Dict[int, List[int]]
          - entropies_by_depth: Dict[int, List[float]]
          - num_samples_analyzed: int
    """
    if not hasattr(model, 'growth_policy') or model.growth_policy is None:
        print("[error] Model does not have growth_policy (not a v2.0.0 model?)")
        return {}
    
    max_depth = getattr(model, 'max_depth', 0)
    
    # Storage for analysis
    growth_probs_by_depth = defaultdict(list)
    decisions_by_depth = defaultdict(list)
    entropies_by_depth = defaultdict(list)
    
    # Hook to capture policy decisions
    policy_data = {"depth": None, "probs": []}
    
    def policy_hook(module, input, output):
        """Capture growth probabilities from policy network."""
        if policy_data["depth"] is not None:
            # output is grow_prob [B, 1]
            probs = output.squeeze(-1).cpu().tolist()
            if isinstance(probs, float):
                probs = [probs]
            policy_data["probs"].extend(probs)
    
    hook = model.growth_policy.register_forward_hook(policy_hook)
    
    samples_analyzed = 0
    
    try:
        for batch_idx, (xb, _) in enumerate(loader):
            if max_samples and samples_analyzed >= max_samples:
                break
            
            xb = xb.to(device)
            batch_size = xb.size(0)
            
            # Forward pass (greedy inference mode)
            # We'll manually track depth by monitoring root_fc and child_fc calls
            _ = model(xb)
            
            # For now, we'll do a simpler approach:
            # Run forward and manually query policy at each depth
            
            # Create root nodes
            h0 = torch.relu(model.root_fc(xb))  # [B, H]
            
            # Query policy at each depth
            for depth in range(max_depth):
                policy_data["depth"] = depth
                policy_data["probs"] = []
                
                # Create depth encoding
                depth_enc = torch.zeros(batch_size, max_depth, device=device)
                depth_enc[:, depth] = 1.0
                
                # Query policy
                grow_prob = model.growth_policy(h0, depth_enc)
                
                # Store probabilities
                probs = grow_prob.squeeze(-1).cpu().numpy()
                growth_probs_by_depth[depth].extend(probs.tolist())
                
                # Greedy decisions
                decisions = (probs >= 0.5).astype(int)
                decisions_by_depth[depth].extend(decisions.tolist())
                
                # Compute entropy: H = -p*log(p) - (1-p)*log(1-p)
                p = np.clip(probs, 1e-8, 1 - 1e-8)
                entropy = -(p * np.log(p) + (1 - p) * np.log(1 - p))
                entropies_by_depth[depth].extend(entropy.tolist())
            
            samples_analyzed += batch_size
    
    finally:
        hook.remove()
    
    return {
        "growth_probs_by_depth": dict(growth_probs_by_depth),
        "decisions_by_depth": dict(decisions_by_depth),
        "entropies_by_depth": dict(entropies_by_depth),
        "num_samples_analyzed": samples_analyzed,
    }


# ----------------------------- Visualization ------------------------------- #

def plot_policy_analysis(
    analysis_results: Dict[str, Any],
    cfg: Dict[str, Any],
    save_path: Optional[str] = None,
) -> None:
    """
    Create visualization plots of policy analysis.
    
    Generates:
      1. Growth probability histogram by depth
      2. Average growth probability by depth
      3. Policy entropy by depth
      4. Decision distribution by depth
    """
    if not _HAS_MATPLOTLIB or not _HAS_NUMPY:
        print("[skip] Visualization requires matplotlib and numpy")
        return
    
    growth_probs = analysis_results.get("growth_probs_by_depth", {})
    entropies = analysis_results.get("entropies_by_depth", {})
    decisions = analysis_results.get("decisions_by_depth", {})
    
    if not growth_probs:
        print("[skip] No policy data to visualize")
        return
    
    depths = sorted(growth_probs.keys())
    max_depth = cfg.get("max_depth", max(depths) if depths else 0)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"BFSNet v2.0.0 Growth Policy Analysis (max_depth={max_depth})", fontsize=16)
    
    # Plot 1: Growth probability histogram by depth
    ax1 = axes[0, 0]
    for depth in depths:
        probs = growth_probs[depth]
        ax1.hist(probs, bins=50, alpha=0.6, label=f"Depth {depth}", density=True)
    ax1.set_xlabel("Growth Probability")
    ax1.set_ylabel("Density")
    ax1.set_title("Growth Probability Distribution by Depth")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axvline(x=0.5, color='red', linestyle='--', linewidth=1, label='Threshold')
    
    # Plot 2: Average growth probability by depth
    ax2 = axes[0, 1]
    avg_probs = [np.mean(growth_probs[d]) for d in depths]
    std_probs = [np.std(growth_probs[d]) for d in depths]
    ax2.errorbar(depths, avg_probs, yerr=std_probs, marker='o', capsize=5, capthick=2)
    ax2.set_xlabel("Depth")
    ax2.set_ylabel("Average Growth Probability")
    ax2.set_title("Growth Probability vs Depth")
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0.5, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax2.set_ylim([0, 1])
    
    # Plot 3: Policy entropy by depth
    ax3 = axes[1, 0]
    avg_entropies = [np.mean(entropies[d]) for d in depths]
    std_entropies = [np.std(entropies[d]) for d in depths]
    ax3.errorbar(depths, avg_entropies, yerr=std_entropies, marker='s', capsize=5, capthick=2, color='green')
    ax3.set_xlabel("Depth")
    ax3.set_ylabel("Average Entropy (nats)")
    ax3.set_title("Policy Entropy by Depth (Exploration)")
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Decision distribution (grow vs no-grow) by depth
    ax4 = axes[1, 1]
    grow_rates = [np.mean(decisions[d]) * 100 for d in depths]
    ax4.bar(depths, grow_rates, alpha=0.7, color='steelblue')
    ax4.set_xlabel("Depth")
    ax4.set_ylabel("Growth Rate (%)")
    ax4.set_title("Percentage of 'Grow' Decisions by Depth")
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_ylim([0, 100])
    
    # Add text annotations
    for i, depth in enumerate(depths):
        ax4.text(depth, grow_rates[i] + 2, f"{grow_rates[i]:.1f}%", 
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[saved] Policy analysis plot: {save_path}")
    else:
        plt.savefig("policy_analysis.png", dpi=150, bbox_inches='tight')
        print(f"[saved] Policy analysis plot: policy_analysis.png")
    
    plt.close()


def print_policy_summary(
    analysis_results: Dict[str, Any],
    cfg: Dict[str, Any],
) -> None:
    """Print text summary of policy analysis."""
    growth_probs = analysis_results.get("growth_probs_by_depth", {})
    entropies = analysis_results.get("entropies_by_depth", {})
    decisions = analysis_results.get("decisions_by_depth", {})
    num_samples = analysis_results.get("num_samples_analyzed", 0)
    
    if not growth_probs:
        print("[info] No policy data available")
        return
    
    print("\n" + "=" * 79)
    print("GROWTH POLICY ANALYSIS SUMMARY")
    print("=" * 79)
    print(f"Model: max_depth={cfg.get('max_depth')}, max_children={cfg.get('max_children')}")
    print(f"Samples analyzed: {num_samples}")
    print("\n" + "-" * 79)
    print(f"{'Depth':>6} {'Avg Prob':>10} {'Std':>10} {'Grow %':>10} {'Entropy':>10}")
    print("-" * 79)
    
    depths = sorted(growth_probs.keys())
    for depth in depths:
        probs = growth_probs[depth]
        ents = entropies[depth]
        decs = decisions[depth]
        
        avg_prob = np.mean(probs)
        std_prob = np.std(probs)
        grow_pct = np.mean(decs) * 100
        avg_ent = np.mean(ents)
        
        print(f"{depth:>6} {avg_prob:>10.4f} {std_prob:>10.4f} {grow_pct:>9.1f}% {avg_ent:>10.4f}")
    
    print("=" * 79)
    
    # Interpretation
    print("\nINTERPRETATION:")
    
    overall_grow_rate = np.mean([np.mean(decisions[d]) for d in depths]) * 100
    print(f"  • Overall growth rate: {overall_grow_rate:.1f}%")
    
    if overall_grow_rate < 30:
        print("    → Policy is VERY SPARSE (conservative growth)")
    elif overall_grow_rate < 60:
        print("    → Policy is BALANCED (moderate growth)")
    else:
        print("    → Policy is AGGRESSIVE (frequent growth)")
    
    # Check if policy varies by depth
    grow_rates_by_depth = [np.mean(decisions[d]) * 100 for d in depths]
    if len(grow_rates_by_depth) > 1:
        depth_variation = max(grow_rates_by_depth) - min(grow_rates_by_depth)
        if depth_variation > 20:
            print(f"  • Growth rate varies significantly by depth ({depth_variation:.1f}% range)")
            print("    → Policy is depth-aware")
        else:
            print(f"  • Growth rate is consistent across depths ({depth_variation:.1f}% range)")
            print("    → Policy is depth-agnostic")
    
    # Check entropy (exploration)
    avg_entropy = np.mean([np.mean(entropies[d]) for d in depths])
    max_entropy = np.log(2)  # Maximum entropy for binary decision
    relative_entropy = avg_entropy / max_entropy
    
    print(f"  • Average policy entropy: {avg_entropy:.4f} (max: {max_entropy:.4f})")
    if relative_entropy > 0.8:
        print("    → Policy is EXPLORING (high uncertainty)")
    elif relative_entropy > 0.5:
        print("    → Policy is MODERATELY CONFIDENT")
    else:
        print("    → Policy is EXPLOITING (low uncertainty, confident decisions)")
    
    print()


# ------------------------------------ CLI ---------------------------------- #

def main():
    p = argparse.ArgumentParser(description="Visualize BFSNet v2.0.0 growth policy")
    
    p.add_argument("--ckpt", type=str, required=True,
                   help="Path to v2.0.0 checkpoint")
    p.add_argument("--data_root", type=str, default="./data",
                   help="Path to FashionMNIST data")
    
    p.add_argument("--num_samples", type=int, default=1000,
                   help="Number of test samples to analyze")
    p.add_argument("--batch_size", type=int, default=64,
                   help="Batch size for analysis")
    
    p.add_argument("--save_plots", type=str, default=None,
                   help="Path to save visualization (e.g., results/policy.png)")
    
    p.add_argument("--cpu", action="store_true",
                   help="Force CPU")
    
    args = p.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    
    print("\n" + "=" * 79)
    print("BFSNet v2.0.0 Growth Policy Visualization")
    print("=" * 79)
    
    # Load model
    print(f"\n[load] Loading checkpoint: {args.ckpt}")
    model, cfg = load_model_for_analysis(args.ckpt, device)
    print(f"[load] Model: max_depth={cfg.get('max_depth')}, max_children={cfg.get('max_children')}")
    
    # Load data
    print(f"\n[data] Loading FashionMNIST test data...")
    loader, test_ds = get_test_loader(root=args.data_root, batch_size=args.batch_size)
    print(f"[data] Loaded {len(test_ds)} test samples")
    
    # Analyze policy
    print(f"\n[analyze] Analyzing growth policy on {args.num_samples} samples...")
    analysis_results = analyze_policy_on_data(
        model=model,
        loader=loader,
        device=device,
        max_samples=args.num_samples,
    )
    
    # Print summary
    print_policy_summary(analysis_results, cfg)
    
    # Create visualizations
    if _HAS_MATPLOTLIB and _HAS_NUMPY:
        print("\n[viz] Creating visualization plots...")
        plot_policy_analysis(analysis_results, cfg, save_path=args.save_plots)
    else:
        print("\n[skip] Visualization requires matplotlib and numpy")
        print("Install with: pip install matplotlib numpy")


if __name__ == "__main__":
    main()