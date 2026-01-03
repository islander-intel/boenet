#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_boenet.py (v3.2.0 - True BFS Language Model with Fixed Node Counting)

Train BoeNet on WikiText-2/Shakespeare/TinyStories with True BFS and REINFORCE

v3.2.0 Critical Fix (2026-01-03) - Node Count Metric Reporting
--------------------------------------------------------------
ISSUE FIXED:
  avg_nodes_per_position was showing 0.0 in CSV despite trees actually expanding.
  Logs showed correct expansion (e.g., total_nodes=7) but metrics reported 0.0.

ROOT CAUSE:
  The model's node_counts returns the TREE STRUCTURE node count per rollout
  (e.g., 7 for a depth-2 tree), NOT multiplied by batch positions.
  
  The old calculation was:
    avg_nodes_per_position = total_nodes_across_all_rollouts / (positions_seen * num_rollouts)
    = 15 / (8192 * 3) = 0.0006 ≈ 0.0  # WRONG!
  
  The correct calculation is:
    avg_nodes_per_position = sum(node_counts) / len(node_counts)
    = 15 / 3 = 5.0  # Average tree size across rollouts

FIX APPLIED:
  1. node_counts from model = [tree_size_rollout_0, tree_size_rollout_1, ...]
     where tree_size is 1, 3, 7, 15, 31 for depth 0, 1, 2, 3, 4
  2. avg_nodes_per_position = mean of node_counts across all rollouts in epoch
  3. Updated depth calculation to use node_counts directly (not divided by positions)
  4. Added detailed logging to verify node counting

v3.1.0 Critical Fixes (2026-01-01) - Tree Expansion During Training
-------------------------------------------------------------------
ISSUE FIXED:
  Trees never expanded during training - always showed:
    [True BFS] Avg nodes/position: 0.00 | Avg depth: 0.00
    [Depth Distribution] d0:100.0%

ROOT CAUSES:
  1. Stochastic rollouts not actually creating tree structures
  2. Policy outputs near threshold boundary led to no expansion
  3. Node counting didn't properly aggregate across rollouts
  4. model.py depth embedding dominated policy decisions

FIXES APPLIED:
  1. Added --min_explore_prob parameter (default 0.1) for epsilon-greedy exploration
     This is passed to model.py v2.2.0 which forces expansion with probability 0.1
  2. Added per-epoch expansion rate tracking and logging
  3. Added WARNING if tree never expands in an epoch (indicates problem)
  4. Improved node count aggregation across rollouts
  5. Added debug mode for detailed rollout logging

v3.0.0 Features (Preserved):
  - True BFS level-by-level expansion (not per-node)
  - NaN detection on outputs, policy_loss, and gradients
  - Separate gradient clipping for policy network
  - BFS tree verification logging
  - Updated theoretical max calculations for binary trees
  - Depth tracking and logging

v2.0.1 Features (Preserved):
  - NaN detection helper functions
  - NaN checks after model forward pass
  - Separate gradient clipping for policy network
  - NaN gradient detection before optimizer.step()
  - Early stopping if too many NaN batches occur

TRUE BFS KEY INSIGHT:
  In True BFS, decisions are made per LEVEL, not per node:
  - The policy makes ONE decision per level
  - If level L expands, ALL 2^L nodes at level L create children
  - This guarantees a BALANCED binary tree
  - Gradient paths are O(log n) instead of O(n)

NODE COUNTING (v3.2.0 - CORRECTED):
  The model returns node_counts as a list where each element is the
  TREE STRUCTURE SIZE for that rollout:
  
  - depth=0 (root only): 1 node per position
  - depth=1: 3 nodes per position (1 + 2)
  - depth=2: 7 nodes per position (1 + 2 + 4)
  - depth=3: 15 nodes per position (1 + 2 + 4 + 8)
  - depth=4: 31 nodes per position (1 + 2 + 4 + 8 + 16)
  
  These are NOT multiplied by batch_size * seq_len. They represent
  the tree structure that is applied to EACH token position.

SPARSITY METRICS (v3.2.0):
  - nodes_per_position: Average tree size across rollouts (1, 3, 7, 15, 31)
  - depth_reached: Actual tree depth achieved (0, 1, 2, 3, 4)
  - sparsity_ratio: nodes_used / max_possible_nodes

Training Loop (v3.0.0):
  1. Forward: outputs, policy_loss, rewards, node_counts = model(x, labels=y, ...)
  2. NaN check on outputs and policy_loss
  3. Language modeling loss: CE(outputs, y) where y is shifted input
  4. Total loss: lm_loss + beta_policy * policy_loss
  5. Backward with NaN gradient detection
  6. Separate gradient clipping for policy network
  7. Log: perplexity, policy loss, avg nodes/position, depth reached

Metrics (v3.0.0):
  - Perplexity = exp(cross_entropy_loss)
  - Lower perplexity = better model
  - Random baseline: PPL = vocab_size (256 for char-level)
  - Depth reached: Computed from node count for balanced tree
  - Sparsity: Actual nodes / max possible nodes

Usage Examples:
---------------
# Basic training on WikiText-2 (DEFAULT):
python3 train_boenet.py --epochs 10 --dataset wikitext2

# Training with True BFS depth sweep:
python3 train_boenet.py \\
    --epochs 20 \\
    --max_depth 4 \\
    --max_children 2 \\
    --lambda_efficiency 0.05 \\
    --greedy_threshold 0.42

# Training with v3.1.0 epsilon-greedy exploration:
python3 train_boenet.py \\
    --epochs 20 \\
    --max_depth 3 \\
    --min_explore_prob 0.15

# Training with BFS verification logging:
python3 train_boenet.py \\
    --debug_node_counts \\
    --debug_bfs_verify \\
    --epochs 5

Available Datasets:
-------------------
  wikitext2:   ~2MB Wikipedia (DEFAULT, RECOMMENDED)
  wikitext103: ~500MB Wikipedia
  shakespeare: ~1MB literary text (via GitHub)
  tinystories: ~2GB children's stories
  bookcorpus:  ~5GB books
  openwebtext: ~40GB web text
  textfile:    Custom local text file

Author: BoeNet project
Version: 3.2.0
Date: 2026-01-03
"""

from __future__ import annotations
import os
import sys
import time
import json
import math
import argparse
import logging
from typing import Dict, List, Tuple, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

try:
    import yaml
    _HAS_YAML = True
except Exception:
    _HAS_YAML = False

# Model
from boenet.model import BoeNet

# BFS indexing functions (v3.0.0)
from boenet.model import (
    get_total_nodes_up_to_level,
    get_level,
    get_nodes_at_level,
)

# Try to import get_num_nodes_at_level, fall back to local implementation
try:
    from boenet.model import get_num_nodes_at_level
except ImportError:
    def get_num_nodes_at_level(level: int) -> int:
        """Return number of nodes at a given level in balanced binary tree."""
        return 1 << level  # 2^level

# Data
from boenet.utils.data_utils import (
    get_dataloaders,
    SplitConfig,
    set_seed,
)

# Losses (v2.0.0 True BFS)
from boenet.losses import (
    compute_perplexity,
)

# Try to import True BFS specific functions
try:
    from boenet.losses import (
        compute_rewards_true_bfs,
        compute_depth_from_nodes,
        get_nodes_for_depth,
    )
except ImportError:
    # Fallback implementations
    def compute_depth_from_nodes(num_nodes: int) -> int:
        if num_nodes <= 0:
            return 0
        return max(0, int(math.floor(math.log2(num_nodes + 1))) - 1)
    
    def get_nodes_for_depth(depth: int) -> int:
        return (1 << (depth + 1)) - 1

# Optional: pruning losses (kept for backwards compatibility)
try:
    from boenet.losses import prune_l1, prune_kl_to_rate
except Exception:
    def prune_l1(prune_soft: torch.Tensor) -> torch.Tensor:
        return prune_soft.abs().mean() if prune_soft is not None else torch.tensor(0.0)
    
    def prune_kl_to_rate(prune_soft: torch.Tensor, prior_keep_rate: float = 0.5) -> torch.Tensor:
        if prune_soft is None:
            return torch.tensor(0.0)
        q = prune_soft.view(-1).clamp(1e-8, 1 - 1e-8)
        r = torch.tensor(float(prior_keep_rate), device=q.device, dtype=q.dtype)
        return (q * (q.log() - r.log()) + (1 - q) * ((1 - q).log() - (1 - r).log())).mean()


# --------------------------------------------------------------------------- #
#                         Module-level logger                                 #
# --------------------------------------------------------------------------- #
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#                      NaN Detection Helpers (v2.0.1)                         #
# --------------------------------------------------------------------------- #

def check_for_nan_in_model(model: nn.Module, step_name: str = "") -> bool:
    """
    Check model parameters for NaN values.
    
    Parameters
    ----------
    model : nn.Module
        The model to check.
    step_name : str
        Identifier for logging (e.g., "batch_42").
        
    Returns
    -------
    bool
        True if NaN detected, False otherwise.
    """
    for name, param in model.named_parameters():
        if param is not None and torch.isnan(param).any():
            nan_count = torch.isnan(param).sum().item()
            logger.error(
                f"[NaN DETECTED] {step_name} - Parameter '{name}' has {nan_count} NaN values!"
            )
            return True
    return False


def check_for_nan_in_gradients(model: nn.Module, step_name: str = "") -> bool:
    """
    Check model gradients for NaN values.
    
    Parameters
    ----------
    model : nn.Module
        The model to check.
    step_name : str
        Identifier for logging (e.g., "batch_42").
        
    Returns
    -------
    bool
        True if NaN gradient detected, False otherwise.
    """
    for name, param in model.named_parameters():
        if param.grad is not None and torch.isnan(param.grad).any():
            nan_count = torch.isnan(param.grad).sum().item()
            logger.error(
                f"[NaN GRADIENT] {step_name} - Parameter '{name}' has {nan_count} NaN gradients!"
            )
            return True
    return False


def check_tensor_for_nan(tensor: torch.Tensor, name: str, step_name: str = "") -> bool:
    """
    Check a tensor for NaN values.
    
    Parameters
    ----------
    tensor : torch.Tensor
        The tensor to check.
    name : str
        Name of the tensor for logging.
    step_name : str
        Identifier for logging.
        
    Returns
    -------
    bool
        True if NaN detected, False otherwise.
    """
    if torch.isnan(tensor).any():
        nan_count = torch.isnan(tensor).sum().item()
        logger.warning(
            f"[NaN TENSOR] {step_name} - Tensor '{name}' has {nan_count} NaN values!"
        )
        return True
    return False


# --------------------------------------------------------------------------- #
#                   True BFS Node Counting Helpers (v3.2.0)                   #
# --------------------------------------------------------------------------- #

def compute_depth_from_tree_nodes(tree_nodes: int) -> int:
    """
    Compute the depth of a complete binary tree from its node count.
    
    v3.2.0: This function takes the TREE STRUCTURE node count (1, 3, 7, 15, 31)
    and returns the depth (0, 1, 2, 3, 4).
    
    For a complete binary tree:
    - depth 0: 1 node
    - depth 1: 3 nodes
    - depth 2: 7 nodes
    - depth 3: 15 nodes
    - depth 4: 31 nodes
    
    Formula: nodes = 2^(depth+1) - 1
    Inverse: depth = floor(log2(nodes + 1)) - 1
    
    Parameters
    ----------
    tree_nodes : int
        Number of nodes in the tree structure (1, 3, 7, 15, 31, ...).
        
    Returns
    -------
    int
        Depth of the tree (0 for root only).
    """
    if tree_nodes <= 0:
        return 0
    if tree_nodes == 1:
        return 0
    # For complete binary tree: nodes = 2^(depth+1) - 1
    # So: depth = log2(nodes + 1) - 1
    return max(0, int(math.floor(math.log2(tree_nodes + 1))) - 1)


def compute_theoretical_max_nodes(max_depth: int) -> int:
    """
    Compute theoretical maximum nodes for a complete binary tree.
    
    Parameters
    ----------
    max_depth : int
        Maximum tree depth.
        
    Returns
    -------
    int
        Maximum possible nodes: 2^(max_depth+1) - 1.
    """
    if max_depth < 0:
        return 1
    return (1 << (max_depth + 1)) - 1  # 2^(max_depth+1) - 1


def format_tree_structure(depth: int) -> str:
    """
    Format a visual representation of the tree structure.
    
    Parameters
    ----------
    depth : int
        Tree depth to visualize.
        
    Returns
    -------
    str
        ASCII representation of tree.
    """
    lines = []
    nodes_for_depth = (1 << (depth + 1)) - 1  # get_nodes_for_depth equivalent
    lines.append(f"Depth {depth} Tree (Total: {nodes_for_depth} nodes):")
    
    for level in range(depth + 1):
        num_nodes = 1 << level  # 2^level
        node_indices = list(range((1 << level) - 1, (1 << (level + 1)) - 1))
        lines.append(f"  Level {level}: {num_nodes} nodes (indices {node_indices[0]}-{node_indices[-1]})")
    
    return "\n".join(lines)


def verify_balanced_tree(node_counts: List[int], max_depth: int) -> Dict[str, Any]:
    """
    Verify that node counts are consistent with balanced binary trees.
    
    v3.2.0: Updated to work with tree structure node counts directly.
    
    In True BFS, each rollout should produce a balanced tree where:
    - node count is one of: 1, 3, 7, 15, 31, ... (2^(d+1) - 1)
    
    Parameters
    ----------
    node_counts : List[int]
        Node counts from each rollout (tree structure size, not multiplied by positions).
    max_depth : int
        Maximum tree depth.
        
    Returns
    -------
    Dict[str, Any]
        Verification results including:
        - is_valid: bool
        - depths: List[int] - depth reached per rollout
        - warnings: List[str]
    """
    valid_node_counts = [(1 << (d + 1)) - 1 for d in range(max_depth + 1)]
    
    result = {
        "is_valid": True,
        "depths": [],
        "warnings": [],
    }
    
    for rollout_idx, tree_nodes in enumerate(node_counts):
        # Compute depth from tree node count
        depth = compute_depth_from_tree_nodes(tree_nodes)
        result["depths"].append(depth)
        
        # Check if tree_nodes is a valid balanced tree count
        if tree_nodes not in valid_node_counts:
            # Allow some tolerance for edge cases
            closest_valid = min(valid_node_counts, key=lambda x: abs(x - tree_nodes))
            if tree_nodes != closest_valid:
                result["is_valid"] = False
                result["warnings"].append(
                    f"Rollout {rollout_idx}: tree_nodes={tree_nodes} "
                    f"not a valid balanced tree count {valid_node_counts}"
                )
    
    return result


# --------------------------------------------------------------------------- #
#                                Small utilities                              #
# --------------------------------------------------------------------------- #

def str2bool(v: Optional[str]) -> bool:
    """Convert string to boolean for argparse."""
    if v is None:
        return True
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("true", "t", "yes", "y", "1"):
        return True
    if s in ("false", "f", "no", "n", "0"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


def current_lr(optimizer: torch.optim.Optimizer) -> float:
    """Get current learning rate from optimizer."""
    for pg in optimizer.param_groups:
        return pg.get("lr", 0.0)
    return 0.0


# --------------------------------------------------------------------------- #
#                          Config loading & merging                           #
# --------------------------------------------------------------------------- #

def _load_yaml_config(path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    if not _HAS_YAML:
        raise ImportError("PyYAML not installed. Install: pip install pyyaml")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Config root must be dict; got {type(cfg)}")
    return cfg


def _flatten_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten nested YAML config to single-level dict."""
    flat: Dict[str, Any] = {}
    
    # Direct passthrough
    passthru = [
        "epochs", "lr", "weight_decay", "grad_clip", "seed",
        "hidden_dim", "embed_dim", "max_depth", "max_children", "no_sibling_embed",
        "use_pruning", "pruning_mode", "pruning_threshold",
        "num_workers", "pin_memory", "batch_size", "val_ratio",
        "save_path", "data_root", "cpu",
        "pooling_mode",
        # Language model specific
        "vocab_size", "seq_len", "dataset", "stride", "text_filepath",
        # Policy parameters
        "num_rollouts", "lambda_efficiency", "beta_entropy", "beta_policy",
        "greedy_threshold",
        # v3.1.0: Epsilon-greedy exploration
        "min_explore_prob",
        # Policy gradient clipping
        "policy_grad_clip",
        # Pruning losses
        "prune_l1_weight", "prune_kl_weight", "prune_keep_rate",
        # Optimizer
        "opt", "momentum", "nesterov", "adamw_beta1", "adamw_beta2", "adamw_eps",
        # LR schedule
        "lr_schedule", "lr_step_size", "lr_gamma",
        # Debug (v3.0.0)
        "debug_node_counts", "debug_bfs_verify",
    ]
    for k in passthru:
        if k in cfg:
            flat[k] = cfg[k]
    
    # Nested sections
    nested_map = {
        "training": ["epochs", "lr", "weight_decay", "grad_clip", "seed", "policy_grad_clip"],
        "model": ["hidden_dim", "embed_dim", "max_depth", "max_children", 
                  "no_sibling_embed", "greedy_threshold", "vocab_size", "min_explore_prob"],
        "data": ["seq_len", "dataset", "stride", "batch_size", "val_ratio", "text_filepath"],
        "pruning": ["use_pruning", "pruning_mode", "pruning_threshold"],
        "dataloader": ["num_workers", "pin_memory"],
        "save": ["save_path", "data_root"],
        "run": ["cpu", "pooling_mode"],
        "policy": ["num_rollouts", "lambda_efficiency", "beta_entropy", 
                   "beta_policy", "greedy_threshold", "policy_grad_clip", "min_explore_prob"],
        "losses": ["prune_l1_weight", "prune_kl_weight", "prune_keep_rate"],
        "optimizer": ["opt", "momentum", "nesterov", "adamw_beta1", 
                      "adamw_beta2", "adamw_eps"],
        "lr": ["lr_schedule", "lr_step_size", "lr_gamma"],
        "debug": ["debug_node_counts", "debug_bfs_verify"],
    }
    for section, keys in nested_map.items():
        if section in cfg and isinstance(cfg[section], dict):
            for k in keys:
                if k in cfg[section]:
                    flat[k] = cfg[section][k]
    
    return flat


def _cli_overrides(parser: argparse.ArgumentParser) -> set:
    """Detect which CLI flags were explicitly set."""
    cli_present = set()
    argv = set(sys.argv[1:])
    for action in parser._actions:
        if not action.option_strings:
            continue
        if any(opt in argv for opt in action.option_strings):
            cli_present.add(action.dest)
    return cli_present


def _apply_config(args: argparse.Namespace, cfg: Dict[str, Any], 
                  cli_overrides: set) -> argparse.Namespace:
    """Apply config to args, respecting CLI overrides."""
    flat = _flatten_config(cfg)
    for k, v in flat.items():
        if k in cli_overrides:
            continue
        if hasattr(args, k):
            setattr(args, k, v)
    return args


# --------------------------------------------------------------------------- #
#                               Eval helpers                                  #
# --------------------------------------------------------------------------- #

@torch.no_grad()
def evaluate(
    loader: torch.utils.data.DataLoader,
    model: nn.Module,
    device: torch.device,
    vocab_size: int,
    max_depth: int,
) -> Tuple[float, float, float, float]:
    """
    Evaluate model on validation set.
    
    v3.0.0: Returns additional metrics for True BFS.
    
    Parameters
    ----------
    loader : DataLoader
        Validation data loader.
    model : nn.Module
        BoeNet model.
    device : torch.device
        Device to run on.
    vocab_size : int
        Vocabulary size (for reference).
    max_depth : int
        Maximum tree depth.
    
    Returns
    -------
    Tuple[float, float, float, float]
        (val_loss, val_perplexity, avg_nodes_per_pos, avg_depth)
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    total_nodes = 0
    num_batches = 0
    
    for input_ids, labels in loader:
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        
        # Forward pass (greedy inference)
        logits = model(input_ids)  # [B, seq_len, vocab_size]
        
        # Check for NaN in logits during evaluation
        if torch.isnan(logits).any():
            logger.warning("[Eval] NaN detected in logits. Skipping batch.")
            continue
        
        # Compute loss
        B, seq_len, V = logits.shape
        logits_flat = logits.view(-1, V)  # [B*seq_len, vocab_size]
        labels_flat = labels.view(-1)      # [B*seq_len]
        
        loss = F.cross_entropy(logits_flat, labels_flat, reduction='sum')
        
        # Check for NaN in loss
        if torch.isnan(loss):
            logger.warning("[Eval] NaN detected in loss. Skipping batch.")
            continue
        
        total_loss += loss.item()
        total_tokens += labels.numel()
        
        # For inference, we assume the model uses its internal node counting
        # Since we're in eval mode, the model uses greedy threshold
        # We can estimate nodes from model.max_nodes or just track batches
        num_batches += 1
    
    avg_loss = total_loss / max(1, total_tokens)
    perplexity = compute_perplexity(avg_loss)
    
    # For inference metrics, we'd need to modify the model to return node counts
    # For now, return placeholders
    avg_nodes_per_pos = 0.0  # Would need model modification to track
    avg_depth = 0.0
    
    return avg_loss, perplexity, avg_nodes_per_pos, avg_depth


# --------------------------------------------------------------------------- #
#                                  Training                                   #
# --------------------------------------------------------------------------- #

def train(args: argparse.Namespace, config_path: Optional[str] = None) -> str:
    """
    Main training function for BoeNet language model with True BFS.
    
    v3.2.0 includes:
    - FIXED: Node count metric reporting now correctly interprets model output
    - node_counts from model = tree structure size (1, 3, 7, 15, 31)
    - avg_nodes_per_position = mean(node_counts) across all rollouts
    
    v3.1.0 includes:
    - Epsilon-greedy exploration via min_explore_prob parameter
    - Warning if trees never expand during an epoch
    - Improved expansion tracking and logging
    
    v3.0.0 includes:
    - True BFS node counting for balanced binary trees
    - BFS tree verification logging
    - Depth tracking per rollout
    - Updated sparsity metrics
    
    v2.0.1 features (preserved):
    - NaN detection on outputs, policy_loss, and gradients
    - Separate gradient clipping for policy network
    - Early stopping on repeated NaN batches
    
    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments.
    config_path : Optional[str]
        Path to YAML config file (if used).
        
    Returns
    -------
    str
        Path to saved checkpoint.
    """
    # Configure logging for training
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    
    logger.info(f"[device] Using device: {device}")
    if device.type == "cuda":
        logger.info(f"[device] GPU: {torch.cuda.get_device_name(0)}")
    
    # ==========================================================================
    # v3.2.0: True BFS Information Banner (Updated)
    # ==========================================================================
    print("\n" + "=" * 79)
    print("🌳 BoeNet v3.2.0 - True BFS Language Model (Fixed Node Count Metrics)")
    print("=" * 79)
    print()
    print("TRUE BFS ARCHITECTURE:")
    print("  - Decisions made per LEVEL (not per node)")
    print("  - Balanced binary tree guaranteed")
    print("  - O(log n) gradient paths (not O(n))")
    print()
    print(f"CONFIGURATION:")
    print(f"  max_depth        = {args.max_depth}")
    print(f"  max_children     = {args.max_children} (True BFS uses binary trees)")
    print(f"  greedy_thresh    = {args.greedy_threshold}")
    print(f"  min_explore_prob = {args.min_explore_prob} (v3.1.0 epsilon-greedy)")
    print()
    
    # v3.1.0: Explain epsilon-greedy
    print("v3.1.0 EPSILON-GREEDY EXPLORATION:")
    print(f"  During training, with probability {args.min_explore_prob:.0%}, the model")
    print("  will force tree expansion regardless of policy output.")
    print("  This ensures trees actually expand during training.")
    print()
    
    # v3.2.0: Explain node counting
    print("v3.2.0 NODE COUNT METRICS (FIXED):")
    print("  node_counts from model = tree structure size per rollout")
    print("  Valid values: 1 (d=0), 3 (d=1), 7 (d=2), 15 (d=3), 31 (d=4)")
    print("  avg_nodes_per_position = mean(node_counts) across all rollouts")
    print()
    
    # Show tree structure for configured depth
    theoretical_max = compute_theoretical_max_nodes(args.max_depth)
    print(f"TREE STRUCTURE (max_depth={args.max_depth}):")
    print(f"  Theoretical max nodes per position: {theoretical_max}")
    print()
    for d in range(args.max_depth + 1):
        nodes = (1 << (d + 1)) - 1  # get_nodes_for_depth equivalent
        print(f"    depth={d}: {nodes:3d} nodes  (levels 0-{d})")
    print()
    print("  True BFS guarantees one of these exact node counts per position.")
    print("=" * 79 + "\n")
    # ==========================================================================
    
    # ==========================================================================
    # Data Loading (Language Model)
    # ==========================================================================
    print(f"[data] Loading {args.dataset} dataset...")
    
    # Build kwargs for get_dataloaders
    loader_kwargs = {
        "batch_size": args.batch_size,
        "seed": args.seed,
        "split": SplitConfig(val_ratio=args.val_ratio, shuffle_before_split=True),
        "seq_len": args.seq_len,
        "stride": args.stride if args.stride > 0 else None,
        "dataloader_num_workers": args.num_workers,
        "dataloader_pin_memory": args.pin_memory,
    }
    
    # Add text_filepath if using textfile dataset
    if args.dataset == "textfile" and args.text_filepath:
        loader_kwargs["text_filepath"] = args.text_filepath
    
    train_loader, val_loader, vocab_size = get_dataloaders(args.dataset, **loader_kwargs)
    
    print(f"[data] vocab_size={vocab_size}, seq_len={args.seq_len}")
    print(f"[data] Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Random baseline perplexity
    random_ppl = vocab_size
    print(f"[data] Random baseline perplexity: {random_ppl:.2f}")
    
    # ==========================================================================
    # Model Creation (v3.1.0: Pass min_explore_prob to model)
    # ==========================================================================
    model = BoeNet(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        max_depth=args.max_depth,
        max_children=args.max_children,
        greedy_threshold=args.greedy_threshold,
        min_explore_prob=args.min_explore_prob,  # v3.1.0: Epsilon-greedy
        sibling_embed=not args.no_sibling_embed,
        use_pruning=args.use_pruning,
        pruning_mode=args.pruning_mode,
        pruning_threshold=args.pruning_threshold,
        pooling_mode=args.pooling_mode,
    ).to(device)
    
    # Log model info
    print(f"\n{model.summary()}")
    print(f"[model] Parameters: {model.num_parameters():,}")
    print(f"[model] max_nodes (per position): {model.max_nodes}")
    print(f"[dims] Language Model: vocab_size={vocab_size}, embed_dim={args.embed_dim}, "
          f"hidden_dim={args.hidden_dim}, seq_len={args.seq_len}")
    print(
        f"[run] epochs={args.epochs} | batch_size={args.batch_size} | "
        f"lr={args.lr} | wd={args.weight_decay} | grad_clip={args.grad_clip}\n"
        f"      policy: num_rollouts={args.num_rollouts}, "
        f"λ_eff={args.lambda_efficiency}, β_ent={args.beta_entropy}, "
        f"β_policy={args.beta_policy}\n"
        f"      greedy_threshold={args.greedy_threshold} (inference)\n"
        f"      min_explore_prob={args.min_explore_prob} (v3.1.0 epsilon-greedy)\n"
        f"      policy_grad_clip={args.policy_grad_clip}\n"
        f"      pooling_mode={args.pooling_mode}\n"
        f"      pruning losses: L1*={args.prune_l1_weight}, "
        f"KL*={args.prune_kl_weight} (keep_rate={args.prune_keep_rate})"
    )
    
    # ==========================================================================
    # v3.2.0: True BFS Node Metrics (Updated Explanation)
    # ==========================================================================
    theoretical_max_per_position = compute_theoretical_max_nodes(args.max_depth)
    
    print(
        f"\n[True BFS Node Metrics v3.2.0]\n"
        f"  Theoretical max nodes per position: {theoretical_max_per_position}\n"
        f"  Model returns node_counts = [tree_size_rollout_0, tree_size_rollout_1, ...]\n"
        f"  Each tree_size is one of: {[(1 << (d + 1)) - 1 for d in range(args.max_depth + 1)]}\n"
        f"  avg_nodes_per_position = mean(all node_counts in epoch)"
    )
    
    # Show valid node counts for debugging
    print(f"\n  Valid tree sizes (balanced binary tree):")
    for d in range(args.max_depth + 1):
        nodes = (1 << (d + 1)) - 1
        print(f"    depth={d}: {nodes} nodes")
    print()
    
    # ==========================================================================
    # Optimizer & Scheduler
    # ==========================================================================
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(args.adamw_beta1, args.adamw_beta2),
        eps=args.adamw_eps,
    ) if args.opt == "adamw" else torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        nesterov=args.nesterov,
    )
    
    # Scheduler
    scheduler = None
    if args.lr_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=0.0
        )
    elif args.lr_schedule == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma
        )
    
    print(f"[opt] {args.opt} | lr={args.lr} | wd={args.weight_decay}")
    print(f"[lr] schedule={args.lr_schedule}")
    
    best_val_loss = float("inf")
    best_val_ppl = float("inf")
    best_epoch = 0
    epoch_times_s: List[float] = []
    t0_total = time.perf_counter()
    
    # ==========================================================================
    # Identify policy parameters for separate gradient clipping
    # ==========================================================================
    policy_param_names = set()
    policy_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if 'growth_policy' in name:
            policy_param_names.add(name)
            policy_params.append(param)
        else:
            other_params.append(param)
    
    logger.info(f"[Policy Params] {len(policy_params)} parameters: {list(policy_param_names)}")
    logger.info(f"[Other Params] {len(other_params)} parameters")
    
    # ==========================================================================
    # Training Loop
    # ==========================================================================
    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        eff_lr = current_lr(optimizer)
        
        print(f"\n[epoch {epoch}/{args.epochs}] lr={eff_lr:.6f}")
        
        model.train()
        total_loss = 0.0
        total_lm_loss = 0.0
        total_policy_loss = 0.0
        total_tokens = 0
        
        # ==========================================================================
        # v3.2.0: FIXED Node Counting
        # ==========================================================================
        # Collect ALL node_counts from ALL rollouts in the epoch
        # Each node_count is the TREE SIZE (1, 3, 7, 15, 31), NOT multiplied by positions
        all_tree_sizes_this_epoch: List[int] = []
        
        # Track depth distribution (v3.0.0)
        depth_counts = {d: 0 for d in range(args.max_depth + 1)}
        
        # v3.1.0: Track if any expansion happened this epoch
        expansion_happened_this_epoch = False
        
        # NaN tracking for early stopping
        nan_batch_count = 0
        max_nan_batches = max(len(train_loader) // 10, 5)  # Allow up to 10% NaN batches
        skipped_batches = 0
        
        # Debug collections
        if args.debug_node_counts:
            debug_node_counts_per_batch = []
        
        # BFS verification (v3.0.0)
        bfs_verification_warnings = []
        
        for batch_idx, (input_ids, labels) in enumerate(train_loader):
            input_ids = input_ids.to(device)  # [B, seq_len]
            labels = labels.to(device)        # [B, seq_len]
            
            optimizer.zero_grad()
            
            # ==================================================================
            # FORWARD with NaN detection
            # ==================================================================
            try:
                outputs, policy_loss, rewards, node_counts = model(
                    input_ids,
                    num_rollouts=args.num_rollouts,
                    lambda_efficiency=args.lambda_efficiency,
                    beta_entropy=args.beta_entropy,
                    labels=labels,
                )
            except RuntimeError as e:
                if "CUDA" in str(e) or "assert" in str(e).lower():
                    logger.error(f"[Batch {batch_idx}] CUDA error during forward: {e}")
                    nan_batch_count += 1
                    skipped_batches += 1
                    if nan_batch_count > max_nan_batches:
                        logger.error(f"[Epoch {epoch}] Too many errors ({nan_batch_count}). Stopping.")
                        break
                    continue
                else:
                    raise
            
            # outputs: [B, seq_len, vocab_size]
            B = input_ids.size(0)
            seq_len_batch = input_ids.size(1)
            
            # ==================================================================
            # Check for NaN in outputs
            # ==================================================================
            if check_tensor_for_nan(outputs, "outputs", f"batch_{batch_idx}"):
                logger.warning(f"[Batch {batch_idx}] NaN in outputs. Skipping batch.")
                nan_batch_count += 1
                skipped_batches += 1
                if nan_batch_count > max_nan_batches:
                    logger.error(f"[Epoch {epoch}] Too many NaN batches ({nan_batch_count}). Stopping.")
                    break
                continue
            
            # ==================================================================
            # Check for NaN in policy loss
            # ==================================================================
            if torch.isnan(policy_loss):
                logger.warning(f"[Batch {batch_idx}] NaN policy loss. Using zero.")
                policy_loss = torch.tensor(0.0, device=device, requires_grad=True)
                nan_batch_count += 1
            
            # Language modeling loss (cross-entropy)
            V = outputs.size(-1)
            outputs_flat = outputs.view(-1, V)     # [B*seq_len, vocab_size]
            labels_flat = labels.view(-1)          # [B*seq_len]
            lm_loss = F.cross_entropy(outputs_flat, labels_flat)
            
            # ==================================================================
            # Check for NaN in LM loss
            # ==================================================================
            if torch.isnan(lm_loss):
                logger.warning(f"[Batch {batch_idx}] NaN LM loss. Skipping batch.")
                nan_batch_count += 1
                skipped_batches += 1
                if nan_batch_count > max_nan_batches:
                    logger.error(f"[Epoch {epoch}] Too many NaN batches ({nan_batch_count}). Stopping.")
                    break
                continue
            
            # Total loss
            total_loss_batch = lm_loss + args.beta_policy * policy_loss
            
            # Backward
            total_loss_batch.backward()
            
            # ==================================================================
            # Check for NaN in gradients BEFORE clipping
            # ==================================================================
            if check_for_nan_in_gradients(model, f"batch_{batch_idx}"):
                logger.warning(f"[Batch {batch_idx}] NaN gradients. Zeroing and skipping.")
                optimizer.zero_grad()
                nan_batch_count += 1
                skipped_batches += 1
                if nan_batch_count > max_nan_batches:
                    logger.error(f"[Epoch {epoch}] Too many NaN batches ({nan_batch_count}). Stopping.")
                    break
                continue
            
            # ==================================================================
            # Separate gradient clipping for policy network
            # ==================================================================
            if args.grad_clip and args.grad_clip > 0:
                # Clip policy gradients more aggressively
                if policy_params and args.policy_grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(policy_params, args.policy_grad_clip)
                
                # Clip other gradients normally
                if other_params:
                    torch.nn.utils.clip_grad_norm_(other_params, args.grad_clip)
            
            # ==================================================================
            # Final NaN check after clipping
            # ==================================================================
            has_nan_after_clip = False
            for name, param in model.named_parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    has_nan_after_clip = True
                    break
            
            if has_nan_after_clip:
                logger.warning(f"[Batch {batch_idx}] NaN still present after clipping. Skipping.")
                optimizer.zero_grad()
                nan_batch_count += 1
                skipped_batches += 1
                continue
            
            optimizer.step()
            
            # ==================================================================
            # Book-keeping
            # ==================================================================
            num_tokens = labels.numel()
            total_loss += total_loss_batch.item() * num_tokens
            total_lm_loss += lm_loss.item() * num_tokens
            total_policy_loss += policy_loss.item() * num_tokens
            total_tokens += num_tokens
            
            # ==================================================================
            # v3.2.0: FIXED Node Counting
            # ==================================================================
            # node_counts is a list of tree sizes, one per rollout
            # e.g., [1, 7, 3] means rollout 0 had 1 node (depth 0),
            #       rollout 1 had 7 nodes (depth 2), rollout 2 had 3 nodes (depth 1)
            all_tree_sizes_this_epoch.extend(node_counts)
            
            # Track depth for each rollout
            for tree_size in node_counts:
                depth = compute_depth_from_tree_nodes(tree_size)
                depth = min(depth, args.max_depth)  # Clamp to max_depth
                depth_counts[depth] += 1
                
                # v3.1.0: Check if expansion happened
                if depth > 0:
                    expansion_happened_this_epoch = True
            
            # ==================================================================
            # v3.0.0: BFS Tree Verification (optional)
            # ==================================================================
            if args.debug_bfs_verify:
                verification = verify_balanced_tree(node_counts, args.max_depth)
                if not verification["is_valid"]:
                    bfs_verification_warnings.extend(verification["warnings"])
            
            # ==================================================================
            # Debug logging for first few batches
            # ==================================================================
            if args.debug_node_counts:
                debug_node_counts_per_batch.append(node_counts)
                
                if batch_idx < 3:  # Log first 3 batches in detail
                    print(
                        f"\n[debug v3.2.0 batch={batch_idx}]\n"
                        f"  node_counts (tree sizes per rollout): {node_counts}\n"
                        f"  batch_size: {B}, seq_len: {seq_len_batch}\n"
                        f"  num_rollouts: {len(node_counts)}\n"
                        f"  mean tree size: {sum(node_counts) / len(node_counts):.2f}\n"
                        f"  theoretical max tree size: {theoretical_max_per_position}\n"
                    )
                    
                    # Show depth per rollout
                    for r_idx, tree_size in enumerate(node_counts):
                        d = compute_depth_from_tree_nodes(tree_size)
                        print(f"    Rollout {r_idx}: tree_size={tree_size} → depth={d}")
                    print()
        
        # ==================================================================
        # Check if we need to stop early due to NaN
        # ==================================================================
        if nan_batch_count > max_nan_batches:
            logger.error(f"[Epoch {epoch}] Stopping training due to too many NaN batches.")
            break
        
        if skipped_batches > 0:
            logger.warning(f"[Epoch {epoch}] Skipped {skipped_batches} batches due to NaN.")
        
        # Scheduler step
        if scheduler is not None:
            scheduler.step()
        
        # ==================================================================
        # Epoch metrics
        # ==================================================================
        if total_tokens > 0:
            train_loss = total_loss / total_tokens
            train_lm_loss = total_lm_loss / total_tokens
            train_policy_loss = total_policy_loss / total_tokens
            train_ppl = compute_perplexity(train_lm_loss)
        else:
            logger.error(f"[Epoch {epoch}] No tokens processed. All batches skipped.")
            train_loss = float('inf')
            train_lm_loss = float('inf')
            train_policy_loss = float('inf')
            train_ppl = float('inf')
        
        # ==================================================================
        # v3.2.0: FIXED Sparsity Metrics
        # ==================================================================
        if len(all_tree_sizes_this_epoch) > 0:
            # Average tree size across all rollouts in the epoch
            avg_nodes_per_position = sum(all_tree_sizes_this_epoch) / len(all_tree_sizes_this_epoch)
        else:
            avg_nodes_per_position = 0.0
        
        # Sparsity ratio: actual / theoretical max
        sparsity_pct = (avg_nodes_per_position / theoretical_max_per_position) * 100 if theoretical_max_per_position > 0 else 0.0
        
        # Compute average depth reached
        total_rollouts = sum(depth_counts.values())
        if total_rollouts > 0:
            avg_depth_reached = sum(d * c for d, c in depth_counts.items()) / total_rollouts
        else:
            avg_depth_reached = 0.0
        
        # ==================================================================
        # Validation
        # ==================================================================
        val_loss, val_ppl, _, _ = evaluate(val_loader, model, device, vocab_size, args.max_depth)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_ppl = val_ppl
            best_epoch = epoch
        
        # ==================================================================
        # Print epoch summary (v3.2.0 format)
        # ==================================================================
        print(
            f"  Train Loss: {train_loss:.4f} (lm={train_lm_loss:.4f}, "
            f"policy={train_policy_loss:.4f}) | Val Loss: {val_loss:.4f}\n"
            f"  Train PPL: {train_ppl:.2f} | Val PPL: {val_ppl:.2f}"
        )
        print(
            f"  [True BFS v3.2.0] Avg tree size: {avg_nodes_per_position:.2f} | "
            f"Avg depth: {avg_depth_reached:.2f} | "
            f"Sparsity: {sparsity_pct:.1f}% of max {theoretical_max_per_position}"
        )
        
        # Print depth distribution
        print(f"  [Depth Distribution] ", end="")
        for d in range(args.max_depth + 1):
            if depth_counts[d] > 0:
                pct = 100.0 * depth_counts[d] / total_rollouts
                print(f"d{d}:{pct:.1f}% ", end="")
        print()
        
        # ==================================================================
        # v3.1.0: Warning if no expansion happened
        # ==================================================================
        if not expansion_happened_this_epoch and args.max_children > 0 and args.max_depth > 0:
            print(
                f"  ⚠️  WARNING: No tree expansion occurred this epoch!\n"
                f"      Trees stayed at depth=0 for all {total_rollouts} rollouts.\n"
                f"      This may indicate:\n"
                f"        - min_explore_prob={args.min_explore_prob} is too low\n"
                f"        - Policy is saturated to 'no expand' decisions\n"
                f"      Consider increasing --min_explore_prob (e.g., 0.15 or 0.2)"
            )
        
        # Log NaN statistics
        if nan_batch_count > 0:
            print(f"  [NaN] Batches with issues: {nan_batch_count}, Skipped: {skipped_batches}")
        
        # BFS verification warnings
        if args.debug_bfs_verify and bfs_verification_warnings:
            print(f"  [BFS Verify] {len(bfs_verification_warnings)} warnings:")
            for w in bfs_verification_warnings[:5]:  # Show first 5
                print(f"    - {w}")
            if len(bfs_verification_warnings) > 5:
                print(f"    ... and {len(bfs_verification_warnings) - 5} more")
        
        # Debug node statistics (v3.2.0)
        if args.debug_node_counts and all_tree_sizes_this_epoch:
            import statistics
            min_tree = min(all_tree_sizes_this_epoch)
            max_tree = max(all_tree_sizes_this_epoch)
            median_tree = statistics.median(all_tree_sizes_this_epoch)
            stddev_tree = statistics.stdev(all_tree_sizes_this_epoch) if len(all_tree_sizes_this_epoch) > 1 else 0
            
            print(
                f"  [Tree Size Stats] min={min_tree}, max={max_tree}, "
                f"median={median_tree:.1f}, stddev={stddev_tree:.1f}, "
                f"count={len(all_tree_sizes_this_epoch)}"
            )
            
            # Show value distribution
            tree_size_dist = {}
            for ts in all_tree_sizes_this_epoch:
                tree_size_dist[ts] = tree_size_dist.get(ts, 0) + 1
            print(f"  [Tree Size Distribution] {tree_size_dist}")
        
        epoch_time = time.perf_counter() - t0
        epoch_times_s.append(float(epoch_time))
        
        # ==================================================================
        # Check model parameters for NaN at end of epoch
        # ==================================================================
        if check_for_nan_in_model(model, f"epoch_{epoch}_end"):
            logger.error(f"[Epoch {epoch}] Model parameters contain NaN. Stopping training.")
            break
    
    total_time_s = time.perf_counter() - t0_total
    
    # ==========================================================================
    # Save Checkpoint
    # ==========================================================================
    save_dir = os.path.dirname(args.save_path) or "."
    os.makedirs(save_dir, exist_ok=True)
    
    ckpt = {
        "model_state_dict": model.state_dict(),
        "config": {
            # Language model specific
            "vocab_size": vocab_size,
            "embed_dim": args.embed_dim,
            "hidden_dim": args.hidden_dim,
            "seq_len": args.seq_len,
            "dataset": args.dataset,
            # Architecture
            "max_depth": args.max_depth,
            "max_children": args.max_children,
            "greedy_threshold": args.greedy_threshold,
            "min_explore_prob": args.min_explore_prob,  # v3.1.0
            "sibling_embed": not args.no_sibling_embed,
            "use_pruning": args.use_pruning,
            "pruning_mode": args.pruning_mode,
            "pruning_threshold": args.pruning_threshold,
            "pooling_mode": args.pooling_mode,
            # Policy parameters
            "num_rollouts": args.num_rollouts,
            "lambda_efficiency": args.lambda_efficiency,
            "beta_entropy": args.beta_entropy,
            "beta_policy": args.beta_policy,
            "policy_grad_clip": args.policy_grad_clip,
            # v3.2.0 additions
            "version": "3.2.0",
            "model_type": "language",
            "bfs_type": "true_bfs",
            "theoretical_max_nodes": theoretical_max_per_position,
        },
        "training_meta": {
            "best_val_loss": best_val_loss,
            "best_val_ppl": best_val_ppl,
            "best_epoch": best_epoch,
            "epochs": args.epochs,
            "epoch_times_s": epoch_times_s,
            "total_time_s": total_time_s,
        },
    }
    torch.save(ckpt, args.save_path)
    print(f"\nSaved checkpoint → {args.save_path}")
    print(f"Best val loss: {best_val_loss:.4f} (PPL: {best_val_ppl:.2f}) at epoch {best_epoch}")
    
    # ==========================================================================
    # POST-TRAINING RECOMMENDATIONS
    # ==========================================================================
    print("\n" + "=" * 79)
    print("POST-TRAINING: True BFS Analysis and Recommendations")
    print("=" * 79)
    print()
    print("Your model was trained with True BFS v3.2.0:")
    print(f"  dataset           = {args.dataset}")
    print(f"  max_depth         = {args.max_depth}")
    print(f"  lambda_efficiency = {args.lambda_efficiency}")
    print(f"  greedy_threshold  = {args.greedy_threshold}")
    print(f"  min_explore_prob  = {args.min_explore_prob} (v3.1.0 epsilon-greedy)")
    print(f"  best_val_ppl      = {best_val_ppl:.2f}")
    print()
    print("TRUE BFS v3.2.0 FEATURES:")
    print("  ✓ Balanced binary trees (no uneven growth)")
    print("  ✓ O(log n) gradient paths (stable training)")
    print("  ✓ Predictable node counts (powers of 2 minus 1)")
    print("  ✓ Epsilon-greedy exploration ensures tree expansion during training")
    print("  ✓ Fixed node count metrics (v3.2.0)")
    print()
    print("NEXT STEPS:")
    print()
    print("1. Run inference with depth analysis:")
    print(f"   python3 infer_boenet.py --ckpt {args.save_path} \\")
    print(f"       --debug_policy --node_samples 1000")
    print()
    print("2. Generate text:")
    print(f"   python3 infer_boenet.py --ckpt {args.save_path} \\")
    print(f"       --generate --max_tokens 200 --temperature 0.8")
    print()
    print("3. Compare to baseline:")
    print(f"   Random PPL = {random_ppl:.2f}")
    print(f"   Your PPL   = {best_val_ppl:.2f}")
    if best_val_ppl > 0 and best_val_ppl < float('inf'):
        print(f"   Improvement: {(random_ppl / best_val_ppl):.2f}x better")
    print()
    print("=" * 79 + "\n")
    # ==========================================================================
    
    # Summary JSON
    summary = {
        "run": {
            "seed": int(args.seed),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "dataset": args.dataset,
        },
        "optimizer": {
            "name": args.opt,
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
        },
        "model": {
            "vocab_size": int(vocab_size),
            "embed_dim": int(args.embed_dim),
            "hidden_dim": int(args.hidden_dim),
            "seq_len": int(args.seq_len),
            "max_depth": int(args.max_depth),
            "max_children": int(args.max_children),
            "pooling_mode": args.pooling_mode,
            "greedy_threshold": float(args.greedy_threshold),
            "min_explore_prob": float(args.min_explore_prob),  # v3.1.0
            "theoretical_max_nodes": int(theoretical_max_per_position),
        },
        "policy": {
            "num_rollouts": int(args.num_rollouts),
            "lambda_efficiency": float(args.lambda_efficiency),
            "beta_entropy": float(args.beta_entropy),
            "beta_policy": float(args.beta_policy),
            "policy_grad_clip": float(args.policy_grad_clip),
        },
        "results": {
            "best_val_loss": float(best_val_loss) if best_val_loss != float('inf') else None,
            "best_val_ppl": float(best_val_ppl) if best_val_ppl != float('inf') else None,
            "best_epoch": int(best_epoch),
            "random_baseline_ppl": float(random_ppl),
        },
        "true_bfs": {
            "avg_nodes_per_position": float(avg_nodes_per_position) if 'avg_nodes_per_position' in dir() else None,
            "avg_depth_reached": float(avg_depth_reached) if 'avg_depth_reached' in dir() else None,
            "sparsity_pct": float(sparsity_pct) if 'sparsity_pct' in dir() else None,
        },
        "time": {
            "total_s": float(total_time_s),
            "epoch_avg_s": float(sum(epoch_times_s) / len(epoch_times_s)) if epoch_times_s else 0.0,
            "last_epoch_s": float(epoch_times_s[-1]) if epoch_times_s else 0.0,
        },
        "artifacts": {
            "checkpoint": args.save_path,
        },
        "version": "3.2.0",
    }
    
    try:
        print("__SUMMARY__ " + json.dumps(summary, ensure_ascii=False), flush=True)
    except Exception:
        print("__SUMMARY__ " + json.dumps(summary, ensure_ascii=True), flush=True)
    
    return args.save_path


# --------------------------------------------------------------------------- #
#                                     CLI                                     #
# --------------------------------------------------------------------------- #

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train BoeNet v3.2.0 True BFS Language Model with REINFORCE",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
---------
# Train on WikiText-2 (DEFAULT):
python3 train_boenet.py --epochs 10 --dataset wikitext2

# Train with True BFS depth 4:
python3 train_boenet.py \\
    --epochs 20 \\
    --max_depth 4 \\
    --max_children 2 \\
    --lambda_efficiency 0.05

# Train with v3.1.0 epsilon-greedy exploration:
python3 train_boenet.py \\
    --epochs 20 \\
    --max_depth 3 \\
    --min_explore_prob 0.15

# Train with BFS verification logging:
python3 train_boenet.py \\
    --debug_node_counts \\
    --debug_bfs_verify \\
    --epochs 5

True BFS Architecture:
----------------------
  v3.2.0 uses True BFS with per-LEVEL decisions:
  - Decisions made per level, not per node
  - Balanced binary tree guaranteed
  - O(log n) gradient paths
  - Epsilon-greedy exploration ensures tree expansion during training
  
  Node counts for balanced trees:
    depth=0: 1 node   (root only)
    depth=1: 3 nodes  (1 + 2)
    depth=2: 7 nodes  (1 + 2 + 4)
    depth=3: 15 nodes (1 + 2 + 4 + 8)
    depth=4: 31 nodes (1 + 2 + 4 + 8 + 16)

  v3.2.0 FIX: Node count metrics now correctly report tree size
  (previously showed 0.0 due to incorrect calculation)

Available Datasets:
-------------------
  wikitext2:   ~2MB Wikipedia (DEFAULT, RECOMMENDED)
  wikitext103: ~500MB Wikipedia
  shakespeare: ~1MB literary text
  tinystories: ~2GB children's stories
  textfile:    Custom local text file

See docs/architecture.md for detailed analysis.
        """
    )
    
    # Config
    p.add_argument("--config", type=str, default=None,
                   help="Path to YAML config")
    
    # Data (Language Model Specific)
    p.add_argument("--dataset", type=str, default="wikitext2",
                   choices=["wikitext2", "wikitext103", "shakespeare", "tinystories", 
                            "bookcorpus", "openwebtext", "textfile"],
                   help="Dataset to use for training (default: wikitext2)")
    p.add_argument("--text_filepath", type=str, default=None,
                   help="Path to text file (for --dataset textfile)")
    p.add_argument("--data_root", type=str, default="./data",
                   help="Root directory for data caching")
    p.add_argument("--val_ratio", type=float, default=0.1,
                   help="Fraction of data for validation")
    p.add_argument("--batch_size", type=int, default=64,
                   help="Batch size for training")
    p.add_argument("--seq_len", type=int, default=128,
                   help="Sequence length for language modeling")
    p.add_argument("--stride", type=int, default=0,
                   help="Stride between samples (0 = seq_len, non-overlapping)")
    p.add_argument("--num_workers", type=int, default=0,
                   help="Number of data loader workers")
    p.add_argument("--pin_memory", type=str2bool, nargs="?", const=True, default=False,
                   help="Pin memory for data loader")
    
    # Model Architecture
    p.add_argument("--embed_dim", type=int, default=64,
                   help="Token embedding dimension")
    p.add_argument("--hidden_dim", type=int, default=128,
                   help="Hidden dimension for BFS tree nodes")
    p.add_argument("--max_depth", type=int, default=2,
                   help="Maximum BFS tree depth (True BFS uses binary trees)")
    p.add_argument("--max_children", type=int, default=2,
                   help="Maximum children per node (True BFS uses 2)")
    p.add_argument("--no_sibling_embed", type=str2bool, nargs="?", const=True, default=False,
                   help="Disable sibling embeddings")
    p.add_argument("--pooling_mode", type=str, default="mean", 
                   choices=["mean", "sum", "learned"],
                   help="Pooling mode for node aggregation")
    
    # Pruning
    p.add_argument("--use_pruning", type=str2bool, nargs="?", const=True, default=False,
                   help="Enable pruning")
    p.add_argument("--pruning_mode", type=str, default="learned", 
                   choices=["learned", "threshold"],
                   help="Pruning mode")
    p.add_argument("--pruning_threshold", type=float, default=1e-3,
                   help="Pruning threshold")
    
    # Policy Gradient Parameters
    p.add_argument("--num_rollouts", type=int, default=3,
                   help="Number of rollouts per input for exploration (1-5)")
    p.add_argument("--lambda_efficiency", type=float, default=0.05,
                   help="Node count penalty in reward (0.0-0.1)")
    p.add_argument("--beta_entropy", type=float, default=0.01,
                   help="Entropy bonus in policy loss (0.001-0.01)")
    p.add_argument("--beta_policy", type=float, default=0.5,
                   help="Policy loss weight in total loss (0.1-1.0)")
    p.add_argument("--greedy_threshold", type=float, default=0.5,
                   help="Threshold for greedy inference decisions (0.3-0.5)")
    
    # v3.1.0: Epsilon-greedy exploration
    p.add_argument("--min_explore_prob", type=float, default=0.1,
                   help="v3.1.0: Minimum probability to force tree expansion during training (0.0-0.3)")
    
    # Policy gradient clipping
    p.add_argument("--policy_grad_clip", type=float, default=0.5,
                   help="Gradient clipping for policy network (default: 0.5)")
    
    # Pruning losses
    p.add_argument("--prune_l1_weight", type=float, default=0.0,
                   help="Weight for L1 pruning loss")
    p.add_argument("--prune_kl_weight", type=float, default=0.0,
                   help="Weight for KL pruning loss")
    p.add_argument("--prune_keep_rate", type=float, default=0.5,
                   help="Target keep rate for KL pruning loss")
    
    # Optimization
    p.add_argument("--opt", type=str, default="adamw", choices=["adamw", "sgd"],
                   help="Optimizer type")
    p.add_argument("--lr", type=float, default=1e-3,
                   help="Learning rate")
    p.add_argument("--weight_decay", type=float, default=0.0,
                   help="Weight decay")
    p.add_argument("--grad_clip", type=float, default=1.0,
                   help="Gradient clipping norm (0 to disable)")
    
    # SGD-specific
    p.add_argument("--momentum", type=float, default=0.9,
                   help="SGD momentum")
    p.add_argument("--nesterov", type=str2bool, nargs="?", const=True, default=True,
                   help="Use Nesterov momentum")
    
    # AdamW-specific
    p.add_argument("--adamw_beta1", type=float, default=0.9,
                   help="AdamW beta1")
    p.add_argument("--adamw_beta2", type=float, default=0.999,
                   help="AdamW beta2")
    p.add_argument("--adamw_eps", type=float, default=1e-8,
                   help="AdamW epsilon")
    
    # LR schedule
    p.add_argument("--lr_schedule", type=str, default="cosine", 
                   choices=["none", "cosine", "step"],
                   help="Learning rate schedule")
    p.add_argument("--lr_step_size", type=int, default=5,
                   help="Step size for StepLR")
    p.add_argument("--lr_gamma", type=float, default=0.5,
                   help="Gamma for StepLR")
    
    p.add_argument("--epochs", type=int, default=10,
                   help="Number of training epochs")
    
    # System
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed")
    p.add_argument("--cpu", type=str2bool, nargs="?", const=True, default=False,
                   help="Force CPU training")
    p.add_argument("--save_path", type=str, default="checkpoints/boenet_wikitext2.pt",
                   help="Path to save checkpoint")
    
    # Debug (v3.0.0)
    p.add_argument("--debug_node_counts", type=str2bool, nargs="?", const=True, default=False,
                   help="Enable detailed node count logging")
    p.add_argument("--debug_bfs_verify", type=str2bool, nargs="?", const=True, default=False,
                   help="Enable BFS tree verification (checks balanced tree property)")
    
    return p


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    
    cfg_path = None
    if args.config is not None:
        cfg_path = args.config
        cfg = _load_yaml_config(cfg_path)
        cli_overrides = _cli_overrides(parser)
        cli_overrides.discard("config")
        args = _apply_config(args, cfg, cli_overrides)
    
    train(args, config_path=cfg_path)