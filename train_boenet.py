#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_boenet.py (v2.0.0 - Language Model)

Train BoeNet on WikiText-2/Shakespeare/TinyStories with REINFORCE Policy Gradients

Converted from train_fmnist_bfs.py (Vision) to train_boenet.py (Language)
--------------------------------------------------------------------------
This script trains a BoeNet language model on character-level text data.
The core training loop (REINFORCE, rollouts, policy loss) is identical
to BFSNet - only the data loading and metrics change.

Key Changes from train_fmnist_bfs.py:
-------------------------------------
REMOVED:
  - FashionMNIST data loading
  - Accuracy metrics
  - Classification loss (cross-entropy on logits)
  - Image flattening utilities (_ensure_flat_784)
  - FASHION_CLASSES
  - Augmentation parameters (aug_pad, aug_hflip, no_normalize)

ADDED:
  - WikiText-2/Shakespeare/TinyStories data loading
  - Perplexity metrics (exp(cross_entropy_loss))
  - Next-token prediction loss
  - Character tokenization
  - seq_len, vocab_size, embed_dim parameters
  - Text generation capabilities

UNCHANGED:
  - REINFORCE policy gradients
  - Multiple rollouts mechanism
  - Greedy threshold handling
  - Policy loss computation
  - Gradient clipping
  - Learning rate scheduling
  - Checkpoint saving
  - Node counting and sparsity metrics

v2.0.0 Dataset Changes:
-----------------------
  - Default dataset changed from shakespeare to wikitext2
  - WikiText-2 uses modern HuggingFace Parquet format (no script issues)
  - Shakespeare now downloads directly from Karpathy's GitHub
  - Added wikitext103, bookcorpus, openwebtext options

v2.0.0 Greedy Threshold (same as BFSNet v2.0.0)
-----------------------------------------------
The greedy_threshold parameter controls inference sparsity:
  - threshold = 0.50 (default): Conservative, often root-only
  - threshold = 0.42-0.45: Balanced, partial expansion
  - threshold = 0.30-0.35: Aggressive, near-full expansion

Training Loop (v2.0.0):
  1. Forward: outputs, policy_loss, rewards, node_counts = model(x, labels=y, ...)
  2. Language modeling loss: CE(outputs, y) where y is shifted input
  3. Total loss: lm_loss + beta_policy * policy_loss
  4. Single backward() flows gradients through both paths
  5. Log: perplexity, policy loss, avg nodes/position, rewards

Metrics:
  - Perplexity = exp(cross_entropy_loss)
  - Lower perplexity = better model
  - Random baseline: PPL = vocab_size (256 for char-level)

Node Counting Metrics (v2.0.0):
  - node_counts: List[int] from model.forward(), one int per rollout
  - Each int is total nodes created for the ENTIRE BATCH in that rollout
  - For language models, this is across all token positions
  - avg_nodes_per_position: Total nodes / (batch_size * seq_len * num_rollouts)

Usage Examples:
---------------
# Basic training on WikiText-2 (DEFAULT):
python3 train_boenet.py --epochs 10 --dataset wikitext2

# Training on Shakespeare (via GitHub download):
python3 train_boenet.py --epochs 10 --dataset shakespeare

# Training with custom hyperparameters:
python3 train_boenet.py \\
    --epochs 20 \\
    --seq_len 128 \\
    --embed_dim 64 \\
    --hidden_dim 128 \\
    --max_depth 2 \\
    --max_children 3 \\
    --lambda_efficiency 0.05 \\
    --greedy_threshold 0.42

# Training on TinyStories (larger dataset):
python3 train_boenet.py --dataset tinystories --epochs 5

# Training on custom text file:
python3 train_boenet.py --dataset textfile --text_filepath path/to/text.txt

Greedy Threshold Selection Guide:
---------------------------------
The greedy_threshold parameter controls which grow decisions pass during
inference. This is CRITICAL for v2.0.0 models.

PROBLEM: Policy learns grow_prob ≈ 0.40-0.50 during training, but default
         threshold=0.5 blocks most decisions → root-only inference.

SOLUTION: Set threshold based on learned distribution:
  - threshold = 0.50: Root-only (max efficiency, higher PPL)
  - threshold = 0.42-0.45: Balanced (~6-8 nodes, lower PPL)
  - threshold = 0.30-0.35: Dense (~10-13 nodes, lowest PPL)

WORKFLOW:
  1. Train with default settings
  2. Run: python3 infer_boenet.py --ckpt <path> --debug_policy
  3. Note the mean_grow_prob from output
  4. Retrain with: --greedy_threshold <mean_grow_prob - 0.03>

Available Datasets:
-------------------
  wikitext2:   ~2MB Wikipedia (DEFAULT, RECOMMENDED)
  wikitext103: ~500MB Wikipedia
  shakespeare: ~1MB literary text (via GitHub)
  tinystories: ~2GB children's stories
  bookcorpus:  ~5GB books
  openwebtext: ~40GB web text
  textfile:    Custom local text file

Author: BoeNet project (converted from BFSNet)
Version: 2.0.0
Date: 2025-12-22
"""

from __future__ import annotations
import os
import sys
import time
import json
import math
import argparse
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

# Data
from boenet.utils.data_utils import (
    get_dataloaders,
    SplitConfig,
    CharTokenizer,
    set_seed,
)

# Losses
from boenet.losses import compute_perplexity

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
        # v2.0.0 policy parameters
        "num_rollouts", "lambda_efficiency", "beta_entropy", "beta_policy",
        "greedy_threshold",
        # Pruning losses
        "prune_l1_weight", "prune_kl_weight", "prune_keep_rate",
        # Optimizer
        "opt", "momentum", "nesterov", "adamw_beta1", "adamw_beta2", "adamw_eps",
        # LR schedule
        "lr_schedule", "lr_step_size", "lr_gamma",
        # Debug
        "debug_node_counts",
    ]
    for k in passthru:
        if k in cfg:
            flat[k] = cfg[k]
    
    # Nested sections
    nested_map = {
        "training": ["epochs", "lr", "weight_decay", "grad_clip", "seed"],
        "model": ["hidden_dim", "embed_dim", "max_depth", "max_children", 
                  "no_sibling_embed", "greedy_threshold", "vocab_size"],
        "data": ["seq_len", "dataset", "stride", "batch_size", "val_ratio", "text_filepath"],
        "pruning": ["use_pruning", "pruning_mode", "pruning_threshold"],
        "dataloader": ["num_workers", "pin_memory"],
        "save": ["save_path", "data_root"],
        "run": ["cpu", "pooling_mode"],
        "policy": ["num_rollouts", "lambda_efficiency", "beta_entropy", 
                   "beta_policy", "greedy_threshold"],
        "losses": ["prune_l1_weight", "prune_kl_weight", "prune_keep_rate"],
        "optimizer": ["opt", "momentum", "nesterov", "adamw_beta1", 
                      "adamw_beta2", "adamw_eps"],
        "lr": ["lr_schedule", "lr_step_size", "lr_gamma"],
        "debug": ["debug_node_counts"],
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
) -> Tuple[float, float]:
    """
    Evaluate model on validation set.
    
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
    
    Returns
    -------
    Tuple[float, float]
        (val_loss, val_perplexity)
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    for input_ids, labels in loader:
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        
        # Forward pass (greedy inference)
        logits = model(input_ids)  # [B, seq_len, vocab_size]
        
        # Compute loss
        B, seq_len, V = logits.shape
        logits_flat = logits.view(-1, V)  # [B*seq_len, vocab_size]
        labels_flat = labels.view(-1)      # [B*seq_len]
        
        loss = F.cross_entropy(logits_flat, labels_flat, reduction='sum')
        
        total_loss += loss.item()
        total_tokens += labels.numel()
    
    avg_loss = total_loss / max(1, total_tokens)
    perplexity = compute_perplexity(avg_loss)
    
    return avg_loss, perplexity


# --------------------------------------------------------------------------- #
#                                  Training                                   #
# --------------------------------------------------------------------------- #

def train(args: argparse.Namespace, config_path: Optional[str] = None) -> str:
    """
    Main training function for BoeNet language model.
    
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
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    
    # ==========================================================================
    # WARNING: Greedy Threshold Mismatch Issue (same as BFSNet v2.0.0)
    # ==========================================================================
    if args.greedy_threshold == 0.5 and args.max_children > 0:
        print("\n" + "=" * 79)
        print("⚠️  WARNING: Default greedy_threshold=0.5 May Result in Root-Only Inference!")
        print("=" * 79)
        print("ISSUE:")
        print("  The policy typically learns grow_prob values around 0.40-0.50 during")
        print("  training. With the default greedy threshold of 0.5, most decisions")
        print("  will FAIL the threshold check, resulting in ZERO children created.")
        print()
        print("EXPECTED BEHAVIOR:")
        print("  - Training mode: Uses ~6-12 nodes (stochastic Bernoulli sampling)")
        print("  - Inference mode: Uses 1 node (root-only, greedy threshold blocks growth)")
        print()
        print("RECOMMENDATION:")
        print("  1. After training, run with --debug_policy to measure learned grow_prob:")
        print(f"     python3 infer_boenet.py --ckpt {args.save_path} --debug_policy --cpu")
        print()
        print("  2. Then retrain with threshold ≈ mean_grow_prob - 0.03:")
        print("     Example: If mean_grow_prob = 0.445, use --greedy_threshold 0.42")
        print()
        print("  3. Or use training matrix to sweep (λ, threshold) pairs:")
        print("     python3 boenet_training_matrix.py --config configs/threshold_sweep.yaml")
        print()
        print("See docs/architecture.md for detailed analysis.")
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
    # Model Creation
    # ==========================================================================
    model = BoeNet(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        max_depth=args.max_depth,
        max_children=args.max_children,
        greedy_threshold=args.greedy_threshold,
        sibling_embed=not args.no_sibling_embed,
        use_pruning=args.use_pruning,
        pruning_mode=args.pruning_mode,
        pruning_threshold=args.pruning_threshold,
        pooling_mode=args.pooling_mode,
    ).to(device)
    
    # Log model info
    print(f"\n{model.summary()}")
    print(f"[model] Parameters: {model.num_parameters():,}")
    print(f"[dims] Language Model: vocab_size={vocab_size}, embed_dim={args.embed_dim}, "
          f"hidden_dim={args.hidden_dim}, seq_len={args.seq_len}")
    print(
        f"[run] epochs={args.epochs} | batch_size={args.batch_size} | "
        f"lr={args.lr} | wd={args.weight_decay} | grad_clip={args.grad_clip}\n"
        f"      policy: num_rollouts={args.num_rollouts}, "
        f"λ_eff={args.lambda_efficiency}, β_ent={args.beta_entropy}, "
        f"β_policy={args.beta_policy}\n"
        f"      greedy_threshold={args.greedy_threshold} (inference)\n"
        f"      pooling_mode={args.pooling_mode}\n"
        f"      pruning losses: L1*={args.prune_l1_weight}, "
        f"KL*={args.prune_kl_weight} (keep_rate={args.prune_keep_rate})"
    )
    
    # Compute theoretical maximum nodes for reference
    if args.max_children > 0 and args.max_depth > 0:
        theoretical_max_per_position = sum(
            args.max_children ** d for d in range(args.max_depth + 1)
        )
    else:
        theoretical_max_per_position = 1  # Just root
    
    print(
        f"[node metrics] Theoretical max nodes per token position per rollout: {theoretical_max_per_position}\n"
        f"               With {args.num_rollouts} rollouts: {theoretical_max_per_position * args.num_rollouts} total\n"
        f"               Batch-level max (B={args.batch_size}, seq_len={args.seq_len}): "
        f"{theoretical_max_per_position * args.batch_size * args.seq_len} per rollout"
    )
    
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
        
        # Node counting
        total_nodes_across_all_rollouts = 0
        positions_seen = 0
        
        if args.debug_node_counts:
            all_rollout_node_counts = []
        
        for batch_idx, (input_ids, labels) in enumerate(train_loader):
            input_ids = input_ids.to(device)  # [B, seq_len]
            labels = labels.to(device)        # [B, seq_len]
            
            optimizer.zero_grad()
            
            # v2.0.0 FORWARD (Language Model)
            outputs, policy_loss, rewards, node_counts = model(
                input_ids,
                num_rollouts=args.num_rollouts,
                lambda_efficiency=args.lambda_efficiency,
                beta_entropy=args.beta_entropy,
                labels=labels,
            )
            # outputs: [B, seq_len, vocab_size]
            
            # Language modeling loss (cross-entropy)
            B, seq_len, V = outputs.shape
            outputs_flat = outputs.view(-1, V)     # [B*seq_len, vocab_size]
            labels_flat = labels.view(-1)          # [B*seq_len]
            lm_loss = F.cross_entropy(outputs_flat, labels_flat)
            
            # Total loss
            total_loss_batch = lm_loss + args.beta_policy * policy_loss
            
            # Backward
            total_loss_batch.backward()
            
            # Gradient clipping
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            
            optimizer.step()
            
            # Book-keeping
            num_tokens = labels.numel()
            total_loss += total_loss_batch.item() * num_tokens
            total_lm_loss += lm_loss.item() * num_tokens
            total_policy_loss += policy_loss.item() * num_tokens
            total_tokens += num_tokens
            
            # Node counting
            total_nodes_this_batch = sum(node_counts)
            total_nodes_across_all_rollouts += total_nodes_this_batch
            positions_seen += B * seq_len  # Token positions
            
            if args.debug_node_counts and batch_idx == 0:
                avg_per_rollout = total_nodes_this_batch / len(node_counts)
                nodes_per_position_this_batch = total_nodes_this_batch / (B * seq_len * len(node_counts))
                print(
                    f"[debug nodes batch={batch_idx}]\n"
                    f"  node_counts (per rollout for batch): {node_counts}\n"
                    f"  sum(node_counts): {total_nodes_this_batch}\n"
                    f"  batch_size: {B}, seq_len: {seq_len}\n"
                    f"  num_rollouts: {len(node_counts)}\n"
                    f"  avg_per_rollout: {avg_per_rollout:.1f}\n"
                    f"  nodes_per_position_per_rollout: {nodes_per_position_this_batch:.2f}\n"
                    f"  theoretical_max_per_position_per_rollout: {theoretical_max_per_position}"
                )
            
            if args.debug_node_counts:
                all_rollout_node_counts.extend(node_counts)
        
        # Scheduler step
        if scheduler is not None:
            scheduler.step()
        
        # Epoch metrics
        train_loss = total_loss / max(1, total_tokens)
        train_lm_loss = total_lm_loss / max(1, total_tokens)
        train_policy_loss = total_policy_loss / max(1, total_tokens)
        train_ppl = compute_perplexity(train_lm_loss)
        
        avg_total_nodes_per_position = total_nodes_across_all_rollouts / max(1, positions_seen)
        theoretical_max_total_per_position = theoretical_max_per_position * args.num_rollouts
        sparsity_ratio = avg_total_nodes_per_position / max(theoretical_max_total_per_position, 1)
        
        # Validation
        val_loss, val_ppl = evaluate(val_loader, model, device, vocab_size)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_ppl = val_ppl
            best_epoch = epoch
        
        print(
            f"  Train Loss: {train_loss:.4f} (lm={train_lm_loss:.4f}, "
            f"policy={train_policy_loss:.4f}) | Val Loss: {val_loss:.4f}\n"
            f"  Train PPL: {train_ppl:.2f} | Val PPL: {val_ppl:.2f} | "
            f"Avg Total Nodes/Position: {avg_total_nodes_per_position:.2f}\n"
            f"  (Sparsity: {sparsity_ratio:.2%} of theoretical max {theoretical_max_total_per_position} nodes)"
        )
        
        if args.debug_node_counts:
            import statistics
            min_nodes = min(all_rollout_node_counts)
            max_nodes = max(all_rollout_node_counts)
            median_nodes = statistics.median(all_rollout_node_counts)
            stddev_nodes = statistics.stdev(all_rollout_node_counts) if len(all_rollout_node_counts) > 1 else 0
            print(
                f"  [node stats] min={min_nodes}, max={max_nodes}, "
                f"median={median_nodes:.1f}, stddev={stddev_nodes:.1f}"
            )
        
        epoch_time = time.perf_counter() - t0
        epoch_times_s.append(float(epoch_time))
    
    total_time_s = time.perf_counter() - t0_total
    
    # ==========================================================================
    # Save Checkpoint
    # ==========================================================================
    save_dir = os.path.dirname(args.save_path) or "."
    os.makedirs(save_dir, exist_ok=True)
    
    ckpt = {
        "model_state_dict": model.state_dict(),
        "config": {
            # Language model specific (NEW keys for v2.0.0)
            "vocab_size": vocab_size,
            "embed_dim": args.embed_dim,
            "hidden_dim": args.hidden_dim,
            "seq_len": args.seq_len,
            "dataset": args.dataset,
            # Architecture
            "max_depth": args.max_depth,
            "max_children": args.max_children,
            "greedy_threshold": args.greedy_threshold,
            "sibling_embed": not args.no_sibling_embed,
            "use_pruning": args.use_pruning,
            "pruning_mode": args.pruning_mode,
            "pruning_threshold": args.pruning_threshold,
            "pooling_mode": args.pooling_mode,
            # v2.0.0 policy parameters
            "num_rollouts": args.num_rollouts,
            "lambda_efficiency": args.lambda_efficiency,
            "beta_entropy": args.beta_entropy,
            "beta_policy": args.beta_policy,
            "version": "2.0.0",
            "model_type": "language",
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
    print("POST-TRAINING: Greedy Threshold Tuning Recommendations")
    print("=" * 79)
    print()
    print("Your model was trained with:")
    print(f"  dataset           = {args.dataset}")
    print(f"  lambda_efficiency = {args.lambda_efficiency}")
    print(f"  greedy_threshold  = {args.greedy_threshold}")
    print(f"  best_val_ppl      = {best_val_ppl:.2f}")
    print()
    print("NEXT STEPS:")
    print()
    print("1. Measure the learned grow_prob distribution:")
    print(f"   python3 infer_boenet.py --ckpt {args.save_path} \\")
    print(f"       --debug_policy --node_samples 1000 --cpu")
    print()
    print("   This will show you:")
    print("   - Mean grow_prob (typically 0.40-0.50)")
    print("   - % of decisions above threshold 0.5")
    print("   - Distribution histogram")
    print()
    print("2. Based on the measured mean_grow_prob, choose optimal threshold:")
    print()
    print("   If mean_grow_prob ≈ 0.445, recommended settings:")
    print("     • threshold = 0.50 → Root-only (1 node, higher PPL)")
    print("     • threshold = 0.42 → Balanced (~6-8 nodes, lower PPL)")
    print("     • threshold = 0.35 → Dense (~10-12 nodes, lowest PPL)")
    print()
    print("   General rule: optimal_threshold ≈ mean_grow_prob - 0.03")
    print()
    print("3. Generate text with the trained model:")
    print(f"   python3 infer_boenet.py --ckpt {args.save_path} \\")
    print(f"       --generate --max_tokens 200 --temperature 0.8")
    print()
    print("4. Compare to random baseline:")
    print(f"   Random PPL = {random_ppl:.2f} (vocab_size)")
    print(f"   Your PPL   = {best_val_ppl:.2f}")
    if best_val_ppl > 0:
        print(f"   Improvement: {(random_ppl / best_val_ppl):.2f}x better")
    print()
    print("EMPIRICAL FINDINGS:")
    print("  • Higher λ (0.05) often gives BETTER perplexity than lower λ (0.01)")
    print("  • Policy learns grow_prob ≈ 0.44-0.45 regardless of λ value")
    print("  • Default threshold=0.5 typically results in root-only inference")
    print("  • Root-only still achieves reasonable PPL (language patterns learned)")
    print()
    print("See docs/architecture.md for complete analysis and recommendations.")
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
        },
        "policy": {
            "num_rollouts": int(args.num_rollouts),
            "lambda_efficiency": float(args.lambda_efficiency),
            "beta_entropy": float(args.beta_entropy),
            "beta_policy": float(args.beta_policy),
        },
        "results": {
            "best_val_loss": float(best_val_loss),
            "best_val_ppl": float(best_val_ppl),
            "best_epoch": int(best_epoch),
            "final_val_ppl": float(val_ppl),
            "random_baseline_ppl": float(random_ppl),
        },
        "time": {
            "total_s": float(total_time_s),
            "epoch_avg_s": float(sum(epoch_times_s) / len(epoch_times_s)) if epoch_times_s else 0.0,
            "last_epoch_s": float(epoch_times_s[-1]) if epoch_times_s else 0.0,
        },
        "artifacts": {
            "checkpoint": args.save_path,
        },
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
        description="Train BoeNet v2.0.0 Language Model with REINFORCE policy gradients",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
---------
# Train on WikiText-2 (DEFAULT):
python3 train_boenet.py --epochs 10 --dataset wikitext2

# Train on Shakespeare:
python3 train_boenet.py --epochs 10 --dataset shakespeare

# Train on TinyStories:
python3 train_boenet.py --epochs 5 --dataset tinystories

# Train on custom text file:
python3 train_boenet.py --dataset textfile --text_filepath path/to/text.txt --epochs 10

# Custom hyperparameters:
python3 train_boenet.py \\
    --seq_len 128 \\
    --embed_dim 64 \\
    --hidden_dim 128 \\
    --max_depth 2 \\
    --max_children 3 \\
    --lambda_efficiency 0.05 \\
    --greedy_threshold 0.42 \\
    --epochs 20

Available Datasets:
-------------------
  wikitext2:   ~2MB Wikipedia (DEFAULT, RECOMMENDED)
  wikitext103: ~500MB Wikipedia
  shakespeare: ~1MB literary text (via GitHub)
  tinystories: ~2GB children's stories
  bookcorpus:  ~5GB books
  openwebtext: ~40GB web text
  textfile:    Custom local text file

Greedy Threshold Selection Guide:
---------------------------------
The greedy_threshold parameter controls which grow decisions pass during
inference. This is CRITICAL for v2.0.0 models.

PROBLEM: Policy learns grow_prob ≈ 0.40-0.50 during training, but default
         threshold=0.5 blocks most decisions → root-only inference.

SOLUTION: Set threshold based on learned distribution:
  - threshold = 0.50: Root-only (max efficiency, higher PPL)
  - threshold = 0.42-0.45: Balanced (~6-8 nodes, lower PPL)
  - threshold = 0.30-0.35: Dense (~10-13 nodes, lowest PPL)

WORKFLOW:
  1. Train with default settings
  2. Run: python3 infer_boenet.py --ckpt <path> --debug_policy
  3. Note the mean_grow_prob from output
  4. Retrain with: --greedy_threshold <mean_grow_prob - 0.03>

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
    
    # Model Architecture (Language Model Specific)
    p.add_argument("--embed_dim", type=int, default=64,
                   help="Token embedding dimension")
    p.add_argument("--hidden_dim", type=int, default=128,
                   help="Hidden dimension for BFS tree nodes")
    p.add_argument("--max_depth", type=int, default=2,
                   help="Maximum BFS tree depth")
    p.add_argument("--max_children", type=int, default=3,
                   help="Maximum children per node")
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
    
    # v2.0.0 Policy Gradient Parameters
    p.add_argument("--num_rollouts", type=int, default=3,
                   help="Number of rollouts per input for exploration (1-5)")
    p.add_argument("--lambda_efficiency", type=float, default=0.05,
                   help="Node count penalty in reward (0.0-0.1). Higher = stronger efficiency pressure. "
                        "NOTE: λ=0.05 often gives BETTER perplexity than λ=0.01 (acts as regularization)")
    p.add_argument("--beta_entropy", type=float, default=0.01,
                   help="Entropy bonus in policy loss (0.001-0.01)")
    p.add_argument("--beta_policy", type=float, default=0.5,
                   help="Policy loss weight in total loss (0.1-1.0)")
    p.add_argument("--greedy_threshold", type=float, default=0.5,
                   help="Threshold for greedy inference decisions (0.3-0.5). "
                        "CRITICAL: Default 0.5 often results in root-only inference! "
                        "Recommended: 0.42-0.45 based on learned grow_prob. "
                        "Run with --debug_policy after training to measure optimal value.")
    
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
    
    # Debug
    p.add_argument("--debug_node_counts", type=str2bool, nargs="?", const=True, default=False,
                   help="Enable detailed node count logging for debugging")
    
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