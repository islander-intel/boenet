#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_boenet.py (v4.0.0 - BPE Tokenizer Support with Scaled Model)

Train BoeNet on WikiText-2/Shakespeare/TinyStories with True BFS and REINFORCE

v4.0.0 Major Update (2026-01-03) - BPE Tokenizer and Scaled Model
-----------------------------------------------------------------
This version adds support for BPE (Byte Pair Encoding) tokenization using
tiktoken's cl100k_base encoding (GPT-4 tokenizer). This enables:

  - Word/subword-level tokenization instead of character-level
  - 100,277 vocabulary size (vs 256 for character-level)
  - ~30% fewer tokens for same text (better efficiency)
  - More coherent text generation

SCALING CHANGES:
  | Parameter     | v3.2.0 (char) | v4.0.0 (BPE)  |
  |---------------|---------------|---------------|
  | vocab_size    | 256           | 100,277       |
  | embed_dim     | 64            | 256           |
  | hidden_dim    | 128           | 512           |
  | Parameters    | ~150K         | ~78M          |

NEW COMMAND LINE OPTIONS:
  --tokenizer_type : "char" or "bpe" (default: "bpe")
  --bpe_encoding   : tiktoken encoding name (default: "cl100k_base")

USAGE EXAMPLES:

  # BPE tokenization (NEW DEFAULT - recommended)
  python3 train_boenet.py --epochs 10 --dataset wikitext2

  # Explicit BPE with cl100k_base
  python3 train_boenet.py --tokenizer_type bpe --bpe_encoding cl100k_base

  # Character-level (legacy, for comparison)
  python3 train_boenet.py --tokenizer_type char --embed_dim 64 --hidden_dim 128

  # Full scaled BPE model
  python3 train_boenet.py \\
      --tokenizer_type bpe \\
      --embed_dim 256 \\
      --hidden_dim 512 \\
      --max_depth 4 \\
      --epochs 20

v3.2.0 Features (Preserved):
  - Fixed node count metric reporting
  - True BFS level-by-level expansion
  - REINFORCE policy gradient training
  - Epsilon-greedy exploration
  - Separate gradient clipping for policy network

Available Datasets:
-------------------
  wikitext2:   ~2MB Wikipedia (DEFAULT, RECOMMENDED)
  wikitext103: ~500MB Wikipedia
  shakespeare: ~1MB literary text (via GitHub)
  tinystories: ~2GB children's stories
  textfile:    Custom local text file

Author: BoeNet project
Version: 4.0.0
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

# BFS indexing functions
try:
    from boenet.model import (
        get_total_nodes_up_to_level,
        get_level,
        get_nodes_at_level,
    )
except ImportError:
    pass

try:
    from boenet.model import get_num_nodes_at_level
except ImportError:
    def get_num_nodes_at_level(level: int) -> int:
        return 1 << level

# v4.0.0: Import tokenizers
from boenet.tokenizer import (
    get_tokenizer,
    CharTokenizer,
    TiktokenWrapper,
    BaseTokenizer,
)

# Try to import get_vocab_size
try:
    from boenet.tokenizer import get_vocab_size
except ImportError:
    def get_vocab_size(tokenizer_type: str = "char", encoding_name: str = "cl100k_base") -> int:
        if tokenizer_type == "char":
            return 256
        tok = get_tokenizer(tokenizer_type, encoding_name)
        return tok.vocab_size

# Data
from boenet.utils.data_utils import (
    get_dataloaders,
    SplitConfig,
    set_seed,
)

# Losses
from boenet.losses import compute_perplexity

try:
    from boenet.losses import (
        compute_rewards_true_bfs,
        compute_depth_from_nodes,
        get_nodes_for_depth,
    )
except ImportError:
    def compute_depth_from_nodes(num_nodes: int) -> int:
        if num_nodes <= 0:
            return 0
        return max(0, int(math.floor(math.log2(num_nodes + 1))) - 1)
    
    def get_nodes_for_depth(depth: int) -> int:
        return (1 << (depth + 1)) - 1

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
#                      NaN Detection Helpers                                  #
# --------------------------------------------------------------------------- #

def check_for_nan_in_model(model: nn.Module, step_name: str = "") -> bool:
    """Check model parameters for NaN values."""
    for name, param in model.named_parameters():
        if param is not None and torch.isnan(param).any():
            nan_count = torch.isnan(param).sum().item()
            logger.error(f"[NaN DETECTED] {step_name} - Parameter '{name}' has {nan_count} NaN values!")
            return True
    return False


def check_for_nan_in_gradients(model: nn.Module, step_name: str = "") -> bool:
    """Check model gradients for NaN values."""
    for name, param in model.named_parameters():
        if param.grad is not None and torch.isnan(param.grad).any():
            nan_count = torch.isnan(param.grad).sum().item()
            logger.error(f"[NaN GRADIENT] {step_name} - Parameter '{name}' has {nan_count} NaN gradients!")
            return True
    return False


def check_tensor_for_nan(tensor: torch.Tensor, name: str, step_name: str = "") -> bool:
    """Check a tensor for NaN values."""
    if torch.isnan(tensor).any():
        nan_count = torch.isnan(tensor).sum().item()
        logger.warning(f"[NaN TENSOR] {step_name} - Tensor '{name}' has {nan_count} NaN values!")
        return True
    return False


# --------------------------------------------------------------------------- #
#                   True BFS Node Counting Helpers                            #
# --------------------------------------------------------------------------- #

def compute_depth_from_tree_nodes(tree_nodes: int) -> int:
    """Compute depth of complete binary tree from node count."""
    if tree_nodes <= 0:
        return 0
    if tree_nodes == 1:
        return 0
    return max(0, int(math.floor(math.log2(tree_nodes + 1))) - 1)


def compute_theoretical_max_nodes(max_depth: int) -> int:
    """Compute theoretical maximum nodes for complete binary tree."""
    if max_depth < 0:
        return 1
    return (1 << (max_depth + 1)) - 1


def verify_balanced_tree(node_counts: List[int], max_depth: int) -> Dict[str, Any]:
    """Verify node counts are consistent with balanced binary trees."""
    valid_node_counts = [(1 << (d + 1)) - 1 for d in range(max_depth + 1)]
    result = {"is_valid": True, "depths": [], "warnings": []}
    
    for rollout_idx, tree_nodes in enumerate(node_counts):
        depth = compute_depth_from_tree_nodes(tree_nodes)
        result["depths"].append(depth)
        if tree_nodes not in valid_node_counts:
            closest_valid = min(valid_node_counts, key=lambda x: abs(x - tree_nodes))
            if tree_nodes != closest_valid:
                result["is_valid"] = False
                result["warnings"].append(
                    f"Rollout {rollout_idx}: tree_nodes={tree_nodes} not valid"
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


def format_params(num_params: int) -> str:
    """Format parameter count with appropriate suffix."""
    if num_params >= 1e9:
        return f"{num_params / 1e9:.2f}B"
    elif num_params >= 1e6:
        return f"{num_params / 1e6:.2f}M"
    elif num_params >= 1e3:
        return f"{num_params / 1e3:.2f}K"
    else:
        return str(num_params)


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
    
    passthru = [
        "epochs", "lr", "weight_decay", "grad_clip", "seed",
        "hidden_dim", "embed_dim", "max_depth", "max_children", "no_sibling_embed",
        "use_pruning", "pruning_mode", "pruning_threshold",
        "num_workers", "pin_memory", "batch_size", "val_ratio",
        "save_path", "data_root", "cpu", "pooling_mode",
        "vocab_size", "seq_len", "dataset", "stride", "text_filepath",
        "tokenizer_type", "bpe_encoding",  # v4.0.0
        "num_rollouts", "lambda_efficiency", "beta_entropy", "beta_policy",
        "greedy_threshold", "min_explore_prob", "policy_grad_clip",
        "prune_l1_weight", "prune_kl_weight", "prune_keep_rate",
        "opt", "momentum", "nesterov", "adamw_beta1", "adamw_beta2", "adamw_eps",
        "lr_schedule", "lr_step_size", "lr_gamma",
        "debug_node_counts", "debug_bfs_verify",
    ]
    for k in passthru:
        if k in cfg:
            flat[k] = cfg[k]
    
    nested_map = {
        "training": ["epochs", "lr", "weight_decay", "grad_clip", "seed", "policy_grad_clip"],
        "model": ["hidden_dim", "embed_dim", "max_depth", "max_children", 
                  "no_sibling_embed", "greedy_threshold", "vocab_size", "min_explore_prob"],
        "data": ["seq_len", "dataset", "stride", "batch_size", "val_ratio", "text_filepath"],
        "tokenizer": ["tokenizer_type", "bpe_encoding"],
        "pruning": ["use_pruning", "pruning_mode", "pruning_threshold"],
        "dataloader": ["num_workers", "pin_memory"],
        "save": ["save_path", "data_root"],
        "run": ["cpu", "pooling_mode"],
        "policy": ["num_rollouts", "lambda_efficiency", "beta_entropy", 
                   "beta_policy", "greedy_threshold", "policy_grad_clip", "min_explore_prob"],
        "losses": ["prune_l1_weight", "prune_kl_weight", "prune_keep_rate"],
        "optimizer": ["opt", "momentum", "nesterov", "adamw_beta1", "adamw_beta2", "adamw_eps"],
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
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    num_batches = 0
    
    for input_ids, labels in loader:
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        
        logits = model(input_ids)
        
        if torch.isnan(logits).any():
            logger.warning("[Eval] NaN detected in logits. Skipping batch.")
            continue
        
        B, seq_len, V = logits.shape
        logits_flat = logits.view(-1, V)
        labels_flat = labels.view(-1)
        
        loss = F.cross_entropy(logits_flat, labels_flat, reduction='sum')
        
        if torch.isnan(loss):
            logger.warning("[Eval] NaN detected in loss. Skipping batch.")
            continue
        
        total_loss += loss.item()
        total_tokens += labels.numel()
        num_batches += 1
    
    avg_loss = total_loss / max(1, total_tokens)
    perplexity = compute_perplexity(avg_loss)
    
    return avg_loss, perplexity, 0.0, 0.0


# --------------------------------------------------------------------------- #
#                                  Training                                   #
# --------------------------------------------------------------------------- #

def train(args: argparse.Namespace, config_path: Optional[str] = None) -> str:
    """
    Main training function for BoeNet language model.
    
    v4.0.0: Adds BPE tokenizer support with cl100k_base encoding.
    """
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
    # v4.0.0: Initialize Tokenizer
    # ==========================================================================
    print("\n" + "=" * 79)
    print("🔤 BoeNet v4.0.0 - Tokenizer Initialization")
    print("=" * 79)
    
    tokenizer = get_tokenizer(
        tokenizer_type=args.tokenizer_type,
        encoding_name=args.bpe_encoding,
    )
    
    vocab_size = tokenizer.vocab_size
    
    print(f"\nTOKENIZER CONFIGURATION:")
    print(f"  tokenizer_type = {args.tokenizer_type}")
    if args.tokenizer_type == "bpe":
        print(f"  bpe_encoding   = {args.bpe_encoding}")
    print(f"  vocab_size     = {vocab_size:,}")
    print(f"  tokenizer      = {tokenizer}")
    
    sample_text = "The quick brown fox jumps over the lazy dog."
    sample_tokens = tokenizer.encode(sample_text)
    print(f"\nTOKENIZATION EXAMPLE:")
    print(f"  Text:   '{sample_text}'")
    print(f"  Tokens: {sample_tokens}")
    print(f"  Count:  {len(sample_tokens)} tokens")
    print(f"  Ratio:  {len(sample_text) / len(sample_tokens):.1f} chars/token")
    
    if hasattr(tokenizer, 'eos_token_id'):
        print(f"\nSPECIAL TOKENS:")
        print(f"  eos_token_id = {tokenizer.eos_token_id}")
        print(f"  pad_token_id = {tokenizer.pad_token_id}")
    
    print("=" * 79 + "\n")
    
    # ==========================================================================
    # Model Info Banner
    # ==========================================================================
    theoretical_max = compute_theoretical_max_nodes(args.max_depth)
    
    print("\n" + "=" * 79)
    print("🌳 BoeNet v4.0.0 - True BFS Language Model with BPE Tokenization")
    print("=" * 79)
    print()
    print(f"CONFIGURATION:")
    print(f"  tokenizer_type   = {args.tokenizer_type}")
    print(f"  vocab_size       = {vocab_size:,}")
    print(f"  embed_dim        = {args.embed_dim}")
    print(f"  hidden_dim       = {args.hidden_dim}")
    print(f"  max_depth        = {args.max_depth}")
    print(f"  max_children     = {args.max_children}")
    print(f"  greedy_thresh    = {args.greedy_threshold}")
    print(f"  min_explore_prob = {args.min_explore_prob}")
    print()
    
    estimated_params = (
        vocab_size * args.embed_dim +
        args.embed_dim * args.hidden_dim + args.hidden_dim +
        args.hidden_dim * args.hidden_dim + args.hidden_dim +
        args.hidden_dim * args.max_children * args.hidden_dim + args.max_children * args.hidden_dim +
        args.hidden_dim * 64 + 64 + 64 + 1 +
        args.hidden_dim * vocab_size + vocab_size
    )
    print(f"ESTIMATED PARAMETERS: {format_params(estimated_params)}")
    print()
    
    print(f"TREE STRUCTURE (max_depth={args.max_depth}):")
    print(f"  Theoretical max nodes per position: {theoretical_max}")
    for d in range(args.max_depth + 1):
        nodes = (1 << (d + 1)) - 1
        print(f"    depth={d}: {nodes:3d} nodes")
    print("=" * 79 + "\n")
    
    # ==========================================================================
    # Data Loading
    # ==========================================================================
    print(f"[data] Loading {args.dataset} dataset with {args.tokenizer_type} tokenizer...")
    
    loader_kwargs = {
        "batch_size": args.batch_size,
        "seed": args.seed,
        "split": SplitConfig(val_ratio=args.val_ratio, shuffle_before_split=True),
        "seq_len": args.seq_len,
        "stride": args.stride if args.stride > 0 else None,
        "dataloader_num_workers": args.num_workers,
        "dataloader_pin_memory": args.pin_memory,
        "tokenizer": tokenizer,  # v4.0.0: Pass tokenizer
    }
    
    if args.dataset == "textfile" and args.text_filepath:
        loader_kwargs["text_filepath"] = args.text_filepath
    
    train_loader, val_loader, loader_vocab_size = get_dataloaders(args.dataset, **loader_kwargs)
    
    if loader_vocab_size != vocab_size:
        logger.warning(f"[data] Vocab size mismatch: tokenizer={vocab_size}, loader={loader_vocab_size}")
    
    print(f"[data] vocab_size={vocab_size:,}, seq_len={args.seq_len}")
    print(f"[data] Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    random_ppl = vocab_size
    print(f"[data] Random baseline perplexity: {random_ppl:,.2f}")
    
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
        min_explore_prob=args.min_explore_prob,
        sibling_embed=not args.no_sibling_embed,
        use_pruning=args.use_pruning,
        pruning_mode=args.pruning_mode,
        pruning_threshold=args.pruning_threshold,
        pooling_mode=args.pooling_mode,
    ).to(device)
    
    print(f"\n{model.summary()}")
    actual_params = model.num_parameters()
    print(f"[model] Parameters: {actual_params:,} ({format_params(actual_params)})")
    print(f"[model] max_nodes (per position): {model.max_nodes}")
    
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
    
    # Identify policy parameters for separate gradient clipping
    policy_params = []
    other_params = []
    for name, param in model.named_parameters():
        if 'growth_policy' in name:
            policy_params.append(param)
        else:
            other_params.append(param)
    
    logger.info(f"[Policy Params] {len(policy_params)} parameters")
    logger.info(f"[Other Params] {len(other_params)} parameters")
    
    theoretical_max_per_position = compute_theoretical_max_nodes(args.max_depth)
    
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
        
        all_tree_sizes_this_epoch: List[int] = []
        depth_counts = {d: 0 for d in range(args.max_depth + 1)}
        expansion_happened_this_epoch = False
        
        nan_batch_count = 0
        max_nan_batches = max(len(train_loader) // 10, 5)
        skipped_batches = 0
        
        for batch_idx, (input_ids, labels) in enumerate(train_loader):
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
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
                    logger.error(f"[Batch {batch_idx}] CUDA error: {e}")
                    nan_batch_count += 1
                    skipped_batches += 1
                    if nan_batch_count > max_nan_batches:
                        break
                    continue
                else:
                    raise
            
            B = input_ids.size(0)
            
            if check_tensor_for_nan(outputs, "outputs", f"batch_{batch_idx}"):
                nan_batch_count += 1
                skipped_batches += 1
                if nan_batch_count > max_nan_batches:
                    break
                continue
            
            if torch.isnan(policy_loss):
                policy_loss = torch.tensor(0.0, device=device, requires_grad=True)
                nan_batch_count += 1
            
            V = outputs.size(-1)
            outputs_flat = outputs.view(-1, V)
            labels_flat = labels.view(-1)
            lm_loss = F.cross_entropy(outputs_flat, labels_flat)
            
            if torch.isnan(lm_loss):
                nan_batch_count += 1
                skipped_batches += 1
                if nan_batch_count > max_nan_batches:
                    break
                continue
            
            total_loss_batch = lm_loss + args.beta_policy * policy_loss
            total_loss_batch.backward()
            
            if check_for_nan_in_gradients(model, f"batch_{batch_idx}"):
                optimizer.zero_grad()
                nan_batch_count += 1
                skipped_batches += 1
                if nan_batch_count > max_nan_batches:
                    break
                continue
            
            if args.grad_clip and args.grad_clip > 0:
                if policy_params and args.policy_grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(policy_params, args.policy_grad_clip)
                if other_params:
                    torch.nn.utils.clip_grad_norm_(other_params, args.grad_clip)
            
            has_nan_after_clip = False
            for name, param in model.named_parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    has_nan_after_clip = True
                    break
            
            if has_nan_after_clip:
                optimizer.zero_grad()
                nan_batch_count += 1
                skipped_batches += 1
                continue
            
            optimizer.step()
            
            num_tokens = labels.numel()
            total_loss += total_loss_batch.item() * num_tokens
            total_lm_loss += lm_loss.item() * num_tokens
            total_policy_loss += policy_loss.item() * num_tokens
            total_tokens += num_tokens
            
            all_tree_sizes_this_epoch.extend(node_counts)
            
            for tree_size in node_counts:
                depth = compute_depth_from_tree_nodes(tree_size)
                depth = min(depth, args.max_depth)
                depth_counts[depth] += 1
                if depth > 0:
                    expansion_happened_this_epoch = True
        
        if nan_batch_count > max_nan_batches:
            logger.error(f"[Epoch {epoch}] Stopping due to too many NaN batches.")
            break
        
        if skipped_batches > 0:
            logger.warning(f"[Epoch {epoch}] Skipped {skipped_batches} batches due to NaN.")
        
        if scheduler is not None:
            scheduler.step()
        
        # Epoch metrics
        if total_tokens > 0:
            train_loss = total_loss / total_tokens
            train_lm_loss = total_lm_loss / total_tokens
            train_policy_loss = total_policy_loss / total_tokens
            train_ppl = compute_perplexity(train_lm_loss)
        else:
            train_loss = train_lm_loss = train_policy_loss = float('inf')
            train_ppl = float('inf')
        
        if len(all_tree_sizes_this_epoch) > 0:
            avg_nodes_per_position = sum(all_tree_sizes_this_epoch) / len(all_tree_sizes_this_epoch)
        else:
            avg_nodes_per_position = 0.0
        
        sparsity_pct = (avg_nodes_per_position / theoretical_max_per_position) * 100 if theoretical_max_per_position > 0 else 0.0
        
        total_rollouts = sum(depth_counts.values())
        if total_rollouts > 0:
            avg_depth_reached = sum(d * c for d, c in depth_counts.items()) / total_rollouts
        else:
            avg_depth_reached = 0.0
        
        val_loss, val_ppl, _, _ = evaluate(val_loader, model, device, vocab_size, args.max_depth)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_ppl = val_ppl
            best_epoch = epoch
        
        print(
            f"  Train Loss: {train_loss:.4f} (lm={train_lm_loss:.4f}, "
            f"policy={train_policy_loss:.4f}) | Val Loss: {val_loss:.4f}\n"
            f"  Train PPL: {train_ppl:.2f} | Val PPL: {val_ppl:.2f}"
        )
        print(
            f"  [True BFS v4.0.0] Avg tree size: {avg_nodes_per_position:.2f} | "
            f"Avg depth: {avg_depth_reached:.2f} | "
            f"Sparsity: {sparsity_pct:.1f}% of max {theoretical_max_per_position}"
        )
        
        print(f"  [Depth Distribution] ", end="")
        for d in range(args.max_depth + 1):
            if depth_counts[d] > 0:
                pct = 100.0 * depth_counts[d] / total_rollouts
                print(f"d{d}:{pct:.1f}% ", end="")
        print()
        
        if not expansion_happened_this_epoch and args.max_children > 0 and args.max_depth > 0:
            print(f"  ⚠️  WARNING: No tree expansion occurred this epoch!")
        
        epoch_time = time.perf_counter() - t0
        epoch_times_s.append(float(epoch_time))
        
        if check_for_nan_in_model(model, f"epoch_{epoch}_end"):
            logger.error(f"[Epoch {epoch}] Model contains NaN. Stopping.")
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
            "vocab_size": vocab_size,
            "embed_dim": args.embed_dim,
            "hidden_dim": args.hidden_dim,
            "seq_len": args.seq_len,
            "dataset": args.dataset,
            "tokenizer_type": args.tokenizer_type,
            "bpe_encoding": args.bpe_encoding,
            "max_depth": args.max_depth,
            "max_children": args.max_children,
            "greedy_threshold": args.greedy_threshold,
            "min_explore_prob": args.min_explore_prob,
            "sibling_embed": not args.no_sibling_embed,
            "use_pruning": args.use_pruning,
            "pruning_mode": args.pruning_mode,
            "pruning_threshold": args.pruning_threshold,
            "pooling_mode": args.pooling_mode,
            "num_rollouts": args.num_rollouts,
            "lambda_efficiency": args.lambda_efficiency,
            "beta_entropy": args.beta_entropy,
            "beta_policy": args.beta_policy,
            "policy_grad_clip": args.policy_grad_clip,
            "version": "4.0.0",
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
    
    # Summary
    print("\n" + "=" * 79)
    print("POST-TRAINING: True BFS v4.0.0 Summary")
    print("=" * 79)
    print(f"  tokenizer: {args.tokenizer_type} ({args.bpe_encoding})")
    print(f"  vocab_size: {vocab_size:,}")
    print(f"  parameters: {format_params(actual_params)}")
    print(f"  best_val_ppl: {best_val_ppl:.2f}")
    print(f"  random_baseline: {random_ppl:,.2f}")
    if best_val_ppl > 0 and best_val_ppl < float('inf'):
        print(f"  improvement: {(random_ppl / best_val_ppl):.2f}x better")
    print("=" * 79 + "\n")
    
    summary = {
        "run": {"seed": args.seed, "epochs": args.epochs, "batch_size": args.batch_size, "dataset": args.dataset},
        "tokenizer": {"type": args.tokenizer_type, "encoding": args.bpe_encoding, "vocab_size": vocab_size},
        "model": {
            "vocab_size": vocab_size, "embed_dim": args.embed_dim, "hidden_dim": args.hidden_dim,
            "max_depth": args.max_depth, "parameters": actual_params,
        },
        "results": {"best_val_loss": best_val_loss, "best_val_ppl": best_val_ppl, "best_epoch": best_epoch},
        "time": {"total_s": total_time_s},
        "version": "4.0.0",
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
        description="Train BoeNet v4.0.0 True BFS Language Model with BPE Tokenization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
---------
# Train with BPE tokenization (NEW DEFAULT):
python3 train_boenet.py --epochs 10 --dataset wikitext2

# Character-level (legacy):
python3 train_boenet.py --tokenizer_type char --embed_dim 64 --hidden_dim 128

# Full scaled BPE model:
python3 train_boenet.py --tokenizer_type bpe --embed_dim 256 --hidden_dim 512 --max_depth 4 --epochs 20
        """
    )
    
    p.add_argument("--config", type=str, default=None, help="Path to YAML config")
    
    # v4.0.0: Tokenizer settings
    p.add_argument("--tokenizer_type", type=str, default="bpe", choices=["char", "bpe"],
                   help="Tokenizer type (default: bpe)")
    p.add_argument("--bpe_encoding", type=str, default="cl100k_base",
                   choices=["gpt2", "r50k_base", "p50k_base", "cl100k_base"],
                   help="BPE encoding (default: cl100k_base)")
    
    # Data
    p.add_argument("--dataset", type=str, default="wikitext2",
                   choices=["wikitext2", "wikitext103", "shakespeare", "tinystories", "textfile"],
                   help="Dataset (default: wikitext2)")
    p.add_argument("--text_filepath", type=str, default=None)
    p.add_argument("--data_root", type=str, default="./data")
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--batch_size", type=int, default=32, help="Batch size (default: 32 for BPE)")
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--stride", type=int, default=0)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--pin_memory", type=str2bool, nargs="?", const=True, default=False)
    
    # Model (v4.0.0: Updated defaults for BPE)
    p.add_argument("--embed_dim", type=int, default=256, help="Embedding dim (default: 256 for BPE)")
    p.add_argument("--hidden_dim", type=int, default=512, help="Hidden dim (default: 512 for BPE)")
    p.add_argument("--max_depth", type=int, default=4)
    p.add_argument("--max_children", type=int, default=2)
    p.add_argument("--no_sibling_embed", type=str2bool, nargs="?", const=True, default=False)
    p.add_argument("--pooling_mode", type=str, default="mean", choices=["mean", "sum", "learned"])
    
    # Pruning
    p.add_argument("--use_pruning", type=str2bool, nargs="?", const=True, default=False)
    p.add_argument("--pruning_mode", type=str, default="learned", choices=["learned", "threshold"])
    p.add_argument("--pruning_threshold", type=float, default=1e-3)
    
    # Policy
    p.add_argument("--num_rollouts", type=int, default=3)
    p.add_argument("--lambda_efficiency", type=float, default=0.05)
    p.add_argument("--beta_entropy", type=float, default=0.01)
    p.add_argument("--beta_policy", type=float, default=0.5)
    p.add_argument("--greedy_threshold", type=float, default=0.5)
    p.add_argument("--min_explore_prob", type=float, default=0.1)
    p.add_argument("--policy_grad_clip", type=float, default=0.5)
    
    # Pruning losses
    p.add_argument("--prune_l1_weight", type=float, default=0.0)
    p.add_argument("--prune_kl_weight", type=float, default=0.0)
    p.add_argument("--prune_keep_rate", type=float, default=0.5)
    
    # Optimization (v4.0.0: Updated defaults for BPE)
    p.add_argument("--opt", type=str, default="adamw", choices=["adamw", "sgd"])
    p.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4 for BPE)")
    p.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay (default: 0.01 for BPE)")
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--nesterov", type=str2bool, nargs="?", const=True, default=True)
    p.add_argument("--adamw_beta1", type=float, default=0.9)
    p.add_argument("--adamw_beta2", type=float, default=0.999)
    p.add_argument("--adamw_eps", type=float, default=1e-8)
    
    # LR schedule
    p.add_argument("--lr_schedule", type=str, default="cosine", choices=["none", "cosine", "step"])
    p.add_argument("--lr_step_size", type=int, default=5)
    p.add_argument("--lr_gamma", type=float, default=0.5)
    
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", type=str2bool, nargs="?", const=True, default=False)
    p.add_argument("--save_path", type=str, default="checkpoints/boenet_bpe.pt")
    
    # Debug
    p.add_argument("--debug_node_counts", type=str2bool, nargs="?", const=True, default=False)
    p.add_argument("--debug_bfs_verify", type=str2bool, nargs="?", const=True, default=False)
    
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