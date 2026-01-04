#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
infer_boenet.py (v4.1.0 - BPE Only)

Load a trained BoeNet checkpoint and:
  1) Evaluate perplexity on test/validation split
  2) Measure per-sample inference latency with statistics (mean, p50, p90, p99)
  3) Measure node usage in greedy inference mode with True BFS verification
  4) DEBUG MODE: Analyze growth policy probabilities (per-LEVEL decisions)
  5) GENERATE TEXT: Autoregressive sampling with temperature/top-k/top-p
  6) Output a __SUMMARY__ JSON line for automated parsing

v4.1.0 Changes (2026-01-03)
---------------------------
BREAKING: BPE Only
  - Removed all CharTokenizer imports and fallbacks
  - Uses boenet.tokenizer.get_tokenizer("bpe") exclusively
  - No more character-level tokenization support
  - Simplified tokenizer initialization

RATIONALE:
  - Model trained with BPE (vocab=100,277)
  - Inference must use same tokenizer
  - CharTokenizer.decode() fails on BPE tokens > 255
  - No backward compatibility needed for this project

v4.0.0 Changes (2026-01-03)
---------------------------
MAJOR: BPE Tokenizer Support
  - Loads tokenizer settings from checkpoint (tokenizer_type, bpe_encoding)
  - Falls back to CLI args if checkpoint doesn't have tokenizer info
  - Supports both character-level (vocab=256) and BPE (vocab=100,277)
  - Uses boenet.tokenizer module for tokenization

v3.1.0 Changes (2026-01-02)
---------------------------
FIXES:
  - Fixed import error: get_num_nodes_at_level now exists in model.py v2.3.0
  - Updated version banner to match model.py
  - Added fallback if function doesn't exist (backward compatibility)

Author: BoeNet project
Version: 4.1.0
Date: 2026-01-03
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from boenet.model import BoeNet

# v3.1.0: Import BFS indexing functions with fallback
try:
    from boenet.model import (
        get_total_nodes_up_to_level,
        get_level,
        get_nodes_at_level,
        get_num_nodes_at_level,
    )
except ImportError:
    print("[infer][warning] Some BFS functions not found in model.py, using local definitions")
    
    def get_level(i: int) -> int:
        if i < 0:
            raise ValueError(f"Node index must be non-negative, got {i}")
        return int(math.floor(math.log2(i + 1)))
    
    def get_nodes_at_level(level: int) -> List[int]:
        if level < 0:
            raise ValueError(f"Level must be non-negative, got {level}")
        start = (1 << level) - 1
        end = (1 << (level + 1)) - 1
        return list(range(start, end))
    
    def get_num_nodes_at_level(level: int) -> int:
        if level < 0:
            raise ValueError(f"Level must be non-negative, got {level}")
        return 1 << level
    
    def get_total_nodes_up_to_level(level: int) -> int:
        if level < 0:
            return 0
        return (1 << (level + 1)) - 1

# v4.1.0: Import BPE tokenizer ONLY from boenet.tokenizer
from boenet.tokenizer import get_tokenizer, TiktokenWrapper

from boenet.utils.data_utils import (
    get_dataloaders,
    SplitConfig,
    set_seed,
)

from boenet.losses import (
    compute_perplexity,
    compute_depth_from_nodes,
    get_nodes_for_depth,
)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def compute_depth_from_total_nodes(total_nodes: int) -> int:
    """Compute depth from total node count."""
    if total_nodes <= 0:
        return 0
    return max(0, int(math.floor(math.log2(total_nodes + 1))) - 1)


def compute_theoretical_max_nodes(max_depth: int) -> int:
    """Compute theoretical max nodes for given depth."""
    if max_depth < 0:
        return 1
    return (1 << (max_depth + 1)) - 1


def get_valid_node_counts(max_depth: int) -> List[int]:
    """Get all valid node counts for depths 0 to max_depth."""
    return [get_nodes_for_depth(d) for d in range(max_depth + 1)]


def compute_percentile(sorted_values: List[float], percentile: float) -> float:
    """Compute percentile from sorted values."""
    if not sorted_values:
        return 0.0
    n = len(sorted_values)
    idx = (percentile / 100.0) * (n - 1)
    lower_idx = int(idx)
    upper_idx = min(lower_idx + 1, n - 1)
    fraction = idx - lower_idx
    return sorted_values[lower_idx] * (1 - fraction) + sorted_values[upper_idx] * fraction


def get_model_size_bytes(ckpt_path: str) -> int:
    """Get checkpoint file size in bytes."""
    try:
        return os.path.getsize(ckpt_path)
    except Exception:
        return 0


def _cfg_get(cfg: Dict[str, Any], key: str, default: Any) -> Any:
    """Get config value with default."""
    return cfg[key] if key in cfg else default


# =============================================================================
# TOKENIZER INITIALIZATION (v4.1.0 - BPE ONLY)
# =============================================================================

def initialize_tokenizer(cfg: Dict[str, Any], bpe_encoding: str = "cl100k_base") -> Tuple[TiktokenWrapper, int]:
    """
    Initialize BPE tokenizer.
    
    v4.1.0: BPE only, no CharTokenizer fallback.
    
    Parameters
    ----------
    cfg : Dict[str, Any]
        Checkpoint configuration.
    bpe_encoding : str
        BPE encoding name (default: cl100k_base).
        
    Returns
    -------
    Tuple[TiktokenWrapper, int]
        (tokenizer, vocab_size)
    """
    ckpt_vocab_size = cfg.get("vocab_size", 100277)
    ckpt_bpe_encoding = cfg.get("bpe_encoding", bpe_encoding)
    
    # v4.1.0: Always use BPE
    tokenizer = get_tokenizer("bpe", encoding_name=ckpt_bpe_encoding)
    vocab_size = tokenizer.vocab_size
    
    print(f"[tokenizer] Created TiktokenWrapper({ckpt_bpe_encoding})")
    print(f"[tokenizer] Vocab size: {vocab_size:,}")
    
    # Verify vocab_size matches checkpoint
    if vocab_size != ckpt_vocab_size:
        print(f"[tokenizer][WARNING] Tokenizer vocab_size ({vocab_size:,}) != checkpoint vocab_size ({ckpt_vocab_size:,})")
        print(f"[tokenizer][WARNING] This may cause incorrect predictions!")
    
    return tokenizer, vocab_size


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_model(ckpt_path: str, device: torch.device) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Load BoeNet checkpoint.
    
    Parameters
    ----------
    ckpt_path : str
        Path to checkpoint file.
    device : torch.device
        Device to load model to.
        
    Returns
    -------
    Tuple[nn.Module, Dict[str, Any]]
        (model, config)
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg: Dict[str, Any] = ckpt.get("config", {})
    
    version = cfg.get("version", "unknown")
    print(f"\n[infer] Loading checkpoint: {ckpt_path}")
    print(f"[infer] Version: {version}")
    
    # Model configuration
    vocab_size = int(_cfg_get(cfg, "vocab_size", 100277))
    embed_dim = int(_cfg_get(cfg, "embed_dim", 64))
    hidden_dim = int(_cfg_get(cfg, "hidden_dim", 644))
    seq_len = int(_cfg_get(cfg, "seq_len", 128))
    max_depth = int(_cfg_get(cfg, "max_depth", 4))
    max_children = int(_cfg_get(cfg, "max_children", 2))
    greedy_threshold = float(_cfg_get(cfg, "greedy_threshold", 0.5))
    min_explore_prob = float(_cfg_get(cfg, "min_explore_prob", 0.1))
    sibling_embed = bool(_cfg_get(cfg, "sibling_embed", True))
    use_pruning = bool(_cfg_get(cfg, "use_pruning", False))
    pruning_mode = str(_cfg_get(cfg, "pruning_mode", "learned"))
    pruning_threshold = float(_cfg_get(cfg, "pruning_threshold", 1e-3))
    pooling_mode = str(_cfg_get(cfg, "pooling_mode", "mean"))
    
    # v4.1.0: Log tokenizer info
    bpe_encoding = cfg.get("bpe_encoding", "cl100k_base")
    
    theoretical_max = compute_theoretical_max_nodes(max_depth)
    valid_counts = get_valid_node_counts(max_depth)
    
    print(f"[infer] Model config: vocab_size={vocab_size:,}, embed_dim={embed_dim}, hidden_dim={hidden_dim}")
    print(f"[infer] Tokenizer: BPE ({bpe_encoding})")
    print(f"[infer] True BFS: max_depth={max_depth}, theoretical_max={theoretical_max}")
    print(f"[infer] Valid node counts: {valid_counts}")
    
    model = BoeNet(
        vocab_size=vocab_size, embed_dim=embed_dim, hidden_dim=hidden_dim,
        max_depth=max_depth, max_children=max_children,
        greedy_threshold=greedy_threshold, min_explore_prob=min_explore_prob,
        sibling_embed=sibling_embed, use_pruning=use_pruning,
        pruning_mode=pruning_mode, pruning_threshold=pruning_threshold,
        pooling_mode=pooling_mode,
    ).to(device)
    
    state = ckpt.get("model_state_dict", ckpt.get("state_dict", None))
    if state is None:
        raise KeyError("Checkpoint is missing 'model_state_dict'")
    
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[infer][note] Missing keys: {sorted(list(missing))[:5]}")
    if unexpected:
        print(f"[infer][note] Unexpected keys: {sorted(list(unexpected))[:5]}")
    
    model.eval()
    return model, cfg


# =============================================================================
# PERPLEXITY EVALUATION
# =============================================================================

@torch.no_grad()
def evaluate_perplexity(model: nn.Module, loader, device: torch.device, vocab_size: int) -> Tuple[float, float]:
    """
    Evaluate model perplexity on data loader.
    
    Parameters
    ----------
    model : nn.Module
        BoeNet model.
    loader : DataLoader
        Validation data loader.
    device : torch.device
        Device.
    vocab_size : int
        Vocabulary size for logging.
        
    Returns
    -------
    Tuple[float, float]
        (average_loss, perplexity)
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    for input_ids, labels in loader:
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        logits = model(input_ids)
        B, seq_len, V = logits.shape
        loss = F.cross_entropy(logits.view(-1, V), labels.view(-1), reduction='sum')
        total_loss += loss.item()
        total_tokens += labels.numel()
    
    avg_loss = total_loss / max(1, total_tokens)
    ppl = compute_perplexity(avg_loss)
    return avg_loss, ppl


# =============================================================================
# LATENCY MEASUREMENT
# =============================================================================

@torch.no_grad()
def measure_latency(model: nn.Module, loader, device: torch.device, 
                    num_samples: int = 1000, warmup: int = 10) -> Dict[str, float]:
    """
    Measure inference latency.
    
    Parameters
    ----------
    model : nn.Module
        BoeNet model.
    loader : DataLoader
        Data loader.
    device : torch.device
        Device.
    num_samples : int
        Number of samples to measure.
    warmup : int
        Number of warmup iterations.
        
    Returns
    -------
    Dict[str, float]
        Latency statistics (mean, p50, p90, p99).
    """
    model.eval()
    all_samples = []
    for input_ids, _ in loader:
        for i in range(input_ids.size(0)):
            all_samples.append(input_ids[i:i+1])
            if len(all_samples) >= num_samples + warmup:
                break
        if len(all_samples) >= num_samples + warmup:
            break
    
    if len(all_samples) < warmup:
        return {
            "latency_ms_mean": None, 
            "latency_ms_p50": None, 
            "latency_ms_p90": None, 
            "latency_ms_p99": None, 
            "num_samples_measured": 0
        }
    
    # Warmup
    for i in range(warmup):
        _ = model(all_samples[i].to(device))
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    # Measure
    latencies_ms = []
    for x in all_samples[warmup:warmup + num_samples]:
        x = x.to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        latencies_ms.append((time.perf_counter() - start) * 1000.0)
    
    latencies_sorted = sorted(latencies_ms)
    return {
        "latency_ms_mean": round(sum(latencies_ms) / len(latencies_ms), 4),
        "latency_ms_p50": round(compute_percentile(latencies_sorted, 50.0), 4),
        "latency_ms_p90": round(compute_percentile(latencies_sorted, 90.0), 4),
        "latency_ms_p99": round(compute_percentile(latencies_sorted, 99.0), 4),
        "num_samples_measured": len(latencies_ms)
    }


# =============================================================================
# NODE USAGE MEASUREMENT
# =============================================================================

@torch.no_grad()
def measure_node_usage_true_bfs(model: nn.Module, loader, device: torch.device, 
                                 num_samples: int = 1000,
                                 debug_nodes: bool = False, 
                                 debug_policy: bool = False,
                                 debug_bfs: bool = False, 
                                 force_growth: bool = False) -> Dict[str, Any]:
    """
    Measure node usage in True BFS mode.
    
    Parameters
    ----------
    model : nn.Module
        BoeNet model.
    loader : DataLoader
        Data loader.
    device : torch.device
        Device.
    num_samples : int
        Number of samples to measure.
    debug_nodes : bool
        Enable node debugging.
    debug_policy : bool
        Enable policy debugging.
    debug_bfs : bool
        Enable BFS structure debugging.
    force_growth : bool
        Force tree expansion for testing.
        
    Returns
    -------
    Dict[str, Any]
        Node usage statistics.
    """
    model.eval()
    
    max_depth = getattr(model, 'max_depth', 0)
    greedy_threshold = getattr(model, 'greedy_threshold', 0.5)
    
    if max_depth == 0:
        return {
            "avg_nodes_per_position": 1.0, 
            "theoretical_max": 1, 
            "sparsity_percent": 0.0,
            "avg_depth": 0.0, 
            "num_samples_measured": num_samples, 
            "bfs_type": "true_bfs"
        }
    
    theoretical_max = compute_theoretical_max_nodes(max_depth)
    valid_node_counts = get_valid_node_counts(max_depth)
    
    level_growth_probs: Dict[int, List[float]] = defaultdict(list)
    hook_call_counter = [0]
    
    def growth_policy_hook(module, input, output):
        probs = output.detach().cpu()
        mean_prob = probs.mean().item()
        current_level = hook_call_counter[0]
        level_growth_probs[current_level].append(mean_prob)
        hook_call_counter[0] += 1
    
    policy_hook = None
    if debug_policy and hasattr(model, 'growth_policy') and model.growth_policy is not None:
        policy_hook = model.growth_policy.register_forward_hook(growth_policy_hook)
    
    original_forward = None
    if force_growth and hasattr(model, 'growth_policy') and model.growth_policy is not None:
        original_forward = model.growth_policy.forward
        def forced_growth_forward(h, level):
            N = h.size(0)
            return torch.full((N,), 0.9, device=h.device, dtype=h.dtype)
        model.growth_policy.forward = forced_growth_forward
        print("[debug] Policy OVERRIDDEN: forcing grow_prob = 0.9")
    
    samples_counted = 0
    total_positions = 0
    
    for input_ids, _ in loader:
        if samples_counted >= num_samples:
            break
        for i in range(input_ids.size(0)):
            if samples_counted >= num_samples:
                break
            x = input_ids[i:i+1].to(device)
            if debug_policy:
                hook_call_counter[0] = 0
            _ = model(x)
            samples_counted += 1
            total_positions += x.size(1)
    
    if policy_hook is not None:
        policy_hook.remove()
    if force_growth and original_forward is not None:
        model.growth_policy.forward = original_forward
        print("[debug] Policy RESTORED")
    
    result = {
        "theoretical_max": theoretical_max, 
        "valid_node_counts": valid_node_counts,
        "num_samples_measured": samples_counted, 
        "total_positions": total_positions,
        "max_depth": max_depth, 
        "greedy_threshold": greedy_threshold, 
        "bfs_type": "true_bfs"
    }
    
    if debug_policy and level_growth_probs:
        print("\n" + "=" * 79)
        print("TRUE BFS GROWTH POLICY ANALYSIS (Per-Level)")
        print("=" * 79)
        print(f"\nModel has max_depth={max_depth}, so at most {max_depth} decisions per position.\n")
        
        all_probs = []
        for level in sorted(level_growth_probs.keys()):
            probs = level_growth_probs[level]
            if not probs:
                continue
            all_probs.extend(probs)
            mean_p = sum(probs) / len(probs)
            above_thresh = sum(1 for p in probs if p >= greedy_threshold) / len(probs) * 100
            status = "EXPAND" if mean_p >= greedy_threshold else "STOP"
            print(f"Level {level} -> {level+1}: mean={mean_p:.4f} ({status}), {above_thresh:.1f}% >= {greedy_threshold}")
        
        if all_probs:
            overall_mean = sum(all_probs) / len(all_probs)
            overall_above = sum(1 for p in all_probs if p >= greedy_threshold) / len(all_probs) * 100
            print(f"\nOverall: mean={overall_mean:.4f}, {overall_above:.1f}% >= threshold")
            result["debug_policy_overall_mean"] = round(overall_mean, 4)
            result["debug_policy_above_threshold_pct"] = round(overall_above, 2)
        print("=" * 79)
    
    if debug_bfs:
        print("\n" + "=" * 79)
        print("TRUE BFS TREE STRUCTURE")
        print("=" * 79)
        print(f"Max depth: {max_depth}, Theoretical max: {theoretical_max}")
        print(f"Valid counts: {valid_node_counts}")
        for d in range(max_depth + 1):
            print(f"  depth={d}: {get_nodes_for_depth(d)} total, {get_num_nodes_at_level(d)} at level")
        print("=" * 79)
    
    return result


# =============================================================================
# TEXT GENERATION (v4.1.0 - BPE ONLY)
# =============================================================================

@torch.no_grad()
def generate_text(model: nn.Module, tokenizer: TiktokenWrapper, prompt: str, 
                  max_tokens: int = 200, temperature: float = 1.0, 
                  top_k: int = 0, top_p: float = 1.0,
                  device: torch.device = torch.device("cpu"),
                  verbose: bool = False) -> str:
    """
    Generate text autoregressively using BPE tokenizer.
    
    v4.1.0: BPE only, uses TiktokenWrapper.
    
    Parameters
    ----------
    model : nn.Module
        BoeNet model.
    tokenizer : TiktokenWrapper
        BPE tokenizer (cl100k_base).
    prompt : str
        Starting text.
    max_tokens : int
        Maximum tokens to generate.
    temperature : float
        Sampling temperature (higher = more random).
    top_k : int
        Top-k sampling (0 = disabled).
    top_p : float
        Nucleus sampling threshold (1.0 = disabled).
    device : torch.device
        Device.
    verbose : bool
        Print progress during generation.
        
    Returns
    -------
    str
        Generated text.
    """
    model.eval()
    
    # Encode prompt using BPE
    tokens = tokenizer.encode(prompt)
    if len(tokens) == 0:
        # Use a space as fallback
        tokens = tokenizer.encode(" ")
        if len(tokens) == 0:
            tokens = [tokenizer.eos_token_id]
    
    generated = list(tokens)
    seq_len = getattr(model, 'seq_len', 128)
    vocab_size = tokenizer.vocab_size
    
    if verbose:
        print(f"[generate] Prompt: {len(tokens)} tokens, seq_len={seq_len}, vocab_size={vocab_size:,}")
    
    for i in range(max_tokens):
        # Get context (last seq_len tokens)
        context = generated[-seq_len:] if len(generated) > seq_len else generated
        
        # Pad if needed (left padding with pad token)
        if len(context) < seq_len:
            pad_token = tokenizer.pad_token_id
            context = [pad_token] * (seq_len - len(context)) + context
        
        input_ids = torch.tensor([context], dtype=torch.long, device=device)
        logits = model(input_ids)
        next_logits = logits[0, -1, :]  # [vocab_size]
        
        # Temperature scaling
        if temperature != 1.0:
            next_logits = next_logits / temperature
        
        # Top-k filtering
        if top_k > 0:
            indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][..., -1, None]
            next_logits[indices_to_remove] = float('-inf')
        
        # Top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(
                dim=-1, index=sorted_indices, src=sorted_indices_to_remove
            )
            next_logits[indices_to_remove] = float('-inf')
        
        # Sample
        probs = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).item()
        generated.append(next_token)
        
        if verbose and (i + 1) % 50 == 0:
            print(f"[generate] Generated {i + 1}/{max_tokens} tokens...")
    
    # Decode all tokens using BPE
    return tokenizer.decode(generated)


# =============================================================================
# SUMMARY JSON
# =============================================================================

def build_summary_json(val_loss: float, val_ppl: float, 
                       latency_stats: Optional[Dict],
                       node_stats: Optional[Dict], 
                       device: torch.device, 
                       cfg: Dict,
                       ckpt_path: str, 
                       num_samples: int,
                       bpe_encoding: str,
                       vocab_size: int) -> Dict[str, Any]:
    """
    Build summary JSON for automated parsing.
    
    v4.1.0: BPE only, simplified tokenizer info.
    """
    summary = {
        "val_loss": round(val_loss, 4), 
        "val_ppl": round(val_ppl, 2),
        "device": str(device), 
        "num_samples": num_samples,
        "model_bytes": get_model_size_bytes(ckpt_path), 
        "checkpoint_path": ckpt_path,
        "version": cfg.get("version", "unknown"), 
        "bfs_type": cfg.get("bfs_type", "true_bfs"),
        # v4.1.0: BPE only
        "tokenizer_type": "bpe",
        "bpe_encoding": bpe_encoding,
        "vocab_size": vocab_size,
    }
    
    if latency_stats:
        summary.update({
            k: latency_stats.get(k) 
            for k in ["latency_ms_mean", "latency_ms_p50", "latency_ms_p90", "latency_ms_p99"]
        })
    
    if node_stats:
        summary.update({
            "theoretical_max_nodes": node_stats.get("theoretical_max"),
            "max_depth": node_stats.get("max_depth"), 
            "greedy_threshold": node_stats.get("greedy_threshold")
        })
        if "debug_policy_overall_mean" in node_stats:
            summary["debug_policy_overall_mean"] = node_stats["debug_policy_overall_mean"]
    
    summary["model_config"] = {
        k: cfg.get(k) 
        for k in [
            "vocab_size", "embed_dim", "hidden_dim", "seq_len", 
            "max_depth", "max_children", "pooling_mode", 
            "greedy_threshold", "min_explore_prob",
            "bpe_encoding"
        ]
    }
    
    return summary


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point."""
    p = argparse.ArgumentParser(
        description="BoeNet v4.1.0 Inference - BPE Only",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic inference
  python infer_boenet.py --ckpt runs/model.pt

  # Generate text
  python infer_boenet.py --ckpt runs/model.pt --generate --prompt "The quick brown"

  # Debug policy decisions
  python infer_boenet.py --ckpt runs/model.pt --debug_policy

  # Evaluate with custom batch size
  python infer_boenet.py --ckpt runs/model.pt --batch_size 8
        """
    )
    
    # Checkpoint
    p.add_argument("--ckpt", type=str, default="checkpoints/boenet_wikitext2.pt",
                   help="Path to checkpoint file")
    
    # v4.1.0: BPE encoding (simplified)
    p.add_argument("--bpe_encoding", type=str, default="cl100k_base",
                   help="BPE encoding name (default: cl100k_base)")
    
    # Data
    p.add_argument("--dataset", type=str, default=None,
                   help="Dataset name (default: from checkpoint)")
    p.add_argument("--batch_size", type=int, default=8,
                   help="Batch size for evaluation (default: 8 for BPE)")
    p.add_argument("--samples", type=int, default=1000,
                   help="Number of samples for latency/node measurement")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed")
    
    # Device
    p.add_argument("--cpu", action="store_true",
                   help="Force CPU inference")
    
    # Latency
    p.add_argument("--latency_warmup", type=int, default=10,
                   help="Number of warmup iterations")
    p.add_argument("--skip_latency", action="store_true",
                   help="Skip latency measurement")
    
    # Nodes
    p.add_argument("--skip_nodes", action="store_true",
                   help="Skip node usage measurement")
    
    # Debug
    p.add_argument("--debug_policy", action="store_true",
                   help="Debug growth policy probabilities")
    p.add_argument("--debug_nodes", action="store_true",
                   help="Debug node usage")
    p.add_argument("--debug_bfs", action="store_true",
                   help="Debug BFS tree structure")
    p.add_argument("--force_growth", action="store_true",
                   help="Force tree expansion (testing only)")
    
    # Generation
    p.add_argument("--generate", action="store_true",
                   help="Generate text instead of evaluating")
    p.add_argument("--prompt", type=str, default="The ",
                   help="Prompt for text generation")
    p.add_argument("--max_tokens", type=int, default=200,
                   help="Maximum tokens to generate")
    p.add_argument("--temperature", type=float, default=0.8,
                   help="Sampling temperature")
    p.add_argument("--top_k", type=int, default=0,
                   help="Top-k sampling (0 = disabled)")
    p.add_argument("--top_p", type=float, default=1.0,
                   help="Nucleus sampling threshold")
    p.add_argument("--verbose_generate", action="store_true",
                   help="Verbose generation progress")
    
    args = p.parse_args()
    
    # Setup
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    
    print("\n" + "=" * 79)
    print("BoeNet v4.1.0 Inference - BPE Only")
    print("=" * 79)
    
    # Load model
    model, cfg = load_model(args.ckpt, device)
    
    # v4.1.0: Initialize BPE tokenizer (no CharTokenizer)
    bpe_encoding = cfg.get("bpe_encoding", args.bpe_encoding)
    tokenizer, vocab_size = initialize_tokenizer(cfg, bpe_encoding)
    
    # Get other config
    seq_len = cfg.get("seq_len", 128)
    dataset_name = args.dataset or cfg.get("dataset", "wikitext2")
    
    print(f"\n[config] Device: {device}")
    print(f"[config] Tokenizer: BPE ({bpe_encoding})")
    print(f"[config] Vocab size: {vocab_size:,}")
    print(f"[config] Sequence length: {seq_len}")
    
    # Text generation mode
    if args.generate:
        print(f"\n[generate] Generating from prompt: \"{args.prompt}\"")
        print(f"[generate] Settings: max_tokens={args.max_tokens}, temp={args.temperature}, top_k={args.top_k}, top_p={args.top_p}")
        
        generated = generate_text(
            model, tokenizer, args.prompt, 
            args.max_tokens, args.temperature, args.top_k, args.top_p, 
            device, args.verbose_generate
        )
        
        print("\n" + "=" * 79)
        print("GENERATED TEXT")
        print("=" * 79)
        print(generated)
        print("=" * 79)
        return
    
    # Evaluation mode
    print(f"\n[data] Loading {dataset_name} with BPE tokenizer...")
    
    # v4.1.0: Pass BPE tokenizer to data loader
    train_loader, val_loader, loader_vocab_size = get_dataloaders(
        dataset_name, 
        batch_size=args.batch_size, 
        seed=args.seed, 
        seq_len=seq_len,
        tokenizer=tokenizer,
    )
    
    print(f"[data] Val batches: {len(val_loader)}")
    print(f"[data] Vocab size from loader: {loader_vocab_size:,}")
    
    # Verify vocab sizes match
    if loader_vocab_size != vocab_size:
        print(f"[WARNING] Loader vocab ({loader_vocab_size:,}) != tokenizer vocab ({vocab_size:,})")
    
    # Perplexity evaluation
    print(f"\n[perplexity] Evaluating on {dataset_name}...")
    val_loss, val_ppl = evaluate_perplexity(model, val_loader, device, vocab_size)
    print(f"[perplexity] Val loss: {val_loss:.4f}")
    print(f"[perplexity] Val PPL: {val_ppl:.2f}")
    print(f"[perplexity] Random baseline: {vocab_size:,}")
    print(f"[perplexity] Improvement: {vocab_size / val_ppl:.2f}x better than random")
    
    # Latency measurement
    latency_stats = None
    if not args.skip_latency:
        print(f"\n[latency] Measuring inference latency...")
        latency_stats = measure_latency(model, val_loader, device, args.samples, args.latency_warmup)
        print(f"[latency] mean={latency_stats['latency_ms_mean']}ms")
        print(f"[latency] p50={latency_stats['latency_ms_p50']}ms")
        print(f"[latency] p90={latency_stats['latency_ms_p90']}ms")
        print(f"[latency] p99={latency_stats['latency_ms_p99']}ms")
    
    # Node usage measurement
    node_stats = None
    if not args.skip_nodes:
        print(f"\n[nodes] Measuring True BFS node usage...")
        node_stats = measure_node_usage_true_bfs(
            model, val_loader, device, args.samples,
            args.debug_nodes, args.debug_policy, args.debug_bfs, args.force_growth
        )
        print(f"[nodes] max_depth={node_stats['max_depth']}")
        print(f"[nodes] theoretical_max={node_stats['theoretical_max']}")
        print(f"[nodes] greedy_threshold={node_stats['greedy_threshold']}")
    
    # Summary
    summary = build_summary_json(
        val_loss, val_ppl, latency_stats, node_stats, 
        device, cfg, args.ckpt, args.samples,
        bpe_encoding, vocab_size
    )
    
    print("\n" + "=" * 79)
    print("INFERENCE SUMMARY")
    print("=" * 79)
    print(f"  Checkpoint: {args.ckpt}")
    print(f"  Tokenizer: BPE ({bpe_encoding})")
    print(f"  Vocab size: {vocab_size:,}")
    print(f"  Val PPL: {val_ppl:.2f}")
    print(f"  Random baseline: {vocab_size:,}")
    print(f"  Improvement: {vocab_size / val_ppl:.2f}x")
    print("=" * 79)
    
    # JSON summary for parsing
    print(f"\n__SUMMARY__ {json.dumps(summary)}")


if __name__ == "__main__":
    main()