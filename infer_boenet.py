#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
infer_boenet.py (v2.0.0 - Language Model)

Load a trained BoeNet checkpoint and:
  1) Evaluate perplexity on test/validation split
  2) Measure per-sample inference latency with statistics (mean, p50, p90, p99)
  3) Measure node usage in greedy inference mode
  4) DEBUG MODE: Analyze growth policy probabilities
  5) GENERATE TEXT: Autoregressive sampling with temperature/top-k/top-p
  6) Output a __SUMMARY__ JSON line for automated parsing

Converted from infer_fmnist_bfs.py (Vision) to infer_boenet.py (Language)
-------------------------------------------------------------------------
Key Changes:
  - REMOVED: FashionMNIST test loading, accuracy calculation, class names
  - ADDED: Text dataset loading, perplexity calculation, text generation
  - UNCHANGED: Latency measurement, node counting, policy debug hooks

v2.0.0 Dataset Changes:
-----------------------
  - Default dataset changed from shakespeare to wikitext2
  - WikiText-2 uses modern HuggingFace Parquet format (no script issues)
  - Shakespeare now downloads directly from Karpathy's GitHub
  - Added wikitext103, bookcorpus, openwebtext options

v2.0.0 Debug Enhancements (same as BFSNet v2.0.0)
-------------------------------------------------
  - --debug_policy: Capture and analyze growth probabilities
  - --debug_nodes: Detailed node creation logging
  - --force_growth: Override policy to test hook mechanism
  - Growth probability statistics reporting
  - Threshold mismatch warnings and recommendations

Text Generation:
----------------
  - --generate: Enable text generation mode
  - --prompt: Starting text for generation
  - --max_tokens: Maximum tokens to generate
  - --temperature: Sampling temperature (higher = more random)
  - --top_k: Top-k sampling (0 = disabled)
  - --top_p: Nucleus sampling threshold (1.0 = disabled)

Usage Examples
--------------
# Basic inference (perplexity evaluation):
python3 infer_boenet.py --ckpt checkpoints/boenet_wikitext2.pt

# Text generation:
python3 infer_boenet.py --ckpt checkpoints/boenet_wikitext2.pt \\
    --generate --prompt "The history of" --max_tokens 200 --temperature 0.8

# Debug mode (analyze policy - RECOMMENDED after training):
python3 infer_boenet.py --ckpt checkpoints/boenet_wikitext2.pt \\
    --debug_policy --samples 1000 --cpu

# Force growth (verify hook works):
python3 infer_boenet.py --ckpt checkpoints/boenet_wikitext2.pt \\
    --force_growth --samples 100 --cpu

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
from boenet.utils.data_utils import (
    get_dataloaders,
    SplitConfig,
    CharTokenizer,
    set_seed,
)
from boenet.losses import compute_perplexity


# ------------------------------ Small utils -------------------------------- #

def compute_percentile(sorted_values: List[float], percentile: float) -> float:
    """Compute the given percentile from a sorted list of values."""
    if not sorted_values:
        return 0.0
    n = len(sorted_values)
    idx = (percentile / 100.0) * (n - 1)
    lower_idx = int(idx)
    upper_idx = min(lower_idx + 1, n - 1)
    fraction = idx - lower_idx
    return sorted_values[lower_idx] * (1 - fraction) + sorted_values[upper_idx] * fraction


def get_model_size_bytes(ckpt_path: str) -> int:
    """Get the file size of the checkpoint in bytes."""
    try:
        return os.path.getsize(ckpt_path)
    except Exception:
        return 0


def _cfg_get(cfg: Dict[str, Any], key: str, default: Any) -> Any:
    """Safely get a value from config dict."""
    return cfg[key] if key in cfg else default


# ------------------------------- Model load -------------------------------- #

def load_model(
    ckpt_path: str,
    device: torch.device,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Load BoeNet v2.0.0 checkpoint.
    
    Parameters
    ----------
    ckpt_path : str
        Path to checkpoint file.
    device : torch.device
        Device to load model onto.
        
    Returns
    -------
    Tuple[nn.Module, Dict[str, Any]]
        (model, config_dict)
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg: Dict[str, Any] = ckpt.get("config", {})
    
    version = cfg.get("version", "unknown")
    model_type = cfg.get("model_type", "unknown")
    
    if version not in ("1.0.0", "2.0.0"):
        print(f"[warning] Checkpoint version is '{version}', expected '1.0.0' or '2.0.0'. Proceeding anyway...")
    if model_type != "language":
        print(f"[warning] Model type is '{model_type}', expected 'language'. Proceeding anyway...")
    
    # Extract config
    vocab_size = int(_cfg_get(cfg, "vocab_size", 256))
    embed_dim = int(_cfg_get(cfg, "embed_dim", 64))
    hidden_dim = int(_cfg_get(cfg, "hidden_dim", 128))
    seq_len = int(_cfg_get(cfg, "seq_len", 128))
    max_depth = int(_cfg_get(cfg, "max_depth", 2))
    max_children = int(_cfg_get(cfg, "max_children", 3))
    greedy_threshold = float(_cfg_get(cfg, "greedy_threshold", 0.5))
    sibling_embed = bool(_cfg_get(cfg, "sibling_embed", True))
    use_pruning = bool(_cfg_get(cfg, "use_pruning", False))
    pruning_mode = str(_cfg_get(cfg, "pruning_mode", "learned"))
    pruning_threshold = float(_cfg_get(cfg, "pruning_threshold", 1e-3))
    pooling_mode = str(_cfg_get(cfg, "pooling_mode", "mean"))
    
    print(f"[infer] Loading v2.0.0 checkpoint: {ckpt_path}")
    print(f"[infer] Model config: vocab_size={vocab_size}, embed_dim={embed_dim}, "
          f"hidden_dim={hidden_dim}, seq_len={seq_len}")
    print(f"[infer] Architecture: max_depth={max_depth}, max_children={max_children}, "
          f"pooling={pooling_mode}")
    print(f"[infer] Greedy threshold: {greedy_threshold}")
    
    model = BoeNet(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        max_depth=max_depth,
        max_children=max_children,
        greedy_threshold=greedy_threshold,
        sibling_embed=sibling_embed,
        use_pruning=use_pruning,
        pruning_mode=pruning_mode,
        pruning_threshold=pruning_threshold,
        pooling_mode=pooling_mode,
    ).to(device)
    
    state = ckpt.get("model_state_dict", ckpt.get("state_dict", None))
    if state is None:
        raise KeyError("Checkpoint is missing 'model_state_dict' (or 'state_dict').")
    
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[infer][note] Missing keys (OK if expected): {sorted(list(missing))[:5]}")
    if unexpected:
        print(f"[infer][note] Unexpected keys (OK): {sorted(list(unexpected))[:5]}")
    
    has_growth_policy = hasattr(model, 'growth_policy') and model.growth_policy is not None
    if not has_growth_policy and (max_depth > 0 and max_children > 0):
        print(f"[warning] Model should have growth_policy for max_depth={max_depth}, "
              f"max_children={max_children}")
    
    model.eval()
    
    return model, cfg


# ------------------------------ Evaluation --------------------------------- #

@torch.no_grad()
def evaluate_perplexity(
    model: nn.Module, 
    loader: torch.utils.data.DataLoader, 
    device: torch.device,
    vocab_size: int,
) -> Tuple[float, float]:
    """
    Evaluate model perplexity on a dataset.
    
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
        
        logits = model(input_ids)  # [B, seq_len, vocab_size]
        
        B, seq_len, V = logits.shape
        logits_flat = logits.view(-1, V)
        labels_flat = labels.view(-1)
        
        loss = F.cross_entropy(logits_flat, labels_flat, reduction='sum')
        
        total_loss += loss.item()
        total_tokens += labels.numel()
    
    avg_loss = total_loss / max(1, total_tokens)
    ppl = compute_perplexity(avg_loss)
    
    return avg_loss, ppl


@torch.no_grad()
def measure_latency(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_samples: int = 1000,
    warmup_iterations: int = 10,
) -> Dict[str, float]:
    """
    Measure per-sample inference latency.
    
    Returns
    -------
    Dict[str, float]
        Latency statistics (mean, p50, p90, p99).
    """
    model.eval()
    
    # Collect samples
    all_samples = []
    for input_ids, labels in loader:
        for i in range(input_ids.size(0)):
            all_samples.append(input_ids[i:i+1])
            if len(all_samples) >= num_samples + warmup_iterations:
                break
        if len(all_samples) >= num_samples + warmup_iterations:
            break
    
    if len(all_samples) < warmup_iterations:
        return {
            "latency_ms_mean": None,
            "latency_ms_p50": None,
            "latency_ms_p90": None,
            "latency_ms_p99": None,
            "num_samples_measured": 0
        }
    
    # Warmup
    for i in range(warmup_iterations):
        x = all_samples[i].to(device)
        _ = model(x)
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    # Measure
    latencies_ms: List[float] = []
    samples_to_measure = all_samples[warmup_iterations:warmup_iterations + num_samples]
    
    for x in samples_to_measure:
        x = x.to(device)
        
        if device.type == "cuda":
            torch.cuda.synchronize()
        
        start_time = time.perf_counter()
        _ = model(x)
        
        if device.type == "cuda":
            torch.cuda.synchronize()
        
        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000.0
        latencies_ms.append(latency_ms)
    
    latencies_sorted = sorted(latencies_ms)
    
    return {
        "latency_ms_mean": round(sum(latencies_ms) / len(latencies_ms), 4),
        "latency_ms_p50": round(compute_percentile(latencies_sorted, 50.0), 4),
        "latency_ms_p90": round(compute_percentile(latencies_sorted, 90.0), 4),
        "latency_ms_p99": round(compute_percentile(latencies_sorted, 99.0), 4),
        "num_samples_measured": len(latencies_ms)
    }


@torch.no_grad()
def measure_node_usage(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_samples: int = 1000,
    debug_nodes: bool = False,
    debug_policy: bool = False,
    force_growth: bool = False,
) -> Dict[str, Any]:
    """
    Measure average node usage in greedy inference mode.
    
    Enhanced with debug capabilities to understand policy behavior.
    
    Parameters
    ----------
    model : nn.Module
        BoeNet model.
    loader : DataLoader
        Data loader for evaluation.
    device : torch.device
        Device.
    num_samples : int
        Number of samples to measure.
    debug_nodes : bool
        Enable detailed node logging.
    debug_policy : bool
        Capture and analyze growth probabilities.
    force_growth : bool
        Override policy to force growth.
        
    Returns
    -------
    Dict[str, Any]
        Node usage statistics.
    """
    model.eval()
    
    max_depth = getattr(model, 'max_depth', 0)
    max_children = getattr(model, 'max_children', 0)
    greedy_threshold = getattr(model, 'greedy_threshold', 0.5)
    
    if max_depth == 0 or max_children == 0:
        return {
            "avg_nodes_per_position": 1.0,
            "theoretical_max": 1,
            "sparsity_percent": 0.0,
            "num_samples_measured": num_samples
        }
    
    # Theoretical max per position
    theoretical_max = sum(max_children ** d for d in range(max_depth + 1))
    
    # ========================================================================
    # DEBUG: Capture ALL growth probabilities
    # ========================================================================
    all_growth_probs = []
    
    def growth_policy_hook(module, input, output):
        """Capture growth probabilities from policy network."""
        probs = output.squeeze(-1).detach().cpu().tolist()
        if isinstance(probs, float):
            probs = [probs]
        all_growth_probs.extend(probs)
    
    policy_hook = None
    if debug_policy and hasattr(model, 'growth_policy') and model.growth_policy is not None:
        policy_hook = model.growth_policy.register_forward_hook(growth_policy_hook)
    
    # ========================================================================
    # DEBUG: Count child_fc calls
    # ========================================================================
    child_fc_call_count = [0]
    child_fc_total_nodes = [0]
    
    def child_fc_debug_hook(module, input, output):
        """Debug hook to verify child_fc is being called."""
        child_fc_call_count[0] += 1
        batch_size = input[0].size(0)
        child_fc_total_nodes[0] += batch_size
        if debug_nodes and child_fc_call_count[0] <= 3:
            print(f"  [debug] child_fc call #{child_fc_call_count[0]}: "
                  f"input shape {input[0].shape}, created {batch_size} nodes")
    
    # ========================================================================
    # FORCE GROWTH: Override policy for testing
    # ========================================================================
    original_forward = None
    if force_growth and hasattr(model, 'growth_policy') and model.growth_policy is not None:
        original_forward = model.growth_policy.forward
        
        def forced_growth_forward(h, depth):
            """Force policy to always grow (grow_prob = 0.9)."""
            batch_size = h.size(0)
            return torch.full((batch_size, 1), 0.9, device=h.device, dtype=h.dtype)
        
        model.growth_policy.forward = forced_growth_forward
        print("[debug] Policy OVERRIDDEN: forcing grow_prob = 0.9 (testing hook)")
    
    # ========================================================================
    # Node counting hook
    # ========================================================================
    child_fc_calls = [0]
    
    def hook_fn(module, input, output):
        child_fc_calls[0] += input[0].size(0)
    
    hook = None
    child_debug_hook = None
    if hasattr(model, 'child_fc'):
        hook = model.child_fc.register_forward_hook(hook_fn)
        if debug_nodes:
            child_debug_hook = model.child_fc.register_forward_hook(child_fc_debug_hook)
    
    # ========================================================================
    # Measure node usage
    # ========================================================================
    total_nodes = 0
    total_positions = 0
    samples_counted = 0
    
    for input_ids, labels in loader:
        if samples_counted >= num_samples:
            break
            
        for i in range(input_ids.size(0)):
            if samples_counted >= num_samples:
                break
                
            x = input_ids[i:i+1].to(device)  # [1, seq_len]
            curr_seq_len = x.size(1)
            
            child_fc_calls[0] = 0
            child_fc_call_count[0] = 0
            child_fc_total_nodes[0] = 0
            
            if debug_nodes and samples_counted < 3:
                print(f"\n[debug] Processing sample {samples_counted}:")
            
            _ = model(x)
            
            # Count nodes: 1 root per position + children
            nodes_this_sample = curr_seq_len + child_fc_calls[0]
            total_nodes += nodes_this_sample
            total_positions += curr_seq_len
            samples_counted += 1
            
            if debug_nodes and samples_counted <= 3:
                print(f"  [debug] Total nodes: {nodes_this_sample} "
                      f"({curr_seq_len} roots + {child_fc_calls[0]} children)")
    
    # ========================================================================
    # Cleanup hooks
    # ========================================================================
    if hook is not None:
        hook.remove()
    if child_debug_hook is not None:
        child_debug_hook.remove()
    if policy_hook is not None:
        policy_hook.remove()
    
    # Restore original policy if overridden
    if force_growth and original_forward is not None:
        model.growth_policy.forward = original_forward
        print("[debug] Policy RESTORED to original")
    
    # ========================================================================
    # Calculate statistics
    # ========================================================================
    avg_nodes_per_position = total_nodes / max(1, total_positions)
    sparsity_percent = (1.0 - avg_nodes_per_position / theoretical_max) * 100.0 if theoretical_max > 0 else 0.0
    
    result = {
        "avg_nodes_per_position": round(avg_nodes_per_position, 2),
        "theoretical_max": theoretical_max,
        "sparsity_percent": round(sparsity_percent, 2),
        "num_samples_measured": samples_counted,
        "total_positions": total_positions,
    }
    
    # ========================================================================
    # DEBUG: Report growth probability statistics
    # ========================================================================
    if debug_policy and all_growth_probs:
        print("\n" + "=" * 79)
        print("GROWTH POLICY ANALYSIS (Greedy Inference)")
        print("=" * 79)
        
        mean_p = sum(all_growth_probs) / len(all_growth_probs)
        min_p = min(all_growth_probs)
        max_p = max(all_growth_probs)
        
        var = sum((p - mean_p) ** 2 for p in all_growth_probs) / len(all_growth_probs)
        std_p = var ** 0.5
        
        above_thresh = sum(1 for p in all_growth_probs if p >= greedy_threshold) / len(all_growth_probs) * 100
        
        print(f"\nTotal grow decisions evaluated: {len(all_growth_probs)}")
        print(f"  Mean grow_prob: {mean_p:.4f}")
        print(f"  Std dev:        {std_p:.4f}")
        print(f"  Min:            {min_p:.4f}")
        print(f"  Max:            {max_p:.4f}")
        print(f"  % >= {greedy_threshold:.2f}:      {above_thresh:.2f}%")
        
        # Histogram
        print(f"\nDistribution:")
        bins = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5), 
                (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]
        for low, high in bins:
            count = sum(1 for p in all_growth_probs if low <= p < high)
            pct = count / len(all_growth_probs) * 100
            bar = '#' * int(pct / 2)
            print(f"  [{low:.1f}-{high:.1f}): {count:>6} ({pct:>5.1f}%) {bar}")
        
        print("=" * 79)
        
        # Interpretation
        if above_thresh < 1.0:
            print(f"\n[!] CRITICAL FINDING:")
            print(f"  -> Only {above_thresh:.2f}% of grow decisions are above threshold ({greedy_threshold:.2f})")
            print(f"  -> Mean grow_prob is {mean_p:.4f} (below threshold)")
            print(f"  -> Greedy mode (threshold {greedy_threshold:.2f}) will create ZERO children")
            print(f"\n[*] RECOMMENDATION:")
            recommended_threshold = max(0.30, mean_p - 0.03)
            print(f"  -> Lower greedy threshold to {recommended_threshold:.2f}")
            print(f"  -> Or retrain with: --greedy_threshold {recommended_threshold:.2f}")
        elif above_thresh > 50:
            print(f"\n[+] Policy learned to grow:")
            print(f"  -> {above_thresh:.2f}% of decisions above threshold ({greedy_threshold:.2f})")
        
        print()
        
        result["debug_policy_mean_grow_prob"] = round(mean_p, 4)
        result["debug_policy_above_threshold_pct"] = round(above_thresh, 2)
    
    return result


# ------------------------------ Text Generation ---------------------------- #

@torch.no_grad()
def generate_text(
    model: nn.Module,
    tokenizer: CharTokenizer,
    prompt: str,
    max_tokens: int = 200,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    device: torch.device = torch.device("cpu"),
) -> str:
    """
    Generate text autoregressively from a prompt.
    
    Parameters
    ----------
    model : nn.Module
        BoeNet model.
    tokenizer : CharTokenizer
        Tokenizer for encoding/decoding.
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
        Device for inference.
        
    Returns
    -------
    str
        Generated text (prompt + new tokens).
    """
    model.eval()
    
    # Encode prompt
    tokens = tokenizer.encode(prompt)
    if len(tokens) == 0:
        tokens = [0]  # Start with padding if empty
    
    generated = list(tokens)
    
    # Get seq_len from model config or use default
    seq_len = 128
    if hasattr(model, 'seq_len'):
        seq_len = model.seq_len
    
    for _ in range(max_tokens):
        # Take last seq_len tokens as context
        context = generated[-seq_len:] if len(generated) > seq_len else generated
        
        # Pad if necessary (left padding)
        if len(context) < seq_len:
            context = [0] * (seq_len - len(context)) + context
        
        # Forward pass
        input_ids = torch.tensor([context], dtype=torch.long, device=device)
        logits = model(input_ids)  # [1, seq_len, vocab_size]
        
        # Get logits for last position
        next_logits = logits[0, -1, :]  # [vocab_size]
        
        # Apply temperature
        if temperature != 1.0:
            next_logits = next_logits / temperature
        
        # Apply top-k
        if top_k > 0:
            indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][..., -1, None]
            next_logits[indices_to_remove] = float('-inf')
        
        # Apply top-p (nucleus sampling)
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
    
    # Decode
    return tokenizer.decode(generated)


def show_generation_samples(
    model: nn.Module,
    tokenizer: CharTokenizer,
    device: torch.device,
    prompts: List[str],
    max_tokens: int = 100,
    temperature: float = 0.8,
) -> None:
    """Display text generation samples."""
    print("\n" + "=" * 79)
    print("TEXT GENERATION SAMPLES")
    print("=" * 79)
    
    for i, prompt in enumerate(prompts):
        print(f"\n[Sample {i+1}] Prompt: \"{prompt}\"")
        print("-" * 40)
        
        generated = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            device=device,
        )
        
        # Show generated part highlighted
        print(f"Generated:\n{generated}")
        print("-" * 40)
    
    print("=" * 79)


# ------------------------------ Summary JSON ------------------------------- #

def build_summary_json(
    val_loss: float,
    val_ppl: float,
    latency_stats: Optional[Dict[str, float]],
    node_stats: Optional[Dict[str, Any]],
    device: torch.device,
    cfg: Dict[str, Any],
    ckpt_path: str,
    num_samples: int,
) -> Dict[str, Any]:
    """Build the summary dictionary for JSON output."""
    summary: Dict[str, Any] = {
        "val_loss": round(val_loss, 4),
        "val_ppl": round(val_ppl, 2),
        "device": str(device),
        "num_samples": num_samples,
        "model_bytes": get_model_size_bytes(ckpt_path),
        "checkpoint_path": ckpt_path,
        "version": cfg.get("version", "unknown"),
    }
    
    if latency_stats is not None:
        summary.update({
            "latency_ms_mean": latency_stats.get("latency_ms_mean"),
            "latency_ms_p50": latency_stats.get("latency_ms_p50"),
            "latency_ms_p90": latency_stats.get("latency_ms_p90"),
            "latency_ms_p99": latency_stats.get("latency_ms_p99"),
            "latency_num_samples": latency_stats.get("num_samples_measured", 0),
        })
    
    if node_stats is not None:
        summary.update({
            "avg_nodes_per_position": node_stats.get("avg_nodes_per_position"),
            "theoretical_max_nodes": node_stats.get("theoretical_max"),
            "sparsity_percent": node_stats.get("sparsity_percent"),
            "node_samples": node_stats.get("num_samples_measured", 0),
        })
        
        if "debug_policy_mean_grow_prob" in node_stats:
            summary["debug_policy_mean_grow_prob"] = node_stats["debug_policy_mean_grow_prob"]
            summary["debug_policy_above_threshold_pct"] = node_stats["debug_policy_above_threshold_pct"]
    
    summary["model_config"] = {
        "vocab_size": cfg.get("vocab_size"),
        "embed_dim": cfg.get("embed_dim"),
        "hidden_dim": cfg.get("hidden_dim"),
        "seq_len": cfg.get("seq_len"),
        "max_depth": cfg.get("max_depth"),
        "max_children": cfg.get("max_children"),
        "pooling_mode": cfg.get("pooling_mode"),
        "greedy_threshold": cfg.get("greedy_threshold"),
        "num_rollouts": cfg.get("num_rollouts"),
        "lambda_efficiency": cfg.get("lambda_efficiency"),
        "beta_entropy": cfg.get("beta_entropy"),
        "beta_policy": cfg.get("beta_policy"),
    }
    
    return summary


# ------------------------------------ CLI ---------------------------------- #

def main():
    p = argparse.ArgumentParser(
        description="BoeNet v2.0.0 Language Model Inference (Debug Enhanced)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage Examples:
---------------
# Basic perplexity evaluation:
python3 infer_boenet.py --ckpt checkpoints/boenet_wikitext2.pt

# Text generation:
python3 infer_boenet.py --ckpt checkpoints/boenet_wikitext2.pt \\
    --generate --prompt "The history of" --max_tokens 200 --temperature 0.8

# Debug policy analysis:
python3 infer_boenet.py --ckpt checkpoints/boenet_wikitext2.pt \\
    --debug_policy --samples 1000 --cpu

Available Datasets:
-------------------
  wikitext2:   ~2MB Wikipedia (DEFAULT, RECOMMENDED)
  wikitext103: ~500MB Wikipedia
  shakespeare: ~1MB literary text (via GitHub)
  tinystories: ~2GB children's stories
  bookcorpus:  ~5GB books
  openwebtext: ~40GB web text
  textfile:    Custom local text file

See docs/architecture.md for complete threshold tuning guide.
        """
    )
    
    # Checkpoint
    p.add_argument("--ckpt", type=str, default="checkpoints/boenet_wikitext2.pt",
                   help="Path to .pt checkpoint")
    p.add_argument("--dataset", type=str, default=None,
                   choices=["wikitext2", "wikitext103", "shakespeare", "tinystories",
                            "bookcorpus", "openwebtext", "textfile"],
                   help="Dataset override (default: use checkpoint config)")
    
    # Evaluation
    p.add_argument("--batch_size", type=int, default=64,
                   help="Batch size for perplexity evaluation")
    p.add_argument("--samples", type=int, default=1000,
                   help="Number of samples for latency/node measurement")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true", help="Force CPU")
    
    # Latency measurement
    p.add_argument("--latency_warmup", type=int, default=10,
                   help="Number of warmup iterations before timing")
    p.add_argument("--skip_latency", action="store_true",
                   help="Skip latency measurement")
    
    # Node usage measurement
    p.add_argument("--skip_nodes", action="store_true",
                   help="Skip node usage measurement")
    
    # Debug options
    p.add_argument("--debug_policy", action="store_true",
                   help="Analyze growth policy probabilities in detail")
    p.add_argument("--debug_nodes", action="store_true",
                   help="Print detailed node creation logs")
    p.add_argument("--force_growth", action="store_true",
                   help="Override policy to force growth (test hook mechanism)")
    
    # Text generation
    p.add_argument("--generate", action="store_true",
                   help="Enable text generation mode")
    p.add_argument("--prompt", type=str, default="The ",
                   help="Starting text for generation")
    p.add_argument("--max_tokens", type=int, default=200,
                   help="Maximum tokens to generate")
    p.add_argument("--temperature", type=float, default=0.8,
                   help="Sampling temperature (higher = more random)")
    p.add_argument("--top_k", type=int, default=0,
                   help="Top-k sampling (0 = disabled)")
    p.add_argument("--top_p", type=float, default=1.0,
                   help="Nucleus sampling threshold (1.0 = disabled)")
    
    args = p.parse_args()
    
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    
    # Load model
    model, cfg = load_model(args.ckpt, device)
    
    # Print key config
    print("\n[config] Key fields:")
    for k in ["vocab_size", "embed_dim", "hidden_dim", "seq_len", "max_depth", 
              "max_children", "pooling_mode", "greedy_threshold", "num_rollouts", 
              "lambda_efficiency", "beta_policy", "dataset"]:
        if k in cfg:
            print(f"  {k}: {cfg[k]}")
    
    # Get vocab_size and seq_len
    vocab_size = cfg.get("vocab_size", 256)
    seq_len = cfg.get("seq_len", 128)
    dataset_name = args.dataset or cfg.get("dataset", "wikitext2")
    
    # Create tokenizer
    tokenizer = CharTokenizer()
    
    # Text generation mode
    if args.generate:
        print(f"\n[generate] Generating text from prompt: \"{args.prompt}\"")
        print(f"[generate] max_tokens={args.max_tokens}, temperature={args.temperature}, "
              f"top_k={args.top_k}, top_p={args.top_p}")
        
        generated = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            device=device,
        )
        
        print("\n" + "=" * 79)
        print("GENERATED TEXT")
        print("=" * 79)
        print(generated)
        print("=" * 79)
        
        # Also show a few more samples with different prompts
        sample_prompts = [
            "Once upon a time",
            "The king said",
            "In the beginning",
        ]
        show_generation_samples(
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompts=sample_prompts,
            max_tokens=100,
            temperature=args.temperature,
        )
        return
    
    # Load data for evaluation
    print(f"\n[data] Loading {dataset_name} dataset for evaluation...")
    train_loader, val_loader, _ = get_dataloaders(
        dataset_name,
        batch_size=args.batch_size,
        seed=args.seed,
        seq_len=seq_len,
    )
    
    print(f"[data] Val batches: {len(val_loader)}")
    
    # Evaluate perplexity
    print(f"\n[perplexity] Evaluating on validation set...")
    val_loss, val_ppl = evaluate_perplexity(model, val_loader, device, vocab_size)
    print(f"[perplexity] Val loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f}")
    print(f"[perplexity] Random baseline PPL: {vocab_size:.2f}")
    
    # Measure latency
    latency_stats: Optional[Dict[str, float]] = None
    if not args.skip_latency:
        print(f"\n[latency] Measuring inference latency ({args.samples} samples)...")
        latency_stats = measure_latency(
            model=model,
            loader=val_loader,
            device=device,
            num_samples=args.samples,
            warmup_iterations=args.latency_warmup,
        )
        print(f"[latency] mean={latency_stats['latency_ms_mean']:.4f}ms | "
              f"p50={latency_stats['latency_ms_p50']:.4f}ms | "
              f"p90={latency_stats['latency_ms_p90']:.4f}ms | "
              f"p99={latency_stats['latency_ms_p99']:.4f}ms")
    
    # Measure node usage
    node_stats: Optional[Dict[str, Any]] = None
    if not args.skip_nodes:
        if args.debug_policy or args.debug_nodes or args.force_growth:
            print(f"\n[nodes] Measuring node usage WITH DEBUG...")
        else:
            print(f"\n[nodes] Measuring node usage ({args.samples} samples)...")
        
        node_stats = measure_node_usage(
            model=model,
            loader=val_loader,
            device=device,
            num_samples=args.samples,
            debug_nodes=args.debug_nodes,
            debug_policy=args.debug_policy,
            force_growth=args.force_growth,
        )
        print(f"[nodes] avg_nodes_per_position={node_stats['avg_nodes_per_position']:.2f} | "
              f"theoretical_max={node_stats['theoretical_max']} | "
              f"sparsity={node_stats['sparsity_percent']:.2f}%")
    
    # Build and output summary JSON
    summary = build_summary_json(
        val_loss=val_loss,
        val_ppl=val_ppl,
        latency_stats=latency_stats,
        node_stats=node_stats,
        device=device,
        cfg=cfg,
        ckpt_path=args.ckpt,
        num_samples=args.samples,
    )
    
    print(f"\n__SUMMARY__ {json.dumps(summary)}")


if __name__ == "__main__":
    main()