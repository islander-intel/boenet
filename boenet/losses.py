#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
boenet/losses.py (v1.1.0 - Language Model)

Loss functions for BoeNet language models with REINFORCE policy gradients.

v1.1.0 Changes (Bug Fixes for K>0 Training)
-------------------------------------------
FIXED:
  - Added reward scaling to prevent gradient explosion
  - Added advantage clamping in policy loss
  - Added NaN detection throughout
  - Improved numerical stability in entropy computation

The gradient explosion causing CUDA assertion failures was traced to:
  1. Large negative rewards (CE loss can be 5-10 for untrained models)
  2. Large advantages causing huge policy gradients
  3. Policy weights exploding → NaN probabilities

Author: BoeNet project
Version: 1.1.0
Date: 2025-12-30
"""

from __future__ import annotations
import math
from typing import Dict, List, Optional, Literal, Sequence
import logging

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

Reduction = Literal["mean", "sum", "none"]

# Numerical stability constants
PROB_CLAMP_MIN = 1e-7
PROB_CLAMP_MAX = 1.0 - 1e-7
LOG_CLAMP_MIN = -20.0
REWARD_SCALE = 5.0  # Scale factor for rewards to prevent gradient explosion
ADVANTAGE_CLAMP = 2.0  # Maximum absolute advantage value


def _reduce(x: torch.Tensor, reduction: Reduction) -> torch.Tensor:
    if reduction == "mean":
        return x.mean()
    if reduction == "sum":
        return x.sum()
    if reduction == "none":
        return x
    raise ValueError(f"Unsupported reduction: {reduction!r}")


def _as_B(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 2 and x.size(1) == 1:
        return x.squeeze(1)
    return x


def _zeros_like_device_of(*tensors: Optional[torch.Tensor]) -> torch.Tensor:
    for t in tensors:
        if isinstance(t, torch.Tensor):
            return torch.zeros((), device=t.device, dtype=t.dtype)
    return torch.tensor(0.0)


def _check_nan(tensor: torch.Tensor, name: str, default: float = 0.0) -> torch.Tensor:
    """Check for NaN and replace with default value."""
    if torch.isnan(tensor).any():
        nan_count = torch.isnan(tensor).sum().item()
        logger.warning(f"[{name}] Contains {nan_count} NaN values. Replacing with {default}.")
        tensor = torch.where(torch.isnan(tensor), torch.full_like(tensor, default), tensor)
    return tensor


def compute_perplexity(cross_entropy_loss: torch.Tensor | float) -> float:
    """Compute perplexity from cross-entropy loss: PPL = exp(CE)."""
    if isinstance(cross_entropy_loss, torch.Tensor):
        cross_entropy_loss = cross_entropy_loss.item()
    # Clamp to prevent overflow
    cross_entropy_loss = min(cross_entropy_loss, 20.0)
    return math.exp(cross_entropy_loss)


def compute_rewards_language(
    outputs: List[torch.Tensor],
    node_counts: List[int],
    labels: torch.Tensor,
    lambda_efficiency: float,
    max_nodes_per_position: int,
    batch_size: int,
    seq_len: int,
    reward_scale: float = REWARD_SCALE,
) -> torch.Tensor:
    """
    Compute per-rollout rewards for REINFORCE (Language Model version).
    
    v1.1.0 CHANGES:
    - Added reward_scale parameter to prevent gradient explosion
    - Added NaN detection in CE loss
    - Clamped efficiency ratio to [0, 1]
    
    Parameters
    ----------
    outputs : List[torch.Tensor]
        Output logits from each rollout, each shape [B * seq_len, vocab_size].
    node_counts : List[int]
        Total node counts from each rollout.
    labels : torch.Tensor
        Ground truth tokens, shape [B, seq_len].
    lambda_efficiency : float
        Efficiency penalty coefficient.
    max_nodes_per_position : int
        Maximum possible nodes per token position.
    batch_size : int
        Batch size B.
    seq_len : int
        Sequence length.
    reward_scale : float
        Scale factor for rewards (default: 5.0).
        
    Returns
    -------
    torch.Tensor
        Scaled rewards, shape [num_rollouts].
    """
    num_positions = batch_size * seq_len
    labels_flat = labels.view(-1)
    
    rewards = []
    for rollout_idx, (out, nodes) in enumerate(zip(outputs, node_counts)):
        # Cross-entropy loss
        ce_loss = F.cross_entropy(out, labels_flat, reduction='mean')
        
        # Check for NaN/Inf
        if torch.isnan(ce_loss) or torch.isinf(ce_loss):
            logger.warning(f"[Reward] Rollout {rollout_idx}: CE loss is {ce_loss.item()}. Using default.")
            ce_loss = torch.tensor(10.0, device=out.device, dtype=out.dtype)
        
        # Reward = negative loss (higher is better)
        reward_quality = -ce_loss
        
        # Efficiency penalty (clamped to [0, 1])
        nodes_per_position = nodes / num_positions
        efficiency_ratio = min(max(nodes_per_position / float(max_nodes_per_position), 0.0), 1.0)
        efficiency_penalty = lambda_efficiency * efficiency_ratio
        
        # Combined and scaled reward
        reward = (reward_quality - efficiency_penalty) / reward_scale
        rewards.append(reward)
    
    return torch.stack(rewards)


def compute_rewards_classification(
    outputs: List[torch.Tensor],
    node_counts: List[int],
    labels: torch.Tensor,
    lambda_efficiency: float,
    max_nodes: int,
) -> torch.Tensor:
    """Compute per-rollout rewards for classification (backward compatibility)."""
    batch_size = outputs[0].size(0) if outputs else 1
    rewards = []
    
    for out, nodes in zip(outputs, node_counts):
        preds = out.argmax(dim=-1)
        accuracy = (preds == labels).float().mean()
        nodes_per_example = nodes / batch_size
        efficiency_penalty = lambda_efficiency * (nodes_per_example / float(max_nodes))
        reward = accuracy - efficiency_penalty
        rewards.append(reward)
    
    return torch.stack(rewards)


def policy_loss_reinforce(
    log_probs: List[torch.Tensor],
    rewards: torch.Tensor,
    beta_entropy: float = 0.01,
    advantage_clamp: float = ADVANTAGE_CLAMP,
) -> torch.Tensor:
    """
    REINFORCE policy loss with baseline, entropy bonus, and numerical stability.
    
    v1.1.0 CHANGES:
    - Added advantage clamping to prevent gradient explosion
    - Added NaN detection in log_probs
    - Improved entropy computation stability
    
    Parameters
    ----------
    log_probs : List[torch.Tensor]
        Log probability tensors from each rollout.
    rewards : torch.Tensor
        Rewards for each rollout (already scaled).
    beta_entropy : float
        Entropy bonus coefficient.
    advantage_clamp : float
        Maximum absolute advantage value.
        
    Returns
    -------
    torch.Tensor
        Scalar policy loss.
    """
    if len(log_probs) == 0:
        return torch.tensor(0.0, requires_grad=True)
    
    # Filter None entries
    valid_pairs = [(lp, r) for lp, r in zip(log_probs, rewards) if lp is not None]
    if len(valid_pairs) == 0:
        return torch.tensor(0.0, device=rewards.device, requires_grad=True)
    
    valid_log_probs = [lp for lp, _ in valid_pairs]
    valid_rewards = torch.stack([r for _, r in valid_pairs])
    
    # Baseline (reduces variance)
    baseline = valid_rewards.mean()
    
    policy_loss = torch.tensor(0.0, device=rewards.device, dtype=rewards.dtype)
    total_entropy = torch.tensor(0.0, device=rewards.device, dtype=rewards.dtype)
    total_decisions = 0
    
    for log_p, reward in zip(valid_log_probs, valid_rewards):
        # Skip if NaN
        if torch.isnan(log_p).any():
            logger.warning("[PolicyLoss] log_p contains NaN. Skipping rollout.")
            continue
        
        # Compute and clamp advantage
        advantage = (reward - baseline).clamp(-advantage_clamp, advantage_clamp)
        
        # REINFORCE gradient (negative because we minimize)
        policy_loss = policy_loss - (log_p * advantage).sum()
        
        # Entropy bonus (for exploration)
        p = torch.exp(log_p).clamp(PROB_CLAMP_MIN, PROB_CLAMP_MAX)
        log_p_safe = torch.log(p).clamp(min=LOG_CLAMP_MIN)
        log_1_minus_p_safe = torch.log(1 - p).clamp(min=LOG_CLAMP_MIN)
        entropy = -(p * log_p_safe + (1 - p) * log_1_minus_p_safe)
        entropy = _check_nan(entropy, "entropy", default=0.0)
        
        total_entropy = total_entropy + entropy.sum()
        total_decisions += log_p.numel()
    
    # Apply entropy bonus
    if total_decisions > 0:
        policy_loss = policy_loss - beta_entropy * total_entropy
    
    # Final NaN check
    if torch.isnan(policy_loss):
        logger.warning("[PolicyLoss] Final loss is NaN. Returning zero.")
        return torch.tensor(0.0, device=rewards.device, requires_grad=True)
    
    return policy_loss


# Pruning gate penalties (unchanged)

def prune_l1(prune_soft: torch.Tensor, *, reduction: Reduction = "mean") -> torch.Tensor:
    """L1 penalty on keep/expand probabilities."""
    q = _as_B(prune_soft).abs()
    return _reduce(q, reduction)


def prune_kl_to_rate(
    prune_soft: torch.Tensor,
    prior_keep_rate: float = 0.5,
    *,
    reduction: Reduction = "mean",
    eps: float = 1e-8,
) -> torch.Tensor:
    """KL divergence from Bernoulli(q) to Bernoulli(r)."""
    q = _as_B(prune_soft).clamp(eps, 1 - eps)
    r = float(prior_keep_rate)
    if not (0.0 < r < 1.0):
        raise ValueError("prior_keep_rate must be in (0,1)")
    r_t = torch.tensor(r, dtype=q.dtype, device=q.device)
    kl = q * (q.log() - r_t.log()) + (1 - q) * ((1 - q).log() - (1 - r_t).log())
    return _reduce(kl, reduction)


# Trace-based regularizers (unchanged)

def expected_children_from_trace(trace: Dict[str, torch.Tensor], batch_size: int) -> torch.Tensor:
    if trace is None or "spawn_counts_sum" not in trace:
        return torch.zeros(0)
    sp = trace["spawn_counts_sum"].detach().float()
    B = max(1, int(batch_size))
    return sp / float(B)


def expected_nodes_from_trace(trace: Dict[str, torch.Tensor], batch_size: int) -> torch.Tensor:
    avg_children = expected_children_from_trace(trace, batch_size)
    if avg_children.numel() == 0:
        return torch.tensor(1.0)
    return 1.0 + avg_children.sum()


def budget_children_mse_from_trace(
    trace: Dict[str, torch.Tensor],
    batch_size: int,
    target_children: float | Sequence[float],
    *,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    avg_children = expected_children_from_trace(trace, batch_size)
    if avg_children.numel() == 0:
        return _zeros_like_device_of()
    if isinstance(target_children, (list, tuple)):
        tgt = torch.tensor(target_children, dtype=avg_children.dtype, device=avg_children.device)
    else:
        tgt = torch.full_like(avg_children, float(target_children))
    diff2 = (avg_children - tgt) ** 2
    return _reduce(diff2, reduction)


def budget_nodes_mse_from_trace(
    trace: Dict[str, torch.Tensor],
    batch_size: int,
    target_total_nodes: float,
    *,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    total_nodes = expected_nodes_from_trace(trace, batch_size)
    diff2 = (total_nodes - float(target_total_nodes)) ** 2
    return _reduce(diff2, reduction)


def depth_kl_to_prior_from_trace(
    trace: Dict[str, torch.Tensor],
    batch_size: int,
    prior_depth: Optional[torch.Tensor | Sequence[float]] = None,
    *,
    reduction: Reduction = "mean",
    eps: float = 1e-8,
) -> torch.Tensor:
    avg_children = expected_children_from_trace(trace, batch_size)
    if avg_children.numel() == 0:
        return _zeros_like_device_of()
    nodes_per_depth = torch.cat([torch.ones_like(avg_children[:1]), avg_children], dim=0)
    p = nodes_per_depth / nodes_per_depth.sum().clamp_min(eps)
    if prior_depth is None:
        q = torch.full_like(p, 1.0 / p.numel())
    else:
        if isinstance(prior_depth, (list, tuple)):
            q = torch.tensor(prior_depth, dtype=p.dtype, device=p.device)
        else:
            q = prior_depth.to(dtype=p.dtype, device=p.device)
        q = q / q.sum().clamp_min(eps)
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    kl = (p * (p.log() - q.log())).sum()
    return _reduce(kl, reduction)


def compute_bfs_regularizers(
    prune_soft: Optional[torch.Tensor] = None,
    *,
    prior_depth: Optional[torch.Tensor | Sequence[float]] = None,
    prior_keep_rate: Optional[float] = None,
    target_children: Optional[float] = None,
    target_children_per_depth: Optional[Sequence[float] | torch.Tensor] = None,
    target_total_nodes: Optional[float] = None,
    trace: Optional[Dict[str, torch.Tensor]] = None,
    batch_size: Optional[int] = None,
    weights: Optional[Dict[str, float]] = None,
    reduction: Reduction = "mean",
) -> Dict[str, torch.Tensor]:
    """Compute weighted sum of BFS auxiliary losses."""
    w = {"prune_l1": 0.0, "prune_kl": 0.0, "budget_children_mse": 0.0, 
         "budget_nodes_mse": 0.0, "depth_kl": 0.0}
    if weights:
        for k, v in weights.items():
            if k in w:
                w[k] = float(v)

    z = _zeros_like_device_of(prune_soft)
    terms: Dict[str, torch.Tensor] = {k: z for k in w}

    if prune_soft is not None:
        if w["prune_l1"] != 0.0:
            terms["prune_l1"] = prune_l1(prune_soft, reduction=reduction)
        if w["prune_kl"] != 0.0 and prior_keep_rate is not None:
            terms["prune_kl"] = prune_kl_to_rate(prune_soft, prior_keep_rate, reduction=reduction)

    have_trace = trace is not None and "spawn_counts_sum" in trace and batch_size is not None
    if have_trace:
        if w["budget_children_mse"] != 0.0 and (target_children_per_depth is not None or target_children is not None):
            tgt = target_children_per_depth if target_children_per_depth is not None else float(target_children)
            terms["budget_children_mse"] = budget_children_mse_from_trace(trace, int(batch_size), tgt, reduction=reduction)
        if w["budget_nodes_mse"] != 0.0 and target_total_nodes is not None:
            terms["budget_nodes_mse"] = budget_nodes_mse_from_trace(trace, int(batch_size), float(target_total_nodes), reduction=reduction)
        if w["depth_kl"] != 0.0:
            terms["depth_kl"] = depth_kl_to_prior_from_trace(trace, int(batch_size), prior_depth=prior_depth, reduction=reduction)

    total = sum(w[k] * terms[k] for k in w)
    terms["total"] = total
    return terms


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    torch.manual_seed(42)
    
    logger.info("=" * 50)
    logger.info("losses.py v1.1.0 Self-Test")
    logger.info("=" * 50)

    B, seq_len, vocab_size = 4, 32, 256
    num_rollouts = 3
    
    # Test compute_rewards_language
    outputs = [torch.randn(B * seq_len, vocab_size) for _ in range(num_rollouts)]
    labels = torch.randint(0, vocab_size, (B, seq_len))
    node_counts = [1024, 896, 1152]
    
    rewards = compute_rewards_language(outputs, node_counts, labels, 0.05, 13, B, seq_len)
    assert rewards.shape == (num_rollouts,)
    assert not torch.isnan(rewards).any()
    logger.info(f"✓ compute_rewards_language OK: {rewards.tolist()}")
    
    # Test policy_loss_reinforce
    log_probs = [torch.randn(100, requires_grad=True) for _ in range(num_rollouts)]
    loss = policy_loss_reinforce(log_probs, rewards, 0.01)
    assert loss.requires_grad
    loss.backward()
    logger.info(f"✓ policy_loss_reinforce OK: {loss.item():.4f}")
    
    # Test compute_perplexity
    ppl = compute_perplexity(2.0)
    assert abs(ppl - math.exp(2.0)) < 1e-6
    logger.info(f"✓ compute_perplexity OK: {ppl:.4f}")
    
    logger.info("=" * 50)
    logger.info("All tests passed!")
    logger.info("=" * 50)