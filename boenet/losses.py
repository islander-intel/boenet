#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
boenet/losses.py (v1.0.0 - Language Model)

Loss functions for BoeNet language models with REINFORCE policy gradients.

Converted from BFSNet v2.0.0 (Vision) to BoeNet v1.0.0 (Language)
-----------------------------------------------------------------
The key change is in the reward computation:

BFSNet (Vision - Classification):
    reward = accuracy - λ * efficiency_penalty
    accuracy = (preds == labels).float().mean()  # 0 or 1

BoeNet (Language - Next Token Prediction):
    reward = -cross_entropy_loss - λ * efficiency_penalty
    We use NEGATIVE loss because lower loss = better = higher reward

Everything else remains identical - the REINFORCE algorithm is task-agnostic.

What this module provides
-------------------------
A) REINFORCE Policy Gradient Losses (v1.0.0 - UPDATED):
   - compute_rewards(outputs, node_counts, labels, λ, max_nodes, batch_size, seq_len)
   - policy_loss_reinforce(log_probs, rewards, β_entropy)

B) Pruning gate helpers (unchanged from BFSNet):
   - prune_l1(z, reduction="mean")
   - prune_kl_to_rate(z, prior_keep_rate=0.5, reduction="mean")

C) Budget & structure-aware regularizers (unchanged from BFSNet):
   - expected_children_from_trace(trace, batch_size) → Tensor[D]
   - expected_nodes_from_trace(trace, batch_size) → scalar
   - budget_children_mse_from_trace(...)
   - budget_nodes_mse_from_trace(...)
   - depth_kl_to_prior_from_trace(...)

D) Convenience combiner (unchanged from BFSNet):
   - compute_bfs_regularizers(...)

E) Language Model Utilities (NEW):
   - compute_perplexity(cross_entropy_loss) → float

Reward Design for Language Models
----------------------------------
In classification (BFSNet), accuracy is bounded [0, 1], making reward scaling easy.

In language modeling, cross-entropy loss is unbounded and varies by task:
  - Untrained model: CE ≈ log(vocab_size) ≈ 5.5 for char-level (vocab=256)
  - Trained model: CE ≈ 1.5-2.5 for good char-level models
  - Perplexity = exp(CE), so CE=2.0 → PPL=7.4

Our reward uses NEGATIVE cross-entropy:
  - reward = -CE - λ * efficiency
  - Higher reward = lower loss = better model
  - Typical reward range: [-5.5, -1.5] for char-level models

The λ efficiency penalty should be scaled appropriately:
  - λ=0.05 with max efficiency=1.0 gives max penalty of 0.05
  - This is small relative to CE range [1.5, 5.5]
  - Consider λ=0.1-0.5 for stronger efficiency pressure in language models

Usage Examples
--------------
>>> # Compute rewards for language model rollouts
>>> outputs = [torch.randn(128, 256) for _ in range(3)]  # 3 rollouts, [B*seq, vocab]
>>> labels = torch.randint(0, 256, (4, 32))  # [B, seq_len]
>>> node_counts = [1024, 896, 1152]  # Total nodes per rollout
>>> 
>>> rewards = compute_rewards_language(
...     outputs, node_counts, labels,
...     lambda_efficiency=0.05,
...     max_nodes_per_position=13,  # 1 + 3 + 9 for depth=2, K=3
...     batch_size=4,
...     seq_len=32
... )
>>> rewards.shape
torch.Size([3])

Author: BoeNet project (converted from BFSNet)
Version: 1.0.0
Date: 2025-12-22
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Literal, Sequence

import torch
import torch.nn.functional as F


Reduction = Literal["mean", "sum", "none"]


# --------------------------------------------------------------------------- #
#                         Shape / utility helpers                             #
# --------------------------------------------------------------------------- #

def _reduce(x: torch.Tensor, reduction: Reduction) -> torch.Tensor:
    """Apply reduction to tensor."""
    if reduction == "mean":
        return x.mean()
    if reduction == "sum":
        return x.sum()
    if reduction == "none":
        return x
    raise ValueError(f"Unsupported reduction: {reduction!r}")


def _as_B(x: torch.Tensor) -> torch.Tensor:
    """Squeeze trailing singleton dims down to [B] if input is [B] or [B,1]."""
    if x.dim() == 2 and x.size(1) == 1:
        return x.squeeze(1)
    return x


def _zeros_like_device_of(*tensors: Optional[torch.Tensor]) -> torch.Tensor:
    """
    Return a detached scalar zero on the first available device among tensors,
    or CPU if all are None.
    """
    for t in tensors:
        if isinstance(t, torch.Tensor):
            return torch.zeros((), device=t.device, dtype=t.dtype)
    return torch.tensor(0.0)


# --------------------------------------------------------------------------- #
#                    Language Model Utilities (NEW)                           #
# --------------------------------------------------------------------------- #

def compute_perplexity(cross_entropy_loss: torch.Tensor | float) -> float:
    """
    Compute perplexity from cross-entropy loss.
    
    Perplexity = exp(cross_entropy_loss)
    
    This is the standard metric for language models:
      - Lower perplexity = better model
      - Perplexity of 1.0 = perfect prediction
      - Random prediction on vocab_size V → PPL = V
    
    Parameters
    ----------
    cross_entropy_loss : torch.Tensor or float
        Cross-entropy loss value.
        
    Returns
    -------
    float
        Perplexity value.
        
    Examples
    --------
    >>> ce_loss = 2.0
    >>> ppl = compute_perplexity(ce_loss)
    >>> ppl  # ≈ 7.39
    7.3890...
    
    >>> # Random prediction on vocab_size=256
    >>> ce_random = math.log(256)  # ≈ 5.545
    >>> compute_perplexity(ce_random)  # ≈ 256.0
    256.0
    """
    if isinstance(cross_entropy_loss, torch.Tensor):
        cross_entropy_loss = cross_entropy_loss.item()
    return math.exp(cross_entropy_loss)


# --------------------------------------------------------------------------- #
#          REINFORCE Policy Gradient Losses (v1.0.0 - Language Model)         #
# --------------------------------------------------------------------------- #

def compute_rewards_language(
    outputs: List[torch.Tensor],
    node_counts: List[int],
    labels: torch.Tensor,
    lambda_efficiency: float,
    max_nodes_per_position: int,
    batch_size: int,
    seq_len: int,
) -> torch.Tensor:
    """
    Compute per-rollout rewards for REINFORCE (Language Model version).
    
    Key difference from BFSNet (classification):
    ---------------------------------------------
    BFSNet:  reward = accuracy - λ * efficiency_penalty
             accuracy ∈ [0, 1]
    
    BoeNet:  reward = -cross_entropy - λ * efficiency_penalty
             -cross_entropy ∈ [-∞, 0] (typically [-5.5, -1.5] for char-level)
    
    We use NEGATIVE cross-entropy because:
      - Lower CE loss = better model
      - REINFORCE maximizes expected reward
      - So higher reward should mean better model
    
    Parameters
    ----------
    outputs : List[torch.Tensor]
        List of output logits from each rollout.
        Each tensor has shape [B * seq_len, vocab_size].
    node_counts : List[int]
        List of TOTAL node counts from each rollout.
        Each count is summed over all positions in the batch.
    labels : torch.Tensor
        Ground truth next tokens, shape [B, seq_len].
    lambda_efficiency : float
        Efficiency penalty coefficient (typically 0.01-0.1).
        For language models, consider higher values (0.1-0.5) since
        the loss range is larger than classification accuracy.
    max_nodes_per_position : int
        Maximum possible nodes per token position.
        Formula: 1 + K + K^2 + ... + K^D for max_children=K, max_depth=D.
    batch_size : int
        Original batch size B.
    seq_len : int
        Sequence length.
        
    Returns
    -------
    torch.Tensor
        Rewards tensor, shape [num_rollouts].
        
    Notes
    -----
    Reward structure:
        reward = -CE_loss - λ * (nodes_used / max_nodes)
    
    Where:
        - CE_loss: Cross-entropy loss (mean over all positions)
        - nodes_used: Actual nodes created (normalized per position)
        - max_nodes: Theoretical maximum nodes per position
    
    The efficiency penalty is computed per position to be comparable:
        efficiency = (total_nodes / num_positions) / max_nodes_per_position
    
    Examples
    --------
    >>> # 3 rollouts with batch_size=4, seq_len=32, vocab_size=256
    >>> outputs = [torch.randn(128, 256) for _ in range(3)]
    >>> labels = torch.randint(0, 256, (4, 32))
    >>> node_counts = [1024, 896, 1152]  # Different sparsity per rollout
    >>> 
    >>> rewards = compute_rewards_language(
    ...     outputs, node_counts, labels,
    ...     lambda_efficiency=0.05,
    ...     max_nodes_per_position=13,
    ...     batch_size=4,
    ...     seq_len=32
    ... )
    >>> print(rewards)  # Typical: tensor([-2.5, -2.3, -2.7])
    """
    num_positions = batch_size * seq_len
    
    # Flatten labels for cross-entropy: [B, seq_len] → [B * seq_len]
    labels_flat = labels.view(-1)
    
    rewards = []
    
    for rollout_idx, (out, nodes) in enumerate(zip(outputs, node_counts)):
        # out shape: [B * seq_len, vocab_size]
        # labels_flat shape: [B * seq_len]
        
        # Cross-entropy loss (mean over all positions)
        ce_loss = F.cross_entropy(out, labels_flat, reduction='mean')
        
        # Reward component: NEGATIVE loss
        # Typical range: [-5.5, -1.5] for char-level
        reward_quality = -ce_loss
        
        # Efficiency penalty (normalized per position)
        nodes_per_position = nodes / num_positions
        efficiency_ratio = nodes_per_position / float(max_nodes_per_position)
        efficiency_penalty = lambda_efficiency * efficiency_ratio
        
        # Combined reward
        reward = reward_quality - efficiency_penalty
        rewards.append(reward)
    
    return torch.stack(rewards)  # [num_rollouts]


def compute_rewards_classification(
    outputs: List[torch.Tensor],
    node_counts: List[int],
    labels: torch.Tensor,
    lambda_efficiency: float,
    max_nodes: int,
) -> torch.Tensor:
    """
    Compute per-rollout rewards for REINFORCE (Classification version).
    
    This is the ORIGINAL BFSNet reward function, kept for backwards compatibility
    and for classification tasks (e.g., sentiment analysis on text).
    
    Reward structure:
        reward = accuracy - λ * (nodes_used / max_nodes)
    
    Parameters
    ----------
    outputs : List[torch.Tensor]
        List of output logits from each rollout, each shape [B, num_classes].
    node_counts : List[int]
        List of node counts from each rollout.
    labels : torch.Tensor
        Ground truth class labels, shape [B].
    lambda_efficiency : float
        Efficiency penalty coefficient (typically 0.01-0.1).
    max_nodes : int
        Maximum possible nodes per example.
        
    Returns
    -------
    torch.Tensor
        Rewards tensor, shape [num_rollouts].
    """
    batch_size = outputs[0].size(0) if outputs else 1
    rewards = []
    
    for out, nodes in zip(outputs, node_counts):
        # Accuracy: fraction of correct predictions
        preds = out.argmax(dim=-1)  # [B]
        correct = (preds == labels).float()  # [B]
        accuracy = correct.mean()  # scalar in [0, 1]
        
        # Efficiency penalty (normalized per example)
        nodes_per_example = nodes / batch_size
        efficiency_penalty = lambda_efficiency * (nodes_per_example / float(max_nodes))
        
        # Combined reward
        reward = accuracy - efficiency_penalty
        rewards.append(reward)
    
    return torch.stack(rewards)


def policy_loss_reinforce(
    log_probs: List[torch.Tensor],
    rewards: torch.Tensor,
    beta_entropy: float = 0.01,
) -> torch.Tensor:
    """
    REINFORCE policy loss with baseline and entropy bonus.
    
    This function is IDENTICAL for both language models and classification -
    the REINFORCE algorithm doesn't care about the task, only about
    log probabilities and rewards.
    
    Loss structure:
        policy_loss = -Σ(log_probs * (reward - baseline)) - β * entropy
    
    Where:
        - log_probs: Log probabilities of sampled actions (grow/stop decisions)
        - reward: Per-rollout rewards from compute_rewards_*
        - baseline: Mean reward (reduces variance)
        - entropy: Encourages exploration (-p log p - (1-p) log(1-p))
        - β (beta_entropy): Entropy bonus coefficient (0.001-0.01)
    
    Parameters
    ----------
    log_probs : List[torch.Tensor]
        List of log probability tensors from each rollout.
        Each tensor contains log probs for all grow/stop decisions.
    rewards : torch.Tensor
        Rewards for each rollout, shape [num_rollouts].
    beta_entropy : float, default=0.01
        Entropy bonus coefficient (higher = more exploration).
        
    Returns
    -------
    torch.Tensor
        Scalar policy loss (minimizing this maximizes expected reward).
        
    Notes
    -----
    The REINFORCE gradient estimator:
        ∇J(θ) ≈ Σ_t ∇log π(a_t|s_t;θ) * (R - b)
    
    Where:
        - π(a_t|s_t;θ): Policy probability of action a_t in state s_t
        - R: Total reward for the trajectory
        - b: Baseline (typically mean reward, reduces variance)
    
    The entropy bonus prevents premature convergence:
        H(π) = -Σ_a π(a) log π(a)
    
    For Bernoulli (grow/stop):
        H = -p log p - (1-p) log(1-p)
    
    Examples
    --------
    >>> # Mock log probs from 3 rollouts
    >>> log_probs = [torch.randn(100, requires_grad=True) for _ in range(3)]
    >>> rewards = torch.tensor([-2.5, -2.3, -2.7])  # Language model rewards
    >>> 
    >>> loss = policy_loss_reinforce(log_probs, rewards, beta_entropy=0.01)
    >>> loss.backward()  # Gradients flow to policy parameters
    """
    if len(log_probs) == 0:
        # No rollouts - return zero loss with gradient
        return torch.tensor(0.0, requires_grad=True)
    
    # Baseline: mean reward across rollouts (reduces variance without adding bias)
    baseline = rewards.mean()
    
    policy_loss = 0.0
    total_entropy = 0.0
    
    for log_p_rollout, reward in zip(log_probs, rewards):
        # Advantage: how much better/worse than average
        advantage = reward - baseline
        
        # REINFORCE gradient: -log_prob * advantage
        # Negative because we minimize loss (equivalent to maximizing reward)
        policy_loss = policy_loss - (log_p_rollout * advantage).sum()
        
        # Entropy bonus (from Bernoulli log probs)
        # For Bernoulli with log_prob = log(p) when action=1, log(1-p) when action=0:
        # We recover p = exp(log_p) for action=1 decisions
        # Entropy = -p*log(p) - (1-p)*log(1-p)
        p = torch.exp(log_p_rollout)
        p = p.clamp(1e-8, 1 - 1e-8)  # Numerical stability
        entropy = -(p * p.log() + (1 - p) * (1 - p).log())
        total_entropy = total_entropy + entropy.sum()
    
    # Average entropy and apply bonus
    # Subtract because higher entropy = more exploration = better
    num_rollouts = len(log_probs)
    avg_entropy = total_entropy / max(num_rollouts, 1)
    policy_loss = policy_loss - beta_entropy * avg_entropy
    
    return policy_loss


# --------------------------------------------------------------------------- #
#                           Pruning gate penalties                             #
# --------------------------------------------------------------------------- #

def prune_l1(
    prune_soft: torch.Tensor,
    *,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """
    L1 penalty on the (soft) keep/expand probabilities.
    
    Encourages sparse expansion by penalizing high keep probabilities.

    Parameters
    ----------
    prune_soft : Tensor [B] or [B,1]
        Values in [0,1]; interpreted as "keep" probability.
    reduction : {"mean","sum","none"}
        Reduction method.

    Returns
    -------
    l1 : Tensor scalar or [B]
        L1 penalty value.
    """
    q = _as_B(prune_soft).abs()  # [B]
    return _reduce(q, reduction)


def prune_kl_to_rate(
    prune_soft: torch.Tensor,
    prior_keep_rate: float = 0.5,
    *,
    reduction: Reduction = "mean",
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    KL divergence from Bernoulli(q) to Bernoulli(r) per example.
    
    Encourages the learned keep probability to match a target rate.

    Parameters
    ----------
    prune_soft : Tensor [B] or [B,1]
        Keep probabilities (q) in [0,1].
    prior_keep_rate : float in (0,1)
        Desired keep rate (r).
    reduction : {"mean","sum","none"}
        Reduction method.
    eps : float
        Small constant for numerical stability.

    Returns
    -------
    kl : Tensor scalar or [B]
        KL(Bern(q) || Bern(r)) = q log(q/r) + (1-q) log((1-q)/(1-r))
    """
    q = _as_B(prune_soft).clamp(eps, 1 - eps)  # [B]
    r = float(prior_keep_rate)
    if not (0.0 < r < 1.0):
        raise ValueError("prior_keep_rate must be in (0,1)")
    r_t = torch.tensor(r, dtype=q.dtype, device=q.device)
    kl = q * (q.log() - r_t.log()) + (1 - q) * ((1 - q).log() - (1 - r_t).log())
    return _reduce(kl, reduction)


# --------------------------------------------------------------------------- #
#                    Trace-based budget / depth regularizers                  #
# --------------------------------------------------------------------------- #

def expected_children_from_trace(trace: Dict[str, torch.Tensor], batch_size: int) -> torch.Tensor:
    """
    Compute per-depth average children per example from the trainer/model trace.

    Required trace keys:
      - "spawn_counts_sum": 1D Tensor[<= max_depth], sum over batch of spawned 
        children at each expansion depth.

    Parameters
    ----------
    trace : Dict[str, torch.Tensor]
        Trace dictionary containing "spawn_counts_sum".
    batch_size : int
        Number of examples in the batch.

    Returns
    -------
    avg_children_per_example : Tensor [D]
        Each entry is (spawn_counts_sum[d] / batch_size).
    """
    if trace is None or "spawn_counts_sum" not in trace:
        return torch.zeros(0)
    sp = trace["spawn_counts_sum"].detach().float()  # [D]
    B = max(1, int(batch_size))
    return sp / float(B)


def expected_nodes_from_trace(trace: Dict[str, torch.Tensor], batch_size: int) -> torch.Tensor:
    """
    Compute expected total nodes per example using the (hard) counts in the trace.
    
    Formula: total_nodes ≈ 1 (root) + Σ_d avg_children_per_example[d]

    Parameters
    ----------
    trace : Dict[str, torch.Tensor]
        Trace dictionary containing "spawn_counts_sum".
    batch_size : int
        Number of examples in the batch.

    Returns
    -------
    total_nodes_mean : Tensor scalar
        Expected total nodes per example.
    """
    avg_children = expected_children_from_trace(trace, batch_size)  # [D]
    if avg_children.numel() == 0:
        # Fallback: at least the root
        return torch.tensor(1.0)
    return 1.0 + avg_children.sum()


def budget_children_mse_from_trace(
    trace: Dict[str, torch.Tensor],
    batch_size: int,
    target_children: float | Sequence[float],
    *,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """
    Penalize per-depth average children per example against a target.

    Parameters
    ----------
    trace : Dict[str, torch.Tensor]
        Trace dictionary containing "spawn_counts_sum".
    batch_size : int
        Number of examples in the batch.
    target_children : float or Sequence[float]
        Target children count per depth. If scalar, same target for all depths.
        If sequence, must match number of depths.
    reduction : {"mean","sum","none"}
        Reduction method.

    Returns
    -------
    mse : Tensor scalar or [D]
        Mean squared error from target.
    """
    avg_children = expected_children_from_trace(trace, batch_size)  # [D]
    if avg_children.numel() == 0:
        return _zeros_like_device_of()  # zero on CPU

    if isinstance(target_children, (list, tuple)):
        tgt = torch.tensor(target_children, dtype=avg_children.dtype, device=avg_children.device)
        if tgt.numel() != avg_children.numel():
            raise ValueError("target_children length must match number of depths in trace['spawn_counts_sum']")
    else:
        tgt = torch.full_like(avg_children, float(target_children))
    diff2 = (avg_children - tgt) ** 2  # [D]
    return _reduce(diff2, reduction)


def budget_nodes_mse_from_trace(
    trace: Dict[str, torch.Tensor],
    batch_size: int,
    target_total_nodes: float,
    *,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """
    Penalize deviation of expected total nodes per example from a scalar target.

    Parameters
    ----------
    trace : Dict[str, torch.Tensor]
        Trace dictionary containing "spawn_counts_sum".
    batch_size : int
        Number of examples in the batch.
    target_total_nodes : float
        Target for total nodes (1 + Σ_d avg_children[d]).
    reduction : {"mean","sum","none"}
        Reduction method.

    Returns
    -------
    mse : Tensor scalar
        Mean squared error from target.
    """
    total_nodes = expected_nodes_from_trace(trace, batch_size)  # scalar
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
    """
    KL between empirical depth distribution (nodes per depth) and a prior.

    Empirical depth distribution p(depth) is built from expected nodes:
      - depth 0 (root): 1.0
      - depth d>=1: avg_children_per_example[d-1]
    
    These are normalized to sum to 1, then KL(p || q) is computed.

    Parameters
    ----------
    trace : Dict[str, torch.Tensor]
        Trace dictionary containing "spawn_counts_sum".
    batch_size : int
        Number of examples in the batch.
    prior_depth : Optional[Tensor or Sequence], length [D+1]
        Prior distribution over depths 0..D. If None, uniform is used.
    reduction : {"mean","sum","none"}
        Reduction method.
    eps : float
        Small constant for numerical stability.

    Returns
    -------
    kl : Tensor scalar
        KL divergence from prior.
    """
    avg_children = expected_children_from_trace(trace, batch_size)  # [D]
    if avg_children.numel() == 0:
        return _zeros_like_device_of()  # zero

    # Nodes per depth: [1, avg_children[0], avg_children[1], ...]
    nodes_per_depth = torch.cat([torch.ones_like(avg_children[:1]), avg_children], dim=0)  # [D+1]
    p = nodes_per_depth / nodes_per_depth.sum().clamp_min(eps)

    if prior_depth is None:
        q = torch.full_like(p, 1.0 / p.numel())
    else:
        if isinstance(prior_depth, (list, tuple)):
            q = torch.tensor(prior_depth, dtype=p.dtype, device=p.device)
        else:
            q = prior_depth.to(dtype=p.dtype, device=p.device)
        if q.numel() != p.numel():
            raise ValueError("prior_depth length must match number of depths (len(spawn_counts_sum)+1)")
        q = q / q.sum().clamp_min(eps)

    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    kl = (p * (p.log() - q.log())).sum()  # scalar
    return _reduce(kl, reduction)


# --------------------------------------------------------------------------- #
#                           Combined convenience API                          #
# --------------------------------------------------------------------------- #

def compute_bfs_regularizers(
    prune_soft: Optional[torch.Tensor] = None,
    *,
    # Priors / targets
    prior_depth: Optional[torch.Tensor | Sequence[float]] = None,
    prior_keep_rate: Optional[float] = None,
    target_children: Optional[float] = None,
    target_children_per_depth: Optional[Sequence[float] | torch.Tensor] = None,
    target_total_nodes: Optional[float] = None,
    # Trace for budget/depth terms
    trace: Optional[Dict[str, torch.Tensor]] = None,
    batch_size: Optional[int] = None,
    # Weights
    weights: Optional[Dict[str, float]] = None,
    reduction: Reduction = "mean",
) -> Dict[str, torch.Tensor]:
    """
    Compute a weighted sum of BFS auxiliary losses.
    
    This function combines multiple regularization terms into a single
    dictionary for easy integration into training loops.
    
    Supported weight keys (missing keys default to 0.0):
      - "prune_l1": L1 penalty on pruning gates
      - "prune_kl": KL divergence to target keep rate
      - "budget_children_mse": MSE from target children per depth
      - "budget_nodes_mse": MSE from target total nodes
      - "depth_kl": KL divergence of depth distribution from prior

    Parameters
    ----------
    prune_soft : Optional[torch.Tensor]
        Pruning gate soft values.
    prior_depth : Optional[Tensor or Sequence]
        Prior distribution over depths.
    prior_keep_rate : Optional[float]
        Target keep rate for KL regularization.
    target_children : Optional[float]
        Target children per depth (scalar).
    target_children_per_depth : Optional[Sequence or Tensor]
        Target children per depth (per-depth values).
    target_total_nodes : Optional[float]
        Target total nodes per example.
    trace : Optional[Dict[str, torch.Tensor]]
        Model trace dictionary.
    batch_size : Optional[int]
        Batch size for trace normalization.
    weights : Optional[Dict[str, float]]
        Dictionary mapping loss names to weights.
    reduction : {"mean","sum","none"}
        Reduction method.

    Returns
    -------
    Dict[str, torch.Tensor]
        Dictionary containing each term and "total" (weighted sum).
    """
    # Default weights
    w = {
        "prune_l1": 0.0,
        "prune_kl": 0.0,
        "budget_children_mse": 0.0,
        "budget_nodes_mse": 0.0,
        "depth_kl": 0.0,
    }
    if weights:
        for k, v in weights.items():
            if k in w:
                w[k] = float(v)

    z = _zeros_like_device_of(prune_soft)

    terms: Dict[str, torch.Tensor] = {
        "prune_l1": z,
        "prune_kl": z,
        "budget_children_mse": z,
        "budget_nodes_mse": z,
        "depth_kl": z,
    }

    # Pruning terms
    if prune_soft is not None:
        if w["prune_l1"] != 0.0:
            terms["prune_l1"] = prune_l1(prune_soft, reduction=reduction)
        if w["prune_kl"] != 0.0 and prior_keep_rate is not None:
            terms["prune_kl"] = prune_kl_to_rate(
                prune_soft, prior_keep_rate=float(prior_keep_rate), reduction=reduction
            )

    # Trace-based budget/depth terms
    have_trace = (
        trace is not None 
        and isinstance(trace, dict) 
        and "spawn_counts_sum" in trace 
        and batch_size is not None
    )
    if have_trace:
        if w["budget_children_mse"] != 0.0 and (target_children_per_depth is not None or target_children is not None):
            tgt = target_children_per_depth if target_children_per_depth is not None else float(target_children)
            terms["budget_children_mse"] = budget_children_mse_from_trace(
                trace, int(batch_size), tgt, reduction=reduction
            )
        if w["budget_nodes_mse"] != 0.0 and target_total_nodes is not None:
            terms["budget_nodes_mse"] = budget_nodes_mse_from_trace(
                trace, int(batch_size), float(target_total_nodes), reduction=reduction
            )
        if w["depth_kl"] != 0.0:
            terms["depth_kl"] = depth_kl_to_prior_from_trace(
                trace, int(batch_size), prior_depth=prior_depth, reduction=reduction
            )

    # Weighted total
    total = (
        w["prune_l1"] * terms["prune_l1"]
        + w["prune_kl"] * terms["prune_kl"]
        + w["budget_children_mse"] * terms["budget_children_mse"]
        + w["budget_nodes_mse"] * terms["budget_nodes_mse"]
        + w["depth_kl"] * terms["depth_kl"]
    )

    terms["total"] = total
    return terms


# --------------------------------------------------------------------------- #
#                                  Self-test                                  #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import logging
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)
    
    torch.manual_seed(42)
    
    logger.info("=" * 60)
    logger.info("BoeNet Losses v1.0.0 Self-Test Suite")
    logger.info("=" * 60)
    
    # Test configuration
    B = 4           # Batch size
    seq_len = 32    # Sequence length
    vocab_size = 256  # Character-level
    num_rollouts = 3
    
    # Test 1: compute_rewards_language
    logger.info("\n[Test 1] compute_rewards_language")
    outputs = [torch.randn(B * seq_len, vocab_size) for _ in range(num_rollouts)]
    labels = torch.randint(0, vocab_size, (B, seq_len))
    node_counts = [1024, 896, 1152]  # Different node counts per rollout
    
    rewards = compute_rewards_language(
        outputs=outputs,
        node_counts=node_counts,
        labels=labels,
        lambda_efficiency=0.05,
        max_nodes_per_position=13,  # depth=2, K=3
        batch_size=B,
        seq_len=seq_len,
    )
    
    assert rewards.shape == (num_rollouts,), f"Expected ({num_rollouts},), got {rewards.shape}"
    logger.info(f"  Rewards: {[f'{r:.4f}' for r in rewards.tolist()]}")
    logger.info(f"  Reward range: [{rewards.min():.4f}, {rewards.max():.4f}]")
    logger.info("  ✓ compute_rewards_language OK")
    
    # Test 2: compute_rewards_classification (backwards compatibility)
    logger.info("\n[Test 2] compute_rewards_classification")
    outputs_cls = [torch.randn(B, 10) for _ in range(num_rollouts)]
    labels_cls = torch.randint(0, 10, (B,))
    node_counts_cls = [48, 36, 52]
    
    rewards_cls = compute_rewards_classification(
        outputs=outputs_cls,
        node_counts=node_counts_cls,
        labels=labels_cls,
        lambda_efficiency=0.05,
        max_nodes=13,
    )
    
    assert rewards_cls.shape == (num_rollouts,), f"Expected ({num_rollouts},), got {rewards_cls.shape}"
    logger.info(f"  Rewards: {[f'{r:.4f}' for r in rewards_cls.tolist()]}")
    logger.info("  ✓ compute_rewards_classification OK")
    
    # Test 3: policy_loss_reinforce
    logger.info("\n[Test 3] policy_loss_reinforce")
    log_probs = [torch.randn(100, requires_grad=True) for _ in range(num_rollouts)]
    
    policy_loss = policy_loss_reinforce(
        log_probs=log_probs,
        rewards=rewards,
        beta_entropy=0.01,
    )
    
    assert policy_loss.requires_grad, "Policy loss should require grad"
    policy_loss.backward()
    
    # Check gradients flowed
    for i, lp in enumerate(log_probs):
        assert lp.grad is not None, f"Log prob {i} should have gradient"
        assert lp.grad.abs().sum() > 0, f"Log prob {i} gradient should be non-zero"
    
    logger.info(f"  Policy loss: {policy_loss.item():.6f}")
    logger.info("  ✓ policy_loss_reinforce OK")
    
    # Test 4: compute_perplexity
    logger.info("\n[Test 4] compute_perplexity")
    ce_loss = 2.0
    ppl = compute_perplexity(ce_loss)
    expected_ppl = math.exp(2.0)
    assert abs(ppl - expected_ppl) < 1e-6, f"Expected {expected_ppl}, got {ppl}"
    logger.info(f"  CE loss: {ce_loss} → Perplexity: {ppl:.4f}")
    
    # Random baseline
    ce_random = math.log(vocab_size)
    ppl_random = compute_perplexity(ce_random)
    logger.info(f"  Random baseline (vocab={vocab_size}): CE={ce_random:.4f} → PPL={ppl_random:.4f}")
    logger.info("  ✓ compute_perplexity OK")
    
    # Test 5: Pruning losses
    logger.info("\n[Test 5] Pruning losses")
    prune_soft = torch.rand(B)
    
    l1_loss = prune_l1(prune_soft)
    kl_loss = prune_kl_to_rate(prune_soft, prior_keep_rate=0.5)
    
    logger.info(f"  prune_l1: {l1_loss.item():.6f}")
    logger.info(f"  prune_kl_to_rate: {kl_loss.item():.6f}")
    logger.info("  ✓ Pruning losses OK")
    
    # Test 6: compute_bfs_regularizers
    logger.info("\n[Test 6] compute_bfs_regularizers")
    trace = {
        "spawn_counts_sum": torch.tensor([12.0, 36.0]),  # Depth 1, Depth 2
    }
    
    regs = compute_bfs_regularizers(
        prune_soft=prune_soft,
        trace=trace,
        batch_size=B,
        prior_keep_rate=0.5,
        target_total_nodes=5.0,
        weights={
            "prune_l1": 0.1,
            "prune_kl": 0.05,
            "budget_nodes_mse": 0.01,
        },
    )
    
    logger.info(f"  Regularizer terms:")
    for key, val in regs.items():
        logger.info(f"    {key}: {val.item():.6f}")
    logger.info("  ✓ compute_bfs_regularizers OK")
    
    logger.info("\n" + "=" * 60)
    logger.info("All self-tests passed!")
    logger.info("=" * 60)