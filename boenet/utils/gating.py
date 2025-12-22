#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utils/gating.py

Modular gating components for BFS-inspired neural networks (v2.0.0).

This module provides composable gates for BFS-expanding neural networks:
  1) Binary growth decisions: GrowthPolicyNet for REINFORCE-based routing
  2) Binary pruning: ScalarGate with straight-through estimation
  3) Magnitude-based pruning: ThresholdPruner (deterministic)
  4) L0 sparsity: HardConcreteGate

v2.0.0 Changes
--------------
- REMOVED: BranchingGate (Gumbel-Softmax categorical branching)
- REMOVED: sample_gumbel, gumbel_softmax_st (no longer needed)
- ADDED: GrowthPolicyNet (binary grow/stop decision per child with depth encoding)

Usage (v2.0.0)
--------------
>>> from utils.gating import GrowthPolicyNet, ScalarGate, ThresholdPruner
>>> 
>>> # Binary growth policy for REINFORCE
>>> policy = GrowthPolicyNet(hidden_dim=64, max_depth=3)
>>> depth_idx = 1  # Current expansion depth
>>> grow_prob = policy(node_hidden, depth_idx)  # [N, 1] probability
>>> 
>>> # Pruning gates (unchanged from v1.4.0)
>>> prune_gate = ScalarGate(in_dim=64, bias=True)
>>> p_soft, z_hard = prune_gate(h)
>>> 
>>> pruner = ThresholdPruner(mode="l2", threshold=0.1)
>>> keep_mask = pruner(h)

Author: William McKeon
Date: 2025-08-19
Updated: 2025-12-18 (v2.0.0 - REINFORCE policy gradients)
"""

from __future__ import annotations
from typing import Tuple, Optional
import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = [
    "GrowthPolicyNet",
    "ScalarGate",
    "HardConcreteGate",
    "ThresholdPruner",
]

# Module-level logger
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#                    GrowthPolicyNet (v2.0.0 - NEW)
# ---------------------------------------------------------------------------

class GrowthPolicyNet(nn.Module):
    """
    Binary grow/stop policy for REINFORCE-based dynamic branching (v2.0.0).
    
    Predicts a probability of growing child j at depth d based on parent
    node hidden state and depth encoding. Used with REINFORCE to learn
    sparse routing through the computation tree.
    
    Architecture
    ------------
    Input: [node_hidden (H), depth_onehot (D)] → concat → [H+D]
    Layers:
      - Linear(H+D, H//2)
      - ReLU
      - Linear(H//2, 1)
      - Sigmoid → probability in (0, 1)
    
    Parameters
    ----------
    hidden_dim : int
        Hidden dimension of node states.
    max_depth : int
        Maximum BFS expansion depth (for depth encoding).
    
    Attributes
    ----------
    hidden_dim : int
        Node hidden dimension.
    max_depth : int
        Maximum depth.
    policy : nn.Sequential
        Policy network: [H+D] → [1].
    
    Notes
    -----
    The depth encoding is a one-hot vector of length max_depth, where
    depth_onehot[d] = 1 and all other entries are 0. This allows the
    policy to learn depth-dependent routing strategies.
    
    During training, decisions are sampled stochastically (Bernoulli).
    During inference, decisions are made greedily (threshold at 0.5).
    
    Examples
    --------
    >>> policy = GrowthPolicyNet(hidden_dim=64, max_depth=3)
    >>> h = torch.randn(32, 64)  # Parent nodes
    >>> depth_idx = 1  # Current depth
    >>> 
    >>> # Get grow probability
    >>> grow_prob = policy(h, depth_idx)
    >>> grow_prob.shape
    torch.Size([32, 1])
    >>> 
    >>> # Sample action (training)
    >>> action = torch.bernoulli(grow_prob)  # {0, 1}
    >>> 
    >>> # Greedy action (inference)
    >>> action = (grow_prob >= 0.5).float()
    """
    
    def __init__(self, hidden_dim: int, max_depth: int):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.max_depth = int(max_depth)
        
        # Policy network: [H + max_depth] → [1]
        input_dim = self.hidden_dim + self.max_depth
        mid_dim = max(self.hidden_dim // 2, 16)  # At least 16 hidden units
        
        self.policy = nn.Sequential(
            nn.Linear(input_dim, mid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mid_dim, 1),
            nn.Sigmoid()
        )
        
        # Initialize weights
        for m in self.policy:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    # Small negative bias = prefer not growing (sparse prior)
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    nn.init.uniform_(m.bias, -bound, -bound * 0.5)
    
    def forward(self, h: torch.Tensor, depth_idx: int) -> torch.Tensor:
        """
        Compute grow probability for nodes at given depth.
        
        Parameters
        ----------
        h : torch.Tensor
            Node hidden states, shape [N, hidden_dim].
        depth_idx : int
            Current expansion depth (0-indexed).
            
        Returns
        -------
        torch.Tensor
            Grow probabilities, shape [N, 1], values in (0, 1).
            
        Examples
        --------
        >>> policy = GrowthPolicyNet(64, max_depth=3)
        >>> h = torch.randn(10, 64)
        >>> p = policy(h, depth_idx=1)
        >>> p.shape
        torch.Size([10, 1])
        >>> (p > 0).all() and (p < 1).all()
        True
        """
        N = h.size(0)
        device = h.device
        
        # Create depth one-hot encoding
        depth_onehot = torch.zeros(N, self.max_depth, device=device, dtype=h.dtype)
        if 0 <= depth_idx < self.max_depth:
            depth_onehot[:, depth_idx] = 1.0
        
        # Concatenate node hidden + depth encoding
        policy_input = torch.cat([h, depth_onehot], dim=-1)  # [N, H+D]
        
        # Compute grow probability
        grow_prob = self.policy(policy_input)  # [N, 1]
        
        return grow_prob


# ---------------------------------------------------------------------------
#                           Scalar (Binary) Gate
# ---------------------------------------------------------------------------

class ScalarGate(nn.Module):
    """
    Learnable binary gate with straight-through rounding.

    Computes p = sigmoid(Wx + b), returns:
      - p_soft ∈ [0, 1] (for logging/regularization)
      - z_hard ∈ {0, 1} via straight-through step for discrete decisions.
    
    This is useful as an expand/stop gate per node in BFS expansion.

    Parameters
    ----------
    in_dim : int
        Input feature dimension.
    bias : bool, default=True
        Include bias term in linear transform.
    threshold : float, default=0.5
        Threshold in [0, 1] used to discretize p_soft into z_hard.

    Attributes
    ----------
    fc : nn.Linear
        Linear projection from in_dim to 1.
    threshold : float
        Decision threshold for binarization.

    Notes
    -----
    Uses a straight-through estimator: forward uses hard z in {0, 1};
    backward uses gradients of p_soft. This enables gradient flow through
    discrete decisions.
    
    Examples
    --------
    >>> gate = ScalarGate(in_dim=64, bias=True, threshold=0.5)
    >>> h = torch.randn(32, 64)
    >>> p_soft, z_hard = gate(h)
    >>> p_soft.shape
    torch.Size([32, 1])
    >>> z_hard.unique()
    tensor([0., 1.])
    """

    def __init__(self, in_dim: int, bias: bool = True, threshold: float = 0.5):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1, bias=bias)
        self.threshold = float(threshold)
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.fc.weight, a=1.0)
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute binary gate decision.

        Parameters
        ----------
        x : torch.Tensor
            Input features, shape [B, in_dim].

        Returns
        -------
        tuple of (torch.Tensor, torch.Tensor)
            p_soft : [B, 1] soft probability in [0, 1]
            z_hard : [B, 1] straight-through binary in {0, 1}
        """
        p = torch.sigmoid(self.fc(x))  # [B, 1]
        z_hard = (p >= self.threshold).float()
        # Straight-through: hard in forward, soft in backward
        z = z_hard.detach() - p.detach() + p
        return p, z


# ---------------------------------------------------------------------------
#                       Hard-Concrete (L0) Binary Gate
# ---------------------------------------------------------------------------

class HardConcreteGate(nn.Module):
    """
    Hard-Concrete gate for L0 regularization.
    
    Implements the Hard-Concrete distribution from Louizos et al., 
    "Learning Sparse Neural Networks through L_0 Regularization" (ICLR 2018).

    Produces approximately binary gates z ∈ {0, 1} with a differentiable
    relaxation, enabling L0-style sparsity penalties. Useful when you want
    *learned* pruning with a continuous surrogate during training.

    Parameters
    ----------
    in_dim : int
        Input feature dimension.
    beta : float, default=2.0
        Temperature for the stretched sigmoid (higher = smoother transitions).
    gamma : float, default=-0.1
        Left boundary for the stretched interval.
    zeta : float, default=1.1
        Right boundary for the stretched interval.

    Attributes
    ----------
    fc : nn.Linear
        Linear projection from in_dim to 1.
    beta : float
        Temperature parameter.
    gamma : float
        Left stretch boundary.
    zeta : float
        Right stretch boundary.

    Notes
    -----
    During training, samples are drawn from a stretched binary concrete
    distribution and hard-clipped to [0, 1]. At evaluation, the deterministic
    mean is used instead.
    
    The expected L0 penalty can be computed as:
        L0 = sigmoid(log_alpha - beta * log(-gamma / zeta))
    where log_alpha are the learned logits.
    
    References
    ----------
    Louizos et al., "Learning Sparse Neural Networks through L_0 Regularization",
    ICLR 2018.
    
    Examples
    --------
    >>> gate = HardConcreteGate(in_dim=64, beta=2.0)
    >>> h = torch.randn(32, 64)
    >>> p_open, z = gate(h, training=True)
    >>> z.shape
    torch.Size([32, 1])
    """

    def __init__(
        self,
        in_dim: int,
        beta: float = 2.0,
        gamma: float = -0.1,
        zeta: float = 1.1,
    ):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1, bias=True)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.zeta = float(zeta)
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.fc.weight, a=1.0)
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)

    def forward(
        self,
        x: torch.Tensor,
        training: Optional[bool] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Hard-Concrete gate values.
        
        Parameters
        ----------
        x : torch.Tensor
            Input features, shape [B, in_dim].
        training : bool, optional
            If None, uses self.training. If True, samples stochastically;
            if False, uses deterministic mean.

        Returns
        -------
        tuple of (torch.Tensor, torch.Tensor)
            p_open : [B, 1] expected gate openness probability (for logging)
            z : [B, 1] gate value in [0, 1] (near-binary; clip to {0, 1} at eval)
        """
        if training is None:
            training = self.training

        logits = self.fc(x)  # [B, 1]

        if training:
            # Sample u ~ Uniform(0, 1) then s = sigmoid((log u - log(1-u) + logits) / beta)
            u = torch.rand_like(logits)
            u = u.clamp(1e-8, 1 - 1e-8)  # Numerical stability
            s = torch.sigmoid((torch.log(u) - torch.log(1 - u) + logits) / self.beta)
        else:
            # Deterministic: use sigmoid(logits / beta) (mean of Concrete)
            s = torch.sigmoid(logits / self.beta)

        # Stretch to (gamma, zeta), then hard clip to [0, 1]
        s_bar = s * (self.zeta - self.gamma) + self.gamma
        z = torch.clamp(s_bar, 0.0, 1.0)

        # Probability of being "on" (rough heuristic for logging)
        p_open = torch.sigmoid(logits)
        return p_open, z


# ---------------------------------------------------------------------------
#                   Deterministic Threshold-Based Pruning
# ---------------------------------------------------------------------------

class ThresholdPruner(nn.Module):
    """
    Simple deterministic pruner based on activation magnitude.
    
    Returns a boolean keep-mask per node based on whether the activation
    magnitude exceeds a threshold. No learnable parameters.

    Parameters
    ----------
    mode : {"l2", "l1", "mean_abs", "max_abs"}, default="l2"
        Metric used to score an activation vector:
        - "l2": Euclidean norm ||h||_2
        - "l1": Manhattan norm ||h||_1
        - "mean_abs": Mean absolute value mean(|h|)
        - "max_abs": Maximum absolute value max(|h|)
    threshold : float, default=1e-3
        Keep a node if score >= threshold; otherwise prune it.

    Attributes
    ----------
    mode : str
        The scoring mode.
    threshold : torch.Tensor
        Registered buffer containing the threshold value.

    Examples
    --------
    >>> pruner = ThresholdPruner(mode="l2", threshold=1.5)
    >>> h = torch.randn(32, 64)
    >>> keep_mask = pruner(h)
    >>> keep_mask.dtype
    torch.bool
    >>> keep_mask.shape
    torch.Size([32])
    """

    def __init__(self, mode: str = "l2", threshold: float = 1e-3):
        super().__init__()
        valid_modes = {"l2", "l1", "mean_abs", "max_abs"}
        if mode not in valid_modes:
            raise ValueError(f"mode must be one of {valid_modes}, got {mode!r}")
        self.mode = mode
        self.register_buffer("threshold", torch.tensor(float(threshold)))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Compute keep mask based on activation magnitude.

        Parameters
        ----------
        h : torch.Tensor
            Hidden states, shape [B, H].

        Returns
        -------
        torch.Tensor
            Boolean mask, shape [B], where True means keep the node.
        """
        if self.mode == "l2":
            score = torch.linalg.vector_norm(h, ord=2, dim=-1)      # [B]
        elif self.mode == "l1":
            score = torch.linalg.vector_norm(h, ord=1, dim=-1)      # [B]
        elif self.mode == "mean_abs":
            score = h.abs().mean(dim=-1)                            # [B]
        else:  # "max_abs"
            score = h.abs().max(dim=-1).values                      # [B]
        return score >= self.threshold


# ---------------------------------------------------------------------------
#                              Minimal Self-Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Configure logging for self-test
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    torch.manual_seed(0)
    B, H = 4, 8
    x = torch.randn(B, H)

    logger.info("=" * 60)
    logger.info("utils/gating.py Self-Test Suite (v2.0.0)")
    logger.info("=" * 60)

    # Test 1: GrowthPolicyNet
    logger.info("\n[Test 1] GrowthPolicyNet")
    policy = GrowthPolicyNet(hidden_dim=H, max_depth=3)
    for depth_idx in range(3):
        grow_prob = policy(x, depth_idx)
        logger.info(f"  depth={depth_idx}: grow_prob shape={grow_prob.shape}, "
                   f"mean={grow_prob.mean().item():.4f}, "
                   f"in (0,1)={(grow_prob > 0).all() and (grow_prob < 1).all()}")
        assert grow_prob.shape == (B, 1), f"grow_prob shape mismatch at depth {depth_idx}"
        assert (grow_prob > 0).all() and (grow_prob < 1).all(), "grow_prob not in (0, 1)"
    logger.info("  ✓ GrowthPolicyNet OK")

    # Test 2: ScalarGate
    logger.info("\n[Test 2] ScalarGate")
    sg = ScalarGate(in_dim=H, bias=True, threshold=0.5)
    p_soft, z = sg(x)
    logger.info(f"  p_soft: {p_soft.squeeze(-1).tolist()}")
    logger.info(f"  z_hard: {z.squeeze(-1).tolist()}")
    assert p_soft.shape == (B, 1), "p_soft shape mismatch"
    assert z.shape == (B, 1), "z shape mismatch"
    logger.info("  ✓ ScalarGate OK")

    # Test 3: HardConcreteGate
    logger.info("\n[Test 3] HardConcreteGate")
    hcg = HardConcreteGate(in_dim=H, beta=2.0)
    p_open, z_cont = hcg(x, training=True)
    logger.info(f"  p_open: {p_open.squeeze(-1).tolist()}")
    logger.info(f"  z_cont (mean): {z_cont.mean().item():.4f}")
    assert p_open.shape == (B, 1), "p_open shape mismatch"
    assert z_cont.shape == (B, 1), "z_cont shape mismatch"
    logger.info("  ✓ HardConcreteGate OK")

    # Test 4: ThresholdPruner
    logger.info("\n[Test 4] ThresholdPruner")
    pruner = ThresholdPruner(mode="l2", threshold=1.5)
    keep = pruner(x)
    logger.info(f"  keep: {keep.tolist()}")
    assert keep.shape == (B,), "keep shape mismatch"
    assert keep.dtype == torch.bool, "keep dtype mismatch"
    logger.info("  ✓ ThresholdPruner OK")

    # Test 5: Gradient flow through GrowthPolicyNet
    logger.info("\n[Test 5] Gradient flow through GrowthPolicyNet")
    policy_grad = GrowthPolicyNet(hidden_dim=H, max_depth=3)
    x_grad = torch.randn(B, H, requires_grad=True)
    grow_prob = policy_grad(x_grad, depth_idx=1)
    loss = grow_prob.sum()
    loss.backward()
    assert x_grad.grad is not None, "Gradients should flow through policy"
    logger.info(f"  input gradient norm: {x_grad.grad.norm().item():.6f}")
    logger.info("  ✓ Gradient flow OK")

    logger.info("\n" + "=" * 60)
    logger.info("All self-tests passed!")
    logger.info("=" * 60)