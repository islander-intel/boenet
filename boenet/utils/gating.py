#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utils/gating.py (v2.1.0)

Modular gating components for BFS-inspired neural networks.

v2.1.0 Fixes: Added logit/probability clamping to prevent CUDA assertion failures.

Components:
  1) GrowthPolicyNet: Binary grow/stop policy for REINFORCE
  2) ScalarGate: Binary pruning with straight-through estimation
  3) ThresholdPruner: Magnitude-based pruning (deterministic)
  4) HardConcreteGate: L0 sparsity gate

Author: William McKeon
Updated: 2025-12-30 (v2.1.0 - Numerical stability fixes)
"""

from __future__ import annotations
from typing import Tuple, Optional
import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["GrowthPolicyNet", "ScalarGate", "HardConcreteGate", "ThresholdPruner"]

logger = logging.getLogger(__name__)

# Numerical stability constants
LOGIT_CLAMP_MIN = -20.0
LOGIT_CLAMP_MAX = 20.0
PROB_CLAMP_MIN = 1e-7
PROB_CLAMP_MAX = 1.0 - 1e-7


class GrowthPolicyNet(nn.Module):
    """
    Binary grow/stop policy for REINFORCE-based dynamic branching (v2.1.0).
    
    v2.1.0 FIXES:
    - Added logit clamping before sigmoid to prevent overflow
    - Added probability clamping after sigmoid as safety net
    - Added NaN detection and handling
    
    Parameters
    ----------
    hidden_dim : int
        Hidden dimension of node states.
    max_depth : int
        Maximum BFS expansion depth (for depth encoding).
    """
    
    def __init__(self, hidden_dim: int, max_depth: int):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.max_depth = int(max_depth)
        
        input_dim = self.hidden_dim + self.max_depth
        mid_dim = max(self.hidden_dim // 2, 16)
        
        self.policy_fc1 = nn.Linear(input_dim, mid_dim)
        self.policy_fc2 = nn.Linear(mid_dim, 1)
        
        # Improved initialization for stability
        nn.init.kaiming_uniform_(self.policy_fc1.weight, a=math.sqrt(5))
        if self.policy_fc1.bias is not None:
            nn.init.constant_(self.policy_fc1.bias, -0.1)
        
        nn.init.xavier_uniform_(self.policy_fc2.weight, gain=0.5)
        if self.policy_fc2.bias is not None:
            nn.init.constant_(self.policy_fc2.bias, -0.2)
        
        # For backward compatibility with code that accesses self.policy[0].weight
        self.policy = nn.Sequential(
            self.policy_fc1,
            nn.ReLU(inplace=True),
            self.policy_fc2,
        )
        
        self._debug = False
    
    def forward(self, h: torch.Tensor, depth_idx: int) -> torch.Tensor:
        """
        Compute grow probability for nodes at given depth.
        
        Returns probabilities GUARANTEED to be in (PROB_CLAMP_MIN, PROB_CLAMP_MAX).
        
        Parameters
        ----------
        h : torch.Tensor
            Node hidden states, shape [N, hidden_dim].
        depth_idx : int
            Current expansion depth (0-indexed).
            
        Returns
        -------
        torch.Tensor
            Grow probabilities, shape [N, 1].
        """
        N = h.size(0)
        device = h.device
        dtype = h.dtype
        
        # Check input for NaN
        if torch.isnan(h).any():
            nan_count = torch.isnan(h).sum().item()
            logger.warning(f"[GrowthPolicyNet] Input has {nan_count} NaN values. Replacing with zeros.")
            h = torch.where(torch.isnan(h), torch.zeros_like(h), h)
        
        # Create depth one-hot encoding
        depth_onehot = torch.zeros(N, self.max_depth, device=device, dtype=dtype)
        if 0 <= depth_idx < self.max_depth:
            depth_onehot[:, depth_idx] = 1.0
        
        # Concatenate and compute
        policy_input = torch.cat([h, depth_onehot], dim=-1)
        hidden = F.relu(self.policy_fc1(policy_input))
        logits = self.policy_fc2(hidden)
        
        # CRITICAL FIX: Clamp logits before sigmoid
        logits_clamped = logits.clamp(LOGIT_CLAMP_MIN, LOGIT_CLAMP_MAX)
        
        # Apply sigmoid
        grow_prob = torch.sigmoid(logits_clamped)
        
        # CRITICAL FIX: Final probability clamp (safety net)
        grow_prob = grow_prob.clamp(PROB_CLAMP_MIN, PROB_CLAMP_MAX)
        
        # Final NaN check
        if torch.isnan(grow_prob).any():
            logger.error("[GrowthPolicyNet] Output has NaN AFTER clamping. Replacing with 0.5.")
            grow_prob = torch.where(torch.isnan(grow_prob), torch.full_like(grow_prob, 0.5), grow_prob)
        
        return grow_prob


class ScalarGate(nn.Module):
    """
    Learnable binary gate with straight-through rounding.
    
    Parameters
    ----------
    in_dim : int
        Input feature dimension.
    bias : bool, default=True
        Include bias term.
    threshold : float, default=0.5
        Threshold for binarization.
    """

    def __init__(self, in_dim: int, bias: bool = True, threshold: float = 0.5):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1, bias=bias)
        self.threshold = float(threshold)
        nn.init.kaiming_uniform_(self.fc.weight, a=1.0)
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (p_soft, z_hard) where z_hard uses straight-through."""
        logits = self.fc(x).clamp(LOGIT_CLAMP_MIN, LOGIT_CLAMP_MAX)
        p = torch.sigmoid(logits).clamp(PROB_CLAMP_MIN, PROB_CLAMP_MAX)
        z_hard = (p >= self.threshold).float()
        z = z_hard.detach() - p.detach() + p
        return p, z


class HardConcreteGate(nn.Module):
    """Hard-Concrete gate for L0 regularization."""

    def __init__(self, in_dim: int, beta: float = 2.0, gamma: float = -0.1, zeta: float = 1.1):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1, bias=True)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.zeta = float(zeta)
        nn.init.kaiming_uniform_(self.fc.weight, a=1.0)
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor, training: Optional[bool] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if training is None:
            training = self.training
        logits = self.fc(x).clamp(LOGIT_CLAMP_MIN, LOGIT_CLAMP_MAX)
        if training:
            u = torch.rand_like(logits).clamp(1e-8, 1 - 1e-8)
            s = torch.sigmoid((torch.log(u) - torch.log(1 - u) + logits) / self.beta)
        else:
            s = torch.sigmoid(logits / self.beta)
        s_bar = s * (self.zeta - self.gamma) + self.gamma
        z = torch.clamp(s_bar, 0.0, 1.0)
        p_open = torch.sigmoid(logits).clamp(PROB_CLAMP_MIN, PROB_CLAMP_MAX)
        return p_open, z


class ThresholdPruner(nn.Module):
    """Deterministic pruner based on activation magnitude."""

    def __init__(self, mode: str = "l2", threshold: float = 1e-3):
        super().__init__()
        valid_modes = {"l2", "l1", "mean_abs", "max_abs"}
        if mode not in valid_modes:
            raise ValueError(f"mode must be one of {valid_modes}")
        self.mode = mode
        self.register_buffer("threshold", torch.tensor(float(threshold)))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if self.mode == "l2":
            score = torch.linalg.vector_norm(h, ord=2, dim=-1)
        elif self.mode == "l1":
            score = torch.linalg.vector_norm(h, ord=1, dim=-1)
        elif self.mode == "mean_abs":
            score = h.abs().mean(dim=-1)
        else:
            score = h.abs().max(dim=-1).values
        return score >= self.threshold


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    torch.manual_seed(0)
    B, H = 4, 8
    x = torch.randn(B, H)

    logger.info("=" * 50)
    logger.info("gating.py v2.1.0 Self-Test")
    logger.info("=" * 50)

    # Test GrowthPolicyNet
    policy = GrowthPolicyNet(hidden_dim=H, max_depth=3)
    for d in range(3):
        p = policy(x, d)
        assert p.shape == (B, 1)
        assert (p >= PROB_CLAMP_MIN).all() and (p <= PROB_CLAMP_MAX).all()
    logger.info("✓ GrowthPolicyNet OK")

    # Test with extreme inputs
    x_large = torch.randn(B, H) * 1000
    p_large = policy(x_large, 0)
    assert not torch.isnan(p_large).any()
    logger.info("✓ GrowthPolicyNet extreme inputs OK")

    # Test ScalarGate
    sg = ScalarGate(in_dim=H)
    p_soft, z = sg(x)
    assert p_soft.shape == (B, 1) and z.shape == (B, 1)
    logger.info("✓ ScalarGate OK")

    # Test HardConcreteGate
    hcg = HardConcreteGate(in_dim=H)
    p_open, z_cont = hcg(x, training=True)
    assert p_open.shape == (B, 1)
    logger.info("✓ HardConcreteGate OK")

    # Test ThresholdPruner
    pruner = ThresholdPruner(mode="l2", threshold=1.5)
    keep = pruner(x)
    assert keep.shape == (B,) and keep.dtype == torch.bool
    logger.info("✓ ThresholdPruner OK")

    logger.info("=" * 50)
    logger.info("All tests passed!")
    logger.info("=" * 50)