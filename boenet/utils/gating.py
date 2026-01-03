#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
boenet/utils/gating.py (v3.1.0 - True BFS with Fixed GrowthPolicyNet)

Modular gating components for BFS-inspired neural networks.

v3.1.0 Changes (2026-01-03) - GrowthPolicyNet Fix
-------------------------------------------------
ISSUE FIXED:
  GrowthPolicyNet was outputting grow_prob=0.0000 for all inputs, causing
  trees to never expand except when epsilon-greedy forced it.

ROOT CAUSE:
  The model.py version had LayerNorm + ReLU architecture that killed all
  activations, producing extremely negative logits before sigmoid.

FIXES APPLIED:
  1. Simplified architecture: hidden -> concat depth_embed -> MLP -> sigmoid
  2. Output shape changed from [N, 1] to [N] to match model.py expectations
  3. Proper initialization to produce ~0.5 probability initially
  4. Added diagnostic printing option
  5. Reduced depth embedding dimension to hidden_dim // 4 for efficiency

v3.0.0 Changes (True BFS Support):
  - GrowthPolicyNet now takes LEVEL-AGGREGATED input (not per-node)
  - Single decision per level for balanced tree expansion
  - Preserved numerical stability from v2.1.0

v2.1.0 Features (Preserved):
  - Logit/probability clamping to prevent CUDA assertion failures
  - NaN detection and handling

Components:
  1) GrowthPolicyNet: Binary grow/stop policy for REINFORCE (LEVEL-BASED)
  2) ScalarGate: Binary pruning with straight-through estimation
  3) ThresholdPruner: Magnitude-based pruning (deterministic)
  4) HardConcreteGate: L0 sparsity gate

Author: BoeNet Project
Version: 3.1.0
Date: 2026-01-03
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

# =============================================================================
# NUMERICAL STABILITY CONSTANTS
# =============================================================================
LOGIT_CLAMP_MIN = -20.0
LOGIT_CLAMP_MAX = 20.0
PROB_CLAMP_MIN = 1e-7
PROB_CLAMP_MAX = 1.0 - 1e-7


# =============================================================================
# GROWTH POLICY NETWORK (v3.1.0 - Fixed Architecture)
# =============================================================================

class GrowthPolicyNet(nn.Module):
    """
    Binary grow/stop policy for REINFORCE-based dynamic branching (v3.1.0).
    
    v3.1.0 FIXES:
    - Simplified architecture that actually produces valid probabilities
    - Output shape is [N] not [N, 1] to match model.py expectations
    - Proper initialization for ~0.5 initial probability
    - No LayerNorm (was killing activations in previous version)
    - Smaller depth embedding (hidden_dim // 4) for efficiency
    
    v3.0.0 FEATURES (Preserved):
    - Takes LEVEL-AGGREGATED hidden states as input
    - Makes ONE decision for entire level (not per node)
    - This enables balanced tree expansion
    
    v2.1.0 Features (Preserved):
    - Logit clamping before sigmoid to prevent overflow
    - Probability clamping after sigmoid as safety net
    - NaN detection and handling
    
    Parameters
    ----------
    hidden_dim : int
        Hidden dimension of node states.
    max_depth : int
        Maximum BFS expansion depth (for depth encoding).
        
    Input
    -----
    h : torch.Tensor
        Level-aggregated hidden states, shape [N, hidden_dim].
        This is the MEAN of all node hidden states at the current level.
        N = batch_size * seq_len for language modeling.
    depth_idx : int
        Current expansion depth (0-indexed).
        
    Output
    ------
    grow_prob : torch.Tensor
        Grow probability for the entire level, shape [N].
        In True BFS, this is averaged across N to get a single level decision.
        Guaranteed to be in (PROB_CLAMP_MIN, PROB_CLAMP_MAX).
    """
    
    def __init__(self, hidden_dim: int, max_depth: int):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.max_depth = int(max_depth)
        
        # v3.1.0: Smaller depth embedding (hidden_dim // 4) for efficiency
        # This reduces the influence of depth on the decision
        self.depth_embed_dim = max(self.hidden_dim // 4, 16)
        self.depth_embed = nn.Embedding(max(self.max_depth + 1, 1), self.depth_embed_dim)
        nn.init.normal_(self.depth_embed.weight, mean=0.0, std=0.02)
        
        # Policy MLP: [hidden + depth_embed] -> [mid] -> [1]
        policy_input_dim = self.hidden_dim + self.depth_embed_dim
        mid_dim = max(self.hidden_dim // 2, 32)
        
        self.policy_fc1 = nn.Linear(policy_input_dim, mid_dim)
        self.policy_fc2 = nn.Linear(mid_dim, 1)
        
        # v3.1.0: Proper initialization for stable training
        # Kaiming for hidden layer (works well with ReLU)
        nn.init.kaiming_uniform_(self.policy_fc1.weight, a=math.sqrt(5))
        if self.policy_fc1.bias is not None:
            # Initialize bias to small positive value to help gradients flow
            nn.init.constant_(self.policy_fc1.bias, 0.01)
        
        # v3.1.0: Initialize output layer to produce ~0.5 probability
        # sigmoid(0) = 0.5, so we want the output to be near 0 initially
        nn.init.xavier_uniform_(self.policy_fc2.weight, gain=0.1)
        if self.policy_fc2.bias is not None:
            nn.init.constant_(self.policy_fc2.bias, 0.0)  # sigmoid(0) = 0.5
        
        # Debug flag for diagnostic output
        self._debug = False
    
    def forward(self, h: torch.Tensor, depth_idx: int) -> torch.Tensor:
        """
        Compute grow probability for a level given aggregated hidden state.
        
        This is used for TRUE BFS where we make ONE decision per level.
        The input `h` should be the mean of all node hidden states at the level.
        
        Parameters
        ----------
        h : torch.Tensor
            Level-aggregated hidden states, shape [N, hidden_dim].
        depth_idx : int
            Current expansion depth (0-indexed).
            
        Returns
        -------
        torch.Tensor
            Grow probabilities, shape [N].
            Guaranteed to be in (PROB_CLAMP_MIN, PROB_CLAMP_MAX).
        """
        N = h.size(0)
        device = h.device
        
        # Check input for NaN
        if torch.isnan(h).any():
            nan_count = torch.isnan(h).sum().item()
            logger.warning(
                f"[GrowthPolicyNet] Input has {nan_count} NaN values. "
                f"Replacing with zeros."
            )
            h = torch.where(torch.isnan(h), torch.zeros_like(h), h)
        
        # Get depth embedding
        # Clamp depth_idx to valid range
        safe_depth = min(max(depth_idx, 0), self.max_depth)
        depth_indices = torch.full((N,), safe_depth, device=device, dtype=torch.long)
        depth_emb = self.depth_embed(depth_indices)  # [N, depth_embed_dim]
        
        # Concatenate hidden state with depth embedding
        policy_input = torch.cat([h, depth_emb], dim=-1)  # [N, hidden_dim + depth_embed_dim]
        
        # Forward through policy MLP
        # v3.1.0: Simple ReLU activation, no LayerNorm
        hidden = F.relu(self.policy_fc1(policy_input))
        logits = self.policy_fc2(hidden)  # [N, 1]
        
        # CRITICAL: Clamp logits before sigmoid to prevent overflow
        logits_clamped = logits.clamp(LOGIT_CLAMP_MIN, LOGIT_CLAMP_MAX)
        
        # Apply sigmoid to get probability
        grow_prob = torch.sigmoid(logits_clamped)
        
        # CRITICAL: Final probability clamp (safety net)
        grow_prob = grow_prob.clamp(PROB_CLAMP_MIN, PROB_CLAMP_MAX)
        
        # v3.1.0: Squeeze to return shape [N] instead of [N, 1]
        grow_prob = grow_prob.squeeze(-1)
        
        # Final NaN check
        if torch.isnan(grow_prob).any():
            logger.error(
                "[GrowthPolicyNet] Output has NaN AFTER clamping. "
                "Replacing with 0.5."
            )
            grow_prob = torch.where(
                torch.isnan(grow_prob),
                torch.full_like(grow_prob, 0.5),
                grow_prob
            )
        
        if self._debug:
            print(f"[GrowthPolicy] depth={depth_idx}, "
                  f"grow_prob: mean={grow_prob.mean().item():.4f}, "
                  f"min={grow_prob.min().item():.4f}, "
                  f"max={grow_prob.max().item():.4f}")
        
        return grow_prob


# =============================================================================
# SCALAR GATE (Unchanged from v2.1.0)
# =============================================================================

class ScalarGate(nn.Module):
    """
    Learnable binary gate with straight-through rounding.
    
    Used for pruning decisions. Outputs both soft probability and
    hard binary decision (with gradient passed through via straight-through).
    
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
        """
        Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Input features, shape [..., in_dim].
            
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            (p_soft, z_hard) where:
            - p_soft: Soft probability, shape [..., 1]
            - z_hard: Hard binary decision with straight-through gradient
        """
        logits = self.fc(x).clamp(LOGIT_CLAMP_MIN, LOGIT_CLAMP_MAX)
        p = torch.sigmoid(logits).clamp(PROB_CLAMP_MIN, PROB_CLAMP_MAX)
        z_hard = (p >= self.threshold).float()
        # Straight-through estimator
        z = z_hard.detach() - p.detach() + p
        return p, z


# =============================================================================
# HARD CONCRETE GATE (Unchanged from v2.1.0)
# =============================================================================

class HardConcreteGate(nn.Module):
    """
    Hard-Concrete gate for L0 regularization.
    
    Implements the hard concrete distribution for differentiable L0 sparsity.
    During training, samples from the stretched distribution.
    During inference, uses the mode.
    
    Parameters
    ----------
    in_dim : int
        Input feature dimension.
    beta : float, default=2.0
        Temperature for the concrete distribution.
    gamma : float, default=-0.1
        Lower bound of the stretched distribution.
    zeta : float, default=1.1
        Upper bound of the stretched distribution.
    """

    def __init__(
        self,
        in_dim: int,
        beta: float = 2.0,
        gamma: float = -0.1,
        zeta: float = 1.1
    ):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1, bias=True)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.zeta = float(zeta)
        nn.init.kaiming_uniform_(self.fc.weight, a=1.0)
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)

    def forward(
        self,
        x: torch.Tensor,
        training: Optional[bool] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Input features.
        training : Optional[bool]
            Override training mode. If None, uses self.training.
            
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            (p_open, z) where:
            - p_open: Probability of gate being open
            - z: Gate value in [0, 1]
        """
        if training is None:
            training = self.training
            
        logits = self.fc(x).clamp(LOGIT_CLAMP_MIN, LOGIT_CLAMP_MAX)
        
        if training:
            # Sample from uniform and apply inverse CDF
            u = torch.rand_like(logits).clamp(1e-8, 1 - 1e-8)
            s = torch.sigmoid(
                (torch.log(u) - torch.log(1 - u) + logits) / self.beta
            )
        else:
            # Use mode during inference
            s = torch.sigmoid(logits / self.beta)
        
        # Stretch to [gamma, zeta] and clamp to [0, 1]
        s_bar = s * (self.zeta - self.gamma) + self.gamma
        z = torch.clamp(s_bar, 0.0, 1.0)
        
        # Probability of gate being open (before stretching)
        p_open = torch.sigmoid(logits).clamp(PROB_CLAMP_MIN, PROB_CLAMP_MAX)
        
        return p_open, z


# =============================================================================
# THRESHOLD PRUNER (Unchanged from v2.1.0)
# =============================================================================

class ThresholdPruner(nn.Module):
    """
    Deterministic pruner based on activation magnitude.
    
    Prunes nodes whose activation magnitude is below a threshold.
    No learnable parameters - purely based on magnitude.
    
    Parameters
    ----------
    mode : str, default="l2"
        Norm type for computing magnitude. One of:
        - "l2": L2 norm
        - "l1": L1 norm
        - "mean_abs": Mean absolute value
        - "max_abs": Maximum absolute value
    threshold : float, default=1e-3
        Pruning threshold. Nodes with magnitude below this are pruned.
    """

    def __init__(self, mode: str = "l2", threshold: float = 1e-3):
        super().__init__()
        valid_modes = {"l2", "l1", "mean_abs", "max_abs"}
        if mode not in valid_modes:
            raise ValueError(f"mode must be one of {valid_modes}, got {mode}")
        self.mode = mode
        self.register_buffer("threshold", torch.tensor(float(threshold)))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Compute keep mask based on activation magnitude.
        
        Parameters
        ----------
        h : torch.Tensor
            Hidden states, shape [..., hidden_dim].
            
        Returns
        -------
        torch.Tensor
            Boolean mask, True for nodes to keep.
        """
        if self.mode == "l2":
            score = torch.linalg.vector_norm(h, ord=2, dim=-1)
        elif self.mode == "l1":
            score = torch.linalg.vector_norm(h, ord=1, dim=-1)
        elif self.mode == "mean_abs":
            score = h.abs().mean(dim=-1)
        else:  # max_abs
            score = h.abs().max(dim=-1).values
        
        return score >= self.threshold


# =============================================================================
# SELF-TEST
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    torch.manual_seed(42)
    B, H = 8, 64
    x = torch.randn(B, H)

    print("=" * 70)
    print("gating.py v3.1.0 Self-Test (Fixed GrowthPolicyNet)")
    print("=" * 70)

    # Test 1: GrowthPolicyNet basic functionality
    print("\n[Test 1] GrowthPolicyNet basic functionality")
    policy = GrowthPolicyNet(hidden_dim=H, max_depth=4)
    policy._debug = True
    
    for depth in range(5):
        p = policy(x, depth)
        assert p.shape == (B,), f"Expected shape ({B},), got {p.shape}"
        assert (p >= PROB_CLAMP_MIN).all(), "Probability below minimum"
        assert (p <= PROB_CLAMP_MAX).all(), "Probability above maximum"
        print(f"  depth={depth}: mean_prob={p.mean().item():.4f}")
    print("  ✓ Basic functionality OK")

    # Test 2: GrowthPolicyNet initial probability near 0.5
    print("\n[Test 2] GrowthPolicyNet initial probability (should be ~0.5)")
    fresh_policy = GrowthPolicyNet(hidden_dim=H, max_depth=4)
    # Use zero input to test bias initialization
    zero_input = torch.zeros(B, H)
    p_zero = fresh_policy(zero_input, 0)
    print(f"  With zero input: mean_prob={p_zero.mean().item():.4f}")
    
    # Use random input
    p_rand = fresh_policy(x, 0)
    print(f"  With random input: mean_prob={p_rand.mean().item():.4f}")
    
    # Check that probability is reasonably close to 0.5 initially
    if 0.3 <= p_rand.mean().item() <= 0.7:
        print("  ✓ Initial probability is in reasonable range [0.3, 0.7]")
    else:
        print("  ⚠ Initial probability may be too extreme")

    # Test 3: GrowthPolicyNet with extreme inputs
    print("\n[Test 3] GrowthPolicyNet with extreme inputs")
    x_large = torch.randn(B, H) * 1000
    p_large = policy(x_large, 0)
    assert not torch.isnan(p_large).any(), "NaN in output with large inputs"
    assert (p_large >= PROB_CLAMP_MIN).all(), "Probability below minimum"
    assert (p_large <= PROB_CLAMP_MAX).all(), "Probability above maximum"
    print(f"  With 1000x input: mean_prob={p_large.mean().item():.4f}")
    print("  ✓ Extreme inputs OK")

    # Test 4: GrowthPolicyNet with NaN inputs
    print("\n[Test 4] GrowthPolicyNet with NaN inputs")
    x_nan = torch.randn(B, H)
    x_nan[0, 0] = float('nan')
    p_nan = policy(x_nan, 0)
    assert not torch.isnan(p_nan).any(), "NaN propagated to output"
    print("  ✓ NaN handling OK")

    # Test 5: GrowthPolicyNet gradient flow
    print("\n[Test 5] GrowthPolicyNet gradient flow")
    x_grad = torch.randn(B, H, requires_grad=True)
    p_grad = policy(x_grad, 2)
    loss = p_grad.sum()
    loss.backward()
    assert x_grad.grad is not None, "No gradient computed"
    assert not torch.isnan(x_grad.grad).any(), "NaN in gradient"
    print(f"  Gradient norm: {x_grad.grad.norm().item():.6f}")
    print("  ✓ Gradient flow OK")

    # Test 6: ScalarGate
    print("\n[Test 6] ScalarGate")
    sg = ScalarGate(in_dim=H)
    p_soft, z = sg(x)
    assert p_soft.shape == (B, 1), f"Expected ({B}, 1), got {p_soft.shape}"
    assert z.shape == (B, 1), f"Expected ({B}, 1), got {z.shape}"
    assert ((z == 0) | (z == 1)).all(), "z should be binary"
    print("  ✓ ScalarGate OK")

    # Test 7: HardConcreteGate
    print("\n[Test 7] HardConcreteGate")
    hcg = HardConcreteGate(in_dim=H)
    p_open, z_cont = hcg(x, training=True)
    assert p_open.shape == (B, 1), f"Expected ({B}, 1), got {p_open.shape}"
    assert (z_cont >= 0).all() and (z_cont <= 1).all(), "z should be in [0, 1]"
    print("  ✓ HardConcreteGate OK")

    # Test 8: ThresholdPruner
    print("\n[Test 8] ThresholdPruner")
    for mode in ["l2", "l1", "mean_abs", "max_abs"]:
        pruner = ThresholdPruner(mode=mode, threshold=1.5)
        keep = pruner(x)
        assert keep.shape == (B,), f"Expected ({B},), got {keep.shape}"
        assert keep.dtype == torch.bool, f"Expected bool, got {keep.dtype}"
    print("  ✓ ThresholdPruner OK")

    # Test 9: Level-aggregated input simulation (True BFS)
    print("\n[Test 9] Level-aggregated input simulation (True BFS)")
    # Simulate what happens in True BFS: aggregate multiple nodes
    num_nodes_at_level = 4
    node_hiddens = torch.randn(num_nodes_at_level, B, H)
    level_aggregated = node_hiddens.mean(dim=0)  # [B, H]
    p_level = policy(level_aggregated, depth_idx=1)
    # In True BFS, we average across batch to get single decision
    p_level_decision = p_level.mean()
    assert p_level_decision.shape == (), "Should be scalar"
    assert PROB_CLAMP_MIN <= p_level_decision <= PROB_CLAMP_MAX
    print(f"  Level decision probability: {p_level_decision.item():.4f}")
    print("  ✓ Level-aggregated input OK")

    # Test 10: Verify output shape is [N] not [N, 1]
    print("\n[Test 10] Output shape verification")
    test_input = torch.randn(16, H)
    output = policy(test_input, 0)
    assert output.shape == (16,), f"Expected (16,), got {output.shape}"
    assert output.dim() == 1, f"Expected 1D tensor, got {output.dim()}D"
    print(f"  Output shape: {output.shape}")
    print("  ✓ Output shape is [N] as expected")

    print("\n" + "=" * 70)
    print("All tests passed! GrowthPolicyNet v3.1.0 is working correctly.")
    print("=" * 70)