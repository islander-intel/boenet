#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
boenet/model.py (v1.1.0 - Language Model)

BoeNet: BFS-Inspired Language Model with REINFORCE Policy Gradients

v1.1.0 Changes (Bug Fixes for K>0 Training)
-------------------------------------------
FIXED:
  - Added torch.clamp() on grow_prob before Bernoulli sampling (prevents CUDA assertion)
  - Added NaN detection and replacement in _rollout() method
  - Added gradient-safe log probability computation
  - Added numerical stability guards throughout

The CUDA error "Assertion `0 <= p4 && p4 <= 1` failed" was caused by:
  1. Policy network producing NaN due to gradient explosion
  2. NaN propagating to torch.bernoulli() which requires probs in [0, 1]
  3. GPU detecting invalid probability values

Converted from BFSNet v2.0.0 (Vision) to BoeNet v1.0.0 (Language)
-----------------------------------------------------------------
This model applies the proven BFS tree expansion + REINFORCE architecture
to language modeling. The core algorithm is identical - only the input/output
layers change.

Key Changes from BFSNet:
------------------------
REMOVED:
  - root_fc = nn.Linear(784, hidden_dim)  # Image input
  - output_fc = nn.Linear(hidden_dim, 10)  # Classification

ADDED:
  - embedding = nn.Embedding(vocab_size, embed_dim)  # Token input
  - embed_proj = nn.Linear(embed_dim, hidden_dim)    # Optional projection
  - output_fc = nn.Linear(hidden_dim, vocab_size)    # Next-token prediction

UNCHANGED:
  - GrowthPolicyNet (binary grow/stop decisions)
  - BFS tree expansion (_rollout)
  - REINFORCE policy gradients
  - Greedy threshold mechanism
  - All utility functions

Architecture Overview (v1.1.0)
------------------------------
Training:
  1. Input: [B, seq_len] token IDs
  2. Embed: [B, seq_len, embed_dim] → reshape to [B*seq_len, embed_dim]
  3. Project: [B*seq_len, hidden_dim] (root nodes)
  4. BFS expansion with num_rollouts stochastic rollouts
  5. Compute rewards: -cross_entropy_loss - λ * efficiency_penalty
  6. Update policy via REINFORCE
  7. Reshape output: [B, seq_len, vocab_size]

Inference:
  1. Single greedy rollout (deterministic)
  2. Output: [B, seq_len, vocab_size] logits

The Algorithm is Input-Agnostic
-------------------------------
The BFS tree expansion doesn't care where vectors come from:
  - Vision: flattened image pixels → root_fc → hidden vector
  - Language: token IDs → embedding → hidden vector

The tree processes hidden vectors and makes grow/stop decisions.
This is why the conversion is minimal - we just swap input layers.

Greedy Threshold Selection (Same as BFSNet)
-------------------------------------------
  - threshold = 0.50 (default): Conservative, often root-only
  - threshold = 0.42-0.45 (recommended): Balanced expansion
  - threshold = 0.30-0.35: Aggressive, near-full expansion

Empirical finding: optimal_threshold ≈ mean_grow_prob - 0.03

Usage Examples
--------------
>>> # Character-level language model
>>> model = BoeNet(vocab_size=256, embed_dim=64, hidden_dim=128,
...                max_depth=2, max_children=3, greedy_threshold=0.42)
>>> 
>>> # Training
>>> model.train()
>>> x = torch.randint(0, 256, (32, 128))  # [B, seq_len]
>>> y = torch.randint(0, 256, (32, 128))  # [B, seq_len] targets
>>> outputs, policy_loss, rewards, node_counts = model(
...     x, num_rollouts=3, lambda_efficiency=0.05,
...     beta_entropy=0.01, labels=y
... )
>>> # outputs: [B, seq_len, vocab_size]
>>> 
>>> # Inference
>>> model.eval()
>>> logits = model(x)  # [B, seq_len, vocab_size]

Author: BoeNet project (converted from BFSNet)
Version: 1.1.0
Date: 2025-12-30
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Literal
import warnings
import math
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import GrowthPolicyNet from gating module (unchanged from BFSNet)
from boenet.utils.gating import GrowthPolicyNet, ScalarGate, ThresholdPruner

# Module-level logger
logger = logging.getLogger(__name__)

# =============================================================================
# NUMERICAL STABILITY CONSTANTS (v1.1.0)
# =============================================================================
# These constants prevent CUDA assertion failures from invalid probabilities
PROB_CLAMP_MIN = 1e-7      # Minimum probability value (prevents log(0))
PROB_CLAMP_MAX = 1.0 - 1e-7  # Maximum probability value (prevents log(1-1)=log(0))
LOG_PROB_CLAMP_MIN = -20.0  # Minimum log probability (prevents -inf)
LOG_PROB_CLAMP_MAX = 0.0    # Maximum log probability (log(1) = 0)

# Safe scatter-add (unchanged from BFSNet)
try:
    from boenet.utils.sparse_utils import safe_index_add
except Exception:
    def safe_index_add(
        dst: torch.Tensor,
        dim: int,
        index: torch.Tensor,
        src: torch.Tensor
    ) -> torch.Tensor:
        """Fallback in-place scatter-add."""
        if index.numel() == 0 or src.numel() == 0:
            return dst
        if index.dtype != torch.long:
            index = index.to(torch.long)
        if index.device != dst.device:
            index = index.to(dst.device)
        if src.dtype != dst.dtype:
            src = src.to(dst.dtype)
        dst.index_add_(dim, index, src)
        return dst


# =============================================================================
# HELPER FUNCTIONS (v1.1.0)
# =============================================================================

def _check_for_nan(tensor: torch.Tensor, name: str, replace_value: float = 0.5) -> torch.Tensor:
    """
    Check tensor for NaN values and replace them with a safe default.
    
    This prevents NaN propagation through the model, which would otherwise
    cause CUDA assertion failures when NaN reaches torch.bernoulli().
    
    Parameters
    ----------
    tensor : torch.Tensor
        Tensor to check.
    name : str
        Name of tensor for logging.
    replace_value : float, default=0.5
        Value to replace NaN with (0.5 is neutral for Bernoulli).
        
    Returns
    -------
    torch.Tensor
        Tensor with NaN values replaced.
    """
    if torch.isnan(tensor).any():
        nan_count = torch.isnan(tensor).sum().item()
        logger.warning(
            f"[NaN DETECTED] {name} contains {nan_count} NaN values. "
            f"Replacing with {replace_value}. This indicates gradient explosion - "
            f"consider reducing learning rate or increasing gradient clipping."
        )
        tensor = torch.where(torch.isnan(tensor), torch.full_like(tensor, replace_value), tensor)
    return tensor


def _safe_clamp_prob(prob: torch.Tensor, name: str = "prob") -> torch.Tensor:
    """
    Safely clamp probability values to valid range [PROB_CLAMP_MIN, PROB_CLAMP_MAX].
    
    This is CRITICAL for preventing CUDA assertion failures.
    The error "Assertion `0 <= p4 && p4 <= 1` failed" occurs when
    torch.bernoulli() receives probabilities outside [0, 1].
    
    Parameters
    ----------
    prob : torch.Tensor
        Probability tensor (expected to be in [0, 1] but may have numerical issues).
    name : str
        Name for logging purposes.
        
    Returns
    -------
    torch.Tensor
        Clamped probability tensor guaranteed to be in [PROB_CLAMP_MIN, PROB_CLAMP_MAX].
    """
    # First check for NaN
    prob = _check_for_nan(prob, name, replace_value=0.5)
    
    # Check for values outside [0, 1] before clamping (for debugging)
    if prob.numel() > 0:
        min_val = prob.min().item()
        max_val = prob.max().item()
        if min_val < 0.0 or max_val > 1.0:
            logger.warning(
                f"[PROB OUT OF RANGE] {name} has values outside [0,1]: "
                f"min={min_val:.6f}, max={max_val:.6f}. Clamping to valid range."
            )
    
    # Clamp to valid range
    return prob.clamp(PROB_CLAMP_MIN, PROB_CLAMP_MAX)


def _safe_log_prob(prob: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """
    Compute log probability for Bernoulli action in a numerically stable way.
    
    For action a ∈ {0, 1} and probability p:
        log_prob = a * log(p) + (1-a) * log(1-p)
    
    This function handles edge cases:
        - p very close to 0 or 1
        - NaN values in p
        - Numerical underflow in log
    
    Parameters
    ----------
    prob : torch.Tensor
        Probability of action=1, shape [...].
    action : torch.Tensor
        Sampled action (0 or 1), same shape as prob.
        
    Returns
    -------
    torch.Tensor
        Log probability of the action, clamped to prevent -inf.
    """
    # Ensure probability is in valid range
    p = prob.clamp(PROB_CLAMP_MIN, PROB_CLAMP_MAX)
    
    # Compute log probabilities safely
    log_p = torch.log(p)
    log_1_minus_p = torch.log(1.0 - p)
    
    # Clamp log probabilities to prevent -inf
    log_p = log_p.clamp(min=LOG_PROB_CLAMP_MIN)
    log_1_minus_p = log_1_minus_p.clamp(min=LOG_PROB_CLAMP_MIN)
    
    # Compute log probability of action
    log_prob = action * log_p + (1.0 - action) * log_1_minus_p
    
    # Final clamp and NaN check
    log_prob = log_prob.clamp(min=LOG_PROB_CLAMP_MIN, max=LOG_PROB_CLAMP_MAX)
    log_prob = _check_for_nan(log_prob, "log_prob", replace_value=LOG_PROB_CLAMP_MIN)
    
    return log_prob


# --------------------------------------------------------------------------- #
#                                   Model                                     #
# --------------------------------------------------------------------------- #

class BoeNet(nn.Module):
    """
    BFS-Inspired Language Model with REINFORCE Policy Gradients (v1.1.0).
    
    This model performs breadth-first expansion where each node can spawn
    0 to K children based on learned binary growth policies. The policies
    are trained via REINFORCE to maximize a reward that balances language
    modeling performance and computational efficiency.
    
    v1.1.0 FIXES:
    - Added probability clamping before torch.bernoulli() to prevent CUDA assertion
    - Added NaN detection and replacement throughout _rollout()
    - Added gradient-safe log probability computation
    - Added numerical stability constants
    
    Parameters
    ----------
    vocab_size : int
        Vocabulary size (256 for char-level, 50257 for GPT-2 BPE).
    embed_dim : int
        Token embedding dimension.
    hidden_dim : int
        Hidden layer dimension for BFS tree nodes.
    max_depth : int, default=2
        Maximum BFS expansion depth.
    max_children : int, default=2
        Maximum children K per parent node.
    greedy_threshold : float, default=0.5
        Threshold for greedy inference: grow if grow_prob >= threshold.
        - 0.5 (default): Conservative, often root-only
        - 0.42-0.45: Balanced, partial expansion
        - 0.30-0.35: Aggressive, near-full expansion
        See module docstring for selection guide.
    sibling_embed : bool, default=True
        Whether to add learnable sibling position embeddings.
    use_pruning : bool, default=False
        Enable parent-level pruning gate.
    pruning_mode : str, default="learned"
        Pruning strategy: "learned" or "threshold".
    pruning_threshold : float, default=1e-3
        Threshold for magnitude-based pruning.
    eps : float, default=1e-8
        Small constant for numerical stability.
    sibling_scale : float, optional
        Scale factor for sibling embeddings. Default: 1/sqrt(hidden_dim).
    pooling_mode : {"mean", "sum", "learned"}, default="mean"
        Aggregation strategy for node contributions.
        
    Attributes
    ----------
    embedding : nn.Embedding
        Token embedding layer.
    embed_proj : nn.Linear
        Projects embed_dim to hidden_dim (identity if equal).
    growth_policy : GrowthPolicyNet
        Binary grow/stop policy network.
    child_fc : nn.Linear
        Transforms parent hidden state to child hidden state.
    output_fc : nn.Linear
        Final next-token prediction head.
    greedy_threshold : float
        Threshold for greedy inference decisions.
    _debug : bool
        If True, print verbose debug information during forward pass.
        
    Notes
    -----
    Training mode (training=True):
        Returns (outputs, policy_loss, rewards, node_counts)
        - outputs: [B, seq_len, vocab_size] averaged over rollouts
        - policy_loss: scalar REINFORCE loss
        - rewards: [num_rollouts] per-rollout rewards
        - node_counts: [num_rollouts] per-rollout node counts
        
    Inference mode (training=False):
        Returns logits: [B, seq_len, vocab_size] from single greedy rollout
    """
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        max_depth: int = 2,
        max_children: int = 2,
        greedy_threshold: float = 0.5,
        sibling_embed: bool = True,
        use_pruning: bool = False,
        pruning_mode: str = "learned",
        pruning_threshold: float = 1e-3,
        eps: float = 1e-8,
        sibling_scale: Optional[float] = None,
        *,
        pooling: Optional[Literal["mean", "sum", "learned"]] = None,
        pooling_mode: Literal["mean", "sum", "learned"] = "mean",
    ):
        super().__init__()
        
        # Handle pooling/pooling_mode alias
        if pooling is not None:
            if pooling_mode != "mean":
                warnings.warn(
                    f"Both 'pooling' and 'pooling_mode' provided. "
                    f"Using 'pooling={pooling}' (pooling_mode={pooling_mode} ignored).",
                    DeprecationWarning,
                    stacklevel=2
                )
            pooling_mode = pooling
        
        # Core dims
        self.vocab_size = int(vocab_size)
        self.embed_dim = int(embed_dim)
        self.hidden_dim = int(hidden_dim)
        
        # BFS controls
        self.max_depth = max(0, int(max_depth))
        self.max_children = max(0, int(max_children))
        self.eps = float(eps)
        
        # Greedy threshold
        self.greedy_threshold = float(greedy_threshold)
        if not (0.0 <= self.greedy_threshold <= 1.0):
            raise ValueError(f"greedy_threshold must be in [0, 1], got {greedy_threshold}")
        
        # Debug flag
        self._debug = False
        
        # ====================================================================
        # INPUT LAYERS (Changed from BFSNet)
        # ====================================================================
        # Token embedding: [vocab_size, embed_dim]
        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        
        # Project embedding to hidden_dim (root node creation)
        # This replaces root_fc from BFSNet
        self.embed_proj = nn.Linear(self.embed_dim, self.hidden_dim)
        nn.init.kaiming_uniform_(self.embed_proj.weight, a=math.sqrt(5))
        if self.embed_proj.bias is not None:
            nn.init.zeros_(self.embed_proj.bias)
        
        # ====================================================================
        # BFS TREE LAYERS (Unchanged from BFSNet)
        # ====================================================================
        # Child node transformation
        self.child_fc = nn.Linear(self.hidden_dim, self.hidden_dim)
        nn.init.kaiming_uniform_(self.child_fc.weight, a=math.sqrt(5))
        if self.child_fc.bias is not None:
            nn.init.zeros_(self.child_fc.bias)
        
        # Growth policy (binary grow/stop decisions)
        self.growth_policy = GrowthPolicyNet(
            hidden_dim=self.hidden_dim,
            max_depth=self.max_depth
        )
        
        # Optional sibling embeddings
        self.sibling_embed = bool(sibling_embed) and (self.max_children > 0)
        if self.sibling_embed:
            self.sibling_embeddings = nn.Embedding(self.max_children, self.hidden_dim)
            nn.init.normal_(self.sibling_embeddings.weight, mean=0.0, std=0.02)
        self.sibling_scale = (
            (1.0 / math.sqrt(self.hidden_dim))
            if sibling_scale is None
            else float(sibling_scale)
        )
        
        # Pruning (unchanged)
        self.use_pruning = bool(use_pruning)
        self.pruning_mode = pruning_mode
        self.pruning_threshold = float(pruning_threshold)
        if self.use_pruning:
            if pruning_mode == "learned":
                self.prune_gate = ScalarGate(self.hidden_dim, bias=True)
                self.pruner = None
            elif pruning_mode == "threshold":
                self.prune_gate = None
                self.pruner = ThresholdPruner(mode="l2", threshold=self.pruning_threshold)
            else:
                raise ValueError("pruning_mode must be 'learned' or 'threshold'")
        else:
            self.prune_gate = None
            self.pruner = None
        
        # Pooling (unchanged)
        self.pooling_mode = pooling_mode
        if self.pooling_mode == "learned":
            self._pool_log_p = nn.Parameter(torch.log(torch.expm1(torch.tensor(1.0))))
        else:
            self.register_parameter("_pool_log_p", None)
        
        # ====================================================================
        # OUTPUT LAYER (Changed from BFSNet)
        # ====================================================================
        # Next-token prediction: hidden_dim → vocab_size
        self.output_fc = nn.Linear(self.hidden_dim, self.vocab_size)
        nn.init.kaiming_uniform_(self.output_fc.weight, a=math.sqrt(5))
        if self.output_fc.bias is not None:
            nn.init.zeros_(self.output_fc.bias)
    
    # ----------------------------- Pooling --------------------------------- #
    
    def _pool(
        self,
        agg_sum: torch.Tensor,
        agg_count: torch.Tensor,
    ) -> torch.Tensor:
        """
        Pool aggregated features.
        
        Parameters
        ----------
        agg_sum : torch.Tensor
            Sum of node contributions, shape [N, H].
        agg_count : torch.Tensor
            Count of contributing nodes, shape [N, 1].
            
        Returns
        -------
        torch.Tensor
            Pooled features, shape [N, H].
        """
        if self.pooling_mode == "sum":
            return agg_sum
        if self.pooling_mode == "mean":
            denom = agg_count.clamp_min(self.eps)
            return agg_sum / denom
        if self.pooling_mode == "learned":
            p = (
                F.softplus(self._pool_log_p)
                if self._pool_log_p is not None
                else torch.tensor(1.0, device=agg_sum.device)
            )
            denom = torch.pow(agg_count.clamp_min(self.eps), p)
            return agg_sum / denom
        raise ValueError(f"Unknown pooling_mode: {self.pooling_mode!r}")
    
    # --------------------------- Rollout Execution ------------------------- #
    
    def _rollout(
        self,
        h0: torch.Tensor,
        greedy: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], int]:
        """
        Execute one rollout (stochastic or greedy) through the BFS tree.
        
        This is the core BFS expansion - IDENTICAL to BFSNet.
        The only difference is h0 comes from embedding instead of root_fc.
        
        v1.1.0 FIXES:
        - Added probability clamping before torch.bernoulli()
        - Added NaN detection on hidden states and probabilities
        - Added gradient-safe log probability computation
        
        Parameters
        ----------
        h0 : torch.Tensor
            Root node hidden states, shape [N, hidden_dim].
            N = B * seq_len for language modeling.
        greedy : bool, default=False
            If True, use deterministic policy (grow_prob >= greedy_threshold).
            If False, sample actions stochastically.
            
        Returns
        -------
        tuple
            output : [N, vocab_size] next-token logits
            log_probs : [T] concatenated log probabilities (None if greedy)
            node_count : int, total nodes created in this rollout
        """
        N = h0.size(0)
        device = h0.device
        
        if self._debug:
            print(f"\n[DEBUG _rollout] Starting rollout: N={N}, greedy={greedy}")
            if greedy:
                print(f"[DEBUG _rollout] Greedy threshold: {self.greedy_threshold:.4f}")
        
        # =======================================================================
        # v1.1.0: Check h0 for NaN before starting
        # =======================================================================
        h0 = _check_for_nan(h0, "h0 (root hidden states)")
        
        # Accumulators
        agg_sum = h0.new_zeros(N, self.hidden_dim)
        agg_count = h0.new_zeros(N, 1)
        
        # Root nodes (h0 already computed from embedding + projection)
        root_idx = torch.arange(N, device=device, dtype=torch.long)
        safe_index_add(agg_sum, 0, root_idx, h0)
        safe_index_add(agg_count, 0, root_idx, h0.new_ones(N, 1))
        
        frontier: List[Tuple[torch.Tensor, torch.Tensor]] = [(h0, root_idx)]
        log_probs_list: List[torch.Tensor] = []
        node_count = N  # Start with root nodes
        
        if self._debug:
            print(f"[DEBUG _rollout] Created {N} root nodes")
        
        # BFS expansion (IDENTICAL to BFSNet)
        for depth in range(self.max_depth):
            next_frontier: List[Tuple[torch.Tensor, torch.Tensor]] = []
            
            if self._debug:
                print(f"\n[DEBUG _rollout] Depth {depth}: {len(frontier)} parent groups")
            
            for parent_group_idx, (parent_h, parent_idx) in enumerate(frontier):
                Np = parent_idx.numel()
                if Np == 0:
                    continue
                
                # =============================================================
                # v1.1.0: Check parent_h for NaN
                # =============================================================
                parent_h = _check_for_nan(parent_h, f"parent_h (depth={depth}, group={parent_group_idx})")
                
                if self._debug:
                    print(f"[DEBUG _rollout]   Parent group {parent_group_idx}: {Np} parents")
                
                # Optional pruning
                if self.use_pruning:
                    if self.prune_gate is not None:
                        prune_soft, z_hard = self.prune_gate(parent_h)
                        keep_mask = (z_hard.squeeze(-1) >= 0.5)
                    else:
                        keep_mask = self.pruner(parent_h)
                else:
                    keep_mask = torch.ones(Np, dtype=torch.bool, device=device)
                
                if keep_mask.sum().item() == 0:
                    if self._debug:
                        print(f"[DEBUG _rollout]     All pruned, skipping")
                    continue
                
                act_h = parent_h[keep_mask]      # [Na, H]
                act_idx = parent_idx[keep_mask]  # [Na]
                Na = act_idx.numel()
                
                if self._debug:
                    print(f"[DEBUG _rollout]     After pruning: {Na} active parents")
                
                if self.max_children == 0:
                    continue
                
                # ============================================================
                # CORE: Decide BEFORE computing (true sparsity)
                # ============================================================
                for j in range(self.max_children):
                    # Get grow probability from policy
                    grow_prob_raw = self.growth_policy(act_h, depth)  # [Na, 1]
                    
                    # ==========================================================
                    # v1.1.0 FIX: CLAMP PROBABILITY TO VALID RANGE
                    # This prevents CUDA assertion "0 <= p4 && p4 <= 1" failure
                    # ==========================================================
                    grow_prob = _safe_clamp_prob(grow_prob_raw, name=f"grow_prob (depth={depth}, child={j})")
                    
                    if self._debug and parent_group_idx == 0 and j == 0:
                        prob_values = grow_prob.squeeze(-1).detach().cpu()
                        raw_values = grow_prob_raw.squeeze(-1).detach().cpu()
                        print(f"[DEBUG _rollout]     Child {j} grow_prob stats:")
                        print(f"[DEBUG _rollout]       raw: mean={raw_values.mean():.4f}, "
                              f"min={raw_values.min():.4f}, max={raw_values.max():.4f}")
                        print(f"[DEBUG _rollout]       clamped: mean={prob_values.mean():.4f}, "
                              f"min={prob_values.min():.4f}, max={prob_values.max():.4f}")
                    
                    if greedy:
                        # Deterministic threshold-based decision
                        action = (grow_prob >= self.greedy_threshold).float()
                        
                        if self._debug:
                            num_growing = action.sum().item()
                            above_thresh = (grow_prob >= self.greedy_threshold).sum().item()
                            print(f"[DEBUG _rollout]     Child {j} greedy decision: "
                                  f"{num_growing}/{Na} growing "
                                  f"(threshold {self.greedy_threshold:.4f}, "
                                  f"{above_thresh}/{Na} above)")
                    else:
                        # =======================================================
                        # v1.1.0 FIX: Stochastic sampling with clamped probability
                        # grow_prob is already clamped by _safe_clamp_prob above
                        # =======================================================
                        action = torch.bernoulli(grow_prob)
                        
                        # =======================================================
                        # v1.1.0 FIX: Gradient-safe log probability computation
                        # =======================================================
                        log_p = _safe_log_prob(grow_prob, action)
                        log_probs_list.append(log_p.squeeze(-1))  # [Na]
                        
                        if self._debug:
                            num_growing = action.sum().item()
                            print(f"[DEBUG _rollout]     Child {j} stochastic sample: "
                                  f"{num_growing}/{Na} growing")
                    
                    # KEY: Only compute if growing
                    grow_mask = (action.squeeze(-1) >= 0.5)  # [Na]
                    num_growing = grow_mask.sum().item()
                    
                    if num_growing == 0:
                        if self._debug:
                            print(f"[DEBUG _rollout]     Child {j}: SKIPPED (no growth)")
                        continue
                    
                    # Extract nodes that are growing
                    h_in = act_h[grow_mask]           # [Ng, H]
                    child_idx = act_idx[grow_mask]    # [Ng]
                    
                    # Add sibling embedding if enabled
                    if self.sibling_embed:
                        h_in = h_in + self.sibling_scale * self.sibling_embeddings.weight[j]
                    
                    # *** ONLY NOW do we compute the child ***
                    child_h = F.relu(self.child_fc(h_in))  # [Ng, H]
                    
                    # ==========================================================
                    # v1.1.0: Check child_h for NaN
                    # ==========================================================
                    child_h = _check_for_nan(child_h, f"child_h (depth={depth}, child={j})")
                    
                    if self._debug:
                        print(f"[DEBUG _rollout]     Child {j}: CREATED {child_h.size(0)} nodes")
                    
                    # Accumulate
                    safe_index_add(agg_sum, 0, child_idx, child_h)
                    safe_index_add(agg_count, 0, child_idx, child_h.new_ones(child_idx.size(0), 1))
                    
                    # Add to next frontier
                    next_frontier.append((child_h, child_idx))
                    node_count += child_h.size(0)
            
            frontier = next_frontier
            if len(frontier) == 0:
                if self._debug:
                    print(f"[DEBUG _rollout] Frontier empty, stopping at depth {depth}")
                break
        
        # =======================================================================
        # v1.1.0: Check aggregated sum for NaN before pooling
        # =======================================================================
        agg_sum = _check_for_nan(agg_sum, "agg_sum (before pooling)")
        
        # Final pooling and output projection
        pooled = self._pool(agg_sum, agg_count)
        
        # v1.1.0: Check pooled for NaN
        pooled = _check_for_nan(pooled, "pooled (after pooling)")
        
        output = self.output_fc(pooled)  # [N, vocab_size]
        
        # v1.1.0: Check output for NaN
        output = _check_for_nan(output, "output (final logits)")
        
        # Concatenate log probs if not greedy
        log_probs = torch.cat(log_probs_list) if len(log_probs_list) > 0 else None
        
        if self._debug:
            print(f"[DEBUG _rollout] Final node count: {node_count}")
            print(f"[DEBUG _rollout] Output shape: {output.shape}")
            if log_probs is not None:
                print(f"[DEBUG _rollout] Log probs: shape={log_probs.shape}, "
                      f"min={log_probs.min():.4f}, max={log_probs.max():.4f}")
            print()
        
        return output, log_probs, node_count
    
    # --------------------------- Reward & Policy Loss ---------------------- #
    
    def _compute_rewards(
        self,
        outputs: List[torch.Tensor],
        node_counts: List[int],
        labels: torch.Tensor,
        lambda_efficiency: float,
        batch_size: int,
        seq_len: int,
    ) -> torch.Tensor:
        """
        Compute per-rollout rewards for REINFORCE (Language Model version).
        
        Reward structure (different from BFSNet):
            reward = -cross_entropy_loss - λ * efficiency_penalty
        
        We use NEGATIVE cross-entropy because:
            - Lower loss = better model = higher reward
            - REINFORCE maximizes expected reward
        
        v1.1.0 CHANGES:
        - Added reward scaling to prevent gradient explosion
        - Added NaN detection in rewards
        - Clamped efficiency penalty to reasonable range
        
        Parameters
        ----------
        outputs : List[torch.Tensor]
            List of output logits from each rollout, each shape [B*seq_len, vocab_size].
        node_counts : List[int]
            List of TOTAL node counts from each rollout.
        labels : torch.Tensor
            Ground truth next tokens, shape [B, seq_len].
        lambda_efficiency : float
            Efficiency penalty coefficient (typically 0.01-0.1).
        batch_size : int
            Original batch size B.
        seq_len : int
            Sequence length.
            
        Returns
        -------
        torch.Tensor
            Rewards tensor, shape [num_rollouts].
        """
        # Compute theoretical maximum nodes per token position
        max_nodes_per_position = 1  # Root node
        for d in range(1, self.max_depth + 1):
            max_nodes_per_position += self.max_children ** d
        
        # Total max nodes = max_per_position * num_positions
        num_positions = batch_size * seq_len
        max_nodes_total = max_nodes_per_position * num_positions
        
        # Flatten labels for cross-entropy computation
        labels_flat = labels.view(-1)  # [B * seq_len]
        
        rewards = []
        for rollout_idx, (out, nodes) in enumerate(zip(outputs, node_counts)):
            # out shape: [B * seq_len, vocab_size]
            # labels_flat shape: [B * seq_len]
            
            # Cross-entropy loss (lower is better)
            ce_loss = F.cross_entropy(out, labels_flat, reduction='mean')
            
            # =================================================================
            # v1.1.0: Check for NaN/Inf in CE loss
            # =================================================================
            if torch.isnan(ce_loss) or torch.isinf(ce_loss):
                logger.warning(
                    f"[REWARD] Rollout {rollout_idx}: CE loss is NaN/Inf ({ce_loss.item()}). "
                    f"Using default penalty value."
                )
                ce_loss = torch.tensor(10.0, device=out.device, dtype=out.dtype)
            
            # Reward component: NEGATIVE loss (so higher reward = lower loss)
            # Typical CE loss for language models: 2-5 (untrained) → 1-2 (trained)
            # We negate so that reward increases as model improves
            reward_accuracy = -ce_loss
            
            # Efficiency penalty: nodes_used / max_nodes
            # Normalize by total positions, not just batch_size
            nodes_per_position = nodes / num_positions
            efficiency_ratio = nodes_per_position / float(max_nodes_per_position)
            
            # =================================================================
            # v1.1.0: Clamp efficiency ratio to [0, 1] to prevent extreme penalties
            # =================================================================
            efficiency_ratio = min(max(efficiency_ratio, 0.0), 1.0)
            efficiency_penalty = lambda_efficiency * efficiency_ratio
            
            # Combined reward
            reward = reward_accuracy - efficiency_penalty
            
            # =================================================================
            # v1.1.0: Scale reward to prevent gradient explosion
            # Typical reward range: [-10, 0] -> scale to [-1, 0]
            # =================================================================
            # Note: We scale by dividing by a constant to keep rewards in a
            # more stable range. This prevents the policy loss from exploding.
            reward_scale = 5.0  # Typical CE loss magnitude
            reward_scaled = reward / reward_scale
            
            rewards.append(reward_scaled)
            
            # Debug logging
            if self._debug and rollout_idx == 0:
                print("\n" + "=" * 79)
                print("REWARD COMPUTATION DEBUG (Rollout 0) - Language Model v1.1.0")
                print("=" * 79)
                print(f"Batch size:               {batch_size}")
                print(f"Sequence length:          {seq_len}")
                print(f"Total positions:          {num_positions}")
                print(f"Cross-entropy loss:       {ce_loss.item():.4f}")
                print(f"Reward (neg loss):        {reward_accuracy.item():.4f}")
                print(f"Total nodes:              {nodes}")
                print(f"Nodes per position:       {nodes_per_position:.2f}")
                print(f"Max nodes per position:   {max_nodes_per_position}")
                print(f"Efficiency ratio:         {efficiency_ratio:.4f}")
                print(f"Lambda efficiency:        {lambda_efficiency:.4f}")
                print(f"Efficiency penalty:       {efficiency_penalty:.6f}")
                print(f"Raw reward:               {reward.item():.6f}")
                print(f"Reward scale factor:      {reward_scale}")
                print(f"Scaled reward:            {reward_scaled.item():.6f}")
                print("=" * 79 + "\n")
        
        return torch.stack(rewards)  # [num_rollouts]
    
    def _compute_policy_loss(
        self,
        log_probs: List[Optional[torch.Tensor]],
        rewards: torch.Tensor,
        beta_entropy: float,
    ) -> torch.Tensor:
        """
        Compute REINFORCE policy loss with baseline and entropy bonus.
        
        IDENTICAL to BFSNet - the policy learning is input-agnostic.
        
        v1.1.0 CHANGES:
        - Added gradient scaling to prevent explosion
        - Added NaN detection in policy loss
        - Clamped advantages to reasonable range
        
        Parameters
        ----------
        log_probs : List[Optional[torch.Tensor]]
            List of log probability tensors from each rollout.
        rewards : torch.Tensor
            Rewards for each rollout.
        beta_entropy : float
            Entropy bonus coefficient.
            
        Returns
        -------
        torch.Tensor
            Scalar policy loss.
        """
        # Filter out None (greedy rollouts)
        valid_log_probs = [lp for lp in log_probs if lp is not None]
        
        if len(valid_log_probs) == 0:
            return torch.tensor(0.0, device=rewards.device, requires_grad=True)
        
        # Baseline (reduces variance)
        baseline = rewards.mean()
        
        policy_loss = torch.tensor(0.0, device=rewards.device, dtype=rewards.dtype)
        total_entropy = torch.tensor(0.0, device=rewards.device, dtype=rewards.dtype)
        total_decisions = 0
        
        for log_p, reward in zip(valid_log_probs, rewards):
            # =================================================================
            # v1.1.0: Check log_p for NaN
            # =================================================================
            if torch.isnan(log_p).any():
                nan_count = torch.isnan(log_p).sum().item()
                logger.warning(
                    f"[POLICY LOSS] log_p contains {nan_count} NaN values. Skipping this rollout."
                )
                continue
            
            # Advantage
            advantage = reward - baseline
            
            # =================================================================
            # v1.1.0: Clamp advantage to prevent extreme gradients
            # Typical advantage range after reward scaling: [-0.5, 0.5]
            # =================================================================
            advantage_clamped = advantage.clamp(-2.0, 2.0)
            
            # REINFORCE gradient
            # Negative because we minimize loss (= maximize reward)
            policy_loss = policy_loss - (log_p * advantage_clamped).sum()
            
            # Entropy bonus
            # For Bernoulli: H = -p*log(p) - (1-p)*log(1-p)
            # We have log_p = log(p) for action=1, log(1-p) for action=0
            # Approximate entropy using the log probs
            p = torch.exp(log_p).clamp(PROB_CLAMP_MIN, PROB_CLAMP_MAX)
            entropy = -(p * torch.log(p) + (1 - p) * torch.log(1 - p))
            
            # Check for NaN in entropy
            entropy = _check_for_nan(entropy, "entropy", replace_value=0.0)
            
            total_entropy = total_entropy + entropy.sum()
            total_decisions += log_p.numel()
        
        # =================================================================
        # v1.1.0: Average entropy across all decisions, not rollouts
        # =================================================================
        if total_decisions > 0:
            avg_entropy = total_entropy / total_decisions
        else:
            avg_entropy = torch.tensor(0.0, device=rewards.device)
        
        # Subtract entropy bonus (we want to maximize entropy = minimize -entropy)
        policy_loss = policy_loss - beta_entropy * avg_entropy * total_decisions
        
        # =================================================================
        # v1.1.0: Check final policy loss for NaN
        # =================================================================
        if torch.isnan(policy_loss):
            logger.warning("[POLICY LOSS] Final policy loss is NaN. Returning zero loss.")
            return torch.tensor(0.0, device=rewards.device, requires_grad=True)
        
        return policy_loss
    
    # ------------------------------- Forward ------------------------------- #
    
    def forward(
        self,
        x: torch.Tensor,
        num_rollouts: int = 3,
        lambda_efficiency: float = 0.05,
        beta_entropy: float = 0.01,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
        """
        Forward pass for language modeling.
        
        Parameters
        ----------
        x : torch.Tensor
            Input token IDs, shape [B, seq_len].
        num_rollouts : int, default=3
            Number of stochastic rollouts during training.
        lambda_efficiency : float, default=0.05
            Efficiency penalty in reward.
        beta_entropy : float, default=0.01
            Entropy bonus in policy loss.
        labels : torch.Tensor, optional
            Target token IDs for reward computation, shape [B, seq_len].
            Required during training.
            
        Returns
        -------
        If training=False (inference):
            logits : [B, seq_len, vocab_size]
        
        If training=True:
            tuple of (outputs, policy_loss, rewards, node_counts)
            - outputs : [B, seq_len, vocab_size] averaged over rollouts
            - policy_loss : scalar REINFORCE loss
            - rewards : [num_rollouts] per-rollout rewards
            - node_counts : [num_rollouts] per-rollout node counts
        """
        B, seq_len = x.shape
        device = x.device
        
        # ====================================================================
        # EMBEDDING (Changed from BFSNet)
        # ====================================================================
        # Embed tokens: [B, seq_len] → [B, seq_len, embed_dim]
        embedded = self.embedding(x)
        
        # Reshape for BFS processing: [B, seq_len, embed_dim] → [B*seq_len, embed_dim]
        embedded_flat = embedded.view(B * seq_len, self.embed_dim)
        
        # Project to hidden_dim (creates root nodes): [B*seq_len, hidden_dim]
        h0 = F.relu(self.embed_proj(embedded_flat))
        
        # =======================================================================
        # v1.1.0: Check h0 for NaN after projection
        # =======================================================================
        h0 = _check_for_nan(h0, "h0 (after embed_proj)")
        
        # ====================================================================
        # BFS EXPANSION (Unchanged from BFSNet)
        # ====================================================================
        if not self.training:
            # Inference: single greedy rollout
            if self._debug:
                print("\n" + "=" * 79)
                print("INFERENCE MODE (Greedy Rollout) - Language Model v1.1.0")
                print(f"Batch size: {B}, Sequence length: {seq_len}")
                print(f"Greedy Threshold: {self.greedy_threshold:.4f}")
                print("=" * 79)
            
            logits_flat, _, _ = self._rollout(h0, greedy=True)
            
            # Reshape output: [B*seq_len, vocab_size] → [B, seq_len, vocab_size]
            logits = logits_flat.view(B, seq_len, self.vocab_size)
            
            if self._debug:
                print("=" * 79 + "\n")
            
            return logits
        
        # Training: multiple rollouts
        if labels is None:
            raise ValueError("labels required during training for reward computation")
        
        if self._debug:
            print("\n" + "=" * 79)
            print(f"TRAINING MODE ({num_rollouts} Rollouts) - Language Model v1.1.0")
            print(f"Batch size: {B}, Sequence length: {seq_len}")
            print(f"Lambda Efficiency: {lambda_efficiency:.4f}")
            print("=" * 79)
        
        all_outputs = []
        all_log_probs = []
        all_node_counts = []
        
        for rollout_idx in range(num_rollouts):
            if self._debug:
                print(f"\n--- Rollout {rollout_idx + 1}/{num_rollouts} ---")
            
            out, log_p, nodes = self._rollout(h0, greedy=False)
            all_outputs.append(out)
            all_log_probs.append(log_p)
            all_node_counts.append(nodes)
        
        # Average outputs for loss computation: [B*seq_len, vocab_size]
        avg_outputs_flat = torch.stack(all_outputs).mean(dim=0)
        
        # =======================================================================
        # v1.1.0: Check averaged outputs for NaN
        # =======================================================================
        avg_outputs_flat = _check_for_nan(avg_outputs_flat, "avg_outputs_flat")
        
        # Compute rewards
        rewards = self._compute_rewards(
            all_outputs, all_node_counts, labels,
            lambda_efficiency, B, seq_len
        )
        
        # =======================================================================
        # v1.1.0: Check rewards for NaN
        # =======================================================================
        rewards = _check_for_nan(rewards, "rewards", replace_value=-1.0)
        
        # Compute policy loss
        policy_loss = self._compute_policy_loss(
            all_log_probs, rewards, beta_entropy
        )
        
        # Reshape output: [B*seq_len, vocab_size] → [B, seq_len, vocab_size]
        avg_outputs = avg_outputs_flat.view(B, seq_len, self.vocab_size)
        
        if self._debug:
            print(f"\nRewards: {rewards.tolist()}")
            print(f"Policy loss: {policy_loss.item():.6f}")
            print(f"Node counts: {all_node_counts}")
            print("=" * 79 + "\n")
        
        return avg_outputs, policy_loss, rewards, all_node_counts
    
    # ---------------------------- Generation ------------------------------- #
    
    @torch.no_grad()
    def generate(
        self,
        prompt: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Autoregressive text generation.
        
        Parameters
        ----------
        prompt : torch.Tensor
            Initial token IDs, shape [B, prompt_len] or [prompt_len].
        max_new_tokens : int, default=100
            Maximum number of tokens to generate.
        temperature : float, default=1.0
            Sampling temperature (higher = more random).
        top_k : int, optional
            If set, only sample from top-k tokens.
        top_p : float, optional
            If set, use nucleus sampling (sample from smallest set with prob >= p).
            
        Returns
        -------
        torch.Tensor
            Generated token IDs, shape [B, prompt_len + max_new_tokens].
        """
        self.eval()
        
        # Handle 1D input
        if prompt.dim() == 1:
            prompt = prompt.unsqueeze(0)  # [1, prompt_len]
        
        B = prompt.size(0)
        generated = prompt.clone()
        
        for _ in range(max_new_tokens):
            # Get logits for last position
            logits = self(generated)  # [B, seq_len, vocab_size]
            next_logits = logits[:, -1, :]  # [B, vocab_size]
            
            # Apply temperature
            if temperature != 1.0:
                next_logits = next_logits / temperature
            
            # Apply top-k filtering
            if top_k is not None:
                v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < v[:, [-1]]] = float('-inf')
            
            # Apply top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                next_logits[indices_to_remove] = float('-inf')
            
            # Sample next token
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # [B, 1]
            
            # Append to generated
            generated = torch.cat([generated, next_token], dim=1)
        
        return generated
    
    # ------------------------------- Summary ------------------------------- #
    
    def summary(self) -> str:
        """Generate human-readable model configuration summary."""
        lines = [
            "BoeNet v1.1.0 - Language Model (",
            f"  vocab_size={self.vocab_size}, embed_dim={self.embed_dim}, hidden_dim={self.hidden_dim},",
            f"  max_depth={self.max_depth}, max_children={self.max_children},",
            f"  greedy_threshold={self.greedy_threshold:.4f},",
            f"  sibling_embed={self.sibling_embed}, sibling_scale={self.sibling_scale:.4f},",
            f"  use_pruning={self.use_pruning}, pruning_mode='{self.pruning_mode}',",
            f"  pooling_mode='{self.pooling_mode}'",
            ")",
        ]
        return "\n".join(lines)
    
    def num_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# -------------------------------- Self-test -------------------------------- #

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    torch.manual_seed(42)
    
    # Test configuration
    B = 4           # Batch size
    seq_len = 32    # Sequence length
    vocab_size = 256  # Character-level vocabulary
    embed_dim = 64
    hidden_dim = 128
    
    logger.info("=" * 60)
    logger.info("BoeNet v1.1.0 Self-Test Suite (Language Model)")
    logger.info("=" * 60)
    
    # Test 1: Inference mode
    logger.info("\n[Test 1] Inference mode (greedy)")
    model = BoeNet(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        max_depth=2,
        max_children=3,
        greedy_threshold=0.42,
    )
    model.eval()
    x = torch.randint(0, vocab_size, (B, seq_len))
    logits = model(x)
    assert logits.shape == (B, seq_len, vocab_size), f"Expected {(B, seq_len, vocab_size)}, got {logits.shape}"
    logger.info(f"  Input: {x.shape}, Output: {logits.shape}")
    logger.info(f"  Parameters: {model.num_parameters():,}")
    logger.info("  ✓ Inference mode OK")
    
    # Test 2: Training mode with K=3 (the problematic case)
    logger.info("\n[Test 2] Training mode with K=3 (BFS tree expansion)")
    model.train()
    y = torch.randint(0, vocab_size, (B, seq_len))  # Target tokens
    outputs, policy_loss, rewards, node_counts = model(
        x, num_rollouts=3, lambda_efficiency=0.05,
        beta_entropy=0.01, labels=y
    )
    assert outputs.shape == (B, seq_len, vocab_size), f"Expected {(B, seq_len, vocab_size)}, got {outputs.shape}"
    assert policy_loss.requires_grad, "Policy loss should require grad"
    assert rewards.shape == (3,), f"Expected rewards shape (3,), got {rewards.shape}"
    assert len(node_counts) == 3, f"Expected 3 node counts, got {len(node_counts)}"
    logger.info(f"  Outputs: {outputs.shape}")
    logger.info(f"  Policy loss: {policy_loss.item():.4f}")
    logger.info(f"  Rewards: {[f'{r:.4f}' for r in rewards.tolist()]}")
    logger.info(f"  Node counts: {node_counts}")
    logger.info("  ✓ Training mode with K=3 OK")
    
    # Test 3: Gradient flow
    logger.info("\n[Test 3] Gradient flow")
    # Flatten outputs and labels for cross-entropy
    outputs_flat = outputs.view(-1, vocab_size)
    labels_flat = y.view(-1)
    classification_loss = F.cross_entropy(outputs_flat, labels_flat)
    total_loss = classification_loss + 0.5 * policy_loss
    total_loss.backward()
    
    embed_grad_norm = model.embedding.weight.grad.norm().item()
    policy_grad_norm = model.growth_policy.policy[0].weight.grad.norm().item()
    
    assert embed_grad_norm > 0, "Embedding should have gradients"
    assert policy_grad_norm > 0, "Policy should have gradients"
    
    logger.info(f"  embedding gradient norm: {embed_grad_norm:.6f}")
    logger.info(f"  growth_policy gradient norm: {policy_grad_norm:.6f}")
    logger.info("  ✓ Gradient flow OK")
    
    # Test 4: NaN detection (v1.1.0 feature)
    logger.info("\n[Test 4] NaN detection (v1.1.0 feature)")
    # Create a tensor with NaN and test the helper function
    test_tensor = torch.tensor([1.0, float('nan'), 3.0])
    fixed_tensor = _check_for_nan(test_tensor, "test_tensor", replace_value=0.0)
    assert not torch.isnan(fixed_tensor).any(), "NaN should be replaced"
    assert fixed_tensor[1].item() == 0.0, f"Expected 0.0, got {fixed_tensor[1].item()}"
    logger.info("  ✓ NaN detection OK")
    
    # Test 5: Probability clamping (v1.1.0 feature)
    logger.info("\n[Test 5] Probability clamping (v1.1.0 feature)")
    # Test with out-of-range probabilities
    bad_probs = torch.tensor([-0.1, 0.5, 1.1, float('nan')])
    clamped_probs = _safe_clamp_prob(bad_probs, "bad_probs")
    assert clamped_probs.min() >= PROB_CLAMP_MIN, f"Min should be >= {PROB_CLAMP_MIN}"
    assert clamped_probs.max() <= PROB_CLAMP_MAX, f"Max should be <= {PROB_CLAMP_MAX}"
    assert not torch.isnan(clamped_probs).any(), "No NaN after clamping"
    logger.info(f"  Bad probs: {bad_probs.tolist()}")
    logger.info(f"  Clamped: {clamped_probs.tolist()}")
    logger.info("  ✓ Probability clamping OK")
    
    # Test 6: Text generation
    logger.info("\n[Test 6] Text generation")
    model.eval()
    prompt = torch.randint(0, vocab_size, (1, 10))  # Start with 10 tokens
    generated = model.generate(prompt, max_new_tokens=20, temperature=0.8)
    assert generated.shape == (1, 30), f"Expected shape (1, 30), got {generated.shape}"
    logger.info(f"  Prompt: {prompt.shape} → Generated: {generated.shape}")
    logger.info("  ✓ Generation OK")
    
    # Test 7: Dense baseline (MLP mode, K=0)
    logger.info("\n[Test 7] Dense baseline (depth=0, K=0)")
    mlp = BoeNet(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        max_depth=0,
        max_children=0,
    )
    mlp.eval()
    logits_mlp = mlp(x)
    assert logits_mlp.shape == (B, seq_len, vocab_size)
    logger.info(f"  MLP output: {logits_mlp.shape}")
    logger.info("  ✓ Dense baseline OK")
    
    # Test 8: Debug mode
    logger.info("\n[Test 8] Debug mode")
    model_debug = BoeNet(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        max_depth=1,
        max_children=2,
        greedy_threshold=0.42,
    )
    model_debug.eval()
    model_debug._debug = True
    x_single = torch.randint(0, vocab_size, (1, 8))
    logger.info("  Enabling _debug flag...")
    logits_debug = model_debug(x_single)
    assert logits_debug.shape == (1, 8, vocab_size)
    logger.info("  ✓ Debug mode OK")
    
    logger.info("\n" + "=" * 60)
    logger.info("All self-tests passed!")
    logger.info(f"Model summary:\n{model.summary()}")
    logger.info("=" * 60)