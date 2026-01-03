#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
boenet/model.py (v2.4.0 - True BFS Language Model with Fixed GrowthPolicyNet)

BoeNet: True BFS-Inspired Language Model with REINFORCE Policy Gradients

v2.4.0 Critical Fix (2026-01-03) - Use GrowthPolicyNet from gating.py
----------------------------------------------------------------------
ISSUE FIXED:
  GrowthPolicyNet defined in model.py was outputting grow_prob=0.0000 for
  ALL inputs, causing trees to never expand except when epsilon-greedy
  forced expansion (~10% of the time).

ROOT CAUSE:
  The LayerNorm + ReLU architecture in the model.py version was killing
  all activations, producing extremely negative logits (-16 or lower)
  before the sigmoid, resulting in probability ≈ 0.

SOLUTION:
  - DELETED the broken GrowthPolicyNet class from model.py
  - IMPORT GrowthPolicyNet from boenet.utils.gating (v3.1.0)
  - The gating.py version uses a simpler, working architecture

v2.3.1 DIAGNOSTIC VERSION (2026-01-02) - Preserved
---------------------------------------------------
  - Diagnostic printing to identify expansion decisions
  - Shows when epsilon-greedy triggers vs policy decision
  - Displays tree structure at end of each rollout
  - Reports node counts per level

v2.3.0 Critical Fixes (2026-01-02)
----------------------------------
  1. Epsilon-greedy exploration forces tree expansion
  2. Log probability computed for actual action taken
  3. Added get_num_nodes_at_level() function for inference

v2.2.0: Epsilon-greedy exploration, neutral policy init
v2.1.0: Fixed in-place tensor operations for gradient computation
v2.0.0: True BFS level-by-level expansion with balanced binary trees

Author: BoeNet project
Version: 2.4.0
Date: 2026-01-03
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Literal
import math
import logging
import random  # For diagnostic rollout IDs

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# =============================================================================
# v2.4.0: IMPORT GrowthPolicyNet FROM gating.py (instead of defining here)
# =============================================================================
from boenet.utils.gating import GrowthPolicyNet

# =============================================================================
# NUMERICAL STABILITY CONSTANTS
# =============================================================================
PROB_CLAMP_MIN = 1e-7
PROB_CLAMP_MAX = 1.0 - 1e-7
LOG_PROB_CLAMP_MIN = -20.0
LOG_PROB_CLAMP_MAX = 0.0

# v2.2.0: Policy control constants
DEPTH_EMBED_SCALE = 0.01  # Reduced from 0.1 (10x reduction)
MIN_EXPLORE_PROB = 0.1     # Minimum expansion probability during training

# v2.3.1: Diagnostic mode - set to True to see all expansion decisions
DIAGNOSTIC_MODE = True  # <-- SET TO FALSE TO DISABLE VERBOSE OUTPUT


# =============================================================================
# BFS INDEXING FUNCTIONS
# =============================================================================

def get_parent_idx(i: int) -> Optional[int]:
    """Get the parent index of node i in a binary tree (BFS order)."""
    if i <= 0:
        return None
    return (i - 1) // 2


def get_left_child_idx(i: int) -> int:
    """Get the left child index of node i in a binary tree (BFS order)."""
    return 2 * i + 1


def get_right_child_idx(i: int) -> int:
    """Get the right child index of node i in a binary tree (BFS order)."""
    return 2 * i + 2


def get_level(i: int) -> int:
    """Get the level (depth) of node i in a binary tree (BFS order)."""
    if i < 0:
        raise ValueError(f"Node index must be non-negative, got {i}")
    return int(math.floor(math.log2(i + 1)))


def get_level_range(level: int) -> Tuple[int, int]:
    """
    Get the range of node indices at a given level.
    
    Parameters
    ----------
    level : int
        The level (depth) in the tree (0 = root).
        
    Returns
    -------
    Tuple[int, int]
        (start_index, end_index) where end_index is exclusive.
    """
    if level < 0:
        raise ValueError(f"Level must be non-negative, got {level}")
    start = (1 << level) - 1      # 2^level - 1
    end = (1 << (level + 1)) - 1  # 2^(level+1) - 1
    return start, end


def get_nodes_at_level(level: int) -> List[int]:
    """
    Get the list of node indices at a given level.
    
    Parameters
    ----------
    level : int
        The level (depth) in the tree (0 = root).
        
    Returns
    -------
    List[int]
        List of node indices at this level.
    """
    start, end = get_level_range(level)
    return list(range(start, end))


def get_num_nodes_at_level(level: int) -> int:
    """
    Get the number of nodes at a given level in a complete binary tree.
    
    Parameters
    ----------
    level : int
        The level (depth) in the tree (0 = root).
        
    Returns
    -------
    int
        Number of nodes at this level: 2^level.
        
    Examples
    --------
    >>> get_num_nodes_at_level(0)
    1
    >>> get_num_nodes_at_level(1)
    2
    >>> get_num_nodes_at_level(2)
    4
    >>> get_num_nodes_at_level(3)
    8
    """
    if level < 0:
        raise ValueError(f"Level must be non-negative, got {level}")
    return 1 << level  # 2^level


def get_total_nodes_up_to_level(level: int) -> int:
    """
    Get the total number of nodes from level 0 to the given level (inclusive).
    
    This is the total node count for a complete binary tree of given depth.
    
    Parameters
    ----------
    level : int
        The maximum level (depth) in the tree.
        
    Returns
    -------
    int
        Total nodes: 2^(level+1) - 1.
        
    Examples
    --------
    >>> get_total_nodes_up_to_level(0)
    1
    >>> get_total_nodes_up_to_level(1)
    3
    >>> get_total_nodes_up_to_level(2)
    7
    >>> get_total_nodes_up_to_level(3)
    15
    """
    if level < 0:
        return 0
    return (1 << (level + 1)) - 1  # 2^(level+1) - 1


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _check_for_nan(tensor: torch.Tensor, name: str, replace_value: float = 0.5) -> torch.Tensor:
    """Check tensor for NaN values and replace them if found."""
    if torch.isnan(tensor).any():
        nan_count = torch.isnan(tensor).sum().item()
        logger.warning(f"[NaN] {name}: {nan_count} NaNs, replacing with {replace_value}")
        tensor = torch.where(torch.isnan(tensor), torch.full_like(tensor, replace_value), tensor)
    return tensor


def _safe_clamp_prob(prob: torch.Tensor, name: str = "prob") -> torch.Tensor:
    """Safely clamp probability values to valid range."""
    prob = _check_for_nan(prob, name, replace_value=0.5)
    return prob.clamp(PROB_CLAMP_MIN, PROB_CLAMP_MAX)


def _safe_log_prob(prob: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """
    Compute log probability of a Bernoulli action given probability.
    
    Parameters
    ----------
    prob : torch.Tensor
        Probability of action=1 (expand).
    action : torch.Tensor
        The action taken (0 or 1).
        
    Returns
    -------
    torch.Tensor
        Log probability of the action: action * log(p) + (1-action) * log(1-p).
    """
    p = prob.clamp(PROB_CLAMP_MIN, PROB_CLAMP_MAX)
    log_p = torch.log(p).clamp(min=LOG_PROB_CLAMP_MIN)
    log_1_minus_p = torch.log(1.0 - p).clamp(min=LOG_PROB_CLAMP_MIN)
    log_prob = action * log_p + (1.0 - action) * log_1_minus_p
    return log_prob.clamp(min=LOG_PROB_CLAMP_MIN, max=LOG_PROB_CLAMP_MAX)


# =============================================================================
# PLACEHOLDER CLASSES (for compatibility)
# =============================================================================

class ScalarGate(nn.Module):
    """Simple scalar gate for pruning."""
    
    def __init__(self, hidden_dim: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, 1, bias=bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.linear(x))


class ThresholdPruner:
    """Threshold-based pruner for node removal."""
    
    def __init__(self, mode: str = "l2", threshold: float = 1e-3):
        self.mode = mode
        self.threshold = threshold


# =============================================================================
# BOENET MODEL (v2.4.0 - Uses GrowthPolicyNet from gating.py)
# =============================================================================

class BoeNet(nn.Module):
    """
    True BFS Language Model with REINFORCE Policy Gradients (v2.4.0).
    
    v2.4.0 KEY CHANGE:
    - GrowthPolicyNet is now imported from boenet.utils.gating
    - The previous model.py version was broken (output prob=0.0000)
    - The gating.py version has a working architecture
    
    v2.3.1 DIAGNOSTIC VERSION (Preserved):
    - Prints detailed information about every expansion decision
    - Helps identify why trees are not expanding during training
    - Set DIAGNOSTIC_MODE = False at top of file to disable
    
    v2.3.0 KEY CHANGES:
    - FIXED: Epsilon-greedy now actually forces expansion
    - FIXED: Log probability computed for actual action taken, not just policy
    - ADDED: get_num_nodes_at_level() function for inference
    - ADDED: Debug logging for expansion decisions
    
    v2.2.0 FEATURES:
    - Epsilon-greedy exploration during training ensures tree expansion
    - Policy initialization produces neutral (~0.5) initial probs
    - Reduced depth embedding influence (content dominates)
    - Per-position stochastic decisions, not just level-mean
    """
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        max_depth: int = 2,
        max_children: int = 2,
        greedy_threshold: float = 0.5,
        min_explore_prob: float = MIN_EXPLORE_PROB,
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
        
        # Handle legacy pooling argument
        if pooling is not None:
            pooling_mode = pooling
        
        # Store configuration
        self.vocab_size = int(vocab_size)
        self.embed_dim = int(embed_dim)
        self.hidden_dim = int(hidden_dim)
        self.max_depth = max(0, int(max_depth))
        self.max_children = max(0, int(max_children))
        self.eps = float(eps)
        self.max_nodes = get_total_nodes_up_to_level(self.max_depth) if self.max_depth >= 0 else 1
        
        # Policy configuration
        self.greedy_threshold = float(greedy_threshold)
        self.min_explore_prob = float(min_explore_prob)
        self._debug = DIAGNOSTIC_MODE  # v2.3.1: Controlled by global flag
        
        # v2.3.0: Track expansion statistics
        self._expansion_stats = {
            'forced_expansions': 0,
            'policy_expansions': 0,
            'policy_stops': 0,
            'total_decisions': 0,
        }
        
        # v2.3.1: Track per-epoch statistics for reporting
        self._epoch_rollout_count = 0
        self._epoch_expanded_count = 0
        
        # Input layers
        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        
        self.embed_proj = nn.Linear(self.embed_dim, self.hidden_dim)
        nn.init.kaiming_uniform_(self.embed_proj.weight, a=math.sqrt(5))
        nn.init.zeros_(self.embed_proj.bias)
        
        # BFS tree layers
        self.child_fc = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        nn.init.kaiming_uniform_(self.child_fc.weight, a=math.sqrt(5))
        nn.init.zeros_(self.child_fc.bias)
        
        # v2.4.0: Growth policy network - IMPORTED from boenet.utils.gating
        # This replaces the broken version that was defined in this file
        self.growth_policy = GrowthPolicyNet(hidden_dim=self.hidden_dim, max_depth=self.max_depth)
        
        # Sibling embeddings
        self.sibling_embed = bool(sibling_embed) and (self.max_children > 0)
        if self.sibling_embed:
            self.sibling_embeddings = nn.Embedding(2, self.hidden_dim)
            nn.init.normal_(self.sibling_embeddings.weight, mean=0.0, std=0.02)
        self.sibling_scale = 1.0 / math.sqrt(self.hidden_dim) if sibling_scale is None else float(sibling_scale)
        
        # v2.2.0: REDUCED depth embedding scale
        self.depth_embed_scale = DEPTH_EMBED_SCALE
        self.depth_embeddings = nn.Embedding(self.max_depth + 1, self.hidden_dim)
        nn.init.normal_(self.depth_embeddings.weight, mean=0.0, std=0.01)
        
        # Pruning (optional)
        self.use_pruning = bool(use_pruning)
        self.pruning_mode = pruning_mode
        self.pruning_threshold = float(pruning_threshold)
        self.prune_gate = ScalarGate(self.hidden_dim) if self.use_pruning and pruning_mode == "learned" else None
        self.pruner = ThresholdPruner(threshold=self.pruning_threshold) if self.use_pruning and pruning_mode == "threshold" else None
        
        # Pooling configuration
        self.pooling_mode = pooling_mode
        if self.pooling_mode == "learned":
            self._pool_log_p = nn.Parameter(torch.log(torch.expm1(torch.tensor(1.0))))
        else:
            self.register_parameter("_pool_log_p", None)
        
        # Output projection
        self.output_fc = nn.Linear(self.hidden_dim, self.vocab_size)
        nn.init.kaiming_uniform_(self.output_fc.weight, a=math.sqrt(5))
        nn.init.zeros_(self.output_fc.bias)
        
        # v2.4.0: Print model configuration with version update
        print(f"\n[BoeNet v2.4.0] Model created (uses GrowthPolicyNet from gating.py):")
        print(f"  max_depth={self.max_depth}, max_nodes={self.max_nodes}")
        print(f"  min_explore_prob={self.min_explore_prob} ({self.min_explore_prob*100:.0f}% forced expansion)")
        print(f"  DIAGNOSTIC_MODE={DIAGNOSTIC_MODE}")
        print()
    
    def _pool(self, agg_sum: torch.Tensor, agg_count: torch.Tensor) -> torch.Tensor:
        """Pool aggregated node representations."""
        if self.pooling_mode == "sum":
            return agg_sum
        if self.pooling_mode == "mean":
            return agg_sum / agg_count.clamp_min(self.eps)
        if self.pooling_mode == "learned":
            p = F.softplus(self._pool_log_p) if self._pool_log_p is not None else 1.0
            return agg_sum / torch.pow(agg_count.clamp_min(self.eps), p)
        raise ValueError(f"Unknown pooling_mode: {self.pooling_mode}")
    
    def reset_expansion_stats(self) -> None:
        """Reset expansion statistics (call at start of epoch)."""
        self._expansion_stats = {
            'forced_expansions': 0,
            'policy_expansions': 0,
            'policy_stops': 0,
            'total_decisions': 0,
        }
        self._epoch_rollout_count = 0
        self._epoch_expanded_count = 0
    
    def get_expansion_stats(self) -> Dict[str, int]:
        """Get expansion statistics for monitoring."""
        stats = self._expansion_stats.copy()
        stats['epoch_rollouts'] = self._epoch_rollout_count
        stats['epoch_expanded'] = self._epoch_expanded_count
        return stats
    
    def _true_bfs_rollout(
        self, h0: torch.Tensor, greedy: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], int]:
        """
        Execute TRUE BFS rollout with v2.4.0 fixed GrowthPolicyNet.
        
        v2.4.0 KEY CHANGE:
        - Uses GrowthPolicyNet from gating.py which outputs valid probabilities
        - The previous version was outputting prob=0.0000 always
        
        v2.3.1 DIAGNOSTIC VERSION:
        - Prints every expansion decision with rollout ID
        - Shows exactly when epsilon-greedy triggers vs policy decision
        - Displays tree structure at end of each rollout
        
        Parameters
        ----------
        h0 : torch.Tensor
            Initial hidden states of shape [N, hidden_dim].
        greedy : bool
            If True, use greedy decisions (for inference).
            If False, use stochastic sampling with epsilon-greedy (for training).
            
        Returns
        -------
        Tuple[torch.Tensor, Optional[torch.Tensor], int]
            (output_logits, log_probs, total_node_count)
        """
        N = h0.size(0)
        device = h0.device
        dtype = h0.dtype
        
        # v2.3.1: Generate rollout ID for tracing
        rollout_id = random.randint(1000, 9999)
        self._epoch_rollout_count += 1
        
        # Validate input
        h0 = _check_for_nan(h0, "h0")
        
        # Track nodes at each level
        # level_nodes[level] = list of node tensors at that level
        level_nodes: List[List[torch.Tensor]] = [[h0]]
        log_probs_list: List[torch.Tensor] = []
        current_depth = 0
        did_expand_at_all = False  # v2.3.1: Track if any expansion happened
        
        # v2.3.1: Print rollout start
        if self._debug:
            print(f"[v2.4.0][R{rollout_id}] START rollout: N={N}, max_depth={self.max_depth}, greedy={greedy}")
        
        # Iterate through levels (0 to max_depth-1, deciding whether to create level+1)
        for level in range(self.max_depth):
            current_level_nodes = level_nodes[level]
            num_nodes_at_level = len(current_level_nodes)
            
            if num_nodes_at_level == 0:
                # No nodes at this level, cannot expand further
                if self._debug:
                    print(f"[v2.4.0][R{rollout_id}] Level {level}: NO NODES - stopping")
                break
            
            # Aggregate all nodes at this level (mean pooling)
            level_stack = torch.stack(current_level_nodes, dim=0)  # [num_nodes, N, hidden_dim]
            level_hidden = level_stack.mean(dim=0)  # [N, hidden_dim]
            
            # Add depth embedding with reduced scale (v2.2.0)
            if level < self.max_depth:
                depth_emb = self.depth_embeddings.weight[level]
                level_hidden = level_hidden + self.depth_embed_scale * depth_emb.unsqueeze(0)
            
            # v2.4.0: Get grow probability from policy network (imported from gating.py)
            # The gating.py version returns shape [N], not [N, 1]
            grow_prob_raw = self.growth_policy(level_hidden, level)
            grow_prob = _safe_clamp_prob(grow_prob_raw, f"grow_prob_level_{level}")
            grow_prob_mean = grow_prob.mean()  # Scalar for level decision
            
            if greedy:
                # ============================================================
                # INFERENCE MODE: Deterministic decision based on threshold
                # ============================================================
                expand = (grow_prob_mean.item() >= self.greedy_threshold)
                if self._debug:
                    print(f"[v2.4.0][R{rollout_id}] Level {level}: GREEDY mode, prob={grow_prob_mean.item():.4f}, threshold={self.greedy_threshold}, expand={expand}")
            else:
                # ============================================================
                # TRAINING MODE: Stochastic with epsilon-greedy exploration
                # ============================================================
                self._expansion_stats['total_decisions'] += 1
                
                # Roll for exploration FIRST
                explore_random = torch.rand(1, device=device).item()
                
                # v2.3.1 DIAGNOSTIC: Print the random roll
                if self._debug:
                    print(f"[v2.4.0][R{rollout_id}] Level {level}: explore_random={explore_random:.4f}, min_explore_prob={self.min_explore_prob}")
                
                if explore_random < self.min_explore_prob:
                    # ========================================================
                    # EPSILON-GREEDY EXPLORATION: Force expansion
                    # ========================================================
                    expand = True
                    action_taken = torch.ones(1, device=device, dtype=dtype)
                    self._expansion_stats['forced_expansions'] += 1
                    
                    if self._debug:
                        print(f"[v2.4.0][R{rollout_id}] Level {level}: *** FORCED EXPANSION *** (rand={explore_random:.4f} < {self.min_explore_prob})")
                else:
                    # ========================================================
                    # EXPLOITATION: Sample from policy
                    # ========================================================
                    expand_prob = grow_prob_mean.unsqueeze(0)
                    action_taken = torch.bernoulli(expand_prob)
                    expand = action_taken.bool().item()
                    
                    if expand:
                        self._expansion_stats['policy_expansions'] += 1
                        if self._debug:
                            print(f"[v2.4.0][R{rollout_id}] Level {level}: Policy EXPAND (sampled 1 from Bernoulli(p={grow_prob_mean.item():.4f}))")
                    else:
                        self._expansion_stats['policy_stops'] += 1
                        if self._debug:
                            print(f"[v2.4.0][R{rollout_id}] Level {level}: Policy STOP (sampled 0 from Bernoulli(p={grow_prob_mean.item():.4f}))")
                
                # ============================================================
                # CRITICAL: Compute log_prob of the ACTION TAKEN
                # ============================================================
                # This is the REINFORCE log probability:
                # - If action_taken=1: log_prob = log(grow_prob_mean)
                # - If action_taken=0: log_prob = log(1 - grow_prob_mean)
                log_p = _safe_log_prob(grow_prob_mean.unsqueeze(0), action_taken)
                log_probs_list.append(log_p)
                
                if self._debug:
                    print(f"[v2.4.0][R{rollout_id}] Level {level}: action_taken={action_taken.item():.0f}, grow_prob={grow_prob_mean.item():.4f}, log_p={log_p.item():.4f}")
            
            # ================================================================
            # EXPANSION DECISION: If not expanding, stop here
            # ================================================================
            if not expand:
                if self._debug:
                    print(f"[v2.4.0][R{rollout_id}] Level {level}: NOT EXPANDING - stopping tree growth")
                break
            
            # Mark that expansion happened
            did_expand_at_all = True
            
            # ================================================================
            # EXPAND: Create children for next level
            # ================================================================
            next_level_nodes: List[torch.Tensor] = []
            
            for node_idx, node_h in enumerate(current_level_nodes):
                # Create two children via linear projection
                children_h = self.child_fc(node_h)  # [N, hidden_dim * 2]
                left_h = F.relu(children_h[:, :self.hidden_dim])
                right_h = F.relu(children_h[:, self.hidden_dim:])
                
                # Add sibling embeddings (distinguishes left from right)
                if self.sibling_embed:
                    left_emb = self.sibling_embeddings.weight[0]
                    right_emb = self.sibling_embeddings.weight[1]
                    left_h = left_h + self.sibling_scale * left_emb.unsqueeze(0)
                    right_h = right_h + self.sibling_scale * right_emb.unsqueeze(0)
                
                # Validate children (check for NaN)
                left_h = _check_for_nan(left_h, f"left_child_level_{level}_node_{node_idx}")
                right_h = _check_for_nan(right_h, f"right_child_level_{level}_node_{node_idx}")
                
                next_level_nodes.append(left_h)
                next_level_nodes.append(right_h)
            
            # Verify children were created correctly
            expected_children = 2 * num_nodes_at_level
            if len(next_level_nodes) != expected_children:
                print(f"[v2.4.0][R{rollout_id}] ERROR: Expected {expected_children} children, got {len(next_level_nodes)}")
            
            level_nodes.append(next_level_nodes)
            current_depth = level + 1
            
            if self._debug:
                print(f"[v2.4.0][R{rollout_id}] Level {level}: EXPANDED -> {len(next_level_nodes)} children at level {level + 1}")
        
        # Track if this rollout had any expansion
        if did_expand_at_all:
            self._epoch_expanded_count += 1
        
        # ================================================================
        # POOLING: Aggregate all nodes across all levels
        # ================================================================
        all_nodes: List[torch.Tensor] = []
        for level_idx, level_node_list in enumerate(level_nodes):
            all_nodes.extend(level_node_list)
        
        total_nodes = len(all_nodes)
        
        # v2.3.1: Print final tree structure
        if self._debug:
            nodes_per_level = [len(level_nodes[i]) for i in range(len(level_nodes))]
            print(f"[v2.4.0][R{rollout_id}] FINAL: depth={current_depth}, total_nodes={total_nodes}, nodes_per_level={nodes_per_level}")
        
        if total_nodes == 0:
            # Fallback to initial hidden state (should never happen)
            print(f"[v2.4.0][R{rollout_id}] WARNING: total_nodes=0, using h0 as fallback")
            pooled = h0
        else:
            all_nodes_stack = torch.stack(all_nodes, dim=0)  # [total_nodes, N, hidden_dim]
            agg_sum = all_nodes_stack.sum(dim=0)  # [N, hidden_dim]
            agg_count = torch.full((N, 1), float(total_nodes), device=device, dtype=dtype)
            pooled = self._pool(agg_sum, agg_count)
        
        # Validate and project to vocabulary
        pooled = _check_for_nan(pooled, "pooled")
        output = self.output_fc(pooled)
        output = _check_for_nan(output, "output")
        
        # Concatenate log_probs from all level decisions
        if log_probs_list:
            log_probs = torch.cat(log_probs_list)
        else:
            log_probs = None
        
        return output, log_probs, total_nodes
    
    def _compute_rewards(
        self, outputs: List[torch.Tensor], node_counts: List[int],
        labels: torch.Tensor, lambda_efficiency: float, batch_size: int, seq_len: int,
    ) -> torch.Tensor:
        """
        Compute rewards for REINFORCE policy gradient.
        
        Parameters
        ----------
        outputs : List[torch.Tensor]
            List of output logits from each rollout.
        node_counts : List[int]
            List of node counts from each rollout.
        labels : torch.Tensor
            Target labels.
        lambda_efficiency : float
            Weight for efficiency penalty.
        batch_size : int
            Batch size.
        seq_len : int
            Sequence length.
            
        Returns
        -------
        torch.Tensor
            Reward tensor of shape [num_rollouts].
        """
        num_positions = batch_size * seq_len
        labels_flat = labels.view(-1)
        rewards = []
        
        for out, nodes in zip(outputs, node_counts):
            # Language modeling reward (negative cross-entropy)
            ce_loss = F.cross_entropy(out, labels_flat, reduction='mean')
            if torch.isnan(ce_loss) or torch.isinf(ce_loss):
                ce_loss = torch.tensor(10.0, device=out.device, dtype=out.dtype)
            
            reward_accuracy = -ce_loss
            
            # Efficiency penalty (penalize using more nodes)
            nodes_per_position = nodes / num_positions
            efficiency_ratio = min(max(nodes_per_position / self.max_nodes, 0.0), 1.0)
            efficiency_penalty = lambda_efficiency * efficiency_ratio
            
            # Combined reward (scaled for stability)
            reward = (reward_accuracy - efficiency_penalty) / 5.0
            rewards.append(reward)
        
        return torch.stack(rewards)
    
    def _compute_policy_loss(
        self, log_probs: List[Optional[torch.Tensor]], rewards: torch.Tensor, beta_entropy: float,
    ) -> torch.Tensor:
        """
        Compute policy loss for REINFORCE with entropy regularization.
        
        Parameters
        ----------
        log_probs : List[Optional[torch.Tensor]]
            Log probabilities of actions from each rollout.
        rewards : torch.Tensor
            Rewards from each rollout.
        beta_entropy : float
            Entropy regularization coefficient.
            
        Returns
        -------
        torch.Tensor
            Policy loss scalar.
        """
        valid_log_probs = [lp for lp in log_probs if lp is not None]
        
        if len(valid_log_probs) == 0:
            return torch.tensor(0.0, device=rewards.device, requires_grad=True)
        
        # Baseline for variance reduction
        baseline = rewards.mean()
        
        policy_loss = torch.tensor(0.0, device=rewards.device, dtype=rewards.dtype)
        total_entropy = torch.tensor(0.0, device=rewards.device, dtype=rewards.dtype)
        total_decisions = 0
        
        for log_p, reward in zip(valid_log_probs, rewards):
            if torch.isnan(log_p).any():
                continue
            
            # Advantage with clipping for stability
            advantage = (reward - baseline).clamp(-2.0, 2.0)
            
            # REINFORCE: -log_prob * advantage
            policy_loss = policy_loss - (log_p * advantage).sum()
            
            # Entropy bonus for exploration
            p = torch.exp(log_p).clamp(PROB_CLAMP_MIN, PROB_CLAMP_MAX)
            entropy = -(p * torch.log(p) + (1 - p) * torch.log(1 - p))
            entropy = _check_for_nan(entropy, "entropy", replace_value=0.0)
            total_entropy = total_entropy + entropy.sum()
            total_decisions += log_p.numel()
        
        # Add entropy regularization (encourages exploration)
        if total_decisions > 0:
            policy_loss = policy_loss - beta_entropy * total_entropy
        
        # Handle NaN in final loss
        if torch.isnan(policy_loss):
            return torch.tensor(0.0, device=rewards.device, requires_grad=True)
        
        return policy_loss
    
    def forward(
        self, x: torch.Tensor, num_rollouts: int = 3, lambda_efficiency: float = 0.05,
        beta_entropy: float = 0.01, labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
        """
        Forward pass with training or inference mode.
        
        Parameters
        ----------
        x : torch.Tensor
            Input token IDs of shape [B, seq_len].
        num_rollouts : int
            Number of policy rollouts for training.
        lambda_efficiency : float
            Efficiency penalty weight.
        beta_entropy : float
            Entropy regularization weight.
        labels : Optional[torch.Tensor]
            Target labels for training.
            
        Returns
        -------
        In inference mode (self.training=False):
            torch.Tensor: Logits of shape [B, seq_len, vocab_size].
        In training mode (self.training=True):
            Tuple of (logits, policy_loss, rewards, node_counts).
        """
        B, seq_len = x.shape
        
        # Embed and project input
        embedded = self.embedding(x)  # [B, seq_len, embed_dim]
        embedded_flat = embedded.view(B * seq_len, self.embed_dim)
        h0 = F.relu(self.embed_proj(embedded_flat))  # [B * seq_len, hidden_dim]
        h0 = _check_for_nan(h0, "h0 (after embed_proj)")
        
        # Inference mode: single greedy rollout
        if not self.training:
            logits_flat, _, _ = self._true_bfs_rollout(h0, greedy=True)
            return logits_flat.view(B, seq_len, self.vocab_size)
        
        # Training mode: multiple stochastic rollouts
        if labels is None:
            raise ValueError("labels required during training")
        
        all_outputs: List[torch.Tensor] = []
        all_log_probs: List[Optional[torch.Tensor]] = []
        all_node_counts: List[int] = []
        
        for rollout_idx in range(num_rollouts):
            out, log_p, nodes = self._true_bfs_rollout(h0, greedy=False)
            all_outputs.append(out)
            all_log_probs.append(log_p)
            all_node_counts.append(nodes)
            
            if self._debug:
                print(f"[v2.4.0] Rollout {rollout_idx}: nodes={nodes}, log_p_shape={log_p.shape if log_p is not None else None}")
        
        # Average outputs across rollouts
        avg_outputs_flat = torch.stack(all_outputs).mean(dim=0)
        avg_outputs_flat = _check_for_nan(avg_outputs_flat, "avg_outputs_flat")
        
        # Compute rewards and policy loss
        rewards = self._compute_rewards(all_outputs, all_node_counts, labels, lambda_efficiency, B, seq_len)
        rewards = _check_for_nan(rewards, "rewards", replace_value=-1.0)
        policy_loss = self._compute_policy_loss(all_log_probs, rewards, beta_entropy)
        
        return avg_outputs_flat.view(B, seq_len, self.vocab_size), policy_loss, rewards, all_node_counts
    
    @torch.no_grad()
    def generate(
        self, prompt: torch.Tensor, max_new_tokens: int = 100, temperature: float = 1.0,
        top_k: Optional[int] = None, top_p: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Generate text autoregressively.
        
        Parameters
        ----------
        prompt : torch.Tensor
            Starting token IDs of shape [seq_len] or [1, seq_len].
        max_new_tokens : int
            Maximum number of tokens to generate.
        temperature : float
            Sampling temperature.
        top_k : Optional[int]
            Top-k sampling parameter.
        top_p : Optional[float]
            Nucleus sampling parameter.
            
        Returns
        -------
        torch.Tensor
            Generated token IDs including prompt.
        """
        self.eval()
        if prompt.dim() == 1:
            prompt = prompt.unsqueeze(0)
        generated = prompt.clone()
        
        for _ in range(max_new_tokens):
            logits = self(generated)
            next_logits = logits[:, -1, :]
            
            # Apply temperature
            if temperature != 1.0:
                next_logits = next_logits / temperature
            
            # Apply top-k
            if top_k is not None:
                v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < v[:, [-1]]] = float('-inf')
            
            # Sample
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)
        
        return generated
    
    def summary(self) -> str:
        """Return a summary string of the model configuration."""
        return (
            f"BoeNet v2.4.0 - True BFS Language Model (Fixed GrowthPolicyNet)\n"
            f"  vocab_size={self.vocab_size}, embed_dim={self.embed_dim}, hidden_dim={self.hidden_dim}\n"
            f"  max_depth={self.max_depth}, max_nodes={self.max_nodes}\n"
            f"  greedy_threshold={self.greedy_threshold:.4f}, min_explore_prob={self.min_explore_prob:.4f}\n"
            f"  depth_embed_scale={self.depth_embed_scale} (reduced from 0.1)\n"
            f"  DIAGNOSTIC_MODE={DIAGNOSTIC_MODE}\n"
            f"  v2.4.0: GrowthPolicyNet imported from boenet.utils.gating (fixed version)"
        )
    
    def num_parameters(self) -> int:
        """Return the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# SELF-TEST
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    torch.manual_seed(42)
    
    B, seq_len, vocab_size, embed_dim, hidden_dim = 2, 8, 256, 64, 128  # Smaller for testing
    
    print("=" * 70)
    print("BoeNet v2.4.0 Self-Test (Fixed GrowthPolicyNet from gating.py)")
    print("=" * 70)
    
    # Test 1: Basic model creation
    print("\n[Test 1] Creating model...")
    model = BoeNet(vocab_size=vocab_size, embed_dim=embed_dim, hidden_dim=hidden_dim, max_depth=2)
    print(f"\n{model.summary()}")
    print(f"Parameters: {model.num_parameters():,}")
    
    # Test 2: Verify get_num_nodes_at_level function
    print("\n[Test 2] get_num_nodes_at_level function:")
    for level in range(5):
        num_nodes = get_num_nodes_at_level(level)
        print(f"  Level {level}: {num_nodes} nodes")
    
    # Test 3: Verify GrowthPolicyNet outputs valid probabilities
    print("\n[Test 3] GrowthPolicyNet probability check:")
    test_hidden = torch.randn(16, hidden_dim)
    for level in range(3):
        prob = model.growth_policy(test_hidden, level)
        print(f"  Level {level}: shape={prob.shape}, mean={prob.mean().item():.4f}, min={prob.min().item():.4f}, max={prob.max().item():.4f}")
        
        # Verify probabilities are not all zeros
        if prob.mean().item() > 0.1:
            print(f"    ✓ Probabilities are reasonable (not stuck at 0)")
        else:
            print(f"    ⚠ WARNING: Probabilities may be too low")
    
    # Test 4: Training forward pass with expansion tracking
    print("\n[Test 4] Training forward pass (watch for expansion decisions):")
    model.train()
    model.reset_expansion_stats()
    
    x = torch.randint(0, vocab_size, (B, seq_len))
    y = torch.randint(0, vocab_size, (B, seq_len))
    
    print("\n--- Starting 3 rollouts ---")
    outputs, policy_loss, rewards, node_counts = model(x, num_rollouts=3, labels=y)
    print("--- Finished rollouts ---\n")
    
    stats = model.get_expansion_stats()
    print(f"  Output shape: {outputs.shape}")
    print(f"  Node counts: {node_counts}")
    print(f"  Policy loss: {policy_loss.item():.6f}")
    print(f"\n  Expansion Statistics:")
    print(f"    Total decisions: {stats['total_decisions']}")
    print(f"    Forced expansions: {stats['forced_expansions']}")
    print(f"    Policy expansions: {stats['policy_expansions']}")
    print(f"    Policy stops: {stats['policy_stops']}")
    print(f"    Epoch rollouts: {stats['epoch_rollouts']}")
    print(f"    Epoch expanded: {stats['epoch_expanded']}")
    
    # Verify results
    print("\n[Verification]")
    if stats['total_decisions'] > 0:
        print(f"  ✓ Decisions were made ({stats['total_decisions']} total)")
    else:
        print(f"  ⚠ No decisions made!")
    
    if stats['forced_expansions'] > 0 or stats['policy_expansions'] > 0:
        print(f"  ✓ Some expansions occurred!")
    else:
        print(f"  ⚠ No expansions at all!")
    
    if any(n > B * seq_len for n in node_counts):
        print(f"  ✓ Trees expanded beyond root (node_counts > {B * seq_len})")
    else:
        print(f"  ⚠ Trees stayed at root only")
    
    # Test 5: Gradient flow through policy
    print("\n[Test 5] Gradient flow check:")
    model.zero_grad()
    loss = policy_loss + outputs.sum() * 0.001
    loss.backward()
    
    policy_grad_norm = 0.0
    for name, param in model.named_parameters():
        if 'growth_policy' in name and param.grad is not None:
            policy_grad_norm += param.grad.norm().item() ** 2
    policy_grad_norm = policy_grad_norm ** 0.5
    print(f"  Policy gradient norm: {policy_grad_norm:.6f}")
    
    if policy_grad_norm > 0:
        print(f"  ✓ Gradients are flowing through policy network")
    else:
        print(f"  ⚠ No gradients in policy network!")
    
    print("\n" + "=" * 70)
    print("Self-test complete!")
    print("=" * 70)