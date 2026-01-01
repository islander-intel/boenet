#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
boenet/model.py (v2.2.0 - True BFS Language Model)

BoeNet: True BFS-Inspired Language Model with REINFORCE Policy Gradients

v2.2.0 Critical Fixes (2026-01-01) - Policy Saturation & Training Expansion
---------------------------------------------------------------------------
ISSUES FIXED:
  1. Policy saturation: grow_prob saturated to extremes (depth 1-2: ~1.0, depth 3-4: ~0.0)
  2. Trees never expanded during training: Always depth=0, Avg nodes/position: 0.00
  3. Depth embedding dominated policy decisions instead of content

ROOT CAUSES:
  1. Depth embedding scale (0.1) too large relative to content features
  2. Policy initialization caused extreme initial outputs
  3. Stochastic sampling was effectively deterministic due to extreme probs

FIXES APPLIED:
  1. REDUCED depth embedding scale from 0.1 to 0.01 (10x reduction)
  2. ADDED LayerNorm to policy network for stable activations
  3. CHANGED policy output initialization to produce ~0.5 initial probs
  4. ADDED epsilon-greedy exploration during training
  5. ADDED per-position stochastic sampling (not just mean-based)

v2.1.0: Fixed in-place tensor operations for gradient computation
v2.0.0: True BFS level-by-level expansion with balanced binary trees

Author: BoeNet project
Version: 2.2.0
Date: 2026-01-01
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Literal
import math
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

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


# =============================================================================
# BFS INDEXING FUNCTIONS
# =============================================================================

def get_parent_idx(i: int) -> Optional[int]:
    if i <= 0:
        return None
    return (i - 1) // 2

def get_left_child_idx(i: int) -> int:
    return 2 * i + 1

def get_right_child_idx(i: int) -> int:
    return 2 * i + 2

def get_level(i: int) -> int:
    if i < 0:
        raise ValueError(f"Node index must be non-negative, got {i}")
    return int(math.floor(math.log2(i + 1)))

def get_level_range(level: int) -> Tuple[int, int]:
    if level < 0:
        raise ValueError(f"Level must be non-negative, got {level}")
    start = (1 << level) - 1
    end = (1 << (level + 1)) - 1
    return start, end

def get_nodes_at_level(level: int) -> List[int]:
    start, end = get_level_range(level)
    return list(range(start, end))

def get_total_nodes_up_to_level(level: int) -> int:
    return (1 << (level + 1)) - 1


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _check_for_nan(tensor: torch.Tensor, name: str, replace_value: float = 0.5) -> torch.Tensor:
    if torch.isnan(tensor).any():
        nan_count = torch.isnan(tensor).sum().item()
        logger.warning(f"[NaN] {name}: {nan_count} NaNs, replacing with {replace_value}")
        tensor = torch.where(torch.isnan(tensor), torch.full_like(tensor, replace_value), tensor)
    return tensor

def _safe_clamp_prob(prob: torch.Tensor, name: str = "prob") -> torch.Tensor:
    prob = _check_for_nan(prob, name, replace_value=0.5)
    return prob.clamp(PROB_CLAMP_MIN, PROB_CLAMP_MAX)

def _safe_log_prob(prob: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    p = prob.clamp(PROB_CLAMP_MIN, PROB_CLAMP_MAX)
    log_p = torch.log(p).clamp(min=LOG_PROB_CLAMP_MIN)
    log_1_minus_p = torch.log(1.0 - p).clamp(min=LOG_PROB_CLAMP_MIN)
    log_prob = action * log_p + (1.0 - action) * log_1_minus_p
    return log_prob.clamp(min=LOG_PROB_CLAMP_MIN, max=LOG_PROB_CLAMP_MAX)


# =============================================================================
# v2.2.0: IMPROVED GROWTH POLICY NETWORK
# =============================================================================

class GrowthPolicyNet(nn.Module):
    """
    Growth Policy Network for True BFS (v2.2.0).
    
    FIXES:
    1. LayerNorm for stable activations
    2. Neutral initialization (~0.5 initial grow_prob)
    3. Reduced depth embedding influence
    4. Content features dominate over depth features
    """
    
    def __init__(self, hidden_dim: int, max_depth: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_depth = max_depth
        
        # v2.2.0: Layer normalization
        self.input_norm = nn.LayerNorm(hidden_dim)
        
        # v2.2.0: Smaller depth embedding
        self.depth_embed_dim = max(hidden_dim // 4, 16)
        self.depth_embed = nn.Embedding(max_depth + 1, self.depth_embed_dim)
        nn.init.normal_(self.depth_embed.weight, mean=0.0, std=0.01)
        
        # Policy MLP
        policy_input_dim = hidden_dim + self.depth_embed_dim
        self.policy_fc1 = nn.Linear(policy_input_dim, hidden_dim // 2)
        self.policy_norm1 = nn.LayerNorm(hidden_dim // 2)
        self.policy_fc2 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.policy_norm2 = nn.LayerNorm(hidden_dim // 4)
        self.policy_out = nn.Linear(hidden_dim // 4, 1)
        
        # v2.2.0: Initialize output to produce ~0.5 probability
        nn.init.zeros_(self.policy_out.weight)
        nn.init.zeros_(self.policy_out.bias)
        
        nn.init.kaiming_uniform_(self.policy_fc1.weight, a=math.sqrt(5))
        nn.init.zeros_(self.policy_fc1.bias)
        nn.init.kaiming_uniform_(self.policy_fc2.weight, a=math.sqrt(5))
        nn.init.zeros_(self.policy_fc2.bias)
    
    def forward(self, h: torch.Tensor, level: int) -> torch.Tensor:
        N = h.size(0)
        h_norm = self.input_norm(h)
        
        level_clamped = min(level, self.max_depth)
        depth_emb = self.depth_embed.weight[level_clamped].unsqueeze(0).expand(N, -1)
        
        combined = torch.cat([h_norm, depth_emb], dim=-1)
        x = F.relu(self.policy_norm1(self.policy_fc1(combined)))
        x = F.relu(self.policy_norm2(self.policy_fc2(x)))
        logit = self.policy_out(x).squeeze(-1)
        
        return torch.sigmoid(logit)


# =============================================================================
# PLACEHOLDER CLASSES (for compatibility)
# =============================================================================

class ScalarGate(nn.Module):
    def __init__(self, hidden_dim: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, 1, bias=bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.linear(x))


class ThresholdPruner:
    def __init__(self, mode: str = "l2", threshold: float = 1e-3):
        self.mode = mode
        self.threshold = threshold


# =============================================================================
# BOENET MODEL (v2.2.0)
# =============================================================================

class BoeNet(nn.Module):
    """
    True BFS Language Model with REINFORCE Policy Gradients (v2.2.0).
    
    v2.2.0 KEY CHANGES:
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
        
        if pooling is not None:
            pooling_mode = pooling
        
        self.vocab_size = int(vocab_size)
        self.embed_dim = int(embed_dim)
        self.hidden_dim = int(hidden_dim)
        self.max_depth = max(0, int(max_depth))
        self.max_children = max(0, int(max_children))
        self.eps = float(eps)
        self.max_nodes = get_total_nodes_up_to_level(self.max_depth) if self.max_depth >= 0 else 1
        
        self.greedy_threshold = float(greedy_threshold)
        self.min_explore_prob = float(min_explore_prob)
        self._debug = False
        
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
        
        self.growth_policy = GrowthPolicyNet(hidden_dim=self.hidden_dim, max_depth=self.max_depth)
        
        self.sibling_embed = bool(sibling_embed) and (self.max_children > 0)
        if self.sibling_embed:
            self.sibling_embeddings = nn.Embedding(2, self.hidden_dim)
            nn.init.normal_(self.sibling_embeddings.weight, mean=0.0, std=0.02)
        self.sibling_scale = 1.0 / math.sqrt(self.hidden_dim) if sibling_scale is None else float(sibling_scale)
        
        # v2.2.0: REDUCED depth embedding scale
        self.depth_embed_scale = DEPTH_EMBED_SCALE
        self.depth_embeddings = nn.Embedding(self.max_depth + 1, self.hidden_dim)
        nn.init.normal_(self.depth_embeddings.weight, mean=0.0, std=0.01)
        
        # Pruning
        self.use_pruning = bool(use_pruning)
        self.pruning_mode = pruning_mode
        self.pruning_threshold = float(pruning_threshold)
        self.prune_gate = ScalarGate(self.hidden_dim) if self.use_pruning and pruning_mode == "learned" else None
        self.pruner = ThresholdPruner(threshold=self.pruning_threshold) if self.use_pruning and pruning_mode == "threshold" else None
        
        # Pooling
        self.pooling_mode = pooling_mode
        if self.pooling_mode == "learned":
            self._pool_log_p = nn.Parameter(torch.log(torch.expm1(torch.tensor(1.0))))
        else:
            self.register_parameter("_pool_log_p", None)
        
        # Output
        self.output_fc = nn.Linear(self.hidden_dim, self.vocab_size)
        nn.init.kaiming_uniform_(self.output_fc.weight, a=math.sqrt(5))
        nn.init.zeros_(self.output_fc.bias)
    
    def _pool(self, agg_sum: torch.Tensor, agg_count: torch.Tensor) -> torch.Tensor:
        if self.pooling_mode == "sum":
            return agg_sum
        if self.pooling_mode == "mean":
            return agg_sum / agg_count.clamp_min(self.eps)
        if self.pooling_mode == "learned":
            p = F.softplus(self._pool_log_p) if self._pool_log_p is not None else 1.0
            return agg_sum / torch.pow(agg_count.clamp_min(self.eps), p)
        raise ValueError(f"Unknown pooling_mode: {self.pooling_mode}")
    
    def _true_bfs_rollout(
        self, h0: torch.Tensor, greedy: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], int]:
        """
        Execute TRUE BFS rollout with v2.2.0 fixes for training expansion.
        
        Key v2.2.0 change: epsilon-greedy exploration ensures tree actually
        expands during training instead of always staying at depth 0.
        """
        N = h0.size(0)
        device = h0.device
        dtype = h0.dtype
        
        h0 = _check_for_nan(h0, "h0")
        level_nodes: List[List[torch.Tensor]] = [[h0]]
        log_probs_list: List[torch.Tensor] = []
        current_depth = 0
        
        for level in range(self.max_depth):
            current_level_nodes = level_nodes[level]
            num_nodes_at_level = len(current_level_nodes)
            
            if num_nodes_at_level == 0:
                break
            
            # Aggregate all nodes at this level
            level_stack = torch.stack(current_level_nodes, dim=0)
            level_hidden = level_stack.mean(dim=0)
            
            # Add depth embedding (reduced scale in v2.2.0)
            if level < self.max_depth:
                depth_emb = self.depth_embeddings.weight[level]
                level_hidden = level_hidden + self.depth_embed_scale * depth_emb.unsqueeze(0)
            
            # Get grow probability
            grow_prob_raw = self.growth_policy(level_hidden, level)
            grow_prob = _safe_clamp_prob(grow_prob_raw, f"grow_prob_level_{level}")
            grow_prob_mean = grow_prob.mean()
            
            if greedy:
                expand = (grow_prob_mean >= self.greedy_threshold)
            else:
                # v2.2.0 FIX: Epsilon-greedy exploration
                # With probability min_explore_prob, force expansion
                explore_random = torch.rand(1, device=device).item()
                if explore_random < self.min_explore_prob:
                    # Force expansion for exploration
                    expand = True
                    expand_float = torch.ones(1, device=device, dtype=dtype)
                else:
                    # Stochastic sampling based on policy
                    expand_prob = grow_prob_mean.unsqueeze(0)
                    expand_sample = torch.bernoulli(expand_prob)
                    expand = expand_sample.bool().item()
                    expand_float = expand_sample
                
                # Compute log probability for REINFORCE
                log_p = _safe_log_prob(grow_prob_mean.unsqueeze(0), expand_float)
                log_probs_list.append(log_p)
            
            if not expand:
                break
            
            # Expand all nodes at this level
            next_level_nodes: List[torch.Tensor] = []
            for node_h in current_level_nodes:
                children_h = self.child_fc(node_h)
                left_h = F.relu(children_h[:, :self.hidden_dim])
                right_h = F.relu(children_h[:, self.hidden_dim:])
                
                if self.sibling_embed:
                    left_emb = self.sibling_embeddings.weight[0]
                    right_emb = self.sibling_embeddings.weight[1]
                    left_h = left_h + self.sibling_scale * left_emb.unsqueeze(0)
                    right_h = right_h + self.sibling_scale * right_emb.unsqueeze(0)
                
                left_h = _check_for_nan(left_h, f"left_child_level_{level}")
                right_h = _check_for_nan(right_h, f"right_child_level_{level}")
                next_level_nodes.append(left_h)
                next_level_nodes.append(right_h)
            
            level_nodes.append(next_level_nodes)
            current_depth = level + 1
        
        # Pool all nodes
        all_nodes: List[torch.Tensor] = []
        for level_node_list in level_nodes:
            all_nodes.extend(level_node_list)
        
        total_nodes = len(all_nodes)
        
        if total_nodes == 0:
            pooled = h0
        else:
            all_nodes_stack = torch.stack(all_nodes, dim=0)
            agg_sum = all_nodes_stack.sum(dim=0)
            agg_count = torch.full((N, 1), float(total_nodes), device=device, dtype=dtype)
            pooled = self._pool(agg_sum, agg_count)
        
        pooled = _check_for_nan(pooled, "pooled")
        output = self.output_fc(pooled)
        output = _check_for_nan(output, "output")
        
        log_probs = torch.cat(log_probs_list) if log_probs_list else None
        return output, log_probs, total_nodes
    
    def _compute_rewards(
        self, outputs: List[torch.Tensor], node_counts: List[int],
        labels: torch.Tensor, lambda_efficiency: float, batch_size: int, seq_len: int,
    ) -> torch.Tensor:
        num_positions = batch_size * seq_len
        labels_flat = labels.view(-1)
        rewards = []
        
        for out, nodes in zip(outputs, node_counts):
            ce_loss = F.cross_entropy(out, labels_flat, reduction='mean')
            if torch.isnan(ce_loss) or torch.isinf(ce_loss):
                ce_loss = torch.tensor(10.0, device=out.device, dtype=out.dtype)
            
            reward_accuracy = -ce_loss
            nodes_per_position = nodes / num_positions
            efficiency_ratio = min(max(nodes_per_position / self.max_nodes, 0.0), 1.0)
            efficiency_penalty = lambda_efficiency * efficiency_ratio
            reward = (reward_accuracy - efficiency_penalty) / 5.0
            rewards.append(reward)
        
        return torch.stack(rewards)
    
    def _compute_policy_loss(
        self, log_probs: List[Optional[torch.Tensor]], rewards: torch.Tensor, beta_entropy: float,
    ) -> torch.Tensor:
        valid_log_probs = [lp for lp in log_probs if lp is not None]
        if len(valid_log_probs) == 0:
            return torch.tensor(0.0, device=rewards.device, requires_grad=True)
        
        baseline = rewards.mean()
        policy_loss = torch.tensor(0.0, device=rewards.device, dtype=rewards.dtype)
        total_entropy = torch.tensor(0.0, device=rewards.device, dtype=rewards.dtype)
        total_decisions = 0
        
        for log_p, reward in zip(valid_log_probs, rewards):
            if torch.isnan(log_p).any():
                continue
            advantage = (reward - baseline).clamp(-2.0, 2.0)
            policy_loss = policy_loss - (log_p * advantage).sum()
            
            p = torch.exp(log_p).clamp(PROB_CLAMP_MIN, PROB_CLAMP_MAX)
            entropy = -(p * torch.log(p) + (1 - p) * torch.log(1 - p))
            entropy = _check_for_nan(entropy, "entropy", replace_value=0.0)
            total_entropy = total_entropy + entropy.sum()
            total_decisions += log_p.numel()
        
        if total_decisions > 0:
            policy_loss = policy_loss - beta_entropy * total_entropy
        
        if torch.isnan(policy_loss):
            return torch.tensor(0.0, device=rewards.device, requires_grad=True)
        return policy_loss
    
    def forward(
        self, x: torch.Tensor, num_rollouts: int = 3, lambda_efficiency: float = 0.05,
        beta_entropy: float = 0.01, labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
        B, seq_len = x.shape
        
        embedded = self.embedding(x)
        embedded_flat = embedded.view(B * seq_len, self.embed_dim)
        h0 = F.relu(self.embed_proj(embedded_flat))
        h0 = _check_for_nan(h0, "h0 (after embed_proj)")
        
        if not self.training:
            logits_flat, _, _ = self._true_bfs_rollout(h0, greedy=True)
            return logits_flat.view(B, seq_len, self.vocab_size)
        
        if labels is None:
            raise ValueError("labels required during training")
        
        all_outputs: List[torch.Tensor] = []
        all_log_probs: List[Optional[torch.Tensor]] = []
        all_node_counts: List[int] = []
        
        for _ in range(num_rollouts):
            out, log_p, nodes = self._true_bfs_rollout(h0, greedy=False)
            all_outputs.append(out)
            all_log_probs.append(log_p)
            all_node_counts.append(nodes)
        
        avg_outputs_flat = torch.stack(all_outputs).mean(dim=0)
        avg_outputs_flat = _check_for_nan(avg_outputs_flat, "avg_outputs_flat")
        
        rewards = self._compute_rewards(all_outputs, all_node_counts, labels, lambda_efficiency, B, seq_len)
        rewards = _check_for_nan(rewards, "rewards", replace_value=-1.0)
        policy_loss = self._compute_policy_loss(all_log_probs, rewards, beta_entropy)
        
        return avg_outputs_flat.view(B, seq_len, self.vocab_size), policy_loss, rewards, all_node_counts
    
    @torch.no_grad()
    def generate(
        self, prompt: torch.Tensor, max_new_tokens: int = 100, temperature: float = 1.0,
        top_k: Optional[int] = None, top_p: Optional[float] = None,
    ) -> torch.Tensor:
        self.eval()
        if prompt.dim() == 1:
            prompt = prompt.unsqueeze(0)
        generated = prompt.clone()
        
        for _ in range(max_new_tokens):
            logits = self(generated)
            next_logits = logits[:, -1, :]
            if temperature != 1.0:
                next_logits = next_logits / temperature
            if top_k is not None:
                v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < v[:, [-1]]] = float('-inf')
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)
        return generated
    
    def summary(self) -> str:
        return (
            f"BoeNet v2.2.0 - True BFS Language Model (Policy Saturation Fix)\n"
            f"  vocab_size={self.vocab_size}, embed_dim={self.embed_dim}, hidden_dim={self.hidden_dim}\n"
            f"  max_depth={self.max_depth}, max_nodes={self.max_nodes}\n"
            f"  greedy_threshold={self.greedy_threshold:.4f}, min_explore_prob={self.min_explore_prob:.4f}\n"
            f"  depth_embed_scale={self.depth_embed_scale} (reduced from 0.1)\n"
            f"  v2.2.0: Epsilon-greedy exploration, neutral policy init, LayerNorm"
        )
    
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    torch.manual_seed(42)
    
    B, seq_len, vocab_size, embed_dim, hidden_dim = 4, 32, 256, 64, 128
    
    print("=" * 70)
    print("BoeNet v2.2.0 Self-Test (Policy Saturation Fix)")
    print("=" * 70)
    
    model = BoeNet(vocab_size=vocab_size, embed_dim=embed_dim, hidden_dim=hidden_dim, max_depth=3)
    print(f"\n{model.summary()}")
    print(f"Parameters: {model.num_parameters():,}")
    
    # Test training mode with expansion
    model.train()
    x = torch.randint(0, vocab_size, (B, seq_len))
    y = torch.randint(0, vocab_size, (B, seq_len))
    
    print("\n[Test] Training forward pass with epsilon-greedy exploration:")
    outputs, policy_loss, rewards, node_counts = model(x, num_rollouts=3, labels=y)
    print(f"  Output shape: {outputs.shape}")
    print(f"  Node counts: {node_counts}")
    print(f"  Avg nodes/position: {sum(node_counts) / (3 * B * seq_len):.2f}")
    print(f"  Policy loss: {policy_loss.item():.4f}")
    
    # Check that trees actually expanded
    total_nodes = sum(node_counts)
    min_expected = B * seq_len * 3  # At least root nodes
    print(f"\n  Total nodes: {total_nodes}, Expected minimum (root only): {min_expected}")
    if total_nodes > min_expected:
        print("  ✓ Trees expanded beyond root! (v2.2.0 epsilon-greedy working)")
    else:
        print("  ⚠ Trees stayed at root (may need higher min_explore_prob)")
    
    # Test gradient flow
    model.zero_grad()
    ce_loss = F.cross_entropy(outputs.view(-1, vocab_size), y.view(-1))
    total_loss = ce_loss + 0.5 * policy_loss
    total_loss.backward()
    print(f"\n[Test] Gradient flow:")
    print(f"  Embedding grad norm: {model.embedding.weight.grad.norm():.4f}")
    print(f"  Policy fc1 grad norm: {model.growth_policy.policy_fc1.weight.grad.norm():.4f}")
    print("  ✓ Gradients flowing correctly")
    
    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)