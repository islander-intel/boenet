# BoeNet Architecture Specification

**Version**: 0.1.0 (Phase 1 - Character-Level)  
**Status**: 🚧 IN PROGRESS - Active Development  
**Purpose**: Complete technical specification for BoeNet implementation  
**Last Updated**: December 20, 2025

**⚠️ Proprietary Software**: This project is closed source. All rights reserved.

---

## 🎯 Document Purpose

This document provides the **complete technical specification** for BoeNet (Biological Optimized Enhanced Net), a language model that applies BFS tree expansion with REINFORCE policy gradients to sequential text processing.

**Audience**: Implementers, developers, researchers working on BoeNet codebase.

**Scope**: Phase 1 (character-level) architecture with notes on Phase 2+ extensions.

---

## 📚 Table of Contents

1. [High-Level Architecture](#1-high-level-architecture)
2. [BFSLanguageCell Specification](#2-bfslanguagecell-specification)
3. [BoeNet Model Specification](#3-boenet-model-specification)
4. [Policy Network Design](#4-policy-network-design)
5. [Reward Function](#5-reward-function)
6. [Training Algorithm](#6-training-algorithm)
7. [Inference Algorithm](#7-inference-algorithm)
8. [Tokenization](#8-tokenization)
9. [Data Pipeline](#9-data-pipeline)
10. [Hyperparameters](#10-hyperparameters)
11. [Implementation Details](#11-implementation-details)
12. [Design Decisions & Rationale](#12-design-decisions--rationale)
13. [Extensions & Future Work](#13-extensions--future-work)

---

## 1. High-Level Architecture

### 1.1 Conceptual Overview
```
┌─────────────────────────────────────────────────────────────┐
│                    BoeNet Architecture                       │
│                   (Phase 1: Character-Level)                 │
└─────────────────────────────────────────────────────────────┘

Input: "The cat sat"
  ↓ Tokenize (character-level)
Tokens: [84, 104, 101, 32, 99, 97, 116, 32, 115, 97, 116]
        (T, h, e, space, c, a, t, space, s, a, t)
  ↓ Embed
Embeddings: [B, seq_len, embed_dim]
  ↓
┌─────────────────────────────────────────────────────────────┐
│  For layer l in [1..num_layers]:                            │
│    hidden[l, 0] = 0  # Initialize                           │
│    For timestep t in [1..seq_len]:                          │
│      hidden[l, t], policy_loss[l, t] = BFSLanguageCell_l(   │
│          input=embeddings[t] if l=1 else hidden[l-1, t],    │
│          hidden_prev=hidden[l, t-1],                         │
│          policy_net=policy_net_l                             │
│      )                                                       │
└─────────────────────────────────────────────────────────────┘
  ↓
Logits: [B, seq_len, vocab_size]
  ↓
Loss: CrossEntropy(logits, targets) + β × policy_loss
```

### 1.2 Key Components

| Component | Description | Input Shape | Output Shape |
|-----------|-------------|-------------|--------------|
| **Tokenizer** | Character-level ASCII encoder | Text string | `[seq_len]` |
| **Embedding** | Token → vector | `[B, seq_len]` | `[B, seq_len, embed_dim]` |
| **BFSLanguageCell** | BFS tree per token | `[B, embed_dim/hidden_dim]` + `[B, hidden_dim]` | `[B, hidden_dim]` + scalar |
| **Output FC** | Hidden → logits | `[B, seq_len, hidden_dim]` | `[B, seq_len, vocab_size]` |

### 1.3 Comparison to BFSNet

| Aspect | BFSNet (Vision) | BoeNet (Language) |
|--------|-----------------|-------------------|
| **Processing** | Single BFS pass | Recurrent BFS per token |
| **Input** | `[B, 784]` (image) | `[B, seq_len]` (tokens) |
| **Hidden State** | None (feedforward) | `[B, hidden_dim]` (recurrent) |
| **Layers** | 1 BFS expansion | `num_layers` stacked cells |
| **Output** | `[B, 10]` (class logits) | `[B, seq_len, vocab_size]` |
| **Metric** | Accuracy | Perplexity |

---

## 2. BFSLanguageCell Specification

### 2.1 Purpose

The **BFSLanguageCell** is the core building block of BoeNet. It processes a single token through a BFS tree expansion, producing:
1. A new hidden state (to be passed to next token)
2. A policy loss (REINFORCE gradient)

**Key Innovation**: Unlike BFSNet which processes entire images, BFSLanguageCell processes one token at a time, making it suitable for variable-length sequences.

### 2.2 Mathematical Formulation

**Inputs**:
- `x_t`: Current token representation, `[B, input_dim]`
  - For layer 1: `x_t = embed(token[t])`, shape `[B, embed_dim]`
  - For layer l>1: `x_t = hidden[l-1, t]`, shape `[B, hidden_dim]`
- `h_{t-1}`: Previous hidden state, `[B, hidden_dim]`

**Outputs**:
- `h_t`: New hidden state, `[B, hidden_dim]`
- `L_policy`: Policy loss (scalar)

**Forward Pass**:
```
# Step 1: Root node (combines current token + previous state)
root = RootFC(concat(x_t, h_{t-1}))
  where root: [B, hidden_dim]

# Step 2: BFS expansion (like BFSNet, but per token)
frontier_0 = [root]
for depth d in [1..max_depth]:
  frontier_d = []
  for parent in frontier_{d-1}:
    # Policy decides: create children?
    grow_prob = PolicyNet_d(parent)  # [B, 1]
    
    # Sample growth decision (REINFORCE)
    if training:
      grow = Bernoulli(grow_prob)
    else:
      grow = (grow_prob >= greedy_threshold)
    
    # Create children if growing
    if grow:
      for k in [1..max_children]:
        child = ChildFC_d(parent)
        frontier_d.append(child)

# Step 3: Pool all nodes to get hidden state
all_nodes = flatten(frontier_0, frontier_1, ..., frontier_max_depth)
h_t = Pool(all_nodes)  # [B, hidden_dim]

# Step 4: Compute policy loss (REINFORCE)
# (Detailed in section 5)
L_policy = ComputePolicyLoss(...)
```

### 2.3 Architecture Diagram
```
┌─────────────────────────────────────────────────────────────┐
│                  BFSLanguageCell Forward Pass                │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Input:                                                      │
│    x_t: [B, input_dim]      (current token embedding)       │
│    h_{t-1}: [B, hidden_dim] (previous hidden state)         │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Root FC: concat(x_t, h_{t-1}) → root                │   │
│  └──────────────────────────────┬───────────────────────┘   │
│                                 │                            │
│                     root: [B, hidden_dim]                    │
│                                 │                            │
│                    ┌────────────┴────────────┐               │
│                    │  Depth 1: BFS Layer     │               │
│                    │  Policy decides growth   │               │
│                    └────────────┬────────────┘               │
│                                 │                            │
│              ┌──────────────────┼──────────────────┐         │
│              │                  │                  │         │
│           child_1            child_2           child_3       │
│              │                  │                  │         │
│         [B, hidden_dim]    [B, hidden_dim]   [B, hidden_dim]│
│              │                  │                  │         │
│              └──────────────────┼──────────────────┘         │
│                                 │                            │
│                    ┌────────────┴────────────┐               │
│                    │  Depth 2: BFS Layer     │               │
│                    │  (if max_depth >= 2)    │               │
│                    └────────────┬────────────┘               │
│                                 │                            │
│                          [more children...]                  │
│                                 │                            │
│                    ┌────────────┴────────────┐               │
│                    │  Pooling: all nodes     │               │
│                    │  → h_t                  │               │
│                    └────────────┬────────────┘               │
│                                 │                            │
│  Output:                        │                            │
│    h_t: [B, hidden_dim]  ───────┘                           │
│    L_policy: scalar                                          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 2.4 Class Definition (PyTorch)
```python
import torch
import torch.nn as nn

class BFSLanguageCell(nn.Module):
    """
    BFS tree expansion for a single token in a sequence.
    
    Processes one token through a BFS tree, producing:
    - New hidden state (to next token)
    - Policy loss (REINFORCE)
    
    Args:
        input_dim (int): Dimension of input (embed_dim for layer 1, hidden_dim for layer >1)
        hidden_dim (int): Dimension of hidden state
        max_depth (int): Maximum BFS tree depth
        max_children (int): Maximum children per node (K)
        greedy_threshold (float): Threshold for greedy inference
        pooling_mode (str): How to pool nodes ('mean', 'sum', 'learned')
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        max_depth: int = 2,
        max_children: int = 3,
        greedy_threshold: float = 0.42,
        pooling_mode: str = 'mean'
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.max_depth = max_depth
        self.max_children = max_children
        self.greedy_threshold = greedy_threshold
        self.pooling_mode = pooling_mode
        
        # Root FC: combines token embedding + previous hidden state
        self.root_fc = nn.Linear(input_dim + hidden_dim, hidden_dim)
        
        # Child FCs: one per depth level
        self.child_fcs = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(max_depth)
        ])
        
        # Policy networks: one per depth level
        self.policy_nets = nn.ModuleList([
            GrowthPolicyNet(hidden_dim, depth=d)
            for d in range(max_depth)
        ])
        
        # Pooling
        if pooling_mode == 'learned':
            self.pool_fc = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(
        self,
        x_t: torch.Tensor,          # [B, input_dim]
        h_prev: torch.Tensor,        # [B, hidden_dim]
        num_rollouts: int = 3,
        lambda_efficiency: float = 0.05,
        reward: torch.Tensor = None  # [B] - provided during training
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through BFS tree.
        
        Args:
            x_t: Current token representation [B, input_dim]
            h_prev: Previous hidden state [B, hidden_dim]
            num_rollouts: Number of REINFORCE rollouts (training only)
            lambda_efficiency: Efficiency penalty weight
            reward: Reward signal for REINFORCE [B] (training only)
        
        Returns:
            h_next: New hidden state [B, hidden_dim]
            policy_loss: REINFORCE policy loss (scalar)
        """
        B = x_t.shape[0]
        device = x_t.device
        
        if self.training:
            return self._forward_train(
                x_t, h_prev, num_rollouts, lambda_efficiency, reward
            )
        else:
            return self._forward_inference(x_t, h_prev)
    
    def _forward_train(
        self,
        x_t: torch.Tensor,
        h_prev: torch.Tensor,
        num_rollouts: int,
        lambda_efficiency: float,
        reward: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Training forward pass with REINFORCE rollouts."""
        B = x_t.shape[0]
        device = x_t.device
        
        # Storage for rollouts
        all_hiddens = []
        all_log_probs = []
        all_nodes_used = []
        
        for rollout in range(num_rollouts):
            # Build BFS tree (stochastic sampling)
            hidden, log_probs, nodes_used = self._build_tree_stochastic(
                x_t, h_prev
            )
            
            all_hiddens.append(hidden)
            all_log_probs.append(log_probs)
            all_nodes_used.append(nodes_used)
        
        # Average hidden states across rollouts
        h_next = torch.stack(all_hiddens).mean(dim=0)  # [B, hidden_dim]
        
        # Compute policy loss (REINFORCE)
        policy_loss = self._compute_policy_loss(
            all_log_probs,
            all_nodes_used,
            reward,
            lambda_efficiency
        )
        
        return h_next, policy_loss
    
    def _forward_inference(
        self,
        x_t: torch.Tensor,
        h_prev: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Inference forward pass (greedy, deterministic)."""
        # Build BFS tree (greedy threshold)
        hidden, _, _ = self._build_tree_greedy(x_t, h_prev)
        
        # No policy loss during inference
        policy_loss = torch.tensor(0.0, device=x_t.device)
        
        return hidden, policy_loss
    
    def _build_tree_stochastic(
        self,
        x_t: torch.Tensor,
        h_prev: torch.Tensor
    ) -> tuple[torch.Tensor, list, torch.Tensor]:
        """
        Build BFS tree with stochastic sampling (training).
        
        Returns:
            hidden: Pooled hidden state [B, hidden_dim]
            log_probs: List of log probabilities for REINFORCE
            nodes_used: Number of nodes created [B]
        """
        B = x_t.shape[0]
        device = x_t.device
        
        # Initialize root
        root_input = torch.cat([x_t, h_prev], dim=-1)  # [B, input_dim + hidden_dim]
        root = torch.relu(self.root_fc(root_input))    # [B, hidden_dim]
        
        # Storage
        all_nodes = [root]
        log_probs = []
        nodes_used = torch.ones(B, device=device)  # Start with 1 (root)
        
        # BFS expansion
        frontier = [root]
        for depth in range(self.max_depth):
            new_frontier = []
            
            for parent in frontier:
                # Policy decision
                grow_prob = self.policy_nets[depth](parent)  # [B, 1]
                
                # Sample growth (Bernoulli)
                grow_dist = torch.distributions.Bernoulli(grow_prob)
                grow = grow_dist.sample()  # [B, 1]
                
                # Store log probability for REINFORCE
                log_prob = grow_dist.log_prob(grow)  # [B, 1]
                log_probs.append(log_prob)
                
                # Create children if growing
                # Note: This is simplified; actual implementation needs masking
                for k in range(self.max_children):
                    child = torch.relu(self.child_fcs[depth](parent))  # [B, hidden_dim]
                    # Mask by growth decision
                    child = child * grow.squeeze(-1).unsqueeze(-1)
                    
                    new_frontier.append(child)
                    all_nodes.append(child)
                    nodes_used += grow.squeeze(-1)  # Increment if grew
            
            frontier = new_frontier
        
        # Pool all nodes
        hidden = self._pool_nodes(all_nodes)  # [B, hidden_dim]
        
        return hidden, log_probs, nodes_used
    
    def _build_tree_greedy(
        self,
        x_t: torch.Tensor,
        h_prev: torch.Tensor
    ) -> tuple[torch.Tensor, list, torch.Tensor]:
        """
        Build BFS tree with greedy threshold (inference).
        
        Returns:
            hidden: Pooled hidden state [B, hidden_dim]
            log_probs: Empty list (no gradients)
            nodes_used: Number of nodes created [B]
        """
        B = x_t.shape[0]
        device = x_t.device
        
        # Initialize root
        root_input = torch.cat([x_t, h_prev], dim=-1)
        root = torch.relu(self.root_fc(root_input))
        
        # Storage
        all_nodes = [root]
        nodes_used = torch.ones(B, device=device)
        
        # BFS expansion
        frontier = [root]
        for depth in range(self.max_depth):
            new_frontier = []
            
            for parent in frontier:
                # Policy decision (greedy threshold)
                grow_prob = self.policy_nets[depth](parent)  # [B, 1]
                grow = (grow_prob >= self.greedy_threshold).float()  # [B, 1]
                
                # Create children if growing
                for k in range(self.max_children):
                    child = torch.relu(self.child_fcs[depth](parent))
                    child = child * grow.squeeze(-1).unsqueeze(-1)
                    
                    new_frontier.append(child)
                    all_nodes.append(child)
                    nodes_used += grow.squeeze(-1)
            
            frontier = new_frontier
        
        # Pool all nodes
        hidden = self._pool_nodes(all_nodes)
        
        return hidden, [], nodes_used
    
    def _pool_nodes(self, nodes: list[torch.Tensor]) -> torch.Tensor:
        """
        Pool all nodes to produce hidden state.
        
        Args:
            nodes: List of [B, hidden_dim] tensors
        
        Returns:
            hidden: [B, hidden_dim]
        """
        if self.pooling_mode == 'mean':
            # Average pooling
            stacked = torch.stack(nodes, dim=1)  # [B, num_nodes, hidden_dim]
            hidden = stacked.mean(dim=1)  # [B, hidden_dim]
        
        elif self.pooling_mode == 'sum':
            # Sum pooling
            stacked = torch.stack(nodes, dim=1)
            hidden = stacked.sum(dim=1)
        
        elif self.pooling_mode == 'learned':
            # Learned pooling
            stacked = torch.stack(nodes, dim=1)
            pooled = stacked.mean(dim=1)
            hidden = torch.relu(self.pool_fc(pooled))
        
        else:
            raise ValueError(f"Unknown pooling mode: {self.pooling_mode}")
        
        return hidden
    
    def _compute_policy_loss(
        self,
        all_log_probs: list,
        all_nodes_used: list,
        reward: torch.Tensor,
        lambda_efficiency: float
    ) -> torch.Tensor:
        """
        Compute REINFORCE policy loss.
        
        Args:
            all_log_probs: List of log probabilities from rollouts
            all_nodes_used: List of nodes used from rollouts
            reward: Base reward (e.g., -perplexity) [B]
            lambda_efficiency: Efficiency penalty weight
        
        Returns:
            policy_loss: Scalar loss
        """
        # Compute efficiency penalty
        max_nodes = 1 + sum(
            self.max_children ** (d + 1) for d in range(self.max_depth)
        )
        
        # Average across rollouts
        num_rollouts = len(all_nodes_used)
        avg_nodes_used = torch.stack(all_nodes_used).mean(dim=0)  # [B]
        
        # Reward with efficiency penalty
        efficiency_penalty = lambda_efficiency * (avg_nodes_used / max_nodes)
        total_reward = reward - efficiency_penalty  # [B]
        
        # REINFORCE: -log_prob × reward (negative because we minimize loss)
        policy_loss = 0.0
        for log_probs in all_log_probs:
            for log_prob in log_probs:
                # log_prob: [B, 1], total_reward: [B]
                policy_loss += -(log_prob.squeeze(-1) * total_reward).mean()
        
        # Average across rollouts
        policy_loss = policy_loss / num_rollouts
        
        return policy_loss
```

### 2.5 Key Design Choices

**1. Root FC Combines Token + Hidden State**:
- **Why**: Natural for recurrent processing (like LSTM's input gate)
- **Alternative**: Separate processing then combine (rejected: more parameters)

**2. Separate Policy Nets per Depth**:
- **Why**: Different depths may need different growth criteria
- **From BFSNet**: This worked well in vision
- **Alternative**: Single shared policy (may try in Phase 2)

**3. Stochastic Training, Greedy Inference**:
- **Why**: REINFORCE requires stochastic sampling
- **Issue**: Training/inference mismatch (BFSNet found this!)
- **Mitigation**: Threshold tuning, `--debug_policy` flag

**4. Pooling Mode = 'mean'**:
- **Why**: BFSNet found 'mean' and 'learned' both work well
- **Start**: 'mean' (simpler, fewer parameters)
- **Alternative**: 'learned' if needed (try in Phase 2)

---

## 3. BoeNet Model Specification

### 3.1 Full Model Architecture
```python
class BoeNet(nn.Module):
    """
    BoeNet: BFS-based language model with recurrent processing.
    
    Architecture:
        Token Embedding
        → Stack of BFSLanguageCell layers (recurrent per layer)
        → Output FC (hidden → vocab logits)
    
    Args:
        vocab_size (int): Vocabulary size
        embed_dim (int): Token embedding dimension
        hidden_dim (int): Hidden state dimension
        num_layers (int): Number of stacked BFSLanguageCell layers
        max_depth (int): BFS tree max depth
        max_children (int): BFS tree branching factor
        greedy_threshold (float): Greedy inference threshold
        pooling_mode (str): Node pooling mode
    """
    
    def __init__(
        self,
        vocab_size: int = 256,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 4,
        max_depth: int = 2,
        max_children: int = 3,
        greedy_threshold: float = 0.42,
        pooling_mode: str = 'mean'
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Token embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Stack of BFS cells
        self.cells = nn.ModuleList()
        for layer_idx in range(num_layers):
            input_dim = embed_dim if layer_idx == 0 else hidden_dim
            
            cell = BFSLanguageCell(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                max_depth=max_depth,
                max_children=max_children,
                greedy_threshold=greedy_threshold,
                pooling_mode=pooling_mode
            )
            self.cells.append(cell)
        
        # Output layer
        self.output_fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(
        self,
        tokens: torch.Tensor,           # [B, seq_len]
        num_rollouts: int = 3,
        lambda_efficiency: float = 0.05,
        targets: torch.Tensor = None    # [B, seq_len] (for training)
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Forward pass through BoeNet.
        
        Args:
            tokens: Input token IDs [B, seq_len]
            num_rollouts: REINFORCE rollouts (training only)
            lambda_efficiency: Efficiency penalty
            targets: Target tokens for next-token prediction [B, seq_len]
        
        Returns:
            logits: Output logits [B, seq_len, vocab_size]
            policy_loss: Total policy loss (scalar)
            info: Dict with metrics (nodes_used, perplexity, etc.)
        """
        B, seq_len = tokens.shape
        device = tokens.device
        
        # Embed tokens
        embeds = self.embedding(tokens)  # [B, seq_len, embed_dim]
        
        # Initialize hidden states (all zeros)
        # hidden[layer, timestep] will be [B, hidden_dim]
        hidden = [[torch.zeros(B, self.hidden_dim, device=device)
                   for _ in range(seq_len + 1)]  # +1 for initial state
                  for _ in range(self.num_layers)]
        
        # Storage for policy losses
        policy_losses = []
        
        # Process sequence (layer by layer, then timestep by timestep)
        for layer_idx in range(self.num_layers):
            for t in range(seq_len):
                # Input for this cell
                if layer_idx == 0:
                    x_t = embeds[:, t, :]  # [B, embed_dim]
                else:
                    x_t = hidden[layer_idx - 1][t + 1]  # [B, hidden_dim]
                
                # Previous hidden state
                h_prev = hidden[layer_idx][t]  # [B, hidden_dim]
                
                # Compute reward for REINFORCE (if training)
                if self.training and targets is not None:
                    # Reward = -loss on this token
                    # (computed later after we have logits)
                    reward = None  # Placeholder
                else:
                    reward = None
                
                # Forward through cell
                h_next, policy_loss = self.cells[layer_idx](
                    x_t=x_t,
                    h_prev=h_prev,
                    num_rollouts=num_rollouts,
                    lambda_efficiency=lambda_efficiency,
                    reward=reward
                )
                
                # Store
                hidden[layer_idx][t + 1] = h_next
                policy_losses.append(policy_loss)
        
        # Get final layer hidden states
        final_hiddens = torch.stack(
            [hidden[-1][t + 1] for t in range(seq_len)],
            dim=1
        )  # [B, seq_len, hidden_dim]
        
        # Output layer
        logits = self.output_fc(final_hiddens)  # [B, seq_len, vocab_size]
        
        # Total policy loss
        total_policy_loss = sum(policy_losses) / len(policy_losses)
        
        # Compute metrics
        info = {}
        if targets is not None:
            # Cross-entropy loss
            ce_loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1),
                reduction='mean'
            )
            info['ce_loss'] = ce_loss.item()
            
            # Perplexity
            perplexity = torch.exp(ce_loss)
            info['perplexity'] = perplexity.item()
        
        return logits, total_policy_loss, info
    
    def generate(
        self,
        prompt_tokens: torch.Tensor,    # [1, prompt_len]
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = None,
        top_p: float = None,
        greedy: bool = False
    ) -> torch.Tensor:
        """
        Autoregressive text generation.
        
        Args:
            prompt_tokens: Starting tokens [1, prompt_len]
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k sampling (None = disabled)
            top_p: Nucleus sampling (None = disabled)
            greedy: Use greedy decoding (argmax)
        
        Returns:
            generated: Full sequence [1, prompt_len + max_new_tokens]
        """
        self.eval()
        device = prompt_tokens.device
        
        # Start with prompt
        generated = prompt_tokens.clone()
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Forward pass (only need last token logits)
                logits, _, _ = self.forward(
                    generated,
                    num_rollouts=1  # Inference uses 1 rollout
                )
                
                # Get last token logits
                next_token_logits = logits[0, -1, :]  # [vocab_size]
                
                # Apply temperature
                next_token_logits = next_token_logits / temperature
                
                # Apply top-k
                if top_k is not None:
                    indices_to_remove = next_token_logits < torch.topk(
                        next_token_logits, top_k
                    )[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Apply top-p (nucleus)
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(
                        next_token_logits, descending=True
                    )
                    cumulative_probs = torch.cumsum(
                        F.softmax(sorted_logits, dim=-1), dim=-1
                    )
                    
                    # Remove tokens with cumulative probability above threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                if greedy:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                else:
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to sequence
                generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
        
        return generated
```

---

## 4. Policy Network Design

### 4.1 GrowthPolicyNet

**Purpose**: Decide whether to grow children from a parent node.

**From BFSNet**: Same architecture, proven to work.
```python
class GrowthPolicyNet(nn.Module):
    """
    Policy network for BFS growth decisions.
    
    Outputs probability of creating children from a parent node.
    
    Args:
        hidden_dim (int): Hidden state dimension
        depth (int): Depth level (for depth-specific policies)
    """
    
    def __init__(self, hidden_dim: int, depth: int = 0):
        super().__init__()
        self.depth = depth
        
        # Simple MLP: hidden → 64 → 1
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.fc2 = nn.Linear(64, 1)
    
    def forward(self, parent: torch.Tensor) -> torch.Tensor:
        """
        Compute growth probability for a parent node.
        
        Args:
            parent: Parent node representation [B, hidden_dim]
        
        Returns:
            grow_prob: Probability of growing [B, 1]
        """
        x = torch.relu(self.fc1(parent))
        grow_prob = torch.sigmoid(self.fc2(x))  # [B, 1]
        return grow_prob
```

### 4.2 Design Rationale

**Depth-Specific Policies**:
- BFSNet found that different depths benefit from separate policies
- Root → depth-1 decisions may differ from depth-1 → depth-2
- **Example**: Depth-0 policy may be more aggressive (always explore), depth-1 more conservative

**Simple MLP**:
- 2-layer MLP is sufficient (BFSNet validation)
- More complex policies (attention, etc.) can be tried in Phase 2+

**Sigmoid Output**:
- Ensures grow_prob ∈ [0, 1]
- Natural interpretation as Bernoulli probability

---

## 5. Reward Function

### 5.1 Formulation

**Goal**: Train policy to balance quality (low perplexity) and efficiency (low FLOPs).

**Reward at Timestep t**:
```
r_t = -loss_t - λ × efficiency_penalty_t

where:
  loss_t = CrossEntropy(logit_t, target_t)  # Next-token prediction loss
  efficiency_penalty_t = (nodes_used_t / max_nodes)
  λ = lambda_efficiency hyperparameter
```

**Aggregated Reward (Sequence-Level)**:
```
R = (1 / seq_len) × Σ_t r_t
  = -perplexity - λ × avg_efficiency_penalty
```

### 5.2 Design Decisions

**Why Negative Loss as Reward?**:
- REINFORCE maximizes reward
- Lower loss → higher reward
- Natural alignment with optimization objective

**Why FLOPs (not Nodes)?**:
- More accurate measure of computational cost
- FLOPs ∝ nodes_used × hidden_dim²
- **Simplification for Phase 1**: Use nodes_used (FLOPs proportional)

**Why Lambda=0.05?**:
- BFSNet found λ=0.05 > λ=0.01 (counter-intuitive!)
- Acts as regularization
- Start with 0.05, sweep [0.01, 0.05, 0.1] in experiments

### 5.3 Implementation
```python
def compute_reward(
    logits: torch.Tensor,          # [B, seq_len, vocab_size]
    targets: torch.Tensor,          # [B, seq_len]
    nodes_used: torch.Tensor,       # [B, num_layers, seq_len]
    max_nodes: int,
    lambda_efficiency: float = 0.05
) -> torch.Tensor:
    """
    Compute reward for REINFORCE.
    
    Returns:
        reward: [B, num_layers, seq_len] - reward per cell
    """
    B, seq_len, vocab_size = logits.shape
    
    # Per-token loss (sample-independent!)
    token_losses = F.cross_entropy(
        logits.view(-1, vocab_size),
        targets.view(-1),
        reduction='none'
    ).view(B, seq_len)  # [B, seq_len]
    
    # Efficiency penalty
    efficiency_penalty = (nodes_used / max_nodes)  # [B, num_layers, seq_len]
    
    # Reward = -loss - λ × efficiency
    # Broadcast token_losses to match nodes_used shape
    reward = -token_losses.unsqueeze(1) - lambda_efficiency * efficiency_penalty
    
    return reward  # [B, num_layers, seq_len]
```

**Critical**: Reward computation must be **sample-independent**!
- BFSNet bug: Used batch norm in reward (batch-dependent)
- BoeNet: No batch norm anywhere near reward calculation

---

## 6. Training Algorithm

### 6.1 Training Loop Pseudocode
```python
def train_boenet(
    model: BoeNet,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 10,
    lr: float = 0.001,
    lambda_efficiency: float = 0.05,
    num_rollouts: int = 3,
    beta_policy: float = 0.5,
    gradient_clip: float = 1.0,
    device: str = 'cuda'
):
    """
    BoeNet training loop.
    
    Args:
        model: BoeNet model
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of epochs
        lr: Learning rate
        lambda_efficiency: Efficiency penalty weight
        num_rollouts: REINFORCE rollouts
        beta_policy: Policy loss weight (0.5 = equal to CE loss)
        gradient_clip: Gradient clipping threshold
        device: Device to train on
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    for epoch in range(num_epochs):
        model.train()
        
        # Training
        for batch in train_loader:
            tokens = batch['tokens'].to(device)      # [B, seq_len]
            targets = batch['targets'].to(device)    # [B, seq_len]
            
            # Forward pass
            logits, policy_loss, info = model(
                tokens=tokens,
                targets=targets,
                num_rollouts=num_rollouts,
                lambda_efficiency=lambda_efficiency
            )
            
            # Cross-entropy loss
            ce_loss = F.cross_entropy(
                logits.view(-1, model.vocab_size),
                targets.view(-1)
            )
            
            # Total loss
            total_loss = ce_loss + beta_policy * policy_loss
            
            # Backward
            optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            
            # Update
            optimizer.step()
            
            # Log
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}: "
                      f"CE Loss: {ce_loss.item():.4f}, "
                      f"Policy Loss: {policy_loss.item():.4f}, "
                      f"Perplexity: {info['perplexity']:.2f}")
        
        # Validation
        val_perplexity = validate(model, val_loader, device)
        print(f"Epoch {epoch} Validation Perplexity: {val_perplexity:.2f}")
        
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_perplexity': val_perplexity
        }, f'checkpoints/boenet_epoch_{epoch}.pt')
```

### 6.2 Key Training Details

**Optimizer**: AdamW (better than Adam for language models)
- Weight decay: 0.01
- Betas: (0.9, 0.999) (default)

**Learning Rate Schedule**:
- Phase 1: Constant lr=0.001 (simple, works for 10M params)
- Phase 2+: Cosine annealing with warmup

**Gradient Clipping**: 1.0 (prevent explosion)

**Beta Policy**: 0.5 (policy loss weighted equally with CE loss)
- Lower (0.1-0.3): Prioritize perplexity
- Higher (0.7-0.9): Prioritize efficiency
- **Start with 0.5**, sweep if needed

**Batch Size**:
- Phase 1 (char-level): 64 (fits on CPU)
- Phase 2 (word-level): 32-64 (GPU memory-dependent)

---

## 7. Inference Algorithm

### 7.1 Greedy Decoding
```python
def inference_greedy(
    model: BoeNet,
    prompt: str,
    tokenizer: CharTokenizer,
    max_new_tokens: int = 100,
    device: str = 'cuda'
) -> str:
    """
    Greedy decoding (argmax at each step).
    
    Fastest, but may be repetitive.
    """
    model.eval()
    
    # Tokenize prompt
    prompt_tokens = tokenizer.encode(prompt)
    tokens = torch.tensor([prompt_tokens], device=device)  # [1, prompt_len]
    
    # Generate
    generated = model.generate(
        prompt_tokens=tokens,
        max_new_tokens=max_new_tokens,
        greedy=True
    )
    
    # Decode
    generated_text = tokenizer.decode(generated[0].tolist())
    return generated_text
```

### 7.2 Sampling Strategies

**Temperature Sampling**:
```python
# Higher temperature → more random
generated = model.generate(
    prompt_tokens=tokens,
    max_new_tokens=100,
    temperature=0.8  # Default: 1.0
)
```

**Top-k Sampling**:
```python
# Only sample from top k most likely tokens
generated = model.generate(
    prompt_tokens=tokens,
    max_new_tokens=100,
    top_k=40
)
```

**Nucleus (Top-p) Sampling**:
```python
# Sample from smallest set with cumulative prob >= p
generated = model.generate(
    prompt_tokens=tokens,
    max_new_tokens=100,
    top_p=0.9
)
```

**Recommended for Phase 1**: Temperature=0.8, top_k=40

---

## 8. Tokenization

### 8.1 Character-Level Tokenizer (Phase 1)
```python
class CharTokenizer:
    """
    Simple character-level tokenizer for Phase 1.
    
    Vocab: ASCII characters (0-255)
    """
    
    def __init__(self, vocab_size: int = 256):
        self.vocab_size = vocab_size
    
    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs."""
        return [ord(c) for c in text]
    
    def decode(self, tokens: list[int]) -> str:
        """Decode token IDs to text."""
        return ''.join([chr(t) for t in tokens])
    
    def __len__(self):
        return self.vocab_size
```

### 8.2 BPE Tokenizer (Phase 2)

**For Phase 2 (word-level)**:
- Use Hugging Face `tokenizers` library
- Train BPE on training data or use GPT-2 tokenizer
- Vocab size: 50,257 (GPT-2 standard)
```python
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

def train_bpe_tokenizer(
    text_files: list[str],
    vocab_size: int = 50257,
    output_path: str = 'data/tokenizers/bpe-gpt2'
):
    """Train BPE tokenizer on text corpus."""
    tokenizer = Tokenizer(BPE())
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=["<|endoftext|>"])
    
    tokenizer.train(files=text_files, trainer=trainer)
    tokenizer.save(output_path)
```

---

## 9. Data Pipeline

### 9.1 Character-Level Dataset (Phase 1)
```python
class CharDataset(torch.utils.data.Dataset):
    """
    Character-level dataset for Phase 1.
    
    Loads text file, creates fixed-length sequences.
    """
    
    def __init__(
        self,
        text_file: str,
        seq_len: int = 128,
        tokenizer: CharTokenizer = None
    ):
        self.seq_len = seq_len
        self.tokenizer = tokenizer or CharTokenizer()
        
        # Load text
        with open(text_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Tokenize entire corpus
        self.tokens = self.tokenizer.encode(text)
        
        # Number of sequences
        self.num_sequences = len(self.tokens) // seq_len
    
    def __len__(self):
        return self.num_sequences
    
    def __getitem__(self, idx):
        # Get sequence
        start = idx * self.seq_len
        end = start + self.seq_len
        
        tokens = self.tokens[start:end]
        targets = self.tokens[start+1:end+1]  # Next-token prediction
        
        return {
            'tokens': torch.tensor(tokens, dtype=torch.long),
            'targets': torch.tensor(targets, dtype=torch.long)
        }
```

### 9.2 DataLoader Configuration
```python
def create_dataloaders(
    train_file: str,
    val_file: str,
    seq_len: int = 128,
    batch_size: int = 64,
    num_workers: int = 4
):
    """Create train and validation dataloaders."""
    train_dataset = CharDataset(train_file, seq_len=seq_len)
    val_dataset = CharDataset(val_file, seq_len=seq_len)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader
```

---

## 10. Hyperparameters

### 10.1 Phase 1 (Character-Level) Default Configuration
```yaml
# Model architecture
model:
  vocab_size: 256              # ASCII
  embed_dim: 64
  hidden_dim: 128
  num_layers: 4                # Stacked BFS cells
  max_depth: 2                 # BFS tree depth
  max_children: 3              # K value
  greedy_threshold: 0.42       # From BFSNet
  pooling_mode: 'mean'

# Dataset
dataset:
  name: 'shakespeare'
  path: 'data/text/shakespeare.txt'
  seq_len: 128
  train_split: 0.9
  val_split: 0.1

# Training
training:
  epochs: 10
  batch_size: 64
  lr: 0.001
  optimizer: 'adamw'
  weight_decay: 0.01
  gradient_clip: 1.0
  
  # REINFORCE
  num_rollouts: 3
  lambda_efficiency: 0.05
  beta_policy: 0.5
  beta_entropy: 0.01

# Inference
inference:
  max_new_tokens: 100
  temperature: 0.8
  top_k: 40
  greedy_threshold: 0.42       # MUST TUNE after training!

# Hardware
hardware:
  device: 'cuda'               # or 'cpu' for Phase 1
  mixed_precision: false       # Phase 2+
```

### 10.2 Hyperparameter Sensitivity (From BFSNet)

| Hyperparameter | Sensitivity | Recommendation |
|----------------|-------------|----------------|
| **lambda_efficiency** | HIGH | Start 0.05, sweep [0.01, 0.05, 0.1] |
| **greedy_threshold** | CRITICAL | MUST tune post-training (use `--debug_policy`) |
| **num_rollouts** | MEDIUM | 3 is good, 1 too noisy, 5+ diminishing returns |
| **beta_policy** | MEDIUM | 0.5 balances quality/efficiency |
| **max_depth** | HIGH | Start 2, try 3 if needed |
| **max_children (K)** | MEDIUM | 3 worked for BFSNet, likely good here |
| **pooling_mode** | LOW | 'mean' and 'learned' both work |

---

## 11. Implementation Details

### 11.1 Critical Implementation Notes

**1. Sample-Independent Rewards** (BFSNet Bug Fix):
```python
# ❌ WRONG (batch-dependent)
reward = -F.cross_entropy(logits, targets)  # Averages across batch

# ✅ CORRECT (sample-independent)
reward = -F.cross_entropy(logits, targets, reduction='none')  # [B]
```

**2. Gradient Flow Verification**:
```python
# After backward(), check all layers have gradients
for name, param in model.named_parameters():
    if param.grad is None:
        print(f"WARNING: {name} has no gradient!")
```

**3. Policy Distribution Monitoring**:
```python
# Track grow_prob statistics during training
grow_probs = []
for layer in model.cells:
    for policy_net in layer.policy_nets:
        # Sample some nodes
        grow_prob = policy_net(sample_nodes)
        grow_probs.append(grow_prob.mean().item())

print(f"Mean grow_prob: {np.mean(grow_probs):.4f}")
print(f"Std grow_prob: {np.std(grow_probs):.4f}")
```

**4. Checkpoint Format**:
```python
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'config': {
        'vocab_size': model.vocab_size,
        'embed_dim': model.embed_dim,
        'hidden_dim': model.hidden_dim,
        'num_layers': model.num_layers,
        'max_depth': model.cells[0].max_depth,
        'max_children': model.cells[0].max_children,
        'greedy_threshold': model.cells[0].greedy_threshold,
    },
    'metrics': {
        'val_perplexity': val_perplexity,
        'mean_grow_prob': mean_grow_prob,  # IMPORTANT for threshold tuning!
        'std_grow_prob': std_grow_prob
    }
}
```

### 11.2 Debugging Tools

**1. `--debug_policy` Flag** (CRITICAL):
```python
def debug_policy(model, val_loader, device):
    """
    Measure policy distribution on validation set.
    
    Returns mean and std of grow_prob across all decisions.
    """
    model.eval()
    all_grow_probs = []
    
    with torch.no_grad():
        for batch in val_loader:
            tokens = batch['tokens'].to(device)
            
            # Forward pass (collect grow_probs)
            # ... (instrument BFSLanguageCell to return grow_probs)
            
            all_grow_probs.extend(grow_probs)
    
    mean_grow_prob = np.mean(all_grow_probs)
    std_grow_prob = np.std(all_grow_probs)
    
    print(f"Policy Distribution:")
    print(f"  Mean: {mean_grow_prob:.4f}")
    print(f"  Std:  {std_grow_prob:.4f}")
    print(f"  Min:  {np.min(all_grow_probs):.4f}")
    print(f"  Max:  {np.max(all_grow_probs):.4f}")
    print(f"  % >= 0.5: {100 * np.mean(np.array(all_grow_probs) >= 0.5):.2f}%")
    
    # Recommend threshold
    recommended_threshold = mean_grow_prob - 0.03
    print(f"\nRecommended greedy_threshold: {recommended_threshold:.2f}")
    
    return mean_grow_prob, std_grow_prob
```

**2. Perplexity Tracking**:
```python
def compute_perplexity(model, data_loader, device):
    """Compute perplexity on dataset."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in data_loader:
            tokens = batch['tokens'].to(device)
            targets = batch['targets'].to(device)
            
            logits, _, _ = model(tokens)
            
            loss = F.cross_entropy(
                logits.view(-1, model.vocab_size),
                targets.view(-1),
                reduction='sum'
            )
            
            total_loss += loss.item()
            total_tokens += targets.numel()
    
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    
    return perplexity
```

**3. Text Generation Quality Check**:
```python
def generate_samples(model, tokenizer, prompts, device):
    """Generate text from multiple prompts for qualitative evaluation."""
    model.eval()
    
    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        
        generated_text = inference_greedy(
            model, prompt, tokenizer, max_new_tokens=100, device=device
        )
        
        print(f"Generated: {generated_text}")
        print("-" * 80)
```

---

## 12. Design Decisions & Rationale

### 12.1 Why Recurrent BFS (Not Sequence BFS)?

**Recurrent BFS** (chosen):
```
Token 1 → BFS Tree → h_1
Token 2 → BFS Tree (uses h_1) → h_2
Token 3 → BFS Tree (uses h_2) → h_3
```

**Sequence BFS** (rejected):
```
All tokens → Single large BFS tree → Output
```

**Rationale**:
1. **Memory**: Recurrent fits in GPU memory (O(seq_len × hidden_dim))
2. **Flexibility**: Variable-length sequences (no padding issues)
3. **Natural**: Matches RNN/LSTM paradigm (proven for sequences)
4. **Efficiency**: Can process long sequences without quadratic blowup

**Trade-off**: Recurrent loses some parallelism (can't process all tokens at once like transformers), but gains memory efficiency.

### 12.2 Why Stack Cells (Not Single Cell)?

**Stacked** (chosen):
```
Layer 1: token → BFS → h_1
Layer 2: h_1 → BFS → h_2
Layer 3: h_2 → BFS → h_3
Layer 4: h_3 → BFS → h_4 → output
```

**Single Cell** (rejected):
```
Layer 1: token → BFS → output
```

**Rationale**:
1. **Expressiveness**: Deeper networks learn better representations
2. **Hierarchical**: Each layer can learn different abstractions
3. **Standard Practice**: All successful LLMs are deep (12-96 layers)
4. **From BFSNet**: Single BFS layer worked for vision, but language is harder

**Phase 1**: Start with 4 layers (manageable, enough depth)  
**Phase 2+**: Scale to 6-24 layers (GPT-2 small has 12)

### 12.3 Why Character-Level First?

**Phase 1: Character-Level** (chosen):
- **Pros**: Simple, small vocab (256), fits on CPU, $0 cost, fast iteration
- **Cons**: Long sequences (War and Peace = 3M chars), slower convergence

**Why Not Word-Level First?**:
- Requires BPE training (extra complexity)
- Larger vocab (50K) → bigger model
- Needs GPU ($500 cost)
- Slower iteration if design is wrong

**Strategy**: Validate architecture on char-level (cheap, fast), then scale to word-level (better quality).

### 12.4 Why REINFORCE (Not PPO/A3C)?

**REINFORCE** (chosen):
- BFSNet validated it works
- Simple, well-understood
- No extra complexity (value networks, etc.)

**Why Not PPO?**:
- More complex (clipping, value network, multiple updates)
- BFSNet didn't need it
- Can try in Phase 2+ if REINFORCE struggles

**Why Not A3C?**:
- Requires multi-process training (complex)
- REINFORCE is simpler
- Can try if needed for stability

### 12.5 Design Philosophy

**From BFSNet Success**:
1. ✅ Keep what worked (REINFORCE, policy nets, efficiency penalties)
2. ✅ Fix what didn't (threshold tuning, sample-independent rewards)
3. ✅ Start simple (character-level, 4 layers, mean pooling)
4. ✅ Iterate based on data (threshold sweeps, λ sweeps)

**Avoid Over-Engineering**:
- Don't add complexity until proven necessary
- Measure first, optimize second
- Simple baselines (LSTM, root-only) are essential

---

## 13. Extensions & Future Work

### 13.1 Phase 2+ Extensions

**Word-Level (Phase 2)**:
- BPE tokenization (GPT-2 vocab)
- Larger model (25M → 125M params)
- Scaled training (multi-GPU)

**Production Scale (Phase 3)**:
- Distributed training (DDP, DeepSpeed)
- Mixed precision (FP16/BF16)
- Flash Attention optimizations
- Gradient checkpointing

**Arcus LLM (Phase 4)**:
- Instruction tuning (RLHF/DPO)
- Tool use (function calling)
- Multi-modal (vision + language)
- Inference optimization (quantization, ONNX)

### 13.2 Research Directions

**Adaptive Thresholds**:
- Learnable threshold (trainable parameter)
- Per-layer thresholds (depth-varying)
- Per-token thresholds (input-dependent)

**Alternative Pooling**:
- Attention-based pooling
- Gated pooling (like LSTM forget gate)
- Hierarchical pooling (tree-structured)

**Hybrid Architectures**:
- BFS + Attention (best of both)
- BFS + MoE (adaptive routing)
- BFS + RNN (explicit memory)

**Efficiency Improvements**:
- Sparse attention within trees
- Dynamic batching (variable tree sizes)
- Cached tree structures (for common patterns)

---

## 14. Validation Checklist

Before declaring Phase 1 complete, verify:

**Architecture**:
- [ ] BFSLanguageCell processes tokens correctly
- [ ] Hidden states propagate across timesteps
- [ ] Policy networks output valid probabilities [0, 1]
- [ ] Gradients flow through all layers
- [ ] No NaN/Inf in any tensor

**Training**:
- [ ] Loss decreases over epochs
- [ ] Perplexity improves on validation set
- [ ] Policy converges (grow_prob stable)
- [ ] No gradient explosion/vanishing
- [ ] Checkpoints save/load correctly

**Inference**:
- [ ] Text generation produces valid characters
- [ ] Generated text is somewhat coherent
- [ ] Greedy threshold tuning improves quality
- [ ] Latency is acceptable (< 100ms per token on CPU)

**Comparison**:
- [ ] Perplexity ≤ LSTM baseline (or within 10%)
- [ ] FLOPs reduction demonstrated (30%+ vs full tree)
- [ ] Root-only baseline tested (validate depth is needed)

**Code Quality**:
- [ ] Unit tests pass (tokenization, BFSLanguageCell, etc.)
- [ ] Integration test passes (full training run)
- [ ] `--debug_policy` flag works
- [ ] Documentation updated

---

## 15. References

### 15.1 BFSNet Foundation
- **BFSNet v2.0.0**: `docs/bfsnet_architecture.md`
- **Lessons Learned**: `docs/bfsnet_lessons_learned.md`
- **Test Results**: `tests/bfsnet/RESULTS.md`

### 15.2 Language Modeling
- **Transformer**: Vaswani et al., 2017
- **GPT-2**: Radford et al., 2019
- **nanoGPT**: Andrej Karpathy (reference implementation)

### 15.3 REINFORCE
- **Original**: Williams, 1992
- **Deep RL**: Sutton & Barto, 2018

---

**Document Version**: 0.1.0  
**Last Updated**: December 20, 2025  
**Status**: Phase 1 Specification - Ready for Implementation  
**Next**: Begin coding BFSLanguageCell

**⚠️ Proprietary Software**: This project is closed source. All rights reserved.