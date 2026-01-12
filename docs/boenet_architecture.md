# BoeNet Architecture Documentation

**Version:** 4.1.0  
**Last Updated:** January 5, 2026

## Table of Contents

1. [Overview](#overview)
2. [Architecture Evolution](#architecture-evolution)
3. [Binary Reasoning Tree](#binary-reasoning-tree)
4. [BPE Tokenizer Integration (v4.0.0)](#bpe-tokenizer-integration-v400)
5. [Parameter Scaling](#parameter-scaling)
6. [Why K=2 is Optimal](#why-k2-is-optimal)
7. [Depth vs Width Trade-off](#depth-vs-width-trade-off)
8. [Sparsity Findings](#sparsity-findings)
9. [Tree Visualizations](#tree-visualizations)
10. [Technical Implementation](#technical-implementation)
11. [Future: Dynamic Depth Algorithm](#future-dynamic-depth-algorithm)

---

## Overview

BoeNet (Binary Optimal Expansion Network) is a language model that uses **learned BFS tree expansion** for next-token prediction. Instead of fixed computation, BoeNet dynamically decides how much to "think" about each prediction.

### Core Concept

```
Traditional LLM:  Input → Fixed Computation → Output
BoeNet:           Input → Adaptive Tree Expansion → Output
```

The model learns TWO things simultaneously:
1. **What to predict** (language modeling via cross-entropy loss)
2. **When to think deeper** (policy learning via REINFORCE)

### Key Properties

| Property | Description |
|----------|-------------|
| Adaptive Compute | 1-31 nodes per token based on complexity |
| Binary Branching | K=2 optimal for speed and quality |
| Learned Sparsity | Model learns when NOT to expand |
| Tree-Structured | Hierarchical reasoning representation |

---

## Architecture Evolution

| Version | Tokenizer | Vocab | Hidden | Parameters | Best PPL |
|---------|-----------|-------|--------|------------|----------|
| v1.0-v2.1 | Character | 256 | 128 | ~150K | 11.64 |
| v4.0.0 | BPE | 100,277 | 644 | 72.2M | 534.60 |
| **v4.1.0** | **BPE** | **100,277** | **644** | **72.2M** | **279.08** |

### Why the Jump?

- **v2.x → v4.0.0**: BPE tokenizer integration required massive parameter scaling
- **v4.0.0 → v4.1.0**: Same architecture, more training (5 → 30 epochs)

---

## Binary Reasoning Tree

### Why Binary (K=2)?

After extensive experiments, **K=2 (binary branching)** emerged as optimal:

| K Value | Description | Best PPL | Training Speed | Verdict |
|---------|-------------|----------|----------------|---------|
| K=0 | Dense (no tree) | 11.55 | 13 sec/ep | Baseline |
| **K=2** | **Binary tree** | **11.64** | **95 sec/ep** | **✅ Winner** |
| K=3 | Ternary tree | 11.63 | 320 sec/ep | 3.4× slower |
| K=4 | Quaternary tree | 11.70 | 330 sec/ep | Worst quality |

### Binary Reasoning Interpretation

K=2 creates a **natural decision structure**:

```
Level 0: Initial thought (root)
         ↓
Level 1: Yes / No (binary decision)
         ↓
Level 2: Strong / Weak (confidence)
         ↓
Level 3: Evidence A / Evidence B
         ↓
Level 4: Context refinement
```

This mirrors human cognition:
- **System 1:** Quick intuition (root node, 1 computation)
- **System 2:** Deliberate reasoning (deeper levels, more computation)

### Mathematical Structure

For K=2 with depth D:
- **Total nodes:** 2^(D+1) - 1
- **Leaf nodes:** 2^D
- **Reasoning levels:** D + 1

| Depth | Total Nodes | Leaf Nodes | Reasoning Levels |
|-------|-------------|------------|------------------|
| 1 | 3 | 2 | 2 |
| 2 | 7 | 4 | 3 |
| 3 | 15 | 8 | 4 |
| **4** | **31** | **16** | **5** |
| 5 | 63 | 32 | 6 |

---

## BPE Tokenizer Integration (v4.0.0)

### Why BPE?

Character-level tokenization (vocab=256) limits model expressiveness:

| Limitation | Character-Level | BPE |
|------------|-----------------|-----|
| Vocabulary | 256 bytes | 100,277 subwords |
| "Hello" | 5 tokens | 1 token |
| Parameters | ~150K | 72M+ |
| Scalability | Poor | Production-ready |

### Tokenizer Comparison

| Tokenizer | Vocab Size | "Hello World" | Tokens |
|-----------|------------|---------------|--------|
| CharTokenizer | 256 | `[72,101,108,108,111,32,87,111,114,108,100]` | 11 |
| **cl100k_base** | **100,277** | `[9906,4435]` | **2** |
| gpt2 | 50,257 | `[15496,2159]` | 2 |

### Implementation

```python
# boenet/tokenizer.py v1.1.0

from boenet.tokenizer import get_tokenizer, TiktokenWrapper

# Factory function
def get_tokenizer(tokenizer_type="bpe", encoding_name="cl100k_base"):
    """Get tokenizer by type."""
    if tokenizer_type == "bpe":
        return TiktokenWrapper(encoding_name)
    elif tokenizer_type == "char":
        return CharTokenizer()
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")

# BPE Wrapper Class
class TiktokenWrapper:
    def __init__(self, encoding_name="cl100k_base"):
        self._encoding = tiktoken.get_encoding(encoding_name)
        self._vocab_size = self._encoding.n_vocab  # 100,277
    
    @property
    def vocab_size(self) -> int:
        return self._vocab_size
    
    def encode(self, text: str) -> List[int]:
        return self._encoding.encode(text)
    
    def decode(self, tokens: List[int]) -> str:
        return self._encoding.decode(tokens)
```

### Training Integration

```python
# train_boenet.py v4.0.0
from boenet.tokenizer import get_tokenizer

# Initialize tokenizer
tokenizer = get_tokenizer("bpe", "cl100k_base")

# Get data loaders with tokenizer
train_loader, val_loader, vocab_size = get_dataloaders(
    dataset_name="wikitext2",
    batch_size=16,
    seq_len=128,
    tokenizer=tokenizer,  # Pass BPE tokenizer
)

# vocab_size will be 100,277
```

### Inference Integration (v4.1.0 Fix)

```python
# infer_boenet.py v4.1.0 - BPE ONLY
from boenet.tokenizer import get_tokenizer, TiktokenWrapper

def initialize_tokenizer(cfg, bpe_encoding="cl100k_base"):
    """Initialize BPE tokenizer for inference."""
    tokenizer = get_tokenizer("bpe", encoding_name=bpe_encoding)
    return tokenizer, tokenizer.vocab_size

# No more CharTokenizer fallback - this was the bug in v4.0.0
```

---

## Parameter Scaling

### Character-Level (v2.x)

```
vocab_size = 256
embed_dim = 64
hidden_dim = 128

Embedding:        256 × 64     =    16K   (10.7%)
Input Projection: 64 × 128     =     8K   (5.3%)
Node Transform:   128 × 128    =    16K   (10.7%)
Child Projection: 128 × 256    =    33K   (22.0%)
Output Head:      128 × 256    =    33K   (22.0%)
Growth Policy:    ~1K                     (0.7%)
Other:            ~43K                    (28.7%)
───────────────────────────────────────────────────
Total:                            ~150K
```

### BPE (v4.0.0+) - Current

```
vocab_size = 100,277
embed_dim = 64
hidden_dim = 644

Embedding:        100,277 × 64  =   6.4M   (8.9%)
Input Projection: 64 × 644      =    41K   (0.1%)
Node Transform:   644 × 644     =   415K   (0.6%)
Child Projection: 644 × 1,288   =   830K   (1.1%)
Output Head:      644 × 100,277 =  64.6M   (89.4%)
Growth Policy:    ~5K                      (0.0%)
───────────────────────────────────────────────────
Total:                            72.23M
Model File Size:                  288.9 MB
```

### Where Parameters Go

| Component | Char (150K) | BPE (72M) | Scaling Factor |
|-----------|-------------|-----------|----------------|
| Embedding | 16K | 6.4M | 400× |
| **Output Head** | 33K | **64.6M** | **1,958×** |
| Tree Logic | 57K | 1.3M | 23× |
| **Total** | 150K | 72.2M | **481×** |

**Key Insight:** The output head dominates parameter count. With vocab=100,277, projecting from hidden_dim to vocab requires `hidden_dim × vocab_size` parameters.

### Memory Bottleneck

The bottleneck during training is the **logits tensor**, not model parameters:

```
logits shape: [batch_size, seq_len, vocab_size]

BPE:  [16, 128, 100,277] = 205M floats = 820MB per batch
Char: [64, 128, 256]     = 2M floats   = 8MB per batch
```

This is why batch_size must be reduced for BPE (64 → 16 → 8).

---

## Why K=2 is Optimal

### 1. Training Speed

K=2 trains **3.4× faster** than K=3/K=4:

| K | Time/Epoch (Char) | Relative Speed |
|---|-------------------|----------------|
| K=0 | 13 sec | 7.3× faster |
| **K=2** | **95 sec** | **1× (baseline)** |
| K=3 | 320 sec | 3.4× slower |
| K=4 | 330 sec | 3.5× slower |

### 2. Inference Latency

K=2 achieves the fastest tree-based inference:

| K | Threshold | Nodes Used | Latency |
|---|-----------|------------|---------|
| K=0 | N/A | 1.0 | 0.24 ms |
| **K=2** | **0.50** | **1.47** | **0.89 ms** |
| K=3 | 0.40 | 13.0 | 1.78 ms |
| K=4 | 0.40 | 21.0 | 3.72 ms |

### 3. Sparsity Handling

K=2 handles sparsity **gracefully** - PPL actually improves with sparsity!

| K | Full Tree PPL | Sparse PPL | Δ PPL |
|---|---------------|------------|-------|
| **K=2** | 11.66 | **11.64** | **-0.02 (better!)** |
| K=4 | 11.70 | 11.91 | +0.21 (worse) |

K=4 quality **degrades** when sparse. K=2 quality **improves**.

### 4. GPU Efficiency

Binary operations align with GPU architecture:

| K | CUDA Warp (32 threads) | Efficiency |
|---|------------------------|------------|
| **K=2** | 32/2 = 16 pairs | **100%** |
| K=3 | 32/3 = 10.67 | 67% (wasted) |
| K=4 | 32/4 = 8 quads | 100% |

K=2 and K=4 are both GPU-aligned, but K=2 is faster and better quality.

---

## Depth vs Width Trade-off

### Key Insight

All K values (K=2, 3, 4) plateau at similar PPL (~11.63-11.70) with depth=2.

| K | Depth | Max Nodes | Best PPL |
|---|-------|-----------|----------|
| K=2 | 2 | 7 | 11.64 |
| K=3 | 2 | 13 | 11.63 |
| K=4 | 2 | 21 | 11.70 |

**Hypothesis:** The bottleneck is **depth** (reasoning levels), not **K** (branching factor).

### Depth Comparison (K=2)

| Depth | Max Nodes | Reasoning Levels | Use Case |
|-------|-----------|------------------|----------|
| 2 | 7 | 3 | Simple patterns |
| 3 | 15 | 4 | Medium complexity |
| **4** | **31** | **5** | **Complex reasoning (current)** |

### Current Configuration (v4.0.0+)

```yaml
max_depth: 4
max_children: 2
theoretical_max_nodes: 31
valid_node_counts: [1, 3, 7, 15, 31]
```

### Binary Efficiency

Even with more depth, K=2 remains efficient:

| Configuration | Nodes | Training Time |
|---------------|-------|---------------|
| K=2 depth=4 | 31 | ~250 sec/epoch |
| K=3 depth=2 | 13 | ~320 sec/epoch |

**K=2 depth=4 is still faster than K=3 depth=2!**

---

## Sparsity Findings

### How Sparsity Works

1. **Training:** Model uses stochastic policy (Bernoulli sampling from grow_prob)
2. **Inference:** Model uses greedy policy (threshold-based: expand if grow_prob ≥ threshold)
3. **Sparsity:** Percentage of nodes NOT expanded

### Threshold Effect

| Threshold | What Passes | Result |
|-----------|-------------|--------|
| 0.40 | grow_prob ≥ 0.40 | Full tree (all nodes) |
| **0.50** | **grow_prob ≥ 0.50** | **Sparse tree (selected nodes)** |
| 0.60 | grow_prob ≥ 0.60 | Very sparse (mostly root) |

### K=2 Sparsity Results (Character-Level)

| Epochs | Threshold | Nodes Used | Sparsity | PPL |
|--------|-----------|------------|----------|-----|
| 10 | 0.50 | 2.07 | 70.5% | 11.68 |
| 15 | 0.50 | 1.47 | **79.1%** | **11.64** |

**Key Finding:** More training → Better sparsity → Same or better PPL!

### BPE Sparsity Observations (v4.0.0)

From the training logs:
```
Level 0: GREEDY mode, prob=0.9852, threshold=0.5, expand=True
Level 0: EXPANDED -> 2 children at level 1
Level 1: GREEDY mode, prob=0.0000, threshold=0.5, expand=False
Level 1: NOT EXPANDING - stopping tree growth
FINAL: depth=1, total_nodes=3, nodes_per_level=[1, 2]
```

The BPE model learned:
- Level 0 → Level 1: **98.5% probability** (always expand)
- Level 1 → Level 2: **0.0% probability** (never expand)
- Result: Consistently uses **3 nodes** (depth=1)

### Why Sparsity Improves Quality

The policy learns to expand **only when necessary**:
- Easy predictions: Root only (1 node)
- Medium predictions: Root + children (3 nodes)
- Hard predictions: Full tree (7-31 nodes)

This acts as **implicit regularization**, preventing overfitting on easy examples.

---

## Tree Visualizations

### K=2, Depth=2 (7 nodes, 3 reasoning levels)

```
                    ┌─────────────┐
                    │    Root     │ Level 0: Initial Thought
                    │   (h₀)      │
                    └──────┬──────┘
                           │
              ┌────────────┴────────────┐
              │                         │
        ┌─────┴─────┐             ┌─────┴─────┐
        │  Child 0  │             │  Child 1  │ Level 1: Yes / No
        │   (Yes)   │             │   (No)    │
        └─────┬─────┘             └─────┬─────┘
              │                         │
        ┌─────┴─────┐             ┌─────┴─────┐
        │           │             │           │
   ┌────┴────┐ ┌────┴────┐  ┌────┴────┐ ┌────┴────┐
   │ GC 0    │ │ GC 1    │  │ GC 2    │ │ GC 3    │ Level 2: Confidence
   │(Strong) │ │(Weak)   │  │(Strong) │ │(Weak)   │
   │  Yes    │ │ Yes     │  │  No     │ │  No     │
   └─────────┘ └─────────┘  └─────────┘ └─────────┘
```

### K=2, Depth=4 (31 nodes, 5 reasoning levels) - Current

```
Level 0: Root                (1 node)   - Initial token representation
Level 1: Yes / No            (2 nodes)  - Primary binary decision
Level 2: Confidence          (4 nodes)  - Strong / Weak signals
Level 3: Evidence            (8 nodes)  - Supporting information
Level 4: Context/Refinement  (16 nodes) - Final adjustments
─────────────────────────────────────────────────────────────────
Total:                       31 nodes maximum
With learned sparsity:       ~3 nodes average (current)
```

### Sparse Tree Example (threshold=0.50)

What actually happens during inference:

```
        ┌─────────────┐
        │    Root     │ ← Always computed
        │   (h₀)      │
        └──────┬──────┘
               │
               ▼
    grow_prob = 0.985 → EXPAND (above 0.50)
               │
        ┌──────┴──────┐
        │             │
   ┌────┴────┐  ┌────┴────┐
   │ Child 0 │  │ Child 1 │ Level 1
   └────┬────┘  └────┬────┘
        │             │
        ▼             ▼
   grow_prob=0.0  grow_prob=0.0
   → STOP          → STOP
   (below 0.50)    (below 0.50)

Final prediction: Pool outputs from 3 nodes
Total nodes used: 3 (90% sparsity vs max 31)
```

---

## Technical Implementation

### Core Components

#### 1. BoeNet Model (`boenet/model.py` v2.4.0)

```python
class BoeNet(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, 
                 max_depth, max_children, greedy_threshold):
        super().__init__()
        
        # Token embedding (scales with vocab)
        self.embed = nn.Embedding(vocab_size, embed_dim)
        
        # Hidden state projection
        self.input_proj = nn.Linear(embed_dim, hidden_dim)
        
        # Node transformation (shared across all tree nodes)
        self.node_transform = nn.Linear(hidden_dim, hidden_dim)
        
        # Child generation (creates K children per node)
        self.child_proj = nn.Linear(hidden_dim, hidden_dim * max_children)
        
        # Growth policy network (decides expand vs stop)
        self.growth_policy = GrowthPolicyNet(hidden_dim)
        
        # Output projection (scales with vocab)
        self.output = nn.Linear(hidden_dim, vocab_size)
        
        # Configuration
        self.max_depth = max_depth
        self.max_children = max_children
        self.greedy_threshold = greedy_threshold
```

#### 2. Tokenizer Module (`boenet/tokenizer.py` v1.1.0)

```python
import tiktoken
from abc import ABC, abstractmethod
from typing import List

class BaseTokenizer(ABC):
    @property
    @abstractmethod
    def vocab_size(self) -> int: ...
    
    @abstractmethod
    def encode(self, text: str) -> List[int]: ...
    
    @abstractmethod
    def decode(self, tokens: List[int]) -> str: ...

class TiktokenWrapper(BaseTokenizer):
    """BPE tokenizer using tiktoken (cl100k_base = GPT-4 tokenizer)."""
    
    def __init__(self, encoding_name: str = "cl100k_base"):
        self._encoding = tiktoken.get_encoding(encoding_name)
        self._vocab_size = self._encoding.n_vocab
    
    @property
    def vocab_size(self) -> int:
        return self._vocab_size
    
    @property
    def eos_token_id(self) -> int:
        return self._encoding.eot_token
    
    def encode(self, text: str) -> List[int]:
        return self._encoding.encode(text)
    
    def decode(self, tokens: List[int]) -> str:
        return self._encoding.decode(tokens)

def get_tokenizer(tokenizer_type: str = "bpe", 
                  encoding_name: str = "cl100k_base"):
    """Factory function for tokenizers."""
    if tokenizer_type == "bpe":
        return TiktokenWrapper(encoding_name)
    elif tokenizer_type == "char":
        return CharTokenizer()
    raise ValueError(f"Unknown tokenizer: {tokenizer_type}")
```

#### 3. Growth Policy (`boenet/utils/gating.py` v2.1.0)

```python
class GrowthPolicyNet(nn.Module):
    """Decides whether to expand a node or stop."""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, hidden: torch.Tensor, depth: int) -> torch.Tensor:
        # Input: node hidden state
        # Output: probability of expanding this node
        logits = self.mlp(hidden)
        
        # v2.1.0: Numerical stability - clamp logits
        logits_clamped = logits.clamp(-20.0, 20.0)
        
        # Convert to probability
        grow_prob = torch.sigmoid(logits_clamped)
        
        # v2.1.0: Clamp probabilities to avoid log(0)
        grow_prob = grow_prob.clamp(1e-7, 1 - 1e-7)
        
        return grow_prob
```

#### 4. Training Loop (`train_boenet.py` v4.0.0)

```python
# Initialize tokenizer
tokenizer = get_tokenizer("bpe", "cl100k_base")

# Get data loaders
train_loader, val_loader, vocab_size = get_dataloaders(
    dataset_name="wikitext2",
    batch_size=16,
    seq_len=128,
    tokenizer=tokenizer,
)

# Initialize model with BPE vocab size
model = BoeNet(
    vocab_size=vocab_size,  # 100,277 for BPE
    embed_dim=64,
    hidden_dim=644,
    max_depth=4,
    max_children=2,
    greedy_threshold=0.50,
)

# Training step
for input_ids, labels in train_loader:
    # Forward with policy gradients
    outputs, policy_loss, rewards, node_counts = model(
        input_ids,
        num_rollouts=3,
        lambda_efficiency=0.05,
        beta_entropy=0.01,
        labels=labels,
    )
    
    # Language modeling loss
    lm_loss = F.cross_entropy(outputs.view(-1, vocab_size), labels.view(-1))
    
    # Combined loss
    total_loss = lm_loss + beta_policy * policy_loss
    
    # Backward pass
    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
```

### Numerical Stability (v2.0.1+)

All probability operations use clamping to prevent NaN/Inf:

```python
# Probability clamping constants
PROB_CLAMP_MIN = 1e-7      # Prevents log(0)
PROB_CLAMP_MAX = 1 - 1e-7  # Prevents log(0) for 1-p
LOGIT_CLAMP_MIN = -20.0    # Prevents sigmoid underflow
LOGIT_CLAMP_MAX = 20.0     # Prevents sigmoid overflow
REWARD_SCALE = 5.0         # Keeps REINFORCE gradients manageable
ADVANTAGE_CLAMP = 2.0      # Prevents gradient explosion
```

### Checkpoint Format (v4.0.0+)

```python
checkpoint = {
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "config": {
        "vocab_size": 100277,
        "embed_dim": 64,
        "hidden_dim": 644,
        "max_depth": 4,
        "max_children": 2,
        "greedy_threshold": 0.50,
        "version": "4.0.0",
        "model_type": "language",
        # v4.0.0: Tokenizer info
        "tokenizer_type": "bpe",
        "bpe_encoding": "cl100k_base",
    },
    "training_meta": {
        "best_val_ppl": 279.08,
        "best_epoch": 30,
        "total_epochs": 30,
        "total_time_s": 10418.0,
    },
}
```

---

## Future: Dynamic Depth Algorithm

### Current Limitation

**Fixed max depth for all inputs:**
```python
max_depth = 4  # Same computation budget for easy AND hard tokens
```

The model learns to use fewer nodes, but the maximum is fixed.

### Proposed Solution

**Truly adaptive depth:**
```python
depth = 0
while not confident_enough(prediction) and depth < max_depth:
    expand_next_level()
    depth += 1
# Easy tokens: stop at depth 0-1
# Hard tokens: go to depth 3-4
```

### Implementation Options

#### Option A: Confidence-Based Stopping
```python
def should_stop(node_output, threshold=0.9):
    confidence = softmax(node_output).max()
    return confidence > threshold
```

**Pros:** Simple, interpretable  
**Cons:** Confidence ≠ correctness

#### Option B: Learned Stop Action
```python
# Policy outputs: [grow_prob, stop_prob]
action = policy(node_hidden)
if action == STOP:
    return prediction
else:
    expand_children()
```

**Pros:** End-to-end learned  
**Cons:** More complex training, credit assignment

#### Option C: Budget-Based
```python
def forward(x, compute_budget=10):
    nodes_used = 0
    while nodes_used < compute_budget:
        expand_next_node()
        nodes_used += 1
```

**Pros:** Guarantees latency SLA  
**Cons:** Not input-adaptive

### Expected Benefits

| Metric | Fixed Depth | Dynamic Depth |
|--------|-------------|---------------|
| Easy token latency | Same as hard | **Much faster** |
| Hard token quality | Same as easy | **Better** |
| Average compute | Fixed | **Lower** |
| Latency variance | Low | Higher |

---

## References

### Internal Documentation
- `docs/RESEARCH_FINDINGS.md` - Complete experiment results
- `configs/experiment-config.yaml` - Sweep configuration

### Related Work
- BFSNet: Original vision model (FashionMNIST)
- Adaptive Computation Time (Graves, 2016)
- Universal Transformers (Dehghani et al., 2018)
- PonderNet (Banino et al., 2021)
- GPT-2 (Radford et al., 2019) - Baseline comparison
- tiktoken - OpenAI's BPE tokenizer library

---

## Changelog

### v4.1.0 (January 5, 2026)
- Fixed tokenizer mismatch in inference (BPE-only mode)
- 30 epoch training: PPL 534 → 279
- Clean progress bars in training matrix

### v4.0.0 (January 3, 2026)
- Added BPE tokenizer integration (cl100k_base)
- Scaled to 72M parameters
- Added tokenizer module (boenet/tokenizer.py)
- Updated parameter breakdown documentation

### v2.4.0 (January 2, 2026)
- Enhanced tree expansion logging
- Per-level debug output

### v2.1.0 (December 31, 2025)
- K=2 identified as optimal
- Depth vs width analysis
- Sparsity findings documented
- Tree visualizations added

### v2.0.1 (December 30, 2025)
- CUDA stability fixes
- Probability clamping
- Logit clamping
- Reward scaling

### v2.0.0 (December 29, 2025)
- Converted from vision to language
- WikiText-2 support

### v1.0.0 (December 2025)
- Initial architecture