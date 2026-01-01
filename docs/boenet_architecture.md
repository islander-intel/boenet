# BoeNet Architecture Documentation

**Version:** 2.1.0  
**Last Updated:** December 31, 2025

## Table of Contents

1. [Overview](#overview)
2. [Binary Reasoning Tree](#binary-reasoning-tree)
3. [Why K=2 is Optimal](#why-k2-is-optimal)
4. [Depth vs Width Trade-off](#depth-vs-width-trade-off)
5. [Sparsity Findings](#sparsity-findings)
6. [Tree Visualizations](#tree-visualizations)
7. [Future: Dynamic Depth Algorithm](#future-dynamic-depth-algorithm)
8. [Technical Implementation](#technical-implementation)

---

## Overview

BoeNet (Binary Optimal Expansion Network) is a language model that uses **learned BFS tree expansion** for next-token prediction. Instead of fixed computation, BoeNet dynamically decides how much to "think" about each prediction.

### Core Concept

```
Traditional LLM:  Input → Fixed Computation → Output
BoeNet:           Input → Adaptive Tree Expansion → Output
```

The model learns TWO things:
1. **What to predict** (language modeling)
2. **When to think deeper** (policy learning via REINFORCE)

---

## Binary Reasoning Tree

### Why Binary (K=2)?

After extensive experiments, **K=2 (binary branching)** emerged as optimal:

| K Value | Description | Best PPL | Verdict |
|---------|-------------|----------|---------|
| K=0 | Dense (no tree) | 11.55 | Baseline |
| **K=2** | **Binary tree** | **11.64** | **✅ Winner** |
| K=3 | Ternary tree | 11.63 | 3.4x slower |
| K=4 | Quaternary tree | 11.70 | Worst quality |

### Binary Reasoning Interpretation

K=2 creates a **natural decision structure**:

```
Level 0: Initial thought
         ↓
Level 1: Yes / No (binary decision)
         ↓
Level 2: Strong / Weak (confidence)
```

This mirrors human cognition:
- **System 1:** Quick intuition (root node)
- **System 2:** Deliberate reasoning (deeper levels)

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
| 4 | 31 | 16 | 5 |

---

## Why K=2 is Optimal

### 1. Training Speed

K=2 trains **3.4x faster** than K=3/K=4:

| K | Time/Epoch | Relative Speed |
|---|------------|----------------|
| K=0 | 13 sec | 7.3x faster |
| **K=2** | **95 sec** | **1x (baseline)** |
| K=3 | 320 sec | 3.4x slower |
| K=4 | 330 sec | 3.5x slower |

### 2. Inference Latency

K=2 achieves the fastest tree-based inference:

| K | Threshold | Nodes Used | Latency |
|---|-----------|------------|---------|
| K=0 | N/A | 1.0 | 0.24 ms |
| **K=2** | **0.50** | **1.47** | **0.89 ms** |
| K=3 | 0.40 | 13.0 | 1.78 ms |
| K=4 | 0.40 | 21.0 | 3.72 ms |

### 3. Sparsity Handling

K=2 handles sparsity **gracefully** - PPL improves with sparsity!

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

### The Pivot: Depth Matters More Than K

**Key Insight:** All K values (2, 3, 4) plateau at similar PPL (~11.63-11.70) when using depth=2.

This suggests the bottleneck is **reasoning depth**, not branching factor.

### Depth Comparison (K=2)

| Depth | Max Nodes | Reasoning Levels | Training Time | Expected PPL |
|-------|-----------|------------------|---------------|--------------|
| 2 | 7 | 3 | ~95 sec/ep | 11.64 (measured) |
| 3 | 15 | 4 | ~150 sec/ep | ~11.58 (testing) |
| 4 | 31 | 5 | ~250 sec/ep | ~11.54 (testing) |

### The Hypothesis

```
More Depth = More Reasoning Levels = Better Predictions
```

Even though K=2 depth=4 has 31 nodes, binary efficiency keeps it fast:
- K=2 depth=4 (31 nodes): ~250 sec/epoch
- K=3 depth=2 (13 nodes): ~320 sec/epoch

**K=2 depth=4 is still faster than K=3 depth=2!**

### When to Use More Depth

| Task Complexity | Recommended Depth | Nodes | Reasoning |
|-----------------|-------------------|-------|-----------|
| Simple patterns | 2 | 7 | 3 levels |
| Medium complexity | 3 | 15 | 4 levels |
| Complex reasoning | 4 | 31 | 5 levels |

---

## Sparsity Findings

### How Sparsity Works

1. **Training:** Model uses stochastic policy (Bernoulli sampling)
2. **Inference:** Model uses greedy policy (threshold-based)
3. **Sparsity:** Percentage of nodes NOT expanded

### Threshold Effect

| Threshold | What Passes | Result |
|-----------|-------------|--------|
| 0.40 | grow_prob ≥ 0.40 | Full tree |
| 0.50 | grow_prob ≥ 0.50 | Sparse tree |
| 0.60 | grow_prob ≥ 0.60 | Root only |

### K=2 Sparsity Results

| Epochs | Threshold | Nodes Used | Sparsity | PPL |
|--------|-----------|------------|----------|-----|
| 10 | 0.50 | 2.07 | 70.5% | 11.68 |
| 15 | 0.50 | 1.47 | **79.1%** | **11.64** |

**Key Finding:** More training → Better sparsity → Same or better PPL!

### Why Sparsity Improves PPL

The policy learns to expand **only when necessary**:
- Easy predictions: Root only (1 node)
- Hard predictions: Full tree (7 nodes)

This acts as **implicit regularization**, preventing overfitting.

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

### K=2, Depth=3 (15 nodes, 4 reasoning levels)

```
                         Root
                        /    \
                      Yes    No
                     /  \   /  \
                    SY  WY SN  WN
                   /\ /\ /\ /\
                  8 great-grandchildren (Level 3: Evidence)
```

### K=2, Depth=4 (31 nodes, 5 reasoning levels)

```
Level 0: Root               (1 node)
Level 1: Yes / No           (2 nodes)
Level 2: Confidence         (4 nodes)
Level 3: Evidence           (8 nodes)
Level 4: Context/Refinement (16 nodes)
─────────────────────────────────────
Total:                      31 nodes
```

### Sparse Tree (threshold=0.50)

With 79% sparsity, only ~1.5 nodes used on average:

```
        ┌─────────────┐
        │    Root     │ ← Always computed
        │   (h₀)      │
        └──────┬──────┘
               │
               ▼
    (Policy decides: expand?)
               │
        ┌──────┴──────┐
        │ grow_prob   │
        │  = 0.48     │ ← Below threshold 0.50
        └─────────────┘
               │
               ▼
        STOP (no expansion)
        Use root prediction
```

---

## Future: Dynamic Depth Algorithm

### Current Limitation

**Fixed depth for all inputs:**
```python
max_depth = 4  # Same for easy AND hard predictions
```

### Proposed Solution

**Learned adaptive depth:**
```python
depth = 0
while not confident_enough(prediction) and depth < max_depth:
    expand_next_level()
    depth += 1
# Easy inputs: stop early (depth 1-2)
# Hard inputs: go deep (depth 4-5)
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
**Cons:** More complex training

#### Option C: Budget-Based Stopping
```python
def forward(x, compute_budget=10):
    nodes_used = 0
    while nodes_used < compute_budget:
        expand_next_node()
        nodes_used += 1
```

**Pros:** Guarantees latency
**Cons:** Not input-adaptive

### Research Questions

1. Does dynamic depth improve quality?
2. Does it reduce compute for easy inputs?
3. Can we learn when to think deeper?
4. How does this compare to transformers?

### Expected Benefits

| Metric | Fixed Depth | Dynamic Depth |
|--------|-------------|---------------|
| Easy input latency | Same as hard | **Faster** |
| Hard input quality | Same as easy | **Better** |
| Average compute | Fixed | **Adaptive** |

---

## Technical Implementation

### Core Components

#### 1. BoeNet Model (`boenet/model.py`)

```python
class BoeNet(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, 
                 max_depth, max_children, greedy_threshold):
        # Token embedding
        self.embed = nn.Embedding(vocab_size, embed_dim)
        
        # Hidden state projection
        self.proj = nn.Linear(embed_dim, hidden_dim)
        
        # Child generation
        self.child_net = nn.Linear(hidden_dim, hidden_dim * max_children)
        
        # Growth policy
        self.growth_policy = GrowthPolicyNet(hidden_dim)
        
        # Output projection
        self.output = nn.Linear(hidden_dim, vocab_size)
```

#### 2. Growth Policy (`boenet/utils/gating.py`)

```python
class GrowthPolicyNet(nn.Module):
    def forward(self, hidden, depth):
        # Input: node hidden state + depth embedding
        # Output: probability of expanding this node
        logits = self.mlp(hidden)
        logits_clamped = logits.clamp(-20.0, 20.0)  # v2.1.0 fix
        grow_prob = torch.sigmoid(logits_clamped)
        grow_prob = grow_prob.clamp(1e-7, 1-1e-7)   # v2.1.0 fix
        return grow_prob
```

#### 3. Training Loop (`train_boenet.py`)

```python
# Forward with policy gradients
outputs, policy_loss, rewards, node_counts = model(
    input_ids,
    num_rollouts=3,
    lambda_efficiency=0.05,
    beta_entropy=0.01,
    labels=labels,
)

# Language modeling loss
lm_loss = F.cross_entropy(outputs, labels)

# Total loss
total_loss = lm_loss + beta_policy * policy_loss

# Backward
total_loss.backward()
```

### Numerical Stability (v2.1.0)

All probability operations are clamped:

```python
# Probability clamping constants
PROB_CLAMP_MIN = 1e-7      # Prevents log(0)
PROB_CLAMP_MAX = 1 - 1e-7  # Prevents log(0)
LOGIT_CLAMP_MIN = -20.0    # Prevents sigmoid underflow
LOGIT_CLAMP_MAX = 20.0     # Prevents sigmoid overflow
REWARD_SCALE = 5.0         # Keeps gradients manageable
ADVANTAGE_CLAMP = 2.0      # Prevents gradient explosion
```

### Checkpoint Format

```python
checkpoint = {
    "model_state_dict": model.state_dict(),
    "config": {
        "vocab_size": 256,
        "embed_dim": 64,
        "hidden_dim": 128,
        "max_depth": 2,
        "max_children": 2,
        "greedy_threshold": 0.50,
        "version": "2.1.0",
        "model_type": "language",
    },
    "training_meta": {
        "best_val_ppl": 11.64,
        "best_epoch": 15,
        "total_time_s": 1440.0,
    },
}
```

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

---

## Changelog

### v2.1.0 (December 31, 2025)
- Added K=2 as optimal finding
- Added depth vs width analysis
- Added sparsity findings
- Added tree visualizations
- Added dynamic depth roadmap

### v2.0.0 (December 30, 2025)
- Converted from vision to language
- Added CUDA stability fixes
- Added greedy threshold documentation

### v1.0.0 (December 2025)
- Initial architecture documentation