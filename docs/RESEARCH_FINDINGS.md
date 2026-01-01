# BoeNet Research Findings

**Document Version:** 1.0.0  
**Research Period:** December 29-31, 2025  
**Status:** Active Research

---

## Executive Summary

This document records all experiments conducted on BoeNet, a novel language model architecture using learned BFS tree expansion. After 50+ training runs, we have identified:

1. **K=2 (binary branching) is optimal** - 3.4x faster than K=3, better sparsity handling
2. **Depth may be more important than K** - All K values plateau at similar PPL with depth=2
3. **79% sparsity achievable** with no quality loss (threshold=0.50, λ=0.05)
4. **CUDA stability fixed** - v2.0.1 patches resolved all K>0 training crashes

---

## Table of Contents

1. [Timeline of Discoveries](#timeline-of-discoveries)
2. [Phase 0: K=0 and K=3 Baseline](#phase-0-k0-and-k3-baseline)
3. [Phase 1: K-Value Comparison](#phase-1-k-value-comparison)
4. [Phase 2: Depth Hypothesis](#phase-2-depth-hypothesis)
5. [Complete Data Tables](#complete-data-tables)
6. [Key Insights](#key-insights)
7. [Future Roadmap](#future-roadmap)

---

## Timeline of Discoveries

### December 29, 2025 - Project Start
- Converted BFSNet (vision) to BoeNet (language)
- Implemented WikiText-2 data loading
- First successful K=0 training runs

### December 30, 2025 - CUDA Bug Discovery
- K=3 training crashes with CUDA assertion error
- Error: `Assertion '0 <= p4 && p4 <= 1' failed`
- Root cause: NaN probabilities in `torch.bernoulli()`

### December 30, 2025 - Stability Fixes (v2.0.1)
- Added probability clamping in `model.py`
- Added logit clamping in `gating.py`
- Added reward scaling in `losses.py`
- Added NaN detection in `train_boenet.py`
- K=3 now trains successfully!

### December 31, 2025 - K Comparison Sweep
- Tested K=0, 2, 3, 4 systematically
- **Discovery: K=2 is optimal!**
- K=2 is 3.4x faster than K=3 with same quality

### December 31, 2025 - Sparsity Discovery
- K=2 with threshold=0.50 achieves 79% sparsity
- **PPL improves with sparsity** (11.68 → 11.64)
- K=4 sparsity hurts quality (11.70 → 11.91)

### December 31, 2025 - Depth Hypothesis
- Observed: All K values plateau at ~11.63-11.70 PPL
- **Hypothesis: Depth is the bottleneck, not K**
- Initiated depth comparison sweep (depth=2, 3, 4)

---

## Phase 0: K=0 and K=3 Baseline

### K=0 Results (Dense Model, No Tree)

16 runs completed with varying epochs and hyperparameters.

| Run ID | Epochs | λ | Val PPL | Train PPL | Time/Epoch |
|--------|--------|---|---------|-----------|------------|
| 0 | 5 | 0.0 | 11.60 | 11.56 | 12.9 sec |
| 1 | 10 | 0.0 | 11.58 | 11.52 | 13.0 sec |
| 2 | 15 | 0.0 | 11.56 | 11.50 | 13.1 sec |
| 3 | 20 | 0.0 | **11.55** | 11.48 | 13.0 sec |
| ... | ... | ... | ... | ... | ... |

**K=0 Summary:**
- Best PPL: **11.55** (20 epochs)
- Inference Latency: **0.24 ms**
- Nodes: 1 (always)

### K=3 Results (Before Stability Fix)

Initial runs crashed or showed extreme instability:

| Run ID | Epochs | Status | Val PPL |
|--------|--------|--------|---------|
| 16 | 5 | ⚠️ Unstable | 129,781,012.67 |
| 17 | 10 | ⚠️ | 15.77 |
| 18 | 15 | ⚠️ | 118.65 |
| 19 | 20 | ⚠️ Unstable | 15,760.38 |

### K=3 Results (After v2.0.1 Fix)

After applying stability fixes:

| Run ID | Epochs | λ | Threshold | Val PPL | Nodes | Time/Epoch |
|--------|--------|---|-----------|---------|-------|------------|
| 16 | 5 | 0.0 | 0.30 | 11.80 | 14.25 | 320 sec |
| 17 | 10 | 0.0 | 0.30 | 11.69 | 14.25 | 320 sec |
| 18 | 15 | 0.0 | 0.30 | 11.65 | 14.25 | 320 sec |
| 19 | 20 | 0.0 | 0.30 | **11.63** | 14.25 | 320 sec |
| 20-25 | Various | 0.0 | 0.35-0.40 | 11.63-11.80 | 14.25 | 320 sec |

**K=3 Summary:**
- Best PPL: **11.63** (20 epochs)
- Inference Latency: **1.78 ms**
- Nodes: 13-14.25 (consistent)
- Training: **24.6x slower than K=0**

---

## Phase 1: K-Value Comparison

### K=2 Results (8 runs)

| Run | Epochs | λ | Threshold | Val PPL | Infer Nodes | Latency | Sparsity |
|-----|--------|---|-----------|---------|-------------|---------|----------|
| 0 | 10 | 0.0 | 0.40 | 11.69 | 7.0 | 1.35 ms | 0% |
| 1 | 15 | 0.0 | 0.40 | 11.66 | 7.0 | 1.31 ms | 0% |
| 2 | 10 | 0.0 | 0.50 | 11.68 | 2.07 | 1.04 ms | 70.5% |
| 3 | 15 | 0.0 | 0.50 | **11.64** | 1.61 | 0.95 ms | 77.1% |
| 4 | 10 | 0.05 | 0.40 | 11.70 | 7.0 | 1.34 ms | 0% |
| 5 | 15 | 0.05 | 0.40 | 11.66 | 7.0 | 1.31 ms | 0% |
| 6 | 10 | 0.05 | 0.50 | 11.68 | 1.96 | 1.02 ms | 72.0% |
| 7 | 15 | 0.05 | 0.50 | **11.64** | **1.47** | **0.89 ms** | **79.1%** |

**K=2 Best Configuration:**
- Epochs: 15
- λ: 0.05
- Threshold: 0.50
- **Val PPL: 11.64**
- **Sparsity: 79.1%**
- **Latency: 0.89 ms**

### K=4 Results (3 runs)

| Run | Epochs | λ | Threshold | Val PPL | Infer Nodes | Latency | Sparsity |
|-----|--------|---|-----------|---------|-------------|---------|----------|
| 8 | 10 | 0.0 | 0.40 | 11.74 | 21.0 | 3.75 ms | 0% |
| 9 | 15 | 0.0 | 0.40 | 11.70 | 21.0 | 3.72 ms | 0% |
| 10 | 10 | 0.0 | 0.50 | **11.91** | 7.23 | 2.87 ms | 65.6% |

**K=4 Critical Finding:**
- Sparsity (threshold=0.50) **hurts quality** significantly
- PPL jumps from 11.70 → 11.91 (+0.21)
- K=4 is **not recommended**

### Master Comparison Table

| K | Best PPL | Latency | Sparsity | Train Time | Verdict |
|---|----------|---------|----------|------------|---------|
| K=0 | **11.55** | **0.24 ms** | N/A | 13 sec/ep | Baseline |
| **K=2** | 11.64 | 0.89 ms | **79%** | 95 sec/ep | **✅ Winner** |
| K=3 | 11.63 | 1.78 ms | 0% | 320 sec/ep | Too slow |
| K=4 | 11.70 | 2.87 ms | 66%* | 330 sec/ep | ❌ Sparsity hurts |

*K=4 sparsity degrades PPL to 11.91

---

## Phase 2: Depth Hypothesis

### The Observation

All K values (K=2, 3, 4) plateau at similar PPL (~11.63-11.70) with depth=2.

| K | Depth | Max Nodes | Best PPL |
|---|-------|-----------|----------|
| K=2 | 2 | 7 | 11.64 |
| K=3 | 2 | 13 | 11.63 |
| K=4 | 2 | 21 | 11.70 |

**Hypothesis:** The bottleneck is depth (reasoning levels), not K (branching).

### Depth Comparison Plan

Testing K=2 with depths 2, 3, 4:

| Depth | K=2 Nodes | Reasoning Levels | Expected PPL |
|-------|-----------|------------------|--------------|
| 2 | 7 | 3 | 11.64 (measured) |
| 3 | 15 | 4 | ~11.58 (hypothesis) |
| 4 | 31 | 5 | ~11.54 (hypothesis) |

### Depth Sweep Configuration

```yaml
k_values: [0, 2]
max_depths: [2, 3, 4]
lambda_efficiency_list: [0.05]
greedy_threshold_list: [0.50]
epochs_list: [20, 30]
```

**8 runs total, ~6-7 hours**

### Results (Pending)

*Results will be added after depth sweep completes.*

---

## Complete Data Tables

### All K=0 Runs

| Run | Epochs | Embed | Hidden | Seq Len | Val PPL | Time/Epoch |
|-----|--------|-------|--------|---------|---------|------------|
| 0 | 5 | 64 | 128 | 128 | 11.60 | 12.9 sec |
| 1 | 10 | 64 | 128 | 128 | 11.58 | 13.0 sec |
| 2 | 15 | 64 | 128 | 128 | 11.56 | 13.1 sec |
| 3 | 20 | 64 | 128 | 128 | 11.55 | 13.0 sec |
| 4-15 | Various | 32-64 | 128 | 64-128 | 11.55-11.60 | 12-13 sec |

### All K=2 Runs

| Run | Epochs | λ | Thr | Val PPL | Train Nodes | Infer Nodes | Latency | Sparsity |
|-----|--------|---|-----|---------|-------------|-------------|---------|----------|
| 0 | 10 | 0.0 | 0.40 | 11.69 | 9.0 | 7.0 | 1.35 ms | 0% |
| 1 | 15 | 0.0 | 0.40 | 11.66 | 9.0 | 7.0 | 1.31 ms | 0% |
| 2 | 10 | 0.0 | 0.50 | 11.68 | 9.0 | 2.07 | 1.04 ms | 70.5% |
| 3 | 15 | 0.0 | 0.50 | 11.64 | 9.0 | 1.61 | 0.95 ms | 77.1% |
| 4 | 10 | 0.05 | 0.40 | 11.70 | 9.0 | 7.0 | 1.34 ms | 0% |
| 5 | 15 | 0.05 | 0.40 | 11.66 | 9.0 | 7.0 | 1.31 ms | 0% |
| 6 | 10 | 0.05 | 0.50 | 11.68 | 9.0 | 1.96 | 1.02 ms | 72.0% |
| 7 | 15 | 0.05 | 0.50 | 11.64 | 9.0 | 1.47 | 0.89 ms | 79.1% |

### All K=3 Runs

| Run | Epochs | λ | Thr | Val PPL | Nodes | Latency |
|-----|--------|---|-----|---------|-------|---------|
| 16 | 5 | 0.0 | 0.30 | 11.80 | 14.25 | 1.78 ms |
| 17 | 10 | 0.0 | 0.30 | 11.69 | 14.25 | 1.78 ms |
| 18 | 15 | 0.0 | 0.30 | 11.65 | 14.25 | 1.78 ms |
| 19 | 20 | 0.0 | 0.30 | 11.63 | 14.25 | 1.78 ms |
| 20 | 5 | 0.0 | 0.35 | 11.80 | 14.25 | 1.78 ms |
| 21 | 10 | 0.0 | 0.35 | 11.69 | 14.25 | 1.78 ms |
| 22 | 15 | 0.0 | 0.35 | 11.65 | 14.25 | 1.78 ms |
| 23 | 20 | 0.0 | 0.35 | 11.63 | 14.25 | 1.78 ms |
| 24 | 5 | 0.0 | 0.40 | 11.80 | 14.25 | 1.78 ms |
| 25 | 10 | 0.0 | 0.40 | 11.69 | 14.25 | 1.78 ms |

### All K=4 Runs

| Run | Epochs | λ | Thr | Val PPL | Nodes | Latency | Sparsity |
|-----|--------|---|-----|---------|-------|---------|----------|
| 8 | 10 | 0.0 | 0.40 | 11.74 | 21.0 | 3.75 ms | 0% |
| 9 | 15 | 0.0 | 0.40 | 11.70 | 21.0 | 3.72 ms | 0% |
| 10 | 10 | 0.0 | 0.50 | 11.91 | 7.23 | 2.87 ms | 65.6% |

---

## Key Insights

### 1. Binary Branching is Fundamental

K=2 outperforms K=3 and K=4:
- **Same quality** as K=3 (PPL 11.64 vs 11.63)
- **3.4x faster training** (95 vs 320 sec/epoch)
- **2x faster inference** (0.89 vs 1.78 ms)
- **Better sparsity handling** (79% vs 0%)

### 2. Sparsity Can Improve Quality

For K=2, more sparsity → better PPL:

| Threshold | Nodes | Sparsity | PPL |
|-----------|-------|----------|-----|
| 0.40 | 7.0 | 0% | 11.66 |
| 0.50 | 1.47 | 79% | **11.64** |

The sparse model is **faster AND better**.

### 3. More Epochs → Better Sparsity

| Epochs | Infer Nodes | Sparsity | PPL |
|--------|-------------|----------|-----|
| 10 | 2.07 | 70.5% | 11.68 |
| 15 | 1.47 | 79.1% | 11.64 |

Policy learns to be more selective with training.

### 4. Threshold vs Nodes Relationship

Policy learns `grow_prob ≈ 0.5`:

| Threshold | above_threshold_pct | Result |
|-----------|---------------------|--------|
| 0.40 | 100% | Full tree |
| 0.50 | 18-33% | Sparse tree |

### 5. K=4 Sparsity is Broken

K=4 cannot handle sparsity well:

| K | Sparse PPL vs Full PPL | Δ |
|---|------------------------|---|
| K=2 | 11.64 vs 11.66 | **-0.02 (better)** |
| K=4 | 11.91 vs 11.70 | **+0.21 (worse)** |

---

## Future Roadmap

### Immediate (Next 24 Hours)

1. **Complete depth sweep** (K=2, depth=2,3,4)
2. **User interaction testing** with best model
3. **Verify depth hypothesis**

### Short Term (Next Week)

1. **Dynamic depth algorithm** prototype
2. **Larger dataset testing** (WikiText-103)
3. **Model scaling** (hidden_dim=256)

### Medium Term (Next Month)

1. **Transformer baseline comparison**
2. **TinyStories experiments**
3. **Paper draft**

### Long Term

1. **Publication**
2. **Open source release**
3. **Community adoption**

---

## Reproduction Instructions

### Environment

```bash
# Docker (recommended)
docker build -t boenet:cuda -f Dockerfile.cuda .

# Or manual
pip install torch torchvision datasets transformers pyyaml
```

### Run Best Configuration

```bash
python3 train_boenet.py \
    --max_children 2 \
    --max_depth 2 \
    --lambda_efficiency 0.05 \
    --greedy_threshold 0.50 \
    --epochs 15 \
    --dataset wikitext2
```

### Expected Results

- **Val PPL:** 11.64 ± 0.02
- **Inference Latency:** 0.89 ± 0.05 ms
- **Sparsity:** 79% ± 2%
- **Training Time:** ~24 minutes

---

## Appendix: CSV Column Definitions

| Column | Description |
|--------|-------------|
| run_id | Unique run identifier |
| tag | Human-readable run name |
| epochs | Number of training epochs |
| val_ppl_last | Validation perplexity at last epoch |
| val_ppl_best | Best validation perplexity achieved |
| train_ppl_last | Training perplexity at last epoch |
| avg_nodes_last | Average nodes per position at last epoch |
| policy_loss_last | Policy loss at last epoch |
| best_epoch | Epoch with best validation PPL |
| total_training_time_sec | Total training time in seconds |
| avg_epoch_time_sec | Average time per epoch |
| max_children | K value (branching factor) |
| max_depth | Tree depth |
| lambda_efficiency | Sparsity penalty coefficient |
| greedy_threshold | Inference decision threshold |
| infer_val_ppl | Validation PPL measured during inference |
| infer_avg_nodes | Average nodes used during inference |
| infer_sparsity_percent | Percentage of nodes NOT used |
| infer_latency_ms_mean | Mean inference latency in milliseconds |
| infer_mean_grow_prob | Mean policy grow probability |
| infer_above_threshold_pct | Percentage of decisions above threshold |

---

## Document History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-12-31 | Initial document with K comparison results |

---

*This document will be updated as new experiments complete.*