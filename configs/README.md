# BFSNet & BoeNet Configuration Files

**Repository**: BoeNet (formerly BFSNet)  
**Purpose**: Configuration documentation for vision (BFSNet - COMPLETE) and language (BoeNet - IN PROGRESS) tasks  
**Last Updated**: December 20, 2025

---

## 📋 Overview

This directory contains YAML configuration files for both:
1. **BFSNet (Vision)** - ✅ COMPLETE - FashionMNIST experiments (historical reference)
2. **BoeNet (Language)** - 🚧 IN PROGRESS - Character/word-level language modeling

### Directory Structure
```
configs/
├── README.md                      # This file
│
├── bfsnet/                        # BFSNet (Vision) - HISTORICAL
│   ├── README.md                  # BFSNet-specific config guide
│   ├── experiment-config.yaml     # Production sweep (FashionMNIST)
│   ├── test-config.yaml           # Integration test config
│   ├── threshold-sweep.yaml       # Greedy threshold tuning sweep
│   └── examples/
│       ├── dense-baseline.yaml    # K=0 MLP baseline
│       ├── small-sweep.yaml       # Quick validation
│       └── full-factorial.yaml    # Comprehensive sweep
│
└── boenet/                        # BoeNet (Language) - ACTIVE
    ├── README.md                  # BoeNet-specific config guide
    ├── char-level-test.yaml       # Phase 1: Shakespeare minimal
    ├── char-level-full.yaml       # Phase 1: War and Peace full
    ├── word-level-tiny.yaml       # Phase 2: TinyStories
    └── examples/
        ├── char-lstm-baseline.yaml    # LSTM comparison
        └── char-transformer-baseline.yaml  # Transformer comparison
```

---

## 🎯 Quick Start

### BFSNet (Vision - Historical Reference)
```bash
# Run BFSNet production sweep on FashionMNIST
python bfs_training_matrix.py \
    --config configs/bfsnet/experiment-config.yaml \
    --infer_script infer_fmnist_bfs.py

# Run minimal test (CI/CD validation)
python bfs_training_matrix.py \
    --config configs/bfsnet/test-config.yaml \
    --infer_script infer_fmnist_bfs.py

# Run threshold sweep (critical for deployment)
python bfs_training_matrix.py \
    --config configs/bfsnet/threshold-sweep.yaml \
    --infer_script infer_fmnist_bfs.py
```

### BoeNet (Language - Active Development)
```bash
# Phase 1: Character-level proof of concept
python train_char_boenet.py \
    --config configs/boenet/char-level-test.yaml

# Phase 1: Full character-level training
python train_char_boenet.py \
    --config configs/boenet/char-level-full.yaml

# Phase 2: Word-level (TinyStories)
python train_word_boenet.py \
    --config configs/boenet/word-level-tiny.yaml
```

---

## 📊 BFSNet Configuration (Vision - COMPLETE)

### Key Parameters for Vision Tasks

| Parameter | Vision Value | Purpose |
|-----------|--------------|---------|
| `input_dim` | 784 | Flattened 28×28 FashionMNIST images |
| `output_dim` | 10 | Number of classes |
| `max_children` (K) | 0-10 | Tree branching factor (K=0 is dense baseline) |
| `max_depth` | 1-10 | BFS expansion depth |
| `lambda_efficiency` | 0.01-0.1 | Efficiency penalty (higher = sparser) |
| `greedy_threshold` | 0.3-0.5 | **CRITICAL**: Inference decision threshold |
| `num_rollouts` | 1-5 | Stochastic rollouts for REINFORCE |
| `beta_entropy` | 0.001-0.1 | Entropy bonus for exploration |

### BFSNet Best Configuration (FashionMNIST)
```yaml
# configs/bfsnet/best-config.yaml
model:
  input_dim: 784
  output_dim: 10
  hidden_dim: 64
  max_children: 3          # K=3 worked best
  max_depth: 2             # Depth=2 sufficient
  greedy_threshold: 0.42   # CRITICAL: NOT 0.5!
  pooling_mode: mean       # or "learned"

training:
  epochs: 5-10
  batch_size: 64
  lr: 0.001
  optimizer: adam
  
  # REINFORCE parameters
  num_rollouts: 3
  lambda_efficiency: 0.05  # Higher = better (counter-intuitive!)
  beta_entropy: 0.01
  
  # Warmup (optional)
  warmup_epochs: 0         # Not necessary for FashionMNIST
  warmup_pooling: sum

inference:
  num_samples: 1000
  greedy_threshold: 0.42   # Must match trained policy distribution
```

### Critical BFSNet Lessons

1. **Lambda Efficiency = Regularization**
   - λ=0.05 achieved 87.42% (best)
   - λ=0.01 achieved 86.62% (worse!)
   - Higher penalty → better accuracy

2. **Threshold Mismatch**
   - Default 0.5 caused root-only inference
   - Policy learned grow_prob ≈ 0.44-0.45
   - **Must tune threshold to ~0.42**

3. **Task-Specific**
   - FashionMNIST didn't require deep trees
   - Root-only achieved 86-87% accuracy
   - Full tree provided <1% improvement

**👉 See `configs/bfsnet/README.md` for complete BFSNet configuration guide**

---

## 🚀 BoeNet Configuration (Language - IN PROGRESS)

### Key Parameters for Language Tasks

| Parameter | Language Value | Purpose | Difference from Vision |
|-----------|----------------|---------|------------------------|
| `vocab_size` | 256 (char), 50257 (GPT-2) | Token vocabulary | Vision used fixed 784 input |
| `seq_len` | 128-512 | Context window | Vision had single images |
| `embed_dim` | 64-256 | Token embedding size | New for language |
| `hidden_dim` | 128-512 | Hidden state size | Similar to vision |
| `max_children` (K) | 2-5 | Tree branching per token | Similar concept |
| `max_depth` | 2-4 | BFS expansion per token | Similar concept |
| `num_layers` | 2-6 | Stacked BFS cells | New: recurrent depth |
| `lambda_efficiency` | 0.01-0.1 | FLOPs penalty | Same concept, different scale |
| `greedy_threshold` | 0.42 (start) | Decision threshold | Same issue expected |

### Architecture: Vision vs. Language
```yaml
# BFSNet (Vision): Single-shot classification
Input: [B, 784] (flattened image)
  ↓
Root FC: [B, hidden_dim]
  ↓
BFS Expansion: [B, num_nodes, hidden_dim]
  ↓
Pooling: [B, hidden_dim]
  ↓
Output FC: [B, 10]

# BoeNet (Language): Recurrent sequence processing
Input: [B, seq_len] (token IDs)
  ↓
Token Embedding: [B, seq_len, embed_dim]
  ↓
For each timestep t:
    BFSLanguageCell(token[t], hidden[t-1])
      ↓ BFS Expansion (per token)
    hidden[t], policy_loss[t]
  ↓
Output FC: [B, seq_len, vocab_size]
```

### BoeNet Phase 1: Character-Level Configuration

**Target**: Match nanoGPT perplexity with 50% fewer FLOPs
```yaml
# configs/boenet/char-level-test.yaml (Shakespeare)
model:
  vocab_size: 256          # ASCII characters
  embed_dim: 64
  hidden_dim: 128
  max_children: 3          # Start with BFSNet's best K
  max_depth: 2             # Start with BFSNet's best depth
  num_layers: 4            # Stack BFS cells
  greedy_threshold: 0.42   # Learned from BFSNet
  
dataset:
  name: shakespeare
  seq_len: 128             # Short context for testing
  batch_size: 64
  train_split: 0.9

training:
  epochs: 10
  lr: 0.001
  optimizer: adamw
  weight_decay: 0.01
  
  # REINFORCE (same as BFSNet)
  num_rollouts: 3
  lambda_efficiency: 0.05  # Start with BFSNet's best
  beta_entropy: 0.01
  
  # Warmup
  warmup_epochs: 3         # More important for language?
  
inference:
  max_new_tokens: 100
  temperature: 0.8
  greedy_threshold: 0.42   # Will need tuning!
```

### BoeNet Phase 2: Word-Level Configuration

**Target**: Coherent 2-3 sentence generation
```yaml
# configs/boenet/word-level-tiny.yaml (TinyStories)
model:
  vocab_size: 50257        # GPT-2 BPE
  embed_dim: 128
  hidden_dim: 256
  max_children: 3-5        # May need more branching
  max_depth: 2-3
  num_layers: 6
  greedy_threshold: TBD    # Measure from char-level
  
dataset:
  name: tinystories
  seq_len: 256
  batch_size: 32
  
training:
  epochs: 20
  lr: 0.0003               # Lower for larger model
  optimizer: adamw
  weight_decay: 0.01
  grad_clip: 1.0
  
  # REINFORCE
  num_rollouts: 3
  lambda_efficiency: 0.05
  beta_entropy: 0.01
  
  # Warmup
  warmup_epochs: 5
  
inference:
  max_new_tokens: 50
  temperature: 0.8
  top_k: 40
  greedy_threshold: TBD
```

---

## 🔄 Key Differences: Vision vs. Language

### 1. Input Representation

| Aspect | Vision (BFSNet) | Language (BoeNet) |
|--------|-----------------|-------------------|
| Input type | Fixed-size images (784-dim) | Variable-length sequences |
| Preprocessing | Flatten → normalize | Tokenize → embed |
| Batch shape | `[B, 784]` | `[B, seq_len]` |
| Context | Single image | Entire sequence history |

### 2. Model Architecture

| Aspect | Vision (BFSNet) | Language (BoeNet) |
|--------|-----------------|-------------------|
| Processing | Single BFS pass | Recurrent BFS per token |
| Hidden state | None (feedforward) | Carried across timesteps |
| Layers | 1 BFS expansion | Multiple stacked BFS cells |
| Output | Single logits vector | Logits per token position |

### 3. Training Dynamics

| Aspect | Vision (BFSNet) | Language (BoeNet) |
|--------|-----------------|-------------------|
| Loss | Cross-entropy (classification) | Cross-entropy (next-token prediction) |
| Metric | Accuracy (%) | Perplexity |
| Efficiency | Nodes per image | FLOPs per token |
| Reward | `acc - λ × (nodes/max_nodes)` | `-perplexity - λ × (flops/max_flops)` |

### 4. Inference

| Aspect | Vision (BFSNet) | Language (BoeNet) |
|--------|-----------------|-------------------|
| Task | Classify single image | Generate sequence autoregressively |
| Latency | ~1ms per image | ~Nms per token (N=seq_len) |
| Output | Class label | Generated text |
| Threshold | Per-node decision | Per-node, per-token decision |

### 5. Datasets

| Aspect | Vision (BFSNet) | Language (BoeNet) |
|--------|-----------------|-------------------|
| Primary dataset | FashionMNIST (60K images) | Shakespeare (300K chars) → TinyStories (2M stories) |
| Download size | ~30MB | ~1MB (Shakespeare) → 2GB (TinyStories) |
| Preprocessing | Normalize pixels | Tokenization (char or BPE) |
| Augmentation | Not used | Not needed (char-level) |

---

## 📋 Configuration File Structure

### Standard YAML Schema

All configuration files follow this structure:
```yaml
# ==============================================================================
# [Project Name] Configuration
# ==============================================================================
# Description: Brief description of what this config does
# Use case: When to use this config
# ==============================================================================

# Metadata
metadata:
  name: "config-name"
  description: "Detailed description"
  version: "1.0"
  created: "2025-12-20"

# Model architecture
model:
  # Architecture-specific parameters
  input_dim: 784           # (Vision only)
  vocab_size: 256          # (Language only)
  embed_dim: 64            # (Language only)
  hidden_dim: 128
  output_dim: 10           # (Vision only)
  max_children: 3          # K value (BFS branching factor)
  max_depth: 2
  num_layers: 4            # (Language only: stacked cells)
  greedy_threshold: 0.42   # CRITICAL: Must tune!
  pooling_mode: "mean"

# Dataset
dataset:
  name: "fashionmnist"     # or "shakespeare", "tinystories"
  seq_len: 128             # (Language only)
  batch_size: 64
  train_split: 0.9
  val_split: 0.05
  test_split: 0.05

# Training hyperparameters
training:
  epochs: 10
  lr: 0.001
  optimizer: "adamw"
  weight_decay: 0.01
  grad_clip: 1.0
  
  # REINFORCE parameters
  num_rollouts: 3
  lambda_efficiency: 0.05
  beta_entropy: 0.01
  
  # Warmup (optional)
  warmup_epochs: 3
  warmup_exec: "soft_full"
  warmup_pooling: "sum"

# Inference settings
inference:
  num_samples: 200         # (Vision: test samples)
  max_new_tokens: 100      # (Language: generation)
  temperature: 0.8         # (Language: sampling temperature)
  top_k: 40                # (Language: top-k sampling)
  greedy_threshold: 0.42   # Must match training distribution!

# Reproducibility
reproducibility:
  seed: 42
  repeats: 1               # Number of runs per config
```

---

## 🔧 Creating Custom Configurations

### For BFSNet (Vision)

1. **Start with a template**:
```bash
   cp configs/bfsnet/experiment-config.yaml configs/bfsnet/my-experiment.yaml
```

2. **Edit key parameters**:
   - `max_children` (K): 0 (dense), 1-10 (BFS)
   - `max_depth`: 1-10
   - `lambda_efficiency`: 0.01-0.1
   - `greedy_threshold`: **CRITICAL** - must tune after training!

3. **Run sweep**:
```bash
   python bfs_training_matrix.py \
       --config configs/bfsnet/my-experiment.yaml \
       --infer_script infer_fmnist_bfs.py
```

4. **Analyze results**:
```python
   import pandas as pd
   df = pd.read_csv('runs/YYYYMMDD_HHMMSS/matrix_results.csv')
   print(df.nlargest(10, 'val_acc_best'))
```

### For BoeNet (Language)

1. **Start with char-level template**:
```bash
   cp configs/boenet/char-level-test.yaml configs/boenet/my-char-experiment.yaml
```

2. **Edit key parameters**:
   - `vocab_size`: 256 (ASCII) or custom
   - `seq_len`: 128-512 (context window)
   - `num_layers`: 2-6 (stacked BFS cells)
   - `lambda_efficiency`: Start with 0.05
   - `greedy_threshold`: Start with 0.42, **MUST TUNE**

3. **Run training**:
```bash
   python train_char_boenet.py \
       --config configs/boenet/my-char-experiment.yaml
```

4. **Measure policy distribution**:
```bash
   python infer_char_boenet.py \
       --ckpt checkpoints/my_model.pt \
       --debug_policy
```

5. **Tune threshold**:
```python
   # From debug output:
   mean_grow_prob = 0.445
   recommended_threshold = mean_grow_prob - 0.03  # ~0.42
   
   # Update config and retrain or adjust inference threshold
```

---

## ⚙️ CLI Overrides

All config values can be overridden via command line:

### BFSNet Examples
```bash
# Override K values
python bfs_training_matrix.py \
    --config configs/bfsnet/experiment-config.yaml \
    --k_values 0,2,4,6 \
    --infer_script infer_fmnist_bfs.py

# Override epochs and lambda
python bfs_training_matrix.py \
    --config configs/bfsnet/experiment-config.yaml \
    --epochs 20 \
    --lambda_efficiency 0.1 \
    --infer_script infer_fmnist_bfs.py

# Force CPU
python bfs_training_matrix.py \
    --config configs/bfsnet/experiment-config.yaml \
    --cpu_only \
    --infer_script infer_fmnist_bfs.py
```

### BoeNet Examples
```bash
# Override vocab size and sequence length
python train_char_boenet.py \
    --config configs/boenet/char-level-test.yaml \
    --vocab_size 128 \
    --seq_len 256

# Override lambda and threshold
python train_char_boenet.py \
    --config configs/boenet/char-level-test.yaml \
    --lambda_efficiency 0.08 \
    --greedy_threshold 0.40

# Override dataset
python train_char_boenet.py \
    --config configs/boenet/char-level-test.yaml \
    --dataset war_and_peace
```

---

## 🧪 Testing Configurations

### Validate Config Before Running
```python
# Python script: validate_config.py
import yaml
import sys

def validate_config(config_path):
    """Validate config file structure and required fields."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Check required sections
    required = ['metadata', 'model', 'dataset', 'training']
    for section in required:
        if section not in config:
            raise ValueError(f"Missing required section: {section}")
    
    # Check model params
    if 'max_children' not in config['model']:
        raise ValueError("model.max_children is required")
    
    # Check greedy_threshold
    if 'greedy_threshold' not in config['model']:
        print("WARNING: greedy_threshold not set, will use default (may be wrong!)")
    
    print(f"✅ Config valid: {config_path}")

if __name__ == '__main__':
    validate_config(sys.argv[1])
```
```bash
# Validate before running
python scripts/validate_config.py configs/boenet/char-level-test.yaml
```

### Quick Test Run
```bash
# BFSNet: 1 epoch, 2 configs
python bfs_training_matrix.py \
    --config configs/bfsnet/test-config.yaml \
    --epochs 1 \
    --infer_script infer_fmnist_bfs.py

# BoeNet: 1 epoch, minimal data
python train_char_boenet.py \
    --config configs/boenet/char-level-test.yaml \
    --epochs 1 \
    --max_samples 1000
```

---

## 📊 Configuration Best Practices

### 1. Version Control

- ✅ **DO**: Commit all `.yaml` files to git
- ✅ **DO**: Use descriptive names (`char-level-test.yaml`, not `config1.yaml`)
- ✅ **DO**: Document what each config does in metadata section
- ❌ **DON'T**: Commit outputs or checkpoints (use `.gitignore`)

### 2. Reproducibility

- ✅ **DO**: Set explicit seeds in config
- ✅ **DO**: Save config copy in output directory
- ✅ **DO**: Record git commit hash in results
- ✅ **DO**: Document any manual overrides in logs

### 3. Experimentation

- ✅ **DO**: Create new config files for experiments (don't overwrite)
- ✅ **DO**: Use `examples/` subdirectory for reference configs
- ✅ **DO**: Name configs by purpose: `threshold-sweep.yaml`, `lambda-comparison.yaml`
- ❌ **DON'T**: Rely on CLI overrides for complex experiments (use explicit configs)

### 4. Documentation

- ✅ **DO**: Add comments explaining non-obvious choices
- ✅ **DO**: Reference related configs in comments
- ✅ **DO**: Document expected runtime and output size
- ✅ **DO**: Note hardware requirements (GPU memory, CPU cores)

---

## 🔍 Debugging Configuration Issues

### Common Issues

1. **Config file not found**
```
   FileNotFoundError: configs/my-config.yaml
```
   - Check path is correct relative to project root
   - Ensure file has `.yaml` extension (not `.yml`)

2. **Invalid YAML syntax**
```
   yaml.scanner.ScannerError: mapping values are not allowed here
```
   - Check indentation (use spaces, not tabs)
   - Ensure colons have space after them (`key: value` not `key:value`)
   - Quote strings with special characters

3. **Missing required fields**
```
   KeyError: 'max_children'
```
   - Validate config with `validate_config.py` (see above)
   - Compare to working example config

4. **Threshold mismatch (BFSNet/BoeNet)**
```
   Inference creates 0 children despite training with many nodes
```
   - **Root cause**: `greedy_threshold` too high
   - **Fix**: Measure policy with `--debug_policy`, set threshold ~0.42
   - **Prevention**: Always run `--debug_policy` after training

---

## 📖 Example Configurations

### BFSNet: Minimal Test (CI/CD)
```yaml
# configs/bfsnet/test-config.yaml
metadata:
  name: "bfsnet-ci-test"
  description: "Minimal config for CI/CD and quick validation"
  version: "1.0"

model:
  input_dim: 784
  output_dim: 10
  hidden_dim: 32          # Small for speed
  max_children: 2         # Just one BFS config
  max_depth: 1
  greedy_threshold: 0.42
  pooling_mode: "sum"

dataset:
  name: "fashionmnist"
  batch_size: 64

training:
  epochs: 1               # Single epoch
  lr: 0.001
  num_rollouts: 1         # Minimal
  lambda_efficiency: 0.05
  beta_entropy: 0.01

inference:
  num_samples: 100        # Small sample
```

### BoeNet: Character-Level (Phase 1)
```yaml
# configs/boenet/char-level-test.yaml
metadata:
  name: "boenet-char-shakespeare"
  description: "Phase 1 character-level proof of concept"
  version: "0.1.0"

model:
  vocab_size: 256         # ASCII
  embed_dim: 64
  hidden_dim: 128
  max_children: 3
  max_depth: 2
  num_layers: 4
  greedy_threshold: 0.42  # Start with BFSNet's value
  pooling_mode: "mean"

dataset:
  name: "shakespeare"
  seq_len: 128
  batch_size: 64
  train_split: 0.9

training:
  epochs: 10
  lr: 0.001
  optimizer: "adamw"
  weight_decay: 0.01
  
  num_rollouts: 3
  lambda_efficiency: 0.05
  beta_entropy: 0.01
  
  warmup_epochs: 3

inference:
  max_new_tokens: 100
  temperature: 0.8
  greedy_threshold: 0.42  # MUST TUNE AFTER TRAINING!
```

---

## 🗺️ Configuration Roadmap

### Phase 1: Character-Level (Current)
- ✅ `char-level-test.yaml` - Shakespeare minimal (CREATED)
- ✅ `char-level-full.yaml` - War and Peace full (CREATED)
- ✅ `char-lstm-baseline.yaml` - LSTM comparison (CREATED)

### Phase 2: Word-Level (Next)
- 🚧 `word-level-tiny.yaml` - TinyStories
- 🚧 `word-level-wiki.yaml` - WikiText-103
- 🚧 `word-transformer-baseline.yaml` - GPT-2 small comparison

### Phase 3: Production Scale (Future)
- ⏳ `production-125m.yaml` - OpenWebText 125M params
- ⏳ `production-1b.yaml` - The Pile 1B params
- ⏳ `production-7b.yaml` - Arcus LLM v1.0

---

## 📞 Support

### For BFSNet (Vision)
- 📚 See `configs/bfsnet/README.md` for detailed guide
- 📖 See `docs/bfsnet_architecture.md` for architecture details
- 🐛 Known issues documented in `BFSNET_FINAL_REPORT.md`

### For BoeNet (Language)
- 📚 See `configs/boenet/README.md` for detailed guide (IN PROGRESS)
- 📖 See `docs/boenet_architecture.md` for architecture details (IN PROGRESS)
- 🎯 See `BOENET_VISION.md` for project goals and roadmap

### General
- 📧 Contact: [your-email@example.com]
- 📝 Issues: Contact project owner (closed source)

---

**Last Updated**: December 20, 2025  
**Status**: BFSNet configs ✅ COMPLETE | BoeNet configs 🚧 IN PROGRESS  
**Next**: Create BoeNet-specific configs for Phase 1 character-level training

**⚠️ Proprietary Software**: This project is closed source. All rights reserved.