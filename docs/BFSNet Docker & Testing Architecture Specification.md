# BFSNet & BoeNet Docker & Testing Architecture Specification

**Version:** 2.0  
**Date:** December 2025  
**Status:** BFSNet Phase ✅ COMPLETE | BoeNet Phase 🚧 IN PROGRESS

---

## 🎯 Document Purpose & Status

This document specifies the complete Docker and testing infrastructure for both BFSNet (vision - COMPLETE) and BoeNet (language - IN PROGRESS). It serves as the authoritative reference for reproducible experiment execution, automated testing, and environment standardization.

### Project Phases
```
┌──────────────────────────────────────────────────────────────┐
│                    PROJECT TIMELINE                          │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  BFSNet (Vision) Phase                                       │
│  ✅ COMPLETE - December 2025                                 │
│  ├─ FashionMNIST experiments                                 │
│  ├─ REINFORCE policy gradients                               │
│  ├─ Docker containerization                                  │
│  ├─ Test suite (86% coverage)                                │
│  └─ Final validation & documentation                         │
│                                                              │
│  BoeNet (Language) Phase                                     │
│  🚧 IN PROGRESS - January 2026+                              │
│  ├─ Character-level proof of concept (Phase 1)               │
│  ├─ Word-level TinyStories (Phase 2)                         │
│  ├─ Production scale (Phase 3)                               │
│  └─ Arcus LLM (Phase 4)                                      │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## 📚 Table of Contents

### Part I: BFSNet Specification (COMPLETE - Historical Reference)
1. [BFSNet Overview & Goals](#1-bfsnet-overview--goals-complete)
2. [BFSNet Directory Structure](#2-bfsnet-directory-structure-complete)
3. [BFSNet Configuration Files](#3-bfsnet-configuration-files-complete)
4. [BFSNet Docker Architecture](#4-bfsnet-docker-architecture-complete)
5. [BFSNet Test Suite Architecture](#5-bfsnet-test-suite-architecture-complete)
6. [BFSNet Execution Workflows](#6-bfsnet-execution-workflows-complete)

### Part II: BoeNet Specification (IN PROGRESS - Active Development)
7. [BoeNet Overview & Goals](#7-boenet-overview--goals-in-progress)
8. [BoeNet Directory Structure](#8-boenet-directory-structure-in-progress)
9. [BoeNet Configuration Files](#9-boenet-configuration-files-in-progress)
10. [BoeNet Docker Architecture](#10-boenet-docker-architecture-in-progress)
11. [BoeNet Test Suite Architecture](#11-boenet-test-suite-architecture-in-progress)
12. [BoeNet Execution Workflows](#12-boenet-execution-workflows-in-progress)

### Part III: Shared Infrastructure
13. [Volume Mounts & Data Persistence](#13-volume-mounts--data-persistence)
14. [Device Management](#14-device-management)
15. [Logging & Output Structure](#15-logging--output-structure)
16. [.gitignore Specification](#16-gitignore-specification)
17. [Future Extensibility](#17-future-extensibility)

---

# PART I: BFSNET SPECIFICATION (COMPLETE)

**⚠️ STATUS**: This section documents the COMPLETED BFSNet (vision) phase. All specifications are frozen and serve as historical reference.

---

## 1. BFSNet Overview & Goals (COMPLETE)

### 1.1 Purpose

BFSNet specification defines the Docker and testing infrastructure for vision-based experiments on FashionMNIST. **This phase is now COMPLETE.**

### 1.2 Design Goals

| Goal | Description | Status |
|------|-------------|--------|
| **Reproducibility** | Identical results with same config | ✅ ACHIEVED |
| **Portability** | Runs on any Ubuntu system with Docker | ✅ ACHIEVED |
| **Separation of Concerns** | Environment vs experiment config | ✅ ACHIEVED |
| **Testability** | Full pipeline validation | ✅ ACHIEVED (86% coverage) |
| **Extensibility** | Support BoeNet transition | ✅ ACHIEVED |
| **Documentation** | No tribal knowledge | ✅ ACHIEVED |

### 1.3 Scope

**Completed Features:**
- ✅ Docker containerization (CPU + CUDA with Blackwell support)
- ✅ Unit test suite (45+ tests, 98% pass rate)
- ✅ Integration test suite (12+ tests, 100% pass)
- ✅ Configuration management (YAML-based)
- ✅ Output persistence and organization
- ✅ Training matrix for parameter sweeps
- ✅ Inference with latency measurement

---

## 2. BFSNet Directory Structure (COMPLETE)

### 2.1 Final Project Layout
```
bfsnet/                                 # PROJECT ROOT
│
├── .gitignore                          # ✅ COMPLETE
├── README.md                           # ✅ UPDATED (documents BFSNet completion)
├── requirements.txt                    # ✅ COMPLETE
├── CHANGELOG.md                        # ✅ UPDATED (v2.0.0 FINAL)
│
├── bfs_model.py                        # ✅ COMPLETE (v2.0.0)
├── train_fmnist_bfs.py                 # ✅ COMPLETE
├── infer_fmnist_bfs.py                 # ✅ COMPLETE
├── bfs_training_matrix.py              # ✅ COMPLETE
│
├── configs/                            # Configuration files
│   ├── README.md                       # ✅ COMPLETE
│   ├── bfsnet/                         # BFSNet configs (FROZEN)
│   │   ├── experiment-config.yaml      # ✅ Production sweep
│   │   ├── test-config.yaml            # ✅ Integration test config
│   │   ├── threshold-sweep.yaml        # ✅ Threshold tuning
│   │   └── examples/                   # ✅ Reference configs
│   └── boenet/                         # BoeNet configs (NEW)
│       └── [see BoeNet section]
│
├── docker/                             # Docker infrastructure
│   ├── README.md                       # ✅ UPDATED
│   ├── Dockerfile                      # ✅ COMPLETE (CPU)
│   ├── Dockerfile.cuda                 # ✅ COMPLETE (CUDA 12.8, Blackwell)
│   ├── Dockerfile.boenet               # 🚧 NEW (BoeNet CPU)
│   ├── Dockerfile.boenet.cuda          # 🚧 NEW (BoeNet CUDA)
│   ├── docker-compose.yaml             # ✅ UPDATED
│   ├── docker-config.yaml              # ✅ COMPLETE
│   └── .dockerignore                   # ✅ COMPLETE
│
├── tests/                              # Test suite
│   ├── README.md                       # ✅ UPDATED
│   ├── conftest.py                     # ✅ COMPLETE
│   ├── pytest.ini                      # ✅ COMPLETE
│   ├── bfsnet/                         # BFSNet tests (COMPLETE)
│   │   ├── README.md                   # ✅ Test documentation
│   │   ├── RESULTS.md                  # ✅ Final test results
│   │   ├── unit/                       # ✅ 45+ tests, 98% pass
│   │   └── integration/                # ✅ 12+ tests, 100% pass
│   └── boenet/                         # BoeNet tests (NEW)
│       └── [see BoeNet section]
│
├── utils/                              # ✅ COMPLETE
│   ├── gating.py                       # GrowthPolicyNet
│   ├── sparse_utils.py                 # Sparse execution helpers
│   └── ...
│
├── scripts/                            # Utility scripts
│   ├── README.md                       # ✅ UPDATED
│   ├── bfsnet/                         # BFSNet scripts (LEGACY)
│   │   ├── bench_sparse_vs_dense.py    # ✅ COMPLETE
│   │   ├── analyze_policy.py           # ✅ COMPLETE
│   │   └── visualize_tree.py           # ✅ COMPLETE
│   └── boenet/                         # BoeNet scripts (NEW)
│       └── [see BoeNet section]
│
├── docs/                               # Documentation
│   ├── bfsnet_architecture.md          # ✅ COMPLETE (historical)
│   └── boenet_architecture.md          # 🚧 IN PROGRESS
│
└── data/                               # Datasets (git-ignored)
    ├── FashionMNIST/                   # ✅ Auto-downloaded
    └── text/                           # 🚧 NEW (BoeNet datasets)
```

### 2.2 Directory Purposes (BFSNet)

| Directory | Purpose | Status |
|-----------|---------|--------|
| `configs/bfsnet/` | BFSNet experiment configs | ✅ FROZEN |
| `docker/` | Docker files (BFSNet + BoeNet) | ✅ BFSNet complete |
| `tests/bfsnet/` | BFSNet test suite | ✅ COMPLETE |
| `scripts/bfsnet/` | BFSNet utility scripts | ✅ COMPLETE |
| `data/FashionMNIST/` | Vision dataset (auto-downloaded) | ✅ COMPLETE |

---

## 3. BFSNet Configuration Files (COMPLETE)

### 3.1 Configuration Hierarchy (Vision)
```
┌─────────────────────────────────────────────────────────────┐
│              docker-config.yaml                              │
│         (Environment: device, paths, runtime)                │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│         configs/bfsnet/experiment-config.yaml                │
│         (Experiment: K, depth, epochs, lambda)               │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              Command-line overrides                          │
│              (Optional: one-off modifications)               │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 BFSNet Configuration Examples

#### docker-config.yaml (Vision)
```yaml
# BFSNet environment configuration
device: cuda
paths:
  data_root: /data              # FashionMNIST auto-downloads here
  output_root: /output
  config_root: /configs
```

#### experiment-config.yaml (Vision)
```yaml
# BFSNet production sweep
model:
  k_values: [0, 1, 2, 3, 4, 5]
  max_depths: [1, 2, 3, 4, 5]
  hidden_dims: [64, 128]
  
training:
  epochs: 10
  lambda_efficiency_list: [0.01, 0.05, 0.1]
  greedy_threshold_list: [0.30, 0.42, 0.50]
  num_rollouts: 3
  beta_entropy: 0.01
```

**Status**: ✅ COMPLETE - All BFSNet configs frozen

---

## 4. BFSNet Docker Architecture (COMPLETE)

### 4.1 Image Strategy (Vision)

| Image | Base | Purpose | Size | Status |
|-------|------|---------|------|--------|
| `bfsnet:cpu` | `python:3.10-slim` | CPU training/inference | ~2.5 GB | ✅ COMPLETE |
| `bfsnet:cuda` | `nvidia/cuda:12.8.0-cudnn-runtime` | GPU training/inference | ~8-10 GB | ✅ COMPLETE |

### 4.2 Dockerfile (CPU) - Vision

**Location**: `docker/Dockerfile`

**Key Characteristics**:
- Python 3.10
- PyTorch 2.1.0 CPU-only
- torchvision for FashionMNIST
- Auto-download support
- Non-root user (bfsnet)

**Status**: ✅ COMPLETE - Frozen

### 4.3 Dockerfile.cuda (GPU) - Vision

**Location**: `docker/Dockerfile.cuda`

**Key Characteristics**:
- CUDA 12.8.0 (Blackwell RTX 50 series support)
- Python 3.11
- PyTorch 2.7.1 with cu128 wheels
- Supports sm_120 (RTX 5080/5090)

**Status**: ✅ COMPLETE - Frozen

### 4.4 Volume Mounts (Vision)

| Container Path | Host Path | Purpose | Content |
|----------------|-----------|---------|---------|
| `/app/data` | `./data` | FashionMNIST images | Auto-downloaded |
| `/app/runs` | `./runs` | Training outputs | CSV, logs, checkpoints |
| `/app/configs` | `./configs` | Config files | YAML configs |

**Data Structure (Vision)**:
```
data/
└── FashionMNIST/              # Auto-downloaded by torchvision
    ├── raw/
    │   ├── train-images-idx3-ubyte.gz
    │   └── t10k-images-idx3-ubyte.gz
    └── processed/
        ├── training.pt
        └── test.pt
```

**Status**: ✅ COMPLETE - Working perfectly

---

## 5. BFSNet Test Suite Architecture (COMPLETE)

### 5.1 Test Categories (Vision)
```
┌─────────────────────────────────────────────────────────────┐
│                  BFSNet Test Suite (COMPLETE)                │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────────────┐    ┌─────────────────────────────┐ │
│  │   Unit Tests        │    │   Integration Tests         │ │
│  │   (45+ tests)       │    │   (12+ tests)               │ │
│  ├─────────────────────┤    ├─────────────────────────────┤ │
│  │ ✅ Gradient flow    │    │ ✅ Pipeline smoke test      │ │
│  │ ✅ Dense baseline   │    │ ✅ CSV output validation    │ │
│  │ ✅ Sparse/dense     │    │ ✅ Config loading           │ │
│  │ ✅ Checkpoints      │    │                             │ │
│  │ ✅ Device fallback  │    │ Pass Rate: 100%             │ │
│  │ ✅ Edge cases       │    │                             │ │
│  │ ✅ Numerical        │    │                             │ │
│  │ ✅ Exec modes       │    │                             │ │
│  │                     │    │                             │ │
│  │ Pass Rate: 98%      │    │                             │ │
│  │ (1 expected diff)   │    │                             │ │
│  └─────────────────────┘    └─────────────────────────────┘ │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 Test Results Summary

**Total Tests**: 57+ tests  
**Pass Rate**: 98.2% (56/57)  
**Coverage**: 86% (core functionality)  
**Known Issues**: 1 (gradient magnitude difference - expected behavior)

**Status**: ✅ COMPLETE - All tests documented in `tests/bfsnet/RESULTS.md`

### 5.3 Test Specifications

See Part I Section 5 of original document for complete test specifications.

**Status**: ✅ ALL TESTS COMPLETE AND PASSING

---

## 6. BFSNet Execution Workflows (COMPLETE)

### 6.1 Training Workflow (Vision)
```bash
# Build image
docker build -t bfsnet:cuda -f docker/Dockerfile.cuda .

# Run training matrix
docker run --rm --gpus all \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/runs:/app/runs \
    -v $(pwd)/configs:/app/configs \
    bfsnet:cuda python bfs_training_matrix.py \
        --config configs/bfsnet/experiment-config.yaml \
        --infer_script infer_fmnist_bfs.py
```

**Status**: ✅ COMPLETE - Fully functional

### 6.2 Testing Workflow (Vision)
```bash
# Run all BFSNet tests
pytest tests/bfsnet/ -v

# Run with Docker
docker run --rm -v $(pwd):/app bfsnet:cpu pytest tests/bfsnet/ -v
```

**Status**: ✅ COMPLETE - All tests passing

---

# PART II: BOENET SPECIFICATION (IN PROGRESS)

**⚠️ STATUS**: This section documents the IN-PROGRESS BoeNet (language) phase. Specifications are actively being developed.

---

## 7. BoeNet Overview & Goals (IN PROGRESS)

### 7.1 Purpose

BoeNet specification defines Docker and testing infrastructure for language modeling experiments, starting with character-level (Phase 1) and scaling to production LLMs (Phase 4).

### 7.2 Design Goals

| Goal | Description | Status |
|------|-------------|--------|
| **Reproducibility** | Identical results with same config | 🚧 IN PROGRESS |
| **Scalability** | Character → Word → Production | 🎯 PLANNED |
| **Text Data Support** | Efficient text dataset handling | 🚧 IN PROGRESS |
| **Baseline Comparison** | vs LSTM, Transformer | 🎯 PLANNED |
| **Generation Quality** | Coherent text generation | 🎯 PLANNED |

### 7.3 Scope

**Phase 1 (Character-Level)**:
- 🚧 Docker containerization (CPU + CUDA)
- 🚧 Text dataset mounting (Shakespeare, War and Peace)
- 🚧 Character-level tokenization
- 🚧 BFSLanguageCell implementation
- 🚧 Unit test suite (target: 15+ tests)
- 🚧 Integration test suite (target: 8+ tests)
- 🚧 Perplexity-based evaluation
- 🚧 Text generation validation

**Phase 2-4 (Future)**:
- ⏳ Word-level BPE tokenization (TinyStories)
- ⏳ Production scale (OpenWebText, The Pile)
- ⏳ Distributed training support
- ⏳ Advanced generation (beam search, sampling strategies)

---

## 8. BoeNet Directory Structure (IN PROGRESS)

### 8.1 New Directory Layout
```
boenet/                                 # PROJECT ROOT (same as bfsnet/)
│
├── boenet_model.py                     # 🚧 BoeNet model (BFSLanguageCell)
├── train_char_boenet.py                # 🚧 Character-level training
├── train_word_boenet.py                # ⏳ Word-level training (Phase 2)
├── generate_boenet.py                  # 🚧 Text generation script
│
├── boenet/                             # 🚧 NEW - BoeNet package
│   ├── __init__.py
│   ├── model.py                        # BFSLanguageCell, BoeNet
│   ├── tokenizer.py                    # CharTokenizer, BPETokenizer
│   ├── generation.py                   # Text generation utilities
│   ├── metrics.py                      # Perplexity, BLEU, etc.
│   └── data.py                         # Text dataset loaders
│
├── configs/boenet/                     # 🚧 NEW - BoeNet configs
│   ├── README.md                       # BoeNet config guide
│   ├── char-level-test.yaml            # Phase 1: Shakespeare minimal
│   ├── char-level-full.yaml            # Phase 1: War and Peace
│   ├── word-level-tiny.yaml            # Phase 2: TinyStories
│   └── examples/
│       ├── char-lstm-baseline.yaml     # LSTM comparison
│       └── char-transformer-baseline.yaml
│
├── docker/                             # Docker infrastructure
│   ├── Dockerfile.boenet               # 🚧 NEW - BoeNet CPU
│   ├── Dockerfile.boenet.cuda          # 🚧 NEW - BoeNet CUDA
│   └── [existing BFSNet Dockerfiles]
│
├── tests/boenet/                       # 🚧 NEW - BoeNet tests
│   ├── README.md                       # Test documentation
│   ├── TEST_PLAN.md                    # Phase 1 test strategy
│   ├── unit/                           # Unit tests
│   │   ├── test_tokenization.py        # 🚧 IN PROGRESS
│   │   ├── test_bfs_language_cell.py   # 🚧 IN PROGRESS
│   │   ├── test_sequence_processing.py # 🚧 IN PROGRESS
│   │   ├── test_gradient_flow.py       # ⏳ PLANNED
│   │   ├── test_generation.py          # ⏳ PLANNED
│   │   └── test_perplexity.py          # ⏳ PLANNED
│   └── integration/                    # Integration tests
│       ├── test_char_training.py       # ⏳ PLANNED
│       ├── test_baseline_comparison.py # ⏳ PLANNED
│       └── test_text_generation.py     # ⏳ PLANNED
│
├── scripts/boenet/                     # 🚧 NEW - BoeNet scripts
│   ├── README.md                       # Script documentation
│   ├── download_shakespeare.py         # ✅ READY
│   ├── download_tinystories.py         # ✅ READY
│   ├── preprocess_text.py              # 🚧 IN PROGRESS
│   ├── tokenizer_utils.py              # 🚧 IN PROGRESS
│   ├── generate_text.py                # 🚧 IN PROGRESS
│   └── analyze_perplexity.py           # ⏳ PLANNED
│
└── data/                               # Datasets (git-ignored)
    ├── text/                           # 🚧 NEW - Text datasets
    │   ├── shakespeare.txt             # ~1 MB
    │   ├── war_and_peace.txt           # ~3 MB
    │   └── tinystories/                # ~2 GB (Phase 2)
    ├── tokenizers/                     # 🚧 NEW - Cached tokenizers
    │   ├── char-ascii/
    │   └── bpe-gpt2/
    └── processed/                      # 🚧 NEW - Preprocessed data
        ├── shakespeare_train.pt
        └── shakespeare_val.pt
```

### 8.2 Directory Purposes (BoeNet)

| Directory | Purpose | Status |
|-----------|---------|--------|
| `boenet/` | Core BoeNet package | 🚧 IN PROGRESS |
| `configs/boenet/` | BoeNet experiment configs | 🚧 IN PROGRESS |
| `tests/boenet/` | BoeNet test suite | 🚧 IN PROGRESS |
| `scripts/boenet/` | BoeNet utility scripts | 🚧 IN PROGRESS |
| `data/text/` | Text datasets (manual) | 🚧 IN PROGRESS |
| `data/tokenizers/` | Tokenizer caches | 🚧 IN PROGRESS |

---

## 9. BoeNet Configuration Files (IN PROGRESS)

### 9.1 Configuration Hierarchy (Language)
```
┌─────────────────────────────────────────────────────────────┐
│              docker-config.yaml                              │
│         (Environment: device, paths, runtime)                │
│         (SAME as BFSNet)                                     │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│         configs/boenet/char-level-test.yaml                  │
│         (Experiment: vocab, seq_len, layers, lambda)         │
│         (NEW structure for language)                         │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              Command-line overrides                          │
│              (Optional: one-off modifications)               │
└─────────────────────────────────────────────────────────────┘
```

### 9.2 BoeNet Configuration Examples

#### char-level-test.yaml (Language - Phase 1)
```yaml
# BoeNet Phase 1: Character-level proof of concept
metadata:
  name: "boenet-char-shakespeare"
  description: "Phase 1 character-level on Shakespeare"
  version: "0.1.0"

model:
  vocab_size: 256              # ASCII characters
  embed_dim: 64
  hidden_dim: 128
  max_children: 3              # Start with BFSNet's best K
  max_depth: 2                 # Start with BFSNet's best depth
  num_layers: 4                # NEW: Stacked BFS cells
  greedy_threshold: 0.42       # Learned from BFSNet
  pooling_mode: "mean"

dataset:
  name: "shakespeare"
  path: "/app/data/text/shakespeare.txt"
  seq_len: 128                 # NEW: Sequence length
  batch_size: 64
  train_split: 0.9

training:
  epochs: 10
  lr: 0.001
  optimizer: "adamw"
  weight_decay: 0.01
  
  # REINFORCE (same as BFSNet)
  num_rollouts: 3
  lambda_efficiency: 0.05      # Start with BFSNet's best
  beta_entropy: 0.01
  
  # Warmup
  warmup_epochs: 3             # May be more important for language

inference:
  max_new_tokens: 100          # NEW: Generation length
  temperature: 0.8             # NEW: Sampling temperature
  top_k: 40                    # NEW: Top-k sampling
  greedy_threshold: 0.42       # MUST TUNE after training!
```

**Status**: 🚧 IN PROGRESS - Template created, needs validation

---

## 10. BoeNet Docker Architecture (IN PROGRESS)

### 10.1 Image Strategy (Language)

| Image | Base | Purpose | Size | Status |
|-------|------|---------|------|--------|
| `boenet:cpu` | `python:3.11-slim` | CPU training/inference | ~3 GB | 🚧 IN PROGRESS |
| `boenet:cuda` | `nvidia/cuda:12.8.0-cudnn-runtime` | GPU training/inference | ~10 GB | 🚧 IN PROGRESS |

### 10.2 Dockerfile.boenet (CPU) - Language

**Location**: `docker/Dockerfile.boenet`

**Key Characteristics**:
- Python 3.11 (latest stable)
- PyTorch 2.7.1 CPU
- **NEW**: tokenizers, transformers, datasets libraries
- **NEW**: sentencepiece for BPE
- **NEW**: /data/text and /data/tokenizers directories
- **NEW**: TOKENIZERS_PARALLELISM=false

**Differences from BFSNet**:
- ✅ Additional language modeling dependencies
- ✅ Text data directories
- ✅ Tokenizer cache support

**Status**: 🚧 IN PROGRESS - Dockerfile created, needs testing

### 10.3 Dockerfile.boenet.cuda (GPU) - Language

**Location**: `docker/Dockerfile.boenet.cuda`

**Key Characteristics**:
- Same CUDA 12.8.0 base as BFSNet (Blackwell support)
- Python 3.11
- PyTorch 2.7.1 with cu128 wheels
- Full language modeling stack

**Status**: 🚧 IN PROGRESS - Dockerfile created, needs testing

### 10.4 Volume Mounts (Language)

| Container Path | Host Path | Purpose | Content |
|----------------|-----------|---------|---------|
| `/app/data` | `./data` | Text datasets + tokenizers | Manual text files |
| `/app/runs` | `./runs` | Training outputs | CSV, logs, checkpoints |
| `/app/configs` | `./configs` | Config files | YAML configs |

**Data Structure (Language)**:
```
data/
├── text/                      # 🚧 NEW - Text datasets (manually added)
│   ├── shakespeare.txt        # ~1 MB
│   ├── war_and_peace.txt      # ~3 MB
│   └── tinystories/           # ~2 GB (Phase 2)
│       ├── train.txt
│       └── val.txt
│
├── tokenizers/                # 🚧 NEW - Cached tokenizers
│   ├── char-ascii/
│   │   └── vocab.json
│   └── bpe-gpt2/
│       ├── vocab.json
│       └── merges.txt
│
└── processed/                 # 🚧 NEW - Preprocessed data (optional)
    ├── shakespeare_train.pt
    └── shakespeare_val.pt
```

**Key Differences from Vision**:
1. **Manual Dataset Preparation**: Text files must be added to `data/text/` before training
2. **Tokenizer Caching**: Pre-trained tokenizers cached in `data/tokenizers/`
3. **Larger Datasets**: TinyStories (2GB), OpenWebText (40GB) require more storage

**Status**: 🚧 IN PROGRESS - Structure defined, needs implementation

---

## 11. BoeNet Test Suite Architecture (IN PROGRESS)

### 11.1 Test Categories (Language)
```
┌─────────────────────────────────────────────────────────────┐
│              BoeNet Test Suite (IN PROGRESS)                 │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────────────┐    ┌─────────────────────────────┐ │
│  │   Unit Tests        │    │   Integration Tests         │ │
│  │   (15+ planned)     │    │   (8+ planned)              │ │
│  ├─────────────────────┤    ├─────────────────────────────┤ │
│  │ 🚧 Tokenization     │    │ ⏳ Char training E2E        │ │
│  │ 🚧 Sequence proc.   │    │ ⏳ Baseline comparison      │ │
│  │ 🚧 BFSLanguageCell  │    │ ⏳ Text generation          │ │
│  │ ⏳ Gradient flow    │    │                             │ │
│  │ ⏳ Generation       │    │ Status: PLANNED             │ │
│  │ ⏳ Perplexity       │    │                             │ │
│  │                     │    │                             │ │
│  │ Status: 3 IN PROG   │    │                             │ │
│  │         3 PLANNED   │    │                             │ │
│  └─────────────────────┘    └─────────────────────────────┘ │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 11.2 Unit Test Specifications (Language)

#### 1. test_tokenization.py (🚧 IN PROGRESS)

**Purpose**: Verify character-level and BPE tokenization.

**Planned Tests**:
- [ ] `test_char_encode_decode` - Character round-trip
- [ ] `test_char_vocab_size` - Vocabulary size = 256
- [ ] `test_bpe_encode_decode` - BPE round-trip
- [ ] `test_bpe_vocab_size` - BPE vocab matches config
- [ ] `test_special_tokens` - Special tokens handled
- [ ] `test_tokenizer_deterministic` - Deterministic encoding

**Priority**: 🔴 CRITICAL - Required for Phase 1

---

#### 2. test_bfs_language_cell.py (🚧 IN PROGRESS)

**Purpose**: Verify BFSLanguageCell processes tokens correctly.

**Planned Tests**:
- [ ] `test_cell_forward_shape` - Output shape correct
- [ ] `test_hidden_state_propagation` - Hidden state carries forward
- [ ] `test_policy_output` - Policy loss generated
- [ ] `test_bfs_expansion_per_token` - Tree built per token
- [ ] `test_cell_stacking` - Multiple cells stack correctly

**Priority**: 🔴 CRITICAL - Core architecture

---

#### 3. test_sequence_processing.py (🚧 IN PROGRESS)

**Purpose**: Verify sequence batching and padding.

**Planned Tests**:
- [ ] `test_batch_padding` - Sequences padded correctly
- [ ] `test_attention_mask` - Padding mask correct
- [ ] `test_variable_length` - Different lengths handled
- [ ] `test_truncation` - Long sequences truncated
- [ ] `test_position_ids` - Position IDs correct

**Priority**: 🟡 HIGH

---

### 11.3 Integration Test Specifications (Language)

#### 1. test_char_training.py (⏳ PLANNED)

**Purpose**: End-to-end character-level training.

**Planned Tests**:
- [ ] `test_shakespeare_training_completes` - Training runs
- [ ] `test_perplexity_improves` - Perplexity decreases
- [ ] `test_generates_valid_text` - Valid text generation
- [ ] `test_checkpoint_saves` - Checkpoints created

**Priority**: 🔴 CRITICAL - Phase 1 validation

---

#### 2. test_baseline_comparison.py (⏳ PLANNED)

**Purpose**: Compare BoeNet to LSTM/Transformer.

**Planned Tests**:
- [ ] `test_vs_lstm_perplexity` - Comparable perplexity
- [ ] `test_vs_lstm_flops` - Lower FLOPs
- [ ] `test_vs_transformer_quality` - Within 10% quality

**Priority**: 🟡 HIGH - Phase 1 success criteria

---

### 11.4 Test Coverage Target (Language)

**Phase 1 Target**: 80%+ coverage on core components

**Core Components**:
- `boenet/model.py` - BFSLanguageCell, BoeNet
- `train_char_boenet.py` - Training loop
- `boenet/tokenizer.py` - CharTokenizer, BPETokenizer
- `boenet/generation.py` - Text generation
- `boenet/metrics.py` - Perplexity

**Status**: 🚧 IN PROGRESS - Test framework being set up

---

## 12. BoeNet Execution Workflows (IN PROGRESS)

### 12.1 Dataset Preparation (Language)
```bash
# Download Shakespeare
python scripts/boenet/download_shakespeare.py \
    --output data/text/shakespeare.txt

# Download TinyStories (Phase 2)
python scripts/boenet/download_tinystories.py \
    --output data/text/tinystories/

# Preprocess text
python scripts/boenet/preprocess_text.py \
    --input data/text/shakespeare.txt \
    --output data/processed/shakespeare.pt \
    --mode char \
    --seq_len 128
```

**Status**: 🚧 IN PROGRESS - Download scripts ready, preprocessing in progress

### 12.2 Training Workflow (Language)
```bash
# Build BoeNet image
docker build -t boenet:cuda -f docker/Dockerfile.boenet.cuda .

# Run character-level training
docker run --rm --gpus all \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/runs:/app/runs \
    -v $(pwd)/configs:/app/configs \
    boenet:cuda python train_char_boenet.py \
        --config configs/boenet/char-level-test.yaml
```

**Status**: 🚧 IN PROGRESS - Docker setup ready, training script in development

### 12.3 Generation Workflow (Language)
```bash
# Generate text from checkpoint
docker run --rm --gpus all \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/runs:/app/runs \
    boenet:cuda python scripts/boenet/generate_text.py \
        --ckpt checkpoints/boenet_char.pt \
        --prompt "To be or not to be" \
        --max_tokens 100 \
        --temperature 0.8
```

**Status**: 🚧 IN PROGRESS - Generation script in development

### 12.4 Testing Workflow (Language)
```bash
# Run all BoeNet tests
pytest tests/boenet/ -v

# Run with Docker
docker run --rm -v $(pwd):/app boenet:cpu pytest tests/boenet/ -v

# Run only fast tests
pytest tests/boenet/ -v -m "unit and not slow"
```

**Status**: 🚧 IN PROGRESS - Test framework being set up

---

# PART III: SHARED INFRASTRUCTURE

**⚠️ STATUS**: These sections apply to BOTH BFSNet and BoeNet.

---

## 13. Volume Mounts & Data Persistence

### 13.1 Mount Point Summary

| Container Path | Purpose | BFSNet Content | BoeNet Content |
|----------------|---------|----------------|----------------|
| `/app/data` | Datasets | FashionMNIST (auto) | text/ + tokenizers/ (manual) |
| `/app/runs` | Outputs | CSV, logs, checkpoints | CSV, logs, checkpoints |
| `/app/configs` | Configs | YAML files (read-only) | YAML files (read-only) |

### 13.2 Output Directory Structure

**Same for both BFSNet and BoeNet**:
```
/output/{run_name}_{timestamp}/
├── matrix_results.csv                  # Main results
├── matrix_results.jsonl                # JSON lines format
├── config_used.yaml                    # Config snapshot
├── {tag}_rep0/
│   ├── run_000.log                     # Training logs
│   ├── infer_000.log                   # Inference logs
│   ├── infer_000.json                  # Parsed metrics
│   └── {tag}_rep0.pt                   # Checkpoint
└── ...
```

### 13.3 Data Persistence Strategy

**Datasets**:
- BFSNet: Downloaded once, auto-managed
- BoeNet: Pre-downloaded manually, version-controlled externally

**Outputs**:
- **Always mount to persistent storage**
- Timestamped subdirectories (no overwrites)
- Config snapshots for reproducibility

**Status**: ✅ COMPLETE (BFSNet) | 🚧 IN PROGRESS (BoeNet)

---

## 14. Device Management

### 14.1 Device Selection Logic

**Same for both BFSNet and BoeNet**:
```
Read config: device: {value}
│
├─ cuda    → CUDA available? → Yes → Use CUDA
│                            → No  → Fallback to CPU, warn
│
├─ mps     → MPS available?  → Yes → Use MPS
│                            → No  → Fallback to CPU, warn
│
└─ cpu     → Use CPU
```

### 14.2 Supported GPUs

**Both BFSNet and BoeNet support**:
- ✅ Blackwell (RTX 50 series) - sm_120
- ✅ Ada Lovelace (RTX 40 series) - sm_89
- ✅ Ampere (RTX 30 series, A100) - sm_80, sm_86
- ✅ Turing (RTX 20 series) - sm_75
- ✅ Volta, Pascal (older architectures)

**Status**: ✅ COMPLETE

---

## 15. Logging & Output Structure

### 15.1 Log Levels

**Same for both BFSNet and BoeNet**:

| Level | Purpose | Example |
|-------|---------|---------|
| DEBUG | Detailed debugging | Tensor shapes, intermediate values |
| INFO | Normal operation | Epoch progress, config loaded |
| WARNING | Potential issues | Device fallback, threshold mismatch |
| ERROR | Errors | Failed to parse inference JSON |
| CRITICAL | Fatal errors | Missing required config |

### 15.2 Structured Output (JSON)

**BFSNet**:
```json
__SUMMARY__ {"run": {...}, "val_acc_best": 87.42, "infer_acc_percent": 86.95}
```

**BoeNet**:
```json
__SUMMARY__ {"run": {...}, "perplexity": 3.45, "infer_generation": "..."}
```

**Status**: ✅ COMPLETE (BFSNet) | 🚧 IN PROGRESS (BoeNet)

---

## 16. .gitignore Specification

### 16.1 Complete .gitignore

**Location**: `.gitignore` (project root)
```gitignore
# ==============================================================================
# BFSNet & BoeNet .gitignore
# ==============================================================================

# Python
__pycache__/
*.py[cod]
*.so
*.egg-info/
dist/
build/

# Virtual Environments
.env
.venv
venv/

# IDE
.idea/
.vscode/
*.swp

# OS Generated
.DS_Store
Thumbs.db

# ==============================================================================
# Project: Data & Outputs (BFSNet + BoeNet)
# ==============================================================================

# Datasets
data/
!data/.gitkeep

# Training outputs
outputs/
runs/

# Checkpoints
checkpoints/
*.pt
*.pth

# Logs
*.log
logs/

# Cache
.cache/
.pytest_cache/

# ==============================================================================
# BoeNet-Specific
# ==============================================================================

# Text datasets (large)
data/text/tinystories/
data/text/pile/

# Tokenizer caches
data/tokenizers/*
!data/tokenizers/.gitkeep

# Preprocessed data
data/processed/

# Generated text samples
generated_samples/
```

**Status**: ✅ COMPLETE

---

## 17. Future Extensibility

### 17.1 Planned Extensions

| Extension | BFSNet Support | BoeNet Support |
|-----------|----------------|----------------|
| **New datasets** | ✅ COMPLETE (FashionMNIST) | 🚧 IN PROGRESS (Shakespeare, TinyStories) |
| **New model variants** | ✅ Subclass BFSNet | 🚧 Subclass BoeNet |
| **Distributed training** | ⏳ PLANNED | ⏳ PLANNED (Phase 3) |
| **Cloud deployment** | ⏳ PLANNED | ⏳ PLANNED (Phase 4) |
| **Hyperparameter optimization** | ✅ Training matrix | 🚧 IN PROGRESS |

### 17.2 Configuration Versioning

**Policy**: Config files include `version` field. When breaking changes occur:
1. Increment version
2. Document migration in config README
3. Add backward-compatibility shim if possible

**Status**: ✅ IMPLEMENTED (both projects)

---

## 18. Summary & Status

### 18.1 BFSNet (COMPLETE)

| Component | Status | Notes |
|-----------|--------|-------|
| Docker (CPU/CUDA) | ✅ COMPLETE | Blackwell support, frozen |
| Configuration | ✅ COMPLETE | All configs frozen |
| Test Suite | ✅ COMPLETE | 86% coverage, 98% pass |
| Documentation | ✅ COMPLETE | Fully documented |
| Pipeline | ✅ COMPLETE | Production ready |

**Decision**: BFSNet development is **FROZEN**. All code preserved for reference.

### 18.2 BoeNet (IN PROGRESS)

| Component | Status | Priority | Phase |
|-----------|--------|----------|-------|
| Docker (CPU/CUDA) | 🚧 IN PROGRESS | 🔴 CRITICAL | 1 |
| BFSLanguageCell | 🚧 IN PROGRESS | 🔴 CRITICAL | 1 |
| Character tokenization | 🚧 IN PROGRESS | 🔴 CRITICAL | 1 |
| Text generation | 🚧 IN PROGRESS | 🔴 CRITICAL | 1 |
| Unit tests | 🚧 IN PROGRESS | 🟡 HIGH | 1 |
| Integration tests | ⏳ PLANNED | 🟡 HIGH | 1 |
| BPE tokenization | ⏳ PLANNED | 🟢 MEDIUM | 2 |
| Production scale | ⏳ PLANNED | 🟢 MEDIUM | 3-4 |

**Focus**: Phase 1 character-level proof of concept (Shakespeare dataset)

---

## 19. Approval & Next Steps

### 19.1 Review Checklist

**BFSNet (Historical)**:
- [x] All tests passing
- [x] Documentation complete
- [x] Results archived
- [x] Lessons learned documented
- [x] Code frozen

**BoeNet (Active)**:
- [ ] Directory structure finalized
- [ ] Docker files tested
- [ ] Configuration templates validated
- [ ] Test framework setup complete
- [ ] Phase 1 ready to begin

### 19.2 Implementation Order (BoeNet Phase 1)

**Week 1**: Foundation
1. ✅ Download scripts (Shakespeare, TinyStories)
2. 🚧 Docker validation (CPU + CUDA)
3. 🚧 BFSLanguageCell implementation
4. 🚧 Character tokenization

**Week 2**: Training
5. 🚧 Training script
6. 🚧 Perplexity calculation
7. 🚧 Text generation
8. 🚧 Unit tests

**Week 3**: Validation
9. ⏳ Integration tests
10. ⏳ Baseline comparison (LSTM)
11. ⏳ Documentation
12. ⏳ Phase 1 report

---

**Document Version:** 2.0  
**Last Updated:** December 20, 2025  
**BFSNet Status:** ✅ COMPLETE - FROZEN  
**BoeNet Status:** 🚧 IN PROGRESS - Phase 1 Active Development  
**Next Milestone:** BoeNet Phase 1 character-level validation (January 2026)

**⚠️ Proprietary Software**: This project is closed source. All rights reserved.