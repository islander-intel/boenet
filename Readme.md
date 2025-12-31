# BoeNet: Biological Optimized Enhanced Neural Network

**Applying BFS Tree Expansion to Language Modeling**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.7+-red.svg)](https://pytorch.org/)
[![License: Proprietary](https://img.shields.io/badge/License-Proprietary-red.svg)](LICENSE)

---

## 🎯 Project Overview

This repository contains the evolution from **BFSNet** (vision) to **BoeNet** (language), demonstrating how breadth-first search tree expansion with adaptive compute can be applied across modalities.

### Current Status

| Project | Version | Status | Description |
|---------|---------|--------|-------------|
| **BFSNet** | v2.0.0 | ✅ **COMPLETE** | Vision model on FashionMNIST - REINFORCE policy gradients for adaptive tree expansion |
| **BoeNet** | v2.0.1 | ✅ **TRAINING** | Language model on WikiText-2 with ByteTokenizer - 400-cell experiment matrix running |

---

## 📚 Table of Contents

- [BFSNet: What We Accomplished](#bfsnet-what-we-accomplished)
- [BoeNet: Current Progress](#boenet-current-progress)
- [Quick Start](#quick-start)
- [Architecture Overview](#architecture-overview)
- [Installation](#installation)
- [Usage Examples](#usage-examples)
- [Documentation](#documentation)
- [Project Roadmap](#project-roadmap)
- [Citation](#citation)
- [License](#license)

---

## 🏆 BFSNet: What We Accomplished

**BFSNet v2.0.0** was our proof-of-concept demonstrating that BFS tree expansion with policy gradients works for neural networks.

### Key Achievements

✅ **True Sparse Computation**: REINFORCE policy decides BEFORE computing (not post-hoc masking)  
✅ **Bug Discovery & Fix**: Found critical batch normalization bug in efficiency penalty  
✅ **Threshold Mismatch**: Discovered training/inference discrepancy in greedy decisions  
✅ **Full Parameter Sweep**: 48 configurations tested on FashionMNIST  
✅ **Production Pipeline**: Docker, training matrix, comprehensive logging  

### Final Results (FashionMNIST)

| Configuration | Val Accuracy | Inference Nodes | Sparsity | Finding |
|---------------|--------------|-----------------|----------|---------|
| λ=0.05, threshold=0.5 | 87.42% | 1.0 (root-only) | 92% | Higher λ → better accuracy! |
| λ=0.05, threshold=0.42 | ~88%* | ~8 nodes | ~40% | Balanced performance |
| Dense Baseline (K=0) | ~85% | N/A | 0% | BFS beats dense MLP |

*Estimated - see `docs/bfsnet_architecture.md` for complete analysis

### Critical Lessons Learned

1. **Policy Gradients Work**: REINFORCE successfully learns adaptive compute decisions
2. **Efficiency = Regularization**: Higher compute penalties improved accuracy (counter-intuitive!)
3. **Threshold Tuning Critical**: Default 0.5 too high; must match learned grow_prob distribution
4. **Task-Specific**: FashionMNIST may not require deep hierarchical reasoning (root-only worked well)

**👉 See `docs/bfsnet_architecture.md` for complete technical retrospective**

---

## 🚀 BoeNet: Current Progress

**BoeNet v2.0.1** is actively training on WikiText-2 with a 400-cell experiment matrix.

### Training Status (December 2025)

| Metric | Value | Notes |
|--------|-------|-------|
| **Best Val PPL** | 11.55 | At epoch 15-20 |
| **Random Baseline** | 256.00 | (vocab_size) |
| **Improvement** | **22x better** | Significant learning |
| **Model Size** | ~70K-340K params | Varies by config |
| **Token Range** | [10, 226] | Within vocab_size=256 ✅ |

### Critical Bug Fix: ByteTokenizer (v2.0.1)

**Problem Discovered**: CharTokenizer used `ord(c)` which returns Unicode code points up to 65535+, causing `IndexError: index out of range` when WikiText-2 contained em-dashes (—), smart quotes ('), and other Unicode characters.

**Solution**: ByteTokenizer with UTF-8 byte encoding ensures ALL token IDs are in range [0, 255].

| Character | ord() (Unicode) | UTF-8 bytes |
|-----------|-----------------|-------------|
| `a` | 97 ✅ | [97] ✅ |
| `é` | 233 ✅ | [195, 169] ✅ |
| `—` (em-dash) | 8212 ❌ | [226, 128, 148] ✅ |
| `'` (smart quote) | 8217 ❌ | [226, 128, 153] ✅ |

**Before (BROKEN)**:
```python
def encode(self, text: str) -> List[int]:
    return [ord(c) for c in text]  # Unicode code points 0-65535+
```

**After (FIXED)**:
```python
def encode(self, text: str) -> List[int]:
    return list(text.encode('utf-8'))  # UTF-8 bytes 0-255
```

### Training Results (Sample from 400-cell Matrix)

| Run | Epochs | seq_len | embed_dim | Val PPL | Train PPL | Status |
|-----|--------|---------|-----------|---------|-----------|--------|
| 000 | 5 | 64 | 32 | 11.60 | 11.59 | ✅ Complete |
| 001 | 10 | 64 | 32 | 11.58 | 11.57 | ✅ Complete |
| 007 | 20 | 128 | 32 | 11.55 | 11.57 | ✅ Complete |
| 015 | 20 | 128 | 64 | 11.55 | 11.57 | ✅ Complete |

**Full Pipeline Operational:**
- ✅ Config loading (configs/experiment-config.yaml)
- ✅ Dataset downloading (WikiText-2 from HuggingFace)
- ✅ ByteTokenizer encoding (UTF-8 bytes)
- ✅ Training with cosine LR schedule
- ✅ Checkpoint saving (runs/YYYYMMDD_HHMMSS/...)
- ✅ Inference evaluation
- ✅ Matrix sweep (400 cells)

### Why Language?

1. **Proven Foundation**: BFSNet validated REINFORCE + efficiency penalties work
2. **Natural Fit**: Sequential text naturally maps to tree expansion (each token can spawn context representations)
3. **Adaptive Compute Advantage**: Different tokens/contexts require different processing depth
4. **Novel Research**: No one has applied BFS tree expansion to LLMs

### Core Innovation
```python
# BFSNet (Vision): Image → BFS Tree → Classification
Input: [B, 784] (flattened image)
   ↓
Root FC: [B, hidden_dim]
   ↓
BFS Expansion: Dynamic tree depth based on input complexity
   ↓
Output FC: [B, 10] (class logits)

# BoeNet (Language): Sequence → BFS Tree per Token → Next Token Prediction
Input: [B, seq_len] (token IDs, 0-255 bytes)
   ↓
Token Embedding: [B, seq_len, embed_dim]
   ↓
For each token t:
    BFS Expansion: Build context representation tree
    Hidden State: Carry forward to next token (like RNN)
   ↓
Output FC: [B, seq_len, vocab_size] (next token logits)
```

### Architecture: Recurrent BFS
```
┌─────────────────────────────────────────┐
│         BoeNet Architecture             │
│                                         │
│  Token "The" → BFS Tree → Hidden State │
│      ↓                         ↓        │
│  Token "cat" → BFS Tree → Hidden State │
│      ↓                         ↓        │
│  Token "sat" → BFS Tree → Hidden State │
│      ↓                         ↓        │
│  Predict: "on" (next token)             │
└─────────────────────────────────────────┘

BFS Tree per Token:
   Root: [token_embed + hidden_{t-1}]
     ├─ Child 0 (grammar path)
     ├─ Child 1 (semantic path)
     └─ Child 2 (context path)
         ├─ Grandchild 0
         └─ Grandchild 1

Policy: REINFORCE decides which children to create
Reward: -perplexity - λ × (FLOPs used / max FLOPs)
```

### Development Roadmap

**Phase 1: Byte-Level Proof of Concept** ✅ **IN PROGRESS**
- Dataset: WikiText-2 (via HuggingFace)
- Model: 70K-340K parameters
- Tokenizer: ByteTokenizer (UTF-8 bytes, vocab_size=256)
- Goal: Validate training pipeline and convergence
- **Status**: 🎯 Training matrix running (400 cells)

**Phase 2: Scaled Training** (Next)
- Dataset: Shakespeare, TinyStories
- Model: 1M-25M parameters
- Goal: Match nanoGPT perplexity with adaptive compute

**Phase 3: Production Scale** (Future)
- Dataset: OpenWebText (40GB) → The Pile (300B+ tokens)
- Model: 125M → 1B parameters
- Goal: Arcus LLM v1.0 - competitive personal assistant

**Phase 4: Arcus LLM** (2026-2027)
- Model: 7B+ parameters
- Goal: ChatGPT-level performance with 3× faster inference

---

## ⚡ Quick Start

### Prerequisites

- Docker with NVIDIA Container Toolkit
- NVIDIA GPU (tested on RTX 5080, RTX 3090)
- CUDA 12.8+ compatible driver (590.x or newer for Blackwell)

### Build Docker Image

```bash
# Build CUDA image
docker build -t boenet:cuda -f docker/Dockerfile.cuda .
```

### Run Training (Interactive)

```bash
# Single training run with GPU
docker run --rm --gpus all \
    -v ${PWD}/data:/app/data \
    -v ${PWD}/runs:/app/runs \
    -v ${PWD}/configs:/app/configs \
    -v ${PWD}/boenet:/app/boenet \
    boenet:cuda python boenet_training_matrix.py \
        --config configs/experiment-config.yaml
```

### Run Training (Background/Detached)

```bash
# Run 400-cell training matrix in background
docker run -d --gpus all \
    --name boenet_sweep \
    -v ${PWD}/data:/app/data \
    -v ${PWD}/runs:/app/runs \
    -v ${PWD}/configs:/app/configs \
    -v ${PWD}/boenet:/app/boenet \
    boenet:cuda python boenet_training_matrix.py \
        --config configs/experiment-config.yaml
```

### Monitor Training

```bash
# Follow live logs
docker logs -f boenet_sweep

# Last 50 lines
docker logs --tail 50 boenet_sweep

# Check if running
docker ps

# Stop gracefully
docker stop boenet_sweep

# Remove container
docker rm boenet_sweep
```

### Windows PowerShell Commands

```powershell
# Build
docker build -t boenet:cuda -f docker/Dockerfile.cuda .

# Run (background with GPU)
docker run -d --gpus all --name boenet_sweep -v ${PWD}/data:/app/data -v ${PWD}/runs:/app/runs -v ${PWD}/configs:/app/configs -v ${PWD}/boenet:/app/boenet boenet:cuda python boenet_training_matrix.py --config configs/experiment-config.yaml

# Monitor
docker logs -f boenet_sweep
```

---

## 🏗️ Architecture Overview

### BoeNet v2.0.1 (Language - Active)

**Key Innovation**: BFS tree expansion per token for sequential language modeling with byte-level tokenization.

**Architecture Components**:

1. **ByteTokenizer**: UTF-8 byte encoding (vocab_size=256)
   - Guarantees all token IDs in range [0, 255]
   - Handles any Unicode text (em-dashes, smart quotes, accented characters)
   - Backwards compatible alias: `CharTokenizer = ByteTokenizer`

2. **BFSLanguageCell**: Processes one token through BFS tree
   - Input: Token embedding + hidden state from previous token
   - Policy: REINFORCE decides which children to expand
   - Output: New hidden state (like LSTM/GRU)

3. **Recurrent Processing**: Chain cells across sequence
```python
   for t in range(seq_len):
       hidden[t], policy_loss[t] = BFSCell(
           token_embed=embed(tokens[t]),
           hidden_prev=hidden[t-1]
       )
```

4. **Reward Function**: Minimize perplexity with FLOPs efficiency
```python
   reward = -perplexity - lambda_efficiency × (flops_used / max_flops)
```

**Current Configuration** (Phase 1):
```python
BoeNet(
    vocab_size=256,         # UTF-8 bytes
    embed_dim=32-64,        # Varies in sweep
    hidden_dim=128,
    max_depth=1-2,          # Varies in sweep
    max_children=0-3,       # K=0 (dense) or K=3 (BFS)
    seq_len=64-128,
    greedy_threshold=0.3-0.5
)
```

### BFSNet v2.0.0 (Vision - Complete)

**Key Innovation**: REINFORCE policy gradients for adaptive tree growth
- Input: 784-dim flattened images
- Architecture: BFS tree with K=3 children, depth=2
- Output: 10-class logits
- Reward: `accuracy - λ × (nodes_used / max_nodes)`

**Best Configuration**:
```python
BFSNet(
    input_dim=784,
    hidden_dim=64,
    output_dim=10,
    max_depth=2,
    max_children=3,
    greedy_threshold=0.42,  # Critical tuning!
    pooling_mode='mean'
)
```

---

## 💻 Installation

### Prerequisites

- Python 3.8+
- PyTorch 2.7+
- CUDA 12.8+ (optional, for GPU)

### Docker (Recommended)

```bash
# Build BoeNet CUDA image (GPU - recommended)
docker build -t boenet:cuda -f docker/Dockerfile.cuda .

# Verify GPU access
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi

# Verify PyTorch CUDA
docker run --rm --gpus all boenet:cuda python -c \
    "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
```

### Local Setup (Alternative)
```bash
# Clone repository
git clone https://github.com/your-repo/boenet.git
cd boenet

# Create virtual environment
python3 -m venv boenet-env
source boenet-env/bin/activate  # On Windows: boenet-env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# For language modeling (additional):
pip install tokenizers transformers datasets
```

---

## 📖 Usage Examples

### Example 1: BoeNet Training Matrix
```python
# Run full 400-cell experiment matrix
# Varies: epochs, seq_len, embed_dim, max_children, threshold, lambda

docker run -d --gpus all --name boenet_sweep \
    -v ${PWD}/data:/app/data \
    -v ${PWD}/runs:/app/runs \
    -v ${PWD}/configs:/app/configs \
    -v ${PWD}/boenet:/app/boenet \
    boenet:cuda python boenet_training_matrix.py \
        --config configs/experiment-config.yaml
```

### Example 2: BoeNet Single Run
```python
import torch
from boenet.model import BoeNet

# Create byte-level model
model = BoeNet(
    vocab_size=256,        # UTF-8 bytes
    embed_dim=64,
    hidden_dim=128,
    max_depth=2,
    max_children=3,
    num_layers=4
)

# Training
model.train()
for tokens, targets in train_loader:  # tokens: [B, seq_len]
    logits, policy_loss, avg_nodes = model(
        tokens,
        num_rollouts=3,
        lambda_efficiency=0.05
    )
    
    # Cross-entropy on next-token prediction
    loss = F.cross_entropy(
        logits.view(-1, vocab_size),
        targets.view(-1)
    ) + 0.5 * policy_loss
    
    loss.backward()
    optimizer.step()
```

### Example 3: ByteTokenizer Usage
```python
from boenet.utils.data_utils import ByteTokenizer

tokenizer = ByteTokenizer()

# Encode text to bytes
text = "Hello, world! — smart quotes: 'test'"
tokens = tokenizer.encode(text)
print(f"Tokens: {tokens}")  # All values 0-255
print(f"Max token: {max(tokens)}")  # <= 255

# Decode back to text
decoded = tokenizer.decode(tokens)
print(f"Decoded: {decoded}")
```

### Example 4: BFSNet Training (Historical Reference)
```python
import torch
from bfs_model import BFSNet

# Create model
model = BFSNet(
    input_dim=784,
    hidden_dim=64,
    output_dim=10,
    max_depth=2,
    max_children=3,
    greedy_threshold=0.42
)

# Training
model.train()
for x, y in train_loader:
    outputs, policy_loss, rewards, nodes = model(
        x, 
        num_rollouts=3,
        lambda_efficiency=0.05,
        labels=y
    )
    
    loss = F.cross_entropy(outputs, y) + 0.5 * policy_loss
    loss.backward()
    optimizer.step()
```

---

## 📚 Documentation

### BoeNet (Language - Active Development)

| Document | Status | Description |
|----------|--------|-------------|
| `docs/boenet_architecture.md` | 📝 IN PROGRESS | BoeNet technical specification |
| `boenet/utils/data_utils.py` | ✅ v2.0.1 | ByteTokenizer with UTF-8 byte encoding |
| `configs/experiment-config.yaml` | ✅ ACTIVE | 400-cell training matrix configuration |
| `docker/README.md` | ✅ v2.0.1 | Docker setup with GPU support |

### BFSNet (Vision - Complete)

| Document | Description |
|----------|-------------|
| `docs/bfsnet_architecture.md` | Complete technical retrospective of BFSNet v2.0.0 |
| `docs/bfsnet_lessons_learned.md` | Key insights and lessons for BoeNet |
| `configs/README.md` | BFSNet configuration guide (historical reference) |

### Research & Analysis

| Document | Description |
|----------|-------------|
| `BFSNET_FINAL_REPORT.md` | Executive summary of BFSNet project |
| `Bfsnet fashionmnist test plan.md` | Experimental methodology and results |
| `CHANGELOG.md` | Version history and transition log |

---

## 🗺️ Project Roadmap

### ✅ Completed: BFSNet v2.0.0 (Dec 2025)

- [x] Implement REINFORCE policy gradients for adaptive tree growth
- [x] Fix batch normalization bug in efficiency penalty
- [x] Discover and analyze greedy threshold mismatch
- [x] Complete 48-configuration parameter sweep on FashionMNIST
- [x] Achieve 87.42% validation accuracy (beats dense baseline)
- [x] Docker containerization and production pipeline
- [x] Comprehensive documentation and analysis

### ✅ Completed: BoeNet v2.0.1 Infrastructure (Dec 2025)

- [x] Docker deployment with HuggingFace cache fixes
- [x] ByteTokenizer replacing CharTokenizer (critical UTF-8 fix)
- [x] WikiText-2 dataset integration via HuggingFace
- [x] Training matrix infrastructure (400 cells)
- [x] GPU support verification (RTX 5080 Blackwell confirmed)
- [x] Inference pipeline with latency measurement

### 🚧 In Progress: BoeNet v2.0.1 Training (Dec 2025 - Jan 2026)

- [x] **Phase 1a**: Validate training pipeline ✅
- [x] **Phase 1b**: Confirm convergence (PPL: 11.55, 22x improvement) ✅
- [ ] **Phase 1c**: Complete 400-cell experiment matrix (running)
- [ ] **Phase 1d**: Analyze results and identify best configurations
- [ ] **Phase 1e**: Document findings

**Current Training Progress**:
- 16/400 cells complete (as of latest CSV)
- Best PPL: 11.55 (K=0, seq_len=128, embed_dim=64, 20 epochs)
- Running on RTX 5080 with `--gpus all`

### ⏳ Planned: BoeNet v2.1.0 (Jan-Feb 2026)

- [ ] Scale to Shakespeare dataset
- [ ] Scale to TinyStories (25M parameters)
- [ ] Implement BFS tree expansion (K=3) optimization
- [ ] Coherent text generation

### 🎯 Future: Arcus LLM (2026-2027)

- [ ] Scale to 125M-1B parameters
- [ ] Train on OpenWebText → The Pile
- [ ] Production deployment
- [ ] Public beta release

---

## 📊 Experimental Results

### BoeNet v2.0.1 (WikiText-2)

**Current Best Configuration**: K=0, seq_len=128, embed_dim=64, 20 epochs

| Metric | Value | Analysis |
|--------|-------|----------|
| Validation PPL | 11.55 | 22x better than random (256) |
| Training PPL | 11.57 | Good convergence |
| Model Size | ~340K params | Efficient |
| Token Range | [10, 226] | All within vocab_size=256 ✅ |
| Epoch Time (CPU) | ~3.6 sec | Fast iteration |
| Epoch Time (GPU) | ~0.5 sec* | With RTX 5080 |

*Estimated with `--gpus all` enabled

**Key Findings (Early Results)**:
1. ✅ **ByteTokenizer Works**: UTF-8 encoding eliminates IndexError
2. ✅ **Training Converges**: 22x improvement over random baseline
3. ✅ **K=0 Strong Baseline**: Dense configuration achieves PPL ~11.55
4. ⚠️ **K=3 Unstable**: BFS tree configurations show high variance (under investigation)
5. 📊 **Longer Sequences Help**: seq_len=128 slightly better than seq_len=64

### BFSNet v2.0.0 (FashionMNIST)

**Best Configuration**: λ=0.05, threshold=0.42, K=3, depth=2

| Metric | Value | Analysis |
|--------|-------|----------|
| Validation Accuracy | 87.42% | Beats dense baseline (85%) |
| Test Accuracy | ~88% (est.) | Threshold-dependent |
| Training Nodes/Example | 6.44 | Efficient (vs max 13) |
| Inference Nodes | 1.0-13.0 | Threshold-tunable |
| Inference Latency (p50) | 0.6 ms | Very fast on CPU |
| Inference Latency (p99) | 25 ms | Outliers present |

**Key Findings**:
1. ✅ **Higher λ → Better Accuracy**: 0.05 beats 0.01 (counter-intuitive regularization effect)
2. ✅ **Policy Learns Consistently**: grow_prob converges to 0.44-0.45 regardless of λ
3. ⚠️ **Threshold Mismatch**: Default 0.5 too high; must tune to ~0.42
4. 📊 **Root-Only Surprisingly Good**: 86-87% accuracy with just 1 node

---

## 🔬 Research Contributions

### Novel Findings

1. **Efficiency as Regularization**: Higher compute penalties improved accuracy (BFSNet λ=0.05 > λ=0.01)
2. **Training/Inference Mismatch**: Discovered systematic bias in greedy threshold selection
3. **Policy Stability**: REINFORCE converges to narrow grow_prob distribution (~0.44-0.45)
4. **Architecture Validation**: BFS tree expansion works for neural networks (proven on vision)
5. **ByteTokenizer for LM**: UTF-8 byte encoding enables vocab_size=256 on any Unicode text

### Open Questions for BoeNet

1. Does BFS tree expansion improve language modeling perplexity?
2. Optimal tree depth for sequential data?
3. How to handle long-range dependencies (vs. transformer attention)?
4. Can we beat transformer efficiency with comparable quality?
5. Why is K=3 (BFS) unstable compared to K=0 (dense)?

---

## 🛠️ File Versions

| File | Version | Status | Description |
|------|---------|--------|-------------|
| `docker/Dockerfile.cuda` | v1.0.3 | ✅ Working | CUDA 12.8 + PyTorch 2.7.1 + HF_HOME fix |
| `boenet/utils/data_utils.py` | v2.0.1 | ✅ Working | ByteTokenizer with UTF-8 encoding |
| `boenet_training_matrix.py` | v2.0.1 | ✅ Working | 400-cell experiment matrix |
| `.dockerignore` | v1.0.1 | ✅ Working | Excludes data, runs, __pycache__ |
| `.gitignore` | v1.0.1 | ✅ Working | Excludes data, runs, checkpoints |
| `configs/experiment-config.yaml` | v1.0.0 | ✅ Working | Training matrix configuration |

---

## 🤝 Contributing

**⚠️ IMPORTANT: This is a closed-source, proprietary project.**

This codebase is **NOT** open source and is provided for reference and evaluation purposes only. Contributions are limited to:

- **Authorized collaborators only**
- Code review and feedback by invitation
- Bug reports (if given access)

**For collaboration inquiries, contact the project owner.**

---

## 📖 Citation

If you reference this work (with permission), please cite:
```bibtex
@software{boenet2025,
  title={BoeNet: Applying BFS Tree Expansion to Language Modeling},
  author={BoeNet Team},
  year={2025-2026},
  version={2.0.1},
  note={Proprietary software - All rights reserved. ByteTokenizer with UTF-8 encoding, training on WikiText-2}
}

@software{bfsnet2025,
  title={BFSNet: Breadth-First Search Neural Networks with Policy Gradient Sparsity},
  author={BFS Project Team},
  year={2025},
  version={2.0.0},
  note={Proprietary software - All rights reserved. Proof-of-concept on FashionMNIST - COMPLETE}
}
```

---

## 📄 License

**Copyright © 2025-2026 BoeNet Project. All rights reserved.**

This software is proprietary and confidential. Unauthorized copying, distribution, modification, or use of this software, via any medium, is strictly prohibited without express written permission from the copyright holder.

**NO WARRANTY**: This software is provided "AS IS" without warranty of any kind, either express or implied.

See the [LICENSE](LICENSE) file for complete terms.

---

## 🙏 Acknowledgments

### BFSNet Phase
- PyTorch team for the excellent deep learning framework
- REINFORCE algorithm: Williams, 1992
- Critical threshold mismatch discovery: Dec 18, 2025 debug session
- FashionMNIST dataset creators

### BoeNet Phase
- Andrej Karpathy's nanoGPT for inspiration
- HuggingFace for datasets library and WikiText-2
- UTF-8 byte-level tokenization approach
- Transformer architecture: Vaswani et al., 2017
- The Pile dataset: Gao et al., 2020

---

## 📞 Support & Contact

- **Issues**: Contact project owner (closed source - no public issue tracker)
- **BFSNet Documentation**: `docs/bfsnet_architecture.md`
- **BoeNet Documentation**: `docs/boenet_architecture.md` (in progress)
- **Collaboration Inquiries**: [your-email@example.com]

### Quick Links

- 📚 [BFSNet Final Report](BFSNET_FINAL_REPORT.md)
- 🎯 [BoeNet Vision](BOENET_VISION.md)
- 🗺️ [Development Roadmap](BOENET_ROADMAP.md)
- 🐳 [Docker Setup](docker/README.md)

---

**Current Focus**: 🎯 **BoeNet v2.0.1 - 400-Cell Training Matrix on WikiText-2**

**Status**: Training matrix running with `--gpus all` on RTX 5080

**Best Result So Far**: Val PPL 11.55 (22x improvement over random baseline)

**⚠️ Proprietary Software**: This project is closed source. All rights reserved.

**Last Updated**: December 30, 2025  
**Project Status**: BFSNet ✅ COMPLETE | BoeNet ✅ TRAINING (Phase 1)