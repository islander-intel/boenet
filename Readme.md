# BoeNet: Biological Optimized Enhanced Neural Network

**Applying BFS Tree Expansion to Language Modeling**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![License: Proprietary](https://img.shields.io/badge/License-Proprietary-red.svg)](LICENSE)

---

## 🎯 Project Overview

This repository contains the evolution from **BFSNet** (vision) to **BoeNet** (language), demonstrating how breadth-first search tree expansion with adaptive compute can be applied across modalities.

### Current Status

| Project | Status | Description |
|---------|--------|-------------|
| **BFSNet v2.0.0** | ✅ **COMPLETE** | Vision model on FashionMNIST - REINFORCE policy gradients for adaptive tree expansion |
| **BoeNet v0.1.0** | 🚧 **IN PROGRESS** | Language model applying BFS principles to sequential text processing |

---

## 📚 Table of Contents

- [BFSNet: What We Accomplished](#bfsnet-what-we-accomplished)
- [BoeNet: The Vision](#boenet-the-vision)
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
2. **Efficiency = Regularization**: Higher lambda penalty improved accuracy (counter-intuitive!)
3. **Threshold Tuning Critical**: Default 0.5 too high; must match learned grow_prob distribution
4. **Task-Specific**: FashionMNIST may not require deep hierarchical reasoning (root-only worked well)

**👉 See `docs/bfsnet_architecture.md` for complete technical retrospective**

---

## 🚀 BoeNet: The Vision

**BoeNet (Biological Optimized Enhanced Net)** applies BFSNet's adaptive compute principles to language modeling, with the ultimate goal of building **Arcus LLM** - a personal language model competitive with ChatGPT.

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
Input: [B, seq_len] (token IDs)
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

**Phase 1: Character-Level Proof of Concept** (4-6 weeks, $0 cost)
- Dataset: Shakespeare (300K characters)
- Model: 10M parameters
- Goal: Match nanoGPT perplexity with 50% fewer FLOPs
- **Status**: 🎯 **NEXT STEP - STEPS 1-3**

**Phase 2: Word-Level (TinyStories)** (4-6 weeks, $500 cost)
- Dataset: TinyStories (2M stories, 2GB)
- Model: 25M parameters
- Goal: Coherent 2-3 sentence generation

**Phase 3: Production Scale** (8-12 weeks, $5K-$10K cost)
- Dataset: OpenWebText (40GB) → The Pile (300B+ tokens)
- Model: 125M → 1B parameters
- Goal: Arcus LLM v1.0 - competitive personal assistant

**Phase 4: Arcus LLM** (6-12 months, $30K-$100K cost)
- Model: 7B+ parameters
- Goal: ChatGPT-level performance with 3× faster inference

---

## ⚡ Quick Start

### BFSNet (Vision - Historical Reference)
```bash
# Train BFSNet on FashionMNIST
python train_fmnist_bfs.py \
    --epochs 10 \
    --lambda_efficiency 0.05 \
    --greedy_threshold 0.42 \
    --save_path checkpoints/bfsnet_final.pt

# Inference with policy analysis
python infer_fmnist_bfs.py \
    --ckpt checkpoints/bfsnet_final.pt \
    --debug_policy \
    --cpu
```

### BoeNet (Language - Active Development)
```bash
# Step 1: Character-level proof of concept
# [COMING SOON - Implementation in progress]

# Download Shakespeare dataset
python scripts/download_shakespeare.py

# Train character-level BoeNet
python train_char_boenet.py \
    --dataset shakespeare \
    --epochs 10 \
    --lambda_efficiency 0.05 \
    --save_path checkpoints/boenet_char.pt

# Generate text
python generate_boenet.py \
    --ckpt checkpoints/boenet_char.pt \
    --prompt "To be or not to be" \
    --max_tokens 100
```

---

## 🏗️ Architecture Overview

### BFSNet (Historical)

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

### BoeNet (Active Development)

**Key Innovation**: BFS tree expansion per token for sequential language modeling

**Architecture Components**:

1. **BFSLanguageCell**: Processes one token through BFS tree
   - Input: Token embedding + hidden state from previous token
   - Policy: REINFORCE decides which children to expand
   - Output: New hidden state (like LSTM/GRU)

2. **Recurrent Processing**: Chain cells across sequence
```python
   for t in range(seq_len):
       hidden[t], policy_loss[t] = BFSCell(
           token_embed=embed(tokens[t]),
           hidden_prev=hidden[t-1]
       )
```

3. **Reward Function**: Minimize perplexity with FLOPs efficiency
```python
   reward = -perplexity - lambda_efficiency × (flops_used / max_flops)
```

**Target Configuration** (Phase 1):
```python
BoeNet(
    vocab_size=256,         # ASCII characters
    embed_dim=64,
    hidden_dim=128,
    max_depth=2,
    max_children=3,
    num_layers=4,           # Stacked BFS cells
    greedy_threshold=0.42   # Learned from BFSNet
)
```

---

## 💻 Installation

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA 12.1+ (optional, for GPU)

### Setup
```bash
# Clone repository
git clone https://github.com/your-repo/boenet.git
cd boenet

# Create virtual environment
python3 -m venv boenet-env
source boenet-env/bin/activate  # On Windows: boenet-env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# For BoeNet language modeling (additional):
pip install tokenizers transformers datasets
```

### Docker (Recommended)
```bash
# Build BFSNet image (vision tasks)
docker build -t bfsnet:cuda -f docker/Dockerfile.cuda .

# Build BoeNet image (language tasks) - COMING SOON
docker build -t boenet:cuda -f docker/Dockerfile.boenet .
```

---

## 📖 Usage Examples

### Example 1: BFSNet Training (Historical)
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

### Example 2: BoeNet Character-Level (In Development)
```python
import torch
from boenet_model import BoeNet

# Create character-level model
model = BoeNet(
    vocab_size=256,        # ASCII
    embed_dim=64,
    hidden_dim=128,
    max_depth=2,
    max_children=3,
    num_layers=4
)

# Training (similar to BFSNet but on sequences)
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

# Generation
model.eval()
prompt = "To be or not to be"
generated = model.generate(
    prompt_tokens=encode(prompt),
    max_new_tokens=100,
    temperature=0.8
)
print(decode(generated))
```

---

## 📚 Documentation

### BFSNet (Vision - Complete)

| Document | Description |
|----------|-------------|
| `docs/bfsnet_architecture.md` | Complete technical retrospective of BFSNet v2.0.0 |
| `docs/bfsnet_lessons_learned.md` | Key insights and lessons for BoeNet |
| `configs/README.md` | BFSNet configuration guide (historical reference) |
| `docker/README.md` | Docker setup for BFSNet (vision tasks) |
| `scripts/README.md` | BFSNet utility scripts |
| `tests/README.md` | BFSNet test suite and results |

### BoeNet (Language - In Progress)

| Document | Status | Description |
|----------|--------|-------------|
| `docs/boenet_architecture.md` | 📝 DRAFT | BoeNet technical specification |
| `docs/boenet_roadmap.md` | 📝 DRAFT | Development phases and milestones |
| `BOENET_VISION.md` | 📝 DRAFT | Project vision and goals |
| `TRANSITION_GUIDE.md` | 📝 DRAFT | BFSNet → BoeNet migration guide |

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

### 🚧 In Progress: BoeNet v0.1.0 (Jan-Feb 2026)

#### Phase 1: Character-Level Validation (Weeks 1-6)

- [ ] **Week 1**: Implement BFSLanguageCell architecture
- [ ] **Week 2**: Implement character-level tokenization and data loading
- [ ] **Week 3**: Train on Shakespeare dataset (300K chars)
- [ ] **Week 4**: Validate perplexity matches or beats baseline LSTM
- [ ] **Week 5**: Measure FLOPs efficiency (target: 50% reduction)
- [ ] **Week 6**: Documentation and analysis

**Success Criteria**:
- Character-level perplexity ≤ baseline LSTM
- 30-50% FLOPs reduction vs. full tree expansion
- Coherent character-by-character generation

#### Phase 2: Word-Level (TinyStories) (Weeks 7-12)

- [ ] Implement BPE tokenization
- [ ] Scale to 25M parameters
- [ ] Train on TinyStories (2M stories)
- [ ] Generate coherent 2-3 sentence stories
- [ ] Compare to small GPT baseline

#### Phase 3: Production Scale (Months 4-6)

- [ ] Scale to 125M-1B parameters
- [ ] Train on OpenWebText → The Pile
- [ ] Benchmark on standard LLM tasks (MMLU, HellaSwag)
- [ ] Optimize inference speed

### 🎯 Future: Arcus LLM (2026-2027)

- [ ] Scale to 7B+ parameters
- [ ] Advanced features (adaptive context, controllable speed/quality)
- [ ] Domain specialization (coding, math)
- [ ] Public release and community engagement

---

## 📊 Experimental Results

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

**Implications for BoeNet**:
- REINFORCE is stable and reliable
- Efficiency penalties work but need careful tuning
- Threshold tuning will be critical for language tasks too
- May need adaptive thresholds for different sequence positions

---

## 🔬 Research Contributions

### Novel Findings

1. **Efficiency as Regularization**: Higher compute penalties improved accuracy (BFSNet λ=0.05 > λ=0.01)
2. **Training/Inference Mismatch**: Discovered systematic bias in greedy threshold selection
3. **Policy Stability**: REINFORCE converges to narrow grow_prob distribution (~0.44-0.45)
4. **Architecture Validation**: BFS tree expansion works for neural networks (proven on vision)

### Open Questions for BoeNet

1. Does BFS tree expansion improve language modeling perplexity?
2. Optimal tree depth for sequential data?
3. How to handle long-range dependencies (vs. transformer attention)?
4. Can we beat transformer efficiency with comparable quality?

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
  version={0.1.0},
  note={Proprietary software - All rights reserved. Evolution from BFSNet v2.0.0 (vision) to BoeNet (language)}
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
- Transformer architecture: Vaswani et al., 2017
- Character-level language modeling: Karpathy et al., 2015
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
- 🔄 [Transition Guide](TRANSITION_GUIDE.md)

---

**Current Focus**: 🎯 **Phase 1 - Character-Level BoeNet (Steps 1-3)**

**Status**: Architecture design complete, implementation starting week of Jan 6, 2026

**⚠️ Proprietary Software**: This project is closed source. All rights reserved.

**Last Updated**: December 20, 2025  
**Project Status**: BFSNet ✅ COMPLETE | BoeNet 🚧 IN PROGRESS (Phase 1)