# Changelog

All notable changes to the BFSNet → BoeNet project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

**⚠️ Proprietary Software**: This project is closed source. All rights reserved.

---

## Project Evolution
```
BFSNet (Vision)          →          BoeNet (Language)
━━━━━━━━━━━━━━━━                    ━━━━━━━━━━━━━━━━━━
FashionMNIST                        Character-level → Word-level → Production
REINFORCE on images                 REINFORCE on sequences
v1.0.0 - v2.0.0 FINAL              v0.1.0 - IN PROGRESS
✅ COMPLETE                         🚧 ACTIVE DEVELOPMENT
```

---

## [BoeNet v0.1.0] - IN PROGRESS (January 2026)

**🚧 PROJECT INITIATED - ACTIVE DEVELOPMENT**

This version marks the transition from BFSNet (vision) to BoeNet (language modeling). BoeNet applies BFS tree expansion with REINFORCE policy gradients to sequential text processing, starting with character-level proof of concept.

### Added

#### Core Architecture
- 🚧 **BFSLanguageCell**: Recurrent BFS cell for processing sequences token-by-token
- 🚧 **BoeNet Model**: Stacked BFSLanguageCell layers with hidden state propagation
- 🚧 **Character-level Tokenization**: ASCII tokenizer for Phase 1 (Shakespeare)
- ⏳ **BPE Tokenization**: Byte-pair encoding for Phase 2 (TinyStories)
- 🚧 **Text Generation**: Autoregressive sampling with temperature/top-k/top-p
- 🚧 **Perplexity Metrics**: Language model evaluation

#### Dataset Support
- ✅ **Shakespeare Download Script**: `scripts/boenet/download_shakespeare.py` (~1MB)
- ✅ **TinyStories Download Script**: `scripts/boenet/download_tinystories.py` (~2GB)
- 🚧 **Text Preprocessing Pipeline**: `scripts/boenet/preprocess_text.py`
- 🚧 **Dataset Loaders**: Character-level and word-level data loaders

#### Docker Infrastructure
- 🚧 **BoeNet CPU Docker**: `docker/Dockerfile.boenet` (Python 3.11, PyTorch 2.7.1 CPU)
- 🚧 **BoeNet CUDA Docker**: `docker/Dockerfile.boenet.cuda` (CUDA 12.8, Blackwell support)
- ✅ **Language Dependencies**: tokenizers, transformers, datasets, sentencepiece
- ✅ **Text Data Volumes**: `/app/data/text`, `/app/data/tokenizers`, `/app/data/processed`

#### Configuration Files
- 🚧 **configs/boenet/char-level-test.yaml**: Phase 1 Shakespeare minimal config
- 🚧 **configs/boenet/char-level-full.yaml**: Phase 1 War and Peace full config
- ⏳ **configs/boenet/word-level-tiny.yaml**: Phase 2 TinyStories config
- ✅ **configs/boenet/README.md**: BoeNet configuration guide

#### Testing Infrastructure
- 🚧 **tests/boenet/unit/test_tokenization.py**: Character and BPE tokenization tests
- 🚧 **tests/boenet/unit/test_bfs_language_cell.py**: BFSLanguageCell unit tests
- 🚧 **tests/boenet/unit/test_sequence_processing.py**: Sequence batching tests
- ⏳ **tests/boenet/unit/test_generation.py**: Text generation tests
- ⏳ **tests/boenet/unit/test_perplexity.py**: Perplexity calculation tests
- ⏳ **tests/boenet/integration/test_char_training.py**: E2E character-level training
- ⏳ **tests/boenet/TEST_PLAN.md**: Phase 1 test strategy

#### Scripts & Utilities
- ✅ **scripts/boenet/download_shakespeare.py**: READY
- ✅ **scripts/boenet/download_tinystories.py**: READY
- 🚧 **scripts/boenet/preprocess_text.py**: Text preprocessing utilities
- 🚧 **scripts/boenet/tokenizer_utils.py**: Tokenizer training and testing
- 🚧 **scripts/boenet/generate_text.py**: Text generation from checkpoints
- ⏳ **scripts/boenet/analyze_perplexity.py**: Perplexity analysis and comparison

#### Documentation
- ✅ **README.md**: Updated with BoeNet vision, roadmap, and architecture
- 🚧 **docs/boenet_architecture.md**: BoeNet technical specification (IN PROGRESS)
- ✅ **configs/README.md**: Updated with BoeNet configuration examples
- ✅ **docker/README.md**: Updated with BoeNet Docker setup
- ✅ **scripts/README.md**: Updated with BoeNet script documentation
- ✅ **tests/README.md**: Updated with BoeNet test strategy
- ✅ **BFSNet Docker & Testing Architecture Specification.md**: Added Part II (BoeNet)

### Changed

#### Architecture Evolution
- **Input Processing**: Images (784-dim) → Sequences (variable-length tokens)
- **Output Format**: Class logits (10-dim) → Token logits per position (vocab_size × seq_len)
- **Metric**: Accuracy (%) → Perplexity
- **Reward Function**: `acc - λ × nodes` → `-perplexity - λ × FLOPs`
- **Processing Model**: Single-shot feedforward → Recurrent BFS per token

#### Configuration Schema
- **Added**: `vocab_size`, `embed_dim`, `seq_len`, `num_layers` (language-specific)
- **Added**: `max_new_tokens`, `temperature`, `top_k`, `top_p` (generation)
- **Retained**: `max_children`, `max_depth`, `lambda_efficiency`, `greedy_threshold` (from BFSNet)
- **Retained**: `num_rollouts`, `beta_entropy` (REINFORCE params)

#### Docker Environment
- **Base Image**: Python 3.10 → Python 3.11 (latest stable)
- **PyTorch**: 2.1.0 (BFSNet CPU) → 2.7.1 (BoeNet CPU/CUDA)
- **Dependencies**: Added tokenizers, transformers, datasets, sentencepiece
- **Volumes**: Images (`/data/FashionMNIST`) → Text files (`/data/text`)

### Lessons Applied from BFSNet

The following insights from BFSNet v2.0.0 informed BoeNet design:

1. **REINFORCE Reliability**: 
   - BFSNet: Policy gradients converged stably (98% test pass rate)
   - BoeNet: Using same REINFORCE approach with confidence

2. **Efficiency as Regularization**:
   - BFSNet: λ=0.05 achieved better accuracy than λ=0.01 (87.42% vs 86.62%)
   - BoeNet: Starting with λ=0.05, treating as regularization parameter

3. **Threshold Mismatch Critical**:
   - BFSNet: Default threshold 0.5 caused zero children (policy learned ~0.44)
   - BoeNet: Implementing `--debug_policy` from day 1, planning adaptive thresholds

4. **Policy Learns Tight Distributions**:
   - BFSNet: 98% of grow_prob in [0.4, 0.5), std=0.0157
   - BoeNet: Expecting similar, planning threshold tuning from start

5. **Batch Normalization Bug**:
   - BFSNet: Batch norm in rewards caused batch-dependent rewards (FIXED)
   - BoeNet: Ensuring sample-independent reward calculation

6. **Root-Only Baseline Strong**:
   - BFSNet: Root-only achieved 86-87% (vs 87.42% full tree)
   - BoeNet: Validating that depth is needed for language (may differ from vision)

7. **Statistical Validation Important**:
   - BFSNet: Single-seed results (repeats=1) - should have used 3+
   - BoeNet: Planning 3+ seeds for Phase 1 validation

8. **Latency Percentiles Matter**:
   - BFSNet: p99 was 42× higher than p50 (outliers!)
   - BoeNet: Tracking p50, p90, p99 from start

### Development Roadmap

**Phase 1: Character-Level (Weeks 1-6) - CURRENT**
- 🚧 Week 1-2: BFSLanguageCell implementation, tokenization
- 🚧 Week 3-4: Training pipeline, perplexity tracking
- ⏳ Week 5-6: Text generation, baseline comparison (LSTM)

**Phase 2: Word-Level (Weeks 7-12) - PLANNED**
- ⏳ BPE tokenization (GPT-2 vocab)
- ⏳ TinyStories dataset (2M stories, 2GB)
- ⏳ 25M parameter model
- ⏳ Coherent 2-3 sentence generation

**Phase 3: Production Scale (Months 4-6) - PLANNED**
- ⏳ 125M-1B parameters
- ⏳ OpenWebText → The Pile datasets
- ⏳ Standard LLM benchmarks (MMLU, HellaSwag)

**Phase 4: Arcus LLM (Months 7-12+) - PLANNED**
- ⏳ 7B+ parameters
- ⏳ ChatGPT-level performance goal
- ⏳ Personal language model

### Success Criteria (Phase 1)

**Minimum Success**:
- [ ] Character-level perplexity ≤ LSTM baseline
- [ ] 30-50% FLOPs reduction vs full tree expansion
- [ ] Coherent character-by-character generation

**Target Success**:
- [ ] Perplexity matches LSTM within 5%
- [ ] Policy converges stably (no NaN, no collapse)
- [ ] Adaptive threshold tuning works

**Stretch Success**:
- [ ] Perplexity beats LSTM by 5%+
- [ ] 50%+ FLOPs reduction
- [ ] Ready to scale to Phase 2 (word-level)

### Known Issues

- ⚠️ Text dataset mounting is manual (no auto-download like FashionMNIST)
- ⚠️ Threshold tuning methodology adapted from vision, may need adjustment
- ⚠️ No baseline implementations yet (LSTM, Transformer for comparison)

### References

- See `docs/boenet_architecture.md` for technical specification (IN PROGRESS)
- See `BOENET_VISION.md` for project goals and motivation
- See `docs/bfsnet_architecture.md` for lessons from vision phase

---

## [BFSNet v2.0.0] - 2025-12-18 **[FINAL RELEASE]**

**✅ PROJECT COMPLETE - NO FURTHER DEVELOPMENT**

This is the **final release** of BFSNet (vision). All development on FashionMNIST experiments is complete. The project successfully demonstrated that BFS tree expansion with REINFORCE policy gradients works for neural networks.

### Summary

BFSNet v2.0.0 achieved **87.42% validation accuracy** on FashionMNIST, beating the dense baseline (85%) by 2.42 percentage points. This validates the core concept of adaptive tree expansion with policy gradients and provides the foundation for BoeNet (language modeling).

### Added

#### Final Features
- ✅ **Complete 48-configuration parameter sweep** on FashionMNIST
- ✅ **Greedy threshold tuning capability** via `--debug_policy` flag
- ✅ **Policy distribution analysis** in inference script
- ✅ **Comprehensive logging** with JSON summary format (`__SUMMARY__` tags)

#### Documentation (COMPLETE)
- ✅ **docs/bfsnet_architecture.md**: Complete technical retrospective
- ✅ **Bfsnet fashionmnist test plan.md**: Final results and lessons learned
- ✅ **tests/bfsnet/RESULTS.md**: Complete test suite results
- ✅ **BFSNET_FINAL_REPORT.md**: Executive summary
- ✅ **BFSNet Docker & Testing Architecture Specification.md**: Part I (Vision)

#### Test Suite (COMPLETE)
- ✅ **57+ tests** total (45 unit, 12 integration)
- ✅ **98% pass rate** (56/57 passed)
- ✅ **86% code coverage** on core functionality
- ✅ All tests documented in `tests/bfsnet/RESULTS.md`

### Changed

#### Critical Fixes
- ✅ **FIXED: Batch Normalization Bug** in efficiency penalty calculation
  - Issue: Batch norm in `_compute_rewards()` made rewards batch-dependent
  - Impact: Incorrect efficiency penalty, rewards not sample-independent
  - Fix: Moved batch norm outside reward calculation
  - Status: VERIFIED in unit tests

- ✅ **IMPROVED: Inference JSON Parsing**
  - Added `__SUMMARY__` JSON tag parsing
  - Robust handling of malformed JSON
  - Validation of metric ranges (accuracy 0-100%)

- ✅ **IMPROVED: Training Matrix CSV Output**
  - All metrics properly captured in CSV
  - Run IDs unique across experiments
  - No NaN in core metrics
  - JSONL format added for easier parsing

### Discovered Issues & Insights

#### 1. Greedy Threshold Mismatch (CRITICAL FINDING)

**Issue**: Default `greedy_threshold=0.5` caused ZERO children to be created in inference.

**Root Cause**:
- Training: Stochastic Bernoulli(grow_prob) sampling
- Inference: Deterministic `grow_prob >= threshold` decision
- Policy learned: grow_prob ≈ 0.44-0.45 (below threshold!)

**Evidence**:
```
Policy Distribution (λ=0.01, 1200 decisions):
  Mean:    0.4457
  Std dev: 0.0157
  Min:     0.3771
  Max:     0.4567
  % ≥ 0.5: 0.00%  ← ZERO decisions above threshold!
```

**Impact**:
- Inference with threshold=0.5 → root-only (1 node) → 86.95% accuracy
- Inference with threshold=0.42 → partial tree (~8 nodes) → ~88% accuracy (estimated)

**Workaround**: Set `greedy_threshold ≈ mean_grow_prob - 0.03` (empirically ~0.42)

**Status**: DOCUMENTED, workaround implemented

---

#### 2. Higher Lambda → Better Accuracy (COUNTER-INTUITIVE)

**Finding**: Stronger efficiency penalty improved accuracy!

**Evidence**:
| λ | Training Nodes | Val Accuracy | Analysis |
|---|----------------|--------------|----------|
| 0.05 | 6.44 | **87.42%** | ✅ Best accuracy |
| 0.01 | 11.80 | **86.62%** | Worse with MORE nodes |

**Hypothesis**:
- Higher λ acts as **regularization** (forces selectivity)
- Fewer nodes → faster forward pass → more gradient steps per epoch
- Task-specific: FashionMNIST may not need deep trees

**Implication**: Efficiency penalty is not just a speed knob, it affects quality!

**Status**: VALIDATED across multiple configurations

---

#### 3. Root-Only Performance Strong

**Finding**: Root-only (1 node) achieved 86-87% accuracy.

**Evidence**:
- Root-only (threshold=0.5): 86.95%
- Full tree (threshold=0.3): 86.95% (no improvement!)
- Partial tree (threshold=0.42): ~88% (marginal gain)

**Interpretation**: FashionMNIST may not require hierarchical reasoning.

**Implication**: Task may be too simple for BFS to shine; language modeling likely better fit.

**Status**: DOCUMENTED

---

#### 4. Policy Learns Narrow Distributions

**Finding**: grow_prob converged to tight range regardless of λ.

**Evidence**:
- λ=0.05: mean=0.4457, std=0.0157
- λ=0.01: mean=0.4450, std=0.0160
- 98% of decisions in [0.4, 0.5)

**Interpretation**: Policy optimized for training dynamics, not final threshold.

**Status**: EXPECTED BEHAVIOR

---

### Fixed

- ✅ Batch normalization in reward calculation (sample independence)
- ✅ Inference JSON parsing robustness
- ✅ CSV output formatting and validation
- ✅ Device fallback warnings (CUDA/MPS → CPU)
- ✅ Gradient flow verification (all layers receive gradients)
- ✅ Checkpoint save/load round-trip

### Performance Metrics (Best Configuration)

**Configuration**: λ=0.05, K=3, depth=2, threshold=0.42

| Metric | Value |
|--------|-------|
| **Validation Accuracy** | **87.42%** |
| **Test Accuracy (threshold=0.5)** | **86.95%** |
| **Test Accuracy (threshold=0.42, est.)** | **~88%** |
| **Training Nodes/Example** | **6.44** |
| **Inference Nodes (threshold=0.5)** | **1.0** (root-only) |
| **Inference Nodes (threshold=0.42)** | **~8** (partial tree) |
| **Inference Latency (p50)** | **0.60 ms** |
| **Inference Latency (p99)** | **25.47 ms** (outliers!) |
| **Dense Baseline** | **~85%** |
| **Improvement over Dense** | **+2.42%** |

### Test Results Summary

**Unit Tests**: 45 tests, 44 passed, 1 known difference
- ✅ Gradient flow verified
- ✅ Dense baseline (K=0) matches MLP
- ⚠️ Sparse/dense gradient magnitudes differ (expected)
- ✅ Checkpoint round-trip works
- ✅ Device fallback robust
- ✅ Edge cases handled
- ✅ Numerically stable
- ✅ Execution modes work

**Integration Tests**: 12 tests, 12 passed
- ✅ Pipeline smoke test
- ✅ CSV output validation
- ✅ Config loading

**Coverage**: 86% (835 statements, 113 missed)

**Pass Rate**: 98.2% (56/57)

### Known Issues (Unfixed)

1. ⚠️ **Greedy threshold must be manually tuned**
   - Not learned automatically
   - Requires post-training measurement via `--debug_policy`
   - **Workaround**: Set threshold ≈ mean_grow_prob - 0.03
   - **Impact**: Medium - workaround documented

2. ⚠️ **Training/inference mode mismatch**
   - Stochastic sampling in training vs deterministic threshold in inference
   - Causes unexpected behavior if threshold not tuned
   - **Workaround**: Threshold tuning
   - **Impact**: Medium - can be mitigated

3. ⚠️ **Latency p99 outliers**
   - 99th percentile 42× higher than median (25ms vs 0.6ms)
   - Likely JIT compilation, CPU scheduling, memory spikes
   - **Impact**: Low for most use cases, problematic for strict SLAs

4. ⚠️ **FashionMNIST may not require BFS**
   - Root-only achieves 86-87% (full tree only adds ~1%)
   - Task may be too simple to benefit from adaptive compute
   - **Impact**: Low - validates architecture, may not generalize to harder tasks

### Deprecated

- ⚠️ v1.x dense-then-mask approach (replaced by true sparse execution)
- ⚠️ Old JSON parsing (replaced by `__SUMMARY__` tag parsing)
- ⚠️ Implicit threshold=0.5 assumption (now explicit and tunable)

### Migration Guide (v1.4.0 → v2.0.0)

**Not Applicable**: BFSNet development is COMPLETE. No migration needed.

For those referencing v1.x code:
1. Replace dense-then-mask with true sparse execution (v2.0.0 model)
2. Add explicit `greedy_threshold` tuning
3. Use `--debug_policy` to measure policy distribution
4. Update configs to include `lambda_efficiency` explicitly

### References

- **Architecture**: `docs/bfsnet_architecture.md`
- **Test Results**: `tests/bfsnet/RESULTS.md`
- **Test Plan**: `Bfsnet fashionmnist test plan.md`
- **Docker Setup**: `docker/README.md`
- **Configuration**: `configs/bfsnet/README.md`

---

## [BFSNet v1.4.0] - 2025-12-11 **[LEGACY]**

**⚠️ SUPERSEDED BY v2.0.0 - Historical reference only**

### Added
- Initial BFS tree expansion implementation
- Dense-then-mask approach (not true sparse)
- FashionMNIST training pipeline
- Basic Docker support

### Known Issues (Fixed in v2.0.0)
- ❌ Dense-then-mask wastes computation
- ❌ No policy distribution analysis
- ❌ Batch norm bug in rewards
- ❌ Threshold hardcoded to 0.5

---

## [BFSNet v1.0.0] - 2025-12-01 **[LEGACY]**

**⚠️ SUPERSEDED BY v2.0.0 - Historical reference only**

### Added
- Initial project setup
- Basic BFS model architecture
- FashionMNIST data loading
- Simple MLP baseline

---

## Legend

### Status Indicators
- ✅ **COMPLETE**: Feature/task is finished and validated
- 🚧 **IN PROGRESS**: Feature/task is being actively developed
- ⏳ **PLANNED**: Feature/task is planned but not started
- ⚠️ **PARTIAL**: Feature/task is partially complete
- ❌ **DEPRECATED**: Feature/task is no longer supported

### Priority Indicators
- 🔴 **CRITICAL**: Must be completed for project to function
- 🟡 **HIGH**: Important for project success
- 🟢 **MEDIUM**: Nice to have, improves project
- ⚪ **LOW**: Optional, future consideration

---

## Version Numbering

This project uses [Semantic Versioning](https://semver.org/):

**Format**: MAJOR.MINOR.PATCH

- **MAJOR**: Incompatible architecture changes (BFSNet → BoeNet = major)
- **MINOR**: Backwards-compatible functionality additions
- **PATCH**: Backwards-compatible bug fixes

**BFSNet Versions**:
- v1.x.x: Initial development (LEGACY)
- v2.0.0: Final release (COMPLETE)

**BoeNet Versions**:
- v0.1.0: Phase 1 character-level (IN PROGRESS)
- v0.2.0: Phase 2 word-level (PLANNED)
- v0.3.0: Phase 3 production scale (PLANNED)
- v1.0.0: Arcus LLM v1.0 (PLANNED)

---

## Contributing

**⚠️ IMPORTANT**: This is a **closed-source, proprietary project**. 

Contributions are limited to:
- Authorized collaborators only
- Code review and feedback by invitation
- Bug reports (if given access)

**For collaboration inquiries, contact the project owner.**

---

## Links

- **Documentation**: `docs/` directory
- **Architecture (BFSNet)**: `docs/bfsnet_architecture.md`
- **Architecture (BoeNet)**: `docs/boenet_architecture.md` (IN PROGRESS)
- **Test Results (BFSNet)**: `tests/bfsnet/RESULTS.md`
- **Test Plan (BoeNet)**: `tests/boenet/TEST_PLAN.md` (IN PROGRESS)
- **Docker Setup**: `docker/README.md`
- **Configuration Guide**: `configs/README.md`

---

## Acknowledgments

### BFSNet Phase
- PyTorch team for the excellent deep learning framework
- REINFORCE algorithm: Williams, 1992
- FashionMNIST dataset creators
- Critical threshold mismatch discovery: December 18, 2025 debug session

### BoeNet Phase
- Andrej Karpathy's nanoGPT for inspiration and methodology
- Transformer architecture: Vaswani et al., 2017
- Character-level language modeling: Karpathy et al., 2015
- The Pile dataset: Gao et al., 2020
- TinyStories dataset: Eldan & Li, 2023

---

**Last Updated**: December 20, 2025  
**Current Version**: BFSNet v2.0.0 (COMPLETE) | BoeNet v0.1.0 (IN PROGRESS)  
**Project Status**: Transitioning from vision to language modeling  
**License**: Proprietary - All rights reserved

**⚠️ Proprietary Software**: This project is closed source. All rights reserved.