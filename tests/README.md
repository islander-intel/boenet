# BFSNet & BoeNet Test Suite

**Repository**: BoeNet (formerly BFSNet)  
**Purpose**: Testing infrastructure for vision (BFSNet - COMPLETE) and language (BoeNet - IN PROGRESS) tasks  
**Last Updated**: December 20, 2025

---

## 📋 Overview

This directory contains comprehensive test suites for both:
1. **BFSNet (Vision)** - ✅ COMPLETE - Test results documented for reference
2. **BoeNet (Language)** - 🚧 IN PROGRESS - Test strategy under development

### Directory Structure
```
tests/
├── README.md                          # This file
├── conftest.py                        # Shared pytest fixtures
├── pytest.ini                         # Pytest configuration
│
├── bfsnet/                            # BFSNet (Vision) - COMPLETE
│   ├── README.md                      # BFSNet test documentation
│   ├── RESULTS.md                     # Final test results
│   ├── unit/                          # Unit tests
│   │   ├── test_gradient_flow.py
│   │   ├── test_dense_baseline.py
│   │   ├── test_sparse_dense_match.py
│   │   ├── test_checkpoint_roundtrip.py
│   │   ├── test_device_fallback.py
│   │   ├── test_edge_cases.py
│   │   ├── test_numerical_stability.py
│   │   └── test_execution_modes.py
│   └── integration/                   # Integration tests
│       ├── test_pipeline_smoke.py
│       ├── test_csv_output.py
│       └── test_config_loading.py
│
└── boenet/                            # BoeNet (Language) - IN PROGRESS
    ├── README.md                      # BoeNet test documentation
    ├── TEST_PLAN.md                   # Phase 1 test strategy
    ├── unit/                          # Unit tests
    │   ├── test_tokenization.py       # Character/BPE tokenization
    │   ├── test_sequence_processing.py # Sequence batching
    │   ├── test_bfs_language_cell.py  # BFSLanguageCell
    │   ├── test_gradient_flow.py      # Sequence gradients
    │   ├── test_generation.py         # Text generation
    │   └── test_perplexity.py         # Perplexity calculation
    └── integration/                   # Integration tests
        ├── test_char_training.py      # Character-level E2E
        ├── test_baseline_comparison.py # vs LSTM/Transformer
        └── test_text_generation.py    # Generation quality
```

---

## 🎯 Quick Start

### Running All Tests
```bash
# Run all tests (BFSNet + BoeNet)
pytest tests/ -v

# Run only BFSNet tests (historical)
pytest tests/bfsnet/ -v

# Run only BoeNet tests (active development)
pytest tests/boenet/ -v

# Run specific test categories
pytest tests/ -v -m unit          # Unit tests only
pytest tests/ -v -m integration   # Integration tests only
pytest tests/ -v -m slow          # Slow tests only
```

### Running with Docker
```bash
# BFSNet tests in Docker
docker run --rm \
    -v $(pwd):/app \
    bfsnet:cpu pytest tests/bfsnet/ -v

# BoeNet tests in Docker
docker run --rm \
    -v $(pwd):/app \
    boenet:cpu pytest tests/boenet/ -v
```

### Quick Validation (CI/CD)
```bash
# Fast unit tests only (~30 seconds)
pytest tests/ -v -m "unit and not slow"

# Integration tests (~5 minutes)
pytest tests/ -v -m integration
```

---

## ✅ BFSNet Test Suite (Vision - COMPLETE)

**⚠️ STATUS**: BFSNet testing is **COMPLETE**. Tests are preserved for:
- Historical record
- Regression testing (if code is reused)
- Methodology reference for BoeNet

### Test Categories

| Category | Tests | Coverage | Status |
|----------|-------|----------|--------|
| **Unit Tests** | 8 files, 45+ tests | ~85% | ✅ COMPLETE |
| **Integration Tests** | 3 files, 12+ tests | Pipeline coverage | ✅ COMPLETE |
| **Total** | 11 files, 57+ tests | Core functionality | ✅ COMPLETE |

### Unit Test Results

#### 1. test_gradient_flow.py (✅ ALL PASSED)

**Purpose**: Verify gradients flow correctly through BFS tree.

**Tests**:
- ✅ `test_logits_require_grad` - Output requires gradients
- ✅ `test_root_fc_receives_gradient` - Root layer gets gradients
- ✅ `test_output_fc_receives_gradient` - Output layer gets gradients
- ✅ `test_branch_gate_receives_gradient` - Policy network gets gradients
- ✅ `test_child_fc_receives_gradient` - Child layers get gradients (K>0)
- ✅ `test_no_detach_on_training_path` - No detached tensors in forward pass

**Result**: All gradients flow correctly, no issues found.

---

#### 2. test_dense_baseline.py (✅ ALL PASSED)

**Purpose**: Verify K=0 behaves exactly like standard MLP.

**Tests**:
- ✅ `test_k0_forward_shape` - Output shape [B, num_classes]
- ✅ `test_k0_deterministic` - Same input → same output
- ✅ `test_k0_parameter_count` - Only root_fc and output_fc have params
- ✅ `test_k0_trains_normally` - Loss decreases over steps
- ✅ `test_k0_matches_manual_mlp` - Outputs match equivalent MLP

**Result**: K=0 is a perfect dense baseline, no BFS overhead.

---

#### 3. test_sparse_dense_match.py (⚠️ 1 KNOWN ISSUE)

**Purpose**: Verify sparse and dense modes produce consistent outputs.

**Tests**:
- ✅ `test_same_weights_same_output` - Modes match (within tolerance)
- ✅ `test_trace_counts_match` - Node counts consistent
- ⚠️ `test_soft_full_vs_sparse_gradient_magnitude` - Gradient magnitudes differ

**Known Issue**:
- Sparse mode has lower gradient magnitudes (expected due to stochastic sampling)
- Not a bug - different optimization dynamics
- **Impact**: Low - both modes train successfully

**Result**: Outputs match, gradient difference is expected behavior.

---

#### 4. test_checkpoint_roundtrip.py (✅ ALL PASSED)

**Purpose**: Verify save/load preserves model state.

**Tests**:
- ✅ `test_save_load_weights_match` - Weights preserved exactly
- ✅ `test_save_load_output_match` - Same outputs after load
- ✅ `test_save_load_config_preserved` - Config dict preserved
- ✅ `test_strict_false_handles_missing_keys` - Backwards compatibility

**Result**: Checkpoint format is robust and reliable.

---

#### 5. test_device_fallback.py (✅ ALL PASSED)

**Purpose**: Verify device selection and fallback logic.

**Tests**:
- ✅ `test_cpu_explicit` - CPU selection works
- ✅ `test_cuda_fallback_to_cpu` - Falls back to CPU when CUDA unavailable
- ✅ `test_mps_fallback_to_cpu` - Falls back to CPU when MPS unavailable
- ✅ `test_fallback_logs_warning` - Warning emitted on fallback
- ✅ `test_model_and_data_same_device` - Model and data on same device

**Result**: Device management is robust and user-friendly.

---

#### 6. test_edge_cases.py (✅ ALL PASSED)

**Purpose**: Verify behavior at boundary conditions.

**Tests**:
- ✅ `test_max_depth_zero` - max_depth=0 works (no expansion)
- ✅ `test_max_children_zero` - max_children=0 works (dense)
- ✅ `test_batch_size_one` - Single sample doesn't crash
- ✅ `test_empty_frontier_early_exit` - Frontier exhaustion handled
- ✅ `test_all_nodes_pruned` - All pruned nodes still produce valid output

**Result**: No edge case failures, architecture is robust.

---

#### 7. test_numerical_stability.py (✅ ALL PASSED)

**Purpose**: Detect NaN/Inf in outputs and gradients.

**Tests**:
- ✅ `test_no_nan_in_logits` - No NaN in outputs
- ✅ `test_no_inf_in_logits` - No Inf in outputs
- ✅ `test_no_nan_in_gradients` - No NaN in gradients
- ✅ `test_no_inf_in_gradients` - No Inf in gradients
- ✅ `test_extreme_input_values` - Handles very large/small inputs

**Result**: Numerically stable, no overflow/underflow issues.

---

#### 8. test_execution_modes.py (✅ ALL PASSED)

**Purpose**: Verify warmup → sparse transition.

**Tests**:
- ✅ `test_soft_full_mode_set` - exec_mode attribute correct
- ✅ `test_sparse_mode_set` - Mode switches after warmup
- ✅ `test_pooling_override_applied` - Warmup pooling override works
- ✅ `test_pooling_override_removed` - Override removed after warmup
- ✅ `test_mode_switch_mid_training` - Trains correctly through transition

**Result**: Execution mode switching is reliable.

---

### Integration Test Results

#### 1. test_pipeline_smoke.py (✅ ALL PASSED)

**Purpose**: End-to-end smoke test of full training pipeline.

**Tests**:
- ✅ `test_training_matrix_completes` - bfs_training_matrix.py exits cleanly
- ✅ `test_csv_created` - matrix_results.csv exists
- ✅ `test_jsonl_created` - matrix_results.jsonl exists
- ✅ `test_logs_created` - Run logs exist
- ✅ `test_no_python_errors` - No exceptions in output

**Result**: Full pipeline works end-to-end.

---

#### 2. test_csv_output.py (✅ ALL PASSED)

**Purpose**: Validate CSV structure and content.

**Tests**:
- ✅ `test_csv_has_required_columns` - All expected columns present
- ✅ `test_csv_no_empty_rows` - No empty rows
- ✅ `test_csv_run_ids_unique` - run_id values unique
- ✅ `test_csv_val_acc_in_range` - Accuracy in [0, 100]
- ✅ `test_csv_no_nan_in_metrics` - No NaN in core metrics
- ✅ `test_csv_row_count_matches_grid` - Row count matches config

**Result**: CSV output format is correct and complete.

---

#### 3. test_config_loading.py (✅ ALL PASSED)

**Purpose**: Verify configuration files parse correctly.

**Tests**:
- ✅ `test_experiment_config_loads` - experiment-config.yaml parses
- ✅ `test_test_config_loads` - test-config.yaml parses
- ✅ `test_missing_config_error` - Missing file raises error
- ✅ `test_invalid_yaml_error` - Malformed YAML raises error
- ✅ `test_unknown_keys_ignored` - Unknown keys don't crash

**Result**: Config loading is robust and validates inputs.

---

### BFSNet Test Summary

**Total Tests**: 57+ tests  
**Pass Rate**: 98.2% (56/57 passed)  
**Known Issues**: 1 (gradient magnitude difference - expected)  
**Coverage**: ~85% (core functionality)  
**Status**: ✅ **COMPLETE** - No further testing needed

**Key Findings**:
1. ✅ All core functionality works correctly
2. ✅ No numerical stability issues
3. ✅ Device fallback is robust
4. ✅ Checkpoint format is reliable
5. ⚠️ Sparse/dense gradient magnitudes differ (expected, not a bug)

---

## 🚀 BoeNet Test Suite (Language - IN PROGRESS)

**⚠️ STATUS**: BoeNet testing is in **ACTIVE DEVELOPMENT**.

### Test Strategy Overview

BoeNet requires fundamentally different tests than BFSNet due to sequential processing:

| Aspect | BFSNet (Vision) | BoeNet (Language) |
|--------|-----------------|-------------------|
| **Input** | Single image | Variable-length sequence |
| **Output** | Class logits | Token logits per position |
| **Metric** | Accuracy | Perplexity |
| **Generation** | N/A | Autoregressive text generation |
| **Temporal** | No temporal dependencies | Hidden state across timesteps |

### Planned Test Categories

| Category | Tests (Planned) | Priority | Status |
|----------|-----------------|----------|--------|
| **Unit Tests** | 15+ tests | 🔴 CRITICAL | 🚧 IN PROGRESS |
| **Integration Tests** | 8+ tests | 🟡 HIGH | ⏳ PLANNED |
| **Baseline Comparisons** | 4+ tests | 🟢 MEDIUM | ⏳ PLANNED |
| **Total** | 27+ tests | - | Phase 1 focus |

---

### Unit Tests (BoeNet)

#### 1. test_tokenization.py (🚧 IN PROGRESS)

**Purpose**: Verify character-level and BPE tokenization.

**Planned Tests**:
- [ ] `test_char_encode_decode` - Character encoding round-trip
- [ ] `test_char_vocab_size` - Vocabulary size correct (256 for ASCII)
- [ ] `test_bpe_encode_decode` - BPE encoding round-trip
- [ ] `test_bpe_vocab_size` - BPE vocab size matches config
- [ ] `test_special_tokens` - Special tokens handled correctly
- [ ] `test_tokenizer_deterministic` - Same input → same tokens

**Status**: 🚧 IN PROGRESS - Critical for Phase 1

**Example Test**:
```python
def test_char_encode_decode():
    """Test character-level tokenization round-trip."""
    from boenet.tokenizer import CharTokenizer
    
    tokenizer = CharTokenizer(vocab_size=256)
    text = "To be or not to be, that is the question."
    
    # Encode
    tokens = tokenizer.encode(text)
    assert len(tokens) == len(text)
    assert all(0 <= t < 256 for t in tokens)
    
    # Decode
    decoded = tokenizer.decode(tokens)
    assert decoded == text
```

---

#### 2. test_sequence_processing.py (🚧 IN PROGRESS)

**Purpose**: Verify sequence batching and padding.

**Planned Tests**:
- [ ] `test_batch_padding` - Sequences padded to same length
- [ ] `test_attention_mask` - Padding mask generated correctly
- [ ] `test_variable_length_sequences` - Different lengths in batch
- [ ] `test_truncation` - Long sequences truncated correctly
- [ ] `test_position_ids` - Position IDs generated correctly

**Status**: 🚧 IN PROGRESS

---

#### 3. test_bfs_language_cell.py (🚧 IN PROGRESS)

**Purpose**: Verify BFSLanguageCell processes tokens correctly.

**Planned Tests**:
- [ ] `test_cell_forward_shape` - Output shape correct
- [ ] `test_hidden_state_propagation` - Hidden state carries forward
- [ ] `test_policy_output` - Policy loss generated
- [ ] `test_bfs_expansion_per_token` - Tree built per token
- [ ] `test_cell_stacking` - Multiple cells stack correctly

**Status**: 🚧 IN PROGRESS - Core architecture test

**Example Test**:
```python
def test_cell_forward_shape():
    """Test BFSLanguageCell output shape."""
    from boenet.model import BFSLanguageCell
    
    cell = BFSLanguageCell(
        embed_dim=64,
        hidden_dim=128,
        max_children=3,
        max_depth=2
    )
    
    batch_size = 4
    token_embed = torch.randn(batch_size, 64)
    hidden_prev = torch.randn(batch_size, 128)
    
    hidden_next, policy_loss = cell(token_embed, hidden_prev)
    
    assert hidden_next.shape == (batch_size, 128)
    assert policy_loss.ndim == 0  # Scalar
```

---

#### 4. test_gradient_flow.py (⏳ PLANNED)

**Purpose**: Verify gradients flow through sequence.

**Planned Tests**:
- [ ] `test_sequence_gradients` - Gradients flow across timesteps
- [ ] `test_bptt_gradients` - Backprop through time works
- [ ] `test_policy_gradients` - REINFORCE gradients correct
- [ ] `test_no_gradient_explosion` - Gradient clipping works

**Status**: ⏳ PLANNED

---

#### 5. test_generation.py (⏳ PLANNED)

**Purpose**: Verify autoregressive text generation.

**Planned Tests**:
- [ ] `test_greedy_generation` - Greedy decoding works
- [ ] `test_temperature_sampling` - Temperature affects diversity
- [ ] `test_top_k_sampling` - Top-k restricts choices
- [ ] `test_top_p_sampling` - Nucleus sampling works
- [ ] `test_generation_stops_at_eos` - Stops at end-of-sequence token

**Status**: ⏳ PLANNED - Critical for Phase 1 validation

---

#### 6. test_perplexity.py (⏳ PLANNED)

**Purpose**: Verify perplexity calculation.

**Planned Tests**:
- [ ] `test_perplexity_calculation` - Math is correct
- [ ] `test_perplexity_vs_cross_entropy` - Relationship correct
- [ ] `test_perplexity_decreases_on_training` - Improves with training
- [ ] `test_perplexity_matches_baseline` - Comparable to known values

**Status**: ⏳ PLANNED

---

### Integration Tests (BoeNet)

#### 1. test_char_training.py (⏳ PLANNED)

**Purpose**: End-to-end character-level training.

**Planned Tests**:
- [ ] `test_shakespeare_training_completes` - Training runs to completion
- [ ] `test_perplexity_improves` - Perplexity decreases over epochs
- [ ] `test_generates_valid_text` - Generated text is valid
- [ ] `test_checkpoint_saves` - Checkpoints created

**Status**: ⏳ PLANNED - Phase 1 validation

---

#### 2. test_baseline_comparison.py (⏳ PLANNED)

**Purpose**: Compare BoeNet to LSTM/Transformer baselines.

**Planned Tests**:
- [ ] `test_vs_lstm_perplexity` - Perplexity comparable to LSTM
- [ ] `test_vs_lstm_flops` - FLOPs lower than LSTM
- [ ] `test_vs_transformer_quality` - Quality comparable (within 10%)

**Status**: ⏳ PLANNED - Phase 1 success criteria

---

#### 3. test_text_generation.py (⏳ PLANNED)

**Purpose**: Validate generated text quality.

**Planned Tests**:
- [ ] `test_generation_coherence` - Text is coherent
- [ ] `test_generation_grammar` - Basic grammar correct
- [ ] `test_generation_diversity` - Multiple samples are different
- [ ] `test_generation_prompt_conditioning` - Prompt affects output

**Status**: ⏳ PLANNED

---

### BoeNet Test Priorities for Phase 1

**Week 1-2 (Character-Level Setup)**:
1. 🔴 `test_tokenization.py` - CRITICAL
2. 🔴 `test_bfs_language_cell.py` - CRITICAL
3. 🟡 `test_sequence_processing.py` - HIGH

**Week 3-4 (Training Validation)**:
4. 🔴 `test_char_training.py` - CRITICAL
5. 🔴 `test_generation.py` - CRITICAL
6. 🟡 `test_perplexity.py` - HIGH

**Week 5-6 (Baseline Comparison)**:
7. 🟡 `test_baseline_comparison.py` - HIGH
8. 🟢 `test_gradient_flow.py` - MEDIUM

---

## 🔧 Test Infrastructure

### Shared Fixtures (conftest.py)

**Location**: `tests/conftest.py`

**Fixtures Provided**:

#### Vision (BFSNet)
```python
@pytest.fixture
def device():
    """Returns appropriate torch.device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture
def small_bfsnet():
    """BFSNet with minimal config for testing."""
    return BFSNet(
        input_dim=784,
        hidden_dim=16,
        output_dim=10,
        max_depth=1,
        max_children=1
    )

@pytest.fixture
def sample_batch():
    """Random image batch [B=4, D=784]."""
    return torch.randn(4, 784)
```

#### Language (BoeNet)
```python
@pytest.fixture
def char_tokenizer():
    """Character-level tokenizer."""
    from boenet.tokenizer import CharTokenizer
    return CharTokenizer(vocab_size=256)

@pytest.fixture
def small_boenet():
    """BoeNet with minimal config for testing."""
    return BoeNet(
        vocab_size=256,
        embed_dim=32,
        hidden_dim=64,
        max_depth=1,
        max_children=2,
        num_layers=2
    )

@pytest.fixture
def sample_sequence():
    """Random token sequence [B=4, seq_len=16]."""
    return torch.randint(0, 256, (4, 16))
```

---

### Pytest Configuration (pytest.ini)

**Location**: `tests/pytest.ini`
```ini
[pytest]
# Test discovery
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*

# Output
addopts = -v --tb=short --strict-markers

# Markers
markers =
    unit: Unit tests (fast, isolated)
    integration: Integration tests (slower, full pipeline)
    bfsnet: BFSNet (vision) tests
    boenet: BoeNet (language) tests
    gpu: Tests requiring GPU (skipped if unavailable)
    slow: Tests that take >10 seconds
    baseline: Baseline comparison tests

# Timeouts (requires pytest-timeout)
timeout = 300

# Warnings
filterwarnings =
    ignore::DeprecationWarning
    ignore::PendingDeprecationWarning
```

---

### Running Specific Test Categories
```bash
# Run only BFSNet tests
pytest tests/ -v -m bfsnet

# Run only BoeNet tests
pytest tests/ -v -m boenet

# Run only unit tests
pytest tests/ -v -m unit

# Run only integration tests
pytest tests/ -v -m integration

# Run only tests that don't require GPU
pytest tests/ -v -m "not gpu"

# Run only fast tests (exclude slow)
pytest tests/ -v -m "not slow"

# Run baseline comparison tests
pytest tests/ -v -m baseline

# Combine markers
pytest tests/ -v -m "boenet and unit and not slow"
```

---

## 📊 Test Coverage

### BFSNet Coverage (COMPLETE)
```bash
# Generate coverage report
pytest tests/bfsnet/ --cov=bfs_model --cov=train_fmnist_bfs --cov-report=html

# View coverage
open htmlcov/index.html
```

**Coverage Results** (Final):
```
Name                    Stmts   Miss  Cover
-------------------------------------------
bfs_model.py              450     68    85%
train_fmnist_bfs.py       180     22    88%
infer_fmnist_bfs.py       120     15    88%
utils/gating.py            85      8    91%
-------------------------------------------
TOTAL                     835    113    86%
```

**Uncovered Code**:
- Error handling paths (hard to test)
- Debug logging (not critical)
- Visualization code (manual testing only)

---

### BoeNet Coverage (Target)

**Phase 1 Target**: 80%+ coverage on core components

**Core Components to Cover**:
- `boenet_model.py` - BFSLanguageCell, BoeNet
- `train_char_boenet.py` - Training loop
- `boenet/tokenizer.py` - CharTokenizer, BPETokenizer
- `boenet/generation.py` - Text generation
- `boenet/metrics.py` - Perplexity calculation

---

## 🐛 Debugging Failed Tests

### Common Test Failures

#### 1. Shape Mismatch
```
AssertionError: assert torch.Size([4, 10]) == torch.Size([4, 128])
```

**Debug**:
```python
# Add print statements
print(f"Expected shape: {expected.shape}")
print(f"Actual shape: {actual.shape}")

# Check intermediate shapes
print(f"Hidden state shape: {hidden.shape}")
print(f"Logits shape: {logits.shape}")
```

#### 2. NaN in Loss
```
AssertionError: Loss contains NaN
```

**Debug**:
```python
# Check inputs
assert not torch.isnan(x).any(), "Input contains NaN"
assert not torch.isinf(x).any(), "Input contains Inf"

# Check gradients
for name, param in model.named_parameters():
    if param.grad is not None:
        if torch.isnan(param.grad).any():
            print(f"NaN gradient in {name}")
```

#### 3. CUDA Out of Memory
```
RuntimeError: CUDA out of memory
```

**Fix**:
```bash
# Reduce batch size in test
pytest tests/boenet/ -v -k test_char_training --batch-size 16

# Or run on CPU
pytest tests/boenet/ -v --device cpu
```

---

## 📈 Continuous Integration (CI)

### GitHub Actions Example
```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test-bfsnet:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov
      
      - name: Run BFSNet tests
        run: pytest tests/bfsnet/ -v --cov=bfs_model
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
  
  test-boenet:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov tokenizers transformers
      
      - name: Run BoeNet tests
        run: pytest tests/boenet/ -v -m "not slow" --cov=boenet
```

---

## 🗺️ Testing Roadmap

### ✅ Completed (BFSNet)
- [x] Unit test suite (45+ tests)
- [x] Integration test suite (12+ tests)
- [x] Coverage reporting
- [x] CI/CD integration
- [x] Documentation of results

### 🚧 In Progress (BoeNet Phase 1)
- [ ] Tokenization tests
- [ ] BFSLanguageCell tests
- [ ] Sequence processing tests
- [ ] Character-level training tests
- [ ] Text generation tests

### ⏳ Planned (BoeNet Phase 2+)
- [ ] BPE tokenization tests
- [ ] Word-level training tests
- [ ] Baseline comparison tests (vs GPT-2)
- [ ] Performance benchmarking tests
- [ ] ONNX export tests

---

## 📞 Support

### BFSNet Tests
- 📚 See `tests/bfsnet/README.md` for detailed documentation
- 📖 See `tests/bfsnet/RESULTS.md` for final test results
- ✅ Status: COMPLETE - All tests documented

### BoeNet Tests
- 📚 See `tests/boenet/README.md` for detailed documentation (IN PROGRESS)
- 📖 See `tests/boenet/TEST_PLAN.md` for Phase 1 strategy (IN PROGRESS)
- 🚧 Status: IN PROGRESS - Active development

### General
- 📧 Contact: [your-email@example.com]
- 📝 Issues: Contact project owner (closed source)

---

## 📖 Best Practices

### Writing Good Tests

1. **Descriptive Names**
```python
   # Good
   def test_tokenizer_handles_empty_string():
       ...
   
   # Bad
   def test_1():
       ...
```

2. **Arrange-Act-Assert Pattern**
```python
   def test_model_forward():
       # Arrange
       model = BoeNet(...)
       x = torch.randn(4, 128)
       
       # Act
       output = model(x)
       
       # Assert
       assert output.shape == (4, vocab_size)
```

3. **Isolated Tests**
   - Each test should be independent
   - Use fixtures, not shared state
   - Clean up resources (files, GPU memory)

4. **Fast Tests**
   - Use minimal configs (small models, few iterations)
   - Mark slow tests with `@pytest.mark.slow`
   - Prefer unit tests over integration tests

5. **Clear Assertions**
```python
   # Good
   assert output.shape == expected_shape, \
       f"Expected {expected_shape}, got {output.shape}"
   
   # Bad
   assert output.shape == expected_shape
```

---

**Last Updated**: December 20, 2025  
**Status**: BFSNet tests ✅ COMPLETE (86% coverage, 98% pass rate) | BoeNet tests 🚧 IN PROGRESS  
**Next Priority**: Complete `test_tokenization.py`, `test_bfs_language_cell.py` for Phase 1

**⚠️ Proprietary Software**: This project is closed source. All rights reserved.