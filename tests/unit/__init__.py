# ============================================================================
# BFSNet Unit Tests Package
# ============================================================================
#
# This package contains unit tests for individual BFSNet components.
# Each test module focuses on a specific aspect of the model's functionality.
#
# Design Principles:
# ------------------
#   - Fast: Each test should complete in under 1 second (except @slow marked)
#   - Isolated: No dependencies between tests
#   - Focused: Test one thing per test function
#   - Reproducible: Use fixed seeds where randomness is involved
#
# Running Tests:
# --------------
#   # Run all unit tests
#   pytest tests/unit -v
#
#   # Run specific test file
#   pytest tests/unit/test_gradient_flow.py -v
#
#   # Run tests excluding slow tests
#   pytest tests/unit -v -m "not slow"
#
#   # Run with coverage
#   pytest tests/unit --cov=. --cov-report=term-missing
#
#   # Run only GPU tests
#   pytest tests/unit -v -m "gpu"
#
#   # Run only CPU tests
#   pytest tests/unit -v -m "cpu"
#
# Test Modules:
# -------------
#   test_gradient_flow.py       - Gradient correctness and backpropagation
#                                 Verifies gradients flow through all layers,
#                                 no vanishing/exploding gradients, and
#                                 Gumbel-softmax straight-through works.
#
#   test_dense_baseline.py      - K=0 MLP equivalence verification
#                                 Confirms BFSNet with K=0 behaves like a
#                                 standard MLP with no branching.
#
#   test_sparse_dense_match.py  - Sparse vs dense output consistency
#                                 Compares soft_full and sparse execution
#                                 modes for output similarity.
#
#   test_checkpoint_roundtrip.py - Save/load correctness
#                                  Tests model state dict persistence,
#                                  optimizer state, and cross-device loading.
#
#   test_device_fallback.py     - Device selection logic (CPU/CUDA)
#                                 Verifies model runs on CPU, CUDA, and
#                                 handles device fallback correctly.
#
#   test_edge_cases.py          - Boundary conditions and edge cases
#                                 Tests unusual inputs, batch sizes,
#                                 dimension values, and special tensors.
#
#   test_numerical_stability.py - NaN/Inf detection and prevention
#                                 Ensures model outputs and gradients
#                                 remain finite under various conditions.
#
#   test_execution_modes.py     - Warmup → sparse transition testing
#                                 Verifies execution mode switching,
#                                 warmup behavior, and mode-specific output.
#
# Fixtures:
# ---------
#   Common fixtures are defined in conftest.py (parent directory).
#   Each test class can also define local fixtures as needed.
#
# Markers:
# --------
#   @pytest.mark.unit        - Unit test (default for this package)
#   @pytest.mark.slow        - Slow test (>1 second)
#   @pytest.mark.gpu         - Requires GPU
#   @pytest.mark.cpu         - CPU-only test
#   @pytest.mark.parametrize - Parametrized test
#
# ============================================================================

"""
BFSNet unit tests package.

This package provides comprehensive unit testing for the BFSNet model,
covering gradient flow, numerical stability, device handling, checkpointing,
and execution mode transitions.
"""

__version__ = "1.0.0"
__author__ = "BFSNet Team"

__all__ = [
    "test_gradient_flow",
    "test_dense_baseline",
    "test_sparse_dense_match",
    "test_checkpoint_roundtrip",
    "test_device_fallback",
    "test_edge_cases",
    "test_numerical_stability",
    "test_execution_modes",
]

# Package-level constants for test configuration
DEFAULT_INPUT_DIM = 784
DEFAULT_OUTPUT_DIM = 10
DEFAULT_HIDDEN_DIM = 64
DEFAULT_MAX_DEPTH = 2
DEFAULT_BATCH_SIZE = 16
DEFAULT_SEED = 42