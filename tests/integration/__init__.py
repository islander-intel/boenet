# ============================================================================
# BFSNet Integration Tests Package
# ============================================================================
# Integration tests for BFSNet end-to-end workflows.
#
# These tests verify that different components work together correctly:
#   - Full training pipeline execution
#   - Configuration file loading and validation
#   - CSV output generation and format correctness
#   - Model save/load across training runs
#
# Integration tests are typically slower than unit tests and may require
# more resources (disk I/O, potentially GPU).
#
# Run integration tests:
#   pytest tests/integration -v
#   pytest tests/integration -v -m "not slow"
#   pytest tests/integration --cov=. --cov-report=term-missing
#
# Test Files:
#   - test_pipeline_smoke.py  : End-to-end smoke tests for training pipeline
#   - test_csv_output.py      : CSV output validation and format checking
#   - test_config_loading.py  : Configuration file parsing and validation
#
# Prerequisites:
#   - BFSNet model (bfs_model.py)
#   - Training matrix script (bfs_training_matrix.py)
#   - Configuration files (configs/*.yaml)
#   - Loss functions (losses/bfs_losses.py)
#   - Utility modules (utils/*.py)
#
# Environment Variables:
#   - BFSNET_TEST_MODE: Set to "1" to enable test mode optimizations
#   - BFSNET_DEVICE: Override device selection (cpu/cuda)
#
# ============================================================================

"""BFSNet integration tests package."""

__all__ = [
    "test_pipeline_smoke",
    "test_csv_output",
    "test_config_loading",
]

# Integration test configuration defaults
INTEGRATION_TEST_CONFIG = {
    # Reduced settings for faster integration tests
    "max_epochs": 3,
    "batch_size": 32,
    "hidden_dim": 32,
    "max_depth": 2,
    "num_repeats": 1,
    "warmup_epochs": 1,
    
    # K values to test (subset for speed)
    "k_values": [0, 2],
    
    # Timeout settings (seconds)
    "single_run_timeout": 60,
    "full_pipeline_timeout": 300,
    
    # Output validation
    "expected_csv_columns": [
        "k", "max_depth", "hidden_dim", "pooling",
        "lr", "batch_size", "weight_decay",
        "temp_schedule", "temp_fixed", "temp_anneal",
        "warmup_epochs", "repeat",
        "final_train_loss", "final_train_acc",
        "final_val_loss", "final_val_acc",
        "best_val_acc", "best_epoch",
        "total_time_sec"
    ],
}