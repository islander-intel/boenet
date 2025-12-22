# ============================================================================
# BFSNet Test Fixtures
# ============================================================================
# Shared pytest fixtures for the BFSNet test suite.
#
# Usage:
#   Fixtures are automatically available to all tests.
#   Use by adding fixture name as function argument:
#
#       def test_example(bfs_model_k2, sample_batch):
#           output = bfs_model_k2(sample_batch)
#           assert output.shape[0] == sample_batch.shape[0]
# ============================================================================

import os
import sys
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import pytest
import torch
import torch.nn as nn
import numpy as np

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# Configuration Constants
# ============================================================================

# Default dimensions for FashionMNIST
DEFAULT_INPUT_DIM = 784  # 28x28 flattened
DEFAULT_OUTPUT_DIM = 10  # 10 classes
DEFAULT_HIDDEN_DIM = 64
DEFAULT_MAX_DEPTH = 2
DEFAULT_BATCH_SIZE = 32

# Random seed for reproducibility
TEST_SEED = 42


# ============================================================================
# Setup and Teardown
# ============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "gpu: mark test as requiring GPU")
    config.addinivalue_line("markers", "unit: mark as unit test")
    config.addinivalue_line("markers", "integration: mark as integration test")
    config.addinivalue_line("markers", "smoke: mark as smoke test")


@pytest.fixture(scope="session", autouse=True)
def set_random_seeds():
    """Set random seeds for reproducibility."""
    torch.manual_seed(TEST_SEED)
    np.random.seed(TEST_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(TEST_SEED)
    yield


# ============================================================================
# Device Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def device() -> torch.device:
    """
    Get the appropriate PyTorch device.
    
    Returns:
        torch.device: CUDA device if available, else CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@pytest.fixture(scope="session")
def cpu_device() -> torch.device:
    """Get CPU device (for tests that must run on CPU)."""
    return torch.device("cpu")


@pytest.fixture(scope="session")
def cuda_available() -> bool:
    """Check if CUDA is available."""
    return torch.cuda.is_available()


# ============================================================================
# Model Fixtures
# ============================================================================

@pytest.fixture
def bfs_model_k0(device):
    """
    Create a dense baseline BFSNet model (K=0, no branching).
    
    This is equivalent to a standard MLP:
        input -> root_fc -> ReLU -> output_fc -> output
    """
    try:
        from bfs_model import BFSNet
    except ImportError:
        pytest.skip("bfs_model not available")
    
    model = BFSNet(
        input_dim=DEFAULT_INPUT_DIM,
        output_dim=DEFAULT_OUTPUT_DIM,
        hidden_dim=DEFAULT_HIDDEN_DIM,
        max_depth=DEFAULT_MAX_DEPTH,
        max_children=0,  # K=0: dense baseline
        pooling="learned"
    ).to(device)
    model.eval()
    return model


@pytest.fixture
def bfs_model_k2(device):
    """
    Create a BFSNet model with K=2 (2 children per node).
    """
    try:
        from bfs_model import BFSNet
    except ImportError:
        pytest.skip("bfs_model not available")
    
    model = BFSNet(
        input_dim=DEFAULT_INPUT_DIM,
        output_dim=DEFAULT_OUTPUT_DIM,
        hidden_dim=DEFAULT_HIDDEN_DIM,
        max_depth=DEFAULT_MAX_DEPTH,
        max_children=2,
        pooling="learned"
    ).to(device)
    model.eval()
    return model


@pytest.fixture
def bfs_model_k3(device):
    """
    Create a BFSNet model with K=3 (3 children per node).
    """
    try:
        from bfs_model import BFSNet
    except ImportError:
        pytest.skip("bfs_model not available")
    
    model = BFSNet(
        input_dim=DEFAULT_INPUT_DIM,
        output_dim=DEFAULT_OUTPUT_DIM,
        hidden_dim=DEFAULT_HIDDEN_DIM,
        max_depth=DEFAULT_MAX_DEPTH,
        max_children=3,
        pooling="learned"
    ).to(device)
    model.eval()
    return model


@pytest.fixture(params=[0, 1, 2, 3, 4, 5])
def bfs_model_all_k(request, device):
    """
    Parametrized fixture that creates BFSNet models for all K values.
    
    Use with @pytest.mark.parametrize or as regular fixture:
        def test_all_k(bfs_model_all_k):
            ...
    """
    try:
        from bfs_model import BFSNet
    except ImportError:
        pytest.skip("bfs_model not available")
    
    k_value = request.param
    model = BFSNet(
        input_dim=DEFAULT_INPUT_DIM,
        output_dim=DEFAULT_OUTPUT_DIM,
        hidden_dim=DEFAULT_HIDDEN_DIM,
        max_depth=DEFAULT_MAX_DEPTH,
        max_children=k_value,
        pooling="learned"
    ).to(device)
    model.eval()
    return model, k_value


@pytest.fixture(params=["learned", "sum", "mean"])
def bfs_model_all_pooling(request, device):
    """
    Parametrized fixture for all pooling modes.
    """
    try:
        from bfs_model import BFSNet
    except ImportError:
        pytest.skip("bfs_model not available")
    
    pooling = request.param
    model = BFSNet(
        input_dim=DEFAULT_INPUT_DIM,
        output_dim=DEFAULT_OUTPUT_DIM,
        hidden_dim=DEFAULT_HIDDEN_DIM,
        max_depth=DEFAULT_MAX_DEPTH,
        max_children=2,
        pooling=pooling
    ).to(device)
    model.eval()
    return model, pooling


@pytest.fixture
def random_model_config() -> Dict[str, Any]:
    """
    Generate a random but valid model configuration.
    """
    np.random.seed(TEST_SEED)
    return {
        "input_dim": DEFAULT_INPUT_DIM,
        "output_dim": DEFAULT_OUTPUT_DIM,
        "hidden_dim": np.random.choice([32, 64, 128, 256]),
        "max_depth": np.random.choice([1, 2, 3]),
        "max_children": np.random.choice([0, 1, 2, 3, 4, 5]),
        "pooling": np.random.choice(["learned", "sum", "mean"])
    }


# ============================================================================
# Data Fixtures
# ============================================================================

@pytest.fixture
def sample_batch(device) -> torch.Tensor:
    """
    Create a random batch of input data.
    
    Returns:
        torch.Tensor: Shape (batch_size, input_dim)
    """
    return torch.randn(DEFAULT_BATCH_SIZE, DEFAULT_INPUT_DIM, device=device)


@pytest.fixture
def sample_batch_small(device) -> torch.Tensor:
    """Create a small batch for quick tests."""
    return torch.randn(4, DEFAULT_INPUT_DIM, device=device)


@pytest.fixture
def sample_batch_large(device) -> torch.Tensor:
    """Create a large batch for stress tests."""
    return torch.randn(256, DEFAULT_INPUT_DIM, device=device)


@pytest.fixture
def sample_labels(device) -> torch.Tensor:
    """
    Create random labels for classification.
    
    Returns:
        torch.Tensor: Shape (batch_size,) with values in [0, output_dim)
    """
    return torch.randint(0, DEFAULT_OUTPUT_DIM, (DEFAULT_BATCH_SIZE,), device=device)


@pytest.fixture
def sample_batch_and_labels(sample_batch, sample_labels) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return both input batch and labels."""
    return sample_batch, sample_labels


@pytest.fixture
def sample_logits(device) -> torch.Tensor:
    """
    Create random logits for loss function tests.
    
    Returns:
        torch.Tensor: Shape (batch_size, output_dim)
    """
    return torch.randn(DEFAULT_BATCH_SIZE, DEFAULT_OUTPUT_DIM, device=device)


@pytest.fixture
def sample_branch_probs(device) -> torch.Tensor:
    """
    Create sample branch probabilities for gating tests.
    
    Returns:
        torch.Tensor: Shape (batch_size, num_nodes, K+1) summing to 1 along last dim
    """
    K = 3  # Example: 3 children + 1 for "stay at parent"
    num_nodes = 7  # Example tree with 7 nodes
    probs = torch.rand(DEFAULT_BATCH_SIZE, num_nodes, K + 1, device=device)
    probs = probs / probs.sum(dim=-1, keepdim=True)  # Normalize to sum to 1
    return probs


@pytest.fixture(scope="session")
def fashionmnist_data():
    """
    Load a small subset of FashionMNIST for integration tests.
    
    Returns:
        Tuple[DataLoader, DataLoader]: Train and test loaders with small batches
    """
    try:
        from torchvision import datasets, transforms
        from torch.utils.data import DataLoader, Subset
    except ImportError:
        pytest.skip("torchvision not available")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.Lambda(lambda x: x.view(-1))  # Flatten to 784
    ])
    
    # Use temp directory for data
    data_dir = tempfile.mkdtemp()
    
    try:
        train_dataset = datasets.FashionMNIST(
            root=data_dir,
            train=True,
            download=True,
            transform=transform
        )
        test_dataset = datasets.FashionMNIST(
            root=data_dir,
            train=False,
            download=True,
            transform=transform
        )
        
        # Use small subsets for speed
        train_subset = Subset(train_dataset, range(500))
        test_subset = Subset(test_dataset, range(100))
        
        train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_subset, batch_size=32, shuffle=False)
        
        yield train_loader, test_loader
        
    finally:
        # Cleanup
        shutil.rmtree(data_dir, ignore_errors=True)


@pytest.fixture
def fashionmnist_batch(fashionmnist_data, device):
    """Get a single batch from FashionMNIST."""
    train_loader, _ = fashionmnist_data
    for x, y in train_loader:
        return x.to(device), y.to(device)


# ============================================================================
# Config Fixtures
# ============================================================================

@pytest.fixture
def test_config() -> Dict[str, Any]:
    """
    Minimal test configuration matching test-config.yaml structure.
    """
    return {
        "sweep": {
            "k_values": [0, 2],
            "max_depths": [2],
            "hidden_dims": [64],
            "poolings": ["learned"],
            "temp_schedules": ["cosine"],
            "fixed_temps": [1.0],
            "cosine_anneal": ["1.4->1.0"],
            "lrs": [0.0025],
            "batch_sizes": [128],
            "weight_decays": [0.0],
            "warmup_epochs_list": [0, 2]
        },
        "training": {
            "epochs": 3,
            "repeats": 1,
            "seed0": 42
        },
        "inference": {
            "infer_samples": 100,
            "cpu_only": True,
            "infer_force_exec": "sparse"
        },
        "paths": {
            "save_root": "test_runs",
            "data_root": "./data",
            "train_script": "train_fmnist_bfs.py",
            "infer_script": "infer_fmnist_bfs.py"
        }
    }


@pytest.fixture
def temp_config_file(test_config, tmp_path) -> Path:
    """
    Create a temporary config file.
    
    Returns:
        Path: Path to the temporary config file
    """
    try:
        import yaml
    except ImportError:
        pytest.skip("PyYAML not available")
    
    config_path = tmp_path / "test_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(test_config, f)
    
    return config_path


@pytest.fixture
def experiment_config() -> Dict[str, Any]:
    """
    Full experiment configuration matching experiment-config.yaml structure.
    """
    return {
        "sweep": {
            "k_values": [0, 1, 2, 3, 4, 5],
            "max_depths": [2, 3],
            "hidden_dims": [128, 256],
            "poolings": ["learned", "mean"],
            "temp_schedules": ["cosine", "none"],
            "fixed_temps": [1.0, 1.2],
            "cosine_anneal": ["1.6->1.0", "1.4->0.8"],
            "lrs": [0.001, 0.0025, 0.005],
            "batch_sizes": [64, 128],
            "weight_decays": [0.0, 0.01],
            "warmup_epochs_list": [3, 5]
        },
        "training": {
            "epochs": 15,
            "repeats": 3,
            "seed0": 42
        },
        "inference": {
            "infer_samples": 1000,
            "cpu_only": True,
            "infer_force_exec": "sparse"
        },
        "paths": {
            "save_root": "runs",
            "data_root": "./data",
            "train_script": "train_fmnist_bfs.py",
            "infer_script": "infer_fmnist_bfs.py"
        }
    }


# ============================================================================
# Temporary File/Directory Fixtures
# ============================================================================

@pytest.fixture
def temp_dir(tmp_path) -> Path:
    """
    Provide a temporary directory that's cleaned up after the test.
    
    Returns:
        Path: Path to temporary directory
    """
    return tmp_path


@pytest.fixture
def temp_checkpoint(bfs_model_k2, tmp_path, device) -> Path:
    """
    Create a temporary model checkpoint.
    
    Returns:
        Path: Path to saved checkpoint
    """
    ckpt_path = tmp_path / "test_model.pt"
    torch.save({
        "model_state_dict": bfs_model_k2.state_dict(),
        "config": {
            "input_dim": DEFAULT_INPUT_DIM,
            "output_dim": DEFAULT_OUTPUT_DIM,
            "hidden_dim": DEFAULT_HIDDEN_DIM,
            "max_depth": DEFAULT_MAX_DEPTH,
            "max_children": 2,
            "pooling": "learned"
        }
    }, ckpt_path)
    return ckpt_path


# ============================================================================
# Loss Function Fixtures
# ============================================================================

@pytest.fixture
def cross_entropy_loss():
    """Standard cross-entropy loss."""
    return nn.CrossEntropyLoss()


@pytest.fixture
def sibling_balance_inputs(device) -> Dict[str, torch.Tensor]:
    """
    Create inputs for sibling balance loss testing.
    
    Returns dictionary with:
        - branch_probs: (B, N, K+1) branch probabilities
        - expected_shape: expected output shape
    """
    B, N, K_plus_1 = 16, 7, 4  # batch=16, nodes=7, K=3 children + 1
    
    # Create valid probability distributions
    probs = torch.rand(B, N, K_plus_1, device=device)
    probs = probs / probs.sum(dim=-1, keepdim=True)
    
    return {
        "branch_probs": probs,
        "batch_size": B,
        "num_nodes": N,
        "K_plus_1": K_plus_1
    }


# ============================================================================
# Gating Function Fixtures
# ============================================================================

@pytest.fixture
def gumbel_softmax_inputs(device) -> Dict[str, torch.Tensor]:
    """
    Create inputs for Gumbel-softmax testing.
    
    Returns dictionary with:
        - logits: unnormalized log probabilities
        - temperature: Gumbel-softmax temperature
    """
    B, N, K = 16, 5, 4  # batch=16, nodes=5, choices=4
    
    return {
        "logits": torch.randn(B, N, K, device=device),
        "temperature": 1.0,
        "shape": (B, N, K)
    }


# ============================================================================
# Sparse Utility Fixtures
# ============================================================================

@pytest.fixture
def sparse_index_inputs(device) -> Dict[str, torch.Tensor]:
    """
    Create inputs for sparse index operation testing.
    """
    B, N, D = 16, 10, 64  # batch=16, indices=10, features=64
    
    # Valid indices (no duplicates, in range)
    indices = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)
    values = torch.randn(B, N, D, device=device)
    target = torch.zeros(B, N * 2, D, device=device)  # Larger target
    
    return {
        "indices": indices,
        "values": values,
        "target": target,
        "dim": 1
    }


# ============================================================================
# Utility Functions for Tests
# ============================================================================

@pytest.fixture
def assert_tensors_equal():
    """
    Fixture providing tensor comparison function.
    """
    def _compare(t1: torch.Tensor, t2: torch.Tensor, rtol: float = 1e-5, atol: float = 1e-8):
        """Assert two tensors are approximately equal."""
        assert t1.shape == t2.shape, f"Shape mismatch: {t1.shape} vs {t2.shape}"
        assert torch.allclose(t1, t2, rtol=rtol, atol=atol), \
            f"Tensor values differ. Max diff: {(t1 - t2).abs().max().item()}"
    
    return _compare


@pytest.fixture
def assert_valid_probs():
    """
    Fixture providing probability validation function.
    """
    def _validate(probs: torch.Tensor, dim: int = -1):
        """Assert tensor represents valid probabilities."""
        assert (probs >= 0).all(), "Probabilities must be non-negative"
        assert (probs <= 1).all(), "Probabilities must be <= 1"
        sums = probs.sum(dim=dim)
        assert torch.allclose(sums, torch.ones_like(sums)), \
            f"Probabilities must sum to 1 along dim {dim}"
    
    return _validate


@pytest.fixture
def assert_no_nan_inf():
    """
    Fixture providing NaN/Inf check function.
    """
    def _check(tensor: torch.Tensor, name: str = "tensor"):
        """Assert tensor contains no NaN or Inf values."""
        assert not torch.isnan(tensor).any(), f"{name} contains NaN values"
        assert not torch.isinf(tensor).any(), f"{name} contains Inf values"
    
    return _check


# ============================================================================
# Gradient Testing Fixtures
# ============================================================================

@pytest.fixture
def gradient_check():
    """
    Fixture providing gradient checking function.
    """
    def _check_gradients(model: nn.Module, input_tensor: torch.Tensor):
        """Verify gradients flow correctly through model."""
        model.train()
        input_tensor.requires_grad_(True)
        
        output = model(input_tensor)
        loss = output.sum()
        loss.backward()
        
        # Check input gradients
        assert input_tensor.grad is not None, "No gradients on input"
        assert not torch.isnan(input_tensor.grad).any(), "NaN in input gradients"
        
        # Check parameter gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
        
        return True
    
    return _check_gradients


# ============================================================================
# Timing Fixtures
# ============================================================================

@pytest.fixture
def timer():
    """
    Fixture providing simple timing context manager.
    """
    import time
    
    class Timer:
        def __init__(self):
            self.elapsed = 0.0
        
        def __enter__(self):
            self.start = time.perf_counter()
            return self
        
        def __exit__(self, *args):
            self.elapsed = time.perf_counter() - self.start
    
    return Timer


# ============================================================================
# Skip Conditions
# ============================================================================

# Decorator for GPU-only tests
requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available"
)

# Decorator for tests requiring multiple GPUs
requires_multi_gpu = pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="Multiple GPUs not available"
)

# Make decorators available to tests
@pytest.fixture
def skip_decorators():
    """Provide skip decorators to tests."""
    return {
        "requires_cuda": requires_cuda,
        "requires_multi_gpu": requires_multi_gpu
    }