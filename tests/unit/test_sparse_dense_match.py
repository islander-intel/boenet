# ============================================================================
# BFSNet Sparse vs Dense Consistency Tests
# ============================================================================
#
# Tests to verify that sparse and dense execution modes produce consistent
# results (accounting for expected differences due to hard vs soft masking).
#
# Important Note on Determinism:
# ------------------------------
#   The BFSNet model has an internal RNG (self._rng) for reproducible
#   stochastic branching. This internal generator is NOT affected by
#   torch.manual_seed(). To ensure deterministic behavior:
#   
#   1. Call torch.manual_seed(seed) for PyTorch's global RNG
#   2. Call model.set_rng_seed(seed) for the model's internal RNG
#   
#   Both are required for full reproducibility.
#
# Run:
#   pytest tests/unit/test_sparse_dense_match.py -v
#
# Updated: 2025-09-17 - Added model.set_rng_seed() calls for determinism tests
# ============================================================================

import pytest
import torch
import torch.nn as nn
import logging
from typing import Dict, Any, Tuple, Optional, List

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

INPUT_DIM = 784
OUTPUT_DIM = 10
HIDDEN_DIM = 64
MAX_DEPTH = 2
BATCH_SIZE = 16


def create_model(k: int, device: torch.device) -> nn.Module:
    """Create a BFSNet model."""
    try:
        from bfs_model import BFSNet
    except ImportError:
        pytest.skip("bfs_model module not available")
    
    model = BFSNet(
        input_dim=INPUT_DIM,
        output_dim=OUTPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        max_depth=MAX_DEPTH,
        max_children=k,
        pooling="learned"
    ).to(device)
    
    return model


def set_execution_mode(model: nn.Module, mode: str) -> bool:
    """Set model execution mode if supported."""
    if hasattr(model, 'set_exec_mode'):
        model.set_exec_mode(mode)
        return True
    elif hasattr(model, 'exec_mode'):
        model.exec_mode = mode
        return True
    return False


def sync_random_state(seed: int = 42, model: nn.Module = None) -> int:
    """
    Synchronize random state for reproducibility.
    
    This sets both PyTorch's global RNG and the model's internal RNG.
    Both are required for deterministic behavior in BFSNet.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    if model is not None and hasattr(model, 'set_rng_seed'):
        model.set_rng_seed(seed)
        logger.debug(f"Set model internal RNG seed to {seed}")
    
    return seed


class TestSparseDenseBasic:
    """Basic tests for sparse vs dense execution."""
    
    @pytest.mark.unit
    @pytest.mark.parametrize("k_value", [1, 2, 3, 4, 5])
    def test_both_modes_produce_valid_output(self, device, k_value):
        """Test both sparse and soft_full modes produce valid output."""
        model = create_model(k=k_value, device=device)
        model.eval()
        
        x = torch.randn(BATCH_SIZE, INPUT_DIM, device=device)
        
        for mode in ['soft_full', 'sparse']:
            if not set_execution_mode(model, mode):
                pytest.skip("Model doesn't support exec_mode")
            
            with torch.no_grad():
                output = model(x)
            
            assert output.shape == (BATCH_SIZE, OUTPUT_DIM)
            assert not torch.isnan(output).any()
            assert not torch.isinf(output).any()
    
    @pytest.mark.unit
    def test_k0_modes_produce_identical_output(self, device):
        """Test that K=0 produces identical output in both modes."""
        model = create_model(k=0, device=device)
        model.eval()
        
        x = torch.randn(BATCH_SIZE, INPUT_DIM, device=device)
        
        outputs = {}
        for mode in ['soft_full', 'sparse']:
            if set_execution_mode(model, mode):
                with torch.no_grad():
                    outputs[mode] = model(x)
        
        if len(outputs) == 2:
            max_diff = (outputs['soft_full'] - outputs['sparse']).abs().max().item()
            assert max_diff < 1e-5, f"K=0 outputs differ: max_diff={max_diff}"


class TestDeterminism:
    """Tests for determinism and random state synchronization."""
    
    @pytest.mark.unit
    def test_deterministic_with_seed(self, device):
        """
        Test that both modes are deterministic with fixed seed.
        
        Note: Both torch.manual_seed() AND model.set_rng_seed() are required
        for deterministic behavior.
        """
        model = create_model(k=2, device=device)
        model.eval()
        
        x = torch.randn(BATCH_SIZE, INPUT_DIM, device=device)
        
        for mode in ['soft_full', 'sparse']:
            if not set_execution_mode(model, mode):
                continue
            
            # Run twice with same seed (using model.set_rng_seed)
            sync_random_state(42, model)
            with torch.no_grad():
                out1 = model(x)
            
            sync_random_state(42, model)
            with torch.no_grad():
                out2 = model(x)
            
            assert torch.allclose(out1, out2, atol=1e-6), \
                f"Mode {mode}: Output not deterministic with same seed"
        
        logger.info("Both modes deterministic with fixed seed")


class TestSparseDenseGradients:
    """Tests for gradient behavior in different modes."""
    
    @pytest.mark.unit
    def test_soft_mode_has_gradients(self, device):
        """Test that soft_full mode produces gradients."""
        model = create_model(k=2, device=device)
        model.train()
        
        if not set_execution_mode(model, 'soft_full'):
            pytest.skip("Model doesn't support exec_mode")
        
        x = torch.randn(BATCH_SIZE, INPUT_DIM, device=device, requires_grad=True)
        output = model(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None, "Input should have gradients in soft mode"
        params_with_grad = sum(1 for p in model.parameters() if p.grad is not None)
        assert params_with_grad > 0, "Model should have gradients in soft mode"


class TestWeightSharing:
    """Tests to verify weights are shared correctly between modes."""
    
    @pytest.mark.unit
    def test_same_weights_used_both_modes(self, device):
        """Test that the same weights are used in both modes."""
        model = create_model(k=2, device=device)
        model.eval()
        
        original_weights = {
            name: param.clone().detach() 
            for name, param in model.named_parameters()
        }
        
        x = torch.randn(BATCH_SIZE, INPUT_DIM, device=device)
        
        for mode in ['soft_full', 'sparse']:
            if set_execution_mode(model, mode):
                with torch.no_grad():
                    _ = model(x)
                
                for name, param in model.named_parameters():
                    assert torch.allclose(param.data, original_weights[name]), \
                        f"Weights changed in {mode} mode: {name}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])