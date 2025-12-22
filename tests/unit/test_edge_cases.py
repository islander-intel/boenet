# ============================================================================
# BFSNet Edge Cases and Boundary Condition Tests
# ============================================================================
# Tests for edge cases and boundary conditions.
#
# These tests verify:
#   - Behavior with minimum/maximum parameter values
#   - Handling of unusual input shapes and sizes
#   - Empty batch handling
#   - Single sample batches
#   - Very large/small values
#   - Boundary K values and depths
#
# Run:
#   pytest tests/unit/test_edge_cases.py -v
# ============================================================================

import pytest
import torch
import torch.nn as nn
import logging
from typing import Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


# ============================================================================
# Test Constants
# ============================================================================

INPUT_DIM = 784
OUTPUT_DIM = 10
HIDDEN_DIM = 64
MAX_DEPTH = 2


# ============================================================================
# Helper Functions
# ============================================================================

def create_model(
    k: int = 2,
    input_dim: int = INPUT_DIM,
    output_dim: int = OUTPUT_DIM,
    hidden_dim: int = HIDDEN_DIM,
    max_depth: int = MAX_DEPTH,
    device: torch.device = None
):
    """Create a BFSNet model with specified parameters."""
    try:
        from bfs_model import BFSNet
    except ImportError:
        pytest.skip("bfs_model not available")
    
    if device is None:
        device = torch.device("cpu")
    
    model = BFSNet(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        max_depth=max_depth,
        max_children=k,
        pooling="learned"
    ).to(device)
    
    return model


# ============================================================================
# Batch Size Edge Cases
# ============================================================================

class TestBatchSizeEdgeCases:
    """Tests for edge cases related to batch size."""
    
    @pytest.mark.unit
    def test_batch_size_1(self, device):
        """Test model works with batch size of 1."""
        model = create_model(k=2, device=device)
        model.eval()
        
        x = torch.randn(1, INPUT_DIM, device=device)
        
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (1, OUTPUT_DIM)
        assert not torch.isnan(output).any()
        
        logger.info("Batch size 1: OK")
    
    @pytest.mark.unit
    def test_batch_size_very_large(self, device):
        """Test model works with large batch size."""
        model = create_model(k=2, device=device)
        model.eval()
        
        batch_size = 512
        x = torch.randn(batch_size, INPUT_DIM, device=device)
        
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (batch_size, OUTPUT_DIM)
        assert not torch.isnan(output).any()
        
        logger.info(f"Batch size {batch_size}: OK")
    
    @pytest.mark.unit
    def test_batch_size_prime_number(self, device):
        """Test model works with prime number batch size."""
        model = create_model(k=2, device=device)
        model.eval()
        
        for batch_size in [7, 13, 31, 97]:
            x = torch.randn(batch_size, INPUT_DIM, device=device)
            
            with torch.no_grad():
                output = model(x)
            
            assert output.shape == (batch_size, OUTPUT_DIM)
        
        logger.info("Prime batch sizes: OK")
    
    @pytest.mark.unit
    def test_zero_batch_size_raises_or_handles(self, device):
        """Test behavior with zero batch size."""
        model = create_model(k=2, device=device)
        model.eval()
        
        x = torch.randn(0, INPUT_DIM, device=device)
        
        try:
            with torch.no_grad():
                output = model(x)
            assert output.shape[0] == 0
            logger.info("Zero batch size: Handled gracefully")
        except (RuntimeError, IndexError):
            logger.info("Zero batch size: Raises error (expected)")


# ============================================================================
# K Value Edge Cases
# ============================================================================

class TestKValueEdgeCases:
    """Tests for edge cases related to K values."""
    
    @pytest.mark.unit
    def test_k_equals_0(self, device):
        """Test K=0 (dense baseline)."""
        model = create_model(k=0, device=device)
        model.eval()
        
        x = torch.randn(16, INPUT_DIM, device=device)
        
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (16, OUTPUT_DIM)
        logger.info("K=0: OK")
    
    @pytest.mark.unit
    def test_k_equals_1(self, device):
        """Test K=1 (minimal branching)."""
        model = create_model(k=1, device=device)
        model.eval()
        
        x = torch.randn(16, INPUT_DIM, device=device)
        
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (16, OUTPUT_DIM)
        logger.info("K=1: OK")
    
    @pytest.mark.unit
    def test_k_large(self, device):
        """Test large K value."""
        model = create_model(k=10, device=device)
        model.eval()
        
        x = torch.randn(16, INPUT_DIM, device=device)
        
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (16, OUTPUT_DIM)
        logger.info("K=10: OK")
    
    @pytest.mark.unit
    def test_negative_k_raises(self, device):
        """Test that negative K raises error."""
        try:
            from bfs_model import BFSNet
        except ImportError:
            pytest.skip("bfs_model not available")
        
        with pytest.raises((ValueError, RuntimeError, AssertionError)):
            BFSNet(
                input_dim=INPUT_DIM,
                output_dim=OUTPUT_DIM,
                hidden_dim=HIDDEN_DIM,
                max_depth=MAX_DEPTH,
                max_children=-1,
                pooling="learned"
            )
        
        logger.info("Negative K: Raises error (correct)")


# ============================================================================
# Depth Edge Cases
# ============================================================================

class TestDepthEdgeCases:
    """Tests for edge cases related to tree depth."""
    
    @pytest.mark.unit
    def test_depth_1(self, device):
        """Test minimum depth (depth=1)."""
        model = create_model(k=2, max_depth=1, device=device)
        model.eval()
        
        x = torch.randn(16, INPUT_DIM, device=device)
        
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (16, OUTPUT_DIM)
        logger.info("Depth=1: OK")
    
    @pytest.mark.unit
    def test_depth_large(self, device):
        """Test large depth value."""
        model = create_model(k=2, max_depth=5, device=device)
        model.eval()
        
        x = torch.randn(16, INPUT_DIM, device=device)
        
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (16, OUTPUT_DIM)
        logger.info("Depth=5: OK")
    
    @pytest.mark.unit
    def test_depth_0_raises_or_handles(self, device):
        """Test behavior with depth=0."""
        try:
            model = create_model(k=2, max_depth=0, device=device)
            x = torch.randn(16, INPUT_DIM, device=device)
            with torch.no_grad():
                output = model(x)
            logger.info("Depth=0: Handled gracefully")
        except (ValueError, RuntimeError, AssertionError):
            logger.info("Depth=0: Raises error (expected)")


# ============================================================================
# Dimension Edge Cases
# ============================================================================

class TestDimensionEdgeCases:
    """Tests for edge cases related to dimensions."""
    
    @pytest.mark.unit
    def test_input_dim_1(self, device):
        """Test with input dimension of 1."""
        model = create_model(k=2, input_dim=1, device=device)
        model.eval()
        
        x = torch.randn(16, 1, device=device)
        
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (16, OUTPUT_DIM)
        logger.info("Input dim=1: OK")
    
    @pytest.mark.unit
    def test_output_dim_1(self, device):
        """Test with output dimension of 1 (regression-like)."""
        model = create_model(k=2, output_dim=1, device=device)
        model.eval()
        
        x = torch.randn(16, INPUT_DIM, device=device)
        
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (16, 1)
        logger.info("Output dim=1: OK")
    
    @pytest.mark.unit
    def test_hidden_dim_small(self, device):
        """Test with very small hidden dimension."""
        model = create_model(k=2, hidden_dim=4, device=device)
        model.eval()
        
        x = torch.randn(16, INPUT_DIM, device=device)
        
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (16, OUTPUT_DIM)
        logger.info("Hidden dim=4: OK")
    
    @pytest.mark.unit
    def test_hidden_dim_large(self, device):
        """Test with large hidden dimension."""
        model = create_model(k=2, hidden_dim=512, device=device)
        model.eval()
        
        x = torch.randn(16, INPUT_DIM, device=device)
        
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (16, OUTPUT_DIM)
        logger.info("Hidden dim=512: OK")
    
    @pytest.mark.unit
    def test_wrong_input_dim_raises(self, device):
        """Test that wrong input dimension raises error."""
        model = create_model(k=2, input_dim=784, device=device)
        model.eval()
        
        x = torch.randn(16, 100, device=device)
        
        with pytest.raises(RuntimeError):
            model(x)
        
        logger.info("Wrong input dim: Raises error (correct)")


# ============================================================================
# Input Value Edge Cases
# ============================================================================

class TestInputValueEdgeCases:
    """Tests for edge cases related to input values."""
    
    @pytest.mark.unit
    def test_all_zeros_input(self, device):
        """Test with all-zeros input."""
        model = create_model(k=2, device=device)
        model.eval()
        
        x = torch.zeros(16, INPUT_DIM, device=device)
        
        with torch.no_grad():
            output = model(x)
        
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
        
        logger.info("All zeros input: OK")
    
    @pytest.mark.unit
    def test_all_ones_input(self, device):
        """Test with all-ones input."""
        model = create_model(k=2, device=device)
        model.eval()
        
        x = torch.ones(16, INPUT_DIM, device=device)
        
        with torch.no_grad():
            output = model(x)
        
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
        
        logger.info("All ones input: OK")
    
    @pytest.mark.unit
    def test_very_large_input_values(self, device):
        """Test with very large input values."""
        model = create_model(k=2, device=device)
        model.eval()
        
        x = torch.randn(16, INPUT_DIM, device=device) * 1000
        
        with torch.no_grad():
            output = model(x)
        
        has_nan = torch.isnan(output).any()
        has_inf = torch.isinf(output).any()
        
        logger.info(f"Large input: NaN={has_nan}, Inf={has_inf}")
    
    @pytest.mark.unit
    def test_very_small_input_values(self, device):
        """Test with very small input values."""
        model = create_model(k=2, device=device)
        model.eval()
        
        x = torch.randn(16, INPUT_DIM, device=device) * 1e-6
        
        with torch.no_grad():
            output = model(x)
        
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
        
        logger.info("Very small input: OK")
    
    @pytest.mark.unit
    def test_negative_input_values(self, device):
        """Test with all negative input values."""
        model = create_model(k=2, device=device)
        model.eval()
        
        x = -torch.abs(torch.randn(16, INPUT_DIM, device=device))
        
        with torch.no_grad():
            output = model(x)
        
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
        
        logger.info("Negative input: OK")
    
    @pytest.mark.unit
    def test_mixed_sign_input(self, device):
        """Test with mixed positive/negative input values."""
        model = create_model(k=2, device=device)
        model.eval()
        
        x = torch.randn(16, INPUT_DIM, device=device)
        
        with torch.no_grad():
            output = model(x)
        
        assert not torch.isnan(output).any()
        
        logger.info("Mixed sign input: OK")


# ============================================================================
# Special Tensor Edge Cases
# ============================================================================

class TestSpecialTensorEdgeCases:
    """Tests for special tensor types and values."""
    
    @pytest.mark.unit
    def test_input_with_inf_raises_or_handles(self, device):
        """Test behavior when input contains Inf."""
        model = create_model(k=2, device=device)
        model.eval()
        
        x = torch.randn(16, INPUT_DIM, device=device)
        x[0, 0] = float('inf')
        
        with torch.no_grad():
            output = model(x)
        
        logger.info(f"Inf input: output has_inf={torch.isinf(output).any()}, "
                   f"has_nan={torch.isnan(output).any()}")
    
    @pytest.mark.unit
    def test_input_with_nan_propagates(self, device):
        """Test that NaN in input propagates to output."""
        model = create_model(k=2, device=device)
        model.eval()
        
        x = torch.randn(16, INPUT_DIM, device=device)
        x[0, 0] = float('nan')
        
        with torch.no_grad():
            output = model(x)
        
        assert torch.isnan(output[0]).any(), "NaN should propagate"
        
        logger.info("NaN input: Propagates correctly")
    
    @pytest.mark.unit
    def test_contiguous_vs_noncontiguous(self, device):
        """Test with non-contiguous tensors."""
        model = create_model(k=2, device=device)
        model.eval()
        
        x_slice = torch.randn(32, INPUT_DIM, device=device)[::2]
        assert not x_slice.is_contiguous()
        
        with torch.no_grad():
            output = model(x_slice)
        
        assert output.shape == (16, OUTPUT_DIM)
        logger.info("Non-contiguous input: OK")


# ============================================================================
# Training Edge Cases
# ============================================================================

class TestTrainingEdgeCases:
    """Tests for training-related edge cases."""
    
    @pytest.mark.unit
    def test_single_sample_training(self, device):
        """Test training with single sample."""
        model = create_model(k=2, device=device)
        model.train()
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        x = torch.randn(1, INPUT_DIM, device=device)
        y = torch.randint(0, OUTPUT_DIM, (1,), device=device)
        
        optimizer.zero_grad()
        output = model(x)
        loss = nn.CrossEntropyLoss()(output, y)
        loss.backward()
        optimizer.step()
        
        logger.info("Single sample training: OK")
    
    @pytest.mark.unit
    def test_zero_learning_rate(self, device):
        """Test training with zero learning rate (weights shouldn't change)."""
        model = create_model(k=2, device=device)
        model.train()
        
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
        
        original_weights = {name: param.clone() 
                          for name, param in model.named_parameters()}
        
        x = torch.randn(16, INPUT_DIM, device=device)
        y = torch.randint(0, OUTPUT_DIM, (16,), device=device)
        
        for _ in range(5):
            optimizer.zero_grad()
            output = model(x)
            loss = nn.CrossEntropyLoss()(output, y)
            loss.backward()
            optimizer.step()
        
        for name, param in model.named_parameters():
            assert torch.allclose(param, original_weights[name])
        
        logger.info("Zero learning rate: Weights unchanged (correct)")
    
    @pytest.mark.unit
    def test_very_high_learning_rate(self, device):
        """Test training with very high learning rate (might diverge)."""
        model = create_model(k=0, device=device)
        model.train()
        
        optimizer = torch.optim.SGD(model.parameters(), lr=100.0)
        
        x = torch.randn(16, INPUT_DIM, device=device)
        y = torch.randint(0, OUTPUT_DIM, (16,), device=device)
        
        losses = []
        for _ in range(5):
            optimizer.zero_grad()
            output = model(x)
            loss = nn.CrossEntropyLoss()(output, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        
        logger.info(f"High learning rate losses: {losses}")


# ============================================================================
# Model State Edge Cases
# ============================================================================

class TestModelStateEdgeCases:
    """Tests for model state edge cases."""
    
    @pytest.mark.unit
    def test_eval_train_mode_switch(self, device):
        """Test switching between train and eval modes."""
        model = create_model(k=2, device=device)
        
        x = torch.randn(16, INPUT_DIM, device=device)
        
        for _ in range(5):
            model.train()
            with torch.no_grad():
                out_train = model(x)
            
            model.eval()
            with torch.no_grad():
                out_eval = model(x)
            
            assert out_train.shape == out_eval.shape
        
        logger.info("Train/eval mode switching: OK")
    
    @pytest.mark.unit
    def test_multiple_forward_passes(self, device):
        """Test multiple sequential forward passes."""
        model = create_model(k=2, device=device)
        model.eval()
        
        x = torch.randn(16, INPUT_DIM, device=device)
        
        for i in range(100):
            with torch.no_grad():
                output = model(x)
            
            if torch.isnan(output).any():
                pytest.fail(f"NaN appeared at iteration {i}")
        
        logger.info("100 forward passes: OK")
    
    @pytest.mark.unit
    def test_no_grad_context(self, device):
        """Test model in no_grad context."""
        model = create_model(k=2, device=device)
        model.eval()
        
        x = torch.randn(16, INPUT_DIM, device=device, requires_grad=True)
        
        with torch.no_grad():
            output = model(x)
        
        assert not output.requires_grad
        
        logger.info("no_grad context: OK")


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])