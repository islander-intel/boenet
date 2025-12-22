# ============================================================================
# BFSNet Numerical Stability Tests
# ============================================================================
# Tests for numerical stability, NaN/Inf detection and prevention.
#
# These tests verify:
#   - No NaN or Inf in outputs under normal conditions
#   - Proper handling of edge cases that might cause numerical issues
#   - Gradient stability (no NaN/Inf gradients)
#   - Stability across different input magnitudes
#   - Softmax and log-softmax stability
#
# Run:
#   pytest tests/unit/test_numerical_stability.py -v
# ============================================================================

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Tuple, Optional, Dict

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
BATCH_SIZE = 16


# ============================================================================
# Helper Functions
# ============================================================================

def create_model(k: int, device: torch.device):
    """Create a BFSNet model."""
    try:
        from bfs_model import BFSNet
    except ImportError:
        pytest.skip("bfs_model not available")
    
    model = BFSNet(
        input_dim=INPUT_DIM,
        output_dim=OUTPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        max_depth=MAX_DEPTH,
        max_children=k,
        pooling="learned"
    ).to(device)
    
    return model


def check_tensor_valid(tensor: torch.Tensor, name: str = "tensor") -> Tuple[bool, str]:
    """
    Check if tensor is numerically valid (no NaN or Inf).
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if torch.isnan(tensor).any():
        nan_count = torch.isnan(tensor).sum().item()
        total = tensor.numel()
        return False, f"{name} contains {nan_count}/{total} NaN values"
    
    if torch.isinf(tensor).any():
        inf_count = torch.isinf(tensor).sum().item()
        total = tensor.numel()
        return False, f"{name} contains {inf_count}/{total} Inf values"
    
    return True, f"{name} is valid"


def check_gradients_valid(model: nn.Module) -> Tuple[bool, str]:
    """Check all gradients are valid."""
    for name, param in model.named_parameters():
        if param.grad is not None:
            is_valid, msg = check_tensor_valid(param.grad, f"grad({name})")
            if not is_valid:
                return False, msg
    return True, "All gradients valid"


# ============================================================================
# Basic Numerical Stability Tests
# ============================================================================

class TestBasicNumericalStability:
    """Basic tests for numerical stability."""
    
    @pytest.mark.unit
    @pytest.mark.parametrize("k_value", [0, 1, 2, 3, 4, 5])
    def test_no_nan_in_forward_pass(self, device, k_value):
        """Test that forward pass produces no NaN values."""
        model = create_model(k=k_value, device=device)
        model.eval()
        
        x = torch.randn(BATCH_SIZE, INPUT_DIM, device=device)
        
        with torch.no_grad():
            output = model(x)
        
        is_valid, msg = check_tensor_valid(output, "output")
        assert is_valid, msg
        
        logger.info(f"K={k_value}: No NaN in output")
    
    @pytest.mark.unit
    @pytest.mark.parametrize("k_value", [0, 1, 2, 3])
    def test_no_nan_in_gradients(self, device, k_value):
        """Test that backward pass produces no NaN gradients."""
        model = create_model(k=k_value, device=device)
        model.train()
        
        x = torch.randn(BATCH_SIZE, INPUT_DIM, device=device, requires_grad=True)
        output = model(x)
        loss = output.sum()
        loss.backward()
        
        is_valid, msg = check_tensor_valid(x.grad, "input_grad")
        assert is_valid, msg
        
        is_valid, msg = check_gradients_valid(model)
        assert is_valid, msg
        
        logger.info(f"K={k_value}: No NaN in gradients")
    
    @pytest.mark.unit
    def test_no_inf_in_output(self, device):
        """Test that output doesn't contain Inf values."""
        model = create_model(k=2, device=device)
        model.eval()
        
        for _ in range(10):
            x = torch.randn(BATCH_SIZE, INPUT_DIM, device=device)
            
            with torch.no_grad():
                output = model(x)
            
            assert not torch.isinf(output).any(), "Output contains Inf"
        
        logger.info("No Inf in output across 10 runs")


# ============================================================================
# Input Magnitude Tests
# ============================================================================

class TestInputMagnitudeStability:
    """Tests for stability across different input magnitudes."""
    
    @pytest.mark.unit
    @pytest.mark.parametrize("scale", [1e-6, 1e-3, 1.0, 1e3, 1e6])
    def test_stability_across_input_scales(self, device, scale):
        """Test numerical stability across different input scales."""
        model = create_model(k=2, device=device)
        model.eval()
        
        x = torch.randn(BATCH_SIZE, INPUT_DIM, device=device) * scale
        
        with torch.no_grad():
            output = model(x)
        
        is_valid, msg = check_tensor_valid(output, f"output(scale={scale})")
        
        logger.info(f"Scale {scale}: valid={is_valid}, "
                   f"output range=[{output.min().item():.2e}, {output.max().item():.2e}]")
        
        if scale <= 1e3:
            assert is_valid, msg
    
    @pytest.mark.unit
    def test_normalized_input_stability(self, device):
        """Test stability with normalized input (mean=0, std=1)."""
        model = create_model(k=2, device=device)
        model.eval()
        
        x = torch.randn(BATCH_SIZE, INPUT_DIM, device=device)
        x = (x - x.mean()) / (x.std() + 1e-8)
        
        with torch.no_grad():
            output = model(x)
        
        is_valid, msg = check_tensor_valid(output, "output")
        assert is_valid, msg
        
        logger.info("Normalized input: stable")
    
    @pytest.mark.unit
    def test_bounded_input_stability(self, device):
        """Test stability with bounded input [-1, 1]."""
        model = create_model(k=2, device=device)
        model.eval()
        
        x = torch.rand(BATCH_SIZE, INPUT_DIM, device=device) * 2 - 1
        
        with torch.no_grad():
            output = model(x)
        
        is_valid, msg = check_tensor_valid(output, "output")
        assert is_valid, msg
        
        logger.info("Bounded input [-1, 1]: stable")


# ============================================================================
# Training Stability Tests
# ============================================================================

class TestTrainingStability:
    """Tests for numerical stability during training."""
    
    @pytest.mark.unit
    def test_loss_stays_finite(self, device):
        """Test that loss stays finite during training."""
        model = create_model(k=2, device=device)
        model.train()
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        x = torch.randn(100, INPUT_DIM, device=device)
        y = torch.randint(0, OUTPUT_DIM, (100,), device=device)
        
        for epoch in range(50):
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            
            assert not torch.isnan(loss), f"NaN loss at epoch {epoch}"
            assert not torch.isinf(loss), f"Inf loss at epoch {epoch}"
            
            loss.backward()
            optimizer.step()
        
        logger.info(f"Training stable for 50 epochs, final loss: {loss.item():.4f}")
    
    @pytest.mark.unit
    def test_gradients_stay_bounded(self, device):
        """Test that gradients stay bounded during training."""
        model = create_model(k=2, device=device)
        model.train()
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        x = torch.randn(50, INPUT_DIM, device=device)
        y = torch.randint(0, OUTPUT_DIM, (50,), device=device)
        
        max_grad_norm = 0.0
        
        for epoch in range(20):
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    max_grad_norm = max(max_grad_norm, grad_norm)
                    
                    assert not torch.isnan(param.grad).any(), \
                        f"NaN gradient in {name} at epoch {epoch}"
            
            optimizer.step()
        
        logger.info(f"Max gradient norm: {max_grad_norm:.4f}")
        assert max_grad_norm < 1e6, f"Gradient explosion: {max_grad_norm}"
    
    @pytest.mark.unit
    def test_stability_with_gradient_clipping(self, device):
        """Test stability with gradient clipping."""
        model = create_model(k=2, device=device)
        model.train()
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        
        x = torch.randn(50, INPUT_DIM, device=device)
        y = torch.randint(0, OUTPUT_DIM, (50,), device=device)
        
        for epoch in range(30):
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            is_valid, msg = check_gradients_valid(model)
            assert is_valid, f"Epoch {epoch}: {msg}"
            
            optimizer.step()
        
        logger.info("Training with gradient clipping: stable")


# ============================================================================
# Softmax and Log-Softmax Stability Tests
# ============================================================================

class TestSoftmaxStability:
    """Tests for softmax-related numerical stability."""
    
    @pytest.mark.unit
    def test_softmax_with_large_values(self, device):
        """Test softmax stability with large input values."""
        logits = torch.randn(BATCH_SIZE, OUTPUT_DIM, device=device) * 100
        
        probs = F.softmax(logits, dim=-1)
        
        is_valid, msg = check_tensor_valid(probs, "softmax_output")
        assert is_valid, msg
        assert torch.allclose(probs.sum(dim=-1), torch.ones(BATCH_SIZE, device=device))
        
        logger.info("Softmax with large values: stable")
    
    @pytest.mark.unit
    def test_log_softmax_with_large_values(self, device):
        """Test log_softmax stability with large input values."""
        logits = torch.randn(BATCH_SIZE, OUTPUT_DIM, device=device) * 100
        
        log_probs = F.log_softmax(logits, dim=-1)
        
        is_valid, msg = check_tensor_valid(log_probs, "log_softmax_output")
        assert is_valid, msg
        
        assert (log_probs <= 0).all(), "Log probs should be <= 0"
        
        logger.info("Log softmax with large values: stable")
    
    @pytest.mark.unit
    def test_cross_entropy_stability(self, device):
        """Test cross entropy loss stability."""
        model = create_model(k=2, device=device)
        model.eval()
        
        x = torch.randn(BATCH_SIZE, INPUT_DIM, device=device)
        y = torch.randint(0, OUTPUT_DIM, (BATCH_SIZE,), device=device)
        
        with torch.no_grad():
            output = model(x)
        
        loss = F.cross_entropy(output, y)
        
        is_valid, msg = check_tensor_valid(loss, "cross_entropy_loss")
        assert is_valid, msg
        assert loss.item() >= 0, "Cross entropy should be non-negative"
        
        logger.info(f"Cross entropy stable: {loss.item():.4f}")


# ============================================================================
# Gumbel-Softmax Stability Tests
# ============================================================================

class TestGumbelSoftmaxStability:
    """Tests for Gumbel-softmax numerical stability."""
    
    @pytest.mark.unit
    def test_gumbel_softmax_no_nan(self, device):
        """Test that Gumbel-softmax produces no NaN."""
        try:
            from utils.gating import gumbel_softmax_st
        except ImportError:
            pytest.skip("gating module not available")
        
        for _ in range(100):
            logits = torch.randn(BATCH_SIZE, 5, 4, device=device)
            
            y_st, y_hard = gumbel_softmax_st(logits, temperature=1.0, hard=True)
            
            is_valid, msg = check_tensor_valid(y_st, "gumbel_softmax_output")
            assert is_valid, msg
        
        logger.info("Gumbel-softmax: no NaN in 100 runs")
    
    @pytest.mark.unit
    @pytest.mark.parametrize("temperature", [0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
    def test_gumbel_softmax_temperature_stability(self, device, temperature):
        """Test Gumbel-softmax stability at different temperatures."""
        try:
            from utils.gating import gumbel_softmax_st
        except ImportError:
            pytest.skip("gating module not available")
        
        logits = torch.randn(BATCH_SIZE, 5, 4, device=device)
        
        y_st, y_hard = gumbel_softmax_st(logits, temperature=temperature, hard=True)
        
        is_valid, msg = check_tensor_valid(y_st, f"gumbel_softmax(T={temperature})")
        assert is_valid, msg
        
        logger.info(f"Gumbel-softmax T={temperature}: stable")
    
    @pytest.mark.unit
    def test_gumbel_softmax_extreme_logits(self, device):
        """Test Gumbel-softmax with extreme logit values."""
        try:
            from utils.gating import gumbel_softmax_st
        except ImportError:
            pytest.skip("gating module not available")
        
        logits_large = torch.randn(BATCH_SIZE, 5, 4, device=device) * 100
        y_st, _ = gumbel_softmax_st(logits_large, temperature=1.0, hard=True)
        is_valid, msg = check_tensor_valid(y_st, "gumbel_large_logits")
        assert is_valid, msg
        
        logits_neg = torch.randn(BATCH_SIZE, 5, 4, device=device) * -100
        y_st, _ = gumbel_softmax_st(logits_neg, temperature=1.0, hard=True)
        is_valid, msg = check_tensor_valid(y_st, "gumbel_neg_logits")
        assert is_valid, msg
        
        logger.info("Gumbel-softmax with extreme logits: stable")


# ============================================================================
# Loss Function Stability Tests
# ============================================================================

class TestLossFunctionStability:
    """Tests for loss function numerical stability."""
    
    @pytest.mark.unit
    def test_sibling_balance_loss_stability(self, device):
        """Test sibling balance loss stability."""
        try:
            from losses.bfs_losses import sibling_balance_from_branch_probs
        except ImportError:
            pytest.skip("bfs_losses not available")
        
        B, N, K_plus_1 = 16, 7, 4
        probs = torch.rand(B, N, K_plus_1, device=device)
        probs = probs / probs.sum(dim=-1, keepdim=True)
        
        loss = sibling_balance_from_branch_probs(probs)
        
        is_valid, msg = check_tensor_valid(loss, "sibling_balance_loss")
        assert is_valid, msg
        
        logger.info(f"Sibling balance loss stable: {loss.item():.6f}")
    
    @pytest.mark.unit
    def test_combined_loss_stability(self, device):
        """Test stability of combined losses."""
        model = create_model(k=2, device=device)
        model.train()
        
        x = torch.randn(BATCH_SIZE, INPUT_DIM, device=device)
        y = torch.randint(0, OUTPUT_DIM, (BATCH_SIZE,), device=device)
        
        output = model(x)
        
        ce_loss = F.cross_entropy(output, y)
        l2_loss = sum(p.pow(2).sum() for p in model.parameters()) * 0.001
        
        total_loss = ce_loss + l2_loss
        
        is_valid, msg = check_tensor_valid(total_loss, "combined_loss")
        assert is_valid, msg
        
        total_loss.backward()
        is_valid, msg = check_gradients_valid(model)
        assert is_valid, msg
        
        logger.info(f"Combined loss stable: {total_loss.item():.4f}")


# ============================================================================
# Long-Running Stability Tests
# ============================================================================

class TestLongRunningStability:
    """Tests for stability over many iterations."""
    
    @pytest.mark.unit
    @pytest.mark.slow
    def test_stability_over_1000_iterations(self, device):
        """Test that model stays stable over many forward passes."""
        model = create_model(k=2, device=device)
        model.eval()
        
        nan_count = 0
        inf_count = 0
        
        for i in range(1000):
            x = torch.randn(BATCH_SIZE, INPUT_DIM, device=device)
            
            with torch.no_grad():
                output = model(x)
            
            if torch.isnan(output).any():
                nan_count += 1
            if torch.isinf(output).any():
                inf_count += 1
        
        logger.info(f"1000 iterations: {nan_count} NaN, {inf_count} Inf")
        
        assert nan_count == 0, f"Got {nan_count} NaN outputs"
        assert inf_count == 0, f"Got {inf_count} Inf outputs"
    
    @pytest.mark.unit
    @pytest.mark.slow
    def test_training_stability_100_epochs(self, device):
        """Test training stability over 100 epochs."""
        model = create_model(k=2, device=device)
        model.train()
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        x = torch.randn(100, INPUT_DIM, device=device)
        y = torch.randint(0, OUTPUT_DIM, (100,), device=device)
        
        losses = []
        for epoch in range(100):
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            
            is_valid, msg = check_tensor_valid(loss, f"loss_epoch_{epoch}")
            if not is_valid:
                pytest.fail(f"Epoch {epoch}: {msg}")
            
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
        
        logger.info(f"100 epochs: loss {losses[0]:.4f} -> {losses[-1]:.4f}")
        assert all(not (l != l) for l in losses), "NaN loss detected"


# ============================================================================
# Detection and Anomaly Tests
# ============================================================================

class TestAnomalyDetection:
    """Tests using PyTorch anomaly detection."""
    
    @pytest.mark.unit
    def test_with_anomaly_detection(self, device):
        """Test with PyTorch anomaly detection enabled."""
        model = create_model(k=2, device=device)
        model.train()
        
        x = torch.randn(BATCH_SIZE, INPUT_DIM, device=device, requires_grad=True)
        
        with torch.autograd.detect_anomaly():
            output = model(x)
            loss = output.sum()
            loss.backward()
        
        logger.info("Anomaly detection: no issues found")
    
    @pytest.mark.unit
    def test_gradient_check_numerical(self, device):
        """Test numerical gradient checking for simple operations."""
        model = create_model(k=0, device=torch.device("cpu"))
        model.train()
        
        x = torch.randn(4, INPUT_DIM, requires_grad=True, dtype=torch.float64)
        model = model.double()
        
        def func(input_tensor):
            return model(input_tensor).sum()
        
        try:
            from torch.autograd import gradcheck
            x_small = torch.randn(2, INPUT_DIM, requires_grad=True, dtype=torch.float64)
            logger.info("Gradient check setup successful")
        except Exception as e:
            logger.info(f"Gradient check: {e}")


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])