# ============================================================================
# BFSNet Device Fallback and Movement Tests
# ============================================================================
#
# Tests for device handling, CPU/GPU fallback, and model movement.
#
# These tests verify:
#   - Model works correctly on both CPU and GPU
#   - Model can be moved between devices
#   - Proper fallback when GPU is unavailable
#   - Float16 support on GPU
#   - RNG state preservation across device moves
#
# Important Note on Determinism:
# ------------------------------
#   The BFSNet model has an internal RNG (self._rng) for reproducible
#   stochastic branching. After moving a model between devices, the internal
#   RNG should be re-seeded using model.set_rng_seed() for determinism.
#
# Important Note on Cross-Device RNG:
# -----------------------------------
#   PyTorch does NOT guarantee that torch.Generator objects produce identical
#   random sequences across CPU and CUDA backends, even with the same seed.
#   This is a known limitation of PyTorch's random number generation.
#   Therefore, tests should NOT expect identical outputs when comparing
#   CPU vs GPU execution with the same seed - only that both produce valid,
#   correctly-shaped outputs.
#
# Run:
#   pytest tests/unit/test_device_fallback.py -v
#
# Updated: 2025-09-17 - Added model.set_rng_seed() calls, fixed device comparison
# Updated: 2025-09-18 - Fixed cross-device test to have realistic expectations
# ============================================================================

import pytest
import torch
import torch.nn as nn
import logging
from typing import Dict, Any, Optional

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

INPUT_DIM = 784
OUTPUT_DIM = 10
HIDDEN_DIM = 64
MAX_DEPTH = 2
BATCH_SIZE = 16


def create_model(k: int, device: torch.device = None) -> nn.Module:
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
    )
    
    if device is not None:
        model = model.to(device)
    
    return model


def set_all_seeds(seed: int, model: nn.Module = None) -> None:
    """
    Set all random seeds for full reproducibility.
    
    This sets both PyTorch's global RNG and the model's internal RNG.
    Both are required for deterministic behavior in BFSNet.
    
    Note: This ensures reproducibility on the SAME device. Cross-device
    reproducibility (CPU vs GPU) is NOT guaranteed by PyTorch.
    
    Parameters
    ----------
    seed : int
        Random seed to set.
    model : nn.Module, optional
        BFSNet model to seed. If provided and model has set_rng_seed(),
        the model's internal RNG will also be seeded.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    if model is not None and hasattr(model, 'set_rng_seed'):
        model.set_rng_seed(seed)


class TestDeviceBasics:
    """Basic device placement tests."""
    
    @pytest.mark.unit
    def test_model_on_cpu(self):
        """Test model works on CPU."""
        model = create_model(k=2, device=torch.device('cpu'))
        model.eval()
        
        x = torch.randn(BATCH_SIZE, INPUT_DIM)
        
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (BATCH_SIZE, OUTPUT_DIM)
        assert output.device.type == 'cpu'
        assert not torch.isnan(output).any()
    
    @pytest.mark.unit
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_model_on_gpu(self):
        """Test model works on GPU."""
        device = torch.device('cuda')
        model = create_model(k=2, device=device)
        model.eval()
        
        x = torch.randn(BATCH_SIZE, INPUT_DIM, device=device)
        
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (BATCH_SIZE, OUTPUT_DIM)
        assert output.device.type == 'cuda'
        assert not torch.isnan(output).any()
    
    @pytest.mark.unit
    def test_all_parameters_on_same_device(self, device):
        """Test all parameters are on the specified device."""
        model = create_model(k=2, device=device)
        
        for name, param in model.named_parameters():
            # Compare device types (not strict equality which fails cuda:0 vs cuda)
            assert param.device.type == device.type, \
                f"Parameter {name} on wrong device: {param.device} vs {device}"


class TestDeviceMovement:
    """Tests for moving models between devices."""
    
    @pytest.mark.unit
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cpu_to_gpu(self):
        """Test moving model from CPU to GPU."""
        model = create_model(k=2, device=torch.device('cpu'))
        
        # Move to GPU
        model = model.cuda()
        
        # Verify all parameters moved
        for name, param in model.named_parameters():
            assert param.device.type == 'cuda', f"{name} not on GPU"
        
        # Test inference works
        x = torch.randn(BATCH_SIZE, INPUT_DIM, device='cuda')
        with torch.no_grad():
            output = model(x)
        
        assert output.device.type == 'cuda'
    
    @pytest.mark.unit
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_to_cpu(self):
        """Test moving model from GPU to CPU."""
        model = create_model(k=2, device=torch.device('cuda'))
        
        # Move to CPU
        model = model.cpu()
        
        # Verify all parameters moved
        for name, param in model.named_parameters():
            assert param.device.type == 'cpu', f"{name} not on CPU"
        
        # Test inference works
        x = torch.randn(BATCH_SIZE, INPUT_DIM)
        with torch.no_grad():
            output = model(x)
        
        assert output.device.type == 'cpu'
    
    @pytest.mark.unit
    def test_model_works_after_device_move(self, device):
        """
        Test model produces valid output after device move.
        
        Note: After moving devices, the model's internal RNG should be
        re-seeded for deterministic behavior.
        """
        # Start on CPU
        model = create_model(k=2, device=torch.device('cpu'))
        model.eval()
        
        # Move to target device
        model = model.to(device)
        
        # Re-seed the model's internal RNG after device move
        set_all_seeds(42, model)
        
        x = torch.randn(BATCH_SIZE, INPUT_DIM, device=device)
        
        with torch.no_grad():
            output1 = model(x)
        
        # Should be deterministic with same seed
        set_all_seeds(42, model)
        with torch.no_grad():
            output2 = model(x)
        
        assert torch.allclose(output1, output2, atol=1e-6), \
            "Model not deterministic after device move"
        
        logger.info(f"Model works after move to {device}")


class TestFloat16Support:
    """Tests for float16 (half precision) support."""
    
    @pytest.mark.unit
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_float16_forward_pass(self):
        """Test forward pass works with float16."""
        device = torch.device('cuda')
        model = create_model(k=2, device=device)
        model = model.half()  # Convert to float16
        model.eval()
        
        x = torch.randn(BATCH_SIZE, INPUT_DIM, device=device, dtype=torch.float16)
        
        with torch.no_grad():
            output = model(x)
        
        assert output.dtype == torch.float16
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
        
        logger.info("Float16 forward pass works")
    
    @pytest.mark.unit
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_float16_training(self):
        """Test training works with float16."""
        device = torch.device('cuda')
        model = create_model(k=2, device=device)
        model = model.half()
        model.train()
        
        # Re-seed after converting to half
        set_all_seeds(42, model)
        
        x = torch.randn(BATCH_SIZE, INPUT_DIM, device=device, dtype=torch.float16)
        y = torch.randint(0, OUTPUT_DIM, (BATCH_SIZE,), device=device)
        
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        
        for _ in range(5):
            optimizer.zero_grad()
            output = model(x)
            loss = nn.CrossEntropyLoss()(output, y)
            loss.backward()
            optimizer.step()
        
        logger.info("Float16 training works")
    
    @pytest.mark.unit
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_mixed_precision_compatibility(self):
        """Test compatibility with automatic mixed precision."""
        device = torch.device('cuda')
        model = create_model(k=2, device=device)
        model.train()
        
        x = torch.randn(BATCH_SIZE, INPUT_DIM, device=device)
        y = torch.randint(0, OUTPUT_DIM, (BATCH_SIZE,), device=device)
        
        # Use updated API to avoid deprecation warning
        scaler = torch.amp.GradScaler('cuda')
        
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        
        for _ in range(3):
            optimizer.zero_grad()
            
            # Use updated API to avoid deprecation warning
            with torch.amp.autocast('cuda'):
                output = model(x)
                loss = nn.CrossEntropyLoss()(output, y)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        logger.info("Mixed precision training works")


class TestDeviceFallback:
    """Tests for device fallback behavior."""
    
    @pytest.mark.unit
    def test_cpu_fallback_when_gpu_unavailable(self):
        """Test model falls back to CPU gracefully."""
        # This test simulates what happens when GPU is requested but unavailable
        device = torch.device('cpu')  # Force CPU
        
        model = create_model(k=2, device=device)
        model.eval()
        
        x = torch.randn(BATCH_SIZE, INPUT_DIM, device=device)
        
        with torch.no_grad():
            output = model(x)
        
        assert output.device.type == 'cpu'
        assert not torch.isnan(output).any()
    
    @pytest.mark.unit
    def test_device_from_config(self, device):
        """Test creating model with device from config/fixture."""
        model = create_model(k=2, device=device)
        model.eval()
        
        x = torch.randn(BATCH_SIZE, INPUT_DIM, device=device)
        
        with torch.no_grad():
            output = model(x)
        
        # Compare device types
        assert output.device.type == device.type


class TestRNGStatePreservation:
    """Tests for RNG state handling across devices."""
    
    @pytest.mark.unit
    def test_rng_seed_works_on_cpu(self):
        """Test RNG seeding works on CPU."""
        model = create_model(k=2, device=torch.device('cpu'))
        model.eval()
        
        x = torch.randn(BATCH_SIZE, INPUT_DIM)
        
        set_all_seeds(42, model)
        with torch.no_grad():
            out1 = model(x)
        
        set_all_seeds(42, model)
        with torch.no_grad():
            out2 = model(x)
        
        assert torch.allclose(out1, out2, atol=1e-6)
    
    @pytest.mark.unit
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_rng_seed_works_on_gpu(self):
        """Test RNG seeding works on GPU."""
        device = torch.device('cuda')
        model = create_model(k=2, device=device)
        model.eval()
        
        x = torch.randn(BATCH_SIZE, INPUT_DIM, device=device)
        
        set_all_seeds(42, model)
        with torch.no_grad():
            out1 = model(x)
        
        set_all_seeds(42, model)
        with torch.no_grad():
            out2 = model(x)
        
        assert torch.allclose(out1, out2, atol=1e-6)
    
    @pytest.mark.unit
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_rng_consistency_across_devices(self):
        """
        Test that both CPU and GPU produce valid outputs with same weights.
        
        IMPORTANT: This test does NOT expect identical outputs between CPU and GPU.
        PyTorch does not guarantee cross-device RNG consistency - the internal
        random number generators on CPU and CUDA produce different sequences
        even with the same seed. This is a known PyTorch limitation.
        
        What this test DOES verify:
        1. Both devices produce outputs with correct shape
        2. Both outputs are valid (no NaN, no Inf)
        3. Weights are correctly shared via state_dict
        4. Each device is individually deterministic (same seed = same output)
        
        This simulates the fallback scenario where:
        - Config requests GPU but GPU is unavailable
        - Code falls back to CPU
        - Model should still work correctly on CPU
        """
        logger.info("Testing cross-device validity (not expecting identical outputs)")
        
        # Create model on CPU
        model_cpu = create_model(k=2, device=torch.device('cpu'))
        model_cpu.eval()
        
        # Create model on GPU with same weights
        model_gpu = create_model(k=2, device=torch.device('cuda'))
        model_gpu.load_state_dict(model_cpu.state_dict())
        model_gpu.eval()
        
        # Verify weights were copied correctly
        for (name_cpu, param_cpu), (name_gpu, param_gpu) in zip(
            model_cpu.named_parameters(), model_gpu.named_parameters()
        ):
            assert name_cpu == name_gpu, f"Parameter name mismatch: {name_cpu} vs {name_gpu}"
            assert torch.allclose(param_cpu, param_gpu.cpu(), atol=1e-6), \
                f"Weight mismatch for {name_cpu}"
        
        logger.info("Weights successfully shared between CPU and GPU models")
        
        # Same input data
        x_cpu = torch.randn(BATCH_SIZE, INPUT_DIM)
        x_gpu = x_cpu.cuda()
        
        # Run on CPU with seed
        set_all_seeds(42, model_cpu)
        with torch.no_grad():
            out_cpu = model_cpu(x_cpu)
        
        # Run on GPU with same seed
        set_all_seeds(42, model_gpu)
        with torch.no_grad():
            out_gpu = model_gpu(x_gpu)
        
        # Verify both outputs are VALID (not necessarily identical)
        # Shape check
        assert out_cpu.shape == (BATCH_SIZE, OUTPUT_DIM), \
            f"CPU output wrong shape: {out_cpu.shape}"
        assert out_gpu.shape == (BATCH_SIZE, OUTPUT_DIM), \
            f"GPU output wrong shape: {out_gpu.shape}"
        
        logger.info(f"CPU output shape: {out_cpu.shape}, GPU output shape: {out_gpu.shape}")
        
        # NaN/Inf check
        assert not torch.isnan(out_cpu).any(), "CPU output contains NaN"
        assert not torch.isnan(out_gpu).any(), "GPU output contains NaN"
        assert not torch.isinf(out_cpu).any(), "CPU output contains Inf"
        assert not torch.isinf(out_gpu).any(), "GPU output contains Inf"
        
        logger.info("Both CPU and GPU outputs are valid (no NaN/Inf)")
        
        # Verify each device is individually deterministic
        # CPU determinism
        set_all_seeds(42, model_cpu)
        with torch.no_grad():
            out_cpu_2 = model_cpu(x_cpu)
        assert torch.allclose(out_cpu, out_cpu_2, atol=1e-6), \
            "CPU model not deterministic with same seed"
        
        logger.info("CPU model is deterministic with same seed")
        
        # GPU determinism
        set_all_seeds(42, model_gpu)
        with torch.no_grad():
            out_gpu_2 = model_gpu(x_gpu)
        assert torch.allclose(out_gpu, out_gpu_2, atol=1e-6), \
            "GPU model not deterministic with same seed"
        
        logger.info("GPU model is deterministic with same seed")
        
        # Log the difference for informational purposes (not a failure condition)
        diff = (out_cpu - out_gpu.cpu()).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        
        logger.info(f"Cross-device output difference (expected due to RNG): "
                   f"max={max_diff:.6f}, mean={mean_diff:.6f}")
        logger.info("Note: Cross-device differences are expected - PyTorch does not "
                   "guarantee RNG consistency across CPU/CUDA backends")
        
        # Final summary
        logger.info("Cross-device test PASSED: Both devices produce valid, "
                   "deterministic outputs (identical outputs not expected)")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])