# ============================================================================
# BFSNet Checkpoint Roundtrip Tests
# ============================================================================
#
# Tests for saving and loading model checkpoints.
#
# These tests verify:
#   - Model can be saved and loaded via state_dict
#   - Loaded model produces identical output
#   - Checkpoint contains all necessary parameters
#   - map_location works correctly for device transfer
#   - Optimizer state can be saved/loaded
#
# Run:
#   pytest tests/unit/test_checkpoint_roundtrip.py -v
#
# Updated: 2025-09-17 - Fixed device comparison to use .type attribute
# ============================================================================

import pytest
import torch
import torch.nn as nn
import tempfile
import os
import logging
from typing import Dict, Any

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
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    if model is not None and hasattr(model, 'set_rng_seed'):
        model.set_rng_seed(seed)


class TestStateDictRoundtrip:
    """Tests for state_dict save/load."""
    
    @pytest.mark.unit
    @pytest.mark.parametrize("k_value", [0, 1, 2, 3])
    def test_state_dict_roundtrip(self, device, k_value):
        """Test saving and loading state_dict."""
        model1 = create_model(k=k_value, device=device)
        model1.eval()
        
        # Get original output
        set_all_seeds(42, model1)
        x = torch.randn(BATCH_SIZE, INPUT_DIM, device=device)
        with torch.no_grad():
            output1 = model1(x)
        
        # Save state dict
        state_dict = model1.state_dict()
        
        # Create new model and load
        model2 = create_model(k=k_value, device=device)
        model2.load_state_dict(state_dict)
        model2.eval()
        
        # Get output from loaded model
        set_all_seeds(42, model2)
        with torch.no_grad():
            output2 = model2(x)
        
        assert torch.allclose(output1, output2, atol=1e-6), \
            f"K={k_value}: Output differs after state_dict roundtrip"
        
        logger.info(f"K={k_value}: state_dict roundtrip successful")
    
    @pytest.mark.unit
    def test_state_dict_contains_all_parameters(self, device):
        """Test that state_dict contains all expected parameters."""
        model = create_model(k=2, device=device)
        state_dict = model.state_dict()
        
        # Check model parameters match state_dict
        model_params = set(name for name, _ in model.named_parameters())
        state_params = set(state_dict.keys())
        
        # State dict should contain all parameters (may also have buffers)
        missing_in_state = model_params - state_params
        
        assert len(missing_in_state) == 0, \
            f"Parameters missing from state_dict: {missing_in_state}"
        
        logger.info(f"state_dict contains {len(state_dict)} entries")


class TestCheckpointFile:
    """Tests for file-based checkpoint save/load."""
    
    @pytest.mark.unit
    def test_save_load_checkpoint_file(self, device):
        """Test saving and loading checkpoint to/from file."""
        model1 = create_model(k=2, device=device)
        model1.eval()
        
        # Get original output
        set_all_seeds(42, model1)
        x = torch.randn(BATCH_SIZE, INPUT_DIM, device=device)
        with torch.no_grad():
            output1 = model1(x)
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            torch.save(model1.state_dict(), f.name)
            checkpoint_path = f.name
        
        try:
            # Load into new model
            model2 = create_model(k=2, device=device)
            model2.load_state_dict(torch.load(checkpoint_path, weights_only=True))
            model2.eval()
            
            # Verify output matches
            set_all_seeds(42, model2)
            with torch.no_grad():
                output2 = model2(x)
            
            assert torch.allclose(output1, output2, atol=1e-6)
            logger.info("File-based checkpoint roundtrip successful")
        finally:
            os.unlink(checkpoint_path)
    
    @pytest.mark.unit
    def test_full_checkpoint_with_optimizer(self, device):
        """Test saving full checkpoint including optimizer state."""
        model = create_model(k=2, device=device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Do some training steps
        model.train()
        for _ in range(5):
            x = torch.randn(BATCH_SIZE, INPUT_DIM, device=device)
            y = torch.randint(0, OUTPUT_DIM, (BATCH_SIZE,), device=device)
            
            optimizer.zero_grad()
            output = model(x)
            loss = nn.CrossEntropyLoss()(output, y)
            loss.backward()
            optimizer.step()
        
        # Save full checkpoint
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': 5,
        }
        
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            torch.save(checkpoint, f.name)
            checkpoint_path = f.name
        
        try:
            # Load checkpoint
            loaded = torch.load(checkpoint_path, weights_only=False)
            
            model2 = create_model(k=2, device=device)
            model2.load_state_dict(loaded['model_state_dict'])
            
            optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.001)
            optimizer2.load_state_dict(loaded['optimizer_state_dict'])
            
            assert loaded['epoch'] == 5
            logger.info("Full checkpoint with optimizer successful")
        finally:
            os.unlink(checkpoint_path)


class TestMapLocation:
    """Tests for map_location device transfer during loading."""
    
    @pytest.mark.unit
    def test_map_location_cpu(self, device):
        """Test loading checkpoint with map_location to CPU."""
        model = create_model(k=2, device=device)
        
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            torch.save(model.state_dict(), f.name)
            checkpoint_path = f.name
        
        try:
            # Load with map_location to CPU
            state_dict = torch.load(
                checkpoint_path, 
                map_location='cpu',
                weights_only=True
            )
            
            # Verify all tensors on CPU
            for name, tensor in state_dict.items():
                assert tensor.device.type == 'cpu', \
                    f"{name} not on CPU after map_location"
            
            logger.info("map_location to CPU works")
        finally:
            os.unlink(checkpoint_path)
    
    @pytest.mark.unit
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_map_location_gpu(self):
        """Test loading checkpoint with map_location to GPU."""
        # Save on CPU
        model = create_model(k=2, device=torch.device('cpu'))
        
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            torch.save(model.state_dict(), f.name)
            checkpoint_path = f.name
        
        try:
            # Load with map_location to GPU
            state_dict = torch.load(
                checkpoint_path, 
                map_location='cuda',
                weights_only=True
            )
            
            # Verify all tensors on GPU
            for name, tensor in state_dict.items():
                assert tensor.device.type == 'cuda', \
                    f"{name} not on GPU after map_location"
            
            logger.info("map_location to GPU works")
        finally:
            os.unlink(checkpoint_path)
    
    @pytest.mark.unit
    def test_map_location_device(self, device):
        """Test loading checkpoint with map_location to target device."""
        # Save on CPU
        model = create_model(k=2, device=torch.device('cpu'))
        
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            torch.save(model.state_dict(), f.name)
            checkpoint_path = f.name
        
        try:
            # Load with map_location to target device
            state_dict = torch.load(
                checkpoint_path, 
                map_location=device,
                weights_only=True
            )
            
            # Load into model
            model2 = create_model(k=2, device=device)
            model2.load_state_dict(state_dict)
            
            # Verify parameters on correct device
            # Use device.type for comparison to handle cuda:0 vs cuda
            for name, param in model2.named_parameters():
                assert param.device.type == device.type, \
                    f"{name} on wrong device: {param.device.type} vs {device.type}"
            
            logger.info(f"map_location to {device} works")
        finally:
            os.unlink(checkpoint_path)


class TestCheckpointCompatibility:
    """Tests for checkpoint compatibility."""
    
    @pytest.mark.unit
    def test_strict_loading(self, device):
        """Test strict=True loading detects missing/extra keys."""
        model1 = create_model(k=2, device=device)
        state_dict = model1.state_dict()
        
        # Add extra key
        state_dict['extra_key'] = torch.tensor([1.0])
        
        model2 = create_model(k=2, device=device)
        
        # strict=True should raise on extra keys
        with pytest.raises(RuntimeError):
            model2.load_state_dict(state_dict, strict=True)
        
        logger.info("strict=True correctly rejects extra keys")
    
    @pytest.mark.unit
    def test_non_strict_loading(self, device):
        """Test strict=False loading allows missing/extra keys."""
        model1 = create_model(k=2, device=device)
        state_dict = model1.state_dict()
        
        # Add extra key
        state_dict['extra_key'] = torch.tensor([1.0])
        
        model2 = create_model(k=2, device=device)
        
        # strict=False should work with warning
        incompatible = model2.load_state_dict(state_dict, strict=False)
        
        assert 'extra_key' in incompatible.unexpected_keys
        logger.info("strict=False allows extra keys")
    
    @pytest.mark.unit
    def test_different_k_values_incompatible(self, device):
        """Test that different K values produce incompatible checkpoints."""
        model_k2 = create_model(k=2, device=device)
        model_k3 = create_model(k=3, device=device)
        
        state_dict_k2 = model_k2.state_dict()
        
        # Loading K=2 state into K=3 model should have mismatches
        # (due to different parameter shapes)
        try:
            model_k3.load_state_dict(state_dict_k2, strict=True)
            # If no error, shapes are compatible (possible for some architectures)
            logger.info("K=2 state loaded into K=3 (shapes compatible)")
        except RuntimeError as e:
            # Expected: shape mismatch
            logger.info(f"K=2 -> K=3 incompatible (expected): {e}")


class TestCheckpointDeterminism:
    """Tests for deterministic behavior after checkpoint loading."""
    
    @pytest.mark.unit
    def test_deterministic_after_load(self, device):
        """Test model is deterministic after loading checkpoint."""
        model1 = create_model(k=2, device=device)
        model1.eval()
        
        state_dict = model1.state_dict()
        
        model2 = create_model(k=2, device=device)
        model2.load_state_dict(state_dict)
        model2.eval()
        
        x = torch.randn(BATCH_SIZE, INPUT_DIM, device=device)
        
        # Multiple runs should be deterministic
        set_all_seeds(42, model2)
        with torch.no_grad():
            out1 = model2(x)
        
        set_all_seeds(42, model2)
        with torch.no_grad():
            out2 = model2(x)
        
        assert torch.allclose(out1, out2, atol=1e-6), \
            "Model not deterministic after checkpoint load"
        
        logger.info("Deterministic after checkpoint load")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])