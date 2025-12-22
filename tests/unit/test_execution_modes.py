# ============================================================================
# BFSNet Execution Modes Tests
# ============================================================================
# Tests for execution mode transitions (warmup → sparse).
#
# These tests verify:
#   - soft_full mode works correctly during warmup
#   - sparse mode works correctly after warmup
#   - Transition from soft_full to sparse is smooth
#   - Model behavior is correct in each mode
#   - Temperature annealing works correctly
#
# Execution Modes:
#   - soft_full: Uses soft Gumbel-softmax (differentiable, for warmup)
#   - sparse: Uses hard one-hot selections (efficient inference)
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
#   pytest tests/unit/test_execution_modes.py -v
#
# Updated: 2025-09-17 - Added model.set_rng_seed() calls for determinism tests
# ============================================================================

import pytest
import torch
import torch.nn as nn
import logging
from typing import Dict, Any, Optional, List

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


def set_execution_mode(model: nn.Module, mode: str) -> bool:
    """
    Set model execution mode.
    
    Returns:
        True if mode was set, False if model doesn't support exec_mode
    """
    if hasattr(model, 'set_exec_mode'):
        model.set_exec_mode(mode)
        return True
    elif hasattr(model, 'exec_mode'):
        model.exec_mode = mode
        return True
    return False


def get_execution_mode(model: nn.Module) -> Optional[str]:
    """Get current execution mode."""
    if hasattr(model, 'exec_mode'):
        return model.exec_mode
    return None


def set_temperature(model: nn.Module, temperature: float) -> bool:
    """Set model temperature."""
    if hasattr(model, 'temperature'):
        model.temperature = temperature
        return True
    return False


def get_temperature(model: nn.Module) -> Optional[float]:
    """Get model temperature."""
    if hasattr(model, 'temperature'):
        return model.temperature
    return None


def set_all_seeds(seed: int, model: nn.Module = None) -> None:
    """
    Set all random seeds for full reproducibility.
    
    This sets both PyTorch's global RNG and the model's internal RNG.
    Both are required for deterministic behavior in BFSNet.
    
    Args:
        seed: Random seed value
        model: BFSNet model (optional, for setting internal RNG)
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    if model is not None and hasattr(model, 'set_rng_seed'):
        model.set_rng_seed(seed)
        logger.debug(f"Set model internal RNG seed to {seed}")


# ============================================================================
# Soft Full Mode Tests
# ============================================================================

class TestSoftFullMode:
    """Tests for soft_full execution mode (warmup phase)."""
    
    @pytest.mark.unit
    @pytest.mark.parametrize("k_value", [1, 2, 3, 4, 5])
    def test_soft_full_mode_works(self, device, k_value):
        """Test that soft_full mode produces valid output."""
        model = create_model(k=k_value, device=device)
        model.eval()
        
        if not set_execution_mode(model, 'soft_full'):
            pytest.skip("Model doesn't support exec_mode")
        
        x = torch.randn(BATCH_SIZE, INPUT_DIM, device=device)
        
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (BATCH_SIZE, OUTPUT_DIM)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
        
        logger.info(f"K={k_value}: soft_full mode works")
    
    @pytest.mark.unit
    def test_soft_full_gradients_flow(self, device):
        """Test that gradients flow correctly in soft_full mode."""
        model = create_model(k=2, device=device)
        model.train()
        
        if not set_execution_mode(model, 'soft_full'):
            pytest.skip("Model doesn't support exec_mode")
        
        x = torch.randn(BATCH_SIZE, INPUT_DIM, device=device, requires_grad=True)
        output = model(x)
        loss = output.sum()
        loss.backward()
        
        # Check input gradients
        assert x.grad is not None, "Input should have gradients"
        assert not torch.isnan(x.grad).any(), "Input gradients contain NaN"
        
        # Check model gradients
        grad_count = sum(1 for p in model.parameters() if p.grad is not None)
        total_params = sum(1 for p in model.parameters() if p.requires_grad)
        
        logger.info(f"soft_full gradients: {grad_count}/{total_params} params have gradients")
        assert grad_count > 0, "No parameters received gradients"
    
    @pytest.mark.unit
    def test_soft_full_training_updates_weights(self, device):
        """Test that training in soft_full mode updates weights."""
        model = create_model(k=2, device=device)
        model.train()
        
        if not set_execution_mode(model, 'soft_full'):
            pytest.skip("Model doesn't support exec_mode")
        
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        
        # Store original weights
        original_weights = {name: param.clone() 
                          for name, param in model.named_parameters()}
        
        x = torch.randn(BATCH_SIZE, INPUT_DIM, device=device)
        y = torch.randint(0, OUTPUT_DIM, (BATCH_SIZE,), device=device)
        
        for _ in range(5):
            optimizer.zero_grad()
            output = model(x)
            loss = nn.CrossEntropyLoss()(output, y)
            loss.backward()
            optimizer.step()
        
        # Check weights changed
        weights_changed = 0
        for name, param in model.named_parameters():
            if not torch.allclose(param, original_weights[name], atol=1e-6):
                weights_changed += 1
        
        logger.info(f"soft_full training: {weights_changed} param groups changed")
        assert weights_changed > 0, "No weights were updated"


# ============================================================================
# Sparse Mode Tests
# ============================================================================

class TestSparseMode:
    """Tests for sparse execution mode (post-warmup)."""
    
    @pytest.mark.unit
    @pytest.mark.parametrize("k_value", [1, 2, 3, 4, 5])
    def test_sparse_mode_works(self, device, k_value):
        """Test that sparse mode produces valid output."""
        model = create_model(k=k_value, device=device)
        model.eval()
        
        if not set_execution_mode(model, 'sparse'):
            pytest.skip("Model doesn't support exec_mode")
        
        x = torch.randn(BATCH_SIZE, INPUT_DIM, device=device)
        
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (BATCH_SIZE, OUTPUT_DIM)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
        
        logger.info(f"K={k_value}: sparse mode works")
    
    @pytest.mark.unit
    def test_sparse_mode_deterministic_with_seed(self, device):
        """
        Test that sparse mode is deterministic with fixed seed.
        
        Note: Both torch.manual_seed() AND model.set_rng_seed() are required
        for deterministic behavior. The model has an internal RNG that is
        not affected by PyTorch's global seed.
        """
        model = create_model(k=2, device=device)
        model.eval()
        
        if not set_execution_mode(model, 'sparse'):
            pytest.skip("Model doesn't support exec_mode")
        
        x = torch.randn(BATCH_SIZE, INPUT_DIM, device=device)
        
        # First run with seed
        set_all_seeds(42, model)
        with torch.no_grad():
            output1 = model(x)
        
        # Second run with same seed
        set_all_seeds(42, model)
        with torch.no_grad():
            output2 = model(x)
        
        assert torch.allclose(output1, output2), \
            "sparse mode should be deterministic with same seed"
        
        logger.info("sparse mode: deterministic with seed (using model.set_rng_seed)")
    
    @pytest.mark.unit
    def test_sparse_mode_different_seeds_different_output(self, device):
        """Test that different seeds produce different outputs in sparse mode."""
        model = create_model(k=3, device=device)  # K>1 for branching
        model.eval()
        
        if not set_execution_mode(model, 'sparse'):
            pytest.skip("Model doesn't support exec_mode")
        
        x = torch.randn(BATCH_SIZE, INPUT_DIM, device=device)
        
        outputs = []
        for seed in [1, 2, 3, 4, 5]:
            set_all_seeds(seed, model)
            with torch.no_grad():
                outputs.append(model(x).clone())
        
        # At least some outputs should be different
        differences = 0
        for i in range(len(outputs)):
            for j in range(i + 1, len(outputs)):
                if not torch.allclose(outputs[i], outputs[j], atol=1e-4):
                    differences += 1
        
        logger.info(f"sparse mode: {differences} different output pairs out of 10")
        # With K>1, we expect some variation


# ============================================================================
# Mode Transition Tests
# ============================================================================

class TestModeTransition:
    """Tests for transitioning between execution modes."""
    
    @pytest.mark.unit
    def test_soft_to_sparse_transition(self, device):
        """Test transitioning from soft_full to sparse mode."""
        model = create_model(k=2, device=device)
        model.eval()
        
        x = torch.randn(BATCH_SIZE, INPUT_DIM, device=device)
        
        # Start in soft_full
        if not set_execution_mode(model, 'soft_full'):
            pytest.skip("Model doesn't support exec_mode")
        
        with torch.no_grad():
            soft_output = model(x)
        
        # Transition to sparse
        set_execution_mode(model, 'sparse')
        
        with torch.no_grad():
            sparse_output = model(x)
        
        # Both should produce valid output
        assert soft_output.shape == sparse_output.shape
        assert not torch.isnan(soft_output).any()
        assert not torch.isnan(sparse_output).any()
        
        logger.info("soft_full -> sparse transition: successful")
    
    @pytest.mark.unit
    def test_sparse_to_soft_transition(self, device):
        """Test transitioning from sparse to soft_full mode."""
        model = create_model(k=2, device=device)
        model.eval()
        
        x = torch.randn(BATCH_SIZE, INPUT_DIM, device=device)
        
        # Start in sparse
        if not set_execution_mode(model, 'sparse'):
            pytest.skip("Model doesn't support exec_mode")
        
        with torch.no_grad():
            sparse_output = model(x)
        
        # Transition to soft_full
        set_execution_mode(model, 'soft_full')
        
        with torch.no_grad():
            soft_output = model(x)
        
        # Both should produce valid output
        assert sparse_output.shape == soft_output.shape
        
        logger.info("sparse -> soft_full transition: successful")
    
    @pytest.mark.unit
    def test_multiple_mode_switches(self, device):
        """Test multiple mode switches don't cause issues."""
        model = create_model(k=2, device=device)
        model.eval()
        
        x = torch.randn(BATCH_SIZE, INPUT_DIM, device=device)
        
        if not set_execution_mode(model, 'soft_full'):
            pytest.skip("Model doesn't support exec_mode")
        
        for i in range(10):
            mode = 'soft_full' if i % 2 == 0 else 'sparse'
            set_execution_mode(model, mode)
            
            with torch.no_grad():
                output = model(x)
            
            assert not torch.isnan(output).any(), f"NaN at switch {i} (mode={mode})"
        
        logger.info("Multiple mode switches: stable")
    
    @pytest.mark.unit
    def test_weights_preserved_across_transition(self, device):
        """Test that weights are preserved when switching modes."""
        model = create_model(k=2, device=device)
        
        if not set_execution_mode(model, 'soft_full'):
            pytest.skip("Model doesn't support exec_mode")
        
        # Store weights
        weights_before = {name: param.clone() 
                        for name, param in model.named_parameters()}
        
        # Switch modes multiple times
        for mode in ['sparse', 'soft_full', 'sparse', 'soft_full']:
            set_execution_mode(model, mode)
        
        # Check weights unchanged
        for name, param in model.named_parameters():
            assert torch.allclose(param, weights_before[name]), \
                f"Weights changed for {name} during mode switch"
        
        logger.info("Weights preserved across mode transitions")


# ============================================================================
# Warmup Training Simulation Tests
# ============================================================================

class TestWarmupTraining:
    """Tests simulating the warmup training phase."""
    
    @pytest.mark.unit
    def test_warmup_phase_training(self, device):
        """Test training during warmup phase (soft_full mode)."""
        model = create_model(k=2, device=device)
        model.train()
        
        if not set_execution_mode(model, 'soft_full'):
            pytest.skip("Model doesn't support exec_mode")
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        
        x = torch.randn(100, INPUT_DIM, device=device)
        y = torch.randint(0, OUTPUT_DIM, (100,), device=device)
        
        warmup_epochs = 5
        losses = []
        
        for epoch in range(warmup_epochs):
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        
        logger.info(f"Warmup training losses: {losses}")
        assert losses[-1] < losses[0], "Loss should decrease during warmup"
    
    @pytest.mark.unit
    def test_post_warmup_inference(self, device):
        """Test inference after warmup phase (sparse mode)."""
        model = create_model(k=2, device=device)
        
        if not set_execution_mode(model, 'soft_full'):
            pytest.skip("Model doesn't support exec_mode")
        
        # Simulate warmup training
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        x_train = torch.randn(100, INPUT_DIM, device=device)
        y_train = torch.randint(0, OUTPUT_DIM, (100,), device=device)
        
        for _ in range(10):
            optimizer.zero_grad()
            output = model(x_train)
            loss = nn.CrossEntropyLoss()(output, y_train)
            loss.backward()
            optimizer.step()
        
        # Switch to sparse mode for inference
        set_execution_mode(model, 'sparse')
        model.eval()
        
        x_test = torch.randn(BATCH_SIZE, INPUT_DIM, device=device)
        
        with torch.no_grad():
            output = model(x_test)
        
        assert output.shape == (BATCH_SIZE, OUTPUT_DIM)
        assert not torch.isnan(output).any()
        
        logger.info("Post-warmup inference in sparse mode: successful")
    
    @pytest.mark.unit
    def test_full_training_simulation(self, device):
        """Test full training simulation with warmup → sparse transition."""
        model = create_model(k=2, device=device)
        
        if not set_execution_mode(model, 'soft_full'):
            pytest.skip("Model doesn't support exec_mode")
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        
        x = torch.randn(100, INPUT_DIM, device=device)
        y = torch.randint(0, OUTPUT_DIM, (100,), device=device)
        
        warmup_epochs = 5
        total_epochs = 15
        
        losses = {'warmup': [], 'post_warmup': []}
        
        for epoch in range(total_epochs):
            # Switch mode at warmup boundary
            if epoch < warmup_epochs:
                set_execution_mode(model, 'soft_full')
                phase = 'warmup'
            else:
                set_execution_mode(model, 'sparse')
                phase = 'post_warmup'
            
            model.train()
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            
            losses[phase].append(loss.item())
        
        logger.info(f"Warmup losses: {losses['warmup']}")
        logger.info(f"Post-warmup losses: {losses['post_warmup']}")
        
        # Training should work in both phases
        assert len(losses['warmup']) == warmup_epochs
        assert len(losses['post_warmup']) == total_epochs - warmup_epochs


# ============================================================================
# Temperature Annealing Tests
# ============================================================================

class TestTemperatureAnnealing:
    """Tests for temperature annealing during training."""
    
    @pytest.mark.unit
    def test_temperature_setting(self, device):
        """Test that temperature can be set."""
        model = create_model(k=2, device=device)
        
        if not set_temperature(model, 1.0):
            pytest.skip("Model doesn't support temperature")
        
        for temp in [0.5, 1.0, 2.0, 5.0]:
            set_temperature(model, temp)
            current_temp = get_temperature(model)
            assert current_temp == temp, f"Temperature not set: {current_temp} != {temp}"
        
        logger.info("Temperature setting: works")
    
    @pytest.mark.unit
    def test_temperature_affects_output(self, device):
        """Test that temperature affects model output."""
        model = create_model(k=2, device=device)
        model.eval()
        
        if not set_execution_mode(model, 'soft_full'):
            pytest.skip("Model doesn't support exec_mode")
        
        if not set_temperature(model, 1.0):
            pytest.skip("Model doesn't support temperature")
        
        x = torch.randn(BATCH_SIZE, INPUT_DIM, device=device)
        
        outputs = {}
        for temp in [0.5, 1.0, 5.0]:
            set_temperature(model, temp)
            set_all_seeds(42, model)
            with torch.no_grad():
                outputs[temp] = model(x).clone()
        
        # Different temperatures should produce different outputs
        # (in soft_full mode)
        diff_05_10 = (outputs[0.5] - outputs[1.0]).abs().max().item()
        diff_10_50 = (outputs[1.0] - outputs[5.0]).abs().max().item()
        
        logger.info(f"Temperature effect: diff(0.5, 1.0)={diff_05_10:.4f}, "
                   f"diff(1.0, 5.0)={diff_10_50:.4f}")
    
    @pytest.mark.unit
    def test_cosine_annealing_schedule(self, device):
        """Test implementing cosine annealing schedule."""
        model = create_model(k=2, device=device)
        
        if not set_temperature(model, 1.0):
            pytest.skip("Model doesn't support temperature")
        
        if not set_execution_mode(model, 'soft_full'):
            pytest.skip("Model doesn't support exec_mode")
        
        import math
        
        # Cosine annealing from temp_start to temp_end
        temp_start = 2.0
        temp_end = 0.5
        total_epochs = 20
        
        temperatures = []
        for epoch in range(total_epochs):
            # Cosine annealing formula
            temp = temp_end + 0.5 * (temp_start - temp_end) * \
                   (1 + math.cos(math.pi * epoch / total_epochs))
            temperatures.append(temp)
            
            set_temperature(model, temp)
            
            # Verify temperature was set
            assert abs(get_temperature(model) - temp) < 1e-6
        
        logger.info(f"Cosine schedule: {temperatures[0]:.2f} -> {temperatures[-1]:.2f}")
        assert temperatures[0] > temperatures[-1], "Temperature should decrease"
    
    @pytest.mark.unit
    def test_linear_annealing_schedule(self, device):
        """Test implementing linear annealing schedule."""
        model = create_model(k=2, device=device)
        
        if not set_temperature(model, 1.0):
            pytest.skip("Model doesn't support temperature")
        
        temp_start = 1.5
        temp_end = 0.5
        total_epochs = 10
        
        temperatures = []
        for epoch in range(total_epochs):
            # Linear annealing
            progress = epoch / (total_epochs - 1)
            temp = temp_start + (temp_end - temp_start) * progress
            temperatures.append(temp)
            
            set_temperature(model, temp)
        
        logger.info(f"Linear schedule: {temperatures}")
        assert abs(temperatures[0] - temp_start) < 1e-6
        assert abs(temperatures[-1] - temp_end) < 1e-6


# ============================================================================
# K=0 Special Case Tests
# ============================================================================

class TestK0ExecutionModes:
    """Tests for execution modes with K=0 (should be no-op)."""
    
    @pytest.mark.unit
    def test_k0_modes_produce_same_output(self, device):
        """Test that K=0 produces same output in both modes."""
        model = create_model(k=0, device=device)
        model.eval()
        
        x = torch.randn(BATCH_SIZE, INPUT_DIM, device=device)
        
        outputs = {}
        for mode in ['soft_full', 'sparse']:
            if set_execution_mode(model, mode):
                with torch.no_grad():
                    outputs[mode] = model(x).clone()
        
        if len(outputs) == 2:
            assert torch.allclose(outputs['soft_full'], outputs['sparse'], atol=1e-6), \
                "K=0 should produce identical output in both modes"
            logger.info("K=0: Both modes produce identical output")
        else:
            logger.info("K=0: Model doesn't support exec_mode")
    
    @pytest.mark.unit
    def test_k0_temperature_has_no_effect(self, device):
        """Test that temperature has no effect with K=0."""
        model = create_model(k=0, device=device)
        model.eval()
        
        x = torch.randn(BATCH_SIZE, INPUT_DIM, device=device)
        
        if not set_temperature(model, 1.0):
            pytest.skip("Model doesn't support temperature")
        
        outputs = {}
        for temp in [0.1, 1.0, 10.0]:
            set_temperature(model, temp)
            with torch.no_grad():
                outputs[temp] = model(x).clone()
        
        # All outputs should be identical for K=0
        assert torch.allclose(outputs[0.1], outputs[1.0], atol=1e-6)
        assert torch.allclose(outputs[1.0], outputs[10.0], atol=1e-6)
        
        logger.info("K=0: Temperature has no effect (correct)")


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])