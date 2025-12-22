# ============================================================================
# BFSNet Gradient Flow Tests
# ============================================================================
#
# Tests for gradient correctness and backpropagation through the BFSNet model.
#
# These tests verify:
#   - Gradients flow correctly through all model components
#   - No vanishing or exploding gradients under normal conditions
#   - Gumbel-softmax straight-through estimator works correctly
#   - All trainable parameters receive gradients during backprop
#   - Gradients are finite (no NaN or Inf values)
#   - Gradient clipping works as expected
#   - Layer-wise gradient analysis
#
# Key Concepts:
# -------------
#   - Straight-Through Estimator (STE): Used with Gumbel-softmax to allow
#     gradients to flow through discrete sampling operations.
#   - Gradient Norm: Total L2 norm of gradients, useful for detecting
#     vanishing (too small) or exploding (too large) gradients.
#
# Important Note on K=0 Gradients:
# --------------------------------
#   When K=0, the child_fc layer is NEVER used in the forward pass because
#   no children are spawned. PyTorch correctly assigns no gradients to unused
#   parameters. This is expected behavior, not a bug. Tests for K=0 should
#   only check that ACTIVE layers (root_fc, output_fc) receive gradients.
#
# Run:
#   pytest tests/unit/test_gradient_flow.py -v
#   pytest tests/unit/test_gradient_flow.py -v -k "test_gradients_flow"
#
# Updated: 2025-09-17 - Fixed K=0 gradient expectations
# ============================================================================

import pytest
import torch
import torch.nn as nn
import logging
from typing import Dict, Any, Tuple, Optional, List

# Configure logging for test output
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Test Constants
# ============================================================================

INPUT_DIM = 784      # FashionMNIST flattened input dimension
OUTPUT_DIM = 10      # Number of classes
HIDDEN_DIM = 64      # Hidden layer dimension
MAX_DEPTH = 2        # Maximum tree depth
BATCH_SIZE = 16      # Default batch size for tests


# ============================================================================
# Helper Functions
# ============================================================================

def create_model(
    k: int,
    pooling: str = "learned",
    device: torch.device = None,
    hidden_dim: int = HIDDEN_DIM,
    max_depth: int = MAX_DEPTH
) -> nn.Module:
    """
    Create a BFSNet model for testing.
    
    Args:
        k: Maximum number of children per node (max_children)
        pooling: Pooling mode ('learned', 'sum', 'mean')
        device: Device to place model on
        hidden_dim: Hidden layer dimension
        max_depth: Maximum tree depth
        
    Returns:
        BFSNet model instance
    """
    try:
        from bfs_model import BFSNet
    except ImportError:
        pytest.skip("bfs_model module not available")
    
    if device is None:
        device = torch.device("cpu")
    
    logger.debug(f"Creating BFSNet: K={k}, pooling={pooling}, "
                f"hidden_dim={hidden_dim}, max_depth={max_depth}")
    
    model = BFSNet(
        input_dim=INPUT_DIM,
        output_dim=OUTPUT_DIM,
        hidden_dim=hidden_dim,
        max_depth=max_depth,
        max_children=k,
        pooling=pooling
    ).to(device)
    
    return model


def get_gradient_stats(model: nn.Module) -> Dict[str, Dict[str, float]]:
    """
    Collect gradient statistics for all parameters.
    
    Args:
        model: PyTorch model with computed gradients
        
    Returns:
        Dictionary mapping parameter name to gradient statistics:
        - mean: Mean gradient value
        - std: Standard deviation of gradients
        - min: Minimum gradient value
        - max: Maximum gradient value
        - norm: L2 norm of gradients
        - has_nan: Whether gradients contain NaN
        - has_inf: Whether gradients contain Inf
        - numel: Number of elements
    """
    stats = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad = param.grad.detach()
            stats[name] = {
                "mean": grad.mean().item(),
                "std": grad.std().item() if grad.numel() > 1 else 0.0,
                "min": grad.min().item(),
                "max": grad.max().item(),
                "norm": grad.norm().item(),
                "has_nan": torch.isnan(grad).any().item(),
                "has_inf": torch.isinf(grad).any().item(),
                "numel": grad.numel(),
            }
        else:
            stats[name] = None
    return stats


def is_k0_active_parameter(name: str) -> bool:
    """
    Check if a parameter should receive gradients in K=0 mode.
    
    When K=0, only these layers are used in the forward pass:
    - root_fc (input projection)
    - output_fc (classification head)
    - _pool_log_p (if pooling="learned")
    
    The following are NOT used when K=0:
    - child_fc (no children spawned)
    - branch_gate (no branching decisions)
    - prune_gate (no pruning)
    - stop_gate (no stopping decisions)
    - sibling_embeddings (no siblings)
    
    Args:
        name: Parameter name
        
    Returns:
        True if the parameter should have gradients in K=0 mode.
    """
    active_patterns = ['root', 'output', '_pool_log_p']
    return any(pattern in name.lower() for pattern in active_patterns)


def check_gradients_valid(model: nn.Module, k: int = None) -> Tuple[bool, str]:
    """
    Check that all gradients are valid (no NaN/Inf, appropriate params have grads).
    
    Args:
        model: PyTorch model with computed gradients
        k: The K value (max_children) of the model. If K=0, only checks
           active layer gradients.
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    for name, param in model.named_parameters():
        if param.requires_grad:
            # For K=0, only active parameters should have gradients
            if k == 0:
                if param.grad is None:
                    if is_k0_active_parameter(name):
                        return False, f"K=0 active parameter '{name}' has no gradient"
                    else:
                        # Expected: inactive parameters in K=0 mode have no gradient
                        continue
            else:
                # For K>0, all parameters should have gradients
                if param.grad is None:
                    return False, f"Parameter '{name}' has no gradient"
            
            # Check for NaN/Inf in gradients that exist
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    nan_count = torch.isnan(param.grad).sum().item()
                    return False, f"Parameter '{name}' has {nan_count} NaN gradient values"
                if torch.isinf(param.grad).any():
                    inf_count = torch.isinf(param.grad).sum().item()
                    return False, f"Parameter '{name}' has {inf_count} Inf gradient values"
    return True, "All gradients valid"


def compute_total_gradient_norm(model: nn.Module) -> float:
    """
    Compute total gradient norm across all parameters.
    
    Args:
        model: PyTorch model with computed gradients
        
    Returns:
        Total L2 norm of all gradients
    """
    total_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            total_norm += param.grad.norm().item() ** 2
    return total_norm ** 0.5


# ============================================================================
# Basic Gradient Flow Tests
# ============================================================================

class TestGradientFlow:
    """Tests for basic gradient flow through the model."""
    
    @pytest.mark.unit
    @pytest.mark.parametrize("k_value", [0, 1, 2, 3, 4, 5])
    def test_gradients_flow_all_k_values(self, device, k_value):
        """
        Test gradients flow correctly for all K values (0-5).
        
        Verifies that backward pass completes without error and appropriate
        trainable parameters receive valid gradients.
        
        Note: For K=0, only active layers (root_fc, output_fc) receive gradients.
        The child_fc layer is never used when K=0, so it correctly has no gradient.
        """
        logger.info(f"Testing gradient flow for K={k_value}")
        
        model = create_model(k=k_value, device=device)
        model.train()
        
        # Forward pass
        x = torch.randn(BATCH_SIZE, INPUT_DIM, device=device, requires_grad=True)
        output = model(x)
        
        logger.debug(f"Output shape: {output.shape}")
        
        # Backward pass
        loss = output.sum()
        loss.backward()
        
        # Check gradients with K-aware validation
        is_valid, msg = check_gradients_valid(model, k=k_value)
        assert is_valid, msg
        
        # Check input gradients
        assert x.grad is not None, "Input tensor should have gradients"
        assert not torch.isnan(x.grad).any(), "Input gradients contain NaN"
        assert not torch.isinf(x.grad).any(), "Input gradients contain Inf"
        
        # Log gradient statistics
        total_norm = compute_total_gradient_norm(model)
        logger.info(f"K={k_value}: Total gradient norm = {total_norm:.6f}")
        
        # Count parameters with/without gradients for logging
        params_with_grad = sum(1 for p in model.parameters() if p.grad is not None)
        total_params = sum(1 for p in model.parameters() if p.requires_grad)
        logger.info(f"K={k_value}: {params_with_grad}/{total_params} parameters have gradients")
    
    @pytest.mark.unit
    @pytest.mark.parametrize("pooling", ["learned", "sum", "mean"])
    def test_gradients_flow_all_pooling_modes(self, device, pooling):
        """
        Test gradients flow correctly for all pooling modes.
        
        Verifies that different pooling strategies don't break gradient flow.
        """
        logger.info(f"Testing gradient flow for pooling={pooling}")
        
        model = create_model(k=2, pooling=pooling, device=device)
        model.train()
        
        x = torch.randn(BATCH_SIZE, INPUT_DIM, device=device, requires_grad=True)
        output = model(x)
        loss = output.sum()
        loss.backward()
        
        is_valid, msg = check_gradients_valid(model, k=2)
        assert is_valid, msg
        
        logger.info(f"pooling={pooling}: Gradient flow verified")
    
    @pytest.mark.unit
    def test_all_parameters_receive_gradients(self, device):
        """
        Test that all trainable parameters receive gradients (for K>0).
        
        This ensures no parameters are accidentally detached from the
        computational graph when K>0.
        """
        logger.info("Testing all parameters receive gradients (K=2)")
        
        model = create_model(k=2, device=device)
        model.train()
        
        x = torch.randn(BATCH_SIZE, INPUT_DIM, device=device)
        output = model(x)
        loss = output.sum()
        loss.backward()
        
        params_without_grad = []
        params_with_grad = []
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                if param.grad is None:
                    params_without_grad.append(name)
                else:
                    params_with_grad.append(name)
        
        logger.debug(f"Parameters with gradients: {len(params_with_grad)}")
        logger.debug(f"Parameters without gradients: {len(params_without_grad)}")
        
        assert len(params_without_grad) == 0, \
            f"Parameters without gradients: {params_without_grad}"
        
        logger.info(f"All {len(params_with_grad)} trainable parameters received gradients")
    
    @pytest.mark.unit
    def test_gradient_accumulation(self, device):
        """
        Test gradient accumulation over multiple batches.
        
        Verifies that calling backward() multiple times without zero_grad()
        properly accumulates gradients.
        """
        logger.info("Testing gradient accumulation")
        
        model = create_model(k=2, device=device)
        model.train()
        
        # First batch
        x1 = torch.randn(BATCH_SIZE, INPUT_DIM, device=device)
        output1 = model(x1)
        loss1 = output1.sum()
        loss1.backward()
        
        # Store first gradients
        grad1 = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad1[name] = param.grad.clone()
        
        # Second batch (accumulate without zero_grad)
        x2 = torch.randn(BATCH_SIZE, INPUT_DIM, device=device)
        output2 = model(x2)
        loss2 = output2.sum()
        loss2.backward()
        
        # Check gradients accumulated (should be different from first)
        accumulated_differs = False
        for name, param in model.named_parameters():
            if param.grad is not None and name in grad1:
                # Check if gradients changed (accumulated)
                if not torch.allclose(param.grad, grad1[name], atol=1e-7):
                    accumulated_differs = True
                    break
        
        assert accumulated_differs, "Gradients should accumulate across batches"
        logger.info("Gradient accumulation verified")
    
    @pytest.mark.unit
    def test_zero_grad_clears_gradients(self, device):
        """
        Test that zero_grad() properly clears all gradients.
        """
        logger.info("Testing zero_grad clears gradients")
        
        model = create_model(k=2, device=device)
        model.train()
        
        # Compute gradients
        x = torch.randn(BATCH_SIZE, INPUT_DIM, device=device)
        output = model(x)
        loss = output.sum()
        loss.backward()
        
        # Verify gradients exist
        has_grad_before = any(p.grad is not None for p in model.parameters())
        assert has_grad_before, "Should have gradients before zero_grad"
        
        # Clear gradients
        model.zero_grad()
        
        # Verify gradients cleared
        for name, param in model.named_parameters():
            if param.grad is not None:
                assert (param.grad == 0).all(), f"{name} gradient not zeroed"
        
        logger.info("zero_grad correctly clears gradients")


# ============================================================================
# Gradient Magnitude Tests
# ============================================================================

class TestGradientMagnitude:
    """Tests for gradient magnitude and stability."""
    
    @pytest.mark.unit
    def test_no_vanishing_gradients(self, device):
        """
        Test that gradients don't vanish (become too small).
        
        Vanishing gradients prevent learning in early layers of deep networks.
        """
        logger.info("Testing for vanishing gradients")
        
        model = create_model(k=2, device=device)
        model.train()
        
        x = torch.randn(BATCH_SIZE, INPUT_DIM, device=device)
        output = model(x)
        loss = output.sum()
        loss.backward()
        
        # Check gradient norms aren't too small
        min_grad_norm_threshold = 1e-10
        vanishing_params = []
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                # Only flag if non-trivial parameter has tiny gradient
                if grad_norm < min_grad_norm_threshold and param.numel() > 1:
                    vanishing_params.append((name, grad_norm))
        
        # Allow some parameters to have small gradients, but not majority
        total_params = sum(1 for p in model.parameters() if p.grad is not None)
        vanishing_ratio = len(vanishing_params) / total_params if total_params > 0 else 0
        
        logger.debug(f"Vanishing params: {vanishing_params}")
        
        assert vanishing_ratio < 0.5, \
            f"Too many vanishing gradients ({len(vanishing_params)}/{total_params}): " \
            f"{vanishing_params[:5]}"
        
        logger.info(f"Vanishing gradient check passed: "
                   f"{len(vanishing_params)}/{total_params} params have tiny gradients")
    
    @pytest.mark.unit
    def test_no_exploding_gradients(self, device):
        """
        Test that gradients don't explode (become too large).
        
        Exploding gradients cause numerical instability and training failure.
        """
        logger.info("Testing for exploding gradients")
        
        model = create_model(k=2, device=device)
        model.train()
        
        x = torch.randn(BATCH_SIZE, INPUT_DIM, device=device)
        output = model(x)
        loss = output.sum()
        loss.backward()
        
        # Check gradient norms aren't too large
        max_grad_norm_threshold = 1e6
        exploding_params = []
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                if grad_norm > max_grad_norm_threshold:
                    exploding_params.append((name, grad_norm))
        
        assert len(exploding_params) == 0, \
            f"Exploding gradients detected: {exploding_params}"
        
        total_norm = compute_total_gradient_norm(model)
        logger.info(f"No exploding gradients detected. Total norm: {total_norm:.4f}")
    
    @pytest.mark.unit
    @pytest.mark.parametrize("scale", [0.01, 0.1, 1.0, 10.0, 100.0])
    def test_gradient_scaling_with_input_magnitude(self, device, scale):
        """
        Test gradient behavior with different input magnitudes.
        
        Verifies model is robust to differently scaled inputs.
        """
        logger.info(f"Testing gradient scaling with input scale={scale}")
        
        model = create_model(k=2, device=device)
        model.train()
        
        x = torch.randn(BATCH_SIZE, INPUT_DIM, device=device) * scale
        output = model(x)
        loss = output.sum()
        loss.backward()
        
        is_valid, msg = check_gradients_valid(model, k=2)
        assert is_valid, f"Input scale {scale}: {msg}"
        
        total_norm = compute_total_gradient_norm(model)
        logger.info(f"Input scale {scale}: gradient norm = {total_norm:.6f}")
    
    @pytest.mark.unit
    def test_gradient_norm_reasonable_range(self, device):
        """
        Test that gradient norms are in a reasonable range for training.
        """
        logger.info("Testing gradient norms are in reasonable range")
        
        model = create_model(k=2, device=device)
        model.train()
        
        # Use realistic training scenario
        x = torch.randn(BATCH_SIZE, INPUT_DIM, device=device)
        y = torch.randint(0, OUTPUT_DIM, (BATCH_SIZE,), device=device)
        
        output = model(x)
        loss = nn.CrossEntropyLoss()(output, y)
        loss.backward()
        
        total_norm = compute_total_gradient_norm(model)
        
        # Reasonable range for gradient norms
        assert 1e-6 < total_norm < 1e4, \
            f"Gradient norm {total_norm} outside reasonable range [1e-6, 1e4]"
        
        logger.info(f"Gradient norm {total_norm:.4f} in reasonable range")


# ============================================================================
# Gumbel-Softmax Gradient Tests
# ============================================================================

class TestGumbelSoftmaxGradients:
    """Tests for Gumbel-softmax straight-through estimator gradients."""
    
    @pytest.mark.unit
    def test_straight_through_estimator_gradients(self, device):
        """
        Test that straight-through estimator passes gradients correctly.
        
        The STE uses soft values for backward pass while hard values for forward,
        enabling gradient flow through discrete sampling.
        """
        logger.info("Testing straight-through estimator gradients")
        
        try:
            from utils.gating import gumbel_softmax_st
        except ImportError:
            pytest.skip("utils.gating module not available")
        
        # Create logits requiring gradients
        logits = torch.randn(BATCH_SIZE, 5, 4, device=device, requires_grad=True)
        
        # Apply Gumbel-softmax with hard=True (STE mode)
        y_st, y_hard = gumbel_softmax_st(logits, temperature=1.0, hard=True)
        
        # y_hard should be one-hot
        assert torch.allclose(y_hard.sum(dim=-1), torch.ones_like(y_hard.sum(dim=-1))), \
            "y_hard should sum to 1 along last dimension"
        
        # Backward through ST output
        loss = y_st.sum()
        loss.backward()
        
        # Check gradients flow back to logits
        assert logits.grad is not None, "Logits should receive gradients through STE"
        assert not torch.isnan(logits.grad).any(), "Logits gradients contain NaN"
        assert not torch.isinf(logits.grad).any(), "Logits gradients contain Inf"
        
        logger.info("Straight-through estimator gradients verified")
    
    @pytest.mark.unit
    @pytest.mark.parametrize("temperature", [0.1, 0.5, 1.0, 2.0, 5.0])
    def test_gradients_at_different_temperatures(self, device, temperature):
        """
        Test gradient flow at different Gumbel-softmax temperatures.
        
        Lower temperatures produce harder decisions but may have gradient issues.
        Higher temperatures are smoother but less decisive.
        """
        logger.info(f"Testing gradients at temperature={temperature}")
        
        model = create_model(k=2, device=device)
        model.train()
        
        # Set temperature if model supports it
        if hasattr(model, 'temperature'):
            model.temperature = temperature
            logger.debug(f"Set model temperature to {temperature}")
        
        x = torch.randn(BATCH_SIZE, INPUT_DIM, device=device)
        output = model(x)
        loss = output.sum()
        loss.backward()
        
        is_valid, msg = check_gradients_valid(model, k=2)
        assert is_valid, f"Temperature {temperature}: {msg}"
        
        total_norm = compute_total_gradient_norm(model)
        logger.info(f"Temperature {temperature}: gradient norm = {total_norm:.6f}")
    
    @pytest.mark.unit
    def test_hard_vs_soft_gradient_comparison(self, device):
        """
        Compare gradient magnitudes between hard and soft Gumbel-softmax modes.
        """
        logger.info("Comparing hard vs soft Gumbel-softmax gradients")
        
        try:
            from utils.gating import gumbel_softmax_st
        except ImportError:
            pytest.skip("utils.gating module not available")
        
        # Fixed logits for comparison
        logits_base = torch.randn(BATCH_SIZE, 5, 4, device=device)
        
        # Soft mode
        logits_soft = logits_base.clone().requires_grad_(True)
        y_soft, _ = gumbel_softmax_st(logits_soft, temperature=1.0, hard=False)
        y_soft.sum().backward()
        soft_grad_norm = logits_soft.grad.norm().item()
        
        # Hard mode (straight-through)
        logits_hard = logits_base.clone().requires_grad_(True)
        y_hard, _ = gumbel_softmax_st(logits_hard, temperature=1.0, hard=True)
        y_hard.sum().backward()
        hard_grad_norm = logits_hard.grad.norm().item()
        
        # Both should have non-zero gradients
        assert soft_grad_norm > 0, "Soft mode should have non-zero gradients"
        assert hard_grad_norm > 0, "Hard mode (STE) should have non-zero gradients"
        
        logger.info(f"Soft gradient norm: {soft_grad_norm:.6f}")
        logger.info(f"Hard gradient norm: {hard_grad_norm:.6f}")
        
        # In STE, hard mode gradients come from soft approximation
        # so they should be similar in magnitude
        ratio = max(soft_grad_norm, hard_grad_norm) / (min(soft_grad_norm, hard_grad_norm) + 1e-10)
        logger.info(f"Gradient norm ratio: {ratio:.2f}")


# ============================================================================
# Layer-wise Gradient Tests
# ============================================================================

class TestLayerwiseGradients:
    """Tests for gradients in specific model layers."""
    
    @pytest.mark.unit
    def test_input_layer_gradients(self, device):
        """
        Test gradients flow through input/root layer.
        """
        logger.info("Testing input layer gradients")
        
        model = create_model(k=2, device=device)
        model.train()
        
        x = torch.randn(BATCH_SIZE, INPUT_DIM, device=device)
        output = model(x)
        loss = output.sum()
        loss.backward()
        
        # Find root/input layer parameters
        root_params = {name: param for name, param in model.named_parameters()
                      if 'root' in name.lower() or 'input' in name.lower()}
        
        if root_params:
            for name, param in root_params.items():
                assert param.grad is not None, f"{name} should have gradients"
                assert param.grad.norm().item() > 0, f"{name} has zero gradients"
                logger.debug(f"{name}: grad_norm = {param.grad.norm().item():.6f}")
            logger.info(f"Root layer gradients verified: {list(root_params.keys())}")
        else:
            logger.warning("No root/input layer parameters found by name")
    
    @pytest.mark.unit
    def test_output_layer_gradients(self, device):
        """
        Test gradients flow through output layer.
        """
        logger.info("Testing output layer gradients")
        
        model = create_model(k=2, device=device)
        model.train()
        
        x = torch.randn(BATCH_SIZE, INPUT_DIM, device=device)
        output = model(x)
        loss = output.sum()
        loss.backward()
        
        # Find output layer parameters
        output_params = {name: param for name, param in model.named_parameters()
                        if 'output' in name.lower() or 'fc_out' in name.lower() 
                        or 'classifier' in name.lower()}
        
        if output_params:
            for name, param in output_params.items():
                assert param.grad is not None, f"{name} should have gradients"
                assert param.grad.norm().item() > 0, f"{name} has zero gradients"
                logger.debug(f"{name}: grad_norm = {param.grad.norm().item():.6f}")
            logger.info(f"Output layer gradients verified: {list(output_params.keys())}")
        else:
            logger.warning("No output layer parameters found by name")
    
    @pytest.mark.unit
    def test_intermediate_layer_gradients(self, device):
        """
        Test gradients flow through intermediate/branching layers (K>0).
        """
        logger.info("Testing intermediate layer gradients")
        
        model = create_model(k=3, device=device)  # K=3 ensures branching
        model.train()
        
        x = torch.randn(BATCH_SIZE, INPUT_DIM, device=device)
        output = model(x)
        loss = output.sum()
        loss.backward()
        
        # Categorize parameters
        root_keywords = ['root', 'input', 'embed']
        output_keywords = ['output', 'fc_out', 'classifier', 'head']
        
        intermediate_params = {}
        for name, param in model.named_parameters():
            is_root = any(kw in name.lower() for kw in root_keywords)
            is_output = any(kw in name.lower() for kw in output_keywords)
            if not is_root and not is_output and param.grad is not None:
                intermediate_params[name] = param
        
        logger.info(f"Found {len(intermediate_params)} intermediate parameters")
        
        for name, param in intermediate_params.items():
            assert not torch.isnan(param.grad).any(), f"{name} has NaN gradients"
            logger.debug(f"{name}: grad_norm = {param.grad.norm().item():.6f}")


# ============================================================================
# Gradient Clipping Tests
# ============================================================================

class TestGradientClipping:
    """Tests for gradient clipping behavior."""
    
    @pytest.mark.unit
    def test_gradient_clipping_by_norm(self, device):
        """
        Test that gradient clipping by norm works correctly.
        
        clip_grad_norm_ scales gradients so total norm <= max_norm.
        """
        logger.info("Testing gradient clipping by norm")
        
        model = create_model(k=2, device=device)
        model.train()
        
        x = torch.randn(BATCH_SIZE, INPUT_DIM, device=device)
        output = model(x)
        loss = output.sum()
        loss.backward()
        
        # Get norm before clipping
        norm_before = compute_total_gradient_norm(model)
        logger.debug(f"Gradient norm before clipping: {norm_before:.4f}")
        
        # Clip gradients
        max_norm = 1.0
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
        
        # Get norm after clipping
        norm_after = compute_total_gradient_norm(model)
        logger.debug(f"Gradient norm after clipping: {norm_after:.4f}")
        
        # Verify clipping worked
        assert norm_after <= max_norm * 1.01, \
            f"Clipping failed: norm {norm_after:.4f} > max {max_norm}"
        
        logger.info(f"Gradient clipping by norm: {norm_before:.4f} -> {norm_after:.4f}")
    
    @pytest.mark.unit
    def test_gradient_clipping_by_value(self, device):
        """
        Test that gradient clipping by value works correctly.
        
        clip_grad_value_ clamps each gradient element to [-clip_value, clip_value].
        """
        logger.info("Testing gradient clipping by value")
        
        model = create_model(k=2, device=device)
        model.train()
        
        x = torch.randn(BATCH_SIZE, INPUT_DIM, device=device)
        output = model(x)
        loss = output.sum()
        loss.backward()
        
        # Clip by value
        clip_value = 0.5
        torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=clip_value)
        
        # Verify all gradients are within bounds
        for name, param in model.named_parameters():
            if param.grad is not None:
                max_val = param.grad.max().item()
                min_val = param.grad.min().item()
                assert max_val <= clip_value + 1e-6, \
                    f"{name} gradient max {max_val} exceeds clip value {clip_value}"
                assert min_val >= -clip_value - 1e-6, \
                    f"{name} gradient min {min_val} below -clip value"
        
        logger.info(f"Gradient value clipping to ±{clip_value} verified")
    
    @pytest.mark.unit
    def test_clipping_preserves_gradient_direction(self, device):
        """
        Test that gradient clipping preserves gradient direction.
        """
        logger.info("Testing clipping preserves gradient direction")
        
        model = create_model(k=2, device=device)
        model.train()
        
        x = torch.randn(BATCH_SIZE, INPUT_DIM, device=device)
        output = model(x)
        loss = output.sum()
        loss.backward()
        
        # Store original gradient directions (normalized)
        original_directions = {}
        for name, param in model.named_parameters():
            if param.grad is not None and param.grad.norm() > 1e-8:
                original_directions[name] = param.grad / param.grad.norm()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
        
        # Check directions preserved
        for name, param in model.named_parameters():
            if name in original_directions and param.grad is not None:
                if param.grad.norm() > 1e-8:
                    new_direction = param.grad / param.grad.norm()
                    cosine_sim = (original_directions[name] * new_direction).sum()
                    assert cosine_sim > 0.99, \
                        f"{name}: gradient direction changed (cosine sim = {cosine_sim:.4f})"
        
        logger.info("Gradient direction preserved after clipping")


# ============================================================================
# Cross-Entropy Loss Gradient Tests
# ============================================================================

class TestLossGradients:
    """Tests for gradients through different loss functions."""
    
    @pytest.mark.unit
    def test_cross_entropy_gradients(self, device):
        """
        Test gradients flow correctly through cross-entropy loss.
        """
        logger.info("Testing cross-entropy loss gradients")
        
        model = create_model(k=2, device=device)
        model.train()
        
        x = torch.randn(BATCH_SIZE, INPUT_DIM, device=device)
        y = torch.randint(0, OUTPUT_DIM, (BATCH_SIZE,), device=device)
        
        output = model(x)
        loss = nn.CrossEntropyLoss()(output, y)
        loss.backward()
        
        is_valid, msg = check_gradients_valid(model, k=2)
        assert is_valid, msg
        
        logger.info(f"Cross-entropy gradients valid, loss = {loss.item():.4f}")
    
    @pytest.mark.unit
    def test_mse_loss_gradients(self, device):
        """
        Test gradients flow correctly through MSE loss (regression scenario).
        """
        logger.info("Testing MSE loss gradients")
        
        model = create_model(k=2, device=device)
        model.train()
        
        x = torch.randn(BATCH_SIZE, INPUT_DIM, device=device)
        target = torch.randn(BATCH_SIZE, OUTPUT_DIM, device=device)
        
        output = model(x)
        loss = nn.MSELoss()(output, target)
        loss.backward()
        
        is_valid, msg = check_gradients_valid(model, k=2)
        assert is_valid, msg
        
        logger.info(f"MSE loss gradients valid, loss = {loss.item():.4f}")
    
    @pytest.mark.unit
    def test_combined_loss_gradients(self, device):
        """
        Test gradients with combined losses (classification + regularization).
        """
        logger.info("Testing combined loss gradients")
        
        model = create_model(k=2, device=device)
        model.train()
        
        x = torch.randn(BATCH_SIZE, INPUT_DIM, device=device)
        y = torch.randint(0, OUTPUT_DIM, (BATCH_SIZE,), device=device)
        
        output = model(x)
        
        # Combined loss: cross-entropy + L2 regularization
        ce_loss = nn.CrossEntropyLoss()(output, y)
        l2_reg = sum(p.pow(2).sum() for p in model.parameters()) * 0.001
        total_loss = ce_loss + l2_reg
        
        total_loss.backward()
        
        is_valid, msg = check_gradients_valid(model, k=2)
        assert is_valid, msg
        
        logger.info(f"Combined loss gradients valid, "
                   f"CE = {ce_loss.item():.4f}, L2 = {l2_reg.item():.6f}")


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x"])