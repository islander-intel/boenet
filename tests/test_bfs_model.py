#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/test_bfs_model.py

Unit tests for BFSNet v2.0.0 core model functionality.

This test suite validates the fundamental behaviors of the BFSNet model
with REINFORCE policy gradients, independent of full training pipelines.

Test Categories
---------------
1. Initialization: Model construction with various configurations
2. Forward pass: Both training and inference modes
3. Output shapes: Verify tensor dimensions
4. Gradient flow: Ensure backpropagation works
5. Device handling: CPU and CUDA compatibility
6. Edge cases: Empty batches, extreme parameters

Author: BFS project
Date: 2025-12-18 (v2.0.0)
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest
from typing import Tuple

# Import model
from bfs_model import BFSNet


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def device():
    """Get available device (CUDA if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def default_config():
    """Default BFSNet configuration for testing."""
    return {
        'input_dim': 784,
        'hidden_dim': 64,
        'output_dim': 10,
        'max_depth': 2,
        'max_children': 3,
        'sibling_embed': True,
        'pooling_mode': 'mean',
    }


@pytest.fixture
def sample_batch(device):
    """Create sample input batch."""
    return torch.randn(16, 784, device=device)


@pytest.fixture
def sample_labels(device):
    """Create sample labels."""
    return torch.randint(0, 10, (16,), device=device)


# ============================================================================
# Test 1: Model Initialization
# ============================================================================

def test_model_initialization_default(default_config, device):
    """Test model initializes with default configuration."""
    model = BFSNet(**default_config).to(device)
    
    assert model.input_dim == 784
    assert model.hidden_dim == 64
    assert model.output_dim == 10
    assert model.max_depth == 2
    assert model.max_children == 3
    assert model.sibling_embed is True
    assert model.pooling_mode == 'mean'


def test_model_initialization_k0_baseline(device):
    """Test dense baseline (K=0, D=0) initialization."""
    model = BFSNet(
        input_dim=784,
        hidden_dim=64,
        output_dim=10,
        max_depth=0,
        max_children=0,
        pooling_mode='mean'
    ).to(device)
    
    assert model.max_depth == 0
    assert model.max_children == 0


def test_model_initialization_various_pooling(device):
    """Test model with different pooling modes."""
    for pooling in ['mean', 'sum', 'learned']:
        model = BFSNet(
            input_dim=784,
            hidden_dim=64,
            output_dim=10,
            max_depth=2,
            max_children=3,
            pooling_mode=pooling
        ).to(device)
        
        assert model.pooling_mode == pooling


def test_model_initialization_with_pruning(device):
    """Test model with pruning enabled."""
    model = BFSNet(
        input_dim=784,
        hidden_dim=64,
        output_dim=10,
        max_depth=2,
        max_children=3,
        use_pruning=True,
        pruning_mode='learned'
    ).to(device)
    
    assert model.use_pruning is True
    assert model.prune_gate is not None


def test_model_has_growth_policy(default_config, device):
    """Test that model has GrowthPolicyNet (v2.0.0 component)."""
    model = BFSNet(**default_config).to(device)
    
    assert hasattr(model, 'growth_policy')
    assert model.growth_policy is not None


# ============================================================================
# Test 2: Forward Pass - Inference Mode
# ============================================================================

def test_forward_inference_mode(default_config, sample_batch, device):
    """Test forward pass in inference mode (greedy)."""
    model = BFSNet(**default_config).to(device)
    model.eval()
    
    with torch.no_grad():
        logits = model(sample_batch)
    
    assert logits.shape == (16, 10)
    assert not torch.isnan(logits).any()
    assert not torch.isinf(logits).any()


def test_forward_inference_deterministic(default_config, sample_batch, device):
    """Test that inference is deterministic (same input → same output)."""
    model = BFSNet(**default_config).to(device)
    model.eval()
    
    with torch.no_grad():
        logits1 = model(sample_batch)
        logits2 = model(sample_batch)
    
    assert torch.allclose(logits1, logits2, rtol=1e-5)


def test_forward_inference_k0_baseline(sample_batch, device):
    """Test inference with K=0 (dense baseline)."""
    model = BFSNet(
        input_dim=784,
        hidden_dim=64,
        output_dim=10,
        max_depth=0,
        max_children=0
    ).to(device)
    model.eval()
    
    with torch.no_grad():
        logits = model(sample_batch)
    
    assert logits.shape == (16, 10)


# ============================================================================
# Test 3: Forward Pass - Training Mode
# ============================================================================

def test_forward_training_mode(default_config, sample_batch, sample_labels, device):
    """Test forward pass in training mode (stochastic rollouts)."""
    model = BFSNet(**default_config).to(device)
    model.train()
    
    outputs, policy_loss, rewards, node_counts = model(
        sample_batch,
        num_rollouts=3,
        lambda_efficiency=0.05,
        beta_entropy=0.01,
        labels=sample_labels
    )
    
    # Check outputs
    assert outputs.shape == (16, 10)
    assert not torch.isnan(outputs).any()
    
    # Check policy loss
    assert policy_loss.requires_grad
    assert not torch.isnan(policy_loss)
    
    # Check rewards
    assert rewards.shape == (3,)  # num_rollouts
    assert not torch.isnan(rewards).any()
    
    # Check node counts
    assert len(node_counts) == 3  # num_rollouts
    assert all(isinstance(c, int) for c in node_counts)


def test_forward_training_various_rollouts(default_config, sample_batch, sample_labels, device):
    """Test training with different num_rollouts."""
    model = BFSNet(**default_config).to(device)
    model.train()
    
    for num_rollouts in [1, 3, 5]:
        outputs, policy_loss, rewards, node_counts = model(
            sample_batch,
            num_rollouts=num_rollouts,
            lambda_efficiency=0.05,
            beta_entropy=0.01,
            labels=sample_labels
        )
        
        assert rewards.shape == (num_rollouts,)
        assert len(node_counts) == num_rollouts


def test_forward_training_k0_raises_error(sample_batch, sample_labels, device):
    """Test that K=0 baseline doesn't support training mode rollouts."""
    model = BFSNet(
        input_dim=784,
        hidden_dim=64,
        output_dim=10,
        max_depth=0,
        max_children=0
    ).to(device)
    model.train()
    
    # K=0 with training mode should still work, just returns minimal structure
    # This test ensures it doesn't crash
    try:
        outputs, policy_loss, rewards, node_counts = model(
            sample_batch,
            num_rollouts=1,
            lambda_efficiency=0.05,
            beta_entropy=0.01,
            labels=sample_labels
        )
        # Should work, just with minimal nodes (only roots)
        assert outputs.shape == (16, 10)
    except Exception as e:
        pytest.fail(f"K=0 training should not raise exception: {e}")


# ============================================================================
# Test 4: Gradient Flow
# ============================================================================

def test_gradient_flow_classification(default_config, sample_batch, sample_labels, device):
    """Test that gradients flow through classification path."""
    model = BFSNet(**default_config).to(device)
    model.train()
    
    outputs, policy_loss, rewards, node_counts = model(
        sample_batch,
        num_rollouts=3,
        lambda_efficiency=0.05,
        beta_entropy=0.01,
        labels=sample_labels
    )
    
    classification_loss = F.cross_entropy(outputs, sample_labels)
    classification_loss.backward()
    
    # Check that root_fc has gradients
    assert model.root_fc.weight.grad is not None
    assert model.root_fc.weight.grad.abs().sum() > 0


def test_gradient_flow_policy(default_config, sample_batch, sample_labels, device):
    """Test that gradients flow through policy path."""
    model = BFSNet(**default_config).to(device)
    model.train()
    
    outputs, policy_loss, rewards, node_counts = model(
        sample_batch,
        num_rollouts=3,
        lambda_efficiency=0.05,
        beta_entropy=0.01,
        labels=sample_labels
    )
    
    policy_loss.backward()
    
    # Check that growth_policy has gradients
    for param in model.growth_policy.parameters():
        if param.requires_grad:
            assert param.grad is not None
            # Note: May be zero if no children were created, which is OK


def test_gradient_flow_combined(default_config, sample_batch, sample_labels, device):
    """Test that gradients flow through combined loss."""
    model = BFSNet(**default_config).to(device)
    model.train()
    
    outputs, policy_loss, rewards, node_counts = model(
        sample_batch,
        num_rollouts=3,
        lambda_efficiency=0.05,
        beta_entropy=0.01,
        labels=sample_labels
    )
    
    classification_loss = F.cross_entropy(outputs, sample_labels)
    total_loss = classification_loss + 0.5 * policy_loss
    total_loss.backward()
    
    # Check both paths have gradients
    assert model.root_fc.weight.grad is not None
    assert model.root_fc.weight.grad.abs().sum() > 0


# ============================================================================
# Test 5: Output Shape Validation
# ============================================================================

@pytest.mark.parametrize("batch_size", [1, 8, 16, 32, 64])
def test_output_shape_various_batch_sizes(batch_size, device):
    """Test output shapes with various batch sizes."""
    model = BFSNet(
        input_dim=784,
        hidden_dim=64,
        output_dim=10,
        max_depth=2,
        max_children=3
    ).to(device)
    model.eval()
    
    x = torch.randn(batch_size, 784, device=device)
    
    with torch.no_grad():
        logits = model(x)
    
    assert logits.shape == (batch_size, 10)


@pytest.mark.parametrize("output_dim", [2, 5, 10, 100])
def test_output_shape_various_classes(output_dim, device):
    """Test output shapes with various number of classes."""
    model = BFSNet(
        input_dim=784,
        hidden_dim=64,
        output_dim=output_dim,
        max_depth=2,
        max_children=3
    ).to(device)
    model.eval()
    
    x = torch.randn(16, 784, device=device)
    
    with torch.no_grad():
        logits = model(x)
    
    assert logits.shape == (16, output_dim)


# ============================================================================
# Test 6: Device Handling
# ============================================================================

def test_model_cpu():
    """Test model works on CPU."""
    model = BFSNet(
        input_dim=784,
        hidden_dim=64,
        output_dim=10,
        max_depth=2,
        max_children=3
    )
    
    x = torch.randn(16, 784)
    
    model.eval()
    with torch.no_grad():
        logits = model(x)
    
    assert logits.device.type == 'cpu'


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_model_cuda():
    """Test model works on CUDA."""
    device = torch.device("cuda")
    model = BFSNet(
        input_dim=784,
        hidden_dim=64,
        output_dim=10,
        max_depth=2,
        max_children=3
    ).to(device)
    
    x = torch.randn(16, 784, device=device)
    
    model.eval()
    with torch.no_grad():
        logits = model(x)
    
    assert logits.device.type == 'cuda'


# ============================================================================
# Test 7: Edge Cases
# ============================================================================

def test_single_example_batch(device):
    """Test model with batch size of 1."""
    model = BFSNet(
        input_dim=784,
        hidden_dim=64,
        output_dim=10,
        max_depth=2,
        max_children=3
    ).to(device)
    model.eval()
    
    x = torch.randn(1, 784, device=device)
    
    with torch.no_grad():
        logits = model(x)
    
    assert logits.shape == (1, 10)


def test_max_depth_zero(device):
    """Test model with max_depth=0 (no BFS expansion)."""
    model = BFSNet(
        input_dim=784,
        hidden_dim=64,
        output_dim=10,
        max_depth=0,
        max_children=0
    ).to(device)
    model.eval()
    
    x = torch.randn(16, 784, device=device)
    
    with torch.no_grad():
        logits = model(x)
    
    assert logits.shape == (16, 10)


def test_max_children_one(device):
    """Test model with max_children=1 (single child per parent)."""
    model = BFSNet(
        input_dim=784,
        hidden_dim=64,
        output_dim=10,
        max_depth=2,
        max_children=1
    ).to(device)
    model.eval()
    
    x = torch.randn(16, 784, device=device)
    
    with torch.no_grad():
        logits = model(x)
    
    assert logits.shape == (16, 10)


def test_large_hidden_dim(device):
    """Test model with large hidden dimension."""
    model = BFSNet(
        input_dim=784,
        hidden_dim=512,
        output_dim=10,
        max_depth=2,
        max_children=3
    ).to(device)
    model.eval()
    
    x = torch.randn(16, 784, device=device)
    
    with torch.no_grad():
        logits = model(x)
    
    assert logits.shape == (16, 10)


# ============================================================================
# Test 8: Model Summary
# ============================================================================

def test_model_summary(default_config):
    """Test that model.summary() returns valid string."""
    model = BFSNet(**default_config)
    summary = model.summary()
    
    assert isinstance(summary, str)
    assert "BFSNet v2.0.0" in summary
    assert "input_dim=784" in summary
    assert "hidden_dim=64" in summary
    assert "max_depth=2" in summary


# ============================================================================
# Test 9: v2.0.0 Specific Features
# ============================================================================

def test_v2_has_no_branching_gate(default_config, device):
    """Test that v2.0.0 does NOT have BranchingGate (removed from v1.4.0)."""
    model = BFSNet(**default_config).to(device)
    
    assert not hasattr(model, 'branch_gate')
    assert not hasattr(model, '_branch_temperature')


def test_v2_has_growth_policy_net(default_config, device):
    """Test that v2.0.0 HAS GrowthPolicyNet (new in v2.0.0)."""
    model = BFSNet(**default_config).to(device)
    
    assert hasattr(model, 'growth_policy')
    from utils.gating import GrowthPolicyNet
    assert isinstance(model.growth_policy, GrowthPolicyNet)


def test_v2_forward_returns_tuple_in_training(default_config, sample_batch, sample_labels, device):
    """Test that v2.0.0 forward returns tuple in training mode."""
    model = BFSNet(**default_config).to(device)
    model.train()
    
    result = model(
        sample_batch,
        num_rollouts=3,
        lambda_efficiency=0.05,
        beta_entropy=0.01,
        labels=sample_labels
    )
    
    assert isinstance(result, tuple)
    assert len(result) == 4
    outputs, policy_loss, rewards, node_counts = result


def test_v2_forward_returns_tensor_in_inference(default_config, sample_batch, device):
    """Test that v2.0.0 forward returns tensor in inference mode."""
    model = BFSNet(**default_config).to(device)
    model.eval()
    
    with torch.no_grad():
        result = model(sample_batch)
    
    assert isinstance(result, torch.Tensor)
    assert result.shape == (16, 10)


# ============================================================================
# Test Runner
# ============================================================================

if __name__ == "__main__":
    # Run with pytest
    pytest.main([__file__, "-v", "--tb=short"])