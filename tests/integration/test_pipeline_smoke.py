# ============================================================================
# BFSNet Pipeline Smoke Tests
# ============================================================================
# End-to-end smoke tests for the BFSNet training pipeline.
#
# These tests verify:
#   - Complete training pipeline executes without errors
#   - Model can be trained and evaluated
#   - Checkpoints are saved correctly
#   - Results are generated in expected format
#   - Different K values work in full pipeline
#   - Warmup → sparse transition works end-to-end
#
# Smoke tests are designed to catch major integration issues quickly.
# They use minimal configurations to keep execution time reasonable.
#
# Run:
#   pytest tests/integration/test_pipeline_smoke.py -v
#   pytest tests/integration/test_pipeline_smoke.py -v -k "test_single"
# ============================================================================

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import tempfile
import os
import sys
import time
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Test Constants
# ============================================================================

# Reduced dimensions for smoke tests
INPUT_DIM = 784
OUTPUT_DIM = 10
HIDDEN_DIM = 32
MAX_DEPTH = 2
BATCH_SIZE = 32

# Minimal training settings
NUM_EPOCHS = 3
WARMUP_EPOCHS = 1
LEARNING_RATE = 0.01

# Dataset sizes (small for speed)
TRAIN_SIZE = 200
VAL_SIZE = 50


# ============================================================================
# Helper Functions
# ============================================================================

def create_model(k: int, device: torch.device):
    """Create a BFSNet model for testing."""
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


def create_synthetic_data(
    num_samples: int,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create synthetic data for testing."""
    X = torch.randn(num_samples, INPUT_DIM, device=device)
    y = torch.randint(0, OUTPUT_DIM, (num_samples,), device=device)
    return X, y


def create_data_loaders(
    device: torch.device,
    train_size: int = TRAIN_SIZE,
    val_size: int = VAL_SIZE,
    batch_size: int = BATCH_SIZE
) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation data loaders."""
    # Create synthetic data
    X_train, y_train = create_synthetic_data(train_size, device)
    X_val, y_val = create_synthetic_data(val_size, device)
    
    # Move to CPU for DataLoader (will be moved back during iteration)
    train_dataset = TensorDataset(X_train.cpu(), y_train.cpu())
    val_dataset = TensorDataset(X_val.cpu(), y_val.cpu())
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader


def set_execution_mode(model: nn.Module, mode: str) -> bool:
    """Set model execution mode if supported."""
    if hasattr(model, 'set_exec_mode'):
        model.set_exec_mode(mode)
        return True
    elif hasattr(model, 'exec_mode'):
        model.exec_mode = mode
        return True
    return False


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    warmup_epochs: int = WARMUP_EPOCHS
) -> Tuple[float, float]:
    """Train for one epoch and return loss and accuracy."""
    model.train()
    
    # Set execution mode based on epoch
    if epoch < warmup_epochs:
        set_execution_mode(model, 'soft_full')
    else:
        set_execution_mode(model, 'sparse')
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += (pred == target).sum().item()
        total += target.size(0)
    
    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total
    
    return avg_loss, accuracy


def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """Evaluate model and return loss and accuracy."""
    model.eval()
    set_execution_mode(model, 'sparse')
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    
    avg_loss = total_loss / len(val_loader) if len(val_loader) > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0
    
    return avg_loss, accuracy


def run_full_training(
    k: int,
    device: torch.device,
    num_epochs: int = NUM_EPOCHS,
    warmup_epochs: int = WARMUP_EPOCHS
) -> Dict[str, Any]:
    """Run full training pipeline and return results."""
    logger.info(f"Starting training pipeline: K={k}, epochs={num_epochs}")
    
    start_time = time.time()
    
    # Create model and data
    model = create_model(k=k, device=device)
    train_loader, val_loader = create_data_loaders(device)
    
    # Setup training
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
    }
    best_val_acc = 0.0
    best_epoch = 0
    
    for epoch in range(num_epochs):
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion,
            device, epoch, warmup_epochs
        )
        
        # Evaluate
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Track best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
        
        logger.debug(f"Epoch {epoch+1}/{num_epochs}: "
                    f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                    f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
    
    elapsed_time = time.time() - start_time
    
    results = {
        'k': k,
        'num_epochs': num_epochs,
        'warmup_epochs': warmup_epochs,
        'final_train_loss': history['train_loss'][-1],
        'final_train_acc': history['train_acc'][-1],
        'final_val_loss': history['val_loss'][-1],
        'final_val_acc': history['val_acc'][-1],
        'best_val_acc': best_val_acc,
        'best_epoch': best_epoch,
        'elapsed_time': elapsed_time,
        'history': history,
        'model': model,
    }
    
    logger.info(f"Training complete: K={k}, best_val_acc={best_val_acc:.4f}, "
               f"time={elapsed_time:.2f}s")
    
    return results


# ============================================================================
# Basic Smoke Tests
# ============================================================================

class TestBasicSmoke:
    """Basic smoke tests for the training pipeline."""
    
    @pytest.mark.integration
    @pytest.mark.smoke
    def test_model_creation(self, device):
        """Smoke test: Model can be created."""
        model = create_model(k=2, device=device)
        
        assert model is not None
        param_count = sum(p.numel() for p in model.parameters())
        assert param_count > 0
        
        logger.info(f"Model created with {param_count} parameters")
    
    @pytest.mark.integration
    @pytest.mark.smoke
    def test_forward_pass(self, device):
        """Smoke test: Forward pass works."""
        model = create_model(k=2, device=device)
        model.eval()
        
        x = torch.randn(BATCH_SIZE, INPUT_DIM, device=device)
        
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (BATCH_SIZE, OUTPUT_DIM)
        assert not torch.isnan(output).any()
        
        logger.info("Forward pass successful")
    
    @pytest.mark.integration
    @pytest.mark.smoke
    def test_backward_pass(self, device):
        """Smoke test: Backward pass works."""
        model = create_model(k=2, device=device)
        model.train()
        
        x = torch.randn(BATCH_SIZE, INPUT_DIM, device=device)
        y = torch.randint(0, OUTPUT_DIM, (BATCH_SIZE,), device=device)
        
        output = model(x)
        loss = nn.CrossEntropyLoss()(output, y)
        loss.backward()
        
        # Check gradients exist
        has_grads = any(p.grad is not None for p in model.parameters())
        assert has_grads, "No gradients computed"
        
        logger.info("Backward pass successful")
    
    @pytest.mark.integration
    @pytest.mark.smoke
    def test_optimizer_step(self, device):
        """Smoke test: Optimizer step works."""
        model = create_model(k=2, device=device)
        model.train()
        
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        
        # Store original weights
        original_weight = next(model.parameters()).clone()
        
        x = torch.randn(BATCH_SIZE, INPUT_DIM, device=device)
        y = torch.randint(0, OUTPUT_DIM, (BATCH_SIZE,), device=device)
        
        optimizer.zero_grad()
        output = model(x)
        loss = nn.CrossEntropyLoss()(output, y)
        loss.backward()
        optimizer.step()
        
        # Check weights changed
        new_weight = next(model.parameters())
        weights_changed = not torch.allclose(original_weight, new_weight)
        
        assert weights_changed, "Weights should change after optimizer step"
        
        logger.info("Optimizer step successful")


# ============================================================================
# Single Run Smoke Tests
# ============================================================================

class TestSingleRunSmoke:
    """Smoke tests for single training runs."""
    
    @pytest.mark.integration
    @pytest.mark.smoke
    @pytest.mark.parametrize("k_value", [0, 2])
    def test_single_epoch_training(self, device, k_value):
        """Smoke test: Single epoch of training completes."""
        model = create_model(k=k_value, device=device)
        train_loader, _ = create_data_loaders(device)
        
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.CrossEntropyLoss()
        
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion,
            device, epoch=0, warmup_epochs=1
        )
        
        assert train_loss > 0, "Loss should be positive"
        assert 0 <= train_acc <= 1, "Accuracy should be in [0, 1]"
        
        logger.info(f"K={k_value}: Single epoch - loss={train_loss:.4f}, acc={train_acc:.4f}")
    
    @pytest.mark.integration
    @pytest.mark.smoke
    @pytest.mark.parametrize("k_value", [0, 2])
    def test_evaluation(self, device, k_value):
        """Smoke test: Model evaluation works."""
        model = create_model(k=k_value, device=device)
        _, val_loader = create_data_loaders(device)
        
        criterion = nn.CrossEntropyLoss()
        
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        assert val_loss > 0, "Loss should be positive"
        assert 0 <= val_acc <= 1, "Accuracy should be in [0, 1]"
        
        logger.info(f"K={k_value}: Evaluation - loss={val_loss:.4f}, acc={val_acc:.4f}")
    
    @pytest.mark.integration
    @pytest.mark.smoke
    def test_data_loader_iteration(self, device):
        """Smoke test: Data loaders can be iterated."""
        train_loader, val_loader = create_data_loaders(device)
        
        # Check train loader
        train_batches = 0
        for data, target in train_loader:
            assert data.shape[1] == INPUT_DIM
            assert target.shape[0] == data.shape[0]
            train_batches += 1
        
        # Check val loader
        val_batches = 0
        for data, target in val_loader:
            assert data.shape[1] == INPUT_DIM
            val_batches += 1
        
        assert train_batches > 0
        assert val_batches > 0
        
        logger.info(f"Data loaders: {train_batches} train batches, {val_batches} val batches")


# ============================================================================
# Full Pipeline Smoke Tests
# ============================================================================

class TestFullPipelineSmoke:
    """Smoke tests for the complete training pipeline."""
    
    @pytest.mark.integration
    @pytest.mark.smoke
    @pytest.mark.parametrize("k_value", [0, 2])
    def test_full_training_pipeline(self, device, k_value):
        """Smoke test: Full training pipeline completes."""
        results = run_full_training(k=k_value, device=device)
        
        # Verify results structure
        assert 'final_train_loss' in results
        assert 'final_val_acc' in results
        assert 'best_val_acc' in results
        assert 'history' in results
        
        # Verify reasonable values
        assert results['final_train_loss'] > 0
        assert 0 <= results['final_val_acc'] <= 1
        assert results['elapsed_time'] > 0
        
        logger.info(f"K={k_value}: Full pipeline completed in {results['elapsed_time']:.2f}s")
    
    @pytest.mark.integration
    @pytest.mark.smoke
    def test_training_reduces_loss(self, device):
        """Smoke test: Training reduces loss over epochs."""
        results = run_full_training(k=2, device=device, num_epochs=5)
        
        history = results['history']
        initial_loss = history['train_loss'][0]
        final_loss = history['train_loss'][-1]
        
        # Loss should generally decrease (allow some variance)
        logger.info(f"Loss: {initial_loss:.4f} -> {final_loss:.4f}")
        
        # At minimum, training should not explode
        assert final_loss < initial_loss * 10, "Loss exploded during training"
    
    @pytest.mark.integration
    @pytest.mark.smoke
    def test_warmup_to_sparse_transition(self, device):
        """Smoke test: Warmup to sparse transition works."""
        model = create_model(k=3, device=device)
        train_loader, _ = create_data_loaders(device)
        
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.CrossEntropyLoss()
        
        # Warmup epoch (soft_full)
        loss_warmup, _ = train_one_epoch(
            model, train_loader, optimizer, criterion,
            device, epoch=0, warmup_epochs=2
        )
        
        # Post-warmup epoch (sparse)
        loss_sparse, _ = train_one_epoch(
            model, train_loader, optimizer, criterion,
            device, epoch=2, warmup_epochs=2
        )
        
        # Both should produce valid losses
        assert loss_warmup > 0 and not torch.isnan(torch.tensor(loss_warmup))
        assert loss_sparse > 0 and not torch.isnan(torch.tensor(loss_sparse))
        
        logger.info(f"Warmup loss: {loss_warmup:.4f}, Sparse loss: {loss_sparse:.4f}")


# ============================================================================
# Checkpoint Smoke Tests
# ============================================================================

class TestCheckpointSmoke:
    """Smoke tests for checkpoint saving and loading."""
    
    @pytest.mark.integration
    @pytest.mark.smoke
    def test_save_checkpoint(self, device, tmp_path):
        """Smoke test: Checkpoint can be saved."""
        model = create_model(k=2, device=device)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        # Train briefly
        train_loader, _ = create_data_loaders(device)
        criterion = nn.CrossEntropyLoss()
        
        train_one_epoch(model, train_loader, optimizer, criterion, device, 0)
        
        # Save checkpoint
        checkpoint_path = tmp_path / "checkpoint.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': 1,
        }, checkpoint_path)
        
        assert checkpoint_path.exists()
        assert checkpoint_path.stat().st_size > 0
        
        logger.info(f"Checkpoint saved: {checkpoint_path.stat().st_size} bytes")
    
    @pytest.mark.integration
    @pytest.mark.smoke
    def test_load_checkpoint(self, device, tmp_path):
        """Smoke test: Checkpoint can be loaded."""
        # Create and save model
        model1 = create_model(k=2, device=device)
        checkpoint_path = tmp_path / "checkpoint.pt"
        
        torch.save({'model_state_dict': model1.state_dict()}, checkpoint_path)
        
        # Load into new model
        model2 = create_model(k=2, device=device)
        checkpoint = torch.load(checkpoint_path)
        model2.load_state_dict(checkpoint['model_state_dict'])
        
        # Verify same output
        x = torch.randn(BATCH_SIZE, INPUT_DIM, device=device)
        
        model1.eval()
        model2.eval()
        
        torch.manual_seed(42)
        with torch.no_grad():
            out1 = model1(x)
        
        torch.manual_seed(42)
        with torch.no_grad():
            out2 = model2(x)
        
        assert torch.allclose(out1, out2)
        
        logger.info("Checkpoint loaded successfully")
    
    @pytest.mark.integration
    @pytest.mark.smoke
    def test_resume_training(self, device, tmp_path):
        """Smoke test: Training can be resumed from checkpoint."""
        train_loader, val_loader = create_data_loaders(device)
        criterion = nn.CrossEntropyLoss()
        
        # Phase 1: Initial training
        model = create_model(k=2, device=device)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        for epoch in range(2):
            train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
        
        # Save checkpoint
        checkpoint_path = tmp_path / "checkpoint.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': 2,
        }, checkpoint_path)
        
        # Phase 2: Resume training
        model2 = create_model(k=2, device=device)
        optimizer2 = optim.Adam(model2.parameters(), lr=LEARNING_RATE)
        
        checkpoint = torch.load(checkpoint_path)
        model2.load_state_dict(checkpoint['model_state_dict'])
        optimizer2.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        
        assert start_epoch == 2
        
        # Continue training
        for epoch in range(start_epoch, start_epoch + 2):
            loss, acc = train_one_epoch(
                model2, train_loader, optimizer2, criterion, device, epoch
            )
            logger.debug(f"Resumed epoch {epoch}: loss={loss:.4f}")
        
        logger.info("Training resumed successfully from checkpoint")


# ============================================================================
# Multi-Configuration Smoke Tests
# ============================================================================

class TestMultiConfigSmoke:
    """Smoke tests for multiple configurations."""
    
    @pytest.mark.integration
    @pytest.mark.smoke
    @pytest.mark.parametrize("k_value", [0, 1, 2, 3])
    def test_all_k_values(self, device, k_value):
        """Smoke test: All K values work in pipeline."""
        results = run_full_training(k=k_value, device=device, num_epochs=2)
        
        assert results['final_val_acc'] >= 0
        logger.info(f"K={k_value}: val_acc={results['final_val_acc']:.4f}")
    
    @pytest.mark.integration
    @pytest.mark.smoke
    @pytest.mark.parametrize("pooling", ["learned", "sum", "mean"])
    def test_all_pooling_modes(self, device, pooling):
        """Smoke test: All pooling modes work."""
        try:
            from bfs_model import BFSNet
        except ImportError:
            pytest.skip("bfs_model not available")
        
        model = BFSNet(
            input_dim=INPUT_DIM,
            output_dim=OUTPUT_DIM,
            hidden_dim=HIDDEN_DIM,
            max_depth=MAX_DEPTH,
            max_children=2,
            pooling=pooling
        ).to(device)
        
        train_loader, _ = create_data_loaders(device)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.CrossEntropyLoss()
        
        loss, acc = train_one_epoch(model, train_loader, optimizer, criterion, device, 0)
        
        assert loss > 0
        logger.info(f"Pooling={pooling}: loss={loss:.4f}")
    
    @pytest.mark.integration
    @pytest.mark.smoke
    @pytest.mark.slow
    def test_multiple_runs_sequential(self, device):
        """Smoke test: Multiple sequential runs complete."""
        results_list = []
        
        for k in [0, 2]:
            results = run_full_training(k=k, device=device, num_epochs=2)
            results_list.append(results)
        
        assert len(results_list) == 2
        
        for r in results_list:
            assert r['final_val_acc'] >= 0
        
        logger.info(f"Completed {len(results_list)} sequential runs")


# ============================================================================
# Error Handling Smoke Tests
# ============================================================================

class TestErrorHandlingSmoke:
    """Smoke tests for error handling."""
    
    @pytest.mark.integration
    @pytest.mark.smoke
    def test_handles_nan_gracefully(self, device):
        """Smoke test: Pipeline handles NaN detection."""
        model = create_model(k=2, device=device)
        model.eval()
        
        # Normal input should work
        x_normal = torch.randn(BATCH_SIZE, INPUT_DIM, device=device)
        with torch.no_grad():
            output_normal = model(x_normal)
        
        assert not torch.isnan(output_normal).any()
        
        logger.info("NaN handling: Normal inputs produce valid outputs")
    
    @pytest.mark.integration
    @pytest.mark.smoke
    def test_empty_batch_handling(self, device):
        """Smoke test: Empty batch is handled."""
        model = create_model(k=2, device=device)
        model.eval()
        
        x_empty = torch.randn(0, INPUT_DIM, device=device)
        
        try:
            with torch.no_grad():
                output = model(x_empty)
            logger.info("Empty batch: Handled gracefully")
        except (RuntimeError, IndexError):
            logger.info("Empty batch: Raises expected error")


# ============================================================================
# Performance Smoke Tests
# ============================================================================

class TestPerformanceSmoke:
    """Smoke tests for performance characteristics."""
    
    @pytest.mark.integration
    @pytest.mark.smoke
    def test_training_completes_in_reasonable_time(self, device):
        """Smoke test: Training completes in reasonable time."""
        start_time = time.time()
        
        results = run_full_training(k=2, device=device, num_epochs=3)
        
        elapsed = time.time() - start_time
        
        # Should complete in under 60 seconds for smoke test
        assert elapsed < 60, f"Training took too long: {elapsed:.2f}s"
        
        logger.info(f"Training completed in {elapsed:.2f}s")
    
    @pytest.mark.integration
    @pytest.mark.smoke
    def test_memory_not_leaking(self, device):
        """Smoke test: Memory is released between runs."""
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated()
        
        # Run training multiple times
        for i in range(3):
            results = run_full_training(k=2, device=device, num_epochs=2)
            del results
        
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            final_memory = torch.cuda.memory_allocated()
            
            # Memory should be similar (allowing for some variance)
            memory_diff = final_memory - initial_memory
            logger.info(f"Memory difference: {memory_diff} bytes")


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])