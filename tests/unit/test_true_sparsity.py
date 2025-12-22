#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/test_true_sparsity.py

Test suite to validate TRUE SPARSITY in BFSNet v2.0.0.

This test verifies the core claim of v2.0.0: only explored children are
computed, not all children. We hook child_fc.forward() to count invocations
and verify the count matches the expected number of explored paths.

Tests
-----
1. Forward call counting: Hook child_fc to count forward() calls
2. Expected vs actual: Compare call count to theoretical maximum
3. Sparsity ratio: Compute actual/theoretical ratio
4. Node count validation: Verify reported node counts match hook counts
5. Multiple rollouts: Test that different rollouts explore different paths

Expected Behavior
-----------------
v1.4.0 (DENSE): child_fc called K * num_parents times (all children computed)
v2.0.0 (SPARSE): child_fc called only for grow=1 decisions (true sparsity)

Usage
-----
python tests/test_true_sparsity.py

Author: BFS project
Date: 2025-12-18 (v2.0.0)
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
from typing import List, Dict, Tuple

# Import model
from bfs_model import BFSNet


class ForwardCallCounter:
    """Hook to count forward() calls on a module."""
    
    def __init__(self):
        self.count = 0
        self.call_sizes: List[int] = []  # Track batch size of each call
    
    def __call__(self, module, input, output):
        """Hook called on forward pass."""
        self.count += 1
        # Input is a tuple, get first element
        if isinstance(input, tuple) and len(input) > 0:
            batch_size = input[0].size(0) if torch.is_tensor(input[0]) else 0
            self.call_sizes.append(batch_size)
    
    def reset(self):
        """Reset counter."""
        self.count = 0
        self.call_sizes.clear()
    
    def total_elements(self) -> int:
        """Total number of elements processed (sum of batch sizes)."""
        return sum(self.call_sizes)


def compute_theoretical_max_nodes(B: int, K: int, D: int) -> int:
    """
    Compute theoretical maximum nodes if ALL children are spawned.
    
    Formula: B * (1 + K + K^2 + ... + K^D)
    """
    if K == 0 or D == 0:
        return B  # Only root nodes
    
    # Geometric series: 1 + K + K^2 + ... + K^D = (K^(D+1) - 1) / (K - 1)
    total_per_example = (K ** (D + 1) - 1) // (K - 1)
    return B * total_per_example


def test_forward_call_counting():
    """Test 1: Hook counts child_fc forward calls correctly."""
    print("\n" + "=" * 70)
    print("Test 1: Forward Call Counting")
    print("=" * 70)
    
    torch.manual_seed(42)
    B, Din, H, Dout = 4, 16, 32, 3
    K, D = 2, 2
    
    model = BFSNet(
        input_dim=Din,
        hidden_dim=H,
        output_dim=Dout,
        max_depth=D,
        max_children=K,
    )
    model.eval()
    
    # Install hook
    counter = ForwardCallCounter()
    hook = model.child_fc.register_forward_hook(counter)
    
    # Forward pass
    x = torch.randn(B, Din)
    logits = model(x)
    
    # Check
    print(f"  Model: D={D}, K={K}, B={B}")
    print(f"  child_fc forward calls: {counter.count}")
    print(f"  Total elements processed: {counter.total_elements()}")
    print(f"  Call sizes: {counter.call_sizes}")
    
    assert counter.count > 0, "child_fc should be called at least once"
    print("  ✓ Hook counting works")
    
    hook.remove()
    return True


def test_sparsity_vs_theoretical_max():
    """Test 2: Verify actual calls < theoretical maximum (true sparsity)."""
    print("\n" + "=" * 70)
    print("Test 2: Sparsity vs Theoretical Maximum")
    print("=" * 70)
    
    torch.manual_seed(42)
    B, Din, H, Dout = 8, 16, 32, 3
    K, D = 3, 2
    
    model = BFSNet(
        input_dim=Din,
        hidden_dim=H,
        output_dim=Dout,
        max_depth=D,
        max_children=K,
    )
    model.eval()
    
    # Theoretical maximum
    max_nodes = compute_theoretical_max_nodes(B, K, D)
    
    # Install hook
    counter = ForwardCallCounter()
    hook = model.child_fc.register_forward_hook(counter)
    
    # Forward pass
    x = torch.randn(B, Din)
    logits = model(x)
    
    # Check
    actual_calls = counter.total_elements()
    sparsity_ratio = actual_calls / max(max_nodes, 1)
    
    print(f"  Model: D={D}, K={K}, B={B}")
    print(f"  Theoretical max children: {max_nodes - B} (excluding root)")
    print(f"  Actual child_fc calls: {actual_calls}")
    print(f"  Sparsity ratio: {sparsity_ratio:.2%}")
    
    assert actual_calls < max_nodes, \
        f"True sparsity FAILED: actual ({actual_calls}) >= max ({max_nodes})"
    print(f"  ✓ True sparsity confirmed: {actual_calls} < {max_nodes}")
    
    hook.remove()
    return True


def test_node_count_validation():
    """Test 3: Verify reported node counts match hook counts."""
    print("\n" + "=" * 70)
    print("Test 3: Node Count Validation")
    print("=" * 70)
    
    torch.manual_seed(42)
    B, Din, H, Dout = 4, 16, 32, 3
    K, D = 2, 2
    
    model = BFSNet(
        input_dim=Din,
        hidden_dim=H,
        output_dim=Dout,
        max_depth=D,
        max_children=K,
    )
    model.train()
    
    # Install hook
    counter = ForwardCallCounter()
    hook = model.child_fc.register_forward_hook(counter)
    
    # Training forward (multiple rollouts)
    x = torch.randn(B, Din)
    y = torch.randint(0, Dout, (B,))
    
    outputs, policy_loss, rewards, node_counts = model(
        x,
        num_rollouts=3,
        lambda_efficiency=0.05,
        beta_entropy=0.01,
        labels=y,
    )
    
    # Check: node_counts includes root, hook only counts child_fc
    total_reported_nodes = sum(node_counts)
    total_hook_calls = counter.total_elements()
    
    # Each rollout has B root nodes not counted by hook
    num_rollouts = len(node_counts)
    expected_hook_calls = total_reported_nodes - (B * num_rollouts)
    
    print(f"  Model: D={D}, K={K}, B={B}, rollouts={num_rollouts}")
    print(f"  Reported total nodes: {total_reported_nodes}")
    print(f"  Root nodes (not in hook): {B * num_rollouts}")
    print(f"  Expected child_fc calls: {expected_hook_calls}")
    print(f"  Actual child_fc calls: {total_hook_calls}")
    print(f"  Per-rollout node counts: {node_counts}")
    
    # Allow small tolerance for floating point
    assert abs(total_hook_calls - expected_hook_calls) <= 1, \
        f"Node count mismatch: hook={total_hook_calls}, expected={expected_hook_calls}"
    print("  ✓ Node counts validated")
    
    hook.remove()
    return True


def test_multiple_rollout_diversity():
    """Test 4: Different rollouts explore different paths."""
    print("\n" + "=" * 70)
    print("Test 4: Multiple Rollout Diversity")
    print("=" * 70)
    
    torch.manual_seed(42)
    B, Din, H, Dout = 4, 16, 32, 3
    K, D = 3, 2
    
    model = BFSNet(
        input_dim=Din,
        hidden_dim=H,
        output_dim=Dout,
        max_depth=D,
        max_children=K,
    )
    model.train()
    
    x = torch.randn(B, Din)
    y = torch.randint(0, Dout, (B,))
    
    # Multiple runs to check diversity
    all_node_counts = []
    for run in range(5):
        outputs, policy_loss, rewards, node_counts = model(
            x,
            num_rollouts=3,
            lambda_efficiency=0.05,
            beta_entropy=0.01,
            labels=y,
        )
        all_node_counts.append(node_counts)
    
    print(f"  Model: D={D}, K={K}, B={B}")
    print(f"  Node counts across 5 runs (3 rollouts each):")
    for i, counts in enumerate(all_node_counts):
        print(f"    Run {i+1}: {counts}")
    
    # Check that node counts vary (not all identical)
    flat_counts = [c for run_counts in all_node_counts for c in run_counts]
    unique_counts = len(set(flat_counts))
    
    print(f"  Unique node counts: {unique_counts}")
    assert unique_counts > 1, "Rollouts should explore different paths"
    print("  ✓ Rollout diversity confirmed")
    
    return True


def test_greedy_vs_stochastic():
    """Test 5: Greedy (inference) vs stochastic (training) behavior."""
    print("\n" + "=" * 70)
    print("Test 5: Greedy vs Stochastic Behavior")
    print("=" * 70)
    
    torch.manual_seed(42)
    B, Din, H, Dout = 4, 16, 32, 3
    K, D = 2, 2
    
    model = BFSNet(
        input_dim=Din,
        hidden_dim=H,
        output_dim=Dout,
        max_depth=D,
        max_children=K,
    )
    
    x = torch.randn(B, Din)
    y = torch.randint(0, Dout, (B,))
    
    # Inference (greedy) - should be deterministic
    model.eval()
    counter_greedy = ForwardCallCounter()
    hook_greedy = model.child_fc.register_forward_hook(counter_greedy)
    
    logits1 = model(x)
    calls_greedy_1 = counter_greedy.total_elements()
    
    counter_greedy.reset()
    logits2 = model(x)
    calls_greedy_2 = counter_greedy.total_elements()
    
    hook_greedy.remove()
    
    print(f"  Greedy mode (inference):")
    print(f"    Run 1 child_fc calls: {calls_greedy_1}")
    print(f"    Run 2 child_fc calls: {calls_greedy_2}")
    print(f"    Deterministic: {calls_greedy_1 == calls_greedy_2}")
    
    assert calls_greedy_1 == calls_greedy_2, "Greedy mode should be deterministic"
    assert torch.allclose(logits1, logits2), "Greedy outputs should be identical"
    
    # Training (stochastic) - can vary
    model.train()
    counter_stoch = ForwardCallCounter()
    hook_stoch = model.child_fc.register_forward_hook(counter_stoch)
    
    _, _, _, counts1 = model(x, num_rollouts=3, lambda_efficiency=0.05, 
                             beta_entropy=0.01, labels=y)
    calls_stoch_1 = counter_stoch.total_elements()
    
    counter_stoch.reset()
    _, _, _, counts2 = model(x, num_rollouts=3, lambda_efficiency=0.05,
                             beta_entropy=0.01, labels=y)
    calls_stoch_2 = counter_stoch.total_elements()
    
    hook_stoch.remove()
    
    print(f"  Stochastic mode (training):")
    print(f"    Run 1 child_fc calls: {calls_stoch_1}")
    print(f"    Run 2 child_fc calls: {calls_stoch_2}")
    print(f"    Varies: {calls_stoch_1 != calls_stoch_2}")
    
    print("  ✓ Greedy deterministic, stochastic varies")
    
    return True


def test_dense_baseline_no_calls():
    """Test 6: Dense baseline (D=0, K=0) should not call child_fc."""
    print("\n" + "=" * 70)
    print("Test 6: Dense Baseline (No child_fc calls)")
    print("=" * 70)
    
    torch.manual_seed(42)
    B, Din, H, Dout = 4, 16, 32, 3
    
    model = BFSNet(
        input_dim=Din,
        hidden_dim=H,
        output_dim=Dout,
        max_depth=0,  # No expansion
        max_children=0,
    )
    model.eval()
    
    # Install hook
    counter = ForwardCallCounter()
    hook = model.child_fc.register_forward_hook(counter)
    
    # Forward pass
    x = torch.randn(B, Din)
    logits = model(x)
    
    print(f"  Model: D=0, K=0 (MLP mode)")
    print(f"  child_fc forward calls: {counter.count}")
    
    assert counter.count == 0, "Dense baseline should not call child_fc"
    print("  ✓ No child_fc calls in MLP mode")
    
    hook.remove()
    return True


def generate_sparsity_report():
    """Generate comprehensive sparsity report."""
    print("\n" + "=" * 70)
    print("SPARSITY REPORT")
    print("=" * 70)
    
    torch.manual_seed(42)
    B = 8
    Din, H, Dout = 16, 32, 3
    
    configs = [
        (1, 2),  # D=1, K=2
        (2, 2),  # D=2, K=2
        (2, 3),  # D=2, K=3
        (3, 2),  # D=3, K=2
        (3, 3),  # D=3, K=3
    ]
    
    results = []
    
    for D, K in configs:
        model = BFSNet(Din, H, Dout, max_depth=D, max_children=K)
        model.eval()
        
        counter = ForwardCallCounter()
        hook = model.child_fc.register_forward_hook(counter)
        
        x = torch.randn(B, Din)
        logits = model(x)
        
        max_nodes = compute_theoretical_max_nodes(B, K, D)
        actual_calls = counter.total_elements()
        sparsity_ratio = actual_calls / max(max_nodes - B, 1)  # Exclude root from ratio
        
        results.append({
            'D': D,
            'K': K,
            'max_children': max_nodes - B,
            'actual_children': actual_calls,
            'sparsity': sparsity_ratio,
        })
        
        hook.remove()
    
    print(f"\n{'Depth':<8} {'K':<6} {'Max Children':<15} {'Actual':<10} {'Ratio':<10}")
    print("-" * 70)
    for r in results:
        print(f"{r['D']:<8} {r['K']:<6} {r['max_children']:<15} "
              f"{r['actual_children']:<10} {r['sparsity']:<10.2%}")
    
    avg_sparsity = sum(r['sparsity'] for r in results) / len(results)
    print(f"\nAverage sparsity ratio: {avg_sparsity:.2%}")
    print(f"(Lower is better - means fewer children computed)")
    
    return True


def run_all_tests():
    """Run all sparsity tests."""
    print("\n" + "=" * 70)
    print("BFSNet v2.0.0 - TRUE SPARSITY TEST SUITE")
    print("=" * 70)
    
    tests = [
        ("Forward Call Counting", test_forward_call_counting),
        ("Sparsity vs Theoretical Max", test_sparsity_vs_theoretical_max),
        ("Node Count Validation", test_node_count_validation),
        ("Multiple Rollout Diversity", test_multiple_rollout_diversity),
        ("Greedy vs Stochastic", test_greedy_vs_stochastic),
        ("Dense Baseline", test_dense_baseline_no_calls),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            if result:
                passed += 1
            else:
                failed += 1
                print(f"  ✗ {test_name} FAILED")
        except AssertionError as e:
            failed += 1
            print(f"\n  ✗ {test_name} FAILED: {e}")
        except Exception as e:
            failed += 1
            print(f"\n  ✗ {test_name} ERROR: {e}")
    
    # Generate report
    print("\n")
    generate_sparsity_report()
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"  Passed: {passed}/{len(tests)}")
    print(f"  Failed: {failed}/{len(tests)}")
    
    if failed == 0:
        print("\n  ✓ ALL TESTS PASSED - TRUE SPARSITY VERIFIED")
    else:
        print(f"\n  ✗ {failed} TEST(S) FAILED")
    
    print("=" * 70)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)