#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_bfs_sparse.py

Unit tests for BFSNet sparse vs dense execution.

What is tested
--------------
1) Logits consistency
   - Dense vs sparse forward logits must be numerically identical (within tolerance).
2) Trace consistency
   - Spawned counts and active-mask sums must be equal across dense and sparse.
3) Early exit
   - If the frontier empties early (no active nodes left), model should stop expanding.

Usage
-----
pytest tests/test_bfs_sparse.py -v
or
python -m unittest tests/test_bfs_sparse.py

Notes
-----
- Assumes BFSNet implements a `sparse_execute` flag (bool).
- Uses small synthetic inputs for speed and determinism.
- All tests run on CPU by default, CUDA if available.

Author: William McKeon
Updated: 2025-08-26
"""

import torch
import pytest
from bfs_model import BFSNet


# --------------------------------------------------------------------------- #
#                              Helper utilities                               #
# --------------------------------------------------------------------------- #

def make_pair(
    B: int = 8,
    D: int = 2,
    K: int = 2,
    H: int = 16,
    Din: int = 8,
    Dout: int = 3,
    seed: int = 123,
):
    """
    Build dense + sparse BFSNet pair with same weights for testing.
    Returns (dense_model, sparse_model, x).
    """
    torch.manual_seed(seed)
    x = torch.randn(B, Din)

    # Dense model
    dense = BFSNet(
        input_dim=Din,
        hidden_dim=H,
        output_dim=Dout,
        max_depth=D,
        max_children=K,
        sibling_embed=True,
        use_pruning=False,
        branch_temperature=1.0,
    )
    # Sparse model with same state_dict
    sparse = BFSNet(
        input_dim=Din,
        hidden_dim=H,
        output_dim=Dout,
        max_depth=D,
        max_children=K,
        sibling_embed=True,
        use_pruning=False,
        branch_temperature=1.0,
    )
    sparse.load_state_dict(dense.state_dict())
    setattr(sparse, "sparse_execute", True)

    return dense, sparse, x


# --------------------------------------------------------------------------- #
#                                    Tests                                    #
# --------------------------------------------------------------------------- #

def test_logits_match_dense_vs_sparse():
    """Dense and sparse logits must be equal within tolerance."""
    dense, sparse, x = make_pair(B=16, D=3, K=2, H=32, Din=8, Dout=4)
    with torch.no_grad():
        ld = dense(x)
        ls = sparse(x)
    diff = (ld - ls).abs().max().item()
    assert diff < 1e-5, f"Logits mismatch: max Δ={diff:.2e}"


def test_trace_counts_match():
    """Spawned counts and active-mask sums should match between modes."""
    dense, sparse, x = make_pair(B=12, D=2, K=3, H=16, Din=10, Dout=5)
    with torch.no_grad():
        _, trace_d = dense(x, return_trace=True)
        _, trace_s = sparse(x, return_trace=True)

    for key in ["spawn_counts_sum", "active_mask_sums", "num_nodes_per_depth"]:
        td = trace_d[key].cpu().numpy()
        ts = trace_s[key].cpu().numpy()
        assert (td == ts).all(), f"Trace mismatch for {key}: dense={td}, sparse={ts}"


def test_early_exit_when_frontier_empties():
    """
    Model should early-exit when no nodes remain in frontier.
    Achieved by forcing max_children=0 (no expansion possible).
    """
    B, Din, Dout = 4, 6, 2
    dense = BFSNet(Din, 8, Dout, max_depth=3, max_children=0)
    sparse = BFSNet(Din, 8, Dout, max_depth=3, max_children=0)
    sparse.load_state_dict(dense.state_dict())
    setattr(sparse, "sparse_execute", True)

    x = torch.randn(B, Din)
    with torch.no_grad():
        _, trace_d = dense(x, return_trace=True)
        _, trace_s = sparse(x, return_trace=True)

    # Expect only depth-0 nodes (root), no spawns
    assert trace_d["num_nodes_per_depth"].max().item() == 1
    assert trace_s["num_nodes_per_depth"].max().item() == 1
    assert trace_d["spawn_counts_sum"].sum().item() == 0
    assert trace_s["spawn_counts_sum"].sum().item() == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
