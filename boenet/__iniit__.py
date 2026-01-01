#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BoeNet v2.0.0 - True BFS Language Model

A language model using true breadth-first search tree expansion
with REINFORCE policy gradients.

Key Features:
- True BFS level-by-level expansion
- Balanced binary tree guarantee
- O(log n) gradient paths
- Proper BFS indexing

Usage:
    from boenet import BoeNet
    
    model = BoeNet(
        vocab_size=256,
        embed_dim=64,
        hidden_dim=128,
        max_depth=3,
        max_children=2,
        greedy_threshold=0.5,
    )
"""

from boenet.model import (
    BoeNet,
    get_parent_idx,
    get_left_child_idx,
    get_right_child_idx,
    get_level,
    get_level_range,
    get_nodes_at_level,
    get_num_nodes_at_level,
    get_total_nodes_up_to_level,
    is_left_child,
    is_right_child,
    get_sibling_idx,
)

__version__ = "2.0.0"
__all__ = [
    "BoeNet",
    "get_parent_idx",
    "get_left_child_idx",
    "get_right_child_idx",
    "get_level",
    "get_level_range",
    "get_nodes_at_level",
    "get_num_nodes_at_level",
    "get_total_nodes_up_to_level",
    "is_left_child",
    "is_right_child",
    "get_sibling_idx",
]