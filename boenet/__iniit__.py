#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BoeNet: BFS-Inspired Language Model with REINFORCE Policy Gradients

Converted from BFSNet v2.0.0 (Vision) to BoeNet v1.0.0 (Language)

This package provides:
  - BoeNet: Language model with adaptive BFS tree expansion
  - CharTokenizer: Character-level tokenization
  - TiktokenWrapper: BPE tokenization (requires tiktoken)
  - Loss functions for REINFORCE policy gradients
  - Data utilities for Shakespeare and TinyStories datasets

The core insight: The BFS+REINFORCE algorithm is vector-agnostic.
Just swap input/output layers, keep everything else unchanged.

Quick Start
-----------
>>> from boenet.model import BoeNet
>>> from boenet.tokenizer import CharTokenizer
>>> from boenet.utils.data_utils import get_dataloaders
>>>
>>> # Load Shakespeare dataset
>>> train_loader, val_loader, vocab_size = get_dataloaders(
...     "shakespeare", batch_size=64, seq_len=128
... )
>>>
>>> # Create model
>>> model = BoeNet(
...     vocab_size=vocab_size,
...     embed_dim=64,
...     hidden_dim=128,
...     max_depth=2,
...     max_children=3,
...     greedy_threshold=0.42,
... )
>>>
>>> # Training loop
>>> for input_ids, labels in train_loader:
...     outputs, policy_loss, rewards, node_counts = model(
...         input_ids, num_rollouts=3, lambda_efficiency=0.05, labels=labels
...     )
...     # Compute loss, backprop, etc.

Author: BoeNet project (converted from BFSNet)
Version: 1.0.0
Date: 2025-12-22
"""

__version__ = "1.0.0"
__author__ = "BoeNet project"

from boenet.model import BoeNet
from boenet.tokenizer import CharTokenizer, get_tokenizer

__all__ = [
    "BoeNet",
    "CharTokenizer",
    "get_tokenizer",
]