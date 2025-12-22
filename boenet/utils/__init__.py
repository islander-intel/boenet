#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BoeNet Utilities Package

This package provides utility functions for BoeNet experiments:
  - data_utils: Dataset loading and preprocessing
  - gating: GrowthPolicyNet and pruning gates (unchanged from BFSNet)
  - sparse_utils: Sparse tensor operations (unchanged from BFSNet)
  - optim: Optimizer builders (unchanged from BFSNet)
  - schedules: Value schedulers (unchanged from BFSNet)
  - config_utils: YAML configuration loading (unchanged from BFSNet)
  - metrics: Evaluation metrics including perplexity
  - profiler: Performance profiling utilities

Author: BoeNet project
Version: 1.0.0
Date: 2025-12-22
"""

from boenet.utils.data_utils import (
    get_dataloaders,
    CharTokenizer,
    TextDataset,
    build_shakespeare_datasets,
    build_tinystories_datasets,
    set_seed,
    get_device,
)

__all__ = [
    "get_dataloaders",
    "CharTokenizer",
    "TextDataset",
    "build_shakespeare_datasets",
    "build_tinystories_datasets",
    "set_seed",
    "get_device",
]