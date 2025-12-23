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

Changelog
---------
v2.0.0 (2025-12-22):
  - BREAKING CHANGE: Updated imports for redesigned data_utils.py
  - REMOVED: build_shakespeare_datasets (replaced by load_shakespeare_from_github)
  - REMOVED: build_tinystories_datasets (replaced by load_huggingface_text_dataset)
  - ADDED: load_shakespeare_from_github - direct GitHub download
  - ADDED: load_huggingface_text_dataset - generic HuggingFace loader
  - ADDED: load_text_file - local text file loader

v1.0.0 (2025-12-22):
  - Initial BoeNet release with language model support

Author: BoeNet project
Version: 2.0.0
Date: 2025-12-22
"""

from boenet.utils.data_utils import (
    # Unified accessor (primary entry point)
    get_dataloaders,
    
    # Tokenization
    CharTokenizer,
    TextDataset,
    
    # Language dataset loaders (NEW in v2.0.0)
    load_shakespeare_from_github,
    load_huggingface_text_dataset,
    load_text_file,
    
    # Utilities
    set_seed,
    get_device,
    SplitConfig,
)

__all__ = [
    # Unified accessor (primary entry point)
    "get_dataloaders",
    
    # Tokenization
    "CharTokenizer",
    "TextDataset",
    
    # Language dataset loaders (NEW in v2.0.0)
    "load_shakespeare_from_github",
    "load_huggingface_text_dataset",
    "load_text_file",
    
    # Utilities
    "set_seed",
    "get_device",
    "SplitConfig",
]