#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
boenet/utils/__init__.py (v2.0.0)

Unified public API for the BoeNet utilities package.

This module re-exports all public classes, functions, and constants from the
utils sub-modules so users can write:

    from boenet.utils import CharTokenizer, get_dataloaders, GrowthPolicyNet

Instead of:

    from boenet.utils.tokenizer import CharTokenizer
    from boenet.utils.data_utils import get_dataloaders
    from boenet.utils.gating import GrowthPolicyNet

v2.0.0 Changes (2025-12-22)
--------------------------
REMOVED imports (no longer exist in data_utils.py):
  - build_shakespeare_datasets
  - build_tinystories_datasets

ADDED imports (new in data_utils.py v2.0.0):
  - load_shakespeare_from_github
  - load_huggingface_text_dataset
  - load_text_file
  - SplitConfig
  - HUGGINGFACE_DATASETS

Author: BoeNet project
Version: 2.0.0
Date: 2025-12-22
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Data utilities (v2.0.0 - updated imports)
# ---------------------------------------------------------------------------
from boenet.utils.data_utils import (
    # Core loaders
    get_dataloaders,
    load_huggingface_text_dataset,
    load_shakespeare_from_github,
    load_text_file,
    # Configuration
    SplitConfig,
    HUGGINGFACE_DATASETS,
    # Tokenizer
    CharTokenizer,
    # Utilities
    set_seed,
)

# ---------------------------------------------------------------------------
# Gating modules (unchanged from v1.0.0)
# ---------------------------------------------------------------------------
from boenet.utils.gating import (
    GrowthPolicyNet,
    ScalarGate,
    HardConcreteGate,
    ThresholdPruner,
)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
__all__ = [
    # Data utilities (v2.0.0)
    "get_dataloaders",
    "load_huggingface_text_dataset",
    "load_shakespeare_from_github",
    "load_text_file",
    "SplitConfig",
    "HUGGINGFACE_DATASETS",
    "CharTokenizer",
    "set_seed",
    # Gating modules
    "GrowthPolicyNet",
    "ScalarGate",
    "HardConcreteGate",
    "ThresholdPruner",
]

__version__ = "2.0.0"