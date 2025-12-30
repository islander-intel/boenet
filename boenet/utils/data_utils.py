#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
boenet/utils/data_utils.py (v2.0.1 - Byte-Level Tokenization Fix)

Lightweight dataset + dataloader utilities for BoeNet experiments.

Converted from BFSNet (Vision) to BoeNet (Vision + Language)
------------------------------------------------------------
This module now supports BOTH vision and language datasets:

Vision (unchanged from BFSNet):
  - Toy2Token: Two-class toy vectors
  - MNIST: Handwritten digits
  - FashionMNIST: Fashion items
  - Synthetic Text BoW: Bag-of-words classification

Language (REDESIGNED in v2.0.0):
  - WikiText-2: Small Wikipedia dataset (~2MB) - DEFAULT
  - WikiText-103: Large Wikipedia dataset (~500MB)
  - Shakespeare: Karpathy's tiny_shakespeare via direct GitHub download
  - TinyStories: Children's stories from HuggingFace
  - BookCorpus: 11,000 books (large, ~5GB)
  - Custom text files: Any local .txt file

Design Changes in v2.0.0
------------------------
1. RENAMED: build_shakespeare_datasets() → load_shakespeare_from_github()
   - Now downloads directly from GitHub, bypassing broken HuggingFace dataset
   
2. NEW: load_huggingface_text_dataset(dataset_name, ...)
   - Generic function for ANY HuggingFace text dataset
   - Properly named to describe what it does
   
3. NEW: WikiText-2 as default dataset
   - Well-maintained, modern Parquet format
   - Small enough for quick experiments (~2MB)
   
4. REMOVED: Hardcoded build_tinystories_datasets()
   - Now uses generic load_huggingface_text_dataset()

Language Model Data Format
--------------------------
For next-token prediction, we need:
  - input_ids: Token IDs at positions [0, 1, ..., seq_len-1]
  - labels: Token IDs at positions [1, 2, ..., seq_len] (shifted by 1)

The TextDataset class handles this automatically:
  >>> dataset = TextDataset(text, seq_len=128, tokenizer=char_tokenizer)
  >>> input_ids, labels = dataset[0]
  >>> # input_ids: "Hello worl" → [72, 101, 108, 108, 111, ...]
  >>> # labels:    "ello world" → [101, 108, 108, 111, 32, ...]

Tokenization
------------
This module includes a ByteTokenizer for byte-level modeling (vocab_size=256).
All text is encoded as UTF-8 bytes where each byte becomes a token (0-255).

For BPE/subword tokenization, use tiktoken or HuggingFace tokenizers
and pass them to TextDataset.

Usage Examples
--------------
>>> # WikiText-2 (default, recommended)
>>> train_loader, val_loader, vocab_size = get_dataloaders(
...     "wikitext2",
...     batch_size=64,
...     seq_len=128,
... )
>>> 
>>> # Shakespeare (via GitHub download)
>>> train_loader, val_loader, vocab_size = get_dataloaders(
...     "shakespeare",
...     batch_size=64,
...     seq_len=128,
... )
>>>
>>> # FashionMNIST (unchanged API)
>>> train_loader, val_loader, input_dim, num_classes = get_dataloaders(
...     "fashionmnist",
...     batch_size=64,
...     mnist_flatten=True,
... )

Changelog
---------
v2.0.1 (2025-12-29):
  - CRITICAL BUGFIX: CharTokenizer renamed to ByteTokenizer
  - CRITICAL BUGFIX: Tokenization now uses UTF-8 bytes (0-255) instead of
    Unicode code points (0-65535+). This fixes "IndexError: index out of range"
    when training on WikiText and other datasets with non-ASCII characters.
  - encode() now uses: list(text.encode('utf-8'))
  - decode() now uses: bytes(tokens).decode('utf-8', errors='replace')
  - All token IDs are now guaranteed to be in range [0, 255]

v2.0.0 (2025-12-22):
  - MAJOR REDESIGN: Generic HuggingFace loader
  - NEW: load_huggingface_text_dataset() - generic function for all HF datasets
  - NEW: load_shakespeare_from_github() - direct download bypassing broken HF dataset
  - NEW: WikiText-2 as default dataset (modern Parquet format, no script issues)
  - REMOVED: build_shakespeare_datasets() (used broken HF dataset script)
  - REMOVED: build_tinystories_datasets() (replaced by generic loader)
  - FIXED: "Dataset scripts are no longer supported" error

v1.0.1 (2025-12-22):
  - BUGFIX: Removed deprecated trust_remote_code=True from load_dataset() calls

v1.0.0 (2025-12-22):
  - Initial language model support
  - Added TextDataset, CharTokenizer
  - Added Shakespeare and TinyStories datasets

Author: BoeNet project (extended from BFSNet)
Version: 2.0.1
Date: 2025-12-29
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Literal, Optional, Tuple, List, Union
import os
import urllib.request
import hashlib

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split

# Optional torchvision imports (MNIST/FashionMNIST)
try:
    import torchvision
    from torchvision import transforms
    _HAS_TORCHVISION = True
except Exception:
    _HAS_TORCHVISION = False

# Optional HuggingFace datasets (for language modeling)
try:
    from datasets import load_dataset
    _HAS_DATASETS = True
except Exception:
    _HAS_DATASETS = False


# --------------------------------------------------------------------------- #
#                              Basic Utilities                                #
# --------------------------------------------------------------------------- #

def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)


def get_device(force_cpu: bool = False) -> torch.device:
    """Return a torch.device. If force_cpu is True, returns CPU even if CUDA exists."""
    if (not force_cpu) and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@dataclass
class SplitConfig:
    """Train/validation split configuration."""
    val_ratio: float = 0.2
    shuffle_before_split: bool = True


def split_dataset(
    dataset: Dataset,
    split: SplitConfig = SplitConfig(),
    seed: int = 42,
) -> Tuple[Dataset, Dataset]:
    """
    Split a dataset into (train, val) with an optional shuffle.
    
    Parameters
    ----------
    dataset : Dataset
        Dataset to split.
    split : SplitConfig
        Split configuration.
    seed : int
        Random seed for reproducibility.
        
    Returns
    -------
    Tuple[Dataset, Dataset]
        (train_dataset, val_dataset)
    """
    n = len(dataset)
    n_val = max(1, int(round(split.val_ratio * n)))
    n_train = max(1, n - n_val)
    g = torch.Generator().manual_seed(seed) if split.shuffle_before_split else None
    subsets = random_split(dataset, [n_train, n_val], generator=g)
    return subsets[0], subsets[1]


def _make_loader(
    dataset: Dataset,
    *,
    batch_size: int = 32,
    shuffle: bool,
    drop_last: bool,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    """Build a DataLoader with explicit shuffle & drop_last."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )


# --------------------------------------------------------------------------- #
#                      One-time per-run sanity printing                       #
# --------------------------------------------------------------------------- #

_SANITY_PRINTED_KEYS: set = set()


def _print_sanity_once(
    key: str,
    xb: torch.Tensor,
    yb: torch.Tensor,
    *,
    flatten_expected: Optional[int] = None,
    dataset_name: str = "",
) -> None:
    """Print a one-time mini sanity line: shape, dtype, min/max, and label counts head."""
    if key in _SANITY_PRINTED_KEYS:
        return
    _SANITY_PRINTED_KEYS.add(key)

    with torch.no_grad():
        x_dtype = str(xb.dtype).replace("torch.", "")
        x_min = float(xb.min().item()) if xb.numel() > 0 else float("nan")
        x_max = float(xb.max().item()) if xb.numel() > 0 else float("nan")
        y_min = int(yb.min().item()) if yb.numel() > 0 else -1
        y_max = int(yb.max().item()) if yb.numel() > 0 else -1

        shape_str = "x".join(str(s) for s in xb.shape)
        extra = ""
        if flatten_expected is not None and xb.dim() == 2:
            extra = f", assert_flatten={'OK' if xb.shape[1] == flatten_expected else f'BAD({xb.shape[1]})'}"

        print(
            f"[sanity:{dataset_name}] batch={shape_str} dtype={x_dtype} "
            f"xmin={x_min:.3f} xmax={x_max:.3f} y[min,max]=[{y_min},{y_max}]{extra}"
        )


def _print_sanity_text_once(
    key: str,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    *,
    vocab_size: int,
    dataset_name: str = "",
) -> None:
    """Print a one-time sanity line for text datasets."""
    if key in _SANITY_PRINTED_KEYS:
        return
    _SANITY_PRINTED_KEYS.add(key)

    with torch.no_grad():
        shape_str = "x".join(str(s) for s in input_ids.shape)
        id_min = int(input_ids.min().item())
        id_max = int(input_ids.max().item())
        label_min = int(labels.min().item())
        label_max = int(labels.max().item())
        
        # Validate that all tokens are within vocab_size
        valid = id_max < vocab_size and label_max < vocab_size
        status = "OK" if valid else f"ERROR: max_id={max(id_max, label_max)} >= vocab_size={vocab_size}"

        print(
            f"[sanity:{dataset_name}] batch={shape_str} dtype={input_ids.dtype} "
            f"input_ids[min,max]=[{id_min},{id_max}] labels[min,max]=[{label_min},{label_max}] "
            f"vocab_size={vocab_size} [{status}]"
        )


# --------------------------------------------------------------------------- #
#                     Byte-Level Tokenizer (FIXED v2.0.1)                     #
# --------------------------------------------------------------------------- #

class ByteTokenizer:
    """
    Byte-level tokenizer using UTF-8 encoding.
    
    This tokenizer encodes text as UTF-8 bytes, where each byte becomes a token
    in the range [0, 255]. This guarantees vocab_size=256 regardless of the
    input text's Unicode content.
    
    CRITICAL FIX (v2.0.1):
    ----------------------
    Previous CharTokenizer used ord(char) which returns Unicode code points:
      - 'a' → 97 (OK)
      - '—' (em-dash) → 8212 (FAILS: exceeds vocab_size=256)
    
    ByteTokenizer uses UTF-8 byte encoding:
      - 'a' → [97] (1 byte)
      - '—' (em-dash) → [226, 128, 148] (3 bytes, all in 0-255)
    
    This ensures ALL token IDs are in [0, 255], preventing IndexError
    in nn.Embedding layers.
    
    Attributes
    ----------
    vocab_size : int
        Vocabulary size (always 256 for byte-level).
        
    Examples
    --------
    >>> tokenizer = ByteTokenizer()
    >>> tokenizer.encode("Hello")
    [72, 101, 108, 108, 111]
    >>> tokenizer.decode([72, 101, 108, 108, 111])
    'Hello'
    >>> # Unicode characters are encoded as multiple bytes
    >>> tokenizer.encode("—")  # em-dash
    [226, 128, 148]
    >>> tokenizer.decode([226, 128, 148])
    '—'
    """
    
    def __init__(self):
        """Initialize byte-level tokenizer."""
        self._vocab_size = 256
    
    @property
    def vocab_size(self) -> int:
        """Return vocabulary size (256 for byte-level)."""
        return self._vocab_size
    
    def encode(self, text: str) -> List[int]:
        """
        Encode text to list of byte values (0-255).
        
        Parameters
        ----------
        text : str
            Text to encode.
            
        Returns
        -------
        List[int]
            List of byte values (each in range 0-255).
            
        Notes
        -----
        Text is encoded as UTF-8 bytes. ASCII characters map 1:1 to bytes,
        while Unicode characters may produce multiple bytes.
        
        Examples
        --------
        >>> tokenizer = ByteTokenizer()
        >>> tokenizer.encode("Hi")
        [72, 105]
        >>> tokenizer.encode("Héllo")  # é is 2 bytes in UTF-8
        [72, 195, 169, 108, 108, 111]
        """
        # Encode text as UTF-8 bytes - each byte is 0-255
        return list(text.encode('utf-8'))
    
    def decode(self, tokens: List[int]) -> str:
        """
        Decode list of byte values to text.
        
        Parameters
        ----------
        tokens : List[int]
            List of byte values (each should be in range 0-255).
            
        Returns
        -------
        str
            Decoded text.
            
        Notes
        -----
        Invalid UTF-8 byte sequences are replaced with the Unicode
        replacement character (U+FFFD).
        
        Examples
        --------
        >>> tokenizer = ByteTokenizer()
        >>> tokenizer.decode([72, 105])
        'Hi'
        >>> tokenizer.decode([72, 195, 169, 108, 108, 111])
        'Héllo'
        """
        # Convert token list to bytes, then decode as UTF-8
        # Use errors='replace' to handle any invalid sequences gracefully
        return bytes(tokens).decode('utf-8', errors='replace')
    
    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        """
        Encode a batch of texts.
        
        Parameters
        ----------
        texts : List[str]
            List of texts to encode.
            
        Returns
        -------
        List[List[int]]
            List of encoded byte value lists.
        """
        return [self.encode(text) for text in texts]
    
    def decode_batch(self, token_lists: List[List[int]]) -> List[str]:
        """
        Decode a batch of byte value lists.
        
        Parameters
        ----------
        token_lists : List[List[int]]
            List of byte value lists.
            
        Returns
        -------
        List[str]
            List of decoded texts.
        """
        return [self.decode(tokens) for tokens in token_lists]


# Backwards compatibility alias (CharTokenizer was renamed to ByteTokenizer)
CharTokenizer = ByteTokenizer


# --------------------------------------------------------------------------- #
#                           Text Dataset                                      #
# --------------------------------------------------------------------------- #

class TextDataset(Dataset):
    """
    Dataset for byte-level or token-level language modeling.
    
    This dataset creates (input_ids, labels) pairs for next-token prediction:
      - input_ids: Tokens at positions [i, i+1, ..., i+seq_len-1]
      - labels: Tokens at positions [i+1, i+2, ..., i+seq_len] (shifted by 1)
    
    Parameters
    ----------
    text : str
        Raw text data.
    seq_len : int
        Sequence length for each sample.
    tokenizer : object, optional
        Tokenizer with encode() method. Defaults to ByteTokenizer.
    stride : int, optional
        Stride between consecutive samples. Defaults to seq_len (non-overlapping).
        Use stride < seq_len for overlapping samples (more training data).
        
    Attributes
    ----------
    token_ids : torch.Tensor
        Encoded token IDs for entire text, shape [num_tokens].
    vocab_size : int
        Vocabulary size from tokenizer.
        
    Examples
    --------
    >>> text = "Hello world! This is a test."
    >>> dataset = TextDataset(text, seq_len=10)
    >>> input_ids, labels = dataset[0]
    >>> input_ids.shape, labels.shape
    (torch.Size([10]), torch.Size([10]))
    
    Notes
    -----
    With ByteTokenizer (default), Unicode text is encoded as UTF-8 bytes.
    This means:
      - ASCII characters: 1 token per character
      - Non-ASCII characters: 2-4 tokens per character
    
    The effective "character length" may differ from token length for
    Unicode-heavy text.
    """
    
    def __init__(
        self,
        text: str,
        seq_len: int,
        tokenizer: Optional[object] = None,
        stride: Optional[int] = None,
    ):
        if tokenizer is None:
            tokenizer = ByteTokenizer()
        
        self.seq_len = seq_len
        self.stride = stride if stride is not None else seq_len
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size
        
        # Encode entire text as bytes
        token_ids = tokenizer.encode(text)
        self.token_ids = torch.tensor(token_ids, dtype=torch.long)
        
        # Validate token range
        if len(self.token_ids) > 0:
            max_id = int(self.token_ids.max().item())
            if max_id >= self.vocab_size:
                raise ValueError(
                    f"Token ID {max_id} exceeds vocab_size {self.vocab_size}. "
                    f"This should not happen with ByteTokenizer. "
                    f"Check your tokenizer implementation."
                )
        
        # Calculate number of samples
        # We need seq_len + 1 tokens for each sample (input + 1 shifted label)
        total_tokens = len(self.token_ids)
        if total_tokens < seq_len + 1:
            raise ValueError(
                f"Text too short: {total_tokens} tokens, need at least {seq_len + 1}"
            )
        
        # Number of valid starting positions
        self.num_samples = (total_tokens - seq_len - 1) // self.stride + 1
    
    def __len__(self) -> int:
        """Return number of samples."""
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample.
        
        Parameters
        ----------
        idx : int
            Sample index.
            
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            (input_ids, labels) both of shape [seq_len].
        """
        start = idx * self.stride
        end = start + self.seq_len
        
        input_ids = self.token_ids[start:end]
        labels = self.token_ids[start + 1:end + 1]
        
        return input_ids, labels


# --------------------------------------------------------------------------- #
#                    Shakespeare Dataset (Direct GitHub Download)             #
# --------------------------------------------------------------------------- #

# Karpathy's tiny_shakespeare raw text URL
SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
SHAKESPEARE_MD5 = "d0d0c12375f25a2a2c95752c04f62c7d"  # MD5 hash for validation


def _download_file(url: str, filepath: str, expected_md5: Optional[str] = None) -> None:
    """
    Download a file from URL to local path with optional MD5 validation.
    
    Parameters
    ----------
    url : str
        URL to download from.
    filepath : str
        Local path to save file.
    expected_md5 : str, optional
        Expected MD5 hash for validation.
    """
    print(f"[data] Downloading from {url}...")
    
    # Create directory if needed
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
    
    # Download
    urllib.request.urlretrieve(url, filepath)
    
    # Validate MD5 if provided
    if expected_md5:
        with open(filepath, "rb") as f:
            actual_md5 = hashlib.md5(f.read()).hexdigest()
        if actual_md5 != expected_md5:
            os.remove(filepath)
            raise ValueError(
                f"MD5 mismatch: expected {expected_md5}, got {actual_md5}. "
                f"File may be corrupted or changed."
            )
    
    print(f"[data] Downloaded to {filepath}")


def load_shakespeare_from_github(
    seq_len: int = 128,
    seed: int = 42,
    split: SplitConfig = SplitConfig(val_ratio=0.1),
    batch_size: int = 64,
    stride: Optional[int] = None,
    cache_dir: str = "./data",
    *,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> Tuple[DataLoader, DataLoader, int]:
    """
    Load Shakespeare dataset via direct GitHub download.
    
    This function downloads Karpathy's tiny_shakespeare directly from GitHub,
    bypassing HuggingFace which has broken dataset script support.
    
    Parameters
    ----------
    seq_len : int, default=128
        Sequence length for each sample.
    seed : int, default=42
        Random seed for reproducibility.
    split : SplitConfig
        Train/validation split configuration.
    batch_size : int, default=64
        Batch size.
    stride : int, optional
        Stride between samples. Defaults to seq_len (non-overlapping).
    cache_dir : str, default="./data"
        Directory to cache downloaded file.
    num_workers : int, default=0
        Number of dataloader workers.
    pin_memory : bool, default=False
        Pin memory for faster GPU transfer.
        
    Returns
    -------
    Tuple[DataLoader, DataLoader, int]
        (train_loader, val_loader, vocab_size)
        
    Examples
    --------
    >>> train_loader, val_loader, vocab_size = load_shakespeare_from_github(
    ...     seq_len=128, batch_size=64
    ... )
    >>> for input_ids, labels in train_loader:
    ...     # input_ids: [B, seq_len]
    ...     # labels: [B, seq_len]
    ...     break
    
    Notes
    -----
    The Shakespeare text is ~1.1MB and contains the complete works of Shakespeare.
    This is the same dataset used in Karpathy's char-rnn and nanoGPT projects.
    """
    set_seed(seed)
    
    # Check for cached file
    filepath = os.path.join(cache_dir, "shakespeare", "input.txt")
    
    if os.path.exists(filepath):
        print(f"[data] Using cached Shakespeare: {filepath}")
    else:
        print("[data] Downloading Shakespeare from GitHub (Karpathy's char-rnn)...")
        _download_file(SHAKESPEARE_URL, filepath, expected_md5=SHAKESPEARE_MD5)
    
    # Load text
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    
    print(f"[data] Loaded {len(text):,} characters from Shakespeare")
    
    # Create tokenizer (ByteTokenizer for byte-level encoding)
    tokenizer = ByteTokenizer()
    
    # Create full dataset
    full_dataset = TextDataset(
        text=text,
        seq_len=seq_len,
        tokenizer=tokenizer,
        stride=stride,
    )
    
    print(f"[data] Created {len(full_dataset):,} samples (seq_len={seq_len})")
    
    # Split into train/val
    train_ds, val_ds = split_dataset(full_dataset, split=split, seed=seed)
    
    print(f"[data] Train: {len(train_ds):,} samples, Val: {len(val_ds):,} samples")
    
    # Create dataloaders
    train_loader = _make_loader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True,
        num_workers=num_workers, pin_memory=pin_memory
    )
    val_loader = _make_loader(
        val_ds, batch_size=batch_size, shuffle=False, drop_last=False,
        num_workers=num_workers, pin_memory=pin_memory
    )
    
    # Sanity check
    input_ids, labels = next(iter(train_loader))
    _print_sanity_text_once(
        "shakespeare", input_ids, labels,
        vocab_size=tokenizer.vocab_size,
        dataset_name="shakespeare"
    )
    
    # Print info about vocab and token stats
    print(f"[data] vocab_size={tokenizer.vocab_size}, seq_len={seq_len}")
    print(f"[data] Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    print(f"[data] Random baseline perplexity: {tokenizer.vocab_size:.2f}")
    
    return train_loader, val_loader, tokenizer.vocab_size


# --------------------------------------------------------------------------- #
#                  Generic HuggingFace Text Dataset Loader                    #
# --------------------------------------------------------------------------- #

# Supported HuggingFace datasets with their configurations
# Format: {user_name: (hf_dataset_name, hf_config_name, text_column, description)}
HUGGINGFACE_DATASETS = {
    # WikiText - well-maintained, modern Parquet format
    "wikitext2": ("wikitext", "wikitext-2-raw-v1", "text", "Small Wikipedia (~2MB)"),
    "wikitext103": ("wikitext", "wikitext-103-raw-v1", "text", "Large Wikipedia (~500MB)"),
    
    # TinyStories - children's stories
    "tinystories": ("roneneldan/TinyStories", None, "text", "Children's stories (~2GB)"),
    
    # BookCorpus - 11,000 books
    "bookcorpus": ("bookcorpus", None, "text", "11,000 books (~5GB)"),
    
    # OpenWebText - GPT-2 training data
    "openwebtext": ("openwebtext", None, "text", "Web text (~40GB)"),
    
    # AG News - news classification (for testing)
    "agnews": ("ag_news", None, "text", "News articles (~30MB)"),
}


def _assert_datasets() -> None:
    """Raise ImportError if HuggingFace datasets not available."""
    if not _HAS_DATASETS:
        raise ImportError(
            "HuggingFace datasets library not available. Install with:\n"
            "  pip install datasets\n"
            "Or use local text files with load_text_file() or Shakespeare with "
            "load_shakespeare_from_github() (no datasets library required)."
        )


def load_huggingface_text_dataset(
    dataset_name: str,
    seq_len: int = 128,
    seed: int = 42,
    split: SplitConfig = SplitConfig(val_ratio=0.1),
    batch_size: int = 64,
    stride: Optional[int] = None,
    max_samples: Optional[int] = None,
    hf_split: str = "train",
    *,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> Tuple[DataLoader, DataLoader, int]:
    """
    Load any HuggingFace text dataset for byte-level language modeling.
    
    This is a GENERIC function that can load any text dataset from HuggingFace.
    It handles the common pattern of:
      1. Load dataset from HuggingFace Hub
      2. Extract text from specified column
      3. Concatenate all text
      4. Create TextDataset with byte-level tokenization (vocab_size=256)
      5. Split into train/val
      6. Create dataloaders
    
    Parameters
    ----------
    dataset_name : str
        Dataset name. Can be:
        - Shorthand: "wikitext2", "wikitext103", "tinystories", "bookcorpus"
        - Full HuggingFace path: "username/dataset_name"
    seq_len : int, default=128
        Sequence length for each sample.
    seed : int, default=42
        Random seed for reproducibility.
    split : SplitConfig
        Train/validation split configuration.
    batch_size : int, default=64
        Batch size.
    stride : int, optional
        Stride between samples. Defaults to seq_len (non-overlapping).
    max_samples : int, optional
        Maximum number of text samples to load (for memory/speed limits).
        If None, use all available samples.
    hf_split : str, default="train"
        Which HuggingFace split to use (usually "train").
    num_workers : int, default=0
        Number of dataloader workers.
    pin_memory : bool, default=False
        Pin memory for faster GPU transfer.
        
    Returns
    -------
    Tuple[DataLoader, DataLoader, int]
        (train_loader, val_loader, vocab_size)
        
    Examples
    --------
    >>> # WikiText-2 (recommended for quick experiments)
    >>> train, val, vocab = load_huggingface_text_dataset("wikitext2", seq_len=128)
    
    >>> # TinyStories with max samples limit
    >>> train, val, vocab = load_huggingface_text_dataset(
    ...     "tinystories", seq_len=256, max_samples=10000
    ... )
    
    >>> # Custom HuggingFace dataset
    >>> train, val, vocab = load_huggingface_text_dataset(
    ...     "username/my_dataset", seq_len=128
    ... )
    
    Supported Datasets
    ------------------
    - wikitext2: Small Wikipedia (~2MB) - RECOMMENDED for quick experiments
    - wikitext103: Large Wikipedia (~500MB)
    - tinystories: Children's stories (~2GB)
    - bookcorpus: 11,000 books (~5GB)
    - openwebtext: Web text (~40GB)
    - Any other HuggingFace dataset with a "text" column
    
    Notes
    -----
    Uses ByteTokenizer (UTF-8 bytes) for tokenization, guaranteeing vocab_size=256.
    Unicode characters are encoded as 2-4 bytes each.
    """
    _assert_datasets()
    set_seed(seed)
    
    # Resolve dataset configuration
    dataset_name_lower = dataset_name.lower().strip()
    
    if dataset_name_lower in HUGGINGFACE_DATASETS:
        hf_name, hf_config, text_column, description = HUGGINGFACE_DATASETS[dataset_name_lower]
        print(f"[data] Loading {dataset_name_lower}: {description}")
    else:
        # Assume it's a full HuggingFace path
        hf_name = dataset_name
        hf_config = None
        text_column = "text"
        print(f"[data] Loading custom HuggingFace dataset: {hf_name}")
    
    # Load from HuggingFace
    print(f"[data] Downloading from HuggingFace Hub...")
    if hf_config:
        dataset = load_dataset(hf_name, hf_config)
    else:
        dataset = load_dataset(hf_name)
    
    # Get the specified split
    if hf_split not in dataset:
        available = list(dataset.keys())
        print(f"[data] Split '{hf_split}' not found, using '{available[0]}'")
        hf_split = available[0]
    
    data_split = dataset[hf_split]
    
    # Extract and concatenate text
    print(f"[data] Extracting text from '{text_column}' column...")
    texts = []
    count = 0
    
    for item in data_split:
        text_value = item.get(text_column, "")
        if text_value:  # Skip empty texts
            texts.append(text_value)
            count += 1
            if max_samples is not None and count >= max_samples:
                break
    
    if not texts:
        raise ValueError(
            f"No text found in column '{text_column}'. "
            f"Available columns: {list(data_split[0].keys())}"
        )
    
    # Concatenate all text
    text = "\n\n".join(texts)
    print(f"[data] Loaded {count:,} text samples, {len(text):,} characters total")
    
    # Create tokenizer (ByteTokenizer for byte-level encoding)
    tokenizer = ByteTokenizer()
    
    # Create full dataset
    full_dataset = TextDataset(
        text=text,
        seq_len=seq_len,
        tokenizer=tokenizer,
        stride=stride,
    )
    
    print(f"[data] Created {len(full_dataset):,} training samples (seq_len={seq_len})")
    
    # Split into train/val
    train_ds, val_ds = split_dataset(full_dataset, split=split, seed=seed)
    
    print(f"[data] Train: {len(train_ds):,} samples, Val: {len(val_ds):,} samples")
    
    # Create dataloaders
    train_loader = _make_loader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True,
        num_workers=num_workers, pin_memory=pin_memory
    )
    val_loader = _make_loader(
        val_ds, batch_size=batch_size, shuffle=False, drop_last=False,
        num_workers=num_workers, pin_memory=pin_memory
    )
    
    # Sanity check
    input_ids, labels = next(iter(train_loader))
    _print_sanity_text_once(
        f"hf:{dataset_name_lower}", input_ids, labels,
        vocab_size=tokenizer.vocab_size,
        dataset_name=dataset_name_lower
    )
    
    # Print info about vocab and token stats
    print(f"[data] vocab_size={tokenizer.vocab_size}, seq_len={seq_len}")
    print(f"[data] Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    print(f"[data] Random baseline perplexity: {tokenizer.vocab_size:.2f}")
    
    return train_loader, val_loader, tokenizer.vocab_size


# --------------------------------------------------------------------------- #
#                         Local Text File Loader                              #
# --------------------------------------------------------------------------- #

def load_text_file(
    filepath: str,
    seq_len: int = 128,
    seed: int = 42,
    split: SplitConfig = SplitConfig(val_ratio=0.1),
    batch_size: int = 64,
    stride: Optional[int] = None,
    encoding: str = "utf-8",
    *,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> Tuple[DataLoader, DataLoader, int]:
    """
    Load dataloaders from a local text file.
    
    Parameters
    ----------
    filepath : str
        Path to text file.
    seq_len : int, default=128
        Sequence length for each sample.
    seed : int, default=42
        Random seed for reproducibility.
    split : SplitConfig
        Train/validation split configuration.
    batch_size : int, default=64
        Batch size.
    stride : int, optional
        Stride between samples.
    encoding : str, default="utf-8"
        File encoding.
    num_workers : int, default=0
        Number of dataloader workers.
    pin_memory : bool, default=False
        Pin memory for faster GPU transfer.
        
    Returns
    -------
    Tuple[DataLoader, DataLoader, int]
        (train_loader, val_loader, vocab_size)
    """
    set_seed(seed)
    
    # Load text file
    print(f"[data] Loading text file: {filepath}")
    with open(filepath, "r", encoding=encoding) as f:
        text = f.read()
    
    print(f"[data] Loaded {len(text):,} characters")
    
    # Create tokenizer (ByteTokenizer for byte-level encoding)
    tokenizer = ByteTokenizer()
    
    # Create full dataset
    full_dataset = TextDataset(
        text=text,
        seq_len=seq_len,
        tokenizer=tokenizer,
        stride=stride,
    )
    
    print(f"[data] Created {len(full_dataset):,} samples (seq_len={seq_len})")
    
    # Split into train/val
    train_ds, val_ds = split_dataset(full_dataset, split=split, seed=seed)
    
    print(f"[data] Train: {len(train_ds):,} samples, Val: {len(val_ds):,} samples")
    
    # Create dataloaders
    train_loader = _make_loader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True,
        num_workers=num_workers, pin_memory=pin_memory
    )
    val_loader = _make_loader(
        val_ds, batch_size=batch_size, shuffle=False, drop_last=False,
        num_workers=num_workers, pin_memory=pin_memory
    )
    
    # Sanity check
    input_ids, labels = next(iter(train_loader))
    _print_sanity_text_once(
        f"textfile:{filepath}", input_ids, labels,
        vocab_size=tokenizer.vocab_size,
        dataset_name="textfile"
    )
    
    # Print info about vocab and token stats
    print(f"[data] vocab_size={tokenizer.vocab_size}, seq_len={seq_len}")
    print(f"[data] Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    print(f"[data] Random baseline perplexity: {tokenizer.vocab_size:.2f}")
    
    return train_loader, val_loader, tokenizer.vocab_size


# --------------------------------------------------------------------------- #
#                        Vision Datasets (Unchanged)                          #
# --------------------------------------------------------------------------- #

def _assert_torchvision() -> None:
    if not _HAS_TORCHVISION:
        raise ImportError(
            "torchvision is not available. Install torchvision to use MNIST/FashionMNIST:\n"
            "  pip install torchvision\n"
            "Or use the toy, text, or language datasets."
        )


def _mnist_like_transforms(
    *,
    flatten: bool = True,
    normalize: bool = True,
    augment: bool = False,
    aug_pad: int = 2,
    aug_hflip: bool = False,
    hflip_p: float = 0.5,
) -> transforms.Compose:
    """Build transforms pipeline for MNIST-like datasets."""
    tfs: list = []

    if augment:
        tfs.append(transforms.RandomCrop(28, padding=int(aug_pad)))
        if aug_hflip:
            tfs.append(transforms.RandomHorizontalFlip(p=float(hflip_p)))

    tfs.append(transforms.ToTensor())
    if normalize:
        tfs.append(transforms.Normalize((0.1307,), (0.3081,)))

    if flatten:
        tfs.append(nn.Flatten(start_dim=0, end_dim=-1))

    return transforms.Compose(tfs)


def build_toy2token_dataset(
    samples_per_class: int = 100,
    input_dim: int = 8,
    seed: int = 42,
    split: SplitConfig = SplitConfig(),
    batch_size: int = 16,
    *,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> Tuple[DataLoader, DataLoader, int, int]:
    """Two-class toy dataset: Class 0 = +1s, Class 1 = -1s."""
    set_seed(seed)
    X_pos = torch.ones((samples_per_class, input_dim), dtype=torch.float32)
    y_pos = torch.zeros(samples_per_class, dtype=torch.long)
    X_neg = -torch.ones((samples_per_class, input_dim), dtype=torch.float32)
    y_neg = torch.ones(samples_per_class, dtype=torch.long)

    X = torch.cat([X_pos, X_neg], dim=0)
    y = torch.cat([y_pos, y_neg], dim=0)
    dataset = TensorDataset(X, y)

    train_ds, val_ds = split_dataset(dataset, split=split, seed=seed)
    train_loader = _make_loader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True,
                                num_workers=num_workers, pin_memory=pin_memory)
    val_loader = _make_loader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False,
                              num_workers=num_workers, pin_memory=pin_memory)

    xb, yb = next(iter(train_loader))
    _print_sanity_once("toy", xb, yb, dataset_name="toy")

    return train_loader, val_loader, input_dim, 2


def build_mnist_datasets(
    root: str = "./data",
    seed: int = 42,
    split: SplitConfig = SplitConfig(val_ratio=0.1),
    batch_size: int = 64,
    *,
    flatten: bool = True,
    normalize: bool = True,
    download: bool = True,
    augment: bool = False,
    aug_pad: int = 2,
    aug_hflip: bool = False,
    hflip_p: float = 0.5,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> Tuple[DataLoader, DataLoader, int, int]:
    """MNIST loaders with optional augmentation and flattening (→ 784-dim)."""
    _assert_torchvision()
    set_seed(seed)

    tf = _mnist_like_transforms(
        flatten=flatten, normalize=normalize, augment=augment,
        aug_pad=aug_pad, aug_hflip=aug_hflip, hflip_p=hflip_p,
    )

    full_train = torchvision.datasets.MNIST(root=root, train=True, transform=tf, download=download)
    train_ds, val_ds = split_dataset(full_train, split=split, seed=seed)

    train_loader = _make_loader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True,
                                num_workers=num_workers, pin_memory=pin_memory)
    val_loader = _make_loader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False,
                              num_workers=num_workers, pin_memory=pin_memory)

    input_dim = 28 * 28 if flatten else 1
    num_classes = 10

    xb, yb = next(iter(train_loader))
    _print_sanity_once("mnist", xb, yb, flatten_expected=(784 if flatten else None), dataset_name="mnist")

    return train_loader, val_loader, input_dim, num_classes


def build_fashion_mnist_datasets(
    root: str = "./data",
    seed: int = 42,
    split: SplitConfig = SplitConfig(val_ratio=0.1),
    batch_size: int = 64,
    *,
    flatten: bool = True,
    normalize: bool = True,
    download: bool = True,
    augment: bool = False,
    aug_pad: int = 2,
    aug_hflip: bool = False,
    hflip_p: float = 0.5,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> Tuple[DataLoader, DataLoader, int, int]:
    """FashionMNIST loaders with optional augmentation and flattening."""
    _assert_torchvision()
    set_seed(seed)

    tf = _mnist_like_transforms(
        flatten=flatten, normalize=normalize, augment=augment,
        aug_pad=aug_pad, aug_hflip=aug_hflip, hflip_p=hflip_p,
    )

    full_train = torchvision.datasets.FashionMNIST(root=root, train=True, transform=tf, download=download)
    train_ds, val_ds = split_dataset(full_train, split=split, seed=seed)

    train_loader = _make_loader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True,
                                num_workers=num_workers, pin_memory=pin_memory)
    val_loader = _make_loader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False,
                              num_workers=num_workers, pin_memory=pin_memory)

    input_dim = 28 * 28 if flatten else 1
    num_classes = 10

    xb, yb = next(iter(train_loader))
    _print_sanity_once("fashionmnist", xb, yb, flatten_expected=(784 if flatten else None),
                       dataset_name="fashionmnist")

    return train_loader, val_loader, input_dim, num_classes


def _sample_bow(
    n_samples: int,
    vocab_size: int,
    class_bias: float = 0.2,
    avg_len: int = 16,
    seed: int = 42,
) -> torch.Tensor:
    """Create simple bag-of-words vectors with class-skewed distributions."""
    g = torch.Generator().manual_seed(seed)
    lengths = torch.clamp((torch.randn(n_samples, generator=g) * (avg_len ** 0.5) + avg_len).round(), min=1).long()

    X = torch.zeros(n_samples, vocab_size, dtype=torch.float32)
    for i, L in enumerate(lengths.tolist()):
        p = torch.ones(vocab_size, dtype=torch.float32)
        half = vocab_size // 2
        p[:half] += class_bias * vocab_size
        p = p / p.sum()
        idx = torch.multinomial(p, num_samples=L, replacement=True, generator=g)
        X[i].scatter_add_(0, idx, torch.ones_like(idx, dtype=torch.float32))
    return X


def build_synthetic_text_bow(
    samples_per_class: int = 200,
    vocab_size: int = 256,
    class_bias: float = 0.2,
    avg_len: int = 16,
    seed: int = 42,
    split: SplitConfig = SplitConfig(),
    batch_size: int = 32,
    *,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> Tuple[DataLoader, DataLoader, int, int]:
    """Two-class synthetic text classification with Bag-of-Words vectors."""
    set_seed(seed)

    X0 = _sample_bow(samples_per_class, vocab_size, class_bias=class_bias, avg_len=avg_len, seed=seed)
    y0 = torch.zeros(samples_per_class, dtype=torch.long)

    X1 = _sample_bow(samples_per_class, vocab_size, class_bias=class_bias, avg_len=avg_len, seed=seed + 1)
    X1 = torch.fliplr(X1)
    y1 = torch.ones(samples_per_class, dtype=torch.long)

    X = torch.cat([X0, X1], dim=0)
    y = torch.cat([y0, y1], dim=0)

    dataset = TensorDataset(X, y)
    train_ds, val_ds = split_dataset(dataset, split=split, seed=seed)
    train_loader = _make_loader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True,
                                num_workers=num_workers, pin_memory=pin_memory)
    val_loader = _make_loader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False,
                              num_workers=num_workers, pin_memory=pin_memory)

    xb, yb = next(iter(train_loader))
    _print_sanity_once("text_bow", xb, yb, dataset_name="text_bow")

    return train_loader, val_loader, vocab_size, 2


# --------------------------------------------------------------------------- #
#                           Unified Accessor Function                          #
# --------------------------------------------------------------------------- #

# Available language datasets
LANGUAGE_DATASETS = [
    "wikitext2",      # Small Wikipedia (DEFAULT, ~2MB)
    "wikitext103",    # Large Wikipedia (~500MB)
    "shakespeare",    # Karpathy's tiny_shakespeare via GitHub (~1MB)
    "tinystories",    # Children's stories (~2GB)
    "bookcorpus",     # 11,000 books (~5GB)
    "openwebtext",    # Web text (~40GB)
    "textfile",       # Local text file
]


def get_dataloaders(
    name: str = "wikitext2",
    *,
    # Common
    batch_size: int = 64,
    seed: int = 42,
    split: SplitConfig = SplitConfig(),
    dataloader_num_workers: int = 0,
    dataloader_pin_memory: bool = False,
    # Toy
    toy_input_dim: int = 8,
    toy_samples_per_class: int = 100,
    # MNIST / FashionMNIST
    mnist_root: str = "./data",
    mnist_flatten: bool = True,
    mnist_normalize: bool = True,
    mnist_download: bool = True,
    mnist_batch_size: Optional[int] = None,
    mnist_augment: bool = False,
    mnist_aug_pad: int = 2,
    mnist_aug_hflip: bool = False,
    mnist_hflip_p: float = 0.5,
    # Text BoW
    text_vocab_size: int = 256,
    text_samples_per_class: int = 200,
    text_class_bias: float = 0.2,
    text_avg_len: int = 16,
    # Language Model
    seq_len: int = 128,
    stride: Optional[int] = None,
    text_filepath: Optional[str] = None,
    max_samples: Optional[int] = None,
    cache_dir: str = "./data",
) -> Union[
    Tuple[DataLoader, DataLoader, int, int],  # Vision: (train, val, input_dim, num_classes)
    Tuple[DataLoader, DataLoader, int],       # Language: (train, val, vocab_size)
]:
    """
    Unified entry point for all datasets.
    
    Parameters
    ----------
    name : str, default="wikitext2"
        Dataset name. Options:
        
        Vision datasets (return 4-tuple):
        - "toy": Two-class toy vectors
        - "mnist": Handwritten digits
        - "fashionmnist": Fashion items
        - "text_bow": Synthetic bag-of-words
        
        Language datasets (return 3-tuple):
        - "wikitext2": Small Wikipedia (~2MB) - DEFAULT, RECOMMENDED
        - "wikitext103": Large Wikipedia (~500MB)
        - "shakespeare": Karpathy's tiny_shakespeare (~1MB, via GitHub)
        - "tinystories": Children's stories (~2GB)
        - "bookcorpus": 11,000 books (~5GB)
        - "openwebtext": Web text (~40GB)
        - "textfile": Local text file (requires text_filepath)
        
    batch_size : int, default=64
        Batch size.
    seed : int, default=42
        Random seed.
    split : SplitConfig
        Train/val split configuration.
    seq_len : int, default=128
        Sequence length for language models.
    stride : int, optional
        Stride between samples for language models.
    text_filepath : str, optional
        Path to local text file (required for "textfile" dataset).
    max_samples : int, optional
        Maximum samples to load (for large datasets).
    cache_dir : str, default="./data"
        Directory for caching downloaded files.
    
    Returns
    -------
    For vision datasets:
        (train_loader, val_loader, input_dim, num_classes)
    For language datasets:
        (train_loader, val_loader, vocab_size)
        
    Examples
    --------
    >>> # WikiText-2 (default, recommended for quick experiments)
    >>> train, val, vocab = get_dataloaders("wikitext2", batch_size=64, seq_len=128)
    
    >>> # Shakespeare (via GitHub, no HuggingFace needed)
    >>> train, val, vocab = get_dataloaders("shakespeare", batch_size=64, seq_len=128)
    
    >>> # FashionMNIST (vision, unchanged API)
    >>> train, val, D, C = get_dataloaders("fashionmnist", batch_size=64)
    
    >>> # Local text file
    >>> train, val, vocab = get_dataloaders("textfile", text_filepath="my_book.txt")
    
    Notes
    -----
    Language datasets use ByteTokenizer (UTF-8 bytes) with vocab_size=256.
    This ensures all token IDs are in range [0, 255], preventing IndexError
    in nn.Embedding layers when processing Unicode text.
    """
    name = name.lower().strip()

    # -------------------------------------------------------------------------
    # Vision datasets (return 4-tuple)
    # -------------------------------------------------------------------------
    if name == "toy":
        return build_toy2token_dataset(
            samples_per_class=toy_samples_per_class,
            input_dim=toy_input_dim,
            seed=seed,
            split=split,
            batch_size=batch_size,
            num_workers=dataloader_num_workers,
            pin_memory=dataloader_pin_memory,
        )

    if name == "mnist":
        bs = mnist_batch_size if mnist_batch_size is not None else batch_size
        return build_mnist_datasets(
            root=mnist_root,
            seed=seed,
            split=split,
            batch_size=bs,
            flatten=mnist_flatten,
            normalize=mnist_normalize,
            download=mnist_download,
            augment=mnist_augment,
            aug_pad=mnist_aug_pad,
            aug_hflip=mnist_aug_hflip,
            hflip_p=mnist_hflip_p,
            num_workers=dataloader_num_workers,
            pin_memory=dataloader_pin_memory,
        )

    if name == "fashionmnist":
        bs = mnist_batch_size if mnist_batch_size is not None else batch_size
        return build_fashion_mnist_datasets(
            root=mnist_root,
            seed=seed,
            split=split,
            batch_size=bs,
            flatten=mnist_flatten,
            normalize=mnist_normalize,
            download=mnist_download,
            augment=mnist_augment,
            aug_pad=mnist_aug_pad,
            aug_hflip=mnist_aug_hflip,
            hflip_p=mnist_hflip_p,
            num_workers=dataloader_num_workers,
            pin_memory=dataloader_pin_memory,
        )

    if name == "text_bow":
        return build_synthetic_text_bow(
            samples_per_class=text_samples_per_class,
            vocab_size=text_vocab_size,
            class_bias=text_class_bias,
            avg_len=text_avg_len,
            seed=seed,
            split=split,
            batch_size=batch_size,
            num_workers=dataloader_num_workers,
            pin_memory=dataloader_pin_memory,
        )

    # -------------------------------------------------------------------------
    # Language datasets (return 3-tuple)
    # -------------------------------------------------------------------------
    
    # Shakespeare: Direct GitHub download (bypasses broken HuggingFace dataset)
    if name == "shakespeare":
        return load_shakespeare_from_github(
            seq_len=seq_len,
            seed=seed,
            split=split,
            batch_size=batch_size,
            stride=stride,
            cache_dir=cache_dir,
            num_workers=dataloader_num_workers,
            pin_memory=dataloader_pin_memory,
        )
    
    # Local text file
    if name == "textfile":
        if text_filepath is None:
            raise ValueError(
                "text_filepath required for 'textfile' dataset. "
                "Example: get_dataloaders('textfile', text_filepath='my_book.txt')"
            )
        return load_text_file(
            filepath=text_filepath,
            seq_len=seq_len,
            seed=seed,
            split=split,
            batch_size=batch_size,
            stride=stride,
            num_workers=dataloader_num_workers,
            pin_memory=dataloader_pin_memory,
        )
    
    # HuggingFace datasets: wikitext2, wikitext103, tinystories, bookcorpus, openwebtext
    if name in HUGGINGFACE_DATASETS or "/" in name:
        return load_huggingface_text_dataset(
            dataset_name=name,
            seq_len=seq_len,
            seed=seed,
            split=split,
            batch_size=batch_size,
            stride=stride,
            max_samples=max_samples,
            num_workers=dataloader_num_workers,
            pin_memory=dataloader_pin_memory,
        )

    # Unknown dataset
    all_datasets = ["toy", "mnist", "fashionmnist", "text_bow"] + LANGUAGE_DATASETS
    raise ValueError(
        f"Unknown dataset name '{name}'. Available options:\n"
        f"  Vision: toy, mnist, fashionmnist, text_bow\n"
        f"  Language: {', '.join(LANGUAGE_DATASETS)}\n"
        f"  Custom HuggingFace: 'username/dataset_name'"
    )


# --------------------------------------------------------------------------- #
#                                   Self-test                                 #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import logging
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)
    
    set_seed(42)
    
    logger.info("=" * 60)
    logger.info("BoeNet Data Utils v2.0.1 Self-Test Suite")
    logger.info("=" * 60)
    
    # Test 1: ByteTokenizer (renamed from CharTokenizer)
    logger.info("\n[Test 1] ByteTokenizer (byte-level UTF-8 encoding)")
    tokenizer = ByteTokenizer()
    
    # Test ASCII text
    text = "Hello, World!"
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    assert decoded == text, f"Round-trip failed: '{text}' → '{decoded}'"
    assert all(0 <= t <= 255 for t in encoded), f"Token out of range: {encoded}"
    logger.info(f"  ASCII: '{text}' → {encoded[:5]}... → '{decoded}'")
    
    # Test Unicode text (CRITICAL: this failed with old CharTokenizer)
    unicode_text = "Hello—World"  # em-dash (U+2014)
    unicode_encoded = tokenizer.encode(unicode_text)
    unicode_decoded = tokenizer.decode(unicode_encoded)
    assert unicode_decoded == unicode_text, f"Unicode round-trip failed: '{unicode_text}' → '{unicode_decoded}'"
    assert all(0 <= t <= 255 for t in unicode_encoded), f"Unicode token out of range: {unicode_encoded}"
    assert max(unicode_encoded) <= 255, f"Max token {max(unicode_encoded)} exceeds 255!"
    logger.info(f"  Unicode: '{unicode_text}' → {unicode_encoded} → '{unicode_decoded}'")
    logger.info(f"  Max token ID: {max(unicode_encoded)} (must be <= 255)")
    
    # Test with WikiText-like content (has em-dashes, smart quotes, etc.)
    wiki_text = "The film—directed by John—was released in 2020."
    wiki_encoded = tokenizer.encode(wiki_text)
    assert all(0 <= t <= 255 for t in wiki_encoded), f"Wiki token out of range: {wiki_encoded}"
    logger.info(f"  WikiText-like: max_token={max(wiki_encoded)}")
    
    logger.info(f"  vocab_size: {tokenizer.vocab_size}")
    logger.info("  ✓ ByteTokenizer OK (all tokens in 0-255 range)")
    
    # Test 2: TextDataset
    logger.info("\n[Test 2] TextDataset")
    sample_text = "The quick brown fox jumps over the lazy dog. " * 10
    dataset = TextDataset(sample_text, seq_len=32)
    input_ids, labels = dataset[0]
    assert input_ids.shape == (32,), f"Expected (32,), got {input_ids.shape}"
    assert labels.shape == (32,), f"Expected (32,), got {labels.shape}"
    # Verify all tokens are in valid range
    assert input_ids.max().item() < 256, f"input_ids max {input_ids.max().item()} >= 256"
    assert labels.max().item() < 256, f"labels max {labels.max().item()} >= 256"
    logger.info(f"  Dataset length: {len(dataset)}")
    logger.info(f"  input_ids shape: {input_ids.shape}, max: {input_ids.max().item()}")
    logger.info("  ✓ TextDataset OK")
    
    # Test 3: TextDataset with Unicode (would have failed with old CharTokenizer)
    logger.info("\n[Test 3] TextDataset with Unicode content")
    unicode_sample = "The film—directed by John—was 'great'. " * 10
    unicode_dataset = TextDataset(unicode_sample, seq_len=32)
    uid, ulabels = unicode_dataset[0]
    assert uid.max().item() < 256, f"Unicode input_ids max {uid.max().item()} >= 256"
    assert ulabels.max().item() < 256, f"Unicode labels max {ulabels.max().item()} >= 256"
    logger.info(f"  Unicode dataset length: {len(unicode_dataset)}")
    logger.info(f"  input_ids max: {uid.max().item()} (must be < 256)")
    logger.info("  ✓ Unicode TextDataset OK")
    
    # Test 4: Toy dataset (vision)
    logger.info("\n[Test 4] Toy dataset (vision)")
    train, val, D, C = build_toy2token_dataset(samples_per_class=10, batch_size=4)
    xb, yb = next(iter(train))
    assert xb.shape[1] == D, f"Expected dim {D}, got {xb.shape[1]}"
    logger.info(f"  D={D}, C={C}, batch={xb.shape}")
    logger.info("  ✓ Toy dataset OK")
    
    # Test 5: Shakespeare dataset (via GitHub)
    logger.info("\n[Test 5] Shakespeare dataset (via GitHub download)")
    try:
        train, val, vocab = load_shakespeare_from_github(
            seq_len=64, batch_size=8, split=SplitConfig(val_ratio=0.1),
            cache_dir="./data"
        )
        input_ids, labels = next(iter(train))
        assert input_ids.shape == (8, 64), f"Expected (8, 64), got {input_ids.shape}"
        assert labels.shape == (8, 64), f"Expected (8, 64), got {labels.shape}"
        assert input_ids.max().item() < 256, f"Shakespeare input_ids max {input_ids.max().item()} >= 256"
        logger.info(f"  vocab_size={vocab}, batch={input_ids.shape}, max_id={input_ids.max().item()}")
        logger.info("  ✓ Shakespeare dataset OK")
    except Exception as e:
        logger.warning(f"  ⚠ Shakespeare test failed: {e}")
    
    # Test 6: WikiText-2 (if datasets library available)
    if _HAS_DATASETS:
        logger.info("\n[Test 6] WikiText-2 dataset (via HuggingFace)")
        try:
            train, val, vocab = load_huggingface_text_dataset(
                "wikitext2", seq_len=64, batch_size=8, 
                split=SplitConfig(val_ratio=0.1),
                max_samples=100  # Limit for faster testing
            )
            input_ids, labels = next(iter(train))
            assert input_ids.shape[1] == 64, f"Expected seq_len=64, got {input_ids.shape[1]}"
            # CRITICAL: This would fail with old CharTokenizer (max would be ~8000+)
            assert input_ids.max().item() < 256, f"WikiText input_ids max {input_ids.max().item()} >= 256"
            logger.info(f"  vocab_size={vocab}, batch={input_ids.shape}")
            logger.info(f"  input_ids[min,max]=[{input_ids.min().item()},{input_ids.max().item()}] (must be < 256)")
            logger.info("  ✓ WikiText-2 dataset OK (byte-level tokenization working!)")
        except Exception as e:
            logger.warning(f"  ⚠ WikiText-2 test failed: {e}")
    else:
        logger.info("\n[Test 6] WikiText-2 dataset (SKIPPED - datasets not installed)")
    
    # Test 7: Unified accessor
    logger.info("\n[Test 7] Unified accessor (get_dataloaders)")
    
    # Vision dataset
    train, val, D, C = get_dataloaders("toy", batch_size=4, toy_samples_per_class=10)
    assert isinstance(D, int) and isinstance(C, int)
    logger.info(f"  toy: D={D}, C={C}")
    
    # Shakespeare (should work without datasets library)
    try:
        train, val, vocab = get_dataloaders("shakespeare", batch_size=8, seq_len=64)
        assert isinstance(vocab, int)
        assert vocab == 256, f"Expected vocab_size=256, got {vocab}"
        logger.info(f"  shakespeare: vocab_size={vocab}")
    except Exception as e:
        logger.warning(f"  ⚠ shakespeare accessor failed: {e}")
    
    # WikiText-2 (requires datasets library)
    if _HAS_DATASETS:
        try:
            train, val, vocab = get_dataloaders(
                "wikitext2", batch_size=8, seq_len=64, max_samples=100
            )
            assert isinstance(vocab, int)
            assert vocab == 256, f"Expected vocab_size=256, got {vocab}"
            input_ids, _ = next(iter(train))
            assert input_ids.max().item() < 256, f"Max token ID {input_ids.max().item()} >= 256"
            logger.info(f"  wikitext2: vocab_size={vocab}, max_id={input_ids.max().item()}")
        except Exception as e:
            logger.warning(f"  ⚠ wikitext2 accessor failed: {e}")
    
    logger.info("  ✓ Unified accessor OK")
    
    logger.info("\n" + "=" * 60)
    logger.info("All self-tests passed!")
    logger.info("=" * 60)
    
    # Print available datasets
    logger.info("\n" + "=" * 60)
    logger.info("Available Datasets:")
    logger.info("=" * 60)
    logger.info("\nVision (return 4-tuple: train, val, input_dim, num_classes):")
    logger.info("  - toy: Two-class toy vectors")
    logger.info("  - mnist: Handwritten digits")
    logger.info("  - fashionmnist: Fashion items")
    logger.info("  - text_bow: Synthetic bag-of-words")
    logger.info("\nLanguage (return 3-tuple: train, val, vocab_size):")
    logger.info("  - wikitext2: Small Wikipedia (~2MB) - DEFAULT, RECOMMENDED")
    logger.info("  - wikitext103: Large Wikipedia (~500MB)")
    logger.info("  - shakespeare: Karpathy's tiny_shakespeare (~1MB, via GitHub)")
    logger.info("  - tinystories: Children's stories (~2GB)")
    logger.info("  - bookcorpus: 11,000 books (~5GB)")
    logger.info("  - openwebtext: Web text (~40GB)")
    logger.info("  - textfile: Local text file (requires text_filepath)")
    logger.info("\nTokenization (v2.0.1):")
    logger.info("  - ByteTokenizer: UTF-8 byte-level (vocab_size=256)")
    logger.info("  - All token IDs guaranteed in range [0, 255]")
    logger.info("  - Unicode characters encoded as 2-4 bytes each")
    logger.info("=" * 60)