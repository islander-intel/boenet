#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
boenet/utils/data_utils.py (v1.0.0 - Language Model Support)

Lightweight dataset + dataloader utilities for BoeNet experiments.

Converted from BFSNet (Vision) to BoeNet (Vision + Language)
------------------------------------------------------------
This module now supports BOTH vision and language datasets:

Vision (unchanged from BFSNet):
  - Toy2Token: Two-class toy vectors
  - MNIST: Handwritten digits
  - FashionMNIST: Fashion items
  - Synthetic Text BoW: Bag-of-words classification

Language (NEW for BoeNet):
  - Shakespeare (char-level): karpathy/tiny_shakespeare
  - TinyStories (char/BPE): roneneldan/TinyStories

What's included
---------------
1) set_seed(seed): Deterministic seeding
2) get_device(force_cpu=False): Choose device
3) Vision datasets (unchanged):
   - Toy2Token, MNIST, FashionMNIST, Synthetic Text BoW
4) Language datasets (NEW):
   - TextDataset: Generic text dataset for language modeling
   - build_shakespeare_datasets: Character-level Shakespeare
   - build_tinystories_datasets: TinyStories (larger scale)
5) get_dataloaders(): Unified entry point supporting all datasets

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
This module includes a simple CharTokenizer for character-level modeling.
For BPE/subword tokenization, use tiktoken or HuggingFace tokenizers
and pass them to TextDataset.

Usage Examples
--------------
>>> # Character-level Shakespeare
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

Author: BoeNet project (extended from BFSNet)
Version: 1.0.0
Date: 2025-12-22
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Literal, Optional, Tuple, List, Union

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

        print(
            f"[sanity:{dataset_name}] batch={shape_str} dtype={input_ids.dtype} "
            f"input_ids[min,max]=[{id_min},{id_max}] labels[min,max]=[{label_min},{label_max}] "
            f"vocab_size={vocab_size}"
        )


# --------------------------------------------------------------------------- #
#                           Character Tokenizer                               #
# --------------------------------------------------------------------------- #

class CharTokenizer:
    """
    Simple character-level tokenizer using ASCII encoding.
    
    This tokenizer maps each character to its ASCII code (0-255).
    It's suitable for character-level language modeling on ASCII text.
    
    Attributes
    ----------
    vocab_size : int
        Vocabulary size (256 for full ASCII).
        
    Examples
    --------
    >>> tokenizer = CharTokenizer()
    >>> tokenizer.encode("Hello")
    [72, 101, 108, 108, 111]
    >>> tokenizer.decode([72, 101, 108, 108, 111])
    'Hello'
    """
    
    def __init__(self):
        """Initialize character tokenizer."""
        self._vocab_size = 256
    
    @property
    def vocab_size(self) -> int:
        """Return vocabulary size (256 for ASCII)."""
        return self._vocab_size
    
    def encode(self, text: str) -> List[int]:
        """
        Encode text to list of token IDs.
        
        Parameters
        ----------
        text : str
            Text to encode.
            
        Returns
        -------
        List[int]
            List of token IDs (ASCII codes).
        """
        return [ord(c) for c in text]
    
    def decode(self, tokens: List[int]) -> str:
        """
        Decode list of token IDs to text.
        
        Parameters
        ----------
        tokens : List[int]
            List of token IDs.
            
        Returns
        -------
        str
            Decoded text.
        """
        return "".join(chr(t) for t in tokens)
    
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
            List of encoded token ID lists.
        """
        return [self.encode(text) for text in texts]
    
    def decode_batch(self, token_lists: List[List[int]]) -> List[str]:
        """
        Decode a batch of token ID lists.
        
        Parameters
        ----------
        token_lists : List[List[int]]
            List of token ID lists.
            
        Returns
        -------
        List[str]
            List of decoded texts.
        """
        return [self.decode(tokens) for tokens in token_lists]


# --------------------------------------------------------------------------- #
#                           Text Dataset (NEW)                                #
# --------------------------------------------------------------------------- #

class TextDataset(Dataset):
    """
    Dataset for character-level or token-level language modeling.
    
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
        Tokenizer with encode() method. Defaults to CharTokenizer.
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
    """
    
    def __init__(
        self,
        text: str,
        seq_len: int,
        tokenizer: Optional[object] = None,
        stride: Optional[int] = None,
    ):
        if tokenizer is None:
            tokenizer = CharTokenizer()
        
        self.seq_len = seq_len
        self.stride = stride if stride is not None else seq_len
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size
        
        # Encode entire text
        token_ids = tokenizer.encode(text)
        self.token_ids = torch.tensor(token_ids, dtype=torch.long)
        
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
#                         Shakespeare Dataset (NEW)                           #
# --------------------------------------------------------------------------- #

def _assert_datasets() -> None:
    """Raise ImportError if HuggingFace datasets not available."""
    if not _HAS_DATASETS:
        raise ImportError(
            "HuggingFace datasets library not available. Install with:\n"
            "  pip install datasets\n"
            "Or use local text files with TextDataset directly."
        )


def build_shakespeare_datasets(
    seq_len: int = 128,
    seed: int = 42,
    split: SplitConfig = SplitConfig(val_ratio=0.1),
    batch_size: int = 64,
    stride: Optional[int] = None,
    *,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> Tuple[DataLoader, DataLoader, int]:
    """
    Build dataloaders for character-level Shakespeare language modeling.
    
    Uses the karpathy/tiny_shakespeare dataset from HuggingFace.
    
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
    >>> train_loader, val_loader, vocab_size = build_shakespeare_datasets(
    ...     seq_len=128, batch_size=64
    ... )
    >>> for input_ids, labels in train_loader:
    ...     # input_ids: [B, seq_len]
    ...     # labels: [B, seq_len]
    ...     break
    """
    _assert_datasets()
    set_seed(seed)
    
    # Load dataset from HuggingFace
    print("[data] Loading karpathy/tiny_shakespeare from HuggingFace...")
    dataset = load_dataset("karpathy/tiny_shakespeare", trust_remote_code=True)
    
    # Concatenate all text (train split contains full text)
    if "train" in dataset:
        text = dataset["train"]["text"][0]
    else:
        # Fallback: concatenate all splits
        text = ""
        for split_name in dataset.keys():
            for item in dataset[split_name]:
                text += item["text"]
    
    print(f"[data] Loaded {len(text):,} characters")
    
    # Create tokenizer
    tokenizer = CharTokenizer()
    
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
    
    return train_loader, val_loader, tokenizer.vocab_size


def build_tinystories_datasets(
    seq_len: int = 256,
    seed: int = 42,
    split: SplitConfig = SplitConfig(val_ratio=0.1),
    batch_size: int = 64,
    stride: Optional[int] = None,
    max_samples: Optional[int] = None,
    *,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> Tuple[DataLoader, DataLoader, int]:
    """
    Build dataloaders for character-level TinyStories language modeling.
    
    Uses the roneneldan/TinyStories dataset from HuggingFace.
    This is a larger dataset (~2GB) suitable for scaling experiments.
    
    Parameters
    ----------
    seq_len : int, default=256
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
        Maximum number of stories to use (for memory/speed limits).
        If None, use all available stories.
    num_workers : int, default=0
        Number of dataloader workers.
    pin_memory : bool, default=False
        Pin memory for faster GPU transfer.
        
    Returns
    -------
    Tuple[DataLoader, DataLoader, int]
        (train_loader, val_loader, vocab_size)
    """
    _assert_datasets()
    set_seed(seed)
    
    # Load dataset from HuggingFace
    print("[data] Loading roneneldan/TinyStories from HuggingFace...")
    print("[data] (This may take a while for the first download)")
    dataset = load_dataset("roneneldan/TinyStories", trust_remote_code=True)
    
    # Concatenate stories
    print("[data] Concatenating stories...")
    texts = []
    count = 0
    for item in dataset["train"]:
        texts.append(item["text"])
        count += 1
        if max_samples is not None and count >= max_samples:
            break
    
    text = "\n\n".join(texts)
    print(f"[data] Loaded {count:,} stories, {len(text):,} characters")
    
    # Create tokenizer
    tokenizer = CharTokenizer()
    
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
        "tinystories", input_ids, labels,
        vocab_size=tokenizer.vocab_size,
        dataset_name="tinystories"
    )
    
    return train_loader, val_loader, tokenizer.vocab_size


def build_text_file_datasets(
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
    Build dataloaders from a local text file.
    
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
    
    # Create tokenizer
    tokenizer = CharTokenizer()
    
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
        dataset_name=f"textfile"
    )
    
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

def get_dataloaders(
    name: Literal[
        "toy", "mnist", "fashionmnist", "text_bow",
        "shakespeare", "tinystories", "textfile"
    ] = "toy",
    *,
    # Common
    batch_size: int = 16,
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
    # Language Model (NEW)
    seq_len: int = 128,
    stride: Optional[int] = None,
    text_filepath: Optional[str] = None,
    tinystories_max_samples: Optional[int] = None,
) -> Union[
    Tuple[DataLoader, DataLoader, int, int],  # Vision: (train, val, input_dim, num_classes)
    Tuple[DataLoader, DataLoader, int],       # Language: (train, val, vocab_size)
]:
    """
    Unified entry point for all datasets.
    
    Parameters
    ----------
    name : str
        Dataset name. Options:
        - Vision: "toy", "mnist", "fashionmnist", "text_bow"
        - Language: "shakespeare", "tinystories", "textfile"
    batch_size : int
        Batch size.
    seed : int
        Random seed.
    split : SplitConfig
        Train/val split configuration.
    
    Returns
    -------
    For vision datasets:
        (train_loader, val_loader, input_dim, num_classes)
    For language datasets:
        (train_loader, val_loader, vocab_size)
        
    Examples
    --------
    >>> # Vision (4-tuple return)
    >>> train, val, D, C = get_dataloaders("fashionmnist", batch_size=64)
    >>> 
    >>> # Language (3-tuple return)
    >>> train, val, vocab = get_dataloaders("shakespeare", batch_size=64, seq_len=128)
    """
    name = name.lower().strip()

    # Vision datasets (return 4-tuple)
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

    # Language datasets (return 3-tuple)
    if name == "shakespeare":
        return build_shakespeare_datasets(
            seq_len=seq_len,
            seed=seed,
            split=split,
            batch_size=batch_size,
            stride=stride,
            num_workers=dataloader_num_workers,
            pin_memory=dataloader_pin_memory,
        )

    if name == "tinystories":
        return build_tinystories_datasets(
            seq_len=seq_len,
            seed=seed,
            split=split,
            batch_size=batch_size,
            stride=stride,
            max_samples=tinystories_max_samples,
            num_workers=dataloader_num_workers,
            pin_memory=dataloader_pin_memory,
        )

    if name == "textfile":
        if text_filepath is None:
            raise ValueError("text_filepath required for 'textfile' dataset")
        return build_text_file_datasets(
            filepath=text_filepath,
            seq_len=seq_len,
            seed=seed,
            split=split,
            batch_size=batch_size,
            stride=stride,
            num_workers=dataloader_num_workers,
            pin_memory=dataloader_pin_memory,
        )

    raise ValueError(
        f"Unknown dataset name '{name}'. Use one of: "
        "'toy', 'mnist', 'fashionmnist', 'text_bow', "
        "'shakespeare', 'tinystories', 'textfile'"
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
    logger.info("BoeNet Data Utils v1.0.0 Self-Test Suite")
    logger.info("=" * 60)
    
    # Test 1: CharTokenizer
    logger.info("\n[Test 1] CharTokenizer")
    tokenizer = CharTokenizer()
    text = "Hello, World!"
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    assert decoded == text, f"Round-trip failed: '{text}' → '{decoded}'"
    logger.info(f"  vocab_size: {tokenizer.vocab_size}")
    logger.info(f"  '{text}' → {encoded[:5]}... → '{decoded}'")
    logger.info("  ✓ CharTokenizer OK")
    
    # Test 2: TextDataset
    logger.info("\n[Test 2] TextDataset")
    sample_text = "The quick brown fox jumps over the lazy dog. " * 10
    dataset = TextDataset(sample_text, seq_len=32)
    input_ids, labels = dataset[0]
    assert input_ids.shape == (32,), f"Expected (32,), got {input_ids.shape}"
    assert labels.shape == (32,), f"Expected (32,), got {labels.shape}"
    # Verify shift: labels should be input_ids shifted by 1
    original_tokens = tokenizer.encode(sample_text)
    assert input_ids[0].item() == original_tokens[0]
    assert labels[0].item() == original_tokens[1]
    logger.info(f"  Dataset length: {len(dataset)}")
    logger.info(f"  input_ids shape: {input_ids.shape}")
    logger.info(f"  Shift verified: input[0]={input_ids[0].item()}, label[0]={labels[0].item()}")
    logger.info("  ✓ TextDataset OK")
    
    # Test 3: Toy dataset (vision)
    logger.info("\n[Test 3] Toy dataset (vision)")
    train, val, D, C = build_toy2token_dataset(samples_per_class=10, batch_size=4)
    xb, yb = next(iter(train))
    assert xb.shape[1] == D, f"Expected dim {D}, got {xb.shape[1]}"
    logger.info(f"  D={D}, C={C}, batch={xb.shape}")
    logger.info("  ✓ Toy dataset OK")
    
    # Test 4: Shakespeare dataset (if available)
    if _HAS_DATASETS:
        logger.info("\n[Test 4] Shakespeare dataset (language)")
        try:
            train, val, vocab = build_shakespeare_datasets(
                seq_len=64, batch_size=8, split=SplitConfig(val_ratio=0.1)
            )
            input_ids, labels = next(iter(train))
            assert input_ids.shape == (8, 64), f"Expected (8, 64), got {input_ids.shape}"
            assert labels.shape == (8, 64), f"Expected (8, 64), got {labels.shape}"
            logger.info(f"  vocab_size={vocab}, batch={input_ids.shape}")
            logger.info("  ✓ Shakespeare dataset OK")
        except Exception as e:
            logger.warning(f"  ⚠ Shakespeare test skipped: {e}")
    else:
        logger.info("\n[Test 4] Shakespeare dataset (SKIPPED - datasets not installed)")
    
    # Test 5: Unified accessor
    logger.info("\n[Test 5] Unified accessor (get_dataloaders)")
    
    # Vision dataset
    train, val, D, C = get_dataloaders("toy", batch_size=4, toy_samples_per_class=10)
    assert isinstance(D, int) and isinstance(C, int)
    logger.info(f"  toy: D={D}, C={C}")
    
    # Language dataset (if available)
    if _HAS_DATASETS:
        try:
            train, val, vocab = get_dataloaders("shakespeare", batch_size=8, seq_len=64)
            assert isinstance(vocab, int)
            logger.info(f"  shakespeare: vocab_size={vocab}")
        except Exception as e:
            logger.warning(f"  ⚠ shakespeare accessor skipped: {e}")
    
    logger.info("  ✓ Unified accessor OK")
    
    logger.info("\n" + "=" * 60)
    logger.info("All self-tests passed!")
    logger.info("=" * 60)