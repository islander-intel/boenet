#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
boenet/tokenizer.py (v1.0.0)

Tokenization utilities for BoeNet language models.

This module provides tokenizers for converting text to token IDs and back.
Currently supports:
  - CharTokenizer: Character-level tokenization (ASCII)
  - TiktokenWrapper: BPE tokenization via tiktoken (optional)

Usage Examples
--------------
>>> # Character-level (built-in, no dependencies)
>>> from boenet.tokenizer import CharTokenizer
>>> tokenizer = CharTokenizer()
>>> tokens = tokenizer.encode("Hello, World!")
>>> text = tokenizer.decode(tokens)

>>> # BPE tokenization (requires tiktoken)
>>> from boenet.tokenizer import TiktokenWrapper
>>> tokenizer = TiktokenWrapper("gpt2")
>>> tokens = tokenizer.encode("Hello, World!")
>>> text = tokenizer.decode(tokens)

Tokenizer Interface
-------------------
All tokenizers implement the following interface:
  - vocab_size: int property
  - encode(text: str) → List[int]
  - decode(tokens: List[int]) → str
  - encode_batch(texts: List[str]) → List[List[int]]
  - decode_batch(token_lists: List[List[int]]) → List[str]

Author: BoeNet project
Version: 1.0.0
Date: 2025-12-22
"""

from __future__ import annotations
from typing import List, Optional, Protocol, runtime_checkable
from abc import ABC, abstractmethod

# Optional tiktoken import
try:
    import tiktoken
    _HAS_TIKTOKEN = True
except ImportError:
    _HAS_TIKTOKEN = False


# --------------------------------------------------------------------------- #
#                           Tokenizer Protocol                                #
# --------------------------------------------------------------------------- #

@runtime_checkable
class TokenizerProtocol(Protocol):
    """Protocol defining the tokenizer interface."""
    
    @property
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        ...
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        ...
    
    def decode(self, tokens: List[int]) -> str:
        """Decode token IDs to text."""
        ...


class BaseTokenizer(ABC):
    """Abstract base class for tokenizers."""
    
    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        pass
    
    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """Encode text to list of token IDs."""
        pass
    
    @abstractmethod
    def decode(self, tokens: List[int]) -> str:
        """Decode list of token IDs to text."""
        pass
    
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
#                           Character Tokenizer                               #
# --------------------------------------------------------------------------- #

class CharTokenizer(BaseTokenizer):
    """
    Simple character-level tokenizer using ASCII encoding.
    
    This tokenizer maps each character to its ASCII code (0-255).
    It's suitable for character-level language modeling on ASCII text.
    
    Non-ASCII characters are replaced with a placeholder (code 0).
    
    Attributes
    ----------
    vocab_size : int
        Vocabulary size (256 for full ASCII).
        
    Examples
    --------
    >>> tokenizer = CharTokenizer()
    >>> tokenizer.vocab_size
    256
    >>> tokenizer.encode("Hello")
    [72, 101, 108, 108, 111]
    >>> tokenizer.decode([72, 101, 108, 108, 111])
    'Hello'
    >>> tokenizer.encode("café")  # Non-ASCII handling
    [99, 97, 102, 0]  # 'é' replaced with 0
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
        Encode text to list of token IDs (ASCII codes).
        
        Parameters
        ----------
        text : str
            Text to encode.
            
        Returns
        -------
        List[int]
            List of token IDs. Non-ASCII characters are encoded as 0.
        """
        result = []
        for c in text:
            code = ord(c)
            if code < 256:
                result.append(code)
            else:
                # Non-ASCII: use 0 as placeholder
                result.append(0)
        return result
    
    def decode(self, tokens: List[int]) -> str:
        """
        Decode list of token IDs to text.
        
        Parameters
        ----------
        tokens : List[int]
            List of token IDs (ASCII codes).
            
        Returns
        -------
        str
            Decoded text.
        """
        return "".join(chr(t % 256) for t in tokens)
    
    def __repr__(self) -> str:
        return f"CharTokenizer(vocab_size={self.vocab_size})"


# --------------------------------------------------------------------------- #
#                           Tiktoken Wrapper                                  #
# --------------------------------------------------------------------------- #

class TiktokenWrapper(BaseTokenizer):
    """
    BPE tokenizer wrapper using tiktoken.
    
    tiktoken is OpenAI's fast BPE tokenizer library, used by GPT models.
    This wrapper provides a consistent interface matching CharTokenizer.
    
    Requires: pip install tiktoken
    
    Parameters
    ----------
    encoding_name : str, default="gpt2"
        Name of the tiktoken encoding to use.
        Options: "gpt2", "r50k_base", "p50k_base", "cl100k_base", etc.
        
    Attributes
    ----------
    vocab_size : int
        Vocabulary size (depends on encoding).
        - gpt2: 50257
        - cl100k_base: 100277
        
    Examples
    --------
    >>> tokenizer = TiktokenWrapper("gpt2")
    >>> tokenizer.vocab_size
    50257
    >>> tokenizer.encode("Hello, World!")
    [15496, 11, 2159, 0]
    >>> tokenizer.decode([15496, 11, 2159, 0])
    'Hello, World!'
    
    Notes
    -----
    tiktoken is much faster than other BPE implementations due to
    its Rust backend. It's recommended for production use.
    """
    
    def __init__(self, encoding_name: str = "gpt2"):
        """
        Initialize tiktoken wrapper.
        
        Parameters
        ----------
        encoding_name : str, default="gpt2"
            Tiktoken encoding name.
            
        Raises
        ------
        ImportError
            If tiktoken is not installed.
        """
        if not _HAS_TIKTOKEN:
            raise ImportError(
                "tiktoken is not available. Install with:\n"
                "  pip install tiktoken\n"
                "Or use CharTokenizer for character-level tokenization."
            )
        
        self.encoding_name = encoding_name
        self._encoding = tiktoken.get_encoding(encoding_name)
        self._vocab_size = self._encoding.n_vocab
    
    @property
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        return self._vocab_size
    
    def encode(self, text: str) -> List[int]:
        """
        Encode text to list of BPE token IDs.
        
        Parameters
        ----------
        text : str
            Text to encode.
            
        Returns
        -------
        List[int]
            List of BPE token IDs.
        """
        return self._encoding.encode(text)
    
    def decode(self, tokens: List[int]) -> str:
        """
        Decode list of BPE token IDs to text.
        
        Parameters
        ----------
        tokens : List[int]
            List of BPE token IDs.
            
        Returns
        -------
        str
            Decoded text.
        """
        return self._encoding.decode(tokens)
    
    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        """
        Encode a batch of texts (optimized for tiktoken).
        
        Parameters
        ----------
        texts : List[str]
            List of texts to encode.
            
        Returns
        -------
        List[List[int]]
            List of encoded token ID lists.
        """
        return self._encoding.encode_batch(texts)
    
    def decode_batch(self, token_lists: List[List[int]]) -> List[str]:
        """
        Decode a batch of token ID lists (optimized for tiktoken).
        
        Parameters
        ----------
        token_lists : List[List[int]]
            List of token ID lists.
            
        Returns
        -------
        List[str]
            List of decoded texts.
        """
        return self._encoding.decode_batch(token_lists)
    
    def __repr__(self) -> str:
        return f"TiktokenWrapper(encoding='{self.encoding_name}', vocab_size={self.vocab_size})"


# --------------------------------------------------------------------------- #
#                           Factory Function                                  #
# --------------------------------------------------------------------------- #

def get_tokenizer(
    tokenizer_type: str = "char",
    encoding_name: str = "gpt2",
) -> BaseTokenizer:
    """
    Factory function to get a tokenizer by type.
    
    Parameters
    ----------
    tokenizer_type : str, default="char"
        Type of tokenizer: "char" for CharTokenizer, "bpe" for TiktokenWrapper.
    encoding_name : str, default="gpt2"
        Encoding name for BPE tokenizer (ignored for char tokenizer).
        
    Returns
    -------
    BaseTokenizer
        Tokenizer instance.
        
    Examples
    --------
    >>> tokenizer = get_tokenizer("char")
    >>> tokenizer.vocab_size
    256
    
    >>> tokenizer = get_tokenizer("bpe", encoding_name="gpt2")
    >>> tokenizer.vocab_size
    50257
    """
    tokenizer_type = tokenizer_type.lower().strip()
    
    if tokenizer_type == "char":
        return CharTokenizer()
    
    if tokenizer_type in ("bpe", "tiktoken"):
        return TiktokenWrapper(encoding_name=encoding_name)
    
    raise ValueError(
        f"Unknown tokenizer type: '{tokenizer_type}'. "
        "Use 'char' for character-level or 'bpe'/'tiktoken' for BPE."
    )


# --------------------------------------------------------------------------- #
#                           Utility Functions                                 #
# --------------------------------------------------------------------------- #

def tokenize_text(
    text: str,
    tokenizer: Optional[BaseTokenizer] = None,
    return_tensor: bool = False,
) -> List[int]:
    """
    Convenience function to tokenize text.
    
    Parameters
    ----------
    text : str
        Text to tokenize.
    tokenizer : BaseTokenizer, optional
        Tokenizer to use. Defaults to CharTokenizer.
    return_tensor : bool, default=False
        If True, return torch.Tensor instead of list.
        
    Returns
    -------
    List[int] or torch.Tensor
        Token IDs.
    """
    if tokenizer is None:
        tokenizer = CharTokenizer()
    
    tokens = tokenizer.encode(text)
    
    if return_tensor:
        import torch
        return torch.tensor(tokens, dtype=torch.long)
    
    return tokens


def detokenize_tokens(
    tokens: List[int],
    tokenizer: Optional[BaseTokenizer] = None,
) -> str:
    """
    Convenience function to detokenize token IDs.
    
    Parameters
    ----------
    tokens : List[int]
        Token IDs to decode.
    tokenizer : BaseTokenizer, optional
        Tokenizer to use. Defaults to CharTokenizer.
        
    Returns
    -------
    str
        Decoded text.
    """
    if tokenizer is None:
        tokenizer = CharTokenizer()
    
    return tokenizer.decode(tokens)


# --------------------------------------------------------------------------- #
#                                  Self-test                                  #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import logging
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info("BoeNet Tokenizer v1.0.0 Self-Test Suite")
    logger.info("=" * 60)
    
    # Test 1: CharTokenizer
    logger.info("\n[Test 1] CharTokenizer")
    char_tok = CharTokenizer()
    
    test_text = "Hello, World! 123"
    encoded = char_tok.encode(test_text)
    decoded = char_tok.decode(encoded)
    
    assert decoded == test_text, f"Round-trip failed: '{test_text}' → '{decoded}'"
    assert char_tok.vocab_size == 256
    
    logger.info(f"  vocab_size: {char_tok.vocab_size}")
    logger.info(f"  '{test_text}' → {encoded}")
    logger.info(f"  {encoded} → '{decoded}'")
    logger.info("  ✓ CharTokenizer OK")
    
    # Test 2: CharTokenizer batch
    logger.info("\n[Test 2] CharTokenizer batch operations")
    texts = ["Hello", "World", "!"]
    encoded_batch = char_tok.encode_batch(texts)
    decoded_batch = char_tok.decode_batch(encoded_batch)
    
    assert decoded_batch == texts, f"Batch round-trip failed"
    logger.info(f"  {texts} → {encoded_batch} → {decoded_batch}")
    logger.info("  ✓ Batch operations OK")
    
    # Test 3: CharTokenizer edge cases
    logger.info("\n[Test 3] CharTokenizer edge cases")
    
    # Empty string
    assert char_tok.encode("") == []
    assert char_tok.decode([]) == ""
    logger.info("  Empty string: OK")
    
    # Single character
    assert char_tok.encode("A") == [65]
    assert char_tok.decode([65]) == "A"
    logger.info("  Single character: OK")
    
    # Special characters
    special = "\n\t\r"
    special_encoded = char_tok.encode(special)
    special_decoded = char_tok.decode(special_encoded)
    assert special_decoded == special
    logger.info(f"  Special chars: {repr(special)} → {special_encoded} → {repr(special_decoded)}")
    logger.info("  ✓ Edge cases OK")
    
    # Test 4: Factory function
    logger.info("\n[Test 4] Factory function (get_tokenizer)")
    
    tok = get_tokenizer("char")
    assert isinstance(tok, CharTokenizer)
    logger.info(f"  get_tokenizer('char') → {tok}")
    logger.info("  ✓ Factory function OK")
    
    # Test 5: TiktokenWrapper (if available)
    if _HAS_TIKTOKEN:
        logger.info("\n[Test 5] TiktokenWrapper (BPE)")
        
        bpe_tok = TiktokenWrapper("gpt2")
        
        test_text = "Hello, World!"
        bpe_encoded = bpe_tok.encode(test_text)
        bpe_decoded = bpe_tok.decode(bpe_encoded)
        
        assert bpe_decoded == test_text
        
        logger.info(f"  vocab_size: {bpe_tok.vocab_size}")
        logger.info(f"  '{test_text}' → {bpe_encoded}")
        logger.info(f"  {bpe_encoded} → '{bpe_decoded}'")
        logger.info("  ✓ TiktokenWrapper OK")
        
        # Test factory
        tok = get_tokenizer("bpe", encoding_name="gpt2")
        assert isinstance(tok, TiktokenWrapper)
        logger.info(f"  get_tokenizer('bpe') → {tok}")
    else:
        logger.info("\n[Test 5] TiktokenWrapper (SKIPPED - tiktoken not installed)")
        logger.info("  Install with: pip install tiktoken")
    
    # Test 6: Utility functions
    logger.info("\n[Test 6] Utility functions")
    
    tokens = tokenize_text("Hello")
    assert tokens == [72, 101, 108, 108, 111]
    logger.info(f"  tokenize_text('Hello') → {tokens}")
    
    text = detokenize_tokens([72, 101, 108, 108, 111])
    assert text == "Hello"
    logger.info(f"  detokenize_tokens([72, 101, 108, 108, 111]) → '{text}'")
    
    logger.info("  ✓ Utility functions OK")
    
    # Test 7: Protocol check
    logger.info("\n[Test 7] Protocol compliance")
    
    assert isinstance(char_tok, TokenizerProtocol)
    logger.info(f"  CharTokenizer implements TokenizerProtocol: True")
    
    if _HAS_TIKTOKEN:
        bpe_tok = TiktokenWrapper("gpt2")
        assert isinstance(bpe_tok, TokenizerProtocol)
        logger.info(f"  TiktokenWrapper implements TokenizerProtocol: True")
    
    logger.info("  ✓ Protocol compliance OK")
    
    logger.info("\n" + "=" * 60)
    logger.info("All self-tests passed!")
    logger.info("=" * 60)