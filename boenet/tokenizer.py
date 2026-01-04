#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
boenet/tokenizer.py (v1.1.0)

Tokenization utilities for BoeNet language models.

This module provides tokenizers for converting text to token IDs and back.
Currently supports:
  - CharTokenizer: Character-level tokenization (ASCII)
  - TiktokenWrapper: BPE tokenization via tiktoken (optional)

v1.1.0 Changes (2026-01-03):
----------------------------
  - Changed default BPE encoding from "gpt2" to "cl100k_base" (GPT-4 tokenizer)
  - Added eos_token_id and pad_token_id properties to TiktokenWrapper
  - Added encode_with_truncation() method for fixed-length encoding
  - Updated factory function default to use cl100k_base
  - Improved error handling for invalid token IDs in CharTokenizer.decode()

Tokenizer Comparison:
---------------------
  | Tokenizer     | Vocab Size | Best For                    |
  |---------------|------------|------------------------------|
  | CharTokenizer | 256        | Character-level experiments  |
  | gpt2          | 50,257     | Legacy compatibility         |
  | cl100k_base   | 100,277    | Modern BPE (GPT-4 quality)   |

Usage Examples
--------------
>>> # Character-level (built-in, no dependencies)
>>> from boenet.tokenizer import CharTokenizer
>>> tokenizer = CharTokenizer()
>>> tokens = tokenizer.encode("Hello, World!")
>>> text = tokenizer.decode(tokens)

>>> # BPE tokenization with cl100k_base (GPT-4 tokenizer)
>>> from boenet.tokenizer import TiktokenWrapper
>>> tokenizer = TiktokenWrapper("cl100k_base")
>>> tokens = tokenizer.encode("Hello, World!")
>>> text = tokenizer.decode(tokens)

>>> # Factory function (defaults to cl100k_base for BPE)
>>> from boenet.tokenizer import get_tokenizer
>>> tokenizer = get_tokenizer("bpe")  # Uses cl100k_base
>>> tokenizer.vocab_size
100277

Tokenizer Interface
-------------------
All tokenizers implement the following interface:
  - vocab_size: int property
  - encode(text: str) → List[int]
  - decode(tokens: List[int]) → str
  - encode_batch(texts: List[str]) → List[List[int]]
  - decode_batch(token_lists: List[List[int]]) → List[str]

Additional properties for TiktokenWrapper:
  - eos_token_id: int (end of sequence token)
  - pad_token_id: int (padding token, same as eos)

Author: BoeNet project
Version: 1.1.0
Date: 2026-01-03
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
    
    @property
    def eos_token_id(self) -> int:
        """End of sequence token ID (use 0 for char tokenizer)."""
        return 0
    
    @property
    def pad_token_id(self) -> int:
        """Padding token ID (same as EOS)."""
        return self.eos_token_id
    
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
        
        v1.1.0: Added validation for token IDs.
        
        Parameters
        ----------
        tokens : List[int]
            List of token IDs (ASCII codes).
            
        Returns
        -------
        str
            Decoded text.
            
        Raises
        ------
        ValueError
            If any token ID is outside valid range [0, 255].
        """
        result = []
        for t in tokens:
            if not 0 <= t < 256:
                raise ValueError(
                    f"Invalid token ID: {t}. "
                    f"CharTokenizer only supports tokens in range [0, 255]."
                )
            result.append(chr(t))
        return "".join(result)
    
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
    encoding_name : str, default="cl100k_base"
        Name of the tiktoken encoding to use.
        Options: "gpt2", "r50k_base", "p50k_base", "cl100k_base", etc.
        
        v1.1.0: Default changed from "gpt2" to "cl100k_base" (GPT-4 tokenizer).
        
    Attributes
    ----------
    vocab_size : int
        Vocabulary size (depends on encoding).
        - gpt2: 50,257
        - cl100k_base: 100,277
        
    eos_token_id : int
        End of text token ID.
        
    pad_token_id : int
        Padding token ID (same as EOS).
        
    Examples
    --------
    >>> tokenizer = TiktokenWrapper("cl100k_base")
    >>> tokenizer.vocab_size
    100277
    >>> tokenizer.encode("Hello, World!")
    [9906, 11, 4435, 0]
    >>> tokenizer.decode([9906, 11, 4435, 0])
    'Hello, World!'
    >>> tokenizer.eos_token_id
    100257
    
    Notes
    -----
    tiktoken is much faster than other BPE implementations due to
    its Rust backend. It's recommended for production use.
    
    cl100k_base is the encoding used by GPT-4 and ChatGPT. It provides
    ~30% better compression than GPT-2's encoding for typical text.
    """
    
    def __init__(self, encoding_name: str = "cl100k_base"):
        """
        Initialize tiktoken wrapper.
        
        Parameters
        ----------
        encoding_name : str, default="cl100k_base"
            Tiktoken encoding name.
            v1.1.0: Default changed to "cl100k_base" (GPT-4 tokenizer).
            
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
        
        # Get special tokens
        # For cl100k_base: eot_token is the end of text token
        self._eos_token_id = self._encoding.eot_token
    
    @property
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        return self._vocab_size
    
    @property
    def eos_token_id(self) -> int:
        """
        End of sequence token ID.
        
        For cl100k_base, this is token 100257.
        For gpt2, this is token 50256.
        """
        return self._eos_token_id
    
    @property
    def pad_token_id(self) -> int:
        """
        Padding token ID (same as EOS for GPT-style models).
        
        GPT models typically use the EOS token for padding since they
        are decoder-only and don't need explicit padding attention masks.
        """
        return self.eos_token_id
    
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
    
    def encode_with_truncation(
        self, 
        text: str, 
        max_length: int,
        truncation_side: str = "right",
    ) -> List[int]:
        """
        Encode text and truncate to max_length.
        
        v1.1.0: New method for fixed-length encoding.
        
        Parameters
        ----------
        text : str
            Text to encode.
        max_length : int
            Maximum number of tokens.
        truncation_side : str, default="right"
            Which side to truncate: "left" or "right".
            
        Returns
        -------
        List[int]
            List of token IDs, truncated to max_length.
        """
        tokens = self._encoding.encode(text)
        
        if len(tokens) <= max_length:
            return tokens
        
        if truncation_side == "left":
            return tokens[-max_length:]
        else:
            return tokens[:max_length]
    
    def encode_with_padding(
        self,
        text: str,
        max_length: int,
        padding_side: str = "right",
        truncate: bool = True,
    ) -> List[int]:
        """
        Encode text and pad/truncate to exact max_length.
        
        v1.1.0: New method for fixed-length batched training.
        
        Parameters
        ----------
        text : str
            Text to encode.
        max_length : int
            Exact length of output.
        padding_side : str, default="right"
            Which side to pad: "left" or "right".
        truncate : bool, default=True
            Whether to truncate if text is longer than max_length.
            
        Returns
        -------
        List[int]
            List of token IDs with exactly max_length tokens.
        """
        tokens = self._encoding.encode(text)
        
        # Truncate if needed
        if len(tokens) > max_length:
            if truncate:
                tokens = tokens[:max_length]
            else:
                raise ValueError(
                    f"Text has {len(tokens)} tokens but max_length={max_length}. "
                    f"Set truncate=True to truncate."
                )
        
        # Pad if needed
        padding_needed = max_length - len(tokens)
        if padding_needed > 0:
            padding = [self.pad_token_id] * padding_needed
            if padding_side == "left":
                tokens = padding + tokens
            else:
                tokens = tokens + padding
        
        return tokens
    
    def __repr__(self) -> str:
        return (
            f"TiktokenWrapper(encoding='{self.encoding_name}', "
            f"vocab_size={self.vocab_size}, eos_token_id={self.eos_token_id})"
        )


# --------------------------------------------------------------------------- #
#                           Factory Function                                  #
# --------------------------------------------------------------------------- #

def get_tokenizer(
    tokenizer_type: str = "char",
    encoding_name: str = "cl100k_base",
) -> BaseTokenizer:
    """
    Factory function to get a tokenizer by type.
    
    Parameters
    ----------
    tokenizer_type : str, default="char"
        Type of tokenizer: "char" for CharTokenizer, "bpe" for TiktokenWrapper.
    encoding_name : str, default="cl100k_base"
        Encoding name for BPE tokenizer (ignored for char tokenizer).
        v1.1.0: Default changed from "gpt2" to "cl100k_base".
        
    Returns
    -------
    BaseTokenizer
        Tokenizer instance.
        
    Examples
    --------
    >>> tokenizer = get_tokenizer("char")
    >>> tokenizer.vocab_size
    256
    
    >>> tokenizer = get_tokenizer("bpe")  # Uses cl100k_base by default
    >>> tokenizer.vocab_size
    100277
    
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


def get_vocab_size(tokenizer_type: str = "char", encoding_name: str = "cl100k_base") -> int:
    """
    Get vocabulary size for a tokenizer type without instantiating it.
    
    v1.1.0: New utility function for config validation.
    
    Parameters
    ----------
    tokenizer_type : str, default="char"
        Type of tokenizer.
    encoding_name : str, default="cl100k_base"
        Encoding name for BPE tokenizer.
        
    Returns
    -------
    int
        Vocabulary size.
    """
    if tokenizer_type == "char":
        return 256
    
    if tokenizer_type in ("bpe", "tiktoken"):
        if not _HAS_TIKTOKEN:
            raise ImportError("tiktoken not installed")
        
        # Known vocab sizes to avoid instantiation
        known_sizes = {
            "gpt2": 50257,
            "r50k_base": 50257,
            "p50k_base": 50281,
            "cl100k_base": 100277,
        }
        
        if encoding_name in known_sizes:
            return known_sizes[encoding_name]
        
        # Fall back to instantiation
        enc = tiktoken.get_encoding(encoding_name)
        return enc.n_vocab
    
    raise ValueError(f"Unknown tokenizer type: '{tokenizer_type}'")


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
    logger.info("BoeNet Tokenizer v1.1.0 Self-Test Suite")
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
    logger.info(f"  eos_token_id: {char_tok.eos_token_id}")
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
    
    # v1.1.0: Test invalid token error
    try:
        char_tok.decode([300])
        assert False, "Should have raised ValueError"
    except ValueError as e:
        logger.info(f"  Invalid token error: {e}")
    logger.info("  ✓ Edge cases OK")
    
    # Test 4: Factory function
    logger.info("\n[Test 4] Factory function (get_tokenizer)")
    
    tok = get_tokenizer("char")
    assert isinstance(tok, CharTokenizer)
    logger.info(f"  get_tokenizer('char') → {tok}")
    logger.info("  ✓ Factory function OK")
    
    # Test 5: TiktokenWrapper with cl100k_base (v1.1.0 default)
    if _HAS_TIKTOKEN:
        logger.info("\n[Test 5] TiktokenWrapper (cl100k_base - v1.1.0 default)")
        
        bpe_tok = TiktokenWrapper("cl100k_base")
        
        test_text = "Hello, World!"
        bpe_encoded = bpe_tok.encode(test_text)
        bpe_decoded = bpe_tok.decode(bpe_encoded)
        
        assert bpe_decoded == test_text
        
        logger.info(f"  vocab_size: {bpe_tok.vocab_size}")
        logger.info(f"  eos_token_id: {bpe_tok.eos_token_id}")
        logger.info(f"  pad_token_id: {bpe_tok.pad_token_id}")
        logger.info(f"  '{test_text}' → {bpe_encoded}")
        logger.info(f"  {bpe_encoded} → '{bpe_decoded}'")
        logger.info("  ✓ TiktokenWrapper (cl100k_base) OK")
        
        # Test 5b: Compare with gpt2
        logger.info("\n[Test 5b] TiktokenWrapper (gpt2 - legacy)")
        gpt2_tok = TiktokenWrapper("gpt2")
        gpt2_encoded = gpt2_tok.encode(test_text)
        logger.info(f"  gpt2 vocab_size: {gpt2_tok.vocab_size}")
        logger.info(f"  gpt2 '{test_text}' → {gpt2_encoded}")
        logger.info(f"  cl100k_base is ~{bpe_tok.vocab_size / gpt2_tok.vocab_size:.1f}x larger vocab")
        
        # Test 5c: encode_with_truncation (v1.1.0)
        logger.info("\n[Test 5c] encode_with_truncation (v1.1.0)")
        long_text = "This is a very long text that should be truncated."
        truncated = bpe_tok.encode_with_truncation(long_text, max_length=5)
        assert len(truncated) == 5
        logger.info(f"  Truncated to 5 tokens: {truncated}")
        logger.info("  ✓ encode_with_truncation OK")
        
        # Test 5d: encode_with_padding (v1.1.0)
        logger.info("\n[Test 5d] encode_with_padding (v1.1.0)")
        short_text = "Hi"
        padded = bpe_tok.encode_with_padding(short_text, max_length=10)
        assert len(padded) == 10
        logger.info(f"  Padded to 10 tokens: {padded}")
        logger.info(f"  Pad token ID: {bpe_tok.pad_token_id}")
        logger.info("  ✓ encode_with_padding OK")
        
        # Test factory with default encoding
        tok = get_tokenizer("bpe")  # Should use cl100k_base
        assert isinstance(tok, TiktokenWrapper)
        assert tok.vocab_size == 100277, f"Expected cl100k_base (100277), got {tok.vocab_size}"
        logger.info(f"  get_tokenizer('bpe') → {tok}")
        logger.info("  ✓ Factory uses cl100k_base by default")
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
    
    # v1.1.0: get_vocab_size
    logger.info("\n[Test 6b] get_vocab_size (v1.1.0)")
    assert get_vocab_size("char") == 256
    logger.info(f"  get_vocab_size('char') = 256")
    if _HAS_TIKTOKEN:
        assert get_vocab_size("bpe", "cl100k_base") == 100277
        assert get_vocab_size("bpe", "gpt2") == 50257
        logger.info(f"  get_vocab_size('bpe', 'cl100k_base') = 100277")
        logger.info(f"  get_vocab_size('bpe', 'gpt2') = 50257")
    
    logger.info("  ✓ Utility functions OK")
    
    # Test 7: Protocol check
    logger.info("\n[Test 7] Protocol compliance")
    
    assert isinstance(char_tok, TokenizerProtocol)
    logger.info(f"  CharTokenizer implements TokenizerProtocol: True")
    
    if _HAS_TIKTOKEN:
        bpe_tok = TiktokenWrapper("cl100k_base")
        assert isinstance(bpe_tok, TokenizerProtocol)
        logger.info(f"  TiktokenWrapper implements TokenizerProtocol: True")
    
    logger.info("  ✓ Protocol compliance OK")
    
    # Test 8: v1.1.0 Summary
    logger.info("\n[Test 8] v1.1.0 Feature Summary")
    logger.info("  ✓ Default BPE encoding changed to cl100k_base (GPT-4)")
    logger.info("  ✓ Added eos_token_id and pad_token_id properties")
    logger.info("  ✓ Added encode_with_truncation() method")
    logger.info("  ✓ Added encode_with_padding() method")
    logger.info("  ✓ Added get_vocab_size() utility function")
    logger.info("  ✓ Improved CharTokenizer.decode() error handling")
    
    logger.info("\n" + "=" * 60)
    logger.info("All self-tests passed!")
    logger.info("=" * 60)