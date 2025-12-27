"""
ULTRA SIMPLE Tokenizer - byte-level

Vocab size = 256 (all bytes)
No BPE, no fancy shit, just works!
"""

from typing import List
import numpy as np


class SimpleByteTokenizer:
    """
    Simplest possible tokenizer - byte level

    Each byte (0-255) = one token
    Vocab size = 256
    """

    def __init__(self):
        self.vocab_size = 256

    def encode(self, text: str) -> List[int]:
        """
        Encode text to byte IDs

        text -> UTF-8 bytes -> list of ints (0-255)
        """
        # UTF-8 encode
        byte_array = text.encode('utf-8', errors='ignore')

        # Convert to list of ints
        return list(byte_array)

    def decode(self, tokens: List[int]) -> str:
        """
        Decode byte IDs to text

        list of ints -> bytes -> UTF-8 decode -> text
        """
        # Clip to valid byte range [0, 255]
        valid_tokens = [max(0, min(255, t)) for t in tokens]

        # Convert to bytes
        byte_array = bytes(valid_tokens)

        # UTF-8 decode (ignore errors)
        try:
            return byte_array.decode('utf-8', errors='ignore')
        except:
            return ""

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """
        Encode batch of texts

        Returns: [batch, max_seq_len] padded array
        """
        # Encode all
        encoded = [self.encode(text) for text in texts]

        # Find max length
        max_len = max(len(seq) for seq in encoded) if encoded else 0

        # Pad with 0
        batch = np.zeros((len(texts), max_len), dtype=np.int32)

        for i, seq in enumerate(encoded):
            batch[i, :len(seq)] = seq

        return batch

    def decode_batch(self, batch: np.ndarray) -> List[str]:
        """Decode batch of token sequences"""
        return [self.decode(seq.tolist()) for seq in batch]
