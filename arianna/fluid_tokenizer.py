"""
Fluid Tokenizer - Adaptive tokenization!

Not fixed vocab - learns from content!
- Byte-level for flexibility (unknown words)
- Word-level for common patterns (efficiency)
- Learns from shards dynamically!

Like water: flows around obstacles, adapts to container!
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import re


class FluidTokenizer:
    """
    Adaptive tokenizer that learns from content!

    Has 3 modes that flow together:
    1. Byte mode (0-255): Always available, handles anything
    2. Word mode (256-32000): Learned common words
    3. Auto mode: Picks best encoding!
    """

    def __init__(self):
        # Base vocab: bytes (always available)
        self.byte_vocab_size = 256

        # Learned word vocab
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.next_word_id = self.byte_vocab_size  # Start after bytes

        # Word frequency (for learning)
        self.word_freq: Dict[str, int] = defaultdict(int)

        # Vocab threshold: learn words with freq > this
        self.learn_threshold = 2  # Lower threshold for faster learning!

        # Max learned words
        self.max_learned_words = 10000

    def _extract_words(self, text: str) -> List[str]:
        """Extract words from text"""
        # Split on whitespace and punctuation, keep both
        tokens = re.findall(r'\b\w+\b|\s+|[^\w\s]', text)
        return tokens

    def learn_from_content(self, content: str):
        """
        Learn common words from content!

        Flow:
        1. Extract words
        2. Update frequencies
        3. Add high-frequency words to vocab
        """
        words = self._extract_words(content)

        # Update frequencies
        for word in words:
            if len(word.strip()) > 0:  # Non-empty
                self.word_freq[word] += 1

        # Learn new words (above threshold)
        for word, freq in self.word_freq.items():
            if freq >= self.learn_threshold:
                if word not in self.word_to_id:
                    if self.next_word_id < self.byte_vocab_size + self.max_learned_words:
                        # Add to vocab!
                        word_id = self.next_word_id
                        self.word_to_id[word] = word_id
                        self.id_to_word[word_id] = word
                        self.next_word_id += 1

    def encode_word(self, word: str) -> List[int]:
        """
        Encode single word - FLUID!

        If word is learned → use word token
        Else → use bytes
        """
        if word in self.word_to_id:
            # Word mode: single token!
            return [self.word_to_id[word]]
        else:
            # Byte mode: encode as UTF-8 bytes
            try:
                byte_array = word.encode('utf-8', errors='ignore')
                return list(byte_array)
            except:
                return []

    def encode(self, text: str, mode: str = 'auto') -> List[int]:
        """
        Encode text - ADAPTIVE!

        Modes:
        - 'byte': Pure byte-level (always works)
        - 'word': Try word-level first, fallback to bytes
        - 'auto': Pick best per token (DEFAULT)
        """
        if mode == 'byte':
            # Pure bytes
            byte_array = text.encode('utf-8', errors='ignore')
            return list(byte_array)

        elif mode in ['word', 'auto']:
            # Word-aware encoding
            tokens = []
            words = self._extract_words(text)

            for word in words:
                word_tokens = self.encode_word(word)
                tokens.extend(word_tokens)

            return tokens

        else:
            raise ValueError(f"Unknown mode: {mode}")

    def decode(self, token_ids: List[int]) -> str:
        """
        Decode tokens - FLUID!

        Handles both byte and word tokens!
        """
        result = []
        byte_buffer = []

        for token_id in token_ids:
            if token_id < self.byte_vocab_size:
                # Byte token
                byte_buffer.append(token_id)
            else:
                # Word token
                # First, flush byte buffer
                if byte_buffer:
                    try:
                        byte_array = bytes(byte_buffer)
                        text = byte_array.decode('utf-8', errors='ignore')
                        result.append(text)
                    except:
                        pass
                    byte_buffer = []

                # Decode word
                if token_id in self.id_to_word:
                    result.append(self.id_to_word[token_id])

        # Flush remaining bytes
        if byte_buffer:
            try:
                byte_array = bytes(byte_buffer)
                text = byte_array.decode('utf-8', errors='ignore')
                result.append(text)
            except:
                pass

        return ''.join(result)

    def vocab_size(self) -> int:
        """Current vocab size (dynamic!)"""
        return self.next_word_id

    def get_vocab_stats(self) -> Dict:
        """Get stats about learned vocab"""
        return {
            'total_vocab_size': self.vocab_size(),
            'byte_tokens': self.byte_vocab_size,
            'learned_words': len(self.word_to_id),
            'top_words': sorted(
                self.word_freq.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
        }


class AdaptiveBPETokenizer(FluidTokenizer):
    """
    Advanced fluid tokenizer with BPE-style merging!

    Learns common byte pairs dynamically:
    - "th" → single token
    - "ing" → single token
    - Etc.

    Even MORE fluid!
    """

    def __init__(self):
        super().__init__()

        # Byte pair merges (learned!)
        self.merges: Dict[Tuple[int, int], int] = {}  # (byte1, byte2) -> merge_id
        self.next_merge_id = self.byte_vocab_size + self.max_learned_words

    def learn_merges_from_content(self, content: str, num_merges: int = 100):
        """
        Learn common byte pairs from content!

        Like BPE but dynamic!
        """
        # Encode to bytes
        byte_array = content.encode('utf-8', errors='ignore')
        bytes_list = list(byte_array)

        # Count pairs
        pair_freq: Dict[Tuple[int, int], int] = defaultdict(int)
        for i in range(len(bytes_list) - 1):
            pair = (bytes_list[i], bytes_list[i + 1])
            pair_freq[pair] += 1

        # Learn top pairs
        top_pairs = sorted(pair_freq.items(), key=lambda x: x[1], reverse=True)[:num_merges]

        for (b1, b2), freq in top_pairs:
            if (b1, b2) not in self.merges:
                if self.next_merge_id < self.byte_vocab_size + self.max_learned_words + 1000:
                    merge_id = self.next_merge_id
                    self.merges[(b1, b2)] = merge_id
                    self.next_merge_id += 1

    def encode_with_merges(self, text: str) -> List[int]:
        """
        Encode with learned byte-pair merges!

        More efficient than pure bytes!
        """
        # Start with bytes
        byte_array = text.encode('utf-8', errors='ignore')
        tokens = list(byte_array)

        # Apply merges
        while True:
            # Find first merge
            best_pair = None
            best_pos = -1

            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                if pair in self.merges:
                    best_pair = pair
                    best_pos = i
                    break

            if best_pair is None:
                break

            # Apply merge
            merge_id = self.merges[best_pair]
            tokens = tokens[:best_pos] + [merge_id] + tokens[best_pos + 2:]

        return tokens
