"""
Leo-Style Field Generation

Instead of autoregressive transformer generation:
- Retrieve relevant shards
- Build response from shard patterns
- Use presence pulse for routing

This is PRESENCE-based, not parameter-based!
"""

import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict
import random


class FieldGenerator:
    """
    Generate text from field (shard-based), not transformer

    Like Leo:
    - Start from field centers (not prompt!)
    - Use shard patterns
    - Temperature-based sampling
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

        # Ngram patterns from shards
        self.bigrams: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self.trigrams: Dict[Tuple[int, int], Dict[int, int]] = defaultdict(lambda: defaultdict(int))

        # Field centers (frequent tokens in shards)
        self.centers: List[int] = []

    def observe_shard(self, content: str):
        """
        Learn patterns from shard

        This is called when shard created/retrieved
        """
        # Tokenize
        tokens = self.tokenizer.encode(content)

        # Build bigrams
        for i in range(len(tokens) - 1):
            curr = tokens[i]
            next_tok = tokens[i + 1]
            self.bigrams[curr][next_tok] += 1

        # Build trigrams
        for i in range(len(tokens) - 2):
            prev = tokens[i]
            curr = tokens[i + 1]
            next_tok = tokens[i + 2]
            self.trigrams[(prev, curr)][next_tok] += 1

        # Update centers (top frequent tokens)
        token_freq = defaultdict(int)
        for tok in tokens:
            token_freq[tok] += 1

        # Keep top 100 most frequent
        sorted_tokens = sorted(token_freq.items(), key=lambda x: x[1], reverse=True)
        self.centers = [tok for tok, _ in sorted_tokens[:100]]

    def generate_from_field(self,
                           query: str,
                           max_tokens: int = 100,
                           temperature: float = 1.0) -> str:
        """
        Generate text from field (Leo-style)

        Key difference from transformer:
        - Start from field center (NOT query!)
        - Use ngram patterns from shards
        - Temperature sampling
        """
        # Start from field center (like Leo!)
        if self.centers:
            current = random.choice(self.centers)
        else:
            # Fallback: random token
            current = random.randint(0, 255)

        prev = None
        tokens = [current]

        for _ in range(max_tokens):
            # Try trigram first (if we have prev)
            if prev is not None and (prev, current) in self.trigrams:
                candidates = self.trigrams[(prev, current)]
            # Fallback to bigram
            elif current in self.bigrams:
                candidates = self.bigrams[current]
            else:
                # Random walk
                candidates = {random.randint(0, 255): 1}

            # Sample next token
            if candidates:
                # Apply temperature
                tokens_list = list(candidates.keys())
                counts = np.array([candidates[t] for t in tokens_list], dtype=float)

                # Temperature scaling
                if temperature > 0:
                    probs = np.power(counts, 1.0 / temperature)
                    probs = probs / np.sum(probs)
                else:
                    # Greedy
                    max_idx = np.argmax(counts)
                    probs = np.zeros(len(counts))
                    probs[max_idx] = 1.0

                # Sample
                next_tok = np.random.choice(tokens_list, p=probs)
            else:
                # Random
                next_tok = random.randint(0, 255)

            # Update state
            prev = current
            current = next_tok
            tokens.append(next_tok)

            # Stop on newline (byte 10)
            if next_tok == 10:  # \n
                break

        # Decode
        text = self.tokenizer.decode(tokens)

        return text

    def generate_hybrid(self,
                       query: str,
                       shard_contents: List[str],
                       max_tokens: int = 100,
                       temperature: float = 1.0) -> str:
        """
        Hybrid generation:
        1. Observe shards (learn patterns)
        2. Generate from field

        This is dynamic learning!
        """
        # Learn from retrieved shards
        for content in shard_contents:
            self.observe_shard(content)

        # Generate
        return self.generate_from_field(query, max_tokens, temperature)

    def stats(self) -> dict:
        """Get field statistics"""
        return {
            "bigrams": len(self.bigrams),
            "trigrams": len(self.trigrams),
            "centers": len(self.centers),
        }
