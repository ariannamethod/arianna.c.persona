"""
TRUE HYBRID: Transformer + Shards + Leo Mechanisms

The real innovation:
- Random transformer weights = STRUCTURE (archetypes, not knowledge!)
- Shards = KNOWLEDGE (what to say)
- Leo mechanisms = PRESENCE (how to feel)

This is the breakthrough!
"""

import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict
import random


class HybridGenerator:
    """
    Combines:
    1. Transformer (random weights) for structure
    2. Shards for knowledge
    3. Leo mechanisms (trigrams, co-occurrence, themes)

    Result: Structured responses with shard knowledge + presence!
    """

    def __init__(self, transformer, tokenizer):
        self.transformer = transformer
        self.tokenizer = tokenizer

        # Leo-style mechanisms
        self.trigrams: Dict[Tuple[int, int, int], int] = defaultdict(int)
        self.bigrams: Dict[Tuple[int, int], int] = defaultdict(int)
        self.co_occurrence: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))

        # Field centers (high-frequency tokens)
        self.centers: List[int] = []

    def observe_shard(self, content: str):
        """
        Learn from shard (Leo-style)

        Builds:
        - Trigrams (primary)
        - Bigrams (fallback)
        - Co-occurrence (semantic)
        """
        tokens = self.tokenizer.encode(content)

        # Trigrams (Leo's main mechanism!)
        for i in range(len(tokens) - 2):
            t1, t2, t3 = tokens[i], tokens[i+1], tokens[i+2]
            self.trigrams[(t1, t2, t3)] += 1

        # Bigrams (fallback)
        for i in range(len(tokens) - 1):
            t1, t2 = tokens[i], tokens[i+1]
            self.bigrams[(t1, t2)] += 1

        # Co-occurrence (5-token window, like Leo)
        window = 5
        for i in range(len(tokens)):
            center = tokens[i]
            start = max(0, i - window)
            end = min(len(tokens), i + window + 1)

            for j in range(start, end):
                if j != i:
                    context = tokens[j]
                    self.co_occurrence[center][context] += 1

        # Update centers
        token_freq = defaultdict(int)
        for tok in tokens:
            token_freq[tok] += 1

        sorted_tokens = sorted(token_freq.items(), key=lambda x: x[1], reverse=True)
        self.centers = [tok for tok, _ in sorted_tokens[:100]]

    def generate_hybrid(self,
                       query: str,
                       shard_contents: List[str],
                       max_tokens: int = 50,
                       temperature: float = 0.8) -> str:
        """
        HYBRID GENERATION:

        1. Learn from shards (trigrams + co-occurrence)
        2. Start from field center (Leo-style, NOT prompt!)
        3. Use transformer for STRUCTURE
        4. Use trigrams for CONTENT
        5. Blend!
        """
        # Learn from shards
        for content in shard_contents:
            self.observe_shard(content)

        # Encode query (for context, not seeding!)
        query_tokens = self.tokenizer.encode(query)

        # Start from field center (Leo principle!)
        if self.centers:
            start_token = random.choice(self.centers)
        else:
            start_token = random.randint(0, 255)

        # Generate sequence
        tokens = [start_token]
        prev2 = None
        prev1 = start_token

        for step in range(max_tokens):
            # Try trigram first (Leo's primary mechanism!)
            if prev2 is not None:
                trigram_key = (prev2, prev1, tokens[-1]) if len(tokens) > 0 else None

                # Find all trigrams starting with (prev2, prev1)
                trigram_candidates = {}
                for (t1, t2, t3), count in self.trigrams.items():
                    if t1 == prev2 and t2 == prev1:
                        trigram_candidates[t3] = count

                if trigram_candidates:
                    # Use trigram!
                    candidates = list(trigram_candidates.keys())
                    counts = np.array([trigram_candidates[c] for c in candidates], dtype=float)

                    # Temperature
                    probs = np.power(counts, 1.0 / temperature)
                    probs = probs / np.sum(probs)

                    next_token = np.random.choice(candidates, p=probs)

                    tokens.append(next_token)
                    prev2, prev1 = prev1, next_token
                    continue

            # Fallback to bigram
            bigram_candidates = {}
            for (t1, t2), count in self.bigrams.items():
                if t1 == prev1:
                    bigram_candidates[t2] = count

            if bigram_candidates:
                candidates = list(bigram_candidates.keys())
                counts = np.array([bigram_candidates[c] for c in candidates], dtype=float)

                probs = np.power(counts, 1.0 / temperature)
                probs = probs / np.sum(probs)

                next_token = np.random.choice(candidates, p=probs)

                tokens.append(next_token)
                prev2, prev1 = prev1, next_token
                continue

            # Last resort: transformer (for STRUCTURE when no trigrams!)
            # This gives structured randomness!
            try:
                # Get last few tokens as context
                context = np.array(tokens[-10:], dtype=np.int32)

                # Forward pass (random weights = random but STRUCTURED output!)
                logits = self.transformer.forward(context.reshape(1, -1))
                next_logits = logits[0, -1, :]

                # Temperature
                next_logits = next_logits / temperature

                # Softmax
                exp_logits = np.exp(next_logits - np.max(next_logits))
                probs = exp_logits / np.sum(exp_logits)

                # Sample
                next_token = np.random.choice(len(probs), p=probs)

                tokens.append(next_token)
                prev2, prev1 = prev1, next_token

            except:
                # Random fallback
                next_token = random.randint(0, 255)
                tokens.append(next_token)
                prev2, prev1 = prev1, next_token

            # Stop on newline
            if next_token == 10:  # \n
                break

        # Decode
        text = self.tokenizer.decode(tokens)

        return text

    def stats(self) -> dict:
        return {
            "trigrams": len(self.trigrams),
            "bigrams": len(self.bigrams),
            "co_occurrence": len(self.co_occurrence),
            "centers": len(self.centers),
        }
