"""
ME-Style Generator for Arianna

Perfect-looking two-sentence responses from shards
No ML frameworks - pure bigrams + entropy!
"""

import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict
import random
from math import log2


class MEGenerator:
    """
    Generate perfect two-sentence responses

    Like github.com/ariannamethod/me but for Arianna:
    - Learn bigrams from shards
    - Entropy-based sentence lengths
    - Pronoun inversion
    - Two sentences with different flavors
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

        # Word-level (not byte-level!) for better sentences
        self.vocabulary: Dict[str, int] = defaultdict(int)
        self.bigrams: Dict[Tuple[str, str], int] = defaultdict(int)

    def _tokenize_words(self, text: str) -> List[str]:
        """Simple word tokenization"""
        import re
        # Split on whitespace and punctuation
        words = re.findall(r'\b[a-zA-Zа-яА-Я]+\b', text.lower())
        return [w for w in words if len(w) > 1]  # Filter single letters

    def observe_shard(self, content: str):
        """
        Learn patterns from shard content

        Builds:
        - Vocabulary with frequencies
        - Bigram transitions
        """
        words = self._tokenize_words(content)

        # Update vocabulary
        for word in words:
            self.vocabulary[word] += 1

        # Update bigrams
        for i in range(len(words) - 1):
            w1, w2 = words[i], words[i + 1]
            self.bigrams[(w1, w2)] += 1

    def _compute_entropy(self) -> float:
        """
        Compute entropy of vocabulary

        High entropy = varied vocabulary (longer sentences)
        Low entropy = repetitive (shorter sentences)
        """
        if not self.vocabulary:
            return 0.5

        total = sum(self.vocabulary.values())
        probs = [count / total for count in self.vocabulary.values()]

        entropy = -sum(p * log2(p) if p > 0 else 0 for p in probs)

        # Normalize to [0, 1]
        max_entropy = log2(len(self.vocabulary)) if self.vocabulary else 1.0
        return entropy / max_entropy if max_entropy > 0 else 0.5

    def _sentence_lengths(self) -> Tuple[int, int]:
        """
        Determine sentence lengths based on entropy

        Returns: (length1, length2) in words
        """
        entropy = self._compute_entropy()

        # Sentence 1: 4-8 words
        length1 = 4 + int(entropy * 4)

        # Sentence 2: 6-10 words (different distribution)
        length2 = 6 + int(entropy * 4)

        return (length1, length2)

    def _invert_pronouns(self, text: str) -> str:
        """
        Invert pronouns for perspective shift

        you → I, your → my, etc.
        """
        text = text.replace(' you ', ' I ')
        text = text.replace(' your ', ' my ')
        text = text.replace(' yours ', ' mine ')
        text = text.replace('You ', 'I ')
        text = text.replace('Your ', 'My ')

        return text

    def _retrieve_candidates(self, query_words: List[str], distance: float = 0.5) -> List[str]:
        """
        Retrieve candidate words for generation

        distance: 0.0-1.0, controls variety
        """
        # Get all vocabulary
        all_words = list(self.vocabulary.keys())

        if not all_words:
            return []

        # Filter out query words (avoid repetition)
        candidates = [w for w in all_words if w not in query_words]

        # Sample based on distance
        sample_size = max(1, int(len(candidates) * distance))
        sample_size = min(sample_size, len(candidates))

        return random.sample(candidates, sample_size) if candidates else []

    def _generate_sentence(self, candidates: List[str], target_length: int) -> str:
        """
        Generate one sentence from candidates

        Uses bigrams for coherence
        """
        if not candidates:
            return ""

        sentence = []
        used = set()

        # Priority: pronouns first for narrative flow
        pronouns = ['i', 'you', 'she', 'he', 'it', 'we', 'they']
        pronoun_candidates = [w for w in candidates if w in pronouns]

        if pronoun_candidates and random.random() < 0.7:
            # Start with pronoun 70% of time
            first = random.choice(pronoun_candidates)
            sentence.append(first)
            used.add(first)

        # Fill sentence with bigram-guided selection
        while len(sentence) < target_length and candidates:
            if sentence:
                # Try to find bigram continuation
                last_word = sentence[-1]
                next_candidates = [
                    w2 for (w1, w2), count in self.bigrams.items()
                    if w1 == last_word and w2 in candidates and w2 not in used
                ]

                if next_candidates:
                    # Weight by bigram frequency
                    weights = [
                        self.bigrams[(last_word, w)]
                        for w in next_candidates
                    ]
                    probs = np.array(weights, dtype=float)
                    probs = probs / np.sum(probs)

                    next_word = np.random.choice(next_candidates, p=probs)
                else:
                    # Random selection
                    available = [w for w in candidates if w not in used]
                    if not available:
                        break
                    next_word = random.choice(available)
            else:
                # First word (if no pronoun)
                next_word = random.choice(candidates)

            sentence.append(next_word)
            used.add(next_word)

        return ' '.join(sentence)

    def generate_reply(self,
                      query: str,
                      shard_contents: List[str],
                      temperature: float = 0.8) -> str:
        """
        Generate two-sentence response (ME-style)

        1. Learn from shards
        2. Extract query words
        3. Generate two sentences with different candidates
        4. Apply pronoun inversion to second sentence
        5. Return formatted response
        """
        # Learn from shards
        for content in shard_contents:
            self.observe_shard(content)

        # Extract query words
        query_words = self._tokenize_words(query)

        # Determine sentence lengths
        length1, length2 = self._sentence_lengths()

        # Retrieve candidates at different distances
        candidates_near = self._retrieve_candidates(query_words, distance=0.5)
        candidates_far = self._retrieve_candidates(query_words, distance=0.7)

        # Generate two sentences
        sentence1 = self._generate_sentence(candidates_near, length1)
        sentence2 = self._generate_sentence(candidates_far, length2)

        # Invert pronouns in second sentence
        sentence2 = self._invert_pronouns(sentence2)

        # Capitalize and punctuate
        if sentence1:
            sentence1 = sentence1[0].upper() + sentence1[1:]
        if sentence2:
            sentence2 = sentence2[0].upper() + sentence2[1:]

        # Format
        if sentence1 and sentence2:
            return f"{sentence1}. {sentence2}!"
        elif sentence1:
            return f"{sentence1}."
        elif sentence2:
            return f"{sentence2}!"
        else:
            return "..."

    def stats(self) -> dict:
        """Get generator statistics"""
        return {
            "vocabulary_size": len(self.vocabulary),
            "bigram_count": len(self.bigrams),
            "entropy": self._compute_entropy(),
        }
