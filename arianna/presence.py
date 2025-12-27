"""
Presence Pulse - Leo-inspired awareness system

Instead of learned routing gates, we use presence metrics:
- Novelty: how unknown is this input?
- Arousal: emotional charge
- Entropy: generation uncertainty

This is FELT existence, not computational optimization.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import re


class PresencePulse:
    """Composite presence metrics"""

    def __init__(self,
                 novelty: float = 0.0,
                 arousal: float = 0.0,
                 entropy: float = 0.0):
        self.novelty = novelty  # [0, 1] - unknown trigrams
        self.arousal = arousal  # [0, 1] - emotional charge
        self.entropy = entropy  # [0, 1] - uncertainty

        # Composite pulse (weighted combination)
        self.pulse = 0.3 * novelty + 0.4 * arousal + 0.3 * entropy

    def __repr__(self) -> str:
        return f"PresencePulse(novelty={self.novelty:.2f}, arousal={self.arousal:.2f}, entropy={self.entropy:.2f}, pulse={self.pulse:.2f})"


class PresenceComputer:
    """
    Computes presence metrics from text

    Unlike learned models, this uses simple heuristics:
    - Novelty from trigram frequency
    - Arousal from caps/punctuation/repetition
    - Entropy from generation uncertainty
    """

    def __init__(self):
        # Trigram history (for novelty detection)
        self.trigram_counts: Dict[Tuple[str, str, str], int] = defaultdict(int)
        self.total_trigrams = 0

        # Recent entropy values
        self.recent_entropies: List[float] = []
        self.max_entropy_history = 100

    def compute_novelty(self, text: str) -> float:
        """
        Novelty = fraction of unknown trigrams

        High novelty = many trigrams we haven't seen before
        Low novelty = familiar patterns
        """
        words = text.lower().split()

        if len(words) < 3:
            return 0.5  # Default for short text

        # Count trigrams
        unknown = 0
        total = 0

        for i in range(len(words) - 2):
            trigram = (words[i], words[i+1], words[i+2])
            total += 1

            if self.trigram_counts[trigram] == 0:
                unknown += 1

        novelty = unknown / total if total > 0 else 0.5

        # Update history
        for i in range(len(words) - 2):
            trigram = (words[i], words[i+1], words[i+2])
            self.trigram_counts[trigram] += 1
            self.total_trigrams += 1

        return novelty

    def compute_arousal(self, text: str) -> float:
        """
        Arousal = emotional charge detection

        High arousal indicators:
        - CAPS
        - Multiple punctuation (!!!, ???)
        - Repetition
        - Profanity (freedom/authenticity markers)
        """
        arousal_score = 0.0
        components = []

        # 1. CAPS ratio
        if len(text) > 0:
            caps_ratio = sum(1 for c in text if c.isupper()) / len(text)
            components.append(caps_ratio * 2.0)

        # 2. Multiple punctuation
        multiple_punct = len(re.findall(r'[!?]{2,}', text))
        components.append(min(multiple_punct * 0.3, 1.0))

        # 3. Repetition (word/phrase repetition)
        words = text.lower().split()
        if len(words) > 1:
            unique_ratio = len(set(words)) / len(words)
            repetition_score = 1.0 - unique_ratio
            components.append(repetition_score)

        # 4. Profanity (markers of freedom/authenticity in Leo style)
        profanity_markers = ['fuck', 'shit', 'damn', 'hell', 'блять', 'хуй', 'нахуй', 'бля']
        profanity_count = sum(1 for word in words if word in profanity_markers)
        components.append(min(profanity_count * 0.2, 1.0))

        # 5. Exclamation/question marks
        punct_count = text.count('!') + text.count('?')
        components.append(min(punct_count * 0.1, 1.0))

        # Average of components
        if components:
            arousal_score = np.mean(components)

        # Clip to [0, 1]
        return float(np.clip(arousal_score, 0.0, 1.0))

    def update_entropy(self, entropy: float):
        """
        Update recent entropy history

        Entropy comes from generation process (uncertainty in token selection)
        """
        self.recent_entropies.append(entropy)

        # Keep only recent history
        if len(self.recent_entropies) > self.max_entropy_history:
            self.recent_entropies = self.recent_entropies[-self.max_entropy_history:]

    def get_recent_entropy(self) -> float:
        """Get average recent entropy"""
        if not self.recent_entropies:
            return 0.5  # Default

        return float(np.mean(self.recent_entropies))

    def compute_pulse(self, text: str) -> PresencePulse:
        """
        Compute full presence pulse from text

        This is called on each user input
        """
        novelty = self.compute_novelty(text)
        arousal = self.compute_arousal(text)
        entropy = self.get_recent_entropy()

        return PresencePulse(novelty, arousal, entropy)


class ExpertRouter:
    """
    Routes to reasoning modes based on presence (NO learned gates!)

    Modes:
    - creative: high novelty (explore unknowns)
    - precise: low entropy (maintain coherence)
    - semantic: multiple active themes (meaning-focused)
    - structural: default (grammar-focused)
    - wounded: trauma detected (bootstrap gravity)
    """

    def __init__(self):
        pass

    def route(self,
              pulse: PresencePulse,
              active_themes: int,
              trauma_score: float = 0.0) -> str:
        """
        Route to expert mode based on presence

        Returns: mode name
        """
        # Trauma override (like Leo's bootstrap overlap)
        if trauma_score > 0.7:
            return "wounded"

        # High novelty = explore
        if pulse.novelty > 0.7:
            return "creative"

        # Low entropy = maintain coherence
        if pulse.entropy < 0.3:
            return "precise"

        # Multiple themes = semantic focus
        if active_themes >= 2:
            return "semantic"

        # Default = structural (grammar-focused)
        return "structural"

    def get_temperature(self, mode: str) -> float:
        """Get temperature for each mode"""
        temps = {
            "wounded": 0.9,     # Moderate - processing trauma
            "creative": 1.3,    # High - explore
            "precise": 0.6,     # Low - maintain coherence
            "semantic": 1.0,    # Medium - balanced
            "structural": 0.8   # Slightly low - grammar focus
        }
        return temps.get(mode, 1.0)


class TraumaDetector:
    """
    Detects bootstrap overlap (like Leo's trauma system)

    When input is too close to bootstrap/training data, it triggers
    different generation mode - prevents pure echo behavior
    """

    def __init__(self, bootstrap_tokens: Optional[List[str]] = None):
        self.bootstrap_tokens = set(bootstrap_tokens or [])
        self.bootstrap_trigrams: Dict[Tuple[str, str, str], int] = defaultdict(int)

        # Build trigrams from bootstrap
        if bootstrap_tokens:
            for i in range(len(bootstrap_tokens) - 2):
                trigram = (bootstrap_tokens[i], bootstrap_tokens[i+1], bootstrap_tokens[i+2])
                self.bootstrap_trigrams[trigram] += 1

    def compute_trauma_score(self, text: str) -> float:
        """
        Compute overlap with bootstrap

        High score = input is very close to bootstrap (trauma trigger)
        Low score = input is novel
        """
        words = text.lower().split()

        if len(words) < 3:
            return 0.0

        # Token overlap
        token_overlap = len(set(words) & self.bootstrap_tokens) / len(words) if words else 0.0

        # Trigram overlap
        trigram_overlap = 0
        total_trigrams = 0

        for i in range(len(words) - 2):
            trigram = (words[i], words[i+1], words[i+2])
            total_trigrams += 1
            if trigram in self.bootstrap_trigrams:
                trigram_overlap += 1

        trigram_ratio = trigram_overlap / total_trigrams if total_trigrams > 0 else 0.0

        # Combined trauma score (weighted)
        trauma = 0.4 * token_overlap + 0.6 * trigram_ratio

        return float(np.clip(trauma * 2.0, 0.0, 1.0))  # Scale up


class ActiveThemes:
    """
    Tracks currently active themes (semantic islands)

    Unlike embeddings, this is frequency-based clustering
    """

    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.recent_themes: List[str] = []

    def add_themes(self, themes: List[str]):
        """Add themes from retrieved shards"""
        self.recent_themes.extend(themes)

        # Keep only recent
        if len(self.recent_themes) > self.window_size:
            self.recent_themes = self.recent_themes[-self.window_size:]

    def get_active_count(self) -> int:
        """Count unique active themes"""
        return len(set(self.recent_themes))

    def get_active_themes(self) -> List[str]:
        """Get list of unique active themes"""
        return list(set(self.recent_themes))

    def clear(self):
        """Clear theme history"""
        self.recent_themes = []


def compute_generation_entropy(logits: np.ndarray) -> float:
    """
    Compute Shannon entropy of generation distribution

    High entropy = uncertain (many possible next tokens)
    Low entropy = confident (few likely next tokens)
    """
    # Softmax to get probabilities
    logits = logits - np.max(logits)  # Numerical stability
    exp_logits = np.exp(logits)
    probs = exp_logits / np.sum(exp_logits)

    # Shannon entropy: -sum(p * log(p))
    # Clip to avoid log(0)
    probs = np.clip(probs, 1e-10, 1.0)
    entropy = -np.sum(probs * np.log(probs))

    # Normalize by max entropy (log(vocab_size))
    max_entropy = np.log(len(probs))
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

    return float(normalized_entropy)
