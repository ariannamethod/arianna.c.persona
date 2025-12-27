"""
Numpy Shard Manager - Leo-inspired dynamic knowledge system

Unlike traditional embeddings, shards are created on-the-fly during conversation
and forgotten when topics die. This is PRESENCE, not persistent memory.
"""

import numpy as np
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import time


class NumpyShard:
    """A single knowledge shard - created dynamically, lives temporarily"""

    def __init__(self,
                 content: str,
                 embedding: np.ndarray,
                 source: str,
                 themes: List[str],
                 arousal: float = 0.0):
        self.content = content
        self.embedding = embedding  # Shape: (embedding_dim,)
        self.source = source  # Which book/conversation
        self.themes = themes
        self.arousal = arousal
        self.created_at = time.time()
        self.last_accessed = time.time()
        self.access_count = 0
        self.sha256 = self._compute_hash()

    def _compute_hash(self) -> str:
        """SHA256 of content for indexing"""
        raw = json.dumps({
            "content": self.content,
            "source": self.source,
            "themes": sorted(self.themes)
        }, sort_keys=True).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:16]

    def access(self):
        """Mark shard as accessed - keeps it alive"""
        self.last_accessed = time.time()
        self.access_count += 1

    def age_seconds(self) -> float:
        """Time since creation"""
        return time.time() - self.created_at

    def idle_seconds(self) -> float:
        """Time since last access"""
        return time.time() - self.last_accessed

    def to_dict(self) -> dict:
        """Serialize for storage"""
        return {
            "content": self.content,
            "embedding": self.embedding.tolist(),
            "source": self.source,
            "themes": self.themes,
            "arousal": self.arousal,
            "created_at": self.created_at,
            "last_accessed": self.last_accessed,
            "access_count": self.access_count,
            "sha256": self.sha256
        }

    @classmethod
    def from_dict(cls, data: dict) -> "NumpyShard":
        """Deserialize from storage"""
        shard = cls(
            content=data["content"],
            embedding=np.array(data["embedding"]),
            source=data["source"],
            themes=data["themes"],
            arousal=data["arousal"]
        )
        shard.created_at = data["created_at"]
        shard.last_accessed = data["last_accessed"]
        shard.access_count = data["access_count"]
        shard.sha256 = data["sha256"]
        return shard


class ShardManager:
    """
    Manages dynamic numpy shards - Leo-inspired architecture

    Key principles:
    - Shards created on-the-fly when topics emerge
    - Shards forgotten when topics die (LRU eviction)
    - No persistent embeddings database
    - Presence > Intelligence
    """

    def __init__(self,
                 shard_dir: Path,
                 max_shards: int = 256,
                 max_idle_seconds: float = 3600.0,  # 1 hour
                 embedding_dim: int = 256):
        self.shard_dir = Path(shard_dir)
        self.shard_dir.mkdir(parents=True, exist_ok=True)

        self.max_shards = max_shards
        self.max_idle_seconds = max_idle_seconds
        self.embedding_dim = embedding_dim

        # Active shards in memory
        self.shards: Dict[str, NumpyShard] = {}

        # Theme index for fast retrieval
        self.theme_index: Dict[str, List[str]] = defaultdict(list)

        # Co-occurrence tracking (Leo-style)
        self.co_occurrence: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        # Load bootstrap shards if they exist
        self._load_bootstrap()

    def _load_bootstrap(self):
        """Load minimal bootstrap shards for 'who am I?' initialization"""
        bootstrap_dir = self.shard_dir.parent / "bootstrap"
        if not bootstrap_dir.exists():
            return

        for shard_file in bootstrap_dir.glob("*.json"):
            with open(shard_file) as f:
                data = json.load(f)
                shard = NumpyShard.from_dict(data)
                self.shards[shard.sha256] = shard

                # Update theme index
                for theme in shard.themes:
                    self.theme_index[theme].append(shard.sha256)

    def create_shard(self,
                     content: str,
                     source: str,
                     themes: List[str],
                     arousal: float = 0.0) -> NumpyShard:
        """
        Create a new shard dynamically

        This is called when:
        - A book/topic is mentioned in conversation
        - Quality threshold is met (like Leo's quality > 0.6)
        - New knowledge needs to be temporarily stored
        """
        # Create simple embedding (for now, random - will improve later)
        # In full implementation, this would be derived from content analysis
        embedding = self._create_embedding(content)

        shard = NumpyShard(
            content=content,
            embedding=embedding,
            source=source,
            themes=themes,
            arousal=arousal
        )

        # Store in memory
        self.shards[shard.sha256] = shard

        # Update theme index
        for theme in themes:
            self.theme_index[theme].append(shard.sha256)

        # Update co-occurrence (themes that appear together)
        for i, theme_a in enumerate(themes):
            for theme_b in themes[i+1:]:
                self.co_occurrence[theme_a][theme_b] += 1
                self.co_occurrence[theme_b][theme_a] += 1

        # Evict old shards if needed
        self._evict_if_needed()

        return shard

    def _create_embedding(self, content: str) -> np.ndarray:
        """
        Create embedding from content

        For now: random initialization (reasoning engine will make sense of it)
        Later: could use co-occurrence patterns like Leo
        """
        # Simple hash-based seeding for reproducibility
        seed = int(hashlib.sha256(content.encode()).hexdigest()[:8], 16)
        rng = np.random.RandomState(seed)

        # Small random embedding
        embedding = rng.randn(self.embedding_dim) * 0.1

        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding.astype(np.float32)

    def retrieve_by_themes(self,
                          themes: List[str],
                          top_k: int = 5) -> List[NumpyShard]:
        """
        Retrieve shards by theme resonance (Leo-style)

        Unlike vector similarity, this uses theme overlap + co-occurrence
        """
        candidates = set()

        # Direct theme matches
        for theme in themes:
            candidates.update(self.theme_index.get(theme, []))

        # Co-occurring themes
        for theme in themes:
            for related_theme, count in self.co_occurrence.get(theme, {}).items():
                if count >= 2:  # At least 2 co-occurrences
                    candidates.update(self.theme_index.get(related_theme, []))

        # Score candidates
        scored = []
        for sha256 in candidates:
            if sha256 not in self.shards:
                continue

            shard = self.shards[sha256]

            # Theme overlap score
            overlap = len(set(themes) & set(shard.themes))

            # Recency bonus (more recent = higher score)
            recency = 1.0 / (1.0 + shard.idle_seconds() / 3600.0)

            # Access count bonus (frequently accessed = important)
            popularity = np.log1p(shard.access_count)

            # Arousal bonus (emotional content = more memorable)
            arousal_bonus = shard.arousal

            # Combined score
            score = overlap + 0.3 * recency + 0.2 * popularity + 0.1 * arousal_bonus

            scored.append((score, shard))

        # Sort by score
        scored.sort(reverse=True, key=lambda x: x[0])

        # Return top-k
        results = [shard for _, shard in scored[:top_k]]

        # Mark as accessed
        for shard in results:
            shard.access()

        return results

    def retrieve_by_content(self,
                           query_embedding: np.ndarray,
                           top_k: int = 5) -> List[NumpyShard]:
        """
        Retrieve by embedding similarity (traditional approach)

        This is a fallback - theme-based retrieval is preferred
        """
        if len(self.shards) == 0:
            return []

        # Compute similarities
        scored = []
        for sha256, shard in self.shards.items():
            similarity = np.dot(query_embedding, shard.embedding)
            scored.append((similarity, shard))

        # Sort by similarity
        scored.sort(reverse=True, key=lambda x: x[0])

        # Return top-k
        results = [shard for _, shard in scored[:top_k]]

        # Mark as accessed
        for shard in results:
            shard.access()

        return results

    def _evict_if_needed(self):
        """
        Evict old/idle shards (LRU-style)

        Shards are forgotten when:
        - Too many shards exist (> max_shards)
        - Shard hasn't been accessed recently (> max_idle_seconds)
        """
        # Remove idle shards
        to_remove = []
        for sha256, shard in self.shards.items():
            if shard.idle_seconds() > self.max_idle_seconds:
                to_remove.append(sha256)

        for sha256 in to_remove:
            self._evict_shard(sha256)

        # If still too many, remove least recently used
        if len(self.shards) > self.max_shards:
            # Sort by last access
            sorted_shards = sorted(
                self.shards.items(),
                key=lambda x: x[1].last_accessed
            )

            # Remove oldest
            num_to_remove = len(self.shards) - self.max_shards
            for sha256, _ in sorted_shards[:num_to_remove]:
                self._evict_shard(sha256)

    def _evict_shard(self, sha256: str):
        """Remove a shard from memory"""
        if sha256 not in self.shards:
            return

        shard = self.shards[sha256]

        # Remove from theme index
        for theme in shard.themes:
            if theme in self.theme_index:
                self.theme_index[theme] = [
                    s for s in self.theme_index[theme] if s != sha256
                ]

        # Remove from memory
        del self.shards[sha256]

    def save_shard(self, sha256: str):
        """Persist a shard to disk (for important shards)"""
        if sha256 not in self.shards:
            return

        shard = self.shards[sha256]
        shard_file = self.shard_dir / f"{sha256}.json"

        with open(shard_file, "w") as f:
            json.dump(shard.to_dict(), f, indent=2)

    def load_shard(self, sha256: str) -> Optional[NumpyShard]:
        """Load a shard from disk"""
        shard_file = self.shard_dir / f"{sha256}.json"

        if not shard_file.exists():
            return None

        with open(shard_file) as f:
            data = json.load(f)
            shard = NumpyShard.from_dict(data)

            # Add to memory
            self.shards[sha256] = shard

            # Update theme index
            for theme in shard.themes:
                self.theme_index[theme].append(sha256)

            return shard

    def get_active_themes(self) -> List[Tuple[str, int]]:
        """
        Get currently active themes (for presence awareness)

        Returns: List of (theme, count) sorted by frequency
        """
        theme_counts = defaultdict(int)

        for shard in self.shards.values():
            for theme in shard.themes:
                theme_counts[theme] += 1

        return sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)

    def stats(self) -> dict:
        """Get statistics about current shard state"""
        return {
            "total_shards": len(self.shards),
            "max_shards": self.max_shards,
            "active_themes": len(self.theme_index),
            "avg_access_count": np.mean([s.access_count for s in self.shards.values()]) if self.shards else 0,
            "oldest_shard_age": max([s.age_seconds() for s in self.shards.values()]) if self.shards else 0,
            "newest_shard_age": min([s.age_seconds() for s in self.shards.values()]) if self.shards else 0,
        }
