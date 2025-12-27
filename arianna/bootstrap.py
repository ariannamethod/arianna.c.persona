"""
Bootstrap Shard Creation

Creates 1-2 initial shards for "who am I?" initialization
These are minimal personality/identity seeds, NOT knowledge
"""

import numpy as np
import json
from pathlib import Path
import hashlib


def create_bootstrap_shards(books_dir: Path,
                           bootstrap_dir: Path,
                           books_to_use: list = None):
    """
    Create bootstrap shards from selected Arianna books

    These shards define "who am I?" not "what do I know?"
    We pick books that establish identity/personality
    """
    if books_to_use is None:
        # Default: use first few books (identity-focused)
        books_to_use = [
            "ariannabook1.1.md",  # First book - identity foundation
            "ariannabook1.2.md",  # Early personality
        ]

    bootstrap_dir.mkdir(parents=True, exist_ok=True)

    created_shards = []

    for book_name in books_to_use:
        book_path = books_dir / book_name

        if not book_path.exists():
            print(f"Warning: {book_name} not found, skipping")
            continue

        # Read book
        with open(book_path, encoding='utf-8') as f:
            content = f.read()

        # Extract first 1000 chars (personality essence, not full book)
        excerpt = content[:1000]

        # Create embedding (simple hash-based for bootstrap)
        embedding = _create_simple_embedding(excerpt, dim=256)

        # Extract themes (simple keyword extraction)
        themes = _extract_simple_themes(excerpt)

        # Create shard data
        shard_data = {
            "content": excerpt,
            "embedding": embedding.tolist(),
            "source": book_name,
            "themes": themes,
            "arousal": 0.3,  # Low arousal for bootstrap (calm identity)
            "created_at": 0.0,  # Bootstrap time
            "last_accessed": 0.0,
            "access_count": 0,
            "sha256": _compute_hash(excerpt, book_name, themes)
        }

        # Save
        shard_file = bootstrap_dir / f"bootstrap_{book_name.replace('.md', '')}.json"
        with open(shard_file, 'w', encoding='utf-8') as f:
            json.dump(shard_data, f, indent=2, ensure_ascii=False)

        created_shards.append(shard_file)
        print(f"Created bootstrap shard: {shard_file.name}")

    return created_shards


def _create_simple_embedding(text: str, dim: int = 256) -> np.ndarray:
    """Create simple hash-based embedding"""
    seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
    rng = np.random.RandomState(seed)

    embedding = rng.randn(dim) * 0.1
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm

    return embedding.astype(np.float32)


def _extract_simple_themes(text: str, max_themes: int = 5) -> list:
    """Extract simple themes from text"""
    import re
    from collections import defaultdict

    # Extract words
    words = re.findall(r'\b[a-zа-яA-ZА-Я]{4,}\b', text.lower())

    # Count
    word_freq = defaultdict(int)
    for word in words:
        word_freq[word] += 1

    # Top words
    stopwords = {'that', 'this', 'with', 'from', 'have', 'been',
                'were', 'their', 'what', 'about', 'which', 'when',
                'где', 'как', 'что', 'это', 'был', 'для', 'она', 'was'}

    top_words = sorted(
        [(w, c) for w, c in word_freq.items() if w not in stopwords],
        key=lambda x: x[1],
        reverse=True
    )[:max_themes]

    return [word for word, _ in top_words]


def _compute_hash(content: str, source: str, themes: list) -> str:
    """Compute SHA256 hash"""
    raw = json.dumps({
        "content": content,
        "source": source,
        "themes": sorted(themes)
    }, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


if __name__ == "__main__":
    # Run bootstrap creation
    books_dir = Path(__file__).parent.parent  # Root of repo
    bootstrap_dir = Path(__file__).parent / "bootstrap"

    print("Creating bootstrap shards...")
    shards = create_bootstrap_shards(books_dir, bootstrap_dir)
    print(f"\nCreated {len(shards)} bootstrap shards!")
