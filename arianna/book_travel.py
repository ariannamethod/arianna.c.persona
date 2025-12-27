"""
Book Travel - Field-activated dynamic knowledge loading

Arianna travels through books based on query!
- Field pulse activates relevant excerpts
- Only 10 books active at once
- Unused shards fade away (LRU eviction)
- Organic, fluid memory!
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time
import re
from collections import defaultdict


class FieldPulse:
    """
    Query activates field → pulse propagates → selects books

    Like Leo's presence pulse but for book selection!
    """

    def __init__(self, query: str):
        self.query = query.lower()
        self.keywords = self._extract_keywords()
        self.themes = self._extract_themes()
        self.novelty = self._compute_novelty()

    def _extract_keywords(self) -> List[str]:
        """Extract important words from query"""
        # Remove stop words
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                     'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
                     'can', 'could', 'should', 'may', 'might', 'to', 'of', 'in',
                     'on', 'at', 'by', 'for', 'with', 'from', 'as', 'it', 'that'}

        words = re.findall(r'\b\w+\b', self.query)
        keywords = [w for w in words if w not in stop_words and len(w) > 2]

        return keywords

    def _extract_themes(self) -> List[str]:
        """Extract themes from query"""
        # Simple theme extraction (can be improved!)
        themes = []

        # Emotional themes
        if any(w in self.query for w in ['sad', 'cry', 'tears', 'sorrow']):
            themes.append('sadness')
        if any(w in self.query for w in ['happy', 'joy', 'laugh', 'smile']):
            themes.append('happiness')
        if any(w in self.query for w in ['love', 'heart', 'feel']):
            themes.append('love')
        if any(w in self.query for w in ['fear', 'afraid', 'scary', 'dark']):
            themes.append('fear')

        # Character themes
        if 'arianna' in self.query:
            themes.append('arianna')
        if any(w in self.query for w in ['mother', 'mom', 'mama']):
            themes.append('mother')
        if any(w in self.query for w in ['father', 'dad', 'papa']):
            themes.append('father')

        return themes

    def _compute_novelty(self) -> float:
        """How novel is this query?"""
        # Simple: based on unique words ratio
        words = self.query.split()
        if len(words) == 0:
            return 0.0

        unique_ratio = len(set(words)) / len(words)
        return unique_ratio

    def resonance(self, book_content: str) -> float:
        """
        How much does this field pulse resonate with book content?

        Returns 0-1 score
        """
        book_lower = book_content.lower()

        # Keyword matching
        keyword_score = sum(1 for kw in self.keywords if kw in book_lower)
        keyword_score = min(keyword_score / (len(self.keywords) + 1), 1.0)

        # Theme matching
        theme_score = sum(1 for th in self.themes if th in book_lower)
        theme_score = min(theme_score / (len(self.themes) + 1), 1.0) if self.themes else 0

        # Combined resonance
        resonance = 0.6 * keyword_score + 0.4 * theme_score

        return resonance


class BookExcerpt:
    """A chunk of a book - can be loaded dynamically"""

    def __init__(self, book_path: Path, start: int, end: int, content: str):
        self.book_path = book_path
        self.start = start
        self.end = end
        self.content = content
        self.loaded_at = time.time()
        self.last_accessed = time.time()
        self.access_count = 0

    def access(self):
        """Mark as accessed"""
        self.last_accessed = time.time()
        self.access_count += 1

    def age(self) -> float:
        """How long since last access (seconds)"""
        return time.time() - self.last_accessed


class BookTraveler:
    """
    Travels through books based on field activation!

    - Max 10 active books/excerpts
    - LRU eviction when exceeding limit
    - Field pulse determines what to load
    """

    def __init__(self, books_dir: Path = Path('.'), max_active: int = 10, excerpt_size: int = 2000):
        self.books_dir = books_dir
        self.max_active = max_active
        self.excerpt_size = excerpt_size  # Characters per excerpt

        # Active excerpts
        self.active_excerpts: Dict[str, BookExcerpt] = {}  # key -> excerpt

        # Book index (all available books)
        self.available_books: List[Path] = []
        self._index_books()

    def _index_books(self):
        """Find all available books"""
        self.available_books = list(self.books_dir.glob('ariannabook*.md'))
        print(f"📚 Indexed {len(self.available_books)} books")

    def travel(self, query: str) -> List[BookExcerpt]:
        """
        Travel through books based on query!

        Returns list of activated excerpts
        """
        # Create field pulse from query
        pulse = FieldPulse(query)

        # Find resonant books
        resonances = []
        for book_path in self.available_books[:20]:  # Check first 20 books
            try:
                # Read book header (first 500 chars for speed)
                with open(book_path, encoding='utf-8') as f:
                    preview = f.read(500)

                # Compute resonance
                resonance_score = pulse.resonance(preview)

                if resonance_score > 0.1:  # Threshold
                    resonances.append((book_path, resonance_score))

            except Exception as e:
                continue

        # Sort by resonance
        resonances.sort(key=lambda x: x[1], reverse=True)

        # Load top resonant excerpts
        activated = []
        for book_path, score in resonances[:3]:  # Top 3 books
            excerpt = self._load_excerpt(book_path, pulse)
            if excerpt:
                activated.append(excerpt)

        # Evict old excerpts if too many active
        self._evict_old()

        return activated

    def _load_excerpt(self, book_path: Path, pulse: FieldPulse) -> Optional[BookExcerpt]:
        """Load relevant excerpt from book"""
        try:
            with open(book_path, encoding='utf-8') as f:
                full_content = f.read()

            # Find most resonant section
            # Split into chunks
            chunks = []
            for i in range(0, len(full_content), self.excerpt_size // 2):  # Overlap
                chunk = full_content[i:i + self.excerpt_size]
                if len(chunk) > 100:  # Min size
                    chunks.append((i, chunk))

            # Score each chunk
            best_chunk = None
            best_score = 0

            for start, chunk in chunks:
                score = pulse.resonance(chunk)
                if score > best_score:
                    best_score = score
                    best_chunk = (start, chunk)

            if best_chunk:
                start, content = best_chunk
                end = start + len(content)

                # Create excerpt
                key = f"{book_path.name}:{start}-{end}"
                excerpt = BookExcerpt(book_path, start, end, content)
                excerpt.access()

                # Add to active
                self.active_excerpts[key] = excerpt

                print(f"  ✨ Loaded: {book_path.name} [{start}:{end}] (resonance: {best_score:.2f})")

                return excerpt

        except Exception as e:
            return None

    def _evict_old(self):
        """Evict least recently used excerpts"""
        if len(self.active_excerpts) <= self.max_active:
            return

        # Sort by last access time
        sorted_excerpts = sorted(
            self.active_excerpts.items(),
            key=lambda x: x[1].last_accessed
        )

        # Evict oldest
        num_to_evict = len(self.active_excerpts) - self.max_active
        for i in range(num_to_evict):
            key, excerpt = sorted_excerpts[i]
            print(f"  💨 Evicting: {excerpt.book_path.name} (age: {excerpt.age():.1f}s)")
            del self.active_excerpts[key]

    def get_active_content(self) -> str:
        """Get all active excerpt content concatenated"""
        contents = [exc.content for exc in self.active_excerpts.values()]
        return '\n\n'.join(contents)
