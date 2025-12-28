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

        # Book index (hierarchical!)
        self.core_books: List[Path] = []  # CORE: always available
        self.story_books: List[Path] = []  # STORIES: LRU + field pulse

        # LRU cache for recently used books
        self.lru_cache: List[Path] = []  # Most recent first
        self.max_lru = 10

        self._index_books()

    def _index_books(self):
        """Find all available books (hierarchical!)"""
        personality_dir = self.books_dir / 'personality'

        # Index CORE books (concepts - always available!)
        core_dir = personality_dir / 'core'
        if core_dir.exists():
            self.core_books = list(core_dir.glob('ariannabook*.md'))

        # Index STORY books (episodic - LRU + field pulse)
        stories_dir = personality_dir / 'stories'
        if stories_dir.exists():
            self.story_books = list(stories_dir.glob('ariannabook*.md'))

        print(f"📚 Indexed {len(self.core_books)} CORE + {len(self.story_books)} STORIES")

    def travel(self, query: str) -> List[BookExcerpt]:
        """
        Travel through books based on query (HIERARCHICAL!)

        3 Levels:
        1. CORE: Always include 1-2 core concept excerpts (50 chars each)
        2. LRU: Check recently used books first (1 book if match)
        3. STORIES: Field pulse through all stories (1-2 books)

        Returns list of activated excerpts
        """
        pulse = FieldPulse(query)
        activated = []

        # LEVEL 1: CORE (always include!)
        if self.core_books:
            # Load 1-2 random core books (short excerpts!)
            import random
            core_sample = random.sample(self.core_books, min(2, len(self.core_books)))
            for book_path in core_sample:
                excerpt = self._load_excerpt(book_path, pulse, max_chars=50)
                if excerpt:
                    activated.append(excerpt)
                    print(f"  🧠 CORE: {book_path.name[:20]}... (always active)")

        # LEVEL 2: LRU CACHE (check recent books first!)
        lru_match = self._check_lru(pulse)
        if lru_match:
            excerpt = self._load_excerpt(lru_match, pulse, max_chars=100)
            if excerpt:
                activated.append(excerpt)
                print(f"  🔄 LRU: {lru_match.name[:20]}... (recent memory)")

        # LEVEL 3: FIELD PULSE (search all stories by resonance)
        resonances = []
        for book_path in self.story_books[:30]:  # Check first 30 stories
            try:
                with open(book_path, encoding='utf-8') as f:
                    preview = f.read(500)
                resonance_score = pulse.resonance(preview)
                if resonance_score > 0.1:
                    resonances.append((book_path, resonance_score))
            except:
                continue

        # Sort by resonance, take top 2
        resonances.sort(key=lambda x: x[1], reverse=True)
        for book_path, score in resonances[:2]:
            excerpt = self._load_excerpt(book_path, pulse, max_chars=100)
            if excerpt:
                activated.append(excerpt)
                self._update_lru(book_path)  # Add to LRU cache
                print(f"  ✨ Loaded: {book_path.name} [0:{len(excerpt.content)}] (resonance: {score:.2f})")

        return activated

    def _check_lru(self, pulse: FieldPulse) -> Optional[Path]:
        """Check LRU cache for matching books"""
        for book_path in self.lru_cache[:5]:  # Check top 5 recent books
            try:
                with open(book_path, encoding='utf-8') as f:
                    preview = f.read(500)
                resonance = pulse.resonance(preview)
                if resonance > 0.15:  # Higher threshold for LRU
                    return book_path
            except:
                continue
        return None

    def _update_lru(self, book_path: Path):
        """Update LRU cache with recently used book"""
        # Remove if already in cache
        if book_path in self.lru_cache:
            self.lru_cache.remove(book_path)
        # Add to front (most recent)
        self.lru_cache.insert(0, book_path)
        # Keep only max_lru entries
        self.lru_cache = self.lru_cache[:self.max_lru]

    def _load_excerpt(self, book_path: Path, pulse: FieldPulse, max_chars: int = 2000) -> Optional[BookExcerpt]:
        """Load relevant excerpt from book (up to max_chars)"""
        try:
            with open(book_path, encoding='utf-8') as f:
                full_content = f.read()

            # For short excerpts (core books), just take beginning
            if max_chars <= 100:
                content = full_content[:max_chars]
                excerpt = BookExcerpt(book_path, 0, len(content), content)
                excerpt.access()
                self.active_excerpts[f"{book_path.name}:0-{len(content)}"] = excerpt
                return excerpt

            # For longer excerpts, find best resonant section
            chunks = []
            for i in range(0, len(full_content), max_chars // 2):  # Overlap
                chunk = full_content[i:i + max_chars]
                if len(chunk) > 50:
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
                excerpt = BookExcerpt(book_path, start, end, content)
                excerpt.access()
                self.active_excerpts[f"{book_path.name}:{start}-{end}"] = excerpt
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
