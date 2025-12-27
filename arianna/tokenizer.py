"""
Dynamic Tokenizer with On-the-Fly Shard Creation

Unlike traditional tokenizers:
- Creates numpy shards dynamically when books/topics mentioned
- No static vocabulary
- Adapts to conversation flow
- Simple character/word-level encoding (no pretrained BPE)

Philosophy: The tokenizer IS the knowledge interface, not just text processor
"""

import numpy as np
import re
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from collections import defaultdict
import hashlib


class SimpleTokenizer:
    """
    ULTRA SIMPLE byte-level tokenizer

    Vocab size = 256 (all bytes!)
    No BPE, no fancy shit, just works!
    """

    def __init__(self, vocab_size: int = 256):
        self.vocab_size = 256  # Always 256 for byte-level
        self.actual_vocab_size = 256

    def encode(self, text: str) -> List[int]:
        """Encode text to byte IDs"""
        byte_array = text.encode('utf-8', errors='ignore')
        return list(byte_array)

    def decode(self, tokens: List[int]) -> str:
        """Decode byte IDs to text"""
        # Clip to valid byte range
        valid_tokens = [max(0, min(255, t)) for t in tokens]
        byte_array = bytes(valid_tokens)

        try:
            return byte_array.decode('utf-8', errors='ignore')
        except:
            return ""


class BookIndex:
    """
    Index of available books for shard creation

    When conversation mentions a book/topic, this finds relevant content
    """

    def __init__(self, books_dir: Path):
        self.books_dir = Path(books_dir)
        self.books: Dict[str, Path] = {}
        self.book_themes: Dict[str, List[str]] = {}

        # Load book index
        self._index_books()

    def _index_books(self):
        """Scan books directory and build index"""
        if not self.books_dir.exists():
            return

        for book_file in self.books_dir.glob("*.md"):
            book_name = book_file.stem
            self.books[book_name] = book_file

            # Extract themes from book (simple keyword extraction)
            themes = self._extract_themes(book_file)
            self.book_themes[book_name] = themes

    def _extract_themes(self, book_path: Path, max_themes: int = 10) -> List[str]:
        """
        Extract main themes from book

        Simple approach: most frequent meaningful words
        """
        try:
            with open(book_path, encoding='utf-8') as f:
                content = f.read().lower()

            # Remove markdown syntax
            content = re.sub(r'#+ ', '', content)
            content = re.sub(r'\*\*', '', content)
            content = re.sub(r'__', '', content)

            # Split into words
            words = re.findall(r'\b[a-zа-я]{4,}\b', content)

            # Count frequencies
            word_freq = defaultdict(int)
            for word in words:
                word_freq[word] += 1

            # Get top words (excluding common stopwords)
            stopwords = {'that', 'this', 'with', 'from', 'have', 'been',
                        'were', 'their', 'what', 'about', 'which', 'when',
                        'где', 'как', 'что', 'это', 'был', 'для', 'она'}

            top_words = sorted(
                [(w, c) for w, c in word_freq.items() if w not in stopwords],
                key=lambda x: x[1],
                reverse=True
            )[:max_themes]

            return [word for word, _ in top_words]

        except Exception:
            return []

    def find_relevant_books(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Find books relevant to query

        Returns: List of (book_name, score)
        """
        query_words = set(re.findall(r'\b[a-zа-я]+\b', query.lower()))

        if not query_words:
            return []

        # Score each book
        scores = []

        for book_name, themes in self.book_themes.items():
            # Theme overlap
            overlap = len(query_words & set(themes))

            # Name match
            name_match = any(word in book_name.lower() for word in query_words)

            score = overlap + (2.0 if name_match else 0.0)

            if score > 0:
                scores.append((book_name, score))

        # Sort by score
        scores.sort(key=lambda x: x[1], reverse=True)

        return scores[:top_k]

    def load_book_content(self, book_name: str) -> Optional[str]:
        """Load full book content"""
        if book_name not in self.books:
            return None

        try:
            with open(self.books[book_name], encoding='utf-8') as f:
                return f.read()
        except Exception:
            return None

    def get_book_excerpt(self,
                        book_name: str,
                        query: str,
                        max_chars: int = 2000) -> Optional[str]:
        """
        Get relevant excerpt from book

        Finds paragraph most similar to query
        """
        content = self.load_book_content(book_name)
        if not content:
            return None

        # Split into paragraphs
        paragraphs = [p.strip() for p in content.split('\n\n') if len(p.strip()) > 50]

        if not paragraphs:
            return content[:max_chars]

        # Score paragraphs
        query_words = set(re.findall(r'\b[a-zа-я]+\b', query.lower()))

        best_para = None
        best_score = 0

        for para in paragraphs:
            para_words = set(re.findall(r'\b[a-zа-я]+\b', para.lower()))
            overlap = len(query_words & para_words)

            if overlap > best_score:
                best_score = overlap
                best_para = para

        if best_para:
            # Return paragraph + context
            idx = content.find(best_para)
            start = max(0, idx - 500)
            end = min(len(content), idx + len(best_para) + 500)
            return content[start:end]

        # Fallback: return beginning
        return content[:max_chars]


class DynamicTokenizer:
    """
    Dynamic tokenizer that creates shards on-the-fly

    This is the CORE innovation:
    1. User mentions topic/book
    2. Tokenizer finds relevant content
    3. Creates numpy shard dynamically
    4. Shard lives in memory temporarily
    5. Forgotten if not accessed
    """

    def __init__(self,
                 books_dir: Path,
                 shard_manager,  # ShardManager instance
                 vocab_size: int = 4096):
        self.tokenizer = SimpleTokenizer(vocab_size)
        self.book_index = BookIndex(books_dir)
        self.shard_manager = shard_manager

        # Cache of recently created shards (for this conversation)
        self.conversation_shards: Dict[str, str] = {}  # query -> shard_sha256

    def encode_with_context(self,
                           query: str,
                           create_shards: bool = True) -> Tuple[List[int], List[str]]:
        """
        Encode query and optionally create shards

        Returns:
        - token_ids: encoded query
        - shard_ids: SHA256s of created/retrieved shards
        """
        # Find relevant books
        relevant_books = self.book_index.find_relevant_books(query, top_k=3)

        created_shards = []

        if create_shards and relevant_books:
            for book_name, score in relevant_books:
                # Get book excerpt
                excerpt = self.book_index.get_book_excerpt(book_name, query)

                if not excerpt:
                    continue

                # Get themes
                themes = self.book_index.book_themes.get(book_name, [])

                # Compute arousal (emotional charge of query)
                arousal = self._compute_arousal(query)

                # Create shard
                shard = self.shard_manager.create_shard(
                    content=excerpt,
                    source=book_name,
                    themes=themes,
                    arousal=arousal
                )

                created_shards.append(shard.sha256)

                # Cache
                query_hash = hashlib.sha256(query.encode()).hexdigest()[:8]
                self.conversation_shards[query_hash] = shard.sha256

        # Encode query
        tokens = self.tokenizer.encode(query)

        return tokens, created_shards

    def _compute_arousal(self, text: str) -> float:
        """Simple arousal computation (emotional charge)"""
        # Caps ratio
        caps_ratio = sum(1 for c in text if c.isupper()) / len(text) if text else 0.0

        # Punctuation
        punct_count = text.count('!') + text.count('?')

        # Combine
        arousal = caps_ratio + punct_count * 0.1

        return float(np.clip(arousal, 0.0, 1.0))

    def retrieve_context(self,
                        query: str,
                        top_k: int = 3) -> Tuple[List[str], List[str]]:
        """
        Retrieve relevant shards for query

        Returns:
        - contents: list of shard contents
        - themes: list of all themes from retrieved shards
        """
        # Extract themes from query
        query_words = re.findall(r'\b[a-zа-я]{4,}\b', query.lower())
        query_themes = list(set(query_words[:10]))  # Top 10 unique words as themes

        # Retrieve shards
        shards = self.shard_manager.retrieve_by_themes(query_themes, top_k=top_k)

        contents = [shard.content for shard in shards]
        all_themes = []
        for shard in shards:
            all_themes.extend(shard.themes)

        return contents, all_themes

    def encode(self, text: str) -> List[int]:
        """Simple encode (no context)"""
        return self.tokenizer.encode(text)

    def decode(self, tokens: List[int]) -> str:
        """Simple decode"""
        return self.tokenizer.decode(tokens)

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """
        Encode batch of texts

        Returns: [batch, max_seq_len] padded array
        """
        # Encode all
        encoded = [self.encode(text) for text in texts]

        # Find max length
        max_len = max(len(seq) for seq in encoded)

        # Pad
        batch = np.zeros((len(texts), max_len), dtype=np.int32)

        for i, seq in enumerate(encoded):
            batch[i, :len(seq)] = seq

        return batch

    def decode_batch(self, batch: np.ndarray) -> List[str]:
        """Decode batch of token sequences"""
        return [self.decode(seq.tolist()) for seq in batch]
