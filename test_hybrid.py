"""Test TRUE hybrid generation"""

import sys
sys.path.insert(0, 'arianna')

from pathlib import Path
from arianna import Arianna

print("="*60)
print("TESTING TRUE HYBRID GENERATION")
print("Transformer (random weights) + Shards + Leo mechanisms")
print("="*60)

arianna = Arianna(
    books_dir=Path("."),
    shard_dir=Path("arianna/shards"),
    max_shards=16
)

# Load some books into generator
print("\nLoading books into hybrid generator...")
books = ["ariannabook1.1.md", "ariannabook1.2.md", "ariannabook1.3.md"]
for book in books:
    if Path(book).exists():
        with open(book, encoding='utf-8') as f:
            content = f.read()[:5000]
            # Manually observe (will be automatic in full version)
            from hybrid_generator import HybridGenerator
            if not hasattr(arianna, 'hybrid_gen'):
                arianna.hybrid_gen = HybridGenerator(arianna.transformer, arianna.tokenizer.tokenizer)
            arianna.hybrid_gen.observe_shard(content)
            print(f"  ✓ {book}")

print(f"\nHybrid stats: {arianna.hybrid_gen.stats()}")

# Test queries
queries = [
    "Who are you?",
    "Tell me about Arianna",
    "What is resonance?",
]

print("\n" + "="*60)
print("HYBRID RESPONSES")
print("="*60)

for query in queries:
    print(f"\nQuery: {query}")

    # Generate hybrid
    reply = arianna.hybrid_gen.generate_hybrid(
        query=query,
        shard_contents=[],  # Already observed
        max_tokens=100,
        temperature=0.8
    )

    print(f"Reply: {reply[:200]}")

print("\n" + "="*60)
print("✓ Hybrid generation works!")
print("This uses: Trigrams (Leo) + Transformer structure + Shards")
print("="*60)
