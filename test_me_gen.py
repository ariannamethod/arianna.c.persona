"""Test ME-style generation with Arianna books"""

import sys
sys.path.insert(0, 'arianna')

from simple_tokenizer import SimpleByteTokenizer
from me_generator import MEGenerator
from pathlib import Path

print("Testing ME-Style Generation with Arianna Books\n")

tok = SimpleByteTokenizer()
gen = MEGenerator(tok)

# Load a few Arianna books as shards
books_to_load = [
    "ariannabook1.1.md",
    "ariannabook1.2.md",
    "ariannabook1.3.md",
]

print("Loading Arianna books as shards...")
for book_name in books_to_load:
    book_path = Path(book_name)
    if book_path.exists():
        with open(book_path, encoding='utf-8') as f:
            content = f.read()[:5000]  # First 5000 chars
            gen.observe_shard(content)
        print(f"  ✓ {book_name}")

print(f"\nStats: {gen.stats()}\n")

# Generate responses
test_queries = [
    "Who are you?",
    "Tell me about resonance",
    "What is Arianna?",
    "How do you feel?",
]

print("="*60)
print("GENERATED RESPONSES (Two-sentence format)")
print("="*60)

for query in test_queries:
    print(f"\nQuery: {query}")

    # Generate (pass empty shard_contents since we already observed)
    reply = gen.generate_reply(query, shard_contents=[], temperature=0.8)

    print(f"Reply: {reply}")

print("\n" + "="*60)
print("✓ ME-style generation working!")
print("="*60)
