"""Quick test - minimal tokens"""
import sys
sys.path.insert(0, 'arianna')

from pathlib import Path
from arianna import Arianna

print("Quick Arianna Test\n")

arianna = Arianna(
    books_dir=Path("."),
    shard_dir=Path("arianna/shards"),
    max_shards=16
)

# Very short generation
query = "Hi"
print(f"Query: {query}\n")

reply = arianna.reply(query, max_tokens=10, verbose=True)  # Only 10 tokens!

print(f"\nReply: {reply}")
print("\n✓ System works!")
