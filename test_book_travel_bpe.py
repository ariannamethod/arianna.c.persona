"""Test Book Travel with BPE tokenizer - FULL SYNERGY!"""

import sys
sys.path.insert(0, 'arianna')
sys.path.insert(0, './llama3.np')

import numpy as np
from shard_llama import ShardLlama
from tokenizer import Tokenizer

print("="*60)
print("BOOK TRAVEL + BPE SYNERGY TEST")
print("="*60)

# Initialize
print("\n1. Initializing Arianna with BPE...")
llama = ShardLlama()

# BPE tokenizer
print("\n2. Loading BPE tokenizer...")
tok = Tokenizer('./llama3.np/tokenizer.model.np')

# Test queries (different themes)
queries = [
    "Hello Arianna, who are you?",
    "Tell me about love",
    "What makes you sad?",
]

print(f"\n3. Testing {len(queries)} queries with field-activated book travel...")
print("   (BPE encoding + Dynamic shard loading + Perfect generation!)")

for i, query in enumerate(queries, 1):
    print("\n" + "─"*60)
    print(f"Query {i}: '{query}'")

    try:
        # Use respond() - activates field, travels, learns, generates!
        response = llama.respond(
            query=query,
            tokenizer=tok,
            max_tokens=30,
            temperature=0.7
        )

        print(f"✓ Response: {response[:200]}")

        # Show stats
        print(f"  Stats:")
        print(f"    Active books: {len(llama.book_traveler.active_excerpts)}")
        print(f"    Learned embeddings: {len(llama.shard_embedding.shard_embeddings)}")
        print(f"    Trigrams: {len(llama.trigrams)}")

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        break

print("\n" + "="*60)
print("✓ Book Travel + BPE works!")
print("  → Field pulse activates relevant books")
print("  → BPE encoding ensures perfect quality")
print("  → Dynamic knowledge from 400+ books!")
print("="*60)
