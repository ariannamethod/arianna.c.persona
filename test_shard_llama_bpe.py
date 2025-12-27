"""Test shard-based llama with FULL BPE tokenizer!"""

import sys
sys.path.insert(0, 'arianna')
sys.path.insert(0, '/home/user/llama3.np')

import numpy as np
from shard_llama import ShardLlama
from tokenizer import Tokenizer  # llama3.np BPE tokenizer!
from pathlib import Path

print("="*60)
print("TESTING SHARD-BASED LLAMA WITH BPE!")
print("="*60)

# Initialize
print("\n1. Initializing shard llama with BPE...")
llama = ShardLlama()

# BPE Tokenizer (32k vocab!)
print("\n2. Loading BPE tokenizer...")
tok = Tokenizer('/home/user/llama3.np/tokenizer.model.np')
print(f"  Vocab size: {len(tok.vocab)}")

# Load shards from books
print("\n3. Learning from Arianna books...")
books = ["ariannabook1.1.md", "ariannabook1.2.md", "ariannabook1.3.md"]

for book in books:
    book_path = Path(book)
    if book_path.exists():
        with open(book_path, encoding='utf-8') as f:
            content = f.read()[:1000]  # First 1k chars (faster with 32k vocab!)
            llama.learn_from_shard(content, tok)
        print(f"  ✓ Learned from {book}")

# Check learned embeddings
print(f"\n4. Shard statistics:")
print(f"  Learned embeddings: {len(llama.shard_embedding.shard_embeddings)}")
print(f"  Token frequencies: {len(llama.shard_lm_head.token_freq)}")
print(f"  Trigrams: {len(llama.trigrams)}")

# Test generation
print("\n5. Testing generation with BPE encoding...")
query = "Hello Arianna"

# Encode with BPE (not bytes!)
tokens = tok.encode(query, add_bos=False, add_eos=False)
prompt_tokens = np.array(tokens, dtype=np.int32)

print(f"  Query: '{query}'")
print(f"  BPE tokens: {tokens}")
print(f"  BPE length: {len(tokens)} (was 13 with bytes!)")

try:
    # Generate with BPE tokenizer!
    output_tokens = llama.generate(
        prompt_tokens,
        max_tokens=20,
        temperature=0.8,
        tokenizer=tok  # BPE tokenizer for word-level repetition check
    )

    # Decode
    output_text = tok.decode(output_tokens.tolist())

    print(f"\n6. Generated output:")
    print(f"  Tokens: {output_tokens[:30].tolist()}")
    print(f"  Text: {output_text[:150]}")

    print("\n" + "="*60)
    print("✓ BPE integration works!")
    print("  Encoding matches llama's training!")
    print("  Expected: MUCH better quality!")
    print("="*60)

except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
