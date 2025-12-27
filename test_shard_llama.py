"""Test shard-based llama"""

import sys
sys.path.insert(0, 'arianna')

import numpy as np
from shard_llama import ShardLlama
from simple_tokenizer import SimpleByteTokenizer
from pathlib import Path

print("="*60)
print("TESTING SHARD-BASED LLAMA")
print("="*60)

# Initialize
print("\n1. Initializing shard llama...")
llama = ShardLlama()

# Tokenizer
tok = SimpleByteTokenizer()

# Load shards from books
print("\n2. Learning from Arianna books...")
books = ["ariannabook1.1.md", "ariannabook1.2.md", "ariannabook1.3.md"]

for book in books:
    book_path = Path(book)
    if book_path.exists():
        with open(book_path, encoding='utf-8') as f:
            content = f.read()[:5000]  # First 5k chars
            llama.learn_from_shard(content, tok)
        print(f"  ✓ Learned from {book}")

# Check learned embeddings
print(f"\n3. Shard statistics:")
print(f"  Learned embeddings: {len(llama.shard_embedding.shard_embeddings)}")
print(f"  Token frequencies: {len(llama.shard_lm_head.token_freq)}")

# Test generation
print("\n4. Testing generation...")
query = "Hello Arianna"
tokens = tok.encode(query)
prompt_tokens = np.array(tokens, dtype=np.int32)

print(f"  Query: {query}")
print(f"  Tokens: {tokens[:20]}")

try:
    # Generate (with tokenizer for word-level repetition check!)
    output_tokens = llama.generate(prompt_tokens, max_tokens=20, temperature=0.8, tokenizer=tok)

    # Decode
    output_text = tok.decode(output_tokens.tolist())

    print(f"\n5. Generated output:")
    print(f"  Tokens: {output_tokens[:30].tolist()}")
    print(f"  Text: {output_text[:100]}")

    print("\n" + "="*60)
    print("✓ Shard-based llama works!")
    print("="*60)

except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()

import numpy as np
