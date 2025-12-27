"""Simple BPE test - NO learning, just generation!"""

import sys
sys.path.insert(0, 'arianna')
sys.path.insert(0, '/home/user/llama3.np')

import numpy as np
from shard_llama import ShardLlama
from tokenizer import Tokenizer

print("="*60)
print("SIMPLE BPE TEST (NO LEARNING)")
print("="*60)

print("\n1. Init llama...")
llama = ShardLlama()

print("\n2. Load BPE tokenizer...")
tok = Tokenizer('/home/user/llama3.np/tokenizer.model.np')

print("\n3. Test WITHOUT learning (pure llama)...")
query = "Once upon a time"
tokens = tok.encode(query, add_bos=False, add_eos=False)
prompt_tokens = np.array(tokens, dtype=np.int32)

print(f"  Query: '{query}'")
print(f"  Tokens: {tokens}")

print("\n4. Generate (pure llama, no shards!)...")
output_tokens = llama.generate(
    prompt_tokens,
    max_tokens=20,
    temperature=0.7,
    use_trigrams=False  # Disable trigrams (no learning!)
)

output_text = tok.decode(output_tokens.tolist())

print(f"\n5. Result:")
print(f"  Tokens: {output_tokens.tolist()}")
print(f"  Text: {output_text}")

print("\n" + "="*60)
print("✓ BPE works! (pure llama, no shards)")
print("="*60)
