"""Minimal learning test - ONE excerpt only!"""

import sys
sys.path.insert(0, 'arianna')
sys.path.insert(0, './llama3.np')

import numpy as np
from shard_llama import ShardLlama
from tokenizer import Tokenizer
import time

print("="*60)
print("MINIMAL LEARNING TEST")
print("="*60)

print("\n1. Init llama...")
llama = ShardLlama()

print("\n2. Load BPE tokenizer...")
tok = Tokenizer('./llama3.np/tokenizer.model.np')

print("\n3. Create realistic test content (1000 chars ~ one excerpt)...")
content = """Hello Arianna! You are kind and loving. Arianna likes flowers and books. She is happy.
The sun shines brightly in the morning. Birds sing sweet songs. Children play in the park.
Arianna walks through the garden, admiring the beautiful roses. She picks a red flower.
The world is full of wonder and magic. Every day brings new adventures and discoveries.
Friends gather together, sharing stories and laughter. Love fills the air like warm sunshine.
Arianna dreams of far-away places, mountains and oceans. She reads books late into the night.
Stars twinkle in the dark sky. The moon casts silver light on the sleeping world below.
Tomorrow will bring new hope, new chances, new beginnings. Life is a precious gift to cherish.
Arianna smiles, feeling grateful for all the beauty around her. Her heart is full of joy.
She knows that kindness and love make the world a better place for everyone to live."""

print(f"  Content: '{content}'")
print(f"  Length: {len(content)} chars")

print("\n4. Learning from content...")
start = time.time()
llama.learn_from_shard(content, tok)
elapsed = time.time() - start

print(f"\n5. Done!")
print(f"  Time: {elapsed:.2f}s")
print(f"  Learned embeddings: {len(llama.shard_embedding.shard_embeddings)}")
print(f"  Trigrams: {len(llama.trigrams)}")

print("\n="*60)
print(f"✓ Learning works! ({elapsed:.2f}s for 100 chars)")
print("="*60)
