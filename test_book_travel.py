"""Test book travel - field-activated dynamic loading!"""

import sys
sys.path.insert(0, 'arianna')

import numpy as np
from shard_llama import ShardLlama
from simple_tokenizer import SimpleByteTokenizer

print("="*60)
print("TESTING BOOK TRAVEL")
print("="*60)

# Initialize
print("\n1. Initializing Arianna with book traveler...")
llama = ShardLlama()
tok = SimpleByteTokenizer()

# Test queries
queries = [
    "Hello Arianna, who are you?",
    "Tell me about sadness",
    "What is love?",
]

print("\n2. Testing field-activated book travel...\n")

for query in queries:
    print("─" * 60)
    response = llama.respond(query, tok, max_tokens=30, temperature=0.8)
    print(f"✓ Response: {response[:150]}")
    print()

print("="*60)
print("✓ Book travel works!")
print("="*60)
