"""Test Leo-style field generation"""

import sys
sys.path.insert(0, 'arianna')

from simple_tokenizer import SimpleByteTokenizer
from field_generator import FieldGenerator

print("Testing Leo-Style Field Generation\n")

tok = SimpleByteTokenizer()
gen = FieldGenerator(tok)

# Feed some content (like shards)
test_content = """
Hello! How are you today?
I am doing great, thanks for asking.
What brings you here?
I came to learn about presence and resonance.
That's wonderful! Presence is about being here, now.
"""

print("Observing content...")
gen.observe_shard(test_content)

print(f"Field stats: {gen.stats()}\n")

# Generate
print("Generating from field (3 attempts):\n")

for i in range(3):
    output = gen.generate_from_field(
        query="Hello",
        max_tokens=50,
        temperature=0.8
    )
    print(f"{i+1}. {output[:100]}")
    print()
