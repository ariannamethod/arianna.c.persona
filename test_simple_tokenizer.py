"""Test ultra simple tokenizer"""

import sys
sys.path.insert(0, 'arianna')

from simple_tokenizer import SimpleByteTokenizer

tok = SimpleByteTokenizer()

print(f"Vocab size: {tok.vocab_size}\n")

# Test encoding
test_texts = [
    "Hello world!",
    "Привет мир!",
    "БЛЯТЬ ОХУЕННО!",
    "123",
]

for text in test_texts:
    encoded = tok.encode(text)
    decoded = tok.decode(encoded)

    print(f"Original: '{text}'")
    print(f"Encoded: {encoded[:20]} {'...' if len(encoded) > 20 else ''}")
    print(f"Decoded: '{decoded}'")
    print(f"Match: {text == decoded}")
    print()
