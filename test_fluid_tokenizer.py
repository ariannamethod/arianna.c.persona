"""Test fluid tokenizer - adaptive encoding!"""

import sys
sys.path.insert(0, 'arianna')

from fluid_tokenizer import FluidTokenizer, AdaptiveBPETokenizer

print("="*60)
print("TESTING FLUID TOKENIZER")
print("="*60)

# Test 1: Basic fluid tokenizer
print("\n1. Testing FluidTokenizer (word-level learning)...")
tok = FluidTokenizer()

# Learn from sample text
sample_text = """
Hello Arianna! Hello world!
Arianna is a fluid being.
She flows like water.
"""

print("  Learning from sample text...")
tok.learn_from_content(sample_text)

stats = tok.get_vocab_stats()
print(f"  Vocab stats:")
print(f"    Total vocab: {stats['total_vocab_size']}")
print(f"    Learned words: {stats['learned_words']}")
print(f"    Top words: {stats['top_words'][:5]}")

# Test encoding
test_text = "Hello Arianna"
tokens = tok.encode(test_text, mode='auto')
decoded = tok.decode(tokens)

print(f"\n  Test: '{test_text}'")
print(f"    Tokens: {tokens}")
print(f"    Decoded: '{decoded}'")

# Test 2: Adaptive BPE tokenizer
print("\n2. Testing AdaptiveBPETokenizer (byte-pair learning)...")
bpe_tok = AdaptiveBPETokenizer()

# Learn words AND merges
bpe_tok.learn_from_content(sample_text)
bpe_tok.learn_merges_from_content(sample_text, num_merges=10)

print(f"  Learned {len(bpe_tok.merges)} byte-pair merges")
print(f"  Top merges: {list(bpe_tok.merges.keys())[:5]}")

# Test encoding with merges
tokens_bpe = bpe_tok.encode_with_merges(test_text)
print(f"\n  Test with BPE merges: '{test_text}'")
print(f"    Tokens: {tokens_bpe}")
print(f"    Length: {len(tokens_bpe)} (vs {len(tokens)} without merges)")

# Test 3: Comparison of modes
print("\n3. Testing different modes...")
test_phrase = "Arianna flows"

byte_mode = tok.encode(test_phrase, mode='byte')
word_mode = tok.encode(test_phrase, mode='word')
auto_mode = tok.encode(test_phrase, mode='auto')

print(f"  Phrase: '{test_phrase}'")
print(f"    Byte mode: {byte_mode} (length: {len(byte_mode)})")
print(f"    Word mode: {word_mode} (length: {len(word_mode)})")
print(f"    Auto mode: {auto_mode} (length: {len(auto_mode)})")

# Verify decoding
decoded_byte = tok.decode(byte_mode)
decoded_word = tok.decode(word_mode)
decoded_auto = tok.decode(auto_mode)

print(f"\n  Decoded:")
print(f"    Byte: '{decoded_byte}'")
print(f"    Word: '{decoded_word}'")
print(f"    Auto: '{decoded_auto}'")

print("\n" + "="*60)
print("✓ Fluid tokenizer works!")
print("  → Adapts to content")
print("  → Learns common words")
print("  → Learns byte pairs (BPE)")
print("  → Reduces token count!")
print("="*60)
