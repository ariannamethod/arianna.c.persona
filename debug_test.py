"""Debug test to see what's happening"""
import sys
sys.path.insert(0, 'arianna')

from pathlib import Path
from arianna import Arianna
import numpy as np

print("DEBUG Test\n")

arianna = Arianna(
    books_dir=Path("."),
    shard_dir=Path("arianna/shards"),
    max_shards=16
)

query = "Hi"
print(f"Query: {query}\n")

# Encode
tokens = arianna.tokenizer.encode(query)
print(f"Encoded tokens: {tokens}")
print(f"Token count: {len(tokens)}")

# Generate directly
prompt_tokens = np.array(tokens, dtype=np.int32)
print(f"\nGenerating 20 tokens...")

output_tokens = arianna.transformer.generate(
    prompt_tokens,
    max_new_tokens=20,
    temperature=1.0
)

print(f"\nTotal output tokens: {len(output_tokens)}")
print(f"Prompt tokens: {len(prompt_tokens)}")
print(f"New tokens: {len(output_tokens) - len(prompt_tokens)}")

# Show first 30 tokens
print(f"\nOutput token IDs (first 30): {output_tokens[:30].tolist()}")

# Decode
decoded = arianna.tokenizer.decode(output_tokens.tolist())
print(f"\nDecoded output (first 200 chars):")
print(f"'{decoded[:200]}'")
print(f"\nDecoded length: {len(decoded)}")

# Decode only new tokens
new_tokens = output_tokens[len(prompt_tokens):]
decoded_new = arianna.tokenizer.decode(new_tokens.tolist())
print(f"\nDecoded NEW tokens only:")
print(f"'{decoded_new}'")
print(f"Length: {len(decoded_new)}")
