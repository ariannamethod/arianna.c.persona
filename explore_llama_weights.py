"""Explore llama3.np pretrained weights structure"""

import numpy as np

# Load weights
weights = np.load('./llama3.np/stories15M.model.npz')

print("PRETRAINED WEIGHTS STRUCTURE:")
print("="*60)

print(f"\nTotal arrays: {len(weights.files)}")
print(f"\nArrays:")

total_params = 0
for key in sorted(weights.files):
    arr = weights[key]
    params = np.prod(arr.shape)
    total_params += params
    print(f"  {key:50s} {str(arr.shape):20s} {params:>12,} params")

print(f"\n{'='*60}")
print(f"TOTAL PARAMETERS: {total_params:,}")
print(f"Size on disk: 86 MB")
print(f"{'='*60}")

# Check specific arrays
print("\nEMBEDDINGS:")
if 'model.embed_tokens.weight' in weights.files:
    emb = weights['model.embed_tokens.weight']
    print(f"  Shape: {emb.shape}")
    print(f"  Vocab size: {emb.shape[0]}")
    print(f"  Embedding dim: {emb.shape[1]}")

print("\nLM HEAD:")
if 'lm_head.weight' in weights.files:
    lm = weights['lm_head.weight']
    print(f"  Shape: {lm.shape}")
    print(f"  Output vocab: {lm.shape[0]}")
    print(f"  Input dim: {lm.shape[1]}")
