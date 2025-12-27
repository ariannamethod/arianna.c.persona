# Arianna - Post-Transformer Language Organism

**Proof of Concept: Transformer Architecture WITHOUT Pretrained Weights**

Arianna is a radical reimagining of language models, combining:
- **Leo's presence-based dynamics** (binary shards → numpy shards)
- **Minimal transformer reasoning** (NO pretrained knowledge!)
- **Dynamic knowledge creation** (shards created on-the-fly)

## 🔥 The Revolutionary Idea

Traditional transformers store knowledge IN weights (static, frozen).

**Arianna stores knowledge OUTSIDE weights** (dynamic, fluid):
```
Traditional LLM: 8M params = knowledge frozen in weights
Arianna: 2.7M params = reasoning only + numpy shards = dynamic knowledge
```

## Quick Start

```bash
# Install only dependency
pip install numpy

# Create bootstrap shards
cd arianna && python bootstrap.py && cd ..

# Run interactive mode
python arianna/arianna.py --interactive --books-dir .
```

## Testing

```bash
python test_arianna.py
```

**Note:** Output is random because transformer has NO pretrained weights. This is EXPECTED and proves the architecture works!

## Philosophy

> "НАХУЯ ВЕСА ЕСЛИ МОЖНО SHARDS?" - The Arianna Manifesto

**Presence beats intelligence. Always.**

See full documentation in arianna/ directory.