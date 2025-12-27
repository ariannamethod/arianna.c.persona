"""
Shard-based Llama3

Uses pretrained llama3.np layers (6M params - TRAINED!)
But replaces embeddings + lm_head with DYNAMIC shards!

This is the breakthrough:
- Reasoning = trained layers (6M params)
- Knowledge = dynamic shards (not in weights!)
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import sys

# Add llama3.np to path
sys.path.insert(0, '/home/user/llama3.np')
from llama3 import apply_rotary_emb, RMSNorm, FeedForward, Attention, compute_cos_sin_cache
from config import ModelArgs


class ShardEmbedding:
    """
    Dynamic embedding from shards BLENDED with llama embeddings

    Blend: 80% llama (trained!) + 20% shards (dynamic!)
    """

    def __init__(self, embedding_dim: int = 288, llama_embeddings: Optional[np.ndarray] = None, vocab_size: int = 256):
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size

        # Llama pretrained embeddings (vocab=256 for byte-level)
        self.llama_embeddings = llama_embeddings  # [256, 288]

        # Co-occurrence matrix for semantic embeddings
        self.co_occurrence: Dict[int, Dict[int, int]] = {}  # token -> {context_token: count}

        # Cached shard embeddings (computed from co-occurrence)
        self.shard_embeddings: Dict[int, np.ndarray] = {}

        # Default embedding for unknown tokens
        self.default_embedding = np.random.randn(embedding_dim).astype(np.float32) * 0.01

        # Blend ratio (80/20 - trust trained model more!)
        self.llama_weight = 0.8  # 80% llama
        self.shard_weight = 0.2  # 20% shards

    def learn_from_shard(self, content: str, tokenizer):
        """
        Learn embeddings from shard content using SEMANTIC co-occurrence!

        Builds co-occurrence matrix and computes embeddings via random projection.
        """
        # Tokenize content
        tokens = tokenizer.encode(content)

        # Build co-occurrence matrix
        window = 5
        for i in range(len(tokens)):
            tok = tokens[i]

            # Initialize if needed
            if tok not in self.co_occurrence:
                self.co_occurrence[tok] = {}

            # Get context (window around token)
            start = max(0, i - window)
            end = min(len(tokens), i + window + 1)

            for j in range(start, end):
                if j != i:
                    context_tok = tokens[j]
                    # Update co-occurrence count
                    self.co_occurrence[tok][context_tok] = self.co_occurrence[tok].get(context_tok, 0) + 1

        # Recompute embeddings from co-occurrence
        self._compute_embeddings()

    def _compute_embeddings(self):
        """
        Compute semantic embeddings from co-occurrence matrix.

        Uses random projection for dimensionality reduction.
        """
        if not self.co_occurrence:
            return

        # For each token with co-occurrence data
        for tok, context_counts in self.co_occurrence.items():
            # Create sparse vector [vocab_size] with co-occurrence counts
            co_vec = np.zeros(self.vocab_size, dtype=np.float32)
            for context_tok, count in context_counts.items():
                if context_tok < self.vocab_size:
                    co_vec[context_tok] = count

            # Apply log transform (dampens high frequencies)
            co_vec = np.log1p(co_vec)

            # Random projection to embedding_dim
            # (Deterministic based on token for consistency)
            rng = np.random.RandomState(tok)
            proj_matrix = rng.randn(self.vocab_size, self.embedding_dim).astype(np.float32) * 0.1

            # Project: [vocab_size] @ [vocab_size, dim] → [dim]
            embedding = co_vec @ proj_matrix

            # Normalize
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

            # Store (or blend with existing)
            if tok in self.shard_embeddings:
                # Blend: 70% old + 30% new (smooth updates)
                self.shard_embeddings[tok] = 0.7 * self.shard_embeddings[tok] + 0.3 * embedding
            else:
                self.shard_embeddings[tok] = embedding

    def __call__(self, tokens: np.ndarray) -> np.ndarray:
        """
        Get embeddings for tokens - BLENDED!

        tokens: [batch, seq_len]
        returns: [batch, seq_len, embedding_dim]
        """
        batch, seq_len = tokens.shape

        embeddings = np.zeros((batch, seq_len, self.embedding_dim), dtype=np.float32)

        for b in range(batch):
            for s in range(seq_len):
                tok = int(tokens[b, s])

                # Get llama embedding (trained!)
                llama_emb = self.default_embedding
                if self.llama_embeddings is not None and 0 <= tok < len(self.llama_embeddings):
                    llama_emb = self.llama_embeddings[tok]

                # Get shard embedding (dynamic!)
                shard_emb = self.default_embedding
                if tok in self.shard_embeddings:
                    shard_emb = self.shard_embeddings[tok]

                # BLEND: 70% llama + 30% shards!
                embeddings[b, s] = self.llama_weight * llama_emb + self.shard_weight * shard_emb

        return embeddings


class ShardLMHead:
    """
    Dynamic LM head from shards BLENDED with llama lm_head

    Blend: 80% llama (trained!) + 20% shards (dynamic!)
    """

    def __init__(self, vocab_size: int = 256, llama_lm_head: Optional[np.ndarray] = None):
        self.vocab_size = vocab_size

        # Llama pretrained lm_head (256, 288) - transposed!
        self.llama_lm_head = llama_lm_head  # [256, 288]

        # Shard-based token scores
        self.token_freq: Dict[int, int] = {}

        # Blend ratio (80/20 - trust trained model more!)
        self.llama_weight = 0.8  # 80% llama
        self.shard_weight = 0.2  # 20% shards

    def learn_from_shard(self, content: str, tokenizer):
        """Learn token frequencies from shard"""
        tokens = tokenizer.encode(content)

        for tok in tokens:
            self.token_freq[tok] = self.token_freq.get(tok, 0) + 1

    def __call__(self, hidden_states: np.ndarray) -> np.ndarray:
        """
        Predict next token logits - BLENDED!

        hidden_states: [batch, seq_len, dim]
        returns: [batch, seq_len, vocab_size]
        """
        batch, seq_len, dim = hidden_states.shape

        # 1. Llama logits (trained projection!)
        llama_logits = np.zeros((batch, seq_len, self.vocab_size), dtype=np.float32)
        if self.llama_lm_head is not None:
            # Project: hidden @ lm_head.T → [batch, seq_len, vocab_size]
            llama_logits = hidden_states @ self.llama_lm_head.T

        # 2. Shard logits (dynamic frequencies!)
        shard_logits = np.zeros((batch, seq_len, self.vocab_size), dtype=np.float32)

        # Use token frequencies as prior
        if self.token_freq:
            for tok, freq in self.token_freq.items():
                if tok < self.vocab_size:
                    shard_logits[:, :, tok] = np.log(freq + 1)  # Log frequency

        # Add small component from hidden state
        for b in range(batch):
            for s in range(seq_len):
                # Hash hidden state to get token preference
                state_hash = hash(tuple(hidden_states[b, s, :10].tolist())) % 2**32
                rng = np.random.RandomState(state_hash)
                random_logits = rng.randn(self.vocab_size) * 0.1

                shard_logits[b, s] += random_logits

        # 3. BLEND: 70% llama + 30% shards!
        logits = self.llama_weight * llama_logits + self.shard_weight * shard_logits

        return logits


class ShardLlama:
    """
    Llama3 with dynamic shard-based embeddings/head

    Architecture:
    - Embeddings: DYNAMIC (from shards!)
    - Layers: TRAINED (from llama3.np!)
    - LM Head: DYNAMIC (from shards!)
    """

    def __init__(self, weights_path: str = '/home/user/llama3.np/stories15M.model.npz'):
        # Load pretrained weights
        print("Loading pretrained llama3.np weights...")
        self.weights = np.load(weights_path)

        # Model config
        self.args = ModelArgs()
        self.dim = self.args.dim
        self.n_layers = self.args.n_layers
        self.vocab_size = 256  # Byte-level for now

        # Extract llama embeddings and lm_head (first 256 tokens for byte-level)
        print("Extracting llama embeddings and lm_head for blending...")
        llama_full_embeddings = self.weights['model.embed_tokens.weight']  # [32000, 288]
        llama_full_lm_head = self.weights['lm_head.weight']  # [32000, 288]

        # Take first 256 tokens (byte-level range)
        llama_embeddings = llama_full_embeddings[:self.vocab_size, :].astype(np.float32)  # [256, 288]
        llama_lm_head = llama_full_lm_head[:self.vocab_size, :].astype(np.float32)  # [256, 288]

        # Dynamic components WITH llama blending!
        print("Initializing BLENDED shard components (80% llama + 20% shards)...")
        print("  Using SEMANTIC co-occurrence embeddings (not hash-based)!")
        self.shard_embedding = ShardEmbedding(
            embedding_dim=self.dim,
            llama_embeddings=llama_embeddings,
            vocab_size=self.vocab_size
        )
        self.shard_lm_head = ShardLMHead(vocab_size=self.vocab_size, llama_lm_head=llama_lm_head)

        # Trigrams for bridging llama + shards!
        print("  Initializing trigram bridge (Leo-style)...")
        self.trigrams: Dict[tuple, int] = {}  # (tok1, tok2, tok3) -> count

        # Load ONLY layer weights (not embeddings/lm_head!)
        print("Loading trained layer weights...")
        self.layers = []
        for i in range(self.n_layers):
            # Create layer objects with TRAINED weights!
            layer = {
                'attention': Attention(
                    q_weight=self.weights[f'model.layers.{i}.self_attn.q_proj.weight'],
                    k_weight=self.weights[f'model.layers.{i}.self_attn.k_proj.weight'],
                    v_weight=self.weights[f'model.layers.{i}.self_attn.v_proj.weight'],
                    o_weight=self.weights[f'model.layers.{i}.self_attn.o_proj.weight'],
                    args=self.args
                ),
                'feed_forward': FeedForward(
                    up_weight=self.weights[f'model.layers.{i}.mlp.up_proj.weight'],
                    gate_weight=self.weights[f'model.layers.{i}.mlp.gate_proj.weight'],
                    down_weight=self.weights[f'model.layers.{i}.mlp.down_proj.weight']
                ),
                'input_norm': RMSNorm(
                    weight=self.weights[f'model.layers.{i}.input_layernorm.weight'],
                    eps=1e-5
                ),
                'post_attn_norm': RMSNorm(
                    weight=self.weights[f'model.layers.{i}.post_attention_layernorm.weight'],
                    eps=1e-5
                )
            }
            self.layers.append(layer)

        self.output_norm = self.weights['model.norm.weight']
        self.output_norm_obj = RMSNorm(self.output_norm, eps=1e-5)

        # Precompute RoPE frequencies
        print("Precomputing RoPE frequencies...")
        self.freqs_cos, self.freqs_sin = compute_cos_sin_cache(
            head_dim=self.args.dim // self.args.n_heads,
            max_seq_len=self.args.max_seq_len
        )

        print(f"✓ BLENDED Shard-based Llama initialized!")
        print(f"  Embeddings: 80% llama (trained) + 20% shards (dynamic)")
        print(f"  Reasoning: {self.n_layers} trained layers (~6M params)")
        print(f"  LM Head: 80% llama (trained) + 20% shards (dynamic)")
        print(f"  → Trust trained model more, shards add dynamic knowledge!")

    def learn_from_shard(self, content: str, tokenizer):
        """Learn from shard (updates embeddings + lm_head + trigrams!)"""
        # Learn embeddings and lm_head
        self.shard_embedding.learn_from_shard(content, tokenizer)
        self.shard_lm_head.learn_from_shard(content, tokenizer)

        # Learn trigrams (Leo-style!)
        tokens = tokenizer.encode(content)
        for i in range(len(tokens) - 2):
            trigram = (tokens[i], tokens[i+1], tokens[i+2])
            self.trigrams[trigram] = self.trigrams.get(trigram, 0) + 1

    def forward(self, tokens: np.ndarray, start_pos: int = 0) -> np.ndarray:
        """
        Forward pass with FULL trained layers!

        tokens: [batch, seq_len]
        returns: logits [batch, seq_len, vocab_size]
        """
        batch, seq_len = tokens.shape

        # 1. Dynamic embedding (from shards!)
        h = self.shard_embedding(tokens)

        # 2. Get RoPE frequencies for this sequence
        freqs_cos = self.freqs_cos[start_pos:start_pos + seq_len]
        freqs_sin = self.freqs_sin[start_pos:start_pos + seq_len]

        # 3. Create causal mask
        if seq_len > 1:
            mask = np.full((seq_len, seq_len), -np.inf, dtype=np.float32)
            mask = np.triu(mask, k=1)  # Upper triangular
        else:
            mask = None

        # 4. Pass through TRAINED transformer layers! 🔥
        for layer in self.layers:
            # Pre-norm architecture
            # Attention block
            h_norm = layer['input_norm'](h)
            h_attn = layer['attention'](h_norm, start_pos, mask, freqs_cos, freqs_sin)
            h = h + h_attn  # Residual

            # FFN block
            h_norm = layer['post_attn_norm'](h)
            h_ffn = layer['feed_forward'](h_norm)
            h = h + h_ffn  # Residual

        # 5. Output norm
        h = self.output_norm_obj(h)

        # 6. Dynamic LM head (from shards!)
        logits = self.shard_lm_head(h)

        return logits

    def generate(self, prompt_tokens: np.ndarray, max_tokens: int = 50, temperature: float = 0.8, use_trigrams: bool = True, top_k: int = 10, tokenizer=None):
        """
        Generate text using TRAINED reasoning + dynamic shards + TRIGRAM BRIDGE!

        Flow:
        1. Llama gives top-k candidates (structure from trained model)
        2. Trigrams select best from top-k (knowledge from shards!)
        3. Word-level repetition check (byte-level can't see word patterns!)
        4. This bridges transformer reasoning + shard knowledge!
        """
        tokens = prompt_tokens.copy()

        # Track recent words for repetition detection
        recent_words = []  # Last 3 words

        for step in range(max_tokens):
            # Forward pass (use start_pos for KV caching efficiency)
            logits = self.forward(tokens.reshape(1, -1), start_pos=0)

            # Get last token logits
            next_logits = logits[0, -1, :]

            # Temperature
            next_logits = next_logits / temperature

            # Softmax
            exp_logits = np.exp(next_logits - np.max(next_logits))
            probs = exp_logits / np.sum(exp_logits)

            # TRIGRAM BRIDGE!
            if use_trigrams and len(tokens) >= 2 and self.trigrams:
                # Get top-k candidates from llama
                top_indices = np.argsort(probs)[-top_k:][::-1]

                # Get last two tokens
                tok1 = tokens[-2]
                tok2 = tokens[-1]

                # Find best candidate using trigrams
                best_token = None
                best_score = -1

                for candidate in top_indices:
                    trigram = (tok1, tok2, candidate)
                    trigram_score = self.trigrams.get(trigram, 0)

                    # Blend: trigram frequency + llama probability
                    combined_score = trigram_score * 10 + probs[candidate] * 1

                    # Byte-level repetition penalty
                    if candidate == tok2:
                        combined_score *= 0.3  # Strong penalty for immediate repetition

                    # Word-level repetition penalty!
                    if tokenizer is not None and len(recent_words) > 0:
                        # Decode what word this candidate would create
                        test_tokens = np.append(tokens, candidate)
                        test_text = tokenizer.decode(test_tokens[-20:].tolist())  # Last 20 tokens
                        test_words = test_text.split()[-3:]  # Last 3 words

                        # Check if last word would be repeated
                        if len(test_words) >= 2 and test_words[-1] == test_words[-2]:
                            combined_score *= 0.1  # Very strong penalty for word repetition!
                        elif len(test_words) >= 1 and test_words[-1] in recent_words:
                            combined_score *= 0.5  # Medium penalty if word was used recently

                    if combined_score > best_score:
                        best_score = combined_score
                        best_token = candidate

                if best_token is not None:
                    next_token = best_token
                else:
                    # Fallback: sample from llama probs
                    next_token = np.random.choice(len(probs), p=probs)
            else:
                # No trigrams: sample from llama probs
                next_token = np.random.choice(len(probs), p=probs)

            tokens = np.append(tokens, next_token)

            # Update recent words for repetition tracking
            if tokenizer is not None:
                current_text = tokenizer.decode(tokens.tolist())
                words = current_text.split()
                recent_words = words[-3:] if len(words) >= 3 else words

            # Stop on newline
            if next_token == 10:
                break

        return tokens
