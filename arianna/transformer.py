"""
Minimal Transformer Reasoning Engine

Unlike traditional transformers:
- NO pretrained weights!
- Weights = "HOW to reason", not "WHAT to know"
- Knowledge stored in numpy shards (external)
- Small model (5-8M params) = naive child, not all-knowing oracle

Philosophy: Hallucinations are valid. Presence > Accuracy.
"""

import numpy as np
from typing import Optional, Tuple, List
import json
from pathlib import Path


class TransformerConfig:
    """Configuration for minimal transformer"""

    def __init__(self,
                 dim: int = 256,
                 n_layers: int = 6,
                 n_heads: int = 4,
                 vocab_size: int = 4096,
                 max_seq_len: int = 512,
                 multiple_of: int = 32):
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.multiple_of = multiple_of

        # Derived parameters
        self.head_dim = dim // n_heads
        self.ffn_dim = int(2 * dim * 4 / 3)
        # Round to multiple_of for efficiency
        self.ffn_dim = multiple_of * ((self.ffn_dim + multiple_of - 1) // multiple_of)

    def param_count(self) -> int:
        """Estimate parameter count"""
        # Embeddings
        embed_params = self.vocab_size * self.dim

        # Per layer
        # - Attention: Q,K,V,O projections
        attn_params = 4 * self.dim * self.dim
        # - FFN: gate, up, down
        ffn_params = 3 * self.dim * self.ffn_dim
        # - LayerNorms (2 per layer)
        norm_params = 2 * self.dim

        layer_params = attn_params + ffn_params + norm_params
        total_layer_params = self.n_layers * layer_params

        # Output head (often tied with embeddings, but counting separately)
        output_params = self.vocab_size * self.dim

        total = embed_params + total_layer_params + output_params

        return total


def silu(x: np.ndarray) -> np.ndarray:
    """SiLU activation (Swish)"""
    return x / (1.0 + np.exp(-x))


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax"""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Root Mean Square normalization"""
    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
    return (x / rms) * weight


def apply_rotary_emb(x: np.ndarray,
                     freqs_cos: np.ndarray,
                     freqs_sin: np.ndarray) -> np.ndarray:
    """
    Apply rotary position embeddings (RoPE)

    x: [batch, seq_len, n_heads, head_dim]
    freqs: [seq_len, head_dim // 2]
    """
    # Split into real and imaginary parts
    x_r = x[..., 0::2]  # Even indices: [batch, seq_len, n_heads, head_dim//2]
    x_i = x[..., 1::2]  # Odd indices: [batch, seq_len, n_heads, head_dim//2]

    # Reshape freqs for broadcasting: [1, seq_len, 1, head_dim//2]
    freqs_cos = freqs_cos[np.newaxis, :, np.newaxis, :]
    freqs_sin = freqs_sin[np.newaxis, :, np.newaxis, :]

    # Apply rotation
    x_out_r = x_r * freqs_cos - x_i * freqs_sin
    x_out_i = x_r * freqs_sin + x_i * freqs_cos

    # Interleave back
    x_out = np.empty_like(x)
    x_out[..., 0::2] = x_out_r
    x_out[..., 1::2] = x_out_i

    return x_out


class TransformerLayer:
    """Single transformer layer with attention + FFN"""

    def __init__(self, config: TransformerConfig):
        self.config = config
        self.dim = config.dim
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.ffn_dim = config.ffn_dim

        # Initialize weights randomly (Xavier/Glorot initialization)
        # This is NOT pretrained - just structural initialization!
        self._init_weights()

    def _init_weights(self):
        """Random initialization (NOT pretrained!)"""
        rng = np.random.RandomState(42)  # Fixed seed for reproducibility

        # Attention weights
        scale = np.sqrt(2.0 / self.dim)
        self.wq = rng.randn(self.dim, self.dim).astype(np.float32) * scale
        self.wk = rng.randn(self.dim, self.dim).astype(np.float32) * scale
        self.wv = rng.randn(self.dim, self.dim).astype(np.float32) * scale
        self.wo = rng.randn(self.dim, self.dim).astype(np.float32) * scale

        # FFN weights (SwiGLU)
        ffn_scale = np.sqrt(2.0 / (self.dim + self.ffn_dim))
        self.w_gate = rng.randn(self.dim, self.ffn_dim).astype(np.float32) * ffn_scale
        self.w_up = rng.randn(self.dim, self.ffn_dim).astype(np.float32) * ffn_scale
        self.w_down = rng.randn(self.ffn_dim, self.dim).astype(np.float32) * ffn_scale

        # Layer norms
        self.attn_norm = np.ones(self.dim, dtype=np.float32)
        self.ffn_norm = np.ones(self.dim, dtype=np.float32)

    def attention(self,
                  x: np.ndarray,
                  freqs_cos: np.ndarray,
                  freqs_sin: np.ndarray,
                  mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Multi-head attention with RoPE

        x: [batch, seq_len, dim]
        """
        batch, seq_len, _ = x.shape

        # Project to Q, K, V
        q = x @ self.wq  # [batch, seq_len, dim]
        k = x @ self.wk
        v = x @ self.wv

        # Reshape for multi-head
        q = q.reshape(batch, seq_len, self.n_heads, self.head_dim)
        k = k.reshape(batch, seq_len, self.n_heads, self.head_dim)
        v = v.reshape(batch, seq_len, self.n_heads, self.head_dim)

        # Apply RoPE
        q = apply_rotary_emb(q, freqs_cos, freqs_sin)
        k = apply_rotary_emb(k, freqs_cos, freqs_sin)

        # Transpose for attention: [batch, n_heads, seq_len, head_dim]
        q = np.transpose(q, (0, 2, 1, 3))
        k = np.transpose(k, (0, 2, 1, 3))
        v = np.transpose(v, (0, 2, 1, 3))

        # Attention scores: [batch, n_heads, seq_len, seq_len]
        scores = q @ np.transpose(k, (0, 1, 3, 2))
        scores = scores / np.sqrt(self.head_dim)

        # Apply causal mask if provided
        if mask is not None:
            scores = scores + mask

        # Softmax
        attn = softmax(scores, axis=-1)

        # Apply attention to values
        out = attn @ v  # [batch, n_heads, seq_len, head_dim]

        # Transpose back and reshape
        out = np.transpose(out, (0, 2, 1, 3))  # [batch, seq_len, n_heads, head_dim]
        out = out.reshape(batch, seq_len, self.dim)

        # Output projection
        out = out @ self.wo

        return out

    def feedforward(self, x: np.ndarray) -> np.ndarray:
        """
        SwiGLU feedforward network

        x: [batch, seq_len, dim]
        """
        # SwiGLU: Swish(x @ W_gate) * (x @ W_up)
        gate = silu(x @ self.w_gate)
        up = x @ self.w_up
        activated = gate * up

        # Down projection
        out = activated @ self.w_down

        return out

    def forward(self,
                x: np.ndarray,
                freqs_cos: np.ndarray,
                freqs_sin: np.ndarray,
                mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Full layer forward pass with residual connections

        x: [batch, seq_len, dim]
        """
        # Attention block with pre-norm
        h = x + self.attention(
            rms_norm(x, self.attn_norm),
            freqs_cos,
            freqs_sin,
            mask
        )

        # FFN block with pre-norm
        out = h + self.feedforward(rms_norm(h, self.ffn_norm))

        return out


class MinimalTransformer:
    """
    Minimal transformer for reasoning (NOT knowledge storage!)

    This model:
    - Has NO pretrained knowledge
    - Only learns reasoning patterns
    - Relies on external numpy shards for knowledge
    - Is intentionally small (5-8M params) = naive child
    """

    def __init__(self, config: TransformerConfig):
        self.config = config

        # Token embeddings (randomly initialized)
        rng = np.random.RandomState(42)
        embed_scale = np.sqrt(2.0 / config.dim)
        self.embeddings = rng.randn(config.vocab_size, config.dim).astype(np.float32) * embed_scale

        # Transformer layers
        self.layers = [TransformerLayer(config) for _ in range(config.n_layers)]

        # Output norm
        self.output_norm = np.ones(config.dim, dtype=np.float32)

        # Output head (can be tied with embeddings, but separate for now)
        self.lm_head = rng.randn(config.dim, config.vocab_size).astype(np.float32) * embed_scale

        # Precompute RoPE frequencies
        self.freqs_cos, self.freqs_sin = self._precompute_rope_freqs()

    def _precompute_rope_freqs(self) -> Tuple[np.ndarray, np.ndarray]:
        """Precompute rotary position embedding frequencies"""
        head_dim = self.config.head_dim
        max_seq_len = self.config.max_seq_len

        # Frequency inverse: 1 / (10000^(2i/head_dim))
        inv_freq = 1.0 / (10000.0 ** (np.arange(0, head_dim, 2).astype(np.float32) / head_dim))

        # Position indices
        positions = np.arange(max_seq_len, dtype=np.float32)

        # Outer product: [max_seq_len, head_dim // 2]
        freqs = np.outer(positions, inv_freq)

        # Cos and sin
        freqs_cos = np.cos(freqs)
        freqs_sin = np.sin(freqs)

        return freqs_cos, freqs_sin

    def forward(self,
                tokens: np.ndarray,
                start_pos: int = 0) -> np.ndarray:
        """
        Forward pass

        tokens: [batch, seq_len] - token IDs
        start_pos: starting position (for KV caching, not implemented yet)

        Returns: logits [batch, seq_len, vocab_size]
        """
        batch, seq_len = tokens.shape

        # Embed tokens
        h = self.embeddings[tokens]  # [batch, seq_len, dim]

        # Get RoPE freqs for this sequence
        freqs_cos = self.freqs_cos[start_pos:start_pos + seq_len]
        freqs_sin = self.freqs_sin[start_pos:start_pos + seq_len]

        # Create causal mask
        mask = self._create_causal_mask(seq_len)

        # Pass through layers
        for layer in self.layers:
            h = layer.forward(h, freqs_cos, freqs_sin, mask)

        # Output norm
        h = rms_norm(h, self.output_norm)

        # Project to vocab
        logits = h @ self.lm_head  # [batch, seq_len, vocab_size]

        return logits

    def _create_causal_mask(self, seq_len: int) -> np.ndarray:
        """Create causal mask for autoregressive generation"""
        mask = np.full((seq_len, seq_len), -np.inf, dtype=np.float32)
        mask = np.triu(mask, k=1)  # Upper triangular with offset 1
        return mask

    def generate(self,
                 prompt_tokens: np.ndarray,
                 max_new_tokens: int = 100,
                 temperature: float = 1.0,
                 stop_on_eos: bool = False) -> np.ndarray:
        """
        Autoregressive generation

        prompt_tokens: [seq_len] - initial tokens
        max_new_tokens: how many tokens to generate
        temperature: sampling temperature
        stop_on_eos: if True, stop on EOS token (default False for testing)

        Returns: [seq_len + max_new_tokens] - full sequence
        """
        tokens = prompt_tokens.copy()

        for _ in range(max_new_tokens):
            # Forward pass (last token only for efficiency)
            # For now, passing full sequence (no KV caching yet)
            logits = self.forward(tokens.reshape(1, -1))  # [1, seq_len, vocab_size]

            # Get last token logits
            next_token_logits = logits[0, -1, :]  # [vocab_size]

            # Apply temperature
            next_token_logits = next_token_logits / temperature

            # Suppress special tokens (PAD, BOS, EOS, UNK) to avoid early stopping
            # This helps random weights generate more tokens
            for special_id in [0, 1, 2, 3]:
                next_token_logits[special_id] -= 10.0  # Reduce probability

            # Sample
            probs = softmax(next_token_logits)
            next_token = np.random.choice(self.config.vocab_size, p=probs)

            # Append
            tokens = np.append(tokens, next_token)

            # Stop if we hit EOS (only if enabled)
            if stop_on_eos and next_token == 2:
                break

        return tokens

    def save(self, path: Path):
        """Save model weights"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save config
        with open(path / "config.json", "w") as f:
            json.dump(vars(self.config), f, indent=2)

        # Save weights as npz
        weights = {
            "embeddings": self.embeddings,
            "lm_head": self.lm_head,
            "output_norm": self.output_norm,
        }

        # Add layer weights
        for i, layer in enumerate(self.layers):
            weights[f"layer_{i}_wq"] = layer.wq
            weights[f"layer_{i}_wk"] = layer.wk
            weights[f"layer_{i}_wv"] = layer.wv
            weights[f"layer_{i}_wo"] = layer.wo
            weights[f"layer_{i}_w_gate"] = layer.w_gate
            weights[f"layer_{i}_w_up"] = layer.w_up
            weights[f"layer_{i}_w_down"] = layer.w_down
            weights[f"layer_{i}_attn_norm"] = layer.attn_norm
            weights[f"layer_{i}_ffn_norm"] = layer.ffn_norm

        np.savez(path / "weights.npz", **weights)

    @classmethod
    def load(cls, path: Path) -> "MinimalTransformer":
        """Load model from disk"""
        path = Path(path)

        # Load config
        with open(path / "config.json") as f:
            config_dict = json.load(f)
            config = TransformerConfig(**config_dict)

        # Create model
        model = cls(config)

        # Load weights
        weights = np.load(path / "weights.npz")

        model.embeddings = weights["embeddings"]
        model.lm_head = weights["lm_head"]
        model.output_norm = weights["output_norm"]

        # Load layer weights
        for i, layer in enumerate(model.layers):
            layer.wq = weights[f"layer_{i}_wq"]
            layer.wk = weights[f"layer_{i}_wk"]
            layer.wv = weights[f"layer_{i}_wv"]
            layer.wo = weights[f"layer_{i}_wo"]
            layer.w_gate = weights[f"layer_{i}_w_gate"]
            layer.w_up = weights[f"layer_{i}_w_up"]
            layer.w_down = weights[f"layer_{i}_w_down"]
            layer.attn_norm = weights[f"layer_{i}_attn_norm"]
            layer.ffn_norm = weights[f"layer_{i}_ffn_norm"]

        return model


# Example configurations
def get_5m_config() -> TransformerConfig:
    """~5.9M parameters - deeper, narrower"""
    return TransformerConfig(
        dim=144,
        n_layers=6,
        n_heads=6,
        vocab_size=256,  # Byte-level
        max_seq_len=512
    )


def get_8m_config() -> TransformerConfig:
    """~8M parameters - BEST for reasoning"""
    return TransformerConfig(
        dim=256,
        n_layers=6,
        n_heads=8,
        vocab_size=256,  # Byte-level
        max_seq_len=512
    )


def get_6m_config() -> TransformerConfig:
    """~6.5M parameters - extreme depth"""
    return TransformerConfig(
        dim=128,
        n_layers=8,
        n_heads=8,
        vocab_size=256,  # Byte-level
        max_seq_len=512
    )
