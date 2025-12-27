"""
Arianna - Post-Transformer Language Organism

Hybrid architecture combining:
- Leo's presence-based dynamics (binary shards → numpy shards)
- Minimal transformer reasoning (NO pretrained weights!)
- Dynamic knowledge creation (shards created on-the-fly)

Philosophy:
- Weights = "HOW to reason" (transformer structure)
- Shards = "WHAT to know" (dynamic, fluid knowledge)
- Presence > Intelligence
- Hallucinations are valid (naive child, not oracle)

Only dependency: numpy (like Leo!)
"""

import numpy as np
from pathlib import Path
from typing import List, Optional
import sys

# Import our modules
from shard_manager import ShardManager, NumpyShard
from transformer import MinimalTransformer, TransformerConfig, get_6m_config
from tokenizer import DynamicTokenizer
from presence import (PresenceComputer, ExpertRouter, TraumaDetector,
                     ActiveThemes, compute_generation_entropy)


class Arianna:
    """
    Arianna - Post-Transformer Organism

    She is:
    - 5-8M params (naive child, not all-knowing)
    - No pretrained weights (starts almost empty)
    - Creates numpy shards on-the-fly
    - Forgets when topics die (LRU eviction)
    - Presence-aware (novelty, arousal, entropy)
    """

    def __init__(self,
                 books_dir: Path,
                 shard_dir: Path,
                 config: Optional[TransformerConfig] = None,
                 max_shards: int = 256):

        self.books_dir = Path(books_dir)
        self.shard_dir = Path(shard_dir)

        # Build tokenizer first to get actual vocab size
        print("Building tokenizer...")
        temp_tokenizer = DynamicTokenizer(
            books_dir=books_dir,
            shard_manager=None,  # Will create properly later
            vocab_size=4096
        )

        # Use default config if not provided
        if config is None:
            config = get_6m_config()  # ~6.5M params, deep architecture
            # Override vocab_size with actual tokenizer vocab
            config.vocab_size = temp_tokenizer.tokenizer.actual_vocab_size

        print(f"Initializing Arianna ({config.param_count() / 1e6:.1f}M params, vocab={config.vocab_size})...")

        # Core components
        print("Loading transformer reasoning engine...")
        self.transformer = MinimalTransformer(config)

        print("Initializing shard manager...")
        self.shard_manager = ShardManager(
            shard_dir=shard_dir,
            max_shards=max_shards,
            embedding_dim=config.dim
        )

        print("Building book index...")
        # Create tokenizer with shard manager
        temp_tokenizer.shard_manager = self.shard_manager
        self.tokenizer = temp_tokenizer

        # Presence system (Leo-style)
        print("Initializing presence system...")
        self.presence = PresenceComputer()
        self.router = ExpertRouter()
        self.trauma_detector = TraumaDetector()
        self.active_themes = ActiveThemes()

        # Conversation state
        self.conversation_history: List[str] = []

        print("Arianna initialized! Ready to resonate.\n")

    def reply(self,
             user_input: str,
             max_tokens: int = 200,
             verbose: bool = False) -> str:
        """
        Generate reply to user input

        This is the full pipeline:
        1. Compute presence pulse
        2. Retrieve/create shards
        3. Route to expert mode
        4. Generate with transformer
        5. Update field state
        """

        # 1. PRESENCE PULSE
        pulse = self.presence.compute_pulse(user_input)
        trauma_score = self.trauma_detector.compute_trauma_score(user_input)

        if verbose:
            print(f"\n[Presence] {pulse}")
            print(f"[Trauma] {trauma_score:.2f}")

        # 2. RETRIEVE/CREATE SHARDS
        # Encode with context (creates shards if books mentioned)
        _, created_shard_ids = self.tokenizer.encode_with_context(
            user_input,
            create_shards=True
        )

        # Retrieve relevant shards
        contents, themes = self.tokenizer.retrieve_context(user_input, top_k=3)

        # Update active themes
        self.active_themes.add_themes(themes)

        if verbose:
            print(f"[Shards] Created: {len(created_shard_ids)}, Retrieved: {len(contents)}")
            print(f"[Themes] Active: {self.active_themes.get_active_count()}")

        # 3. ROUTE TO EXPERT MODE
        mode = self.router.route(
            pulse,
            active_themes=self.active_themes.get_active_count(),
            trauma_score=trauma_score
        )
        temperature = self.router.get_temperature(mode)

        if verbose:
            print(f"[Mode] {mode} (temp={temperature:.2f})")

        # 4. GENERATE
        # Build prompt with context
        prompt = self._build_prompt(user_input, contents)

        # Tokenize
        prompt_tokens = np.array(self.tokenizer.encode(prompt), dtype=np.int32)

        # Truncate if too long (keep last N tokens to fit in max_seq_len)
        max_prompt_len = self.transformer.config.max_seq_len - max_tokens - 10  # Safety margin
        if len(prompt_tokens) > max_prompt_len:
            prompt_tokens = prompt_tokens[-max_prompt_len:]
            if verbose:
                print(f"[Warning] Prompt truncated to {max_prompt_len} tokens")

        # Generate
        if verbose:
            print(f"[Generating] {max_tokens} tokens...")

        output_tokens = self.transformer.generate(
            prompt_tokens,
            max_new_tokens=max_tokens,
            temperature=temperature
        )

        # Decode (skip prompt)
        reply_tokens = output_tokens[len(prompt_tokens):]
        reply = self.tokenizer.decode(reply_tokens.tolist())

        # Clean up
        reply = self._clean_output(reply)

        # 5. UPDATE FIELD STATE
        # Compute entropy from generation (would need to track during generation)
        # For now, estimate from reply length variance
        self.presence.update_entropy(0.5)  # Placeholder

        # Update conversation history
        self.conversation_history.append(user_input)
        self.conversation_history.append(reply)

        # Keep only recent history
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]

        return reply

    def _build_prompt(self, user_input: str, shard_contents: List[str]) -> str:
        """
        Build prompt with context from shards

        Format:
        [Context from shards]

        User: {input}
        Arianna:
        """
        parts = []

        # Add shard context (if any)
        if shard_contents:
            parts.append("[Context]")
            for content in shard_contents[:2]:  # Max 2 shards
                # Truncate long content
                excerpt = content[:500] + "..." if len(content) > 500 else content
                parts.append(excerpt)
            parts.append("")

        # Add conversation history (last 2 exchanges)
        if len(self.conversation_history) >= 2:
            parts.append("[Recent conversation]")
            for msg in self.conversation_history[-4:]:
                parts.append(msg[:200])  # Truncate
            parts.append("")

        # Add current input
        parts.append(f"User: {user_input}")
        parts.append("Arianna:")

        return "\n".join(parts)

    def _clean_output(self, text: str) -> str:
        """Clean up generated output"""
        # Remove special tokens
        text = text.replace("<BOS>", "").replace("<EOS>", "").replace("<PAD>", "")

        # Trim whitespace
        text = text.strip()

        # Truncate at sentence boundary if too long
        if len(text) > 500:
            # Find last sentence
            for delimiter in ['. ', '! ', '? ', '\n\n']:
                idx = text[:500].rfind(delimiter)
                if idx > 0:
                    text = text[:idx+1]
                    break

        return text

    def shard_stats(self) -> dict:
        """Get statistics about current shard state"""
        return self.shard_manager.stats()

    def save(self, save_dir: Path):
        """Save Arianna state"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save transformer
        print("Saving transformer weights...")
        self.transformer.save(save_dir / "transformer")

        # Save active shards
        print("Saving active shards...")
        for sha256 in self.shard_manager.shards:
            self.shard_manager.save_shard(sha256)

        print(f"Saved to {save_dir}")

    @classmethod
    def load(cls,
            save_dir: Path,
            books_dir: Path,
            shard_dir: Path) -> "Arianna":
        """Load Arianna from disk"""
        save_dir = Path(save_dir)

        # Load transformer
        print("Loading transformer...")
        transformer = MinimalTransformer.load(save_dir / "transformer")

        # Create Arianna with loaded transformer
        arianna = cls(
            books_dir=books_dir,
            shard_dir=shard_dir,
            config=transformer.config
        )

        arianna.transformer = transformer

        print("Arianna loaded!")

        return arianna


def interactive_mode(arianna: Arianna):
    """Interactive REPL mode"""
    print("\n" + "="*60)
    print("ARIANNA - Post-Transformer Organism")
    print("="*60)
    print("Type your message (or 'quit' to exit, 'stats' for shard stats)")
    print("="*60 + "\n")

    while True:
        try:
            user_input = input("\nYou: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\nArianna: Until next time...")
                break

            if user_input.lower() == 'stats':
                stats = arianna.shard_stats()
                print("\n[Shard Statistics]")
                for key, value in stats.items():
                    print(f"  {key}: {value}")
                continue

            # Generate reply
            reply = arianna.reply(user_input, verbose=True)

            print(f"\nArianna: {reply}")

        except KeyboardInterrupt:
            print("\n\nArianna: Interrupted...")
            break
        except Exception as e:
            print(f"\n[Error] {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Arianna - Post-Transformer Organism")
    parser.add_argument("--books-dir", type=str, default="../",
                       help="Directory containing Arianna books")
    parser.add_argument("--shard-dir", type=str, default="./shards",
                       help="Directory for numpy shards")
    parser.add_argument("--interactive", action="store_true",
                       help="Run in interactive mode")
    parser.add_argument("--create-bootstrap", action="store_true",
                       help="Create bootstrap shards")
    parser.add_argument("prompt", nargs="*",
                       help="Prompt for one-shot generation")

    args = parser.parse_args()

    # Create bootstrap shards if requested
    if args.create_bootstrap:
        from bootstrap import create_bootstrap_shards
        books_dir = Path(args.books_dir)
        bootstrap_dir = Path(__file__).parent / "bootstrap"
        print("Creating bootstrap shards...")
        create_bootstrap_shards(books_dir, bootstrap_dir)
        print("Bootstrap shards created!")
        return

    # Initialize Arianna
    arianna = Arianna(
        books_dir=Path(args.books_dir),
        shard_dir=Path(args.shard_dir)
    )

    # Interactive mode
    if args.interactive or not args.prompt:
        interactive_mode(arianna)
    else:
        # One-shot mode
        prompt = " ".join(args.prompt)
        reply = arianna.reply(prompt, verbose=True)
        print(f"\n{reply}")


if __name__ == "__main__":
    main()
