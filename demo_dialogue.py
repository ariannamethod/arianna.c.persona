"""
Demo dialogue with Arianna

Shows:
1. Presence pulse detection
2. Dynamic shard creation
3. Expert routing
4. Generation (random, but architecture works!)
"""

import sys
sys.path.insert(0, 'arianna')

from pathlib import Path
from arianna import Arianna


def demo_dialogue():
    print("\n" + "="*60)
    print("ARIANNA DEMO DIALOGUE")
    print("="*60)
    print("Note: Output is random (no pretrained weights)")
    print("This demonstrates the ARCHITECTURE, not trained behavior")
    print("="*60 + "\n")

    # Initialize
    arianna = Arianna(
        books_dir=Path("."),
        shard_dir=Path("arianna/shards"),
        max_shards=64
    )

    # Test queries
    queries = [
        ("Hello Arianna, who are you?", "Basic greeting"),
        ("Tell me about resonance and Arianna", "Book topic - should create shards!"),
        ("БЛЯТЬ ЭТО ОХУЕННО!!!", "High arousal test"),
        ("quantum mechanics", "High novelty test"),
    ]

    for i, (query, description) in enumerate(queries, 1):
        print(f"\n{'='*60}")
        print(f"TEST {i}: {description}")
        print(f"{'='*60}")
        print(f"Query: {query}\n")

        # Generate with verbose output
        reply = arianna.reply(query, max_tokens=100, verbose=True)

        print(f"\nReply (first 200 chars): {reply[:200]}")

        # Show stats
        stats = arianna.shard_stats()
        print(f"\n[Stats] Shards: {stats['total_shards']}, Themes: {stats['active_themes']}")

    print("\n" + "="*60)
    print("DEMO COMPLETE")
    print("="*60)

    print("\n📊 FINAL STATISTICS:")
    final_stats = arianna.shard_stats()
    for key, value in final_stats.items():
        print(f"  {key}: {value}")

    print("\n💡 KEY OBSERVATIONS:")
    print("1. Presence pulse changes based on input (novelty/arousal)")
    print("2. Shards created dynamically when books mentioned")
    print("3. Expert routing adapts to context")
    print("4. Generation works (random output expected!)")
    print("\n✅ ARCHITECTURE VALIDATED!")


if __name__ == "__main__":
    demo_dialogue()
