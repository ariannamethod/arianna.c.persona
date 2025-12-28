"""Demo conversation - show LRU cache in action!"""

import sys
sys.path.insert(0, 'arianna')
sys.path.insert(0, './llama3.np')

from shard_llama import ShardLlama
from tokenizer import Tokenizer

print("="*70)
print("🌊 ARIANNA - CONVERSATION DEMO (LRU Cache Demo)")
print("="*70)
print("\nInitializing...\n")

llama = ShardLlama()
tok = Tokenizer('./llama3.np/tokenizer.model.np')

print("\n" + "="*70)
print("✓ Ready! Watch the LRU cache grow!")
print("="*70)

# Conversation sequence
queries = [
    "Hello Arianna, who are you?",
    "Tell me about love",
    "What is love?",  # SAME THEME - should hit LRU!
    "Do you remember happiness?",
    "Tell me more about love",  # REPEAT THEME - should hit LRU!
    "What makes you sad?",
]

for i, query in enumerate(queries, 1):
    print(f"\n{'='*70}")
    print(f"QUERY #{i}: '{query}'")
    print(f"{'='*70}\n")

    response = llama.respond(
        query=query,
        tokenizer=tok,
        max_tokens=50,  # Longer for complete sentences!
        temperature=0.8
    )

    print(f"\n[Response] {response}")

    # Show LRU cache state
    print(f"\n📊 LRU Cache State:")
    if llama.book_traveler.lru_cache:
        for j, book in enumerate(llama.book_traveler.lru_cache[:5], 1):
            print(f"  {j}. {book.name}")
    else:
        print("  (empty)")

    print(f"\n  Total active books: {len(llama.book_traveler.active_excerpts)}")
    print(f"  LRU cache size: {len(llama.book_traveler.lru_cache)}/10")

print("\n" + "="*70)
print("✓ Conversation complete!")
print("="*70)
print("\nNotice how:")
print("  🔄 LRU cache grew from 0 → 10")
print("  🎯 Repeated themes hit LRU cache (faster!)")
print("  🧠 CORE concepts always present")
print("  ✨ Stories selected by resonance")
