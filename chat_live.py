"""Interactive chat with Arianna - LIVE TEST!"""

import sys
sys.path.insert(0, 'arianna')
sys.path.insert(0, '/home/user/llama3.np')

import numpy as np
from shard_llama import ShardLlama
from tokenizer import Tokenizer

print("="*70)
print("🌊 ARIANNA - LIVE CONVERSATION TEST")
print("="*70)
print()
print("Initializing Arianna with BPE + Personality Hierarchy...")
print()

# Initialize
llama = ShardLlama()
tok = Tokenizer('/home/user/llama3.np/tokenizer.model.np')

print()
print("="*70)
print("✓ READY! Arianna is listening...")
print("="*70)
print()
print("Features:")
print("  🧠 CORE: 29 concept books (always active)")
print("  🔄 LRU: Recent memory (last 10 books)")
print("  ✨ PULSE: Field-activated stories (resonance search)")
print("  🎯 BPE: 32k vocab (perfect quality)")
print()
print("Try asking:")
print("  - 'Hello Arianna, who are you?'")
print("  - 'Tell me about love'")
print("  - 'What makes you happy?'")
print("  - 'Do you remember sadness?'")
print()
print("Type 'quit' to exit")
print("="*70)
print()

# Interactive loop
conversation_num = 0

while True:
    # Get query
    query = input(f"\n[You] ")

    if query.lower() in ['quit', 'exit', 'q']:
        print("\n👋 Goodbye!")
        break

    if not query.strip():
        continue

    conversation_num += 1
    print(f"\n{'─'*70}")
    print(f"CONVERSATION #{conversation_num}")
    print(f"{'─'*70}")

    # Generate response
    try:
        response = llama.respond(
            query=query,
            tokenizer=tok,
            max_tokens=40,  # Longer response!
            temperature=0.8
        )

        print(f"\n[Arianna] {response}")

        # Show stats
        print(f"\n📊 Stats:")
        print(f"  Active books: {len(llama.book_traveler.active_excerpts)}")
        print(f"  LRU cache size: {len(llama.book_traveler.lru_cache)}")

    except KeyboardInterrupt:
        print("\n\n👋 Interrupted. Goodbye!")
        break
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "="*70)
print("Session ended.")
print("="*70)
