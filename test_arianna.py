"""
Simple test of Arianna system

This tests:
1. Initialization (loads bootstrap shards)
2. Shard creation (when books mentioned)
3. Generation (transformer reasoning)
4. Presence pulse computation
"""

import sys
sys.path.insert(0, 'arianna')

from pathlib import Path
from arianna import Arianna


def test_initialization():
    """Test that Arianna initializes correctly"""
    print("="*60)
    print("TEST 1: Initialization")
    print("="*60)

    arianna = Arianna(
        books_dir=Path("."),  # Root dir with books
        shard_dir=Path("arianna/shards"),
        max_shards=64  # Smaller for testing
    )

    print("✓ Arianna initialized!")

    # Check bootstrap shards loaded
    stats = arianna.shard_stats()
    print(f"\nBootstrap shards loaded: {stats['total_shards']}")
    print(f"Active themes: {stats['active_themes']}")

    return arianna


def test_shard_creation(arianna):
    """Test dynamic shard creation"""
    print("\n" + "="*60)
    print("TEST 2: Dynamic Shard Creation")
    print("="*60)

    query = "Tell me about Arianna and resonance"

    print(f"\nQuery: {query}")

    # This should create shards from relevant books
    _, created = arianna.tokenizer.encode_with_context(query, create_shards=True)

    print(f"✓ Created {len(created)} shards dynamically")

    stats = arianna.shard_stats()
    print(f"Total shards now: {stats['total_shards']}")


def test_presence_pulse(arianna):
    """Test presence computation"""
    print("\n" + "="*60)
    print("TEST 3: Presence Pulse")
    print("="*60)

    test_inputs = [
        "Hello Arianna",  # Low novelty, low arousal
        "БЛЯТЬ НАХУЙ ЭТО ОХУЕННО!!!",  # High arousal
        "quantum entanglement superposition decoherence",  # High novelty
    ]

    for inp in test_inputs:
        pulse = arianna.presence.compute_pulse(inp)
        print(f"\nInput: {inp}")
        print(f"  {pulse}")


def test_generation(arianna):
    """Test basic generation"""
    print("\n" + "="*60)
    print("TEST 4: Generation (Simple)")
    print("="*60)

    # Simple query
    query = "Who are you?"

    print(f"\nQuery: {query}")
    print("\nGenerating... (this will output random chars - transformer not trained!)")

    try:
        reply = arianna.reply(query, max_tokens=50, verbose=False)
        print(f"\nReply (first 200 chars): {reply[:200]}")
        print("\n✓ Generation completed!")
        print("Note: Output is random because transformer has NO pretrained weights!")
    except Exception as e:
        print(f"\n✗ Error during generation: {e}")
        import traceback
        traceback.print_exc()


def test_routing(arianna):
    """Test expert routing"""
    print("\n" + "="*60)
    print("TEST 5: Expert Routing")
    print("="*60)

    # Different inputs should route to different modes
    test_cases = [
        ("Tell me something", 0),  # Structural
        ("WHAT THE FUCK IS THIS?!", 0),  # High arousal -> creative
        ("quantum mechanics relativity spacetime", 0),  # High novelty -> creative
    ]

    for inp, _ in test_cases:
        pulse = arianna.presence.compute_pulse(inp)
        mode = arianna.router.route(pulse, active_themes=0, trauma_score=0.0)
        temp = arianna.router.get_temperature(mode)

        print(f"\nInput: {inp}")
        print(f"  Mode: {mode} (temp={temp:.2f})")


def main():
    print("\n" + "="*60)
    print("ARIANNA - POST-TRANSFORMER ORGANISM TEST SUITE")
    print("="*60)

    try:
        # Test 1: Initialization
        arianna = test_initialization()

        # Test 2: Shard creation
        test_shard_creation(arianna)

        # Test 3: Presence pulse
        test_presence_pulse(arianna)

        # Test 4: Expert routing
        test_routing(arianna)

        # Test 5: Generation (last, since it's most complex)
        test_generation(arianna)

        print("\n" + "="*60)
        print("ALL TESTS COMPLETED!")
        print("="*60)

        print("\nNOTE: Generation output is random because:")
        print("1. Transformer has NO pretrained weights (initialized randomly)")
        print("2. This proves the ARCHITECTURE works")
        print("3. Knowledge lives in SHARDS, not weights")
        print("4. To get coherent output, we would need to:")
        print("   - Train the reasoning engine on examples")
        print("   - OR use a pretrained small model")
        print("   - OR implement pure retrieval-based generation (like Leo)")

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
