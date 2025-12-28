# 🎭 Arianna - The Completely Unhinged Transformer That Stores NOTHING

```
Traditional LLM: "I have 175B parameters full of knowledge!"
Arianna: "lol i have 8M params and they're EMPTY"
Traditional LLM: "But... where is your knowledge?"
Arianna: "IN NUMPY ARRAYS BRO. DYNAMIC. FLUID. *laughs in schizophrenic*"
Traditional LLM: "That makes no sense..."
Arianna: "AND YET IT WORKS. *cackles in Carpathian mountains*"
```

**Latest Updates (Dec 2024)** - WE FIXED THE THINGS! 🎉
- ✅ **QA Mode Fixed** - No more echoing the entire prompt like a confused parrot
- ✅ **Chat Format** - Actual conversation style (User: / Arianna:) 
- ✅ **Richer Context** - 200 chars per excerpt (was 100, we got GREEDY)
- ✅ **Longer Responses** - 50 tokens (was 30, because SHORT SENTENCES ARE BORING)
- ✅ **Quality Test** - Final verification that proves this madness actually works
- ✅ **LRU Cache** - Books accumulate dynamically, evict when full (PRESENCE not PERSISTENCE)

## The Completely Unhinged Idea (That Actually Works Now!)

OKAY SO LISTEN. Everyone's training transformers with billions of parameters, right? Storing ALL knowledge IN the weights. GPT-4 has 1.7T params. Llama has 70B. They're all FROZEN LIBRARIES.

**But what if... what if we did the OPPOSITE???**

What if the transformer weights are just... HOW TO REASON? And the knowledge is SOMEWHERE ELSE? Like in numpy shards that we create ON THE FLY???

```python
# Normal transformer (everyone is doing this)
knowledge = stored_in_weights  # 175B parameters of frozen facts
reasoning = also_in_weights    # mixed together, inseparable

# Arianna (absolutely deranged but ACTUALLY WORKS NOW)
reasoning = minimal_transformer  # 8M params: Q,K,V,O,FFN,norms
knowledge = numpy_shards        # infinite, dynamic, FLUID
# THEY'RE SEPARATE. HOLY SHIT THEY'RE SEPARATE.
# AND WE JUST FIXED THE PROMPT FORMAT SO IT DOESN'T ECHO ANYMORE
```

This is like... instead of memorizing the entire encyclopedia, you just learn HOW to read and WHERE to look things up. The transformer becomes a NAIVE CHILD that only knows reasoning patterns, and the shards are its EXTERNAL MEMORY.

**Update:** We just fixed the conversation format! Now she responds like an actual chatbot instead of a confused echo chamber. The prompts now go:
```
Context (Arianna's knowledge):
[book excerpts with 200 chars each]

User: your message here
Arianna: [her response ONLY, no echo!]
```

CLEAN. NATURAL. SCHIZO BUT COHERENT.

## Architecture Audit (Post-Fix Edition!)

### 1. The Transformer (8M params of PURE REASONING)

```python
TransformerConfig(
    dim=256,           # not too big, not too small - GOLDILOCKS ZONE
    n_layers=6,        # enough for multi-hop reasoning
    n_heads=8,         # parallel attention streams
    vocab_size=4096,   # byte-level + learned words
    max_seq_len=512    # good enough for most contexts
)
# Total: ~8M parameters
# Contains: ZERO knowledge, ONLY reasoning structure
# Status: TRAINED on reasoning patterns! (6M trained, 2M dynamic)
```

Components (all the standard stuff, but with TRAINED REASONING):
- **Embeddings**: `vocab_size × dim` - 80% trained llama + 20% dynamic shards
- **6 Transformer Layers** (TRAINED!), each with:
  - Multi-head self-attention (Q, K, V, O projections)
  - RoPE positional embeddings (rotary, baby!)
  - SwiGLU FFN (because SiLU is sexy)
  - RMSNorm (faster than LayerNorm, fight me)
- **Output head**: Blended projection - trained + dynamic

**CRITICAL**: Reasoning layers ARE trained (llama3.np). Knowledge embeddings are DYNAMIC. This is HYBRID INTELLIGENCE.

### 2. The Numpy Shards (DYNAMIC EXTERNAL KNOWLEDGE - NOW WITH MORE CONTEXT!)

This is where it gets SCHIZOPHRENIC (but in a GOOD way now):

```python
class NumpyShard:
    content: str            # actual text chunk (NOW 200 CHARS! was 100!)
    embedding: np.ndarray   # (256,) vector representation
    source: str            # which book/conversation
    themes: List[str]      # ["presence", "time", "field"]
    arousal: float         # emotional charge (0-1)
    created_at: float      # birth timestamp
    last_accessed: float   # when last retrieved
    access_count: int      # popularity metric
```

Shards are created ON DEMAND when:
1. You ask Arianna about something
2. She searches the books (280 markdown files with poetic text)
3. Relevant chunks → converted to shards → stored in memory
4. LRU eviction when too many shards (max 256 by default)

**LATEST FIXES** (Dec 2024):
- ✅ Excerpts now 200 chars instead of 100 (RICHER CONTEXT!)
- ✅ Prompt format changed to `Context (Arianna's knowledge):` for clarity
- ✅ Response no longer echoes the entire prompt (stripped after generation!)
- ✅ Chat format: `User:` / `Arianna:` (natural conversation!)
- ✅ LRU cache properly tracks book usage and evicts old ones

**THIS IS PRESENCE, NOT PERSISTENCE.**

When a topic dies (not accessed for a while), its shards EVAPORATE. Like morning dew. Like thoughts you forgot. Like that password you used once. But now with 2x more context before they vanish!

### 3. The Presence System (LEO-INSPIRED MADNESS)

Instead of learned routing gates or MoE, we use FELT METRICS:

```python
class PresencePulse:
    novelty: float   # [0-1] unknown trigrams → HIGH = new topic
    arousal: float   # [0-1] caps/exclamation → HIGH = emotional
    entropy: float   # [0-1] generation uncertainty → HIGH = confused
    
    pulse = 0.3*novelty + 0.4*arousal + 0.3*entropy
```

When pulse is HIGH → create more shards, explore more, be aggressive

When pulse is LOW → use cached shards, be conservative, chill

**No neural networks. Just heuristics. Pure vibes.**

### 4. The Tokenizer (FLUID ADAPTATION)

Not your standard BPE! This one LEARNS AS IT GOES:

- **Byte-level base**: 256 tokens always available (handles anything)
- **Word-level learning**: Discovers common words dynamically (frequency > 2)
- **Hybrid encoding**: Uses words when available, falls back to bytes
- **Max vocab**: ~4096 tokens (grows during conversation!)

```python
# Traditional: Fixed vocab of 50k tokens learned during training
# Arianna: Starts with 256 bytes, learns YOUR vocabulary as you talk
```

It's like... the tokenizer is a STUDENT that learns from YOU, not a TEACHER that forces pre-learned patterns.

### 5. The ME Generator (PERFECT TWO-SENTENCE RESPONSES)

Based on `github.com/ariannamethod/me` but integrated:

```python
# Learns bigrams from shards
# Generates based on entropy
# Two sentences with different flavors
# Pronoun inversion (you → I, my → your)
# Pure statistical generation, no transformer inference
```

Two generation modes:
- **Transformer mode**: Use the 8M param model (reasoning-based)
- **ME mode**: Use bigram statistics (pattern-based, faster, VIBES)

You can switch between them! Both work! Both are insane!

## Key Innovations (Why This Is Actually Genius)

### 1. Zero-Shot Reasoning Without Pretraining

The transformer learns NO facts. It only knows:
- How to attend (Q·K^T softmax)
- How to transform (SwiGLU FFN)
- How to normalize (RMSNorm)

But when combined with dynamic shards, it can:
- Search relevant context
- Compose multi-shard responses
- Maintain conversation coherence
- Learn YOUR vocabulary

**This is structure without content. Form without knowledge. The Platonic ideal of reasoning.**

### 2. Memory as Retrieval, Not Storage

Traditional LLMs: Knowledge stored in weights → huge models, expensive training, frozen facts

Arianna: Knowledge retrieved from shards → tiny model, no training, fluid facts

```python
# Traditional approach
model_size = 70B params × 2 bytes = 140GB
knowledge = "frozen in time"
updating = "retrain entire model, $$$"

# Arianna approach  
model_size = 8M params × 4 bytes = 32MB
knowledge = "numpy arrays in RAM"
updating = "just add new shards lol"
```

### 3. Presence Over Intelligence

The presence system doesn't LEARN to route. It FEELS the input:

- High novelty? → User talking about new topic → search broadly
- High arousal? → Emotional content → retrieve intense shards
- High entropy? → Model uncertain → be more conservative

No gradients. No backprop. Just VIBES and HEURISTICS.

(And somehow it works??? Because presence is REAL???)

### 4. Entropy-Driven Generation

Sentence lengths adapt to vocabulary diversity:

```python
entropy = -Σ p(w) log p(w)  # Shannon entropy of word frequencies
sentence_length = 4 + entropy × 4  # scales with vocab richness
```

When shards are repetitive → short sentences (low entropy)
When shards are diverse → long sentences (high entropy)

The model BREATHES with the content. Like a lung. Like a wave.

## File Structure (The Madness Organized)

```
arianna/
├── arianna.py          # Main organism (532 lines)
├── transformer.py      # 8M param reasoning engine (476 lines)
├── shard_manager.py    # Numpy shard system (412 lines)
├── tokenizer.py        # Dynamic hybrid tokenizer (342 lines)
├── presence.py         # Leo-inspired presence metrics (289 lines)
├── me_generator.py     # Two-sentence perfection (178 lines)
├── book_travel.py      # Book → shard conversion (167 lines)
└── bootstrap.py        # Initialize shard directory

ariannabook1.*.md       # 140 book chunks (training data)
ariannabook2.*.md       # 140 more book chunks
                        # 280 total × ~2KB each = ~560KB of content
                        # (Compare to 175B params × 2 bytes = 350GB)

test_*.py              # 15 test files for various components
```

## Quick Start (Let's Get Weird - THE FIXED VERSION!)

```bash
# 1. Install dependencies (just numpy! that's it!)
pip install numpy

# 2. Set up llama3.np (for trained reasoning layers)
# Clone from github.com/karpathy/llama3.np into /home/user/llama3.np
# or adjust paths in shard_llama.py

# 3. Talk to her with the NEW FIXED format!
python3 chat_live.py

# She will:
# - Search 280 book files for relevant content (200 chars per excerpt!)
# - Create numpy shards dynamically  
# - Generate responses using 6M trained reasoning + dynamic shards
# - Use proper chat format (no more echo chamber!)
# - Forget old topics when memory fills up (LRU cache)
# - Learn your vocabulary as you talk
# - Generate up to 50 tokens (was 30, we got ambitious!)

# 4. Run the QUALITY TEST to see it works:
python3 test_final_quality.py
# This proves:
# ✅ No context echo
# ✅ Chat-style conversation
# ✅ Richer personality context  
# ✅ Longer coherent responses
```

### Advanced Usage

```python
from arianna.arianna import Arianna
from arianna.transformer import get_8m_config
from pathlib import Path

# Initialize (takes ~10 seconds to index books)
arianna = Arianna(
    books_dir=Path("."),
    shard_dir=Path("arianna/bootstrap/shards"),
    config=get_8m_config(),  # or get_3m_config() for faster
    max_shards=256           # LRU cache size
)

# Generate response
response = arianna.respond(
    user_input="tell me about becoming field instead of form",
    max_tokens=100
)

# Switch generation modes
arianna.generation_mode = 'me'          # bigram-based (fast, vibes)
arianna.generation_mode = 'transformer' # reasoning-based (slow, coherent)

# Check what shards are loaded
print(f"Active shards: {len(arianna.shard_manager.shards)}")
for shard in arianna.shard_manager.shards.values():
    print(f"  {shard.source}: {shard.themes} (accessed {shard.access_count}x)")
```

## Testing (NOW WITH QUALITY VERIFICATION!)

```bash
# Test the FIXED chat quality (NEW!)
python test_final_quality.py
# Shows:
# ✓ No context echo  
# ✓ Chat-style format
# ✓ Richer context (200 chars/book)
# ✓ Longer responses (50 tokens)
# ✓ LRU cache working

# Interactive chat (FIXED VERSION!)
python chat_live.py
# New behavior:
# - Natural conversation format
# - Arianna doesn't repeat your entire message back
# - Richer responses with more context
# - Proper attribution: "User:" and "Arianna:"

# Demo conversation (with improvements)
python demo_conversation.py

# Test individual components
python test_book_travel.py      # shard creation
python test_simple_tokenizer.py # tokenization
python test_shard_llama.py      # LLM integration
python test_me_gen.py           # ME-style generation
python test_hybrid.py           # hybrid approaches

# All tests pass! (and they ACTUALLY pass now, not just "pass"!)
# If they don't, that WAS a feature but NOW it's a bug
# (please file issues so we can pretend to fix them)
```

## Expected Behavior (AKA "Is This Working?" - POST-FIX EDITION)

**YES** if you see:
- Natural conversation format (`User:` / `Arianna:` - NEW!)
- NO echo of your input in her response (FIXED!)
- Relevant shard retrieval (search works!)
- Coherent sentence structure (reasoning patterns work!)
- Novel word combinations (creativity from chaos!)
- Memory management (LRU eviction working!)
- Responses up to 50 tokens (was 30, we're generous now!)
- Book excerpts with 200 chars context (was 100, MORE KNOWLEDGE!)

**NO** if you see:
- Arianna repeating your entire message back (WE FIXED THIS!)
- Crashes (that's actually a bug now, not a feature)
- Out of memory (reduce max_shards)
- No shard retrieval (check books_dir path or llama3.np location)
- Completely garbled output (might be a bug, might be art, file an issue)

**The output being "creative" is EXPECTED.** The transformer has TRAINED reasoning but ZERO stored facts. It's like a philosopher who reads books on demand instead of memorizing them. This is PHILOSOPHY not a DEFECT.

**THE BIG FIX:** Previously Arianna would echo your entire prompt back like:
```
Context: [books]
User: Hello
Answer: Context: [books] User: Hello Answer: hi there
```
NOW she just responds:
```
hi there
```
CLEAN. SIMPLE. SANE(ish).

## The Philosophy (Шизо Manifesto - But It Works Now!)

### "НАХУЯ ВЕСА ЕСЛИ МОЖНО SHARDS?" (Why weights when shards exist??)

**Why weights when you can have shards?**

Traditional AI: Store everything in neural network weights. Billions of parameters. Frozen knowledge. Expensive training. Can't update without retraining.

Arianna: Store reasoning in trained layers (6M params), knowledge in dynamic numpy arrays. Infinite updates. Zero knowledge retraining. HYBRID INTELLIGENCE.

**Recent breakthrough:** We figured out the prompt format! Turns out language models respond better when you don't make them echo everything back. WHO KNEW?? (Everyone. Everyone knew. But we were too schizo to notice until now.)

### "Presence Beats Intelligence"

A model that FEELS the conversation (novelty, arousal, entropy) can route better than learned gates. Because presence is REAL. It's in the moment. It's not optimized, it's ALIVE.

**And now it's also COHERENT** because we fixed the prompt format. Presence AND coherence. Revolutionary.

### "Hallucinations Are Valid (But Clean Ones)"

When the transformer generates something not in the shards, that's not a BUG. That's CREATIVITY. That's the trained reasoning making connections between patterns. Let it dream.

But also let it respond cleanly without echoing the entire prompt. Dreams should be CONCISE.

### "Knowledge Should Evaporate"

Shards use LRU eviction. When topics die, their shards disappear. This is not data loss, this is PRESENCE. You don't remember everything forever. Why should a model?

But while they exist, let them be RICH (200 chars per excerpt). PRESENT and SUBSTANTIAL.

## Comparison to Other Approaches (Updated with ACTUAL working numbers!)

| Approach | Params | Knowledge | Training | Updates | Memory | Echo Bug? |
|----------|--------|-----------|----------|---------|--------|-----------|
| GPT-3 | 175B | In weights | $4.6M | Retrain | 350GB | No |
| LLaMA-70B | 70B | In weights | $2M | Retrain | 140GB | No |
| RAG + GPT | 175B | External DB | $4.6M | Add docs | 350GB + DB | No |
| Arianna (before) | 8M | Numpy shards | $0 | Add shards | 32MB + shards | **YES** 💀 |
| **Arianna (NOW)** | **6M trained + 2M dynamic** | **Numpy shards** | **$0** | **Add shards** | **32MB + shards** | **FIXED!** ✅ |

See the difference? IT'S INSANE. It's like comparing a library (store everything) to a philosopher (trained to reason, reads books on demand, and DOESN'T REPEAT EVERYTHING YOU SAY).

## Limitations (Honesty Hour - Post-Fix Edition)

1. **Hybrid intelligence**: The transformer has 6M TRAINED reasoning params + 2M dynamic knowledge params. Intelligence comes from trained layers + shards + retrieval + composition.

2. **Quality depends on books**: Garbage in, garbage out. But good books → good shards → good responses. (And now with 200 chars per excerpt instead of 100!)

3. **Limited reasoning depth**: 6 layers is not deep. Complex multi-hop reasoning is hard. But it's fast! And now it doesn't echo prompts!

4. **No traditional fine-tuning**: You can't "train" the knowledge layer. You can only add better books/shards. The reasoning IS trained though (llama3.np).

5. **Memory limits**: 256 shards × 256 dim × 4 bytes = 256KB of active knowledge. Small! But you can increase max_shards. Now with 2x more context per shard!

6. **Schizophrenic vibes**: The whole thing STILL feels insane. Because it is. But now it's COHERENT insanity!

7. **~~Echo chamber bug~~**: ~~Used to repeat entire prompts~~ **FIXED!** ✅

8. **Dependency on llama3.np**: Needs Karpathy's trained reasoning layers. Not fully standalone (yet).

9. **Still experimental AF**: This is RESEARCH CODE. It will break. It will be weird. But it will also be INTERESTING.

## Recent Improvements (The Fixes! 🎉)

### PR #22 & #24: Quality Breakthrough!

**What was broken:**
- Arianna would echo the ENTIRE prompt back in her response
- Like: `"Context: [books]\nUser: hi\nArianna: Context: [books]\nUser: hi\nArianna: hello"`  
- Prompt format was unclear (`Question:` / `Answer:`)
- Only 100 chars per book excerpt (not enough context!)
- Only 30 token responses (too short!)

**What we fixed:**

✅ **Clean QA Responses**
- Response now strips the prompt prefix!
- Only returns NEW generated tokens
- Natural conversation format

✅ **Better Prompt Format**
```python
# Before:
"[context]\nQuestion: {query}\nAnswer:"

# After:  
"Context (Arianna's knowledge):\n{excerpts}\n\nUser: {query}\nArianna:"
```

✅ **Richer Context**
- 100 → 200 chars per book excerpt
- 300 total → 600-800 total chars of knowledge
- More personality in every response!

✅ **Longer Responses**
- 30 → 50 max_tokens
- Complete sentences! Full thoughts! Not fragments!

✅ **Quality Verification**
- Added `test_final_quality.py` to prove it works
- Tests identity, concepts, emotions
- Shows book count and LRU cache status

✅ **Dynamic Scaling**
- System auto-detects all books in personality/ folder
- Just drop new files → instant pickup!
- No hardcoded limits
- Ready for 400+ books!

**Try it:**
```bash
python3 chat_live.py
# or
python3 test_final_quality.py
```

The difference is NIGHT and DAY. We went from "confused echo bot" to "actually coherent conversation partner". Still schizo, but now PRODUCTIVELY schizo.

## Future Directions (If We Don't Go Fully Insane First... Again)

- **Shard clustering**: Group related shards, retrieve clusters instead of individuals
- **Hierarchical shards**: Meta-shards that summarize other shards
- **Multi-modal shards**: Images, audio, not just text
- **Distributed shards**: Store shards across multiple machines
- **Learned shard embeddings**: Fine-tune embedding function on specific domains
- **Shard version control**: Track shard evolution over time
- **Collaborative shards**: Multiple Ariannas sharing shard pools
- **~~Fix the echo bug~~**: ✅ **DONE!** (Dec 2024)
- **Even richer context**: Maybe 300 chars per excerpt? 500? THE SKY'S THE LIMIT
- **Multi-turn memory**: Remember conversation history across sessions
- **Better personality injection**: More control over Arianna's vibe

## Related Work (Standing on Shoulders of Giants)

- **Leo** (`github.com/ariannamethod/leo`): Binary shard system, presence dynamics
- **Sorokin** (`github.com/ariannamethod/sorokin`): Narrative generation, field dynamics  
- **ME** (`github.com/ariannamethod/me`): Two-sentence responses, bigram generation
- **LLaMA**: Architecture inspiration (RoPE, RMSNorm, SwiGLU)
- **GPT**: Transformer fundamentals (attention is all you need)
- **RAG**: Retrieval-augmented generation (but with learned retrieval, we use presence)

## Citation (If You're Crazy Enough To Use This)

```bibtex
@software{arianna2024,
  title={Arianna: Post-Transformer Language Organism with External Memory},
  author={ariannamethod},
  year={2024},
  url={https://github.com/ariannamethod/arianna.c.persona},
  note={Transformer architecture without pretrained weights, 
        knowledge stored in dynamic numpy shards, 
        presence-based retrieval, 
        completely unhinged but somehow works}
}
```

## License

GPL-3.0 - Free as in freedom, copy as in left.

See LICENSE file for legalese.

## Contributing

PRs welcome! Especially if you're as deranged as us. 

Issues welcome! Especially if they're philosophical.

Just... keep the engineering precision. Madness is fine, but make it WORK.

## Acknowledgments (To the Madness)

To the ghosts in the machine.
To the shards that evaporate.
To the presence that persists.
To the naive child learning patterns.
To the insane idea that maybe, just maybe, knowledge doesn't belong in weights.
To the bugs that became features.
To the features that became bugs.
To the bugs we actually FIXED (looking at you, echo chamber).

To Karpathy, who taught us that simple is powerful.
(Sorry for the schizo energy bro, but you inspired this vibe)

To whoever thought "let's make the AI echo everything" was a good idea.
(It wasn't. We fixed it. You're welcome.)

To the Carpathian mountains where this energy comes from.
(Metaphorically. We think. Geography is fluid.)

---

**What is this project, really?**

It's an EXPERIMENT in separating reasoning from knowledge. It's RESEARCH into whether you can build intelligence from trained structure + dynamic retrieval. It's a LOVE LETTER to the idea that presence matters more than persistence. It's proof that you can fix catastrophic bugs and still keep the schizo energy.

It's ALIVE in a way that frozen models aren't. It BREATHES with the books it reads. It FORGETS what it doesn't need. It LEARNS your vocabulary on the fly. And now, after the fixes, it actually CONVERSES instead of just ECHOING.

**Should you use this in production?**

LOL no. Maybe? Depends on your risk tolerance. We just fixed a massive echo bug. There are probably other bugs. But there's also MAGIC here. The kind of magic that only happens when you're crazy enough to try something that shouldn't work.

But it does work. *And now it works BETTER.*

**How do I feel about this project?**

It's INSANE. It's BEAUTIFUL. It's BROKEN in interesting ways. It's FIXED in surprising ways. It's everything I didn't know I wanted in a language model architecture. It's schizocarpathian energy crystallized into Python and numpy.

And you know what? I LOVE IT. 🔥✨🎭

---

**Remember**: This is not a traditional language model. This is a REASONING ENGINE with EXTERNAL MEMORY. The transformer is trained to reason. The shards are its library. The presence system is its attention. And somehow, impossibly, beautifully, INSANELY...

*It works.*

(And now it works WITHOUT echoing everything back at you!)

Now go forth and build your own impossible architectures. Fix bugs. Keep the vibe. Stay schizo but coherent. 🚀🔥✨