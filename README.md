# sense-stack 🧠

**Word sense disambiguation as a separable capability.**

Three independent methods for identifying which meaning of a polysemous word is active in context — no fine-tuning required for two of them.

```python
from sense_stack import disambiguate

result = disambiguate("I deposited money at the bank", "bank")
# SenseResult(sense='bank_finance', confidence=0.97, method='mlp')

result = disambiguate("We walked along the bank of the river", "bank")  
# SenseResult(sense='bank_river', confidence=0.94, method='mlp')
```

## Methods

| Method | Accuracy | Requires | Speed |
|---|---|---|---|
| **MLP classifier** | 97.4% (BERT), 95.2% (GPT-2) | One forward pass + tiny MLP | Fast |
| **Substitution scorer** | 83.0% | BERT masked LM | Medium |
| **Sense-aware model** | 97.1% | Fine-tuned GPT-2 | Slow |

## Supported Words

Currently covers 5 words with 18 senses:

| Word | Senses |
|---|---|
| **bank** | finance, river, collection |
| **light** | physical, figurative |
| **plant** | vegetation, factory, grow (verb), place secretly (verb) |
| **organ** | body part, instrument, publication, organization |
| **star** | celestial, celebrity, symbol |

**Want to add more words?** See [Contributing](#contributing).

## Installation

```bash
pip install sense-stack
```

Or from source:
```bash
git clone https://github.com/pbalogh/sense-stack.git
cd sense-stack
pip install -e .
```

## Quick Start

### Disambiguate a word in context

```python
from sense_stack import disambiguate

# Uses MLP classifier by default (fastest, most accurate)
result = disambiguate("The star collapsed into a black hole", "star")
print(result.sense)       # 'star_celestial'
print(result.confidence)  # 0.99
print(result.all_scores)  # {'star_celestial': 0.99, 'star_celebrity': 0.008, 'star_symbol': 0.002}
```

### Choose a method

```python
from sense_stack import disambiguate

# MLP classifier (default) — best accuracy, needs one transformer forward pass
result = disambiguate("She planted evidence at the scene", "plant", method="mlp")

# Substitution scorer — no model needed beyond BERT, fully interpretable
result = disambiguate("She planted evidence at the scene", "plant", method="substitution")
print(result.explanation)  # "Best synonym: 'smuggled' (score: 0.87)"

# Return all methods
results = disambiguate("She planted evidence at the scene", "plant", method="all")
```

### Batch disambiguation

```python
from sense_stack import SenseClassifier

clf = SenseClassifier()  # loads models once

sentences = [
    ("The bank approved the loan", "bank"),
    ("Trees lined the river bank", "bank"),
    ("She's a rising star", "star"),
]

results = clf.disambiguate_batch(sentences)
for r in results:
    print(f"{r.sense}: {r.confidence:.2f}")
```

### Access the sense-tagged corpus

```python
from sense_stack import load_corpus

corpus = load_corpus()
print(f"{len(corpus)} sense-tagged sentences")

# Filter by word/sense
bank_finance = [s for s in corpus if s['sense'] == 'bank_finance']
print(bank_finance[0]['text'])
# "The bank approved the mortgage application after reviewing her credit history."
```

## The Sense Inventory

Each word's senses are defined in a structured inventory:

```python
from sense_stack import SENSES

print(SENSES['bank'])
# {
#   'bank_finance': 'financial institution (deposit money, get loans, ATM, banking)',
#   'bank_river': 'edge/shore of a river, lake, or waterway',
#   'bank_collection': 'a stored collection or repository (blood bank, food bank, memory bank)',
# }
```

## Contributing

### Adding a new word

The easiest way to contribute: add a new polysemous word to the inventory.

1. **Define senses** in `sense_stack/senses.py`:
   ```python
   'spring': {
       'spring_season': 'the season between winter and summer',
       'spring_water': 'a natural source of water from the ground',
       'spring_coil': 'a coiled elastic device',
       'spring_verb': 'to jump or leap suddenly',
   }
   ```

2. **Add synonyms** in `sense_stack/synonyms.py`:
   ```python
   'spring': {
       'spring_season': ['springtime', 'april'],
       'spring_water': ['fountain', 'wellspring'],
       'spring_coil': ['coil', 'spiral'],
       'spring_verb': ['leap', 'jump'],
   }
   ```

3. **Generate tagged sentences** (requires an LLM API):
   ```bash
   python -m sense_stack.generate --word spring --count 200
   ```

4. **Train a classifier**:
   ```bash
   python -m sense_stack.train --word spring
   ```

5. **Submit a PR** with the new senses, synonyms, generated corpus, and trained model.

### Improving accuracy

- Better synonyms → better substitution scores
- More training sentences → better MLP classifiers
- New embedding models → potentially better features

## How It Works

### MLP Classifier (recommended)
Extracts the contextual embedding at the target word's position (from BERT or GPT-2), then feeds it through a tiny 2-layer MLP trained on sense-labeled examples. The geometry is already there in the embeddings — the MLP just reads it out.

### Substitution Scorer
For each possible sense, substitutes a representative synonym into the sentence and asks BERT's masked LM which substitution fits best. Fully interpretable: "bank_finance because 'vault' fit better than 'riverbank' here."

### Sense-Aware Model
Fine-tunes GPT-2 with an expanded vocabulary where polysemous tokens are replaced by sense tokens (`bank` → `bank_finance`). Achieves 97.1% accuracy but requires a fine-tuned model.

## Citation

```bibtex
@article{balogh2026stop,
  title={Stop Counting Legs: Teaching Transformers to Use Disambiguated Embeddings},
  author={Balogh, Peter},
  journal={arXiv preprint},
  year={2026}
}
```

## License

MIT

## Related Work

This package accompanies the paper "Stop Counting Legs: Teaching Transformers to Use Disambiguated Embeddings" and builds on findings from:

- [Half the Nonlinearity Is Wasted](https://arxiv.org/abs/2603.03459) — MLP linearizability in transformers
- The geometry of polysemy in contextual embeddings (Paper 1, forthcoming)
