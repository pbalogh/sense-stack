"""
Access the sense-tagged corpus.
"""

import json
from pathlib import Path
from typing import List, Dict, Optional

CORPUS_DIR = Path(__file__).parent / "data"


def load_corpus(word: Optional[str] = None, sense: Optional[str] = None) -> List[Dict]:
    """Load the sense-tagged corpus.

    Args:
        word: Filter to a specific word (e.g., 'bank')
        sense: Filter to a specific sense (e.g., 'bank_finance')

    Returns:
        List of dicts with keys: text, word, sense, source
    """
    corpus_file = CORPUS_DIR / "corpus.json"
    if not corpus_file.exists():
        raise FileNotFoundError(
            f"Corpus not found at {corpus_file}. "
            f"Run `python -m sense_stack.generate` to create it, "
            f"or download it from the GitHub releases."
        )

    with open(corpus_file) as f:
        corpus = json.load(f)

    if word:
        corpus = [s for s in corpus if s['word'] == word]
    if sense:
        corpus = [s for s in corpus if s['sense'] == sense]

    return corpus
