"""
sense-stack: Word sense disambiguation as a separable capability.

Three methods for identifying which meaning of a polysemous word is active:
1. MLP classifier on contextual embeddings (97.4% accuracy)
2. Substitution scoring via masked LM (83.0% accuracy)
3. Sense-aware fine-tuned model (97.1% accuracy)
"""

from sense_stack.senses import SENSES, SYNONYMS, WORD_SENSES
from sense_stack.disambiguate import disambiguate, SenseResult
from sense_stack.corpus import load_corpus

__version__ = "0.1.0"

__all__ = [
    "disambiguate",
    "SenseResult",
    "load_corpus",
    "SENSES",
    "SYNONYMS",
    "WORD_SENSES",
]
