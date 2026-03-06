"""
Core disambiguation API.

Three methods:
1. mlp — MLP classifier on contextual embeddings (default, 97.4%)
2. substitution — synonym substitution scoring via BERT masked LM (83%)
3. all — run all methods and return results from each
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Literal

from sense_stack.senses import SENSES, SYNONYMS, WORD_SENSES, SUPPORTED_WORDS


@dataclass
class SenseResult:
    """Result of a disambiguation call."""
    sense: str
    confidence: float
    method: str
    word: str
    all_scores: Dict[str, float] = field(default_factory=dict)
    explanation: Optional[str] = None

    def __repr__(self):
        return f"SenseResult(sense='{self.sense}', confidence={self.confidence:.3f}, method='{self.method}')"


def disambiguate(
    sentence: str,
    word: str,
    method: Literal["mlp", "substitution", "all"] = "mlp",
    model: str = "bert-base-uncased",
    device: Optional[str] = None,
) -> SenseResult:
    """Disambiguate a polysemous word in context.

    Args:
        sentence: The sentence containing the word.
        word: The polysemous word to disambiguate.
        method: 'mlp' (default), 'substitution', or 'all'.
        model: Transformer model for feature extraction.
        device: PyTorch device ('cpu', 'cuda', 'mps'). Auto-detected if None.

    Returns:
        SenseResult with the predicted sense, confidence, and scores.

    Raises:
        ValueError: If the word is not in the supported vocabulary.
    """
    word = word.lower()
    if word not in SUPPORTED_WORDS:
        raise ValueError(
            f"'{word}' is not supported. Supported words: {SUPPORTED_WORDS}. "
            f"See README.md for how to add new words."
        )

    if method == "mlp":
        return _disambiguate_mlp(sentence, word, model, device)
    elif method == "substitution":
        return _disambiguate_substitution(sentence, word, device)
    elif method == "all":
        results = {}
        results['mlp'] = _disambiguate_mlp(sentence, word, model, device)
        results['substitution'] = _disambiguate_substitution(sentence, word, device)
        # Return the MLP result as primary (highest accuracy)
        primary = results['mlp']
        primary.explanation = (
            f"MLP: {primary.sense} ({primary.confidence:.1%}) | "
            f"Substitution: {results['substitution'].sense} ({results['substitution'].confidence:.1%})"
        )
        return primary
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'mlp', 'substitution', or 'all'.")


# ============================================================
# MLP Classifier
# ============================================================

_mlp_models = {}  # cache: (word, model_name) -> (classifier, label_encoder)
_embedding_model = {}  # cache: model_name -> (model, tokenizer)


def _get_embedding_model(model_name, device):
    """Load and cache the transformer model for feature extraction."""
    if model_name not in _embedding_model:
        import torch
        from transformers import AutoModel, AutoTokenizer

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else \
                     'mps' if torch.backends.mps.is_available() else 'cpu'

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device)
        model.eval()
        _embedding_model[model_name] = (model, tokenizer, device)

    return _embedding_model[model_name]


def _extract_embedding(sentence, word, model_name, device):
    """Extract contextual embedding at the target word's position."""
    import torch

    model, tokenizer, device = _get_embedding_model(model_name, device)

    inputs = tokenizer(sentence, return_tensors='pt', truncation=True, max_length=128)
    input_ids = inputs['input_ids'][0]

    # Find the target word's token position
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    word_lower = word.lower()

    # Find token(s) matching the word
    target_idx = None
    for i, token in enumerate(tokens):
        clean = token.replace('##', '').replace('Ġ', '').replace('▁', '').lower()
        if clean == word_lower:
            target_idx = i
            break

    # Fallback: partial match
    if target_idx is None:
        for i, token in enumerate(tokens):
            clean = token.replace('##', '').replace('Ġ', '').replace('▁', '').lower()
            if word_lower.startswith(clean) and len(clean) > 1:
                target_idx = i
                break

    if target_idx is None:
        raise ValueError(f"Could not find '{word}' in tokenized sentence: {tokens}")

    with torch.no_grad():
        outputs = model(inputs['input_ids'].to(device), attention_mask=inputs['attention_mask'].to(device))
        hidden = outputs.last_hidden_state[0]  # (seq_len, hidden_dim)

    return hidden[target_idx].cpu().numpy()


def _get_mlp_classifier(word, model_name):
    """Load a trained MLP classifier for a word."""
    import torch
    from pathlib import Path

    key = (word, model_name)
    if key not in _mlp_models:
        model_dir = Path(__file__).parent / "models"
        model_file = model_dir / f"classifier_{word}_{model_name.replace('/', '_')}.pt"

        if not model_file.exists():
            raise FileNotFoundError(
                f"No trained classifier for '{word}' with {model_name}. "
                f"Expected at {model_file}. "
                f"Run `python -m sense_stack.train --word {word}` to train one."
            )

        checkpoint = torch.load(model_file, map_location='cpu', weights_only=False)
        _mlp_models[key] = checkpoint

    return _mlp_models[key]


def _disambiguate_mlp(sentence, word, model_name, device) -> SenseResult:
    """Disambiguate using MLP classifier on contextual embeddings."""
    import torch
    import numpy as np

    embedding = _extract_embedding(sentence, word, model_name, device)
    checkpoint = _get_mlp_classifier(word, model_name)

    model = checkpoint['model']
    model.eval()
    senses = checkpoint['senses']  # ordered list of sense labels

    with torch.no_grad():
        logits = model(torch.tensor(embedding, dtype=torch.float32).unsqueeze(0))
        probs = torch.softmax(logits, dim=-1)[0].numpy()

    best_idx = np.argmax(probs)
    scores = {sense: float(probs[i]) for i, sense in enumerate(senses)}

    return SenseResult(
        sense=senses[best_idx],
        confidence=float(probs[best_idx]),
        method='mlp',
        word=word,
        all_scores=scores,
    )


# ============================================================
# Substitution Scorer
# ============================================================

_bert_model = {}


def _get_bert(device):
    """Load and cache BERT for masked LM scoring."""
    if 'model' not in _bert_model:
        import torch
        from transformers import BertForMaskedLM, BertTokenizer

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else \
                     'mps' if torch.backends.mps.is_available() else 'cpu'

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForMaskedLM.from_pretrained('bert-base-uncased').to(device)
        model.eval()
        _bert_model['model'] = model
        _bert_model['tokenizer'] = tokenizer
        _bert_model['device'] = device

    return _bert_model['model'], _bert_model['tokenizer'], _bert_model['device']


def _disambiguate_substitution(sentence, word, device) -> SenseResult:
    """Disambiguate using synonym substitution scoring."""
    import torch
    import numpy as np

    model, tokenizer, device = _get_bert(device)
    senses = WORD_SENSES[word]
    synonyms = SYNONYMS[word]

    scores = {}
    best_synonyms = {}

    for sense in senses:
        sense_synonyms = synonyms.get(sense, [])
        if not sense_synonyms:
            scores[sense] = float('-inf')
            continue

        best_score = float('-inf')
        best_syn = None

        for synonym in sense_synonyms:
            # Replace target word with [MASK]
            masked = re.sub(
                rf'(?<![a-zA-Z]){re.escape(word)}(?![a-zA-Z])',
                '[MASK]',
                sentence,
                count=1,
                flags=re.IGNORECASE
            )

            if masked == sentence:
                continue

            inputs = tokenizer(masked, return_tensors='pt', truncation=True, max_length=128)
            input_ids = inputs['input_ids'][0]
            mask_idx = (input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[0]

            if len(mask_idx) == 0:
                continue

            mask_idx = mask_idx[0].item()
            synonym_id = tokenizer.convert_tokens_to_ids(synonym.lower())

            with torch.no_grad():
                outputs = model(inputs['input_ids'].to(device), attention_mask=inputs['attention_mask'].to(device))
                logits = outputs.logits[0, mask_idx]
                log_probs = torch.log_softmax(logits, dim=-1)
                score = log_probs[synonym_id].item()

            if score > best_score:
                best_score = score
                best_syn = synonym

        scores[sense] = best_score
        best_synonyms[sense] = best_syn

    # Normalize scores to probabilities
    import numpy as np
    score_values = np.array([scores[s] for s in senses])
    # Softmax
    exp_scores = np.exp(score_values - score_values.max())
    probs = exp_scores / exp_scores.sum()
    prob_dict = {sense: float(probs[i]) for i, sense in enumerate(senses)}

    best_sense = max(prob_dict, key=prob_dict.get)

    return SenseResult(
        sense=best_sense,
        confidence=prob_dict[best_sense],
        method='substitution',
        word=word,
        all_scores=prob_dict,
        explanation=f"Best synonym: '{best_synonyms.get(best_sense, '?')}'"
    )
