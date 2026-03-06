"""
Train MLP sense classifiers.

Usage:
    python -m sense_stack.train                    # train all words
    python -m sense_stack.train --word bank        # train one word
    python -m sense_stack.train --model gpt2       # use GPT-2 features
"""

import argparse
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sense_stack.senses import SENSES, SENSE_COARSE
from sense_stack.corpus import load_corpus


MODEL_DIR = Path(__file__).parent / "models"


class SenseClassifierMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def extract_embeddings(sentences, words, model_name, device):
    """Extract contextual embeddings at target word positions."""
    from transformers import AutoModel, AutoTokenizer
    import re

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    embeddings = []
    valid_indices = []

    for i, (sentence, word) in enumerate(zip(sentences, words)):
        try:
            inputs = tokenizer(sentence, return_tensors='pt', truncation=True, max_length=128)
            tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

            # Find target word
            target_idx = None
            word_lower = word.lower()
            for j, token in enumerate(tokens):
                clean = token.replace('##', '').replace('Ġ', '').replace('▁', '').lower()
                if clean == word_lower:
                    target_idx = j
                    break

            if target_idx is None:
                for j, token in enumerate(tokens):
                    clean = token.replace('##', '').replace('Ġ', '').replace('▁', '').lower()
                    if word_lower.startswith(clean) and len(clean) > 1:
                        target_idx = j
                        break

            if target_idx is None:
                continue

            with torch.no_grad():
                outputs = model(inputs['input_ids'].to(device),
                                attention_mask=inputs['attention_mask'].to(device))
                hidden = outputs.last_hidden_state[0]

            embeddings.append(hidden[target_idx].cpu().numpy())
            valid_indices.append(i)
        except Exception:
            continue

    return np.array(embeddings), valid_indices


def train_word(word, model_name='bert-base-uncased', device=None, epochs=50, lr=1e-3):
    """Train an MLP classifier for a single word."""
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else \
                 'mps' if torch.backends.mps.is_available() else 'cpu'

    corpus = load_corpus(word=word)

    # Map to coarse senses
    senses_list = list(SENSES[word].keys())
    skip = {'other', f'{word}_proper'}

    sentences = []
    words = []
    labels = []
    for item in corpus:
        sense = SENSE_COARSE.get(item['sense'], item['sense'])
        if sense in skip or sense not in senses_list:
            continue
        sentences.append(item['text'])
        words.append(word)
        labels.append(senses_list.index(sense))

    print(f"\n{word}: {len(sentences)} samples, {len(senses_list)} senses")

    # Extract embeddings
    print(f"  Extracting {model_name} embeddings...")
    embeddings, valid_idx = extract_embeddings(sentences, words, model_name, device)
    labels = np.array([labels[i] for i in valid_idx])

    print(f"  Got {len(embeddings)} embeddings ({embeddings.shape[1]}d)")

    # Train/val split
    np.random.seed(42)
    perm = np.random.permutation(len(embeddings))
    split = int(0.85 * len(perm))
    train_idx, val_idx = perm[:split], perm[split:]

    X_train = torch.tensor(embeddings[train_idx], dtype=torch.float32)
    y_train = torch.tensor(labels[train_idx], dtype=torch.long)
    X_val = torch.tensor(embeddings[val_idx], dtype=torch.float32)
    y_val = torch.tensor(labels[val_idx], dtype=torch.long)

    # Train
    model = SenseClassifierMLP(embeddings.shape[1], len(senses_list)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)

    best_acc = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()

        # Validate
        model.eval()
        with torch.no_grad():
            logits = model(X_val.to(device))
            preds = logits.argmax(dim=-1).cpu()
            acc = (preds == y_val).float().mean().item()

        if acc > best_acc:
            best_acc = acc
            best_state = model.state_dict().copy()

    model.load_state_dict(best_state)
    print(f"  Best val accuracy: {best_acc:.1%}")

    # Save
    MODEL_DIR.mkdir(exist_ok=True)
    save_path = MODEL_DIR / f"classifier_{word}_{model_name.replace('/', '_')}.pt"
    torch.save({
        'model': model.cpu(),
        'senses': senses_list,
        'model_name': model_name,
        'input_dim': embeddings.shape[1],
        'accuracy': best_acc,
    }, save_path)
    print(f"  Saved to {save_path}")

    return best_acc


def main():
    parser = argparse.ArgumentParser(description="Train sense classifiers")
    parser.add_argument('--word', type=str, help='Train for a specific word')
    parser.add_argument('--model', default='bert-base-uncased', help='Feature extractor model')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=50)
    args = parser.parse_args()

    words = [args.word] if args.word else list(SENSES.keys())

    results = {}
    for word in words:
        acc = train_word(word, args.model, args.device, args.epochs)
        results[word] = acc

    print(f"\n{'='*40}")
    print("Summary:")
    for word, acc in results.items():
        print(f"  {word}: {acc:.1%}")
    print(f"  Overall: {np.mean(list(results.values())):.1%}")


if __name__ == '__main__':
    main()
