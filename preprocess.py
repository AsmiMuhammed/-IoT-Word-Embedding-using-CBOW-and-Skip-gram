import re
from collections import Counter

def preprocess_text(corpus):
    text = " ".join(corpus).lower()
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = text.split()
    return tokens

def build_vocab(tokens):
    freq = Counter(tokens)
    vocab = {word: idx for idx, word in enumerate(freq.keys())}
    return vocab, freq
