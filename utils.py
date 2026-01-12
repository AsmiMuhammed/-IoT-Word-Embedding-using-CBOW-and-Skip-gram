import numpy as np

def one_hot(index, vocab_size):
    vec = np.zeros(vocab_size)
    vec[index] = 1
    return vec

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
