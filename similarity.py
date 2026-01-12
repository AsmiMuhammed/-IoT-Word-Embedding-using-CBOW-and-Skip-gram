import pandas as pd
from utils import cosine_similarity

emb = pd.read_csv("embeddings_cbow.csv").values

vocab = {}
with open("vocab.txt") as f:
    for line in f:
        word, idx, _ = line.strip().split(",")
        vocab[word] = int(idx)

queries = ["sensor", "mqtt", "gateway", "firmware", "battery"]

with open("similarity_results.txt", "w") as f:
    for q in queries:
        if q not in vocab:
            continue
        q_vec = emb[vocab[q]]

        sims = []
        for w, i in vocab.items():
            sims.append((w, cosine_similarity(q_vec, emb[i])))

        sims = sorted(sims, key=lambda x: x[1], reverse=True)[1:6]

        f.write(f"\nQuery: {q}\n")
        for w, s in sims:
            f.write(f"{w}: {s:.4f}\n")
