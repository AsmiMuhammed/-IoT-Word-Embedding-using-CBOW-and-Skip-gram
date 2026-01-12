import csv
from corpus import corpus
from preprocess import preprocess_text, build_vocab

W = 4

tokens = preprocess_text(corpus)
vocab, freq = build_vocab(tokens)

# Save vocabulary 
with open("vocab.txt", "w") as f:
    f.write("word,word_id,frequency\n")
    for word, idx in vocab.items():
        f.write(f"{word},{idx},{freq[word]}\n")

encoded = [vocab[word] for word in tokens]

cbow_rows = []
skipgram_rows = []

for i in range(W, len(encoded) - W):
    context = encoded[i-W:i] + encoded[i+1:i+W+1]
    target = encoded[i]

    cbow_rows.append(context + [target])

    for ctx in context:
        skipgram_rows.append([target, ctx])

# Save CBOW dataset 
with open("cbow_dataset.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "context_1","context_2","context_3","context_4",
        "context_5","context_6","context_7","context_8",
        "target"
    ])
    writer.writerows(cbow_rows)

# Save Skip-gram dataset
with open("skipgram_dataset.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["target_word", "context_word"])
    writer.writerows(skipgram_rows)
