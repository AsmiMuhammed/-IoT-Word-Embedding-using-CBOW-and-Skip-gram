import argparse
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from utils import one_hot

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--lr", type=float, required=True)
args = parser.parse_args()

data = pd.read_csv("cbow_dataset.csv")

vocab_size = data.max().max() + 1

X, y = [], []

for _, row in data.iterrows():
    context_ids = row[:-1].values.astype(int)
    target_id = int(row[-1])

    context_vecs = [one_hot(i, vocab_size) for i in context_ids]
    X.append(np.mean(context_vecs, axis=0))
    y.append(one_hot(target_id, vocab_size))

X, y = np.array(X), np.array(y)

model = Sequential([
    Dense(10, activation="linear", input_shape=(vocab_size,)),
    Dense(vocab_size, activation="softmax")
])

model.compile(
    optimizer=SGD(learning_rate=args.lr),
    loss="categorical_crossentropy"
)

history = model.fit(X, y, epochs=args.epochs, verbose=1)

# Save loss 
loss_file = f"loss_cbow_lr_{args.lr}.txt"
with open(loss_file, "w") as f:
    f.write("epoch,loss,learning_rate\n")
    for i, loss in enumerate(history.history["loss"], start=1):
        f.write(f"{i},{loss},{args.lr}\n")

# Save embeddings
embeddings = model.layers[0].get_weights()[0]
pd.DataFrame(embeddings).to_csv("embeddings_cbow.csv", index=False)
