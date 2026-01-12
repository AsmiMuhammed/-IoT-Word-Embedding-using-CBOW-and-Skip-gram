import argparse
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from utils import one_hot

# -----------------------------
# Command line arguments
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument(
    "--lrs",
    nargs="+",
    type=float,
    default=[0.1, 0.01, 0.001],
    help="List of learning rates"
)
args = parser.parse_args()

# -----------------------------
# Load Skip-gram dataset
# -----------------------------
data = pd.read_csv("skipgram_dataset.csv")

vocab_size = data.max().max() + 1

X = np.array([one_hot(int(i), vocab_size) for i in data["target_word"]])
y = np.array([one_hot(int(i), vocab_size) for i in data["context_word"]])

# -----------------------------
# Train model for each LR
# -----------------------------
for lr in args.lrs:
    print(f"\nTraining Skip-gram with learning rate = {lr}")

    model = Sequential([
        Dense(10, activation="linear", input_shape=(vocab_size,)),
        Dense(vocab_size, activation="softmax")
    ])

    model.compile(
        optimizer=SGD(learning_rate=lr),
        loss="categorical_crossentropy"
    )

    history = model.fit(X, y, epochs=args.epochs, verbose=1)

    # -----------------------------
    # Save loss
    # -----------------------------
    loss_file = f"loss_skipgram_lr_{lr}.txt"
    with open(loss_file, "w") as f:
        f.write("epoch,loss,learning_rate\n")
        for epoch, loss in enumerate(history.history["loss"], start=1):
            f.write(f"{epoch},{loss},{lr}\n")

    # -----------------------------
    # Save embeddings
    # -----------------------------
    embeddings = model.layers[0].get_weights()[0]
    emb_file = f"embeddings_skipgram_lr_{lr}.csv"
    pd.DataFrame(embeddings).to_csv(emb_file, index=False)

print("\nSkip-gram training completed for all learning rates.")
