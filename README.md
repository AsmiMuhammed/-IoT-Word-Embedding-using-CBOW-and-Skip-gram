# Custom Word Embeddings using CBOW and Skip-gram (IoT Corpus)

This project implements **custom word embedding learning from scratch** using
**Python and Keras (TensorFlow backend)**.  
Two models are implemented:
- **CBOW (Continuous Bag of Words)**
- **Skip-gram**

The models are trained **only on the provided IoT paragraph** and use **one-hot
encoded vectors** as inputs.  
No pre-trained embeddings such as **Word2Vec, GloVe, or FastText** are used.

---

## 1. Objective

The main objective of this project is to:

- Learn **10-dimensional (D = 10) word embeddings** from scratch
- Implement **CBOW** and **Skip-gram** models using Keras
- Use a **fixed context window size W = 4**
- Train neural networks using **one-hot encoded inputs**
- Evaluate learned embeddings using **cosine similarity**
- Strictly follow the constraints given in the problem statement

---

## 2. Corpus Used

Only the following **IoT paragraph provided in the question** is used as the corpus:

An IoT deployment collects sensor readings such as temperature, vibration, and
humidity from edge devices. Data is transmitted using MQTT to a gateway, then
forwarded to a cloud dashboard for alerting and analytics. Power management is
critical for battery-operated nodes, so sampling rate and sleep cycles are tuned
carefully. Firmware updates must be secure to prevent device hijacking.


- No additional paragraphs are added  
- No external or internet data is used  
- The paragraph is treated as a **single training corpus**

---

## 3. Project Structure



iot_word_embedding/
│
├── corpus.py
├── preprocess.py
├── dataset_builder.py
├── utils.py
├── train_cbow.py
├── train_skipgram.py
├── similarity.py
│
├── vocab.txt
├── cbow_dataset.csv
├── skipgram_dataset.csv
├── loss_cbow.txt
├── loss_skipgram.txt
├── embeddings_cbow.csv
├── embeddings_skipgram.csv
├── similarity_results.txt
│
├── README.md
└── REPORT.pdf


---

## 4. Description of Files

- **corpus.py**  
  Contains the given IoT paragraph used as the corpus.

- **preprocess.py**  
  Handles preprocessing steps such as:
  - Lowercasing
  - Removing punctuation
  - Tokenization
  - Vocabulary creation

- **dataset_builder.py**  
  - Builds the vocabulary
  - Assigns a unique ID and frequency to each word
  - Generates CBOW and Skip-gram datasets using window size W = 4
  - Saves `vocab.txt`, `cbow_dataset.csv`, and `skipgram_dataset.csv`

- **utils.py**  
  Contains helper functions for:
  - One-hot encoding
  - Cosine similarity calculation

- **train_cbow.py**  
  Trains the CBOW model using one-hot encoded context words.

- **train_skipgram.py**  
  Trains the Skip-gram model using one-hot encoded target words.

- **similarity.py**  
  Evaluates the learned embeddings using cosine similarity and prints
  Top-5 nearest words.

---

## 5. Requirements

- Python 3.x
- TensorFlow (Keras backend)
- NumPy
- Pandas

Install dependencies (one time):

```bash
pip install tensorflow numpy pandas

6. Running the Project in VS Code
Step 1: Open Project Folder

Open Visual Studio Code

Click File → Open Folder

Select the folder iot_word_embedding

Step 2: Open Terminal

Press Ctrl + `

Or go to Terminal → New Terminal

Make sure the terminal path points to the project folder.

Step 3: Build Vocabulary and Datasets
python dataset_builder.py


This generates:

vocab.txt

cbow_dataset.csv

skipgram_dataset.csv

Step 4: Train CBOW Model
python train_cbow.py --epochs 100 --lr 0.01


Outputs:

loss_cbow.txt

embeddings_cbow.csv

Step 5: Train Skip-gram Model
python train_skipgram.py --epochs 100 --lr 0.01


Outputs:

loss_skipgram.txt

embeddings_skipgram.csv

Step 6: Evaluate Embeddings
python similarity.py


Output:

similarity_results.txt

7. Model Architecture
CBOW Model
Context words (one-hot)
        ↓
Dense layer (10 units, linear)  ← Word Embedding
        ↓
Dense layer (V units, softmax)

Skip-gram Model
Target word (one-hot)
        ↓
Dense layer (10 units, linear)  ← Word Embedding
        ↓
Dense layer (V units, softmax)


The hidden Dense layer represents the 10-dimensional embedding

Linear activation is used (no non-linear activation)

Optimizer: Stochastic Gradient Descent (SGD)

8. Learning Rate Experiments

The models can be trained using different learning rates to observe
loss convergence:

python train_cbow.py --epochs 100 --lr 0.1
python train_cbow.py --epochs 100 --lr 0.01
python train_cbow.py --epochs 100 --lr 0.001


Loss values are recorded in loss_cbow.txt and loss_skipgram.txt.

9. Evaluation Using Cosine Similarity

Selected IoT-related query words (e.g., sensor, mqtt, gateway, firmware)

Computed cosine similarity between word embeddings

Printed Top-5 nearest words for each query

Results saved in similarity_results.txt

10. Compliance with Problem Statement

Implemented strictly in Python using Keras

Uses only the provided IoT corpus

No pre-trained embeddings used

One-hot encoding used as input

Embedding dimension fixed at D = 10

Context window fixed at W = 4

11. Conclusion

This project demonstrates the complete implementation of CBOW and Skip-gram
models from scratch using Keras. Even with a small IoT-specific corpus, the
models are able to learn meaningful word representations and capture semantic
relationships between domain-specific terms.