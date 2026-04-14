<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f0c29,50:302b63,100:24243e&height=200&section=header&text=Next%20Word%20Predictor&fontSize=48&fontColor=FF6B35&fontAlignY=38&desc=LSTM%20%E2%9A%94%EF%B8%8F%20GRU%20%7C%20Sequential%20Language%20Modeling&descAlignY=58&descSize=18&descColor=a0aec0&animation=fadeIn" width="100%"/>

<br/>

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Keras](https://img.shields.io/badge/Keras-Sequential-D00000?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)
[![NLP](https://img.shields.io/badge/NLP-Language%20Modeling-00D4AA?style=for-the-badge&logo=databricks&logoColor=white)]()
[![License](https://img.shields.io/badge/License-MIT-FF6B35?style=for-the-badge)](LICENSE)

<br/>

> **"Teaching machines to think one word at a time — comparing two titans of sequence modeling."**

<br/>

</div>

---

## 🧠 What Is This?

This project explores **next-word prediction** using two powerful Recurrent Neural Network architectures — **LSTM** (Long Short-Term Memory) and **GRU** (Gated Recurrent Unit). Both models are trained on the same text corpus and evaluated head-to-head, providing a clear empirical comparison of their performance on sequential language modeling.

Given a seed phrase, the model predicts the most probable next word — the foundation of autocomplete systems, text generators, and intelligent writing assistants.

```
Input:   "The quick brown fox"
LSTM →   "jumps"
GRU  →   "jumps"
```

---

## ⚔️ LSTM vs GRU — Architecture Duel

| Feature | LSTM | GRU |
|---|---|---|
| **Gates** | Input · Forget · Output (3 gates) | Update · Reset (2 gates) |
| **Cell State** | ✅ Separate cell state | ❌ No separate cell state |
| **Parameters** | More (heavier) | Fewer (lighter) |
| **Training Speed** | Slower | Faster |
| **Long-term Memory** | Excellent | Good |
| **Best For** | Complex, long sequences | Shorter sequences, faster training |

> Both architectures solve the **vanishing gradient problem** of vanilla RNNs, but differ in complexity and compute cost.

---

## 🏗️ Model Architecture

```
Text Corpus
    │
    ▼
┌─────────────────────────┐
│   Text Preprocessing    │  ← Tokenization, Lowercasing, Cleaning
└────────────┬────────────┘
             │
    ▼
┌─────────────────────────┐
│   Keras Tokenizer       │  ← Word → Integer mapping
│   Sequence Generation   │  ← N-gram sliding window
│   Padding               │  ← pre-padding to max_len
└────────────┬────────────┘
             │
    ┌────────┴────────┐
    ▼                 ▼
┌────────┐       ┌────────┐
│  LSTM  │       │  GRU   │
│ Model  │       │ Model  │
└───┬────┘       └────┬───┘
    │                 │
    ▼                 ▼
┌─────────────────────────┐
│  Embedding Layer        │
│  RNN Layer (128 units)  │
│  Dense + Softmax        │
└────────────┬────────────┘
             │
    ▼
 Predicted Next Word
```

---

## 🗂️ Project Structure

```
📁 Predicting-Next-Word-using-LSTM-and-GRU/
│
├── 📓 Next_Word_Prediction.ipynb    # Main notebook (LSTM + GRU training & eval)
├── 📄 README.md                     # You are here
│
├── 📁 data/
│   └── corpus.txt                   # Training text corpus
│
├── 📁 models/
│   ├── lstm_model.h5                # Saved LSTM model
│   └── gru_model.h5                 # Saved GRU model
│
└── 📁 tokenizer/
    └── tokenizer.pkl                # Saved Keras tokenizer
```

---

## 🛠️ Tech Stack

```python
core = {
    "language"   : "Python 3.9+",
    "framework"  : "TensorFlow 2.x / Keras",
    "environment": "Jupyter Notebook",
    "nlp"        : "Keras Tokenizer",
    "numerics"   : "NumPy",
}

models = ["LSTM", "GRU"]
layers = ["Embedding", "RNN Layer", "Dense (Softmax)"]
loss   = "categorical_crossentropy"
optim  = "Adam"
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/aryxn-builds/Predicting-Next-Word-using-LSTM-and-GRU.git
cd Predicting-Next-Word-using-LSTM-and-GRU
```

### 2. Install Dependencies

```bash
pip install tensorflow numpy jupyter
```

### 3. Launch the Notebook

```bash
jupyter notebook Next_Word_Prediction.ipynb
```

### 4. Run Prediction

```python
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle, numpy as np

# Load saved model & tokenizer
model     = load_model("models/lstm_model.h5")      # or gru_model.h5
tokenizer = pickle.load(open("tokenizer/tokenizer.pkl", "rb"))

def predict_next_word(seed_text, model, tokenizer, max_len):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_len-1, padding='pre')
    predicted  = np.argmax(model.predict(token_list), axis=-1)
    for word, index in tokenizer.word_index.items():
        if index == predicted:
            return word

# Example
print(predict_next_word("To be or not to", model, tokenizer, max_len=10))
# → "be"
```

---

## 📊 Results

| Metric | LSTM | GRU |
|---|---|---|
| **Training Accuracy** | ~85% | ~87% |
| **Validation Accuracy** | ~78% | ~80% |
| **Training Speed** | Slower | ~20% faster |
| **Parameters** | More | Fewer |
| **Convergence** | Stable | Slightly faster |

> 💡 **Key Takeaway:** GRU achieves comparable or slightly better accuracy with fewer parameters and faster training — making it a strong default choice for sequence tasks without extremely long dependencies.

---

## 🔬 How It Works

```
1. TOKENIZATION
   Raw text → Keras Tokenizer → Integer-encoded vocabulary

2. SEQUENCE GENERATION
   Sliding N-gram window → Input/Output sequence pairs

3. PADDING
   All sequences padded to uniform length (pre-padding)

4. ONE-HOT ENCODING
   Target word → One-hot vector over vocabulary

5. MODEL TRAINING
   Embedding → RNN (LSTM or GRU) → Dense (Softmax) → argmax → Predicted word

6. EVALUATION
   Training & Validation accuracy plotted per epoch
```

---

## 📈 Training Curves

Both models were trained with:

- **Optimizer:** Adam
- **Loss:** Categorical Crossentropy
- **Epochs:** 50–100
- **Batch Size:** 64
- **Embedding Dim:** 100

Training and validation loss/accuracy curves are plotted inside the notebook for side-by-side comparison.

---

## 🧩 Key Concepts Demonstrated

- ✅ Text corpus preprocessing & tokenization
- ✅ N-gram sequence generation with sliding window
- ✅ Keras Embedding layer for dense word representations
- ✅ LSTM architecture with cell state & 3-gate mechanism
- ✅ GRU architecture with simplified 2-gate mechanism
- ✅ Multi-class classification with Softmax output
- ✅ Model serialization (`.h5` + pickle tokenizer)
- ✅ Empirical comparison of RNN variants on the same task

---

## 🌐 Applications

> Next-word prediction powers some of the most widely used NLP systems in the world:

- 📱 **Mobile keyboard autocomplete** (SwiftKey, Gboard)
- ✍️ **AI writing assistants** (Grammarly, Notion AI)
- 🤖 **Chatbots & conversational agents**
- 🔍 **Search query suggestion**
- 🎓 **Language learning tools**

---

## 👨‍💻 Author

<div align="center">

**Aryan Yadav**
*AI/ML Engineer · Founder @ Blazion.io · B.Tech CSE (AI/ML)*

[![GitHub](https://img.shields.io/badge/GitHub-aryxn--builds-181717?style=for-the-badge&logo=github)](https://github.com/aryxn-builds)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=for-the-badge&logo=linkedin)](https://linkedin.com/in/aryan-yadav)

</div>

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f0c29,50:302b63,100:24243e&height=100&section=footer" width="100%"/>

*Built with 🔥 — If this helped you, drop a ⭐ on the repo!*

</div>