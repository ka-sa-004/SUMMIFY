# 🎬 IMDB Sentiment Analysis — GRU Neural Network

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange?style=flat-square&logo=pytorch)](https://pytorch.org)
[![Accuracy](https://img.shields.io/badge/Accuracy-85.6%25-brightgreen?style=flat-square)](/)
[![License](https://img.shields.io/badge/License-MIT-purple?style=flat-square)](LICENSE)

A deep learning project that classifies IMDB movie reviews as **Positive** or **Negative** using a GRU-based Recurrent Neural Network built with PyTorch.

---

## 🚀 Live Demo

[Click here to view the project] - (https://aradhy1234-star.github.io/MACHINE-LEARNING-PROJECTS/Sentiment_Analysis/)

---

## 📊 Results

| Metric | Score |
|---|---|
| Validation Accuracy | **85.6%** |
| Training Samples | 39,665 |
| Test Samples | 9,917 |
| Model | GRU (2-layer, hidden=128) |
| Features | TF-IDF bigrams (5,000 features) |

---

## 🗂 Project Structure

```
sentiment-analysis/
│
├── Sentiment_analysis_improved.ipynb   # Main notebook (training + evaluation)
├── sentiment_analysis_webpage.html     # Live demo webpage
├── IMDB_Dataset.csv                    # Dataset (50,000 reviews)
├── requirements.txt                    # Python dependencies
└── README.md
```

> After running the notebook, these files will also be generated:
> - `sentiment_model.pt` — saved model weights + config
> - `tfidf_vectorizer.pkl` — fitted TF-IDF vectorizer
> - `training_curves.png` — loss & accuracy plots
> - `confusion_matrix.png` — evaluation heatmap

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/sentiment-analysis.git
cd sentiment-analysis
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the notebook
```bash
jupyter notebook Sentiment_analysis_improved.ipynb
```

---

## 🧠 Model Architecture

```
Input: TF-IDF vector (5,000 features)
   ↓
GRU (2 layers, hidden=128, dropout=0.3)
   ↓
Dropout (0.3)
   ↓
Linear (128 → 1)
   ↓
Sigmoid → Positive/Negative
```

**Why GRU over vanilla RNN?**
- Better at capturing long-range dependencies in text
- Fewer parameters than LSTM but similar performance
- Less prone to vanishing gradients

---

## 🔄 Preprocessing Pipeline

1. **Lowercase** — normalize all text
2. **Remove HTML tags** — clean `<br />` artifacts from IMDB
3. **Remove URLs** — strip any web links
4. **Remove punctuation & digits** — keep only alphabetic tokens
5. **Remove stopwords** — filter common English words (NLTK)
6. **Porter Stemming** — reduce words to their root form

---

## 🔍 Inference — Use the Model

```python
import torch, joblib, re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Load saved artifacts
tfidf = joblib.load("tfidf_vectorizer.pkl")
checkpoint = torch.load("sentiment_model.pt", map_location="cpu")

# Rebuild model (copy SentimentRNN class from notebook)
cfg = checkpoint["model_config"]
model = SentimentRNN(**cfg)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Predict
result = predict_sentiment("This movie was absolutely fantastic!")
print(result)
# → {'label': 'Positive 😊', 'confidence': '94.21%', 'probability': 0.9421}
```

---

## 📈 Training Improvements vs Original

| Feature | Original | Improved |
|---|---|---|
| RNN type | Vanilla RNN | **GRU** |
| Loss function | BCELoss + sigmoid | **BCEWithLogitsLoss** |
| Layers | 1 | **2** |
| Dropout | ✗ | **✓ 0.3** |
| Gradient clipping | ✗ | **✓** |
| LR scheduler | ✗ | **✓ StepLR** |
| TF-IDF ngrams | unigram | **bigram** |
| Stratified split | ✗ | **✓** |
| Model checkpointing | ✗ | **✓** |
| URL regex | broken | **fixed** |
| Stopword removal | buggy | **fixed** |

---

## 📦 Dataset

**IMDB Movie Review Dataset**
- 50,000 reviews (25,000 positive, 25,000 negative)
- Balanced binary classification task
- Source: [Kaggle IMDB Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

---

## 🛠 Tech Stack

- **Python 3.9+**
- **PyTorch** — model training & inference
- **scikit-learn** — TF-IDF vectorization, metrics
- **NLTK** — text preprocessing
- **Pandas / NumPy** — data handling
- **Matplotlib / Seaborn** — visualizations

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙌 Acknowledgements

- IMDB Dataset from Kaggle
- PyTorch documentation & community
- NLTK for NLP utilities
