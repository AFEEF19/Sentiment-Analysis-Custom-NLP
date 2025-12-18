# Sentiment Analysis using Custom NLP Pipeline

## 📌 Overview
This project implements a complete sentiment analysis pipeline from scratch as part of a µLearn AI task.
The goal is to understand how text processing, embeddings, and classification work at a low level,
and to compare a custom-built pipeline with a stronger pre-trained baseline.

---

## 🎯 Objective
- Implement a **custom subword tokenizer** from scratch (BPE-based)
- Train **custom word embeddings**
- Use these embeddings in a **text classification model**
- Compare performance with a **pre-trained baseline**
- Evaluate using standard metrics (Accuracy, Precision, Recall, F1-score)

---

## 📂 Dataset
**IMDB Movie Reviews Dataset**
- 50,000 reviews
- Balanced positive and negative classes
- Used for sentiment classification

Dataset Source:  
https://ai.stanford.edu/~amaas/data/sentiment/

---

## 🔹 Custom Subword Tokenizer
- Implemented a basic **Byte Pair Encoding (BPE)** tokenizer from scratch
- No external tokenizer libraries were used
- Tokenizer trained on a subset of the dataset
- Learned vocabulary saved as `subword_vocab.json`
- Sample sentences were tokenized to demonstrate tokenizer behavior

---

## 🔹 Custom Word Embeddings

The word embeddings were trained from scratch using a **data-driven co-occurrence
learning objective**. A co-occurrence matrix was constructed from the dataset,
and embeddings were learned by minimizing reconstruction error using
**stochastic gradient descent (SGD)**.

This approach is inspired by matrix factorization methods (similar to GloVe)
and ensures that embeddings are **genuinely learned from data**, rather than
randomly initialized. Training was performed on a representative subset of the
dataset to keep computation efficient.


---

## 🔹 Text Classification
### Custom Pipeline
- Reviews converted to vectors using custom embeddings
- Logistic Regression used as the classifier
- Evaluated using Accuracy, Precision, Recall, and F1-score

### Baseline Comparison
- TF-IDF vectorization used as a strong baseline
- Logistic Regression applied on TF-IDF features
- Results compared with the custom pipeline

---

## 📊 Results & Comparison

- The custom tokenizer and embeddings pipeline demonstrates a complete NLP
  workflow built entirely from scratch.
- Due to limited training data and simplified tokenization, the custom pipeline
  performs lower than the TF-IDF baseline.
- The TF-IDF baseline benefits from strong statistical text representations and
  performs significantly better.
- This comparison highlights the trade-off between custom-built NLP pipelines
  and pre-trained or statistically optimized representations.

---

## 🧠 Learning Outcomes
- Understanding subword tokenization concepts
- Training word embeddings from scratch
- Building an end-to-end NLP pipeline
- Comparing custom and pre-trained representations
- Evaluating models using standard ML metrics

---

## ⚙️ Tools & Technologies
- Python
- NumPy
- Pandas
- Scikit-learn
- Google Colab

---

## 🏁 Conclusion
This project demonstrates the complete workflow of sentiment analysis,
from raw text processing to classification and evaluation,
while emphasizing the trade-offs between custom NLP pipelines
and pre-trained text representations.
