# Fake News Detection with Neural Network (TF-IDF + Feedforward NN)

A machine learning project aimed at detecting fake news articles using a lightweight, interpretable pipeline — without transformers like BERT. This project was built for portfolio purposes to demonstrate end-to-end ML capability: from data preprocessing to training, evaluation, and inference.

---

## Overview

This project classifies news articles as **Real** or **Fake** using traditional NLP techniques (TF-IDF) and a regularized feedforward neural network (FNN). It avoids using large transformer models, focusing instead on optimizing smaller architectures with strong generalization.

---

## Model Architecture

- Input: TF-IDF vectors (max 5000 features)
- Dense(64) + ReLU + L2 regularization + Dropout(0.6)
- Dense(64) + ReLU + L2 regularization + Dropout(0.6)
- Output: Dense(1) with Sigmoid

Optimized using Adam (lr = 0.0001) and Binary Crossentropy loss.

---

## Features

- TF-IDF vectorization with unigrams, bigrams, and trigrams
- Simple feedforward neural network with dropout & L2 regularization
- Early stopping to prevent overfitting
- Evaluation on Train, Dev, and Test sets
- Real-world inference on custom inputs
- Model & vectorizer saving for reuse

---

## Challenge Faced: Overfitting

Despite the balanced dataset, the model **initially overfitted** the training data due to:

- High feature sparsity from TF-IDF
- Lack of semantic representation
- Shallow neural architecture

### 🔧 Solutions Implemented:

- **Dropout (0.6)** on each hidden layer
- **L2 Regularization** (λ = 0.05)
- **EarlyStopping** based on validation loss
- **Stratified splitting** of data to maintain class distribution

These strategies significantly reduced the overfitting gap and improved generalization.

---

## Final Performance

| Dataset | Accuracy | Precision | Recall | F1 Score |
|---------|----------|-----------|--------|----------|
| Train   | ~99%     | 99%       | 99%    | 99%      |
| Dev     | ~99%     | 99%       | 99%    | 99%      |
| Test    | ~99%     | 99%       | 99%    | 99%      |

Note: While the scores are high, real-world performance may vary due to TF-IDF's limitations in capturing context.

---
