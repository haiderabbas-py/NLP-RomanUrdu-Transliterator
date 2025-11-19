# RomanUrdu-Transliterator

A complete **Neural Machine Translation (NMT)** system that converts **Urdu text into Roman Urdu** using a **Bidirectional LSTM Encoder** and **LSTM Decoder** implemented in **PyTorch**. This project was developed as part of my **NLP course assignment** and includes preprocessing, model architecture, training pipeline, evaluation metrics, and an optional Streamlit deployment.

---

# 📌 Project Overview

This project performs **character-level transliteration** from *Urdu script* → *Roman Urdu* using a deep learning approach. Rather than using simple rule-based mappings, the model learns how Roman Urdu is formed from Urdu characters using a sequence-to-sequence architecture.

---

# 🎯 Objectives

* Build a **Seq2Seq NMT system** using:

  * **BiLSTM Encoder** (2 layers)
  * **LSTM Decoder** (4 layers)
* Train on *low-resource poetic Urdu data*
* Compare different hyperparameters
* Evaluate using **BLEU**, **Perplexity**, and **CER**
* Deploy the final model using **Streamlit**

---

# 📂 Dataset

Dataset used: **urdu_ghazals_rekhta**

* Contains Urdu text, transliteration, and Hindi.
* We extract **Urdu → Roman Urdu pairs**.
* Additional preprocessing applied (normalization, diacritics removal).

Dataset link:
👉 [https://github.com/amir9ume/urdu_ghazals_rekhta](https://github.com/amir9ume/urdu_ghazals_rekhta)

---

# 🧹 Preprocessing Steps

### ✔ Unicode normalization

### ✔ Diacritics removal

### ✔ Standardization of Alef/Yeh forms

### ✔ Custom Urdu → Roman Urdu mapping rules

### ✔ Tokenization (character-level)

These steps are implemented inside `preprocessing/text_cleaning.py`.

---

# 🧠 Model Architecture

### **🔹 Encoder – BiLSTM**

* Learns bidirectional context of Urdu characters
* 2 layers
* Hidden size: 256/512
* Embedding size: 128/256/512

### **🔹 Decoder – LSTM**

* 4 layers
* Uses teacher forcing
* Predicts Roman Urdu characters
* Optional attention-like context (mean of encoder outputs)

### **🔹 Seq2Seq Wrapper**

* Connects encoder and decoder
* Handles training loop token-by-token

---

# 🏋️ Training Pipeline

* Train/Val/Test split: **50% / 25% / 25%**
* Optimizer: **Adam**
* Loss: **CrossEntropyLoss**
* Batch sizes: 32 / 64 / 128
* Learning rates tested: 1e-3, 5e-4, 1e-4
* Teacher forcing ratio: 0.5

Training file is located at: `training/train.py`

---

# 📊 Evaluation Metrics

The following metrics were implemented:

### ✔ BLEU Score (main NMT metric)

### ✔ Perplexity (model confidence)

### ✔ CER – Character Error Rate

### ✔ Levenshtein Distance

Sample evaluation implemented in: `evaluation/evaluate.py`

---

# 🌐 Deployment (Streamlit App)

A simple **Streamlit UI** is included to test the model:

* User enters Urdu text
* Model outputs Roman Urdu
* Deployed using `streamlit run deployment/app.py`


---

# 📁 Project Structure

```
📦 Urdu-to-Roman-Urdu-NMT
├── data/
│   └── (dataset files go here)
├── preprocessing/
│   └── text_cleaning.py
├── models/
│   ├── encoder.py
│   ├── decoder.py
│   └── seq2seq.py
├── training/
│   └── train.py
├── evaluation/
│   └── evaluate.py
├── deployment/
│   └── app.py
├── notebooks/
│   └── nlp-a1-22f-8781-22f-3606.ipynb
├── requirements.txt
└── README.md
```

---


**Haider Abbas**
FAST NUCES — NLP Course Assignment
