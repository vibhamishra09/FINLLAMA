# FinLLaMA: Financial Sentiment Analysis with LoRA-Fine-Tuned LLM

## 📌 Overview

**FinLLaMA** is a reproducible research pipeline for financial sentiment analysis using a **LoRA-fine-tuned Large Language Model (LLM)**.

The project implements a **teacher-student framework**, where:

* A pretrained financial model (FinBERT) generates **pseudo-labels**
* A lightweight LLM (LLaMA) is fine-tuned using **parameter-efficient techniques (LoRA)**

The pipeline supports:

* Gated dataset ingestion
* Automated labeling
* Model fine-tuning
* Inference & evaluation
* Portfolio backtesting

---

## ⚙️ Key Features

* 🔹 End-to-end reproducible pipeline
* 🔹 Teacher-student training (FinBERT → FinLLaMA)
* 🔹 Parameter-efficient fine-tuning (LoRA + quantization)
* 🔹 Financial sentiment classification
* 🔹 Backtesting trading strategies based on sentiment
* 🔹 Modular and extensible codebase

---

## 🧠 Architecture

```
Raw Financial News Data
        ↓
Data Ingestion & Cleaning
        ↓
FinBERT (Teacher Model)
        ↓
Pseudo-Labeled Dataset
        ↓
LoRA Fine-Tuning (LLaMA)
        ↓
FinLLaMA Model
        ↓
Evaluation + Portfolio Backtesting
```

---

## 📁 Repository Structure

### Core Files

* `README.md` — Project documentation
* `requirements.txt` — Dependencies
* `CODEBASE_INDEX.md` — Detailed codebase breakdown

---

### 📊 Data

* `data/financial_news_shards/`
  Raw unlabeled financial news (Parquet shards)

* `data/financial_news_shards_labeled/`
  FinBERT-labeled data with sentiment + confidence

* `data/llm_training_data/`
  Final train/val/test splits

---

### 🤖 Models

* `models/Llama-3.2-1B/`
  Base LLaMA model

* `models/finllama_lora/`
  Fine-tuned LoRA adapters

* `models/finbert-tone/`
  FinBERT teacher model

---

### 📈 Results

* `results/predictions/` — Model outputs
* `results/figures/` — Evaluation & backtesting plots

---

## 🧩 Source Modules

### `src/data_prep/`

Handles dataset ingestion and labeling

* `financial_multisource_loader.py`
  → Loads and preprocesses financial news

* `label_finbert_on_parquet_shards.py`
  → Generates sentiment labels using FinBERT

* `prepare_llm_data.py`
  → Creates training/validation/test datasets

---

### `src/modeling/`

Model training and inference

* `train_model.py`
  → Fine-tunes LLaMA using LoRA

* `run_finllama_inference.py`
  → Inference with fine-tuned model

* `run_base_llama_inference.py`
  → Baseline model inference

---

### `src/evaluation/`

Performance evaluation

* `evaluate_all.py`
  → Accuracy, F1, precision, recall, confusion matrix

* `plot_roc.py`
  → ROC curves and AUC analysis

---

### `src/portfolio/`

Trading strategy simulation

* `backtest_portfolio.py`
  → Converts sentiment → trading signals
  → Simulates returns and evaluates performance

---

### `src/setup/`

Environment setup utilities

* `setup_huggingface_auth.py`
  → Handles Hugging Face authentication for gated datasets

---

## 🚀 Installation

```bash
git clone https://github.com/Manasvee16/FINLLAMA.git
cd FINLLAMA

pip install -r requirements.txt
```

---

## 🔐 Setup (Hugging Face Access)

```bash
python src/setup/setup_huggingface_auth.py
```

You will need:

* A Hugging Face account
* Access token for gated datasets

---

## ▶️ Execution Pipeline

Run the pipeline in order:

```bash
# 1. Data ingestion
python src/data_prep/financial_multisource_loader.py

# 2. Label generation (FinBERT)
python src/data_prep/label_finbert_on_parquet_shards.py

# 3. Dataset preparation
python src/data_prep/prepare_llm_data.py

# 4. Model training (FinLLaMA)
python src/modeling/train_model.py

# 5. Inference
python src/modeling/run_finllama_inference.py
python src/modeling/run_base_llama_inference.py

# 6. Evaluation
python src/evaluation/evaluate_all.py
python src/evaluation/plot_roc.py

# 7. Portfolio backtesting
python src/portfolio/backtest_portfolio.py
```

---

## 📊 Evaluation Metrics

* Accuracy
* Weighted F1 Score
* Precision / Recall
* ROC-AUC
* Confusion Matrix

---

## 💡 Research Insights

* Uses **teacher-student learning** for scalable labeling
* Achieves efficiency via **LoRA adapters**
* Reduces compute cost with **quantization**
* Bridges NLP → Finance via **strategy backtesting**

---

## 📌 Future Work

* Multi-modal financial signals (news + price + social media)
* Real-time inference pipeline
* Reinforcement learning for trading strategies
* Larger LLM variants (e.g., 7B+)

---

## 📚 References

* LLaMA-3.2 Model — https://huggingface.co/meta-llama/Llama-3.2-1B
* LoRA (PEFT) — https://huggingface.co/docs/peft/methods/lora
* FinBERT — https://huggingface.co/yiyanghkust/finbert-tone
* Financial News Dataset — https://huggingface.co/datasets/Brianferrell787/financial-news-multisource

---
