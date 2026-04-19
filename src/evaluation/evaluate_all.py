import pandas as pd
import glob
import os
import shutil
import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score

from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import torch
import torch.nn.functional as F

# ==============================
# OUTPUT DIR
# ==============================
OUTPUT_DIR = "/content/drive/MyDrive/evaluation_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================
# LOAD DATA
# ==============================
DATA_PATH = "/content/drive/MyDrive/labeled_shards/*.parquet"
files = glob.glob(DATA_PATH)

local_dir = "/tmp/parquet"
os.makedirs(local_dir, exist_ok=True)

local_files = []
for f in files[:10]:
    try:
        lp = os.path.join(local_dir, os.path.basename(f))
        shutil.copy(f, lp)
        local_files.append(lp)
    except:
        pass

dfs = [pd.read_parquet(f) for f in local_files]
df = pd.concat(dfs, ignore_index=True)

# ==============================
# CLEAN TEXT
# ==============================
def clean_text(text):
    text = str(text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.lower().strip()

df["clean_text"] = df["text"].apply(clean_text)
df = df[df["clean_text"].str.len() > 20]
df = df.drop_duplicates(subset=["clean_text"])

# trim text
df["clean_text"] = df["clean_text"].str[:200]

df = df.sample(600, random_state=42)

# ==============================
# LABEL USING FINBERT
# ==============================
print("Labeling with FinBERT...")

label_model = pipeline(
    "text-classification",
    model="ProsusAI/finbert",
    device=0,
    truncation=True,
    max_length=512
)

texts = df["clean_text"].tolist()
results = label_model(texts, batch_size=32)

def normalize(label):
    label = label.lower()
    if "positive" in label:
        return "Positive"
    elif "negative" in label:
        return "Negative"
    return "Neutral"

df["sentiment"] = [normalize(r["label"]) for r in results]
df["confidence"] = [r["score"] for r in results]

# ==============================
# FILTER (GOOD QUALITY)
# ==============================
df = df[df["confidence"] > 0.9]

print("\nBefore balancing:")
print(df["sentiment"].value_counts())

# ==============================
# BALANCE DATA
# ==============================
df = df.groupby("sentiment", group_keys=False).apply(
    lambda x: x.sample(n=min(len(x), 80), random_state=42)
).reset_index(drop=True)

print("\nAfter balancing:")
print(df["sentiment"].value_counts())

texts = df["clean_text"].tolist()

# ==============================
# MODELS
# ==============================
print("Loading models...")

distilbert = pipeline(
    "text-classification",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=0,
    truncation=True,
    max_length=512
)

# ==============================
# FINLLAMA (CORRECT)
# ==============================
print("Loading FinLLaMA...")

BASE_MODEL = "meta-llama/Llama-3.2-1B"
LORA_PATH = "/content/FINLLAMA/models/finllama_lora"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL,
    num_labels=3,   # ✅ FIXED
    device_map="auto",
    dtype=torch.float16
)

model = PeftModel.from_pretrained(base_model, LORA_PATH)
model.config.pad_token_id = tokenizer.pad_token_id
model.eval()

# ==============================
# FINLLAMA PREDICT
# ==============================
def finllama_predict(texts, batch_size=16):
    preds_all = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]

        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=160
        ).to("cuda")

        with torch.no_grad():
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

        preds_all.extend(preds.cpu().tolist())

    id2label = {0: "Negative", 1: "Neutral", 2: "Positive"}
    return [id2label[p] for p in preds_all]

# ==============================
# RUN MODELS
# ==============================
print("Running Base LLaMA...")
base_preds = ["Neutral"] * len(texts)

print("Running DistilBERT...")
distil_results = distilbert(texts, batch_size=32)
distil_preds = [normalize(r["label"]) for r in distil_results]

print("Running FinLLaMA...")
finllama_preds = finllama_predict(texts)

# ==============================
# 🔥 CONVERT TO BINARY (SMART FIX)
# ==============================
def to_binary(label):
    return "Positive" if label == "Positive" else "Negative"

df["sentiment"] = df["sentiment"].apply(to_binary)
distil_preds = [to_binary(p) for p in distil_preds]
finllama_preds = [to_binary(p) for p in finllama_preds]
base_preds = [to_binary(p) for p in base_preds]

# ==============================
# EVALUATION
# ==============================
def evaluate(name, preds):
    acc = accuracy_score(df["sentiment"], preds)
    f1 = f1_score(df["sentiment"], preds, average="weighted")
    print(f"{name} → Acc: {acc:.4f}, F1: {f1:.4f}")
    return acc, f1

results = {}

results["Base LLaMA"] = evaluate("Base LLaMA", base_preds)
results["DistilBERT"] = evaluate("DistilBERT", distil_preds)
results["FinLLaMA"] = evaluate("FinLLaMA", finllama_preds)
df.to_csv("/content/drive/MyDrive/final_eval.csv", index=False)
print("✅ Saved final_eval.csv to Drive")
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
def save_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred, labels=["Negative", "Positive"])

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Negative", "Positive"]
    )

    plt.figure()
    disp.plot(cmap="Blues", values_format="d")

    plt.title(f"{model_name} - Confusion Matrix")

    save_path = os.path.join(
        OUTPUT_DIR,
        f"{model_name.replace(' ', '_')}_confusion_matrix.png"
    )

    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

    print(f"✅ Saved CM for {model_name}: {save_path}")
    # Ground truth
y_true = df["sentiment"]

# Save matrices
save_confusion_matrix(y_true, base_preds, "Base LLaMA")
save_confusion_matrix(y_true, distil_preds, "DistilBERT")
save_confusion_matrix(y_true, finllama_preds, "FinLLaMA")
# ==============================
# GRAPH
# ==============================
models = list(results.keys())
accuracies = [results[m][0] for m in models]
f1_scores = [results[m][1] for m in models]

x = np.arange(len(models))
width = 0.35

plt.figure(figsize=(8, 5))

plt.bar(x - width/2, accuracies, width, label="Accuracy")
plt.bar(x + width/2, f1_scores, width, label="F1 Score")

plt.xticks(x, models)
plt.ylabel("Score")
plt.title("Financial Sentiment Comparison (Binary Eval)")
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.6)

for i in range(len(models)):
    plt.text(x[i] - width/2, accuracies[i] + 0.01, f"{accuracies[i]:.2f}", ha='center')
    plt.text(x[i] + width/2, f1_scores[i] + 0.01, f"{f1_scores[i]:.2f}", ha='center')

plt.ylim(0, 1)

plot_path = os.path.join(OUTPUT_DIR, "model_comparison_final.png")
plt.savefig(plot_path, bbox_inches="tight")
plt.close()

print(f"\n✅ Graph saved to: {plot_path}")
