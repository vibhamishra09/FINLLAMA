import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# --- Configuration & Paths ---
MODEL_NAME = "meta-llama/Llama-3.2-1B"
DATA_DIR = Path("/content/drive/MyDrive/llm_training_data")
OUTPUT_DIR = Path("./model_output/finllama")

# Extreme VRAM Optimizations for GTX 1650
MAX_LENGTH = 128      # Hard cap to save memory
BATCH_SIZE = 2        # Micro-batching
GRAD_ACCUM_STEPS = 8  # Simulates a batch size of 16 (2 x 8)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted", zero_division=0
    )
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

def main():
    print("="*60)
    print("🚀 INITIATING FINLLAMA QLORA TRAINING PIPELINE")
    print("="*60)

    # 1. Load the dense 15k dataset
    print("Loading LLM datasets...")
    dataset = load_dataset(
        "parquet",
        data_files={
            "train": str(DATA_DIR / "train_llm.parquet"),
            "validation": str(DATA_DIR / "val_llm.parquet"),
            "test": str(DATA_DIR / "test_llm.parquet"),
        }
    )

    # 2. Load Tokenizer
    print("Loading Llama Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Llama models don't have a default padding token, so we assign one
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(example):
        tokenized = tokenizer(
            example["text"],
            padding="max_length",
            truncation=True,
            max_length=512,
        )
    
    # ALWAYS match batch size of input_ids
    batch_size = len(tokenized["input_ids"])
    tokenized["labels"] = [0] * batch_size
    
    return tokenized

    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text", "date"])

    # 3. Configure 4-Bit Quantization (The Black Magic)
    print("Configuring 4-bit Quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    # 4. Load the Base Llama Model in 4-bit
    print("Loading Base Llama Model (This will take a moment)...")
    id2label = {0: "Negative", 1: "Neutral", 2: "Positive"}
    label2id = {"Negative": 0, "Neutral": 1, "Positive": 2}

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
        id2label=id2label,
        label2id=label2id,
        quantization_config=bnb_config,
        device_map="auto",  # Automatically places the model on your GPU
        local_files_only=False,
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    # 5. Apply LoRA Adapters
    print("Injecting LoRA Adapters...")
    model = prepare_model_for_kbit_training(model)
    
    lora_config = LoraConfig(
        r=16, 
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"], # Target the attention brain
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_CLS" # Sequence Classification
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters() # This will prove we are only training ~1% of the model!

    # 6. Training Arguments
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-4,          # LoRA needs a slightly higher learning rate
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        num_train_epochs=2,          # 2 epochs is plenty for a 1B model on 12k rows
        weight_decay=0.01,
        fp16=True,                   # 16-bit math for training stability
        logging_steps=20,
        optim="paged_adamw_8bit",    # 8-bit optimizer to save even more VRAM
        report_to="none"
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 7. Train!
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("\n🔥 STARTING FINLLAMA TRAINING 🔥")
    trainer.train()

    # 8. Evaluate and Save
    print("\nEvaluating on Test Set...")
    test_results = trainer.evaluate(eval_dataset=tokenized_datasets["test"])
    print("\n--- FINAL TEST RESULTS ---")
    for key, value in test_results.items():
        if key.startswith("eval_"):
            print(f"{key.replace('eval_', '')}: {value:.4f}")

    print(f"\nSaving FinLLaMA LoRA adapters to {OUTPUT_DIR}/final_model")
    trainer.save_model(str(OUTPUT_DIR / "final_model"))
    print("Done! You have officially built FinLLaMA.")

if __name__ == "__main__":
    main()