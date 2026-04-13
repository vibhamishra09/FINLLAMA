import torch
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

# --- Configuration ---
BASE_MODEL_PATH = str(Path("meta-llama/Llama-3.2-1B").resolve())
ADAPTER_PATH = str(Path("/content/drive/MyDrive/finllama_model").resolve())
TEST_DATA_PATH = Path("/content/drive/MyDrive/llm_training_data")
OUTPUT_CSV_PATH = Path("/content/drive/MyDrive/FinLLaMA_predictions.csv")

def main():
    print("="*60)
    print("🧠 INITIATING FINLLAMA INFERENCE PIPELINE")
    print("="*60)

    # 1. Load the Test Data
    print(f"Loading unseen test data from {TEST_DATA_PATH.name}...")
    test_df = pd.read_parquet(TEST_DATA_PATH)
    
    # 2. Load Tokenizer
    print("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, local_files_only=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3. Load Base Model (in standard 16-bit to fit in 4GB for inference)
    print("Loading Base Llama Model...")
    id2label = {0: "Negative", 1: "Neutral", 2: "Positive"}
    
    base_model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_PATH,
        num_labels=3,
        id2label=id2label,
        device_map="auto",
        torch_dtype=torch.float16,
        local_files_only=False
    )
    base_model.config.pad_token_id = tokenizer.pad_token_id

    # 4. Attach the LoRA Adapters
    print("Attaching your custom FinLLaMA LoRA Adapters...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval() # Set to evaluation mode (no training)

    # 5. Run Inference
    print(f"\nRunning predictions on {len(test_df)} articles...")
    predictions = []
    confidences = []

    # Using tqdm for a nice progress bar
    for text in tqdm(test_df["text"], desc="Predicting Sentiment"):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(model.device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            
            # Convert logits to probabilities using Softmax
            probs = torch.nn.functional.softmax(logits, dim=-1)
            predicted_class_id = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][predicted_class_id].item()
            
            predictions.append(predicted_class_id)
            confidences.append(confidence)

    # 6. Save Results
    print("\nSaving predictions to CSV...")
    results_df = test_df.copy()
    results_df["predicted_sentiment"] = predictions
    results_df["prediction_confidence"] = confidences
    
    results_df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"✅ Success! Predictions saved to: {OUTPUT_CSV_PATH.resolve()}")

if __name__ == "__main__":
    main()