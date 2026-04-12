import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --- Configuration ---
BASE_MODEL_PATH = str(Path("./Llama-3.2-1B").resolve())
TEST_DATA_PATH = Path("./llm_training_data/test_llm.parquet")
OUTPUT_CSV_PATH = Path("./Base_Llama_predictions.csv")

def main():
    print("="*60)
    print("🧊 INITIATING BASE LLAMA (UNTRAINED) INFERENCE")
    print("="*60)

    test_df = pd.read_parquet(TEST_DATA_PATH)
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading Base Llama Model (Without LoRA)...")
    id2label = {0: "Negative", 1: "Neutral", 2: "Positive"}
    
    # We load it directly, no PeftModel wrapper
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_PATH,
        num_labels=3,
        id2label=id2label,
        device_map="auto",
        torch_dtype=torch.float16,
        local_files_only=True
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()

    predictions, confidences = [], []

    print(f"\nRunning predictions on {len(test_df)} articles...")
    for text in tqdm(test_df["text"], desc="Predicting (Random Head)"):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(model.device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            pred_class = torch.argmax(probs, dim=-1).item()
            conf = probs[0][pred_class].item()
            
            predictions.append(pred_class)
            confidences.append(conf)

    results_df = test_df.copy()
    results_df["predicted_sentiment"] = predictions
    results_df["prediction_confidence"] = confidences
    
    results_df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"\n✅ Success! Baseline predictions saved to: {OUTPUT_CSV_PATH.name}")

if __name__ == "__main__":
    main()