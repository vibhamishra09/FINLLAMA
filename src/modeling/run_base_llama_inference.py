import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig

# ============================================================
# CONFIGURATION
# ============================================================
BASE_MODEL_PATH = "meta-llama/Llama-3.2-1B"
TEST_DATA_PATH = Path("/content/drive/MyDrive/llm_training_data")
OUTPUT_CSV_PATH = Path("/content/drive/MyDrive/Base_Llama_predictions.csv")


def load_model_and_tokenizer():
    print("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading Model Config...")
    config = AutoConfig.from_pretrained(BASE_MODEL_PATH)

    # 🔥 Fix LLaMA rope issue
    if hasattr(config, "rope_scaling"):
        config.rope_scaling = None

    config.num_labels = 3
    config.problem_type = "single_label_classification"

    print("Loading Base Llama Model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_PATH,
        config=config,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()

    return model, tokenizer


def run_inference(model, tokenizer, df):
    predictions = []
    confidences = []

    print(f"\nRunning predictions on {len(df)} samples...\n")

    for text in tqdm(df["text"], desc="Predicting (Baseline)"):
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128
        ).to(model.device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

            probs = torch.nn.functional.softmax(logits, dim=-1)
            pred = torch.argmax(probs, dim=-1).item()
            conf = probs[0][pred].item()

        predictions.append(pred)
        confidences.append(conf)

    return predictions, confidences


def main():
    print("=" * 60)
    print("🧊 INITIATING BASE LLAMA (UNTRAINED) INFERENCE")
    print("=" * 60)

    # ========================================================
    # LOAD DATA
    # ========================================================
    print(f"Loading test data from: {TEST_DATA_PATH}")
    test_df = pd.read_parquet(TEST_DATA_PATH)

    # ========================================================
    # LOAD MODEL + TOKENIZER
    # ========================================================
    model, tokenizer = load_model_and_tokenizer()

    # ========================================================
    # RUN INFERENCE
    # ========================================================
    predictions, confidences = run_inference(model, tokenizer, test_df)

    # ========================================================
    # SAVE RESULTS
    # ========================================================
    print("\nSaving results...")

    results_df = test_df.copy()
    results_df["predicted_sentiment"] = predictions
    results_df["confidence"] = confidences

    results_df.to_csv(OUTPUT_CSV_PATH, index=False)

    print(f"✅ Done! Results saved at:\n{OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    main()