import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from pathlib import Path
import numpy as np

# --- Configuration ---
PREDICTION_FILES = {
    "Base Llama-3.2-1B (Untrained)": Path("../../results/predictions/Base_Llama_predictions.csv"),
    "FinLLaMA (LoRA Tuned)": Path("../../results/predictions/FinLLaMA_predictions.csv")
}
OUTPUT_DIR = Path("../../results/figures")
LABELS = ["Negative", "Neutral", "Positive"]

def main():
    print("="*60)
    print("MODEL PERFORMANCE EVALUATION (DEEP DIVE)")
    print("="*60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    # 1. Process model predictions and print detailed reports
    for model_name, file_path in PREDICTION_FILES.items():
        if not file_path.exists():
            print(f"Warning: Skipping {model_name} - Could not find {file_path.resolve()}")
            continue
            
        df = pd.read_csv(file_path)
        if "sentiment" in df.columns:
            y_true = df["sentiment"]
        else:
            y_true = df.iloc[:, 0] # fallback

        y_pred = df["predicted_sentiment"]
        
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="weighted")
        
        results.append({
            "Model": model_name,
            "Accuracy": acc,
            "F1-Score": f1
        })

        # --- NEW: Print detailed Precision, Recall, and F1 for each class ---
        print(f"\n[{model_name}] Detailed Classification Report:")
        print(classification_report(y_true, y_pred, target_names=LABELS, zero_division=0))

        # --- NEW: Generate Confusion Matrix specifically for FinLLaMA ---
        if "FinLLaMA" in model_name:
            print(f"Generating Confusion Matrix for {model_name}...")
            cm = confusion_matrix(y_true, y_pred)
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=LABELS, yticklabels=LABELS)
            plt.title('FinLLaMA (LoRA Tuned) - Confusion Matrix', pad=20, fontsize=14, fontweight='bold')
            plt.ylabel('True Sentiment (FinBERT Teacher)')
            plt.xlabel('Predicted Sentiment (FinLLaMA Student)')
            
            cm_path = OUTPUT_DIR / "finllama_confusion_matrix.png"
            plt.tight_layout()
            plt.savefig(cm_path, dpi=300)
            print(f"Confusion Matrix saved to: {cm_path.resolve()}")

    if not results: return

    # 2. Format overall metrics output
    results_df = pd.DataFrame(results)
    print("\n" + "="*60)
    print("OVERALL METRICS SUMMARY")
    print("="*60)
    print(results_df.to_string(index=False))

    # 3. Generate performance bar chart (Same as before)
    print("\nGenerating comparison bar chart...")
    plt.figure(figsize=(9, 6))
    sns.set_theme(style="whitegrid")
    melted_df = results_df.melt(id_vars="Model", var_name="Metric", value_name="Score")
    ax = sns.barplot(data=melted_df, x="Model", y="Score", hue="Metric", palette=["#e74c3c", "#2ecc71"])
    plt.title("Financial Sentiment Analysis: Base LLM vs. LoRA Fine-Tuned", pad=20, fontsize=14, fontweight='bold')
    plt.ylim(0.0, 1.0)
    plt.ylabel("Score (0.0 to 1.0)", fontsize=12)
    plt.xlabel("")
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=3)

    plot_path = OUTPUT_DIR / "lora_impact_metrics.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    
    plt.show()

if __name__ == "__main__":
    main()