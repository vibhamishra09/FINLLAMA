import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import LabelBinarizer
from pathlib import Path

# --- Configuration ---
FINLLAMA_PREDS_PATH = Path("../../results/predictions/FinLLaMA_predictions.csv")
OUTPUT_DIR = Path("../../results/figures")
LABELS = ["Negative", "Neutral", "Positive"]

def main():
    print("="*60)
    print("GENERATING ROC CURVE AND AUC SCORES")
    print("="*60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    if not FINLLAMA_PREDS_PATH.exists():
        print(f"Error: Could not find {FINLLAMA_PREDS_PATH.resolve()}")
        return

    print("Loading FinLLaMA predictions...")
    df = pd.read_csv(FINLLAMA_PREDS_PATH)
    y_true = df["sentiment"]
    y_pred = df["predicted_sentiment"]

    # Binarize the labels for One-vs-Rest (OvR) evaluation
    lb = LabelBinarizer()
    y_true_bin = lb.fit_transform(y_true)
    y_pred_bin = lb.transform(y_pred)

    # Calculate overall Macro AUC Score
    macro_roc_auc_ovr = roc_auc_score(y_true_bin, y_pred_bin, multi_class="ovr", average="macro")
    print(f"\nOverall Macro AUC Score: {macro_roc_auc_ovr:.4f}")

    # Set up the plot
    plt.figure(figsize=(10, 8))
    sns.set_theme(style="whitegrid")
    
    colors = ['#e74c3c', '#95a5a6', '#2ecc71'] # Red, Grey, Green

    # Plot an ROC curve for each of the 3 classes
    for i, color in zip(range(len(LABELS)), colors):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_bin[:, i])
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, color=color, lw=2,
                 label=f"ROC curve of class {LABELS[i]} (AUC = {roc_auc:.2f})")
        
        print(f" - {LABELS[i]} AUC: {roc_auc:.4f}")

    # Plot the random guessing baseline
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label="Random Guessing (AUC = 0.50)")

    # Formatting
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) - FinLLaMA One-vs-Rest', pad=20, fontsize=15, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)

    # Save and display
    plot_path = OUTPUT_DIR / "finllama_roc_curve.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    print(f"\n✅ ROC chart saved to: {plot_path.resolve()}")
    
    plt.show()

if __name__ == "__main__":
    main()