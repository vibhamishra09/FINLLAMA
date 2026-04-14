import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import LabelBinarizer
from pathlib import Path

# --- Configuration ---
PRED_PATH = Path("/content/drive/MyDrive/FinLLaMA_predictions.csv")
TRUE_PATH = Path("/content/drive/MyDrive/labeled_shards")  # use labeled shards
OUTPUT_DIR = Path("/content/drive/MyDrive/figures")

LABELS = ["Negative", "Neutral", "Positive"]

def main():
    print("="*60)
    print("GENERATING ROC CURVE AND AUC SCORES")
    print("="*60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ==============================
    # LOAD DATA
    # ==============================
    df_pred = pd.read_csv(PRED_PATH)

    # Load labeled shards (ground truth)
    import glob
    files = glob.glob(str(TRUE_PATH / "*.parquet"))
    df_true = pd.concat([pd.read_parquet(f) for f in files[:10]])

    # Align size
    df_true = df_true.iloc[:len(df_pred)]

    # ==============================
    # PREPARE LABELS
    # ==============================
    y_true = df_true["sentiment"]
    y_pred = df_pred["predicted_sentiment"]

    # Convert numeric → text if needed
    if y_true.dtype != "object":
        id2label = {0: "Negative", 1: "Neutral", 2: "Positive"}
        y_true = y_true.map(id2label)

    if y_pred.dtype != "object":
        id2label = {0: "Negative", 1: "Neutral", 2: "Positive"}
        y_pred = y_pred.map(id2label)

    # ==============================
    # BINARIZATION
    # ==============================
    lb = LabelBinarizer()
    y_true_bin = lb.fit_transform(y_true)
    y_pred_bin = lb.transform(y_pred)

    # ==============================
    # AUC SCORE
    # ==============================
    macro_auc = roc_auc_score(y_true_bin, y_pred_bin, average="macro")
    print(f"\nOverall Macro AUC Score: {macro_auc:.4f}")

    # ==============================
    # PLOT ROC
    # ==============================
    plt.figure(figsize=(10, 8))
    sns.set_theme(style="whitegrid")

    colors = ['#e74c3c', '#95a5a6', '#2ecc71']

    for i, color in zip(range(len(LABELS)), colors):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_bin[:, i])
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, color=color, lw=2,
                 label=f"{LABELS[i]} (AUC = {roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], 'k--', lw=2)

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - FinLLaMA')
    plt.legend(loc="lower right")

    # Save
    path = OUTPUT_DIR / "roc_curve.png"
    plt.savefig(path, dpi=300)
    print(f"\n✅ Saved at: {path}")

    plt.show()

if __name__ == "__main__":
    main()