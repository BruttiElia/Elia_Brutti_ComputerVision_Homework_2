"""Test-set metrics for all approaches, computed from the saved predictions."""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                  # save figures to file without opening windows
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score
from torchvision import datasets

from utils import save_json, save_csv

RESULTS_DIR = Path("results")
APPROACHES = ["full", "head", "features"]


def load_predictions(path):
    """Read predictions.csv; return true and predicted labels as arrays."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    y_true = np.array([int(row["true"]) for row in rows])          
    y_pred = np.array([int(row["pred"]) for row in rows])        
    return y_true, y_pred


def compute_metrics(y_true, y_pred):
    """Accuracy and macro-averaged F1 on the test set."""
    return {
        "test_accuracy": float(accuracy_score(y_true, y_pred)),
        "test_macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }


def get_class_names():
    """Breed names in label order (0-36), read from the dataset."""
    return datasets.OxfordIIITPet(root="data", split="test").classes


def save_confusion_matrix(y_true, y_pred, class_names, path, title):
    """Row-normalized confusion matrix with breed names."""
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    fig, ax = plt.subplots(figsize=(14, 14))
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(
        ax=ax, cmap="Blues", include_values=False, xticks_rotation=90, colorbar=False
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

def top_confused_pairs(y_true, y_pred, class_names, k=10):
    """The k (true, predicted) breed pairs with the most test errors."""
    cm = confusion_matrix(y_true, y_pred)       # counts: rows = true breed, columns = predicted
    np.fill_diagonal(cm, 0)                     # ignore correct predictions

    pairs = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if cm[i, j] > 0:
                pairs.append({
                    "true": class_names[i],
                    "predicted": class_names[j],
                    "count": int(cm[i, j]),
                    "share_of_true_class": round(float(cm[i, j] / (y_true == i).sum()), 3),
                })

    pairs.sort(key=lambda p: p["count"], reverse=True)
    return pairs[:k]

def save_examples(y_true, y_pred, class_names, path, n=6, seed=42):
    """Grid of test images: n correct predictions (top row) and n errors (bottom row)."""
    test_images = datasets.OxfordIIITPet(root="data", split="test")   # original images, no transform
    rng = np.random.default_rng(seed)                                  # fixed seed: same examples every run

    correct_idx = np.where(y_true == y_pred)[0]         
    wrong_idx = np.where(y_pred != y_true)[0]            
    rows = [rng.choice(correct_idx, n, replace=False), rng.choice(wrong_idx, n, replace=False)]

    fig, axes = plt.subplots(2, n, figsize=(3 * n, 7))
    for r, indices in enumerate(rows):
        for c, idx in enumerate(indices):
            image, _ = test_images[idx]
            ax = axes[r, c]
            ax.imshow(image)
            ok = y_true[idx] == y_pred[idx]
            ax.set_title(f"True: {class_names[y_true[idx]]}\nPred: {class_names[y_pred[idx]]}",
                         fontsize=9, color="green" if ok else "red")
            ax.axis("off")
    fig.suptitle("Approach 3 (features + logistic regression): correct (top) and wrong (bottom) predictions")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

if __name__ == "__main__":
    class_names = get_class_names()           # once, before the loop: same names for all approaches

    for approach in APPROACHES:
        y_true, y_pred = load_predictions(RESULTS_DIR / approach / "predictions.csv")
        metrics = compute_metrics(y_true, y_pred)
        save_json(metrics, RESULTS_DIR / approach / "metrics.json")
        print(f"{approach:<10} accuracy={metrics['test_accuracy']:.4f}  "
              f"macro_f1={metrics['test_macro_f1']:.4f}")

        save_confusion_matrix(
            y_true, y_pred, class_names,
            RESULTS_DIR / approach / "confusion_matrix.png",
            f"Confusion matrix - {approach} (test set, row-normalized)",
        )

        pairs = top_confused_pairs(y_true, y_pred, class_names)
        save_csv(pairs, RESULTS_DIR / approach / "confused_pairs.csv")

    # Example predictions for the recommended approach only
    y_true, y_pred = load_predictions(RESULTS_DIR / "features" / "predictions.csv")
    save_examples(y_true, y_pred, class_names, RESULTS_DIR / "features" / "examples.png")
        

        