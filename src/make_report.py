"""Summary table and figures for the report, built from the saved results."""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # save figures to file without opening windows
import matplotlib.pyplot as plt

from utils import save_csv

RESULTS_DIR = Path("results")
FIGURES_DIR = RESULTS_DIR / "figures" # figures used in the report
LABELS = {"full": "1. Full fine-tuning", "head": "2. Classifier head", "features": "3. Features + LogReg"}  # names in tables and titles
LOGREG_PARAMS = 576 * 37 + 37      # weights + biases of the scikit-learn classifier (approach 3)


def load_json(path):
    """Read a JSON file (config.json, metrics.json)."""
    with open(path) as f:
        return json.load(f)


def load_csv(path):
    """Read a CSV file as a list of dicts, one per row."""
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def build_summary():
    """One row per approach with performance and computational cost."""
    rows = []
    for approach in ["full", "head", "features"]:
        config = load_json(RESULTS_DIR / approach / "config.json")
        metrics = load_json(RESULTS_DIR / approach / "metrics.json")

        if approach == "features":
            trainable = f"0 (network) + {LOGREG_PARAMS} (classifier)"
            time_s = config["extraction_time_s"] + config["classifier_time_s"]  # extraction + classifier training (incl. search over C)
        else:
            trainable = config["trainable_params"]
            time_s = config["train_time_s"]

        rows.append({
            "approach": LABELS[approach],
            "trainable_params": trainable,
            "time_s": round(time_s),
            "val_acc": round(config["best_val_acc"], 4),
            "train_acc": round(config["train_acc_eval"], 4),
            "test_acc": round(metrics["test_accuracy"], 4),
            "test_macro_f1": round(metrics["test_macro_f1"], 4),
            "train_test_gap": round(config["train_acc_eval"] - metrics["test_accuracy"], 4),  # > 0: better on train than on test
        })
    return rows

def plot_training_curves():
    """Train and validation accuracy per epoch for approaches 1 and 2."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, approach in zip(axes, ["full", "head"]):
        history = load_csv(RESULTS_DIR / approach / "history.csv")
        best_epoch = load_json(RESULTS_DIR / approach / "config.json")["best_epoch"] 
        epochs = [int(r["epoch"]) for r in history]

        ax.plot(epochs, [float(r["train_acc"]) for r in history], label="train (augmented, during epoch)")
        ax.plot(epochs, [float(r["val_acc"]) for r in history], label="validation")
        ax.axvline(best_epoch, linestyle="--", color="gray", label=f"selected epoch ({best_epoch})")  # checkpoint reloaded at the end
        ax.set_title(LABELS[approach])
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend()
    axes[0].set_ylabel("accuracy")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "training_curves.png", dpi=150)
    plt.close(fig)


def plot_c_curve():
    """Validation accuracy of the logistic regression for each value of C (approach 3)."""
    rows = load_csv(RESULTS_DIR / "features" / "val_scores.csv")
    best_c = load_json(RESULTS_DIR / "features" / "config.json")["best_c"]
    c_values = [float(r["C"]) for r in rows]
    val_accs = [float(r["val_acc"]) for r in rows]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogx(c_values, val_accs, marker="o")      # log scale: C goes from 1e-4 to 10
    ax.axvline(best_c, linestyle="--", color="gray", label=f"selected C = {best_c}")
    ax.set_xlabel("C (log scale)")
    ax.set_ylabel("validation accuracy")
    ax.set_title("3. Features + LogReg: choice of C")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "features_c_curve.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    summary = build_summary()
    save_csv(summary, RESULTS_DIR / "summary.csv") # results table of the report
    for row in summary:
        print(row)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_training_curves()
    plot_c_curve()