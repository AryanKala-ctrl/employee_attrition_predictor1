"""Evaluate models: accuracy, precision, recall, F1, confusion matrix."""

from pathlib import Path

import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)

from src.data_loader import OUTPUT_DIR


def evaluate_model(model, X_test, y_test, X_train=None, y_train=None):
    """Compute metrics for one trained model. 'Yes' (1) is the positive class."""
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    result = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "true_negative": int(cm[0, 0]),
        "false_positive": int(cm[0, 1]),
        "false_negative": int(cm[1, 0]),
        "true_positive": int(cm[1, 1]),
    }
    if X_train is not None and y_train is not None:
        result["train_accuracy"] = accuracy_score(y_train, model.predict(X_train))
    return result


def compare_models(models, X_test, y_test, X_train=None, y_train=None):
    """Evaluate every model and return a comparison table (one row per model)."""
    rows = {name: evaluate_model(m, X_test, y_test, X_train, y_train) for name, m in models.items()}
    return pd.DataFrame(rows).T.rename_axis("model").reset_index()


def print_comparison(results):
    """Print the comparison in a readable way."""
    show = results.copy()
    metric_cols = [c for c in ["train_accuracy", "accuracy", "precision", "recall", "f1_score"]
                   if c in show.columns]
    for col in metric_cols:
        show[col] = show[col].astype(float).round(3)
    print("\nModel comparison (test set; positive class = Attrition 'Yes')")
    print(show[["model"] + metric_cols].to_string(index=False))
    print("\nConfusion matrix counts (rows = actual, columns = predicted)")
    for _, row in results.iterrows():
        print(f"\n  {row['model']}:")
        print(f"                 Pred No   Pred Yes")
        print(f"    Actual No   {int(row['true_negative']):8d}   {int(row['false_positive']):8d}")
        print(f"    Actual Yes  {int(row['false_negative']):8d}   {int(row['true_positive']):8d}")
    best = results.loc[results["f1_score"].astype(float).idxmax(), "model"]
    print(f"\nHighest F1-score on this test set: {best}")
    print("Note: the data is synthetic, so these numbers say nothing about real-world performance.")


def save_metrics_csv(results, output_dir=OUTPUT_DIR):
    """Save the comparison table to outputs/model_metrics.csv."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "model_metrics.csv"
    results.to_csv(path, index=False)
    return path


def save_confusion_matrix_plots(results, output_dir=OUTPUT_DIR):
    """Save one confusion-matrix PNG per model (optional; skipped if matplotlib is missing)."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # no window needed
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed - skipping confusion-matrix plots.")
        return []

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for _, row in results.iterrows():
        cm = [[int(row["true_negative"]), int(row["false_positive"])],
              [int(row["false_negative"]), int(row["true_positive"])]]
        fig, ax = plt.subplots(figsize=(4, 3.5))
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1], labels=["No", "Yes"])
        ax.set_yticks([0, 1], labels=["No", "Yes"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"{row['model']}")
        top = max(max(r) for r in cm)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i][j], ha="center", va="center",
                        color="white" if cm[i][j] > top / 2 else "black")
        fig.tight_layout()
        path = output_dir / f"confusion_matrix_{row['model'].lower().replace(' ', '_')}.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        paths.append(path)
    return paths
