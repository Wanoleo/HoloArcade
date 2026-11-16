"""Evaluate trained KNN, SVM, and Random Forest models (Milestone 4)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import joblib
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless plot generation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.config import (
    DATASET_PROCESSED_DIR,
    LANDMARK_COLUMNS,
    MODELS_DIR,
    RESULTS_METRICS_DIR,
    RESULTS_PLOTS_DIR,
)


MODEL_NAMES = ("knn", "svm", "random_forest")


def _plot_confusion_matrix(
    cm: np.ndarray,
    labels: list[str],
    model_name: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=30, colorbar=False)
    ax.set_title(f"Confusion Matrix — {model_name.upper()}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_model_comparison(comparison_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(comparison_df))
    width = 0.25

    ax.bar(x - width, comparison_df["accuracy"], width, label="Accuracy")
    ax.bar(x, comparison_df["f1_macro"], width, label="Macro F1")
    ax.bar(x + width, comparison_df["recall_macro"], width, label="Macro Recall")

    ax.set_xticks(x)
    ax.set_xticklabels([name.upper() for name in comparison_df["model"]])
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("HoloEscape Model Comparison (Held-Out Test Set)")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def evaluate_all_models(
    processed_dir: Path = DATASET_PROCESSED_DIR,
    models_dir: Path = MODELS_DIR,
    metrics_dir: Path = RESULTS_METRICS_DIR,
    plots_dir: Path = RESULTS_PLOTS_DIR,
) -> pd.DataFrame:
    """Evaluate all saved classifiers and write metrics CSV, JSON, and PNG plots."""
    test_path = processed_dir / "test.csv"
    if not test_path.exists():
        raise FileNotFoundError(
            f"Processed test set not found at '{test_path}'. Run train_models first."
        )

    test_df = pd.read_csv(test_path)
    X_test = test_df[list(LANDMARK_COLUMNS)].to_numpy(dtype=np.float32)
    y_test = test_df["label"].to_numpy()
    labels = sorted(np.unique(y_test).tolist())

    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    reports: dict[str, object] = {}

    for name in MODEL_NAMES:
        model_path = models_dir / f"{name}_model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Trained model not found at '{model_path}'.")

        model = joblib.load(model_path)
        start = time.perf_counter()
        y_pred = model.predict(X_test)
        total_ms = (time.perf_counter() - start) * 1000.0
        latency_ms = total_ms / max(len(X_test), 1)

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
        rec = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
        f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))

        rows.append(
            {
                "model": name,
                "accuracy": round(acc, 4),
                "precision_macro": round(prec, 4),
                "recall_macro": round(rec, 4),
                "f1_macro": round(f1, 4),
                "latency_ms_per_sample": round(latency_ms, 4),
            }
        )

        reports[name] = classification_report(
            y_test, y_pred, labels=labels, output_dict=True, zero_division=0
        )

        cm = confusion_matrix(y_test, y_pred, labels=labels)
        _plot_confusion_matrix(cm, labels, name, plots_dir / f"cm_{name}.png")

    comparison_df = pd.DataFrame(rows)
    comparison_df.to_csv(metrics_dir / "model_comparison.csv", index=False)

    with (metrics_dir / "classification_reports.json").open("w", encoding="utf-8") as handle:
        json.dump(reports, handle, indent=2)

    _plot_model_comparison(comparison_df, plots_dir / "model_comparison.png")
    return comparison_df


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate trained gesture classifiers.")
    parser.add_argument("--processed-dir", type=Path, default=DATASET_PROCESSED_DIR)
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--metrics-dir", type=Path, default=RESULTS_METRICS_DIR)
    parser.add_argument("--plots-dir", type=Path, default=RESULTS_PLOTS_DIR)
    args = parser.parse_args()

    df = evaluate_all_models(
        processed_dir=args.processed_dir,
        models_dir=args.models_dir,
        metrics_dir=args.metrics_dir,
        plots_dir=args.plots_dir,
    )
    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
