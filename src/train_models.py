"""Train and compare KNN, SVM, and Random Forest gesture classifiers (Milestone 4)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC

from src.config import (
    DATASET_PROCESSED_DIR,
    DATASET_RAW_DIR,
    LANDMARK_COLUMNS,
    MODELS_DIR,
)
from src.preprocessing import extract_geometric_features, prepare_processed_dataset


def build_candidate_models(random_state: int = 42) -> dict[str, Pipeline]:
    """Create the three PRD-specified classifiers with geometric feature augmentation."""
    geom = FunctionTransformer(extract_geometric_features, validate=False)
    return {
        "knn": Pipeline(
            [
                ("geom", geom),
                ("scaler", StandardScaler()),
                ("clf", KNeighborsClassifier(n_neighbors=7, weights="distance")),
            ]
        ),
        "svm": Pipeline(
            [
                ("geom", geom),
                ("scaler", StandardScaler()),
                (
                    "clf",
                    SVC(
                        kernel="rbf",
                        C=5.0,
                        probability=True,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("geom", geom),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=200,
                        max_depth=18,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }


def _measure_latency_ms(model: Pipeline, X_test: np.ndarray) -> float:
    """Measure average single-sample prediction latency in milliseconds."""
    if len(X_test) == 0:
        return 0.0
    sample = X_test[:1]
    start = time.perf_counter()
    repeats = 50
    for _ in range(repeats):
        model.predict(sample)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return elapsed_ms / repeats


def train_all_models(
    processed_dir: Path = DATASET_PROCESSED_DIR,
    models_dir: Path = MODELS_DIR,
    raw_dir: Path = DATASET_RAW_DIR,
    auto_preprocess: bool = True,
) -> dict[str, object]:
    """Train KNN, SVM, and Random Forest on the same split and save artifacts."""
    train_path = processed_dir / "train.csv"
    test_path = processed_dir / "test.csv"

    if auto_preprocess or not (train_path.exists() and test_path.exists()):
        prepare_processed_dataset(raw_dir=raw_dir, processed_dir=processed_dir)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df[list(LANDMARK_COLUMNS)].to_numpy(dtype=np.float32)
    y_train = train_df["label"].to_numpy()
    X_test = test_df[list(LANDMARK_COLUMNS)].to_numpy(dtype=np.float32)
    y_test = test_df["label"].to_numpy()

    models_dir.mkdir(parents=True, exist_ok=True)
    candidates = build_candidate_models()
    results: dict[str, dict[str, float]] = {}

    best_name = ""
    best_score_key = (-1.0, -1.0, float("-inf"))  # (f1_macro, accuracy, -latency)
    best_model: Pipeline | None = None

    for name, pipeline in candidates.items():
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        acc = float(accuracy_score(y_test, y_pred))
        f1_macro = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
        latency_ms = float(_measure_latency_ms(pipeline, X_test))

        model_file = models_dir / f"{name}_model.joblib"
        joblib.dump(pipeline, model_file)

        results[name] = {
            "accuracy": round(acc, 4),
            "f1_macro": round(f1_macro, 4),
            "latency_ms": round(latency_ms, 4),
        }

        score_key = (f1_macro, acc, -latency_ms)
        if score_key > best_score_key:
            best_score_key = score_key
            best_name = name
            best_model = pipeline

    if best_model is not None:
        joblib.dump(best_model, models_dir / "best_model.joblib")

    metadata = {
        "selected_model": best_name,
        "classes": sorted(np.unique(y_train).tolist()),
        "models": results,
    }
    with (models_dir / "model_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description="Train KNN, SVM, and Random Forest.")
    parser.add_argument("--raw-dir", type=Path, default=DATASET_RAW_DIR)
    parser.add_argument("--processed-dir", type=Path, default=DATASET_PROCESSED_DIR)
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    args = parser.parse_args()

    metadata = train_all_models(
        processed_dir=args.processed_dir,
        models_dir=args.models_dir,
        raw_dir=args.raw_dir,
    )
    print(f"[HoloEscape] Training complete. Selected model: {metadata['selected_model']}")
    print(json.dumps(metadata["models"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
