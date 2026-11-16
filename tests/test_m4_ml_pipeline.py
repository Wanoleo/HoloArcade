"""Milestone 4 tests for preprocessing, model training (KNN/SVM/RF), and evaluation."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np

from src.config import GESTURE_CLASSES, RAW_CSV_COLUMNS
from src.evaluate_models import evaluate_all_models
from src.preprocessing import (
    clean_dataset,
    normalize_landmarks,
    prepare_processed_dataset,
    split_train_test,
)
from src.train_models import train_all_models


def _write_synthetic_raw_dataset(raw_dir: Path, samples_per_class_per_session: int = 8) -> None:
    rng = np.random.default_rng(42)
    sessions = ["s1", "s2", "s3", "s4"]
    csv_path = raw_dir / "synthetic_gestures.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(RAW_CSV_COLUMNS)
        for sess in sessions:
            for class_idx, label in enumerate(GESTURE_CLASSES):
                # Distinct base pattern per gesture class so classifiers can separate them.
                base = np.linspace(0.05 * (class_idx + 1), 0.1 * (class_idx + 1) + 0.3, 63)
                for _ in range(samples_per_class_per_session):
                    noise = rng.normal(0.0, 0.01, size=63)
                    vec = base + noise
                    writer.writerow([sess, f"user_{sess}", "2026-11-16", "Right", label, *vec.tolist()])


class TestMilestone4MLPipeline(unittest.TestCase):
    def test_normalize_landmarks_translation_and_scale_invariant(self) -> None:
        base = np.linspace(0.1, 0.7, 63, dtype=np.float32)
        shifted_scaled = (base.reshape(21, 3) * 2.5 + np.array([1.2, -0.4, 0.3])).reshape(63)

        norm_a = normalize_landmarks(base)
        norm_b = normalize_landmarks(shifted_scaled)

        self.assertEqual(norm_a.shape, (63,))
        np.testing.assert_allclose(norm_a[:3], [0.0, 0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(norm_a, norm_b, atol=1e-5)

    def test_session_grouped_split_has_no_session_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw_dir = Path(tmp) / "raw"
            processed_dir = Path(tmp) / "processed"
            raw_dir.mkdir()
            _write_synthetic_raw_dataset(raw_dir, samples_per_class_per_session=5)

            summary = prepare_processed_dataset(raw_dir=raw_dir, processed_dir=processed_dir)
            self.assertEqual(summary["split_strategy"], "grouped_by_session")

            import pandas as pd
            train_df = pd.read_csv(processed_dir / "train.csv")
            test_df = pd.read_csv(processed_dir / "test.csv")
            overlap = set(train_df["session_id"]).intersection(set(test_df["session_id"]))
            self.assertEqual(overlap, set())

    def test_train_and_evaluate_all_three_models(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_dir = root / "raw"
            processed_dir = root / "processed"
            models_dir = root / "models"
            metrics_dir = root / "metrics"
            plots_dir = root / "plots"
            raw_dir.mkdir()

            _write_synthetic_raw_dataset(raw_dir, samples_per_class_per_session=8)
            metadata = train_all_models(
                processed_dir=processed_dir,
                models_dir=models_dir,
                raw_dir=raw_dir,
                auto_preprocess=True,
            )

            self.assertIn(metadata["selected_model"], ("knn", "svm", "random_forest"))
            for name in ("knn", "svm", "random_forest"):
                self.assertTrue((models_dir / f"{name}_model.joblib").exists())
            self.assertTrue((models_dir / "best_model.joblib").exists())
            self.assertTrue((models_dir / "model_metadata.json").exists())

            best_model = joblib.load(models_dir / "best_model.joblib")
            self.assertEqual(len(best_model.classes_), len(GESTURE_CLASSES))

            df = evaluate_all_models(
                processed_dir=processed_dir,
                models_dir=models_dir,
                metrics_dir=metrics_dir,
                plots_dir=plots_dir,
            )
            self.assertEqual(set(df["model"]), {"knn", "svm", "random_forest"})
            self.assertTrue((metrics_dir / "model_comparison.csv").exists())
            self.assertTrue((metrics_dir / "classification_reports.json").exists())
            self.assertTrue((plots_dir / "model_comparison.png").exists())
            for name in ("knn", "svm", "random_forest"):
                self.assertTrue((plots_dir / f"cm_{name}.png").exists())

            with (metrics_dir / "classification_reports.json").open("r", encoding="utf-8") as h:
                reports = json.load(h)
            self.assertEqual(set(reports.keys()), {"knn", "svm", "random_forest"})


if __name__ == "__main__":
    unittest.main()
