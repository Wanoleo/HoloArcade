"""Milestone 5 tests for GestureRecognizer smoothing, confidence gating, and debounce."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np
from sklearn.dummy import DummyClassifier

from src.gesture_recognition import GestureRecognizer
from src.hand_detection import NUM_FEATURES, NUM_LANDMARKS, HandDetectionResult


class _FixedProbaModel:
    """Simple picklable classifier stub for testing smoothing and confidence thresholds."""

    def __init__(self, classes: list[str], probs: list[float]) -> None:
        self.classes_ = np.array(classes)
        self._probs = np.array(probs, dtype=np.float32)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.tile(self._probs, (len(X), 1))

    def predict(self, X: np.ndarray) -> np.ndarray:
        label = self.classes_[int(np.argmax(self._probs))]
        return np.array([label] * len(X))


class TestMilestone5GestureRecognition(unittest.TestCase):
    def _make_detection(self) -> HandDetectionResult:
        landmarks = np.linspace(0.1, 0.9, NUM_FEATURES, dtype=np.float32).reshape(
            NUM_LANDMARKS, 3
        )
        return HandDetectionResult(
            landmarks_xyz=landmarks,
            features=landmarks.reshape(NUM_FEATURES),
            handedness="Right",
            confidence=0.95,
            num_hands_detected=1,
            pointer_xy=(float(landmarks[8, 0]), float(landmarks[8, 1])),
        )

    def test_missing_model_raises_file_not_found(self) -> None:
        with self.assertRaises(FileNotFoundError):
            GestureRecognizer(model_path="models/non_existent_model.joblib")

    def test_corrupt_model_raises_runtime_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bad_path = Path(tmp) / "corrupt.joblib"
            bad_path.write_text("not a joblib file", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                GestureRecognizer(model_path=bad_path)

    def test_no_hand_resets_state_and_returns_neutral(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "model.joblib"
            joblib.dump(_FixedProbaModel(["fist", "open_palm"], [0.1, 0.9]), model_path)
            rec = GestureRecognizer(model_path=model_path, window_size=5, min_votes=3)

            empty = HandDetectionResult(None, None, "None", 0.0, 0, None)
            pred = rec.predict(empty, now=1.0)
            self.assertEqual(pred.status, "no_hand")
            self.assertIsNone(pred.smoothed_label)
            self.assertIsNone(pred.triggered_gesture)

    def test_low_confidence_returns_uncertain(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "model.joblib"
            joblib.dump(_FixedProbaModel(["fist", "open_palm"], [0.45, 0.55]), model_path)
            rec = GestureRecognizer(
                model_path=model_path,
                confidence_threshold=0.70,
                window_size=5,
                min_votes=3,
            )
            pred = rec.predict(self._make_detection(), now=1.0)
            self.assertEqual(pred.status, "uncertain")
            self.assertIsNone(pred.triggered_gesture)

    def test_smoothing_and_debounce_trigger_once_per_hold(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "model.joblib"
            joblib.dump(_FixedProbaModel(["fist", "thumbs_up"], [0.05, 0.95]), model_path)
            rec = GestureRecognizer(
                model_path=model_path,
                window_size=5,
                min_votes=3,
                confidence_threshold=0.55,
                cooldown_seconds=0.5,
            )
            det = self._make_detection()

            p1 = rec.predict(det, now=1.0)
            p2 = rec.predict(det, now=1.1)
            p3 = rec.predict(det, now=1.2)
            p4 = rec.predict(det, now=2.0)

            self.assertEqual(p1.status, "stabilizing")
            self.assertEqual(p2.status, "stabilizing")
            self.assertEqual(p3.status, "stable")
            self.assertEqual(p3.smoothed_label, "thumbs_up")
            self.assertEqual(p3.triggered_gesture, "thumbs_up")
            # Holding the same gesture should not re-trigger even after cooldown.
            self.assertEqual(p4.status, "stable")
            self.assertIsNone(p4.triggered_gesture)

    def test_geometric_fallback_recovers_low_confidence_real_hand(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "model.joblib"
            joblib.dump(_FixedProbaModel(["fist", "pointing"], [0.49, 0.51]), model_path)
            rec = GestureRecognizer(
                model_path=model_path,
                confidence_threshold=0.75,
                window_size=2,
                min_votes=2,
                cooldown_seconds=0.1,
            )
            lm = np.zeros((NUM_LANDMARKS, 3), dtype=np.float32)
            lm[0] = [0.50, 0.80, 0.0]   # wrist
            lm[5] = [0.45, 0.62, 0.0]   # index_mcp
            lm[6] = [0.45, 0.50, 0.0]   # index_pip
            lm[8] = [0.45, 0.32, 0.0]   # index_tip (extended)
            lm[9] = [0.50, 0.60, 0.0]   # middle_mcp
            lm[10] = [0.50, 0.56, 0.0]
            lm[12] = [0.50, 0.65, 0.0]  # middle_tip (curled)
            lm[13] = [0.54, 0.62, 0.0]  # ring_mcp
            lm[14] = [0.54, 0.58, 0.0]
            lm[16] = [0.54, 0.66, 0.0]
            lm[17] = [0.58, 0.65, 0.0]  # pinky_mcp
            lm[18] = [0.58, 0.61, 0.0]
            lm[20] = [0.58, 0.68, 0.0]
            lm[3] = [0.41, 0.70, 0.0]
            lm[4] = [0.43, 0.66, 0.0]

            det = HandDetectionResult(lm, lm.reshape(NUM_FEATURES), "Right", 0.95, 1, (0.45, 0.32))
            rec.predict(det, now=1.0)
            p2 = rec.predict(det, now=1.1)
            self.assertEqual(p2.status, "stable")
            self.assertEqual(p2.smoothed_label, "pointing")
            self.assertEqual(p2.triggered_gesture, "pointing")


if __name__ == "__main__":
    unittest.main()
