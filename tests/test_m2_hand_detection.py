"""Milestone 2 tests for MediaPipe HandTracker and 63-feature extraction."""

from __future__ import annotations

from dataclasses import dataclass
import unittest

import numpy as np

from src.hand_detection import (
    NUM_FEATURES,
    NUM_LANDMARKS,
    HandDetectionResult,
    HandTracker,
    _select_primary_hand_index,
    draw_hand_landmarks,
)


@dataclass
class _DummyLandmark:
    x: float
    y: float
    z: float


@dataclass
class _DummyCategory:
    category_name: str
    score: float


class TestMilestone2HandDetection(unittest.TestCase):
    def test_missing_model_raises_file_not_found(self) -> None:
        with self.assertRaises(FileNotFoundError):
            HandTracker(model_path="models/does_not_exist.task")

    def test_blank_frame_returns_no_hand(self) -> None:
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        with HandTracker() as tracker:
            result = tracker.detect(blank)
        self.assertFalse(result.detected)
        self.assertIsNone(result.features)
        self.assertIsNone(result.landmarks_xyz)
        self.assertEqual(result.num_hands_detected, 0)

    def test_empty_frame_returns_no_hand(self) -> None:
        empty = np.zeros((0, 0, 3), dtype=np.uint8)
        with HandTracker() as tracker:
            result = tracker.detect(empty)
        self.assertFalse(result.detected)

    def test_multi_hand_selection_prefers_highest_confidence(self) -> None:
        hand_a = [_DummyLandmark(0.1, 0.1, 0.0) for _ in range(NUM_LANDMARKS)]
        hand_b = [_DummyLandmark(0.5, 0.5, 0.0) for _ in range(NUM_LANDMARKS)]
        handedness = [
            [_DummyCategory("Left", 0.65)],
            [_DummyCategory("Right", 0.92)],
        ]
        idx, label, score = _select_primary_hand_index([hand_a, hand_b], handedness)
        self.assertEqual(idx, 1)
        self.assertEqual(label, "Right")
        self.assertAlmostEqual(score, 0.92)

    def test_draw_landmarks_with_valid_result_preserves_shape(self) -> None:
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        landmarks = np.linspace(0.2, 0.8, NUM_FEATURES, dtype=np.float32).reshape(
            NUM_LANDMARKS, 3
        )
        result = HandDetectionResult(
            landmarks_xyz=landmarks,
            features=landmarks.reshape(NUM_FEATURES),
            handedness="Right",
            confidence=0.95,
            num_hands_detected=1,
            pointer_xy=(float(landmarks[8, 0]), float(landmarks[8, 1])),
        )
        drawn = draw_hand_landmarks(frame, result)
        self.assertEqual(drawn.shape, frame.shape)
        self.assertGreater(int(drawn.sum()), 0)


if __name__ == "__main__":
    unittest.main()
