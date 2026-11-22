"""Tests for two-hand support and ergonomic HoloEscape controls."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np

from src.game_logic import CHEST_CLEAR_BOX, CHEST_PAD_BOXES, SAFE_POS, EscapeRoomGame
from src.gesture_recognition import DualHandGestureRecognizer
from src.hand_detection import (
    NUM_FEATURES,
    NUM_LANDMARKS,
    HandDetectionResult,
    SingleHandData,
    draw_hand_landmarks,
)


def _box_center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    return (box[0] + box[2] / 2.0, box[1] + box[3] / 2.0)


class _StubModel:
    def __init__(self, classes: list[str], probs: list[float]) -> None:
        self.classes_ = np.array(classes)
        self._probs = np.array(probs, dtype=np.float32)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.tile(self._probs, (len(X), 1))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self.classes_[int(np.argmax(self._probs))]] * len(X))


class TestTwoHandsAndErgonomics(unittest.TestCase):
    def test_dual_hand_detection_and_drawing(self) -> None:
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        lm1 = np.linspace(0.1, 0.4, NUM_FEATURES, dtype=np.float32).reshape(NUM_LANDMARKS, 3)
        lm2 = np.linspace(0.5, 0.9, NUM_FEATURES, dtype=np.float32).reshape(NUM_LANDMARKS, 3)
        sec = SingleHandData(
            landmarks_xyz=lm2,
            features=lm2.reshape(NUM_FEATURES),
            handedness="Left",
            confidence=0.90,
            pointer_xy=(float(lm2[8, 0]), float(lm2[8, 1])),
        )
        res = HandDetectionResult(
            landmarks_xyz=lm1,
            features=lm1.reshape(NUM_FEATURES),
            handedness="Right",
            confidence=0.95,
            num_hands_detected=2,
            pointer_xy=(float(lm1[8, 0]), float(lm1[8, 1])),
            secondary_hand=sec,
        )
        drawn = draw_hand_landmarks(frame, res)
        self.assertGreater(int(drawn.sum()), 0)

        with tempfile.TemporaryDirectory() as tmp:
            mpath = Path(tmp) / "m.joblib"
            joblib.dump(_StubModel(["pinch", "pointing"], [0.9, 0.1]), mpath)
            dual = DualHandGestureRecognizer(model_path=mpath, window_size=2, min_votes=2, cooldown_seconds=0.1)
            dual.predict(res, now=1.0)
            p1, p2 = dual.predict(res, now=1.1)
            self.assertEqual(p1.smoothed_label, "pinch")
            self.assertEqual(p2.smoothed_label, "pinch")

    def test_pinch_selects_and_fist_preserves_code(self) -> None:
        game = EscapeRoomGame()
        # Pinch opens safe
        game.update("pinch", "pinch", SAFE_POS, "stable", now=1.0)
        self.assertTrue(game.state.safe_panel_open)

        # Pinch enters '3' (index 2)
        game.update("pinch", "pinch", _box_center(CHEST_PAD_BOXES[2]), "stable", now=2.0)
        self.assertEqual(game.state.dials, [3])

        # Accidental fist closes panel WITHOUT clearing entered code
        game.update("fist", "fist", _box_center(CHEST_PAD_BOXES[2]), "stable", now=2.5)
        self.assertFalse(game.state.safe_panel_open)
        self.assertEqual(game.state.dials, [3])

        # Reopen safe and use CLEAR CODE button to clear
        game.update("pinch", "pinch", SAFE_POS, "stable", now=3.0)
        self.assertTrue(game.state.safe_panel_open)
        game.update("pinch", "pinch", _box_center(CHEST_CLEAR_BOX), "stable", now=3.5)
        self.assertEqual(game.state.dials, [])

    def test_two_hand_confirm_and_dwell_select(self) -> None:
        game = EscapeRoomGame()
        game.update("pinch", "pinch", SAFE_POS, "stable", now=1.0)
        self.assertTrue(game.state.safe_panel_open)

        # Hand 1 points at digit '7' (index 6), Hand 2 triggers pinch -> enters 7
        game.update(
            None,
            "pointing",
            _box_center(CHEST_PAD_BOXES[6]),
            "stable",
            now=2.0,
            secondary_triggered="pinch",
            secondary_held="pinch",
        )
        self.assertEqual(game.state.dials, [7])

        # Dwell-to-select by holding pointing steadily over digit '2' (index 1)
        pad_2_center = _box_center(CHEST_PAD_BOXES[1])
        t = 3.0
        for step in range(12):
            game.update(None, "pointing", pad_2_center, "stable", now=t + step * 0.09)
        self.assertEqual(game.state.dials, [7, 2])


if __name__ == "__main__":
    unittest.main()
