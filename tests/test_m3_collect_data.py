"""Milestone 3 tests for gesture dataset collection CSV helpers."""

from __future__ import annotations

import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.collect_data import append_sample_to_csv, build_sample_row
from src.config import GESTURE_CLASSES, RAW_CSV_COLUMNS
from src.hand_detection import NUM_FEATURES, NUM_LANDMARKS, HandDetectionResult


class TestMilestone3CollectData(unittest.TestCase):
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

    def test_build_sample_row_valid(self) -> None:
        row = build_sample_row(
            session_id="sess_01",
            participant="alice",
            label="open_palm",
            detection=self._make_detection(),
            timestamp="2026-11-16T01:40:00.000",
        )
        self.assertEqual(len(row), len(RAW_CSV_COLUMNS))
        self.assertEqual(row[0], "sess_01")
        self.assertEqual(row[1], "alice")
        self.assertEqual(row[4], "open_palm")

    def test_build_sample_row_rejects_invalid_label(self) -> None:
        with self.assertRaises(ValueError):
            build_sample_row("s1", "alice", "invalid_gesture", self._make_detection())

    def test_build_sample_row_rejects_no_hand(self) -> None:
        empty = HandDetectionResult(None, None, "None", 0.0, 0, None)
        with self.assertRaises(ValueError):
            build_sample_row("s1", "alice", GESTURE_CLASSES[0], empty)

    def test_append_sample_to_csv_writes_single_header(self) -> None:
        detection = self._make_detection()
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "sample.csv"
            row1 = build_sample_row("s1", "alice", "fist", detection, "t1")
            row2 = build_sample_row("s1", "alice", "pinch", detection, "t2")
            append_sample_to_csv(csv_path, row1)
            append_sample_to_csv(csv_path, row2)

            with csv_path.open("r", encoding="utf-8") as handle:
                rows = list(csv.reader(handle))

        self.assertEqual(len(rows), 3)
        self.assertEqual(tuple(rows[0]), RAW_CSV_COLUMNS)
        self.assertEqual(rows[1][4], "fist")
        self.assertEqual(rows[2][4], "pinch")


if __name__ == "__main__":
    unittest.main()
