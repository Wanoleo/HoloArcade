"""MediaPipe Hand Landmarker wrapper for 21-landmark extraction and visualization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import numpy as np


DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "hand_landmarker.task"
NUM_LANDMARKS = 21
NUM_FEATURES = NUM_LANDMARKS * 3  # x, y, z per landmark = 63 features

# Standard 21-point MediaPipe hand skeleton connections.
HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index finger
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle finger
    (0, 9), (9, 10), (10, 11), (11, 12),
    # Ring finger
    (0, 13), (13, 14), (14, 15), (15, 16),
    # Pinky finger
    (0, 17), (17, 18), (18, 19), (19, 20),
    # Palm arch
    (5, 9), (9, 13), (13, 17),
)


@dataclass(frozen=True)
class HandDetectionResult:
    """Validated single-hand landmark output for one frame."""

    landmarks_xyz: np.ndarray | None
    features: np.ndarray | None
    handedness: str
    confidence: float
    num_hands_detected: int
    pointer_xy: tuple[float, float] | None

    @property
    def detected(self) -> bool:
        """Return True when a valid 21-landmark hand is present."""
        return self.landmarks_xyz is not None and self.features is not None


def _empty_result(num_hands_detected: int = 0) -> HandDetectionResult:
    return HandDetectionResult(
        landmarks_xyz=None,
        features=None,
        handedness="None",
        confidence=0.0,
        num_hands_detected=num_hands_detected,
        pointer_xy=None,
    )


def _select_primary_hand_index(
    hand_landmarks_list: list,
    handedness_list: list,
) -> tuple[int, str, float]:
    """Select one hand consistently when multiple hands are present.

    Rule (PRD v1): pick the hand with the highest detection/handedness score;
    break ties by largest 2D bounding-box area in normalized image coordinates.
    """
    best_idx = 0
    best_score = -1.0
    best_area = -1.0
    best_label = "Unknown"

    for idx, landmarks in enumerate(hand_landmarks_list):
        score = 0.0
        label = "Unknown"
        if idx < len(handedness_list) and handedness_list[idx]:
            top_cat = handedness_list[idx][0]
            score = float(getattr(top_cat, "score", 0.0))
            label = str(getattr(top_cat, "category_name", "Unknown"))

        xs = [lm.x for lm in landmarks]
        ys = [lm.y for lm in landmarks]
        area = (max(xs) - min(xs)) * (max(ys) - min(ys)) if xs and ys else 0.0

        if (score > best_score) or (score == best_score and area > best_area):
            best_idx = idx
            best_score = score
            best_area = area
            best_label = label

    return best_idx, best_label, max(0.0, best_score)


class HandTracker:
    """Detects one hand per frame and extracts 63 (x, y, z) landmark features."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        max_num_hands: int = 2,
        min_hand_detection_confidence: float = 0.5,
        min_hand_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"MediaPipe hand landmarker model not found at '{self.model_path}'. "
                "Download 'hand_landmarker.task' into the models/ directory."
            )

        base_options = mp_python.BaseOptions(model_asset_path=str(self.model_path))
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_hand_detection_confidence,
            min_hand_presence_confidence=min_hand_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._last_timestamp_ms = -1

    def detect(self, frame_bgr: np.ndarray) -> HandDetectionResult:
        """Detect a hand in a BGR frame and return validated 21-landmark features."""
        if frame_bgr is None or frame_bgr.size == 0:
            return _empty_result()

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        timestamp_ms = int(time.monotonic() * 1000)
        if timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = timestamp_ms

        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        hand_landmarks_list = result.hand_landmarks or []
        handedness_list = result.handedness or []
        num_hands = len(hand_landmarks_list)

        if num_hands == 0:
            return _empty_result(0)

        idx, label, score = _select_primary_hand_index(hand_landmarks_list, handedness_list)
        selected = hand_landmarks_list[idx]
        if len(selected) != NUM_LANDMARKS:
            return _empty_result(num_hands)

        coords = np.array(
            [[lm.x, lm.y, lm.z] for lm in selected],
            dtype=np.float32,
        )
        if coords.shape != (NUM_LANDMARKS, 3) or not np.all(np.isfinite(coords)):
            return _empty_result(num_hands)

        features = coords.reshape(NUM_FEATURES)
        pointer_xy = (float(coords[8, 0]), float(coords[8, 1]))

        return HandDetectionResult(
            landmarks_xyz=coords,
            features=features,
            handedness=label,
            confidence=score,
            num_hands_detected=num_hands,
            pointer_xy=pointer_xy,
        )

    def close(self) -> None:
        """Release MediaPipe resources."""
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

    def __enter__(self) -> "HandTracker":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


def draw_hand_landmarks(
    frame_bgr: np.ndarray,
    result: HandDetectionResult,
) -> np.ndarray:
    """Draw the 21 hand landmarks and skeleton connections onto a BGR frame copy."""
    annotated = frame_bgr.copy()
    if not result.detected or result.landmarks_xyz is None:
        return annotated

    height, width = annotated.shape[:2]
    points = [
        (
            int(np.clip(lm[0], 0.0, 1.0) * width),
            int(np.clip(lm[1], 0.0, 1.0) * height),
        )
        for lm in result.landmarks_xyz
    ]

    for start_idx, end_idx in HAND_CONNECTIONS:
        cv2.line(annotated, points[start_idx], points[end_idx], (80, 220, 160), 2)

    for idx, pt in enumerate(points):
        if idx == 0:
            color = (0, 200, 255)  # Wrist
            radius = 6
        elif idx == 8:
            color = (255, 180, 0)  # Index fingertip (pointer)
            radius = 6
        else:
            color = (240, 240, 240)
            radius = 4
        cv2.circle(annotated, pt, radius, color, -1)
        cv2.circle(annotated, pt, radius, (20, 30, 40), 1)

    return annotated
