"""Live gesture recognition engine with smoothing, confidence gating, and debounce (Milestone 5)."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
import time

import joblib
import numpy as np

from src.config import MODELS_DIR
from src.hand_detection import HandDetectionResult
from src.preprocessing import normalize_landmarks


DEFAULT_MODEL_PATH = MODELS_DIR / "best_model.joblib"


def _geometric_gesture_fallback(landmarks_xyz: np.ndarray) -> str | None:
    """Classify clear hand poses from 21 MediaPipe landmarks when ML confidence is low."""
    if landmarks_xyz.shape != (21, 3):
        return None

    wrist = landmarks_xyz[0, :2]
    index_mcp = landmarks_xyz[5, :2]
    middle_mcp = landmarks_xyz[9, :2]
    pinky_mcp = landmarks_xyz[17, :2]

    # Reject degenerate/collinear synthetic landmarks
    v1 = index_mcp - wrist
    v2 = pinky_mcp - wrist
    palm_area = abs(float(v1[0] * v2[1] - v1[1] * v2[0]))
    if palm_area < 1e-4:
        return None

    hand_scale = float(np.linalg.norm(middle_mcp - wrist))
    if hand_scale < 1e-4:
        return None

    def _dist(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a - b))

    def _is_extended(pip_idx: int, tip_idx: int) -> bool:
        d_pip = _dist(landmarks_xyz[pip_idx, :2], wrist)
        d_tip = _dist(landmarks_xyz[tip_idx, :2], wrist)
        return d_tip > d_pip * 1.12 and d_tip > hand_scale * 1.25

    index_ext = _is_extended(6, 8)
    middle_ext = _is_extended(10, 12)
    ring_ext = _is_extended(14, 16)
    pinky_ext = _is_extended(18, 20)

    thumb_tip = landmarks_xyz[4, :2]
    thumb_ip = landmarks_xyz[3, :2]
    index_tip = landmarks_xyz[8, :2]

    if _dist(thumb_tip, index_tip) < hand_scale * 0.42 and not (middle_ext and ring_ext and pinky_ext):
        return "pinch"
    if index_ext and middle_ext and ring_ext and pinky_ext:
        return "open_palm"
    if index_ext and middle_ext and not ring_ext and not pinky_ext:
        return "two_fingers"
    if index_ext and not middle_ext and not ring_ext and not pinky_ext:
        return "pointing"
    if not index_ext and not middle_ext and not ring_ext and not pinky_ext:
        if thumb_tip[1] < thumb_ip[1] - hand_scale * 0.12 and _dist(thumb_tip, index_mcp) > hand_scale * 0.55:
            return "thumbs_up"
        return "fist"
    return None


class PointerSmoother:
    """Exponential moving average (EMA) smoother with deadzone to reduce hand jitter."""

    def __init__(self, alpha: float = 0.16, deadzone: float = 0.012) -> None:
        self.alpha = max(0.05, min(1.0, alpha))
        self.deadzone = max(0.0, deadzone)
        self._pos: tuple[float, float] | None = None

    def reset(self) -> None:
        """Reset stored pointer position."""
        self._pos = None

    def update(self, raw_xy: tuple[float, float] | None, freeze: bool = False) -> tuple[float, float] | None:
        """Return smoothed (x, y) pointer coordinates, or None when no hand is present."""
        if raw_xy is None:
            self._pos = None
            return None

        if freeze and self._pos is not None:
            return self._pos

        rx = max(0.0, min(1.0, float(raw_xy[0])))
        ry = max(0.0, min(1.0, float(raw_xy[1])))
        if self._pos is None:
            self._pos = (rx, ry)
            return self._pos

        px, py = self._pos
        dx = rx - px
        dy = ry - py
        if (dx * dx + dy * dy) ** 0.5 < self.deadzone:
            return self._pos

        nx = px + self.alpha * dx
        ny = py + self.alpha * dy
        self._pos = (nx, ny)
        return self._pos


@dataclass(frozen=True)
class GesturePrediction:
    """Output of the live gesture recognizer for one frame."""

    status: str  # "no_hand" | "uncertain" | "stabilizing" | "stable"
    raw_label: str | None
    smoothed_label: str | None
    triggered_gesture: str | None
    confidence: float
    latency_ms: float


class GestureRecognizer:
    """Predicts gestures from live hand landmarks with temporal smoothing and cooldown."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        window_size: int = 3,
        confidence_threshold: float = 0.50,
        min_votes: int = 2,
        cooldown_seconds: float = 0.35,
    ) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Trained gesture model not found at '{self.model_path}'. "
                "Run 'python -m src.collect_data' and 'python -m src.train_models' first."
            )

        try:
            self._model = joblib.load(self.model_path)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load gesture model '{self.model_path}': {exc}. "
                "Re-run 'python -m src.train_models' to regenerate the model file."
            ) from exc

        if not hasattr(self._model, "predict"):
            raise RuntimeError(
                f"Loaded object from '{self.model_path}' is not a valid scikit-learn classifier."
            )

        self.window_size = max(1, window_size)
        self.confidence_threshold = confidence_threshold
        self.min_votes = max(1, min(min_votes, self.window_size))
        self.cooldown_seconds = cooldown_seconds

        self._history: deque[str | None] = deque(maxlen=self.window_size)
        self._last_triggered_label: str | None = None
        self._last_stable_label: str | None = None
        self._last_trigger_time: float = -float("inf")

    def reset(self) -> None:
        """Clear rolling prediction window and trigger state."""
        self._history.clear()
        self._last_triggered_label = None
        self._last_stable_label = None

    def predict(
        self,
        detection: HandDetectionResult,
        now: float | None = None,
    ) -> GesturePrediction:
        """Predict and smooth a gesture from one HandDetectionResult."""
        current_time = time.monotonic() if now is None else now

        if not detection.detected or detection.features is None:
            self.reset()
            return GesturePrediction(
                status="no_hand",
                raw_label=None,
                smoothed_label=None,
                triggered_gesture=None,
                confidence=0.0,
                latency_ms=0.0,
            )

        try:
            norm_features = normalize_landmarks(detection.features).reshape(1, -1)
        except ValueError:
            self._history.append(None)
            return GesturePrediction(
                status="uncertain",
                raw_label=None,
                smoothed_label=None,
                triggered_gesture=None,
                confidence=0.0,
                latency_ms=0.0,
            )

        start = time.perf_counter()
        if hasattr(self._model, "predict_proba"):
            probs = self._model.predict_proba(norm_features)[0]
            best_idx = int(np.argmax(probs))
            confidence = float(probs[best_idx])
            raw_label = str(self._model.classes_[best_idx])
        else:
            raw_label = str(self._model.predict(norm_features)[0])
            confidence = 1.0
        latency_ms = (time.perf_counter() - start) * 1000.0

        if confidence < self.confidence_threshold:
            geo_label = (
                _geometric_gesture_fallback(detection.landmarks_xyz)
                if detection.landmarks_xyz is not None
                else None
            )
            if geo_label is not None:
                raw_label = geo_label
                confidence = max(confidence, self.confidence_threshold)
                self._history.append(geo_label)
            else:
                self._history.append(None)
        else:
            self._history.append(raw_label)

        valid_votes = [item for item in self._history if item is not None]
        if not valid_votes:
            self._last_triggered_label = None
            return GesturePrediction(
                status="uncertain",
                raw_label=raw_label,
                smoothed_label=None,
                triggered_gesture=None,
                confidence=confidence,
                latency_ms=latency_ms,
            )

        top_label, top_count = Counter(valid_votes).most_common(1)[0]
        if top_count < self.min_votes:
            return GesturePrediction(
                status="stabilizing",
                raw_label=raw_label,
                smoothed_label=self._last_stable_label,
                triggered_gesture=None,
                confidence=confidence,
                latency_ms=latency_ms,
            )

        self._last_stable_label = top_label
        triggered: str | None = None
        if (
            top_label != self._last_triggered_label
            and (current_time - self._last_trigger_time) >= self.cooldown_seconds
        ):
            triggered = top_label
            self._last_triggered_label = top_label
            self._last_trigger_time = current_time

        return GesturePrediction(
            status="stable",
            raw_label=raw_label,
            smoothed_label=top_label,
            triggered_gesture=triggered,
            confidence=confidence,
            latency_ms=latency_ms,
        )


class DualHandGestureRecognizer:
    """Simultaneously tracks and smooths gestures for primary and secondary hands."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        window_size: int = 3,
        confidence_threshold: float = 0.50,
        min_votes: int = 2,
        cooldown_seconds: float = 0.35,
    ) -> None:
        self.primary = GestureRecognizer(
            model_path=model_path,
            window_size=window_size,
            confidence_threshold=confidence_threshold,
            min_votes=min_votes,
            cooldown_seconds=cooldown_seconds,
        )
        self.secondary = GestureRecognizer(
            model_path=model_path,
            window_size=window_size,
            confidence_threshold=confidence_threshold,
            min_votes=min_votes,
            cooldown_seconds=cooldown_seconds,
        )

    def reset(self) -> None:
        """Reset smoothing and trigger state for both hands."""
        self.primary.reset()
        self.secondary.reset()

    def predict(
        self,
        detection: HandDetectionResult,
        now: float | None = None,
    ) -> tuple[GesturePrediction, GesturePrediction]:
        """Return (primary_prediction, secondary_prediction) for the current frame."""
        p1 = self.primary.predict(detection, now=now)
        p2 = self.secondary.predict(detection.secondary_as_result, now=now)
        return p1, p2
