"""Shared project constants for HoloEscape."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_RAW_DIR = PROJECT_ROOT / "dataset" / "raw"
DATASET_PROCESSED_DIR = PROJECT_ROOT / "dataset" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_PLOTS_DIR = PROJECT_ROOT / "results" / "plots"
RESULTS_METRICS_DIR = PROJECT_ROOT / "results" / "metrics"

GESTURE_CLASSES: tuple[str, ...] = (
    "open_palm",
    "fist",
    "pointing",
    "thumbs_up",
    "pinch",
    "two_fingers",
)

GESTURE_DESCRIPTIONS: dict[str, str] = {
    "open_palm": "Open menu / show controls",
    "fist": "Close menu / cancel action",
    "pointing": "Select / highlight item",
    "thumbs_up": "Confirm action",
    "pinch": "Grab, hold, or release object",
    "two_fingers": "Rotate / inspect object",
}

LANDMARK_COLUMNS: tuple[str, ...] = tuple(
    f"{axis}{idx}" for idx in range(21) for axis in ("x", "y", "z")
)

RAW_CSV_COLUMNS: tuple[str, ...] = (
    "session_id",
    "participant",
    "timestamp",
    "handedness",
    "label",
    *LANDMARK_COLUMNS,
)
