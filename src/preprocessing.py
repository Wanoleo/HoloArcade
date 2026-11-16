"""Landmark preprocessing, normalization, and session-grouped splitting (Milestone 4)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from src.config import (
    DATASET_PROCESSED_DIR,
    DATASET_RAW_DIR,
    GESTURE_CLASSES,
    LANDMARK_COLUMNS,
)


def normalize_landmarks(features: np.ndarray) -> np.ndarray:
    """Normalize 63-feature (21 x 3) landmark vector(s) relative to wrist and hand scale.

    Supports both a single sample of shape (63,) and a batch of shape (N, 63).
    """
    arr = np.asarray(features, dtype=np.float32)
    single_sample = arr.ndim == 1
    if single_sample:
        arr = arr.reshape(1, -1)

    if arr.ndim != 2 or arr.shape[1] != len(LANDMARK_COLUMNS):
        raise ValueError(
            f"Expected landmark array with 63 features, got shape {arr.shape}."
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError("Landmark array contains NaN or infinite values.")

    reshaped = arr.reshape(-1, 21, 3)
    wrist = reshaped[:, 0:1, :]
    centered = reshaped - wrist

    # Scale by max Euclidean distance from wrist across all 21 landmarks.
    norms = np.linalg.norm(centered, axis=2)
    scale = np.max(norms, axis=1, keepdims=True)[..., np.newaxis]
    scale = np.where(scale < 1e-6, 1.0, scale)

    normalized = (centered / scale).reshape(-1, len(LANDMARK_COLUMNS))
    return normalized[0] if single_sample else normalized


def extract_geometric_features(X: np.ndarray) -> np.ndarray:
    """Augment 63 normalized landmarks with handedness-invariant finger distances & curl ratios."""
    arr = np.asarray(X, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    pts = arr.reshape(-1, 21, 3)

    wrist = pts[:, 0, :]
    thumb_tip = pts[:, 4, :]
    index_tip = pts[:, 8, :]
    middle_tip = pts[:, 12, :]
    ring_tip = pts[:, 16, :]
    pinky_tip = pts[:, 20, :]

    # Distances from wrist to each fingertip (5)
    d_wrist_tips = [
        np.linalg.norm(pts[:, tip, :] - wrist, axis=1, keepdims=True)
        for tip in (4, 8, 12, 16, 20)
    ]
    # Tip-to-PIP extension ratios (5)
    d_curl_ratios = []
    for pip_idx, tip_idx in ((3, 4), (6, 8), (10, 12), (14, 16), (18, 20)):
        d_pip = np.linalg.norm(pts[:, pip_idx, :] - wrist, axis=1, keepdims=True)
        d_tip = np.linalg.norm(pts[:, tip_idx, :] - wrist, axis=1, keepdims=True)
        d_curl_ratios.append(d_tip / np.maximum(d_pip, 1e-4))

    # Thumb-tip to each fingertip distances (4) + index-to-middle spread (1)
    d_pinch = [
        np.linalg.norm(thumb_tip - tip, axis=1, keepdims=True)
        for tip in (index_tip, middle_tip, ring_tip, pinky_tip)
    ]
    d_spread = [np.linalg.norm(index_tip - middle_tip, axis=1, keepdims=True)]

    extra = np.hstack([*d_wrist_tips, *d_curl_ratios, *d_pinch, *d_spread]).astype(np.float32)
    return np.hstack([arr, extra])


def load_raw_dataset(raw_dir: Path = DATASET_RAW_DIR) -> pd.DataFrame:
    """Load and concatenate all non-empty CSV files from dataset/raw/."""
    csv_files = sorted(raw_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No raw CSV files found in '{raw_dir}'. "
            "Run 'python -m src.collect_data' to collect gesture samples first."
        )

    frames = []
    for path in csv_files:
        if path.stat().st_size == 0:
            continue
        df = pd.read_csv(path)
        if not df.empty:
            frames.append(df)

    if not frames:
        raise ValueError(f"All CSV files in '{raw_dir}' are empty.")

    combined = pd.concat(frames, ignore_index=True)
    required_cols = {"session_id", "label", *LANDMARK_COLUMNS}
    missing = required_cols - set(combined.columns)
    if missing:
        raise ValueError(f"Raw dataset is missing required columns: {sorted(missing)}")

    return combined


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Drop invalid labels, missing values, and degenerate zero-scale rows."""
    cleaned = df.copy()
    cleaned = cleaned[cleaned["label"].isin(GESTURE_CLASSES)]
    cleaned = cleaned.dropna(subset=["session_id", "label", *LANDMARK_COLUMNS])

    feature_values = cleaned[list(LANDMARK_COLUMNS)].to_numpy(dtype=np.float32)
    finite_mask = np.all(np.isfinite(feature_values), axis=1)
    cleaned = cleaned.loc[finite_mask].reset_index(drop=True)

    if cleaned.empty:
        raise ValueError("No valid rows remain after cleaning the raw dataset.")

    # Normalize features in-place for processed storage.
    normalized = normalize_landmarks(cleaned[list(LANDMARK_COLUMNS)].to_numpy(dtype=np.float32))
    cleaned[list(LANDMARK_COLUMNS)] = normalized
    return cleaned


def split_train_test(
    df: pd.DataFrame,
    test_size: float = 0.25,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Split dataset into train/test, grouping by session_id when multiple sessions exist."""
    unique_groups = df["session_id"].nunique()
    if unique_groups >= 2:
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=test_size, random_state=random_state
        )
        train_idx, test_idx = next(splitter.split(df, df["label"], groups=df["session_id"]))
        return (
            df.iloc[train_idx].reset_index(drop=True),
            df.iloc[test_idx].reset_index(drop=True),
            "grouped_by_session",
        )

    # Fallback when only one session exists (warn caller).
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["label"] if df["label"].nunique() > 1 else None,
    )
    return (
        train_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
        "stratified_single_session_fallback",
    )


def prepare_processed_dataset(
    raw_dir: Path = DATASET_RAW_DIR,
    processed_dir: Path = DATASET_PROCESSED_DIR,
    test_size: float = 0.25,
    random_state: int = 42,
) -> dict[str, object]:
    """Run the full preprocessing pipeline and save train.csv / test.csv."""
    raw_df = load_raw_dataset(raw_dir)
    cleaned_df = clean_dataset(raw_df)
    train_df, test_df, split_strategy = split_train_test(
        cleaned_df, test_size=test_size, random_state=random_state
    )

    processed_dir.mkdir(parents=True, exist_ok=True)
    train_path = processed_dir / "train.csv"
    test_path = processed_dir / "test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    return {
        "total_samples": len(cleaned_df),
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "split_strategy": split_strategy,
        "train_path": str(train_path),
        "test_path": str(test_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Preprocess raw gesture CSVs.")
    parser.add_argument("--raw-dir", type=Path, default=DATASET_RAW_DIR)
    parser.add_argument("--processed-dir", type=Path, default=DATASET_PROCESSED_DIR)
    parser.add_argument("--test-size", type=float, default=0.25)
    args = parser.parse_args()

    summary = prepare_processed_dataset(
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        test_size=args.test_size,
    )
    print(f"[HoloEscape] Preprocessing complete: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
