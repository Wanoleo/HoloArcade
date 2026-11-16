"""Developer-facing gesture dataset collection tool (Milestone 3).

Captures 21 MediaPipe hand landmarks (63 x, y, z features) and saves labelled
samples to CSV under dataset/raw/ with session and participant metadata so
train/test splits can be grouped by session or person.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
import sys
import time

import numpy as np
import pygame

from src.camera import CameraError, WebcamCapture, bgr_to_pygame_surface
from src.config import DATASET_RAW_DIR, GESTURE_CLASSES, RAW_CSV_COLUMNS
from src.hand_detection import HandDetectionResult, HandTracker, draw_hand_landmarks


WINDOW_TITLE = "HoloEscape — Gesture Dataset Collector (M3)"
TARGET_FPS = 30


def build_sample_row(
    session_id: str,
    participant: str,
    label: str,
    detection: HandDetectionResult,
    timestamp: str | None = None,
) -> list[str | float]:
    """Build one CSV row from a validated HandDetectionResult."""
    if label not in GESTURE_CLASSES:
        raise ValueError(f"Unknown gesture label '{label}'. Expected one of {GESTURE_CLASSES}.")
    if not detection.detected or detection.features is None:
        raise ValueError("Cannot build sample row without a valid hand detection.")

    ts = timestamp or datetime.now().isoformat(timespec="milliseconds")
    feature_values = [round(float(v), 6) for v in detection.features.tolist()]
    return [session_id, participant, ts, detection.handedness, label, *feature_values]


def append_sample_to_csv(csv_path: Path, row: list[str | float]) -> None:
    """Append a single labelled sample row to a raw dataset CSV file."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if write_header:
            writer.writerow(RAW_CSV_COLUMNS)
        writer.writerow(row)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line options for data collection."""
    default_session = datetime.now().strftime("session_%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description="Collect labelled hand gesture samples.")
    parser.add_argument("--participant", default="user1", help="Participant ID (e.g. user1).")
    parser.add_argument("--session", default=default_session, help="Recording session ID.")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV webcam index (default: 0).")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output CSV path (default: dataset/raw/gestures_<session>.csv).",
    )
    parser.add_argument(
        "--sample-interval",
        type=float,
        default=0.1,
        help="Minimum seconds between recorded frames in continuous mode (default: 0.1).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Optional frame limit for automated smoke tests (0 = run until quit).",
    )
    return parser.parse_args(argv)


def run_collector(
    participant: str = "user1",
    session_id: str | None = None,
    camera_index: int = 0,
    output_path: Path | None = None,
    sample_interval: float = 0.1,
    max_frames: int = 0,
) -> int:
    """Run the interactive Pygame gesture collection loop."""
    session = session_id or datetime.now().strftime("session_%Y%m%d_%H%M%S")
    csv_path = output_path or (DATASET_RAW_DIR / f"gestures_{session}.csv")

    active_idx = 0
    recording = False
    last_saved_time = 0.0
    counts = {label: 0 for label in GESTURE_CLASSES}

    try:
        with WebcamCapture(camera_index=camera_index) as cam, HandTracker() as tracker:
            first_frame = cam.read_frame()
            frame_height, frame_width = first_frame.shape[:2]

            pygame.init()
            screen = pygame.display.set_mode((frame_width, frame_height))
            pygame.display.set_caption(WINDOW_TITLE)
            font = pygame.font.SysFont("consolas", 17)
            clock = pygame.time.Clock()

            frame_count = 0
            running = True
            current_frame = first_frame

            while running:
                single_capture = False
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key in (pygame.K_ESCAPE, pygame.K_q):
                            running = False
                        elif event.key == pygame.K_SPACE:
                            recording = not recording
                        elif event.key == pygame.K_s:
                            single_capture = True
                        elif pygame.K_1 <= event.key <= pygame.K_6:
                            active_idx = event.key - pygame.K_1
                            recording = False

                if frame_count > 0:
                    current_frame = cam.read_frame()

                detection = tracker.detect(current_frame)
                active_label = GESTURE_CLASSES[active_idx]

                now = time.monotonic()
                should_record = single_capture or (
                    recording and (now - last_saved_time) >= sample_interval
                )
                if should_record and detection.detected:
                    row = build_sample_row(session, participant, active_label, detection)
                    append_sample_to_csv(csv_path, row)
                    counts[active_label] += 1
                    last_saved_time = now

                display_frame = draw_hand_landmarks(current_frame, detection)
                screen.blit(bgr_to_pygame_surface(display_frame), (0, 0))

                # Top status bar
                hud_bg = pygame.Surface((frame_width, 82), pygame.SRCALPHA)
                hud_bg.fill((12, 18, 28, 200))
                screen.blit(hud_bg, (0, 0))

                mode_text = "RECORDING (SPACE to pause)" if recording else "PAUSED (SPACE=record, S=single)"
                mode_color = (255, 110, 110) if recording else (180, 220, 255)
                line1 = font.render(
                    f"Session: {session} | User: {participant} | File: {csv_path.name}",
                    True,
                    (230, 240, 250),
                )
                line2 = font.render(
                    f"Active [{active_idx + 1}]: {active_label} | {mode_text}",
                    True,
                    mode_color,
                )
                hand_msg = (
                    f"Hand: {detection.handedness} ({detection.confidence:.2f})"
                    if detection.detected
                    else "No hand detected — sample will not be saved"
                )
                line3 = font.render(
                    f"{hand_msg} | Keys 1-6: class | Q/ESC: quit",
                    True,
                    (130, 240, 175) if detection.detected else (255, 200, 110),
                )
                screen.blit(line1, (12, 6))
                screen.blit(line2, (12, 30))
                screen.blit(line3, (12, 54))

                # Bottom class counter bar
                footer_bg = pygame.Surface((frame_width, 34), pygame.SRCALPHA)
                footer_bg.fill((12, 18, 28, 200))
                screen.blit(footer_bg, (0, frame_height - 34))
                summary = " | ".join(
                    f"{idx + 1}:{name}={counts[name]}"
                    for idx, name in enumerate(GESTURE_CLASSES)
                )
                footer_text = font.render(summary, True, (220, 230, 240))
                screen.blit(footer_text, (10, frame_height - 26))

                pygame.display.flip()
                clock.tick(TARGET_FPS)

                frame_count += 1
                if max_frames > 0 and frame_count >= max_frames:
                    running = False

    except (CameraError, FileNotFoundError) as exc:
        print(f"[HoloEscape] Collector startup error: {exc}", file=sys.stderr)
        return 1
    finally:
        pygame.quit()

    total = sum(counts.values())
    print(f"[HoloEscape] Saved {total} samples to {csv_path}")
    return 0


def main() -> int:
    args = parse_args()
    return run_collector(
        participant=args.participant,
        session_id=args.session,
        camera_index=args.camera,
        output_path=args.output,
        sample_interval=args.sample_interval,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    raise SystemExit(main())
