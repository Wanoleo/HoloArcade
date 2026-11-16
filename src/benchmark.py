"""Runtime FPS and latency benchmark for HoloEscape (Milestone 8)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import pygame

from src.camera import WebcamCapture, bgr_to_pygame_surface
from src.config import RESULTS_METRICS_DIR
from src.game_logic import EscapeRoomGame
from src.gesture_recognition import DEFAULT_MODEL_PATH, GestureRecognizer
from src.hand_detection import HandTracker, draw_hand_landmarks
from src.renderer import GameRenderer


def run_benchmark(
    num_frames: int = 60,
    camera_index: int = 0,
    model_path: Path = DEFAULT_MODEL_PATH,
    output_path: Path = RESULTS_METRICS_DIR / "runtime_benchmark.json",
) -> dict[str, float | int | str]:
    """Measure live end-to-end FPS, hand-tracking time, and ML inference latency."""
    recognizer = GestureRecognizer(model_path=model_path)
    game = EscapeRoomGame()

    frame_times_ms: list[float] = []
    tracker_times_ms: list[float] = []
    ml_times_ms: list[float] = []

    with WebcamCapture(camera_index=camera_index) as cam, HandTracker() as tracker:
        first_frame = cam.read_frame()
        height, width = first_frame.shape[:2]

        pygame.init()
        try:
            screen = pygame.display.set_mode((width, height))
            pygame.display.set_caption("HoloEscape — Runtime Benchmark")
            renderer = GameRenderer()

            bench_start = time.perf_counter()
            for idx in range(num_frames):
                pygame.event.pump()
                t0 = time.perf_counter()
                frame = first_frame if idx == 0 else cam.read_frame()

                t_track_start = time.perf_counter()
                detection = tracker.detect(frame)
                tracker_times_ms.append((time.perf_counter() - t_track_start) * 1000.0)

                pred = recognizer.predict(detection)
                if detection.detected:
                    ml_times_ms.append(pred.latency_ms)

                state = game.update(
                    triggered_gesture=pred.triggered_gesture,
                    held_gesture=pred.smoothed_label,
                    pointer_xy=detection.pointer_xy,
                    gesture_status=pred.status,
                )
                drawn = draw_hand_landmarks(frame, detection)
                surf = bgr_to_pygame_surface(drawn)
                renderer.render(
                    screen=screen,
                    camera_surface=surf,
                    state=state,
                    prediction=pred,
                    pointer_xy=detection.pointer_xy,
                    fps=0.0,
                    model_name=model_path.name,
                )
                pygame.display.flip()
                frame_times_ms.append((time.perf_counter() - t0) * 1000.0)

            total_seconds = time.perf_counter() - bench_start
        finally:
            pygame.quit()

    avg_frame_ms = sum(frame_times_ms) / max(len(frame_times_ms), 1)
    avg_tracker_ms = sum(tracker_times_ms) / max(len(tracker_times_ms), 1)
    avg_ml_ms = sum(ml_times_ms) / max(len(ml_times_ms), 1) if ml_times_ms else 0.0
    measured_fps = num_frames / max(total_seconds, 1e-6)

    summary: dict[str, float | int | str] = {
        "frames_measured": num_frames,
        "resolution": f"{width}x{height}",
        "model_file": model_path.name,
        "measured_fps": round(measured_fps, 2),
        "avg_frame_ms": round(avg_frame_ms, 2),
        "avg_hand_tracker_ms": round(avg_tracker_ms, 2),
        "avg_ml_inference_ms": round(avg_ml_ms, 4),
        "hand_detected_frames": len(ml_times_ms),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark HoloEscape runtime FPS and latency.")
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()

    summary = run_benchmark(
        num_frames=args.frames,
        camera_index=args.camera,
        model_path=args.model,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
