"""HoloEscape & HoloSoccer entry point.

Launches the Pygame Gesture Game Hub where the player can choose between:
1. HoloEscape — Narration-driven Sector 4 escape room
2. HoloSoccer — Gesture-controlled 1v1 football
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pygame

from src.camera import CameraError, WebcamCapture, bgr_to_pygame_surface
from src.game_logic import EscapeRoomGame, _point_in_box
from src.gesture_recognition import DEFAULT_MODEL_PATH, GestureRecognizer, PointerSmoother
from src.hand_detection import HandTracker, draw_hand_landmarks
from src.narrator import Narrator
from src.renderer import HUB_ESCAPE_BOX, HUB_SOCCER_BOX, GameRenderer
from src.soccer_game import HoloSoccerGame


WINDOW_TITLE = "HoloArcade — Gesture-Controlled Escape Room & Football"
TARGET_FPS = 30


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description="Run HoloArcade.")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV webcam index (default: 0).")
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to saved .joblib gesture model (default: models/best_model.joblib).",
    )
    parser.add_argument(
        "--game",
        choices=("hub", "escape", "soccer"),
        default="hub",
        help="Initial screen (default: hub).",
    )
    parser.add_argument(
        "--fullscreen",
        action="store_true",
        help="Launch the game in fullscreen mode (press F11 anytime to toggle).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Optional frame limit for automated smoke tests (0 = run until quit).",
    )
    return parser.parse_args(argv)


def run_game(
    camera_index: int = 0,
    model_path: Path = DEFAULT_MODEL_PATH,
    max_frames: int = 0,
    initial_mode: str = "hub",
    fullscreen: bool = False,
) -> int:
    """Launch the HoloArcade webcam + ML gesture game loop."""
    try:
        recognizer = GestureRecognizer(model_path=model_path)
        pointer_smoother = PointerSmoother(alpha=0.25, deadzone=0.004)
        escape_game = EscapeRoomGame()
        soccer_game = HoloSoccerGame()
        narrator = Narrator(enabled=(max_frames == 0))
        active_mode = initial_mode
        hub_selected: str | None = None

        if active_mode == "escape":
            pending, _ = escape_game.consume_narration_events()
            if pending:
                narrator.speak(pending)

        with WebcamCapture(camera_index=camera_index) as cam, HandTracker() as tracker:
            first_frame = cam.read_frame()
            frame_height, frame_width = first_frame.shape[:2]

            pygame.init()
            is_fullscreen = fullscreen
            if is_fullscreen:
                screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
            else:
                screen = pygame.display.set_mode((frame_width, frame_height), pygame.RESIZABLE)
            pygame.display.set_caption(WINDOW_TITLE)
            renderer = GameRenderer()
            clock = pygame.time.Clock()

            frame_count = 0
            running = True
            current_frame = first_frame

            while running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key in (pygame.K_ESCAPE, pygame.K_q):
                            running = False
                        elif event.key == pygame.K_m:
                            active_mode = "hub"
                            narrator.stop()
                        elif event.key == pygame.K_1 and active_mode == "hub":
                            active_mode = "escape"
                            escape_game.restart()
                            pending, _ = escape_game.consume_narration_events()
                            if pending:
                                narrator.speak(pending)
                        elif event.key == pygame.K_2 and active_mode == "hub":
                            active_mode = "soccer"
                            soccer_game.restart()
                            narrator.stop()
                        elif event.key == pygame.K_r:
                            recognizer.reset()
                            if active_mode == "escape":
                                escape_game.restart()
                            elif active_mode == "soccer":
                                soccer_game.restart()
                        elif event.key == pygame.K_F11:
                            is_fullscreen = not is_fullscreen
                            if is_fullscreen:
                                screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
                            else:
                                screen = pygame.display.set_mode(
                                    (frame_width, frame_height), pygame.RESIZABLE
                                )
                        elif event.key == pygame.K_n and active_mode == "escape":
                            narrator.replay()

                if frame_count > 0:
                    current_frame = cam.read_frame()

                detection = tracker.detect(current_frame)
                prediction = recognizer.predict(detection)
                freeze_ptr = active_mode in ("hub", "escape") and (
                    prediction.status == "stabilizing" or prediction.smoothed_label == "thumbs_up"
                )
                smooth_pointer = pointer_smoother.update(detection.pointer_xy, freeze=freeze_ptr)

                display_frame = draw_hand_landmarks(current_frame, detection)
                camera_surface = bgr_to_pygame_surface(display_frame)

                if active_mode == "hub":
                    hovered = None
                    if _point_in_box(smooth_pointer, HUB_ESCAPE_BOX):
                        hovered = "escape"
                    elif _point_in_box(smooth_pointer, HUB_SOCCER_BOX):
                        hovered = "soccer"
                    if hovered is not None:
                        hub_selected = hovered
                    target = hovered or hub_selected
                    if prediction.triggered_gesture == "thumbs_up" and target is not None:
                        active_mode = target
                        if active_mode == "escape":
                            escape_game.restart()
                            pending, _ = escape_game.consume_narration_events()
                            if pending:
                                narrator.speak(pending)
                        elif active_mode == "soccer":
                            soccer_game.restart()
                    renderer.render_hub(
                        screen=screen,
                        camera_surface=camera_surface,
                        prediction=prediction,
                        pointer_xy=smooth_pointer,
                        hovered_card=target,
                        fps=clock.get_fps(),
                        model_name=model_path.name,
                    )
                elif active_mode == "escape":
                    keys = pygame.key.get_pressed()
                    mx = float(keys[pygame.K_d]) - float(keys[pygame.K_a])
                    my = float(keys[pygame.K_s] or keys[pygame.K_DOWN]) - float(keys[pygame.K_w] or keys[pygame.K_UP])
                    turn = float(keys[pygame.K_RIGHT] or keys[pygame.K_e]) - float(keys[pygame.K_LEFT] or keys[pygame.K_q])
                    interact = bool(keys[pygame.K_SPACE] or keys[pygame.K_f])
                    if keys[pygame.K_t]:
                        escape_game.state.npc_dialog_open = True
                    state = escape_game.update(
                        triggered_gesture=prediction.triggered_gesture,
                        held_gesture=prediction.smoothed_label,
                        pointer_xy=smooth_pointer,
                        gesture_status=prediction.status,
                        move_dir=(mx, my),
                        turn_dir=turn,
                        interact_trigger=interact,
                    )
                    pending, replay = escape_game.consume_narration_events()
                    if pending:
                        narrator.speak(pending)
                    elif replay:
                        narrator.replay()
                    renderer.render(
                        screen=screen,
                        camera_surface=camera_surface,
                        state=state,
                        prediction=prediction,
                        pointer_xy=smooth_pointer,
                        fps=clock.get_fps(),
                        model_name=model_path.name,
                    )
                elif active_mode == "soccer":
                    keys = pygame.key.get_pressed()
                    mx = float(keys[pygame.K_d] or keys[pygame.K_RIGHT]) - float(keys[pygame.K_a] or keys[pygame.K_LEFT])
                    my = float(keys[pygame.K_s] or keys[pygame.K_DOWN]) - float(keys[pygame.K_w] or keys[pygame.K_UP])
                    kick_trig = "shoot" if keys[pygame.K_SPACE] else ("pass" if keys[pygame.K_e] else None)
                    s_state = soccer_game.update(
                        triggered_gesture=prediction.triggered_gesture,
                        held_gesture=prediction.smoothed_label,
                        pointer_xy=smooth_pointer,
                        gesture_status=prediction.status,
                        move_dir=(mx, my),
                        kick_trigger=kick_trig,
                    )
                    renderer.render_soccer(
                        screen=screen,
                        camera_surface=camera_surface,
                        state=s_state,
                        prediction=prediction,
                        pointer_xy=smooth_pointer,
                        fps=clock.get_fps(),
                        model_name=model_path.name,
                    )

                pygame.display.flip()
                clock.tick(TARGET_FPS)

                frame_count += 1
                if max_frames > 0 and frame_count >= max_frames:
                    running = False

    except (CameraError, FileNotFoundError, RuntimeError) as exc:
        print(f"[HoloEscape] Startup error: {exc}", file=sys.stderr)
        return 1
    finally:
        if "narrator" in locals():
            narrator.stop()
        pygame.quit()

    return 0


# Backwards-compatible alias for earlier milestone tests.
run_preview = run_game


def main() -> int:
    args = parse_args()
    return run_game(
        camera_index=args.camera,
        model_path=args.model,
        max_frames=args.max_frames,
        initial_mode=args.game,
        fullscreen=args.fullscreen,
    )


if __name__ == "__main__":
    raise SystemExit(main())
