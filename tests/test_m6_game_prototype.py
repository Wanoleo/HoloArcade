"""Tests for room-exploration EscapeRoomGame, Narrator, and GameRenderer."""

from __future__ import annotations

import unittest

import pygame

from src.game_logic import (
    CHEST_PAD_BOXES,
    EXIT_DOOR_POS,
    EXIT_PAD_BOXES,
    HOLO_CORE_START,
    PEDESTAL_POS,
    RESTART_BOX,
    SAFE_POS,
    EscapeRoomGame,
)
from src.gesture_recognition import GesturePrediction
from src.narrator import Narrator
from src.renderer import GameRenderer


def _box_center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    return (box[0] + box[2] / 2.0, box[1] + box[3] / 2.0)


class TestMilestone6GameLogicAndRenderer(unittest.TestCase):
    def test_narrator_tracks_lines_and_replay_when_disabled(self) -> None:
        narrator = Narrator(enabled=False)
        narrator.speak("Three past midnight, seven alarms, two guards.")
        self.assertEqual(narrator.current_line, "Three past midnight, seven alarms, two guards.")
        narrator.replay()
        self.assertEqual(len(narrator.history), 1)

    def test_menu_open_and_close_with_open_palm_and_fist(self) -> None:
        game = EscapeRoomGame()
        game.update("open_palm", "open_palm", (0.5, 0.5), "stable", now=1.0)
        self.assertTrue(game.state.menu_open)
        game.update("fist", "fist", (0.5, 0.5), "stable", now=2.0)
        self.assertFalse(game.state.menu_open)

    def test_player_walks_and_completes_narration_escape_room(self) -> None:
        game = EscapeRoomGame(move_speed=2.0)
        pending, _ = game.consume_narration_events()
        self.assertIsNone(pending)
        self.assertIn("dr. vance", game.state.npc_line.lower())

        # Aim directly at Locked Chest in First-Person POV + thumbs_up
        game.update("thumbs_up", "thumbs_up", SAFE_POS, "stable", now=2.0)
        self.assertTrue(game.state.safe_panel_open)

        # Enter 3 - 7 - 2 on the 0-9 Chest Keypad (indices 2, 6, 1 in CHEST_PAD_VALUES)
        for pad_idx in (2, 6, 1):
            game.update(
                "thumbs_up",
                "thumbs_up",
                _box_center(CHEST_PAD_BOXES[pad_idx]),
                "stable",
                now=3.0 + pad_idx,
            )
        self.assertEqual(game.state.stage, "mystery_object")

        # Pinch Holo Core in POV, drag to Right Portal, release
        game.update("pinch", "pinch", HOLO_CORE_START, "stable", now=15.0)
        self.assertTrue(game.state.artifact_grabbed)
        game.update(None, "pinch", PEDESTAL_POS, "stable", now=16.0)
        game.update(None, "open_palm", PEDESTAL_POS, "stable", now=16.2)
        self.assertEqual(game.state.stage, "final_exit")

        # Aim at Back Exit Terminal in POV + thumbs_up, enter 7 - 2 - 4
        game.update("thumbs_up", "thumbs_up", EXIT_DOOR_POS, "stable", now=20.0)
        self.assertTrue(game.state.exit_panel_open)

        for pad_idx in (2, 0, 1):
            game.update(
                "thumbs_up",
                "thumbs_up",
                _box_center(EXIT_PAD_BOXES[pad_idx]),
                "stable",
                now=21.0 + pad_idx,
            )
        self.assertEqual(game.state.stage, "completed")

        # Restart via gesture
        game.update("thumbs_up", "thumbs_up", _box_center(RESTART_BOX), "stable", now=25.0)
        self.assertEqual(game.state.stage, "chest_lock")
        self.assertEqual(game.state.dials, [])

    def test_renderer_draws_all_stages_and_pip_without_error(self) -> None:
        pygame.init()
        try:
            screen = pygame.Surface((960, 540))
            cam_surf = pygame.Surface((640, 480))
            renderer = GameRenderer()
            pred = GesturePrediction("stable", "pointing", "pointing", None, 0.9, 1.2)
            game = EscapeRoomGame()

            for stage in ("chest_lock", "mystery_object", "final_exit", "completed"):
                game.state.stage = stage
                game.state.safe_panel_open = stage == "chest_lock"
                game.state.exit_panel_open = stage == "final_exit"
                game.state.menu_open = stage == "mystery_object"
                renderer.render(
                    screen=screen,
                    camera_surface=cam_surf,
                    state=game.state,
                    prediction=pred,
                    pointer_xy=(0.5, 0.5),
                    fps=30.0,
                    model_name="best_model.joblib",
                )
        finally:
            pygame.quit()

    def test_pov_movement_via_pointing_and_move_dir_updates_camera(self) -> None:
        from src.game_logic import pov_camera_params

        game = EscapeRoomGame()
        start_pos = tuple(game.state.player_pos)
        start_zoom, _, _ = pov_camera_params(game.state, (0.5, 0.5))

        # Point forward-left to walk in First-Person POV
        game.update(None, "pointing", (0.25, 0.50), "stable", now=10.0)
        game.update(None, "pointing", (0.25, 0.50), "stable", now=10.1)
        self.assertTrue(game.state.is_moving)
        self.assertLess(game.state.player_pos[0], start_pos[0])
        self.assertLess(game.state.player_pos[1], start_pos[1])

        moved_zoom, moved_off_x, _ = pov_camera_params(game.state, (0.5, 0.5))
        self.assertGreater(moved_zoom, start_zoom)
        self.assertGreater(moved_off_x, 0.0)

        # Keyboard move_dir works even when no_hand is active
        pos_before_key = tuple(game.state.player_pos)
        game.update(None, None, None, "no_hand", now=10.2, move_dir=(1.0, 0.0))
        self.assertTrue(game.state.is_moving)
        self.assertGreater(game.state.player_pos[0], pos_before_key[0])

    def test_npc_dialogue_interaction_and_close(self) -> None:
        from src.game_logic import NPC_DIALOG_BOXES, NPC_POS

        game = EscapeRoomGame()
        # Aim at Dr. Vance + thumbs_up opens dialogue
        game.update("thumbs_up", "thumbs_up", NPC_POS, "stable", now=1.0)
        self.assertTrue(game.state.npc_dialog_open)

        # Select first dialogue option ("Ask about Cyber Vault Code")
        game.update("thumbs_up", "thumbs_up", _box_center(NPC_DIALOG_BOXES[0]), "stable", now=2.0)
        self.assertEqual(game.state.npc_topic, "safe")
        self.assertIn("3 - 7 - 2", game.state.npc_line)

        # Fist closes dialogue
        game.update("fist", "fist", _box_center(NPC_DIALOG_BOXES[0]), "stable", now=3.0)
        self.assertFalse(game.state.npc_dialog_open)


if __name__ == "__main__":
    unittest.main()
