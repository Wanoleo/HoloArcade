"""Tests for HoloBoard (Air-Drawing & Gesture Text Board) and 3-card Hub rendering."""

from __future__ import annotations

import unittest

import pygame

from src.board_game import (
    CANVAS_BOX,
    KEYBOARD_BUTTONS,
    TOOLBAR_BUTTONS,
    HoloBoardGame,
)
from src.gesture_recognition import GesturePrediction
from src.renderer import GameRenderer


def _box_center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    return (box[0] + box[2] / 2.0, box[1] + box[3] / 2.0)


def _find_btn(btn_id: str) -> tuple[float, float]:
    for bid, _, box in (*TOOLBAR_BUTTONS, *KEYBOARD_BUTTONS):
        if bid == btn_id:
            return _box_center(box)
    raise KeyError(btn_id)


class TestHoloBoard(unittest.TestCase):
    def test_air_drawing_strokes_undo_and_clear(self) -> None:
        board = HoloBoardGame()
        cx, cy = _box_center(CANVAS_BOX)

        # Pinch inside canvas starts stroke and records points
        board.update("pinch", "pinch", (cx - 0.05, cy), "stable", now=1.0)
        board.update(None, "pinch", (cx + 0.05, cy), "stable", now=1.1)
        self.assertIsNotNone(board.state.active_stroke)
        self.assertEqual(len(board.state.active_stroke.points), 2)

        # Releasing pinch finishes stroke
        board.update(None, "pointing", (cx + 0.05, cy), "stable", now=1.2)
        self.assertIsNone(board.state.active_stroke)
        self.assertEqual(len(board.state.strokes), 1)

        # Two-hand draw: Hand 1 points, Hand 2 pinches
        board.update(
            None,
            "pointing",
            (cx, cy - 0.05),
            "stable",
            now=2.0,
            secondary_held="pinch",
        )
        board.update(
            None,
            "pointing",
            (cx, cy + 0.05),
            "stable",
            now=2.1,
            secondary_held="pinch",
        )
        board.update(None, "pointing", (cx, cy + 0.05), "stable", now=2.2)
        self.assertEqual(len(board.state.strokes), 2)

        # Undo removes last stroke
        board.update("pinch", "pinch", _find_btn("undo"), "stable", now=3.0)
        self.assertEqual(len(board.state.strokes), 1)

        # Clear draw removes remaining strokes
        board.update("pinch", "pinch", _find_btn("clear_draw"), "stable", now=3.5)
        self.assertEqual(len(board.state.strokes), 0)

    def test_virtual_keyboard_typing_dwell_and_speak(self) -> None:
        board = HoloBoardGame()
        # Pinch types 'H', 'I'
        board.update("pinch", "pinch", _find_btn("key_H"), "stable", now=1.0)
        board.update("pinch", "pinch", _find_btn("key_I"), "stable", now=1.5)
        self.assertEqual(board.state.text, "HI")

        # Dwell-to-select types '!'? Let's dwell on SPACE then '1'
        space_pos = _find_btn("key_SPACE")
        for step in range(10):
            board.update(None, "pointing", space_pos, "stable", now=2.0 + step * 0.08)
        self.assertEqual(board.state.text, "HI ")

        # Two-hand confirm types '1'
        board.update(
            None,
            "pointing",
            _find_btn("key_1"),
            "stable",
            now=3.0,
            secondary_triggered="pinch",
            secondary_held="pinch",
        )
        self.assertEqual(board.state.text, "HI 1")

        # SPEAK queues TTS
        board.update("pinch", "pinch", _find_btn("speak"), "stable", now=3.5)
        self.assertEqual(board.consume_speech(), "HI 1")
        self.assertIsNone(board.consume_speech())

        # BACK removes last char
        board.update("pinch", "pinch", _find_btn("key_BACK"), "stable", now=4.0)
        self.assertEqual(board.state.text, "HI ")

    def test_render_board_and_three_card_hub(self) -> None:
        pygame.init()
        try:
            screen = pygame.Surface((960, 540))
            cam = pygame.Surface((640, 480))
            renderer = GameRenderer()
            pred = GesturePrediction("stable", "pointing", "pointing", None, 0.92, 1.0)
            board = HoloBoardGame()
            board.update("pinch", "pinch", _box_center(CANVAS_BOX), "stable", now=1.0)
            board.update("pinch", "pinch", _find_btn("key_A"), "stable", now=1.2)

            renderer.render_hub(screen, cam, pred, (0.8, 0.4), "board", 30.0, "best_model.joblib")
            renderer.render_board(screen, cam, board.state, pred, (0.5, 0.35), 30.0, "best_model.joblib")
        finally:
            pygame.quit()


if __name__ == "__main__":
    unittest.main()
