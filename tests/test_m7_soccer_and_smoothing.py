"""Tests for PointerSmoother, 4v4 HoloSoccerGame (with pass), and Hub/Soccer rendering."""

from __future__ import annotations

import unittest

import pygame

from src.gesture_recognition import GesturePrediction, PointerSmoother
from src.renderer import GameRenderer
from src.soccer_game import PITCH_RIGHT, HoloSoccerGame


class TestSoccerAndPointerSmoothing(unittest.TestCase):
    def test_pointer_smoother_dampens_jumps_and_resets_on_none(self) -> None:
        smoother = PointerSmoother(alpha=0.25, deadzone=0.01)
        p1 = smoother.update((0.2, 0.2))
        self.assertEqual(p1, (0.2, 0.2))

        p_jit = smoother.update((0.203, 0.203))
        self.assertEqual(p_jit, (0.2, 0.2))

        p_move = smoother.update((0.8, 0.2))
        self.assertIsNotNone(p_move)
        self.assertAlmostEqual(p_move[0], 0.35, places=4)
        self.assertIsNone(smoother.update(None))

    def test_soccer_has_three_teammates_and_three_opponents_and_pass_switches_player(self) -> None:
        game = HoloSoccerGame(move_speed=0.5)
        self.assertEqual(len(game.state.teammates), 4)  # Active + 3 teammates
        self.assertEqual(len(game.state.opponents), 3)  # 3 opponents (+ keeper)

        # Place ball at controlled player's feet and aim pass toward teammate 1
        game.state.teammates[0].pos = [0.40, 0.50]
        game.state.ball_pos = [0.43, 0.50]
        mate1_pos = tuple(game.state.teammates[1].pos)

        game.update("two_fingers", "two_fingers", mate1_pos, "stable", now=1.0)
        self.assertEqual(game.state.controlled_idx, 1)
        self.assertIsNotNone(game.state.pass_line)
        self.assertGreater(abs(game.state.ball_vel[0]) + abs(game.state.ball_vel[1]), 0.3)

    def test_soccer_movement_fist_shot_and_goal(self) -> None:
        game = HoloSoccerGame(move_speed=0.5)
        # Move opponents away from shot lane
        for opp in game.state.opponents:
            opp.pos = [0.30, 0.20]
        game.state.teammates[0].pos = [0.76, 0.50]
        game.state.ball_pos = [0.79, 0.50]
        game.state.keeper_pos = [0.80, 0.43]

        game.update("fist", "fist", (0.86, 0.50), "stable", now=1.0)
        self.assertGreater(game.state.ball_vel[0], 0.5)

        for step in range(15):
            game.update(None, None, None, "no_hand", now=1.05 + step * 0.05)

        self.assertEqual(game.state.score_player, 1)
        self.assertGreater(game.state.goal_banner_timer, 0.0)
        self.assertLess(game.state.ball_pos[0], PITCH_RIGHT)

    def test_hub_and_soccer_render_without_error(self) -> None:
        pygame.init()
        try:
            screen = pygame.Surface((960, 540))
            cam = pygame.Surface((640, 480))
            renderer = GameRenderer()
            pred = GesturePrediction("stable", "pointing", "pointing", None, 0.9, 1.0)
            soccer = HoloSoccerGame()

            renderer.render_hub(screen, cam, pred, (0.25, 0.45), "escape", 30.0, "best_model.joblib")
            renderer.render_soccer(screen, cam, soccer.state, pred, (0.5, 0.5), 30.0, "best_model.joblib")
        finally:
            pygame.quit()


if __name__ == "__main__":
    unittest.main()
