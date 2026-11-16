"""Milestone 1 smoke tests for environment imports and webcam preview."""

from __future__ import annotations

import unittest

import numpy as np
import pygame

from main import run_preview
from src.camera import CameraError, WebcamCapture, bgr_to_pygame_surface


class TestMilestone1Foundation(unittest.TestCase):
    def test_required_packages_import(self) -> None:
        import cv2
        import joblib
        import matplotlib
        import mediapipe
        import pandas
        import sklearn

        self.assertTrue(hasattr(cv2, "VideoCapture"))
        self.assertTrue(hasattr(mediapipe, "tasks"))
        self.assertTrue(hasattr(sklearn, "__version__"))
        self.assertTrue(hasattr(pandas, "__version__"))
        self.assertTrue(hasattr(matplotlib, "__version__"))
        self.assertTrue(hasattr(joblib, "__version__"))

    def test_bgr_to_pygame_surface_shape(self) -> None:
        pygame.init()
        try:
            dummy_bgr = np.zeros((120, 160, 3), dtype=np.uint8)
            surface = bgr_to_pygame_surface(dummy_bgr)
            self.assertEqual(surface.get_size(), (160, 120))
        finally:
            pygame.quit()

    def test_invalid_camera_raises_camera_error(self) -> None:
        with self.assertRaises(CameraError):
            with WebcamCapture(camera_index=99):
                pass

    def test_run_preview_handles_missing_camera_gracefully(self) -> None:
        exit_code = run_preview(camera_index=99, max_frames=1)
        self.assertEqual(exit_code, 1)


if __name__ == "__main__":
    unittest.main()
