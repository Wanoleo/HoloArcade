"""Webcam capture utilities using OpenCV and Pygame surface conversion."""

from __future__ import annotations

import cv2
import numpy as np
import pygame


class CameraError(RuntimeError):
    """Raised when the webcam cannot be opened or a frame cannot be read."""


class WebcamCapture:
    """Simple, safe wrapper around OpenCV VideoCapture."""

    def __init__(
        self,
        camera_index: int = 0,
        width: int = 960,
        height: int = 540,
        mirror: bool = True,
    ) -> None:
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.mirror = mirror
        self._cap: cv2.VideoCapture | None = None

    def open(self) -> None:
        """Open the webcam and configure capture resolution."""
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            cap.release()
            raise CameraError(
                f"Could not open webcam at index {self.camera_index}. "
                "Check that a camera is connected and not in use by another app, "
                "and that camera permissions are enabled in Windows Settings."
            )

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap = cap

    def read_frame(self) -> np.ndarray:
        """Read a single BGR frame from the webcam."""
        if self._cap is None or not self._cap.isOpened():
            raise CameraError("Webcam is not open. Call open() before reading frames.")

        ok, frame = self._cap.read()
        if not ok or frame is None:
            raise CameraError("Failed to read a frame from the webcam.")

        if self.mirror:
            frame = cv2.flip(frame, 1)
        return frame

    def release(self) -> None:
        """Release the webcam resource cleanly."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "WebcamCapture":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()


def bgr_to_pygame_surface(frame_bgr: np.ndarray) -> pygame.Surface:
    """Convert an OpenCV BGR frame to a Pygame Surface."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    height, width = frame_rgb.shape[:2]
    return pygame.image.frombuffer(frame_rgb.tobytes(), (width, height), "RGB")
