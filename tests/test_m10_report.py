"""Tests for the Model Evaluation Report screen, HTML export, and Hub button."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import pygame

from src.gesture_recognition import GesturePrediction
from src.renderer import GameRenderer
from src.report_view import (
    BACK_HUB_BOX,
    MODEL_SELECT_BOXES,
    TAB_BOXES,
    HoloReportScreen,
    export_html_report,
    load_report_data,
)


def _box_center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    return (box[0] + box[2] / 2.0, box[1] + box[3] / 2.0)


class TestModelReportView(unittest.TestCase):
    def test_load_report_data_and_export_html(self) -> None:
        data = load_report_data()
        self.assertIn(data.selected_model, ("random_forest", "svm", "knn"))
        self.assertGreater(data.total_raw_samples, 0)
        self.assertEqual(len(data.comparison_rows), 3)

        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "report.html"
            written = export_html_report(data, output_path=out_path)
            self.assertTrue(written.exists())
            content = written.read_text(encoding="utf-8")
            self.assertIn("HoloArcade", content)
            self.assertIn("RANDOM_FOREST", content)

    def test_report_navigation_tabs_models_dwell_and_back(self) -> None:
        screen_ctrl = HoloReportScreen()
        self.assertEqual(screen_ctrl.state.active_tab, "overview")

        # Pinch on Per-Class & CM tab
        classes_tab_pt = _box_center(TAB_BOXES[1][2])
        screen_ctrl.update("pinch", "pinch", classes_tab_pt, "stable", now=1.0)
        self.assertEqual(screen_ctrl.state.active_tab, "classes")

        # Dwell-to-select on SVM model button
        svm_pt = _box_center(MODEL_SELECT_BOXES[1][2])
        for step in range(10):
            screen_ctrl.update(None, "pointing", svm_pt, "stable", now=2.0 + step * 0.08)
        self.assertEqual(screen_ctrl.state.selected_model, "svm")

        # Back button sets return_to_hub
        screen_ctrl.update("pinch", "pinch", _box_center(BACK_HUB_BOX), "stable", now=3.5)
        self.assertTrue(screen_ctrl.state.return_to_hub)

    def test_render_report_all_tabs_and_hub_button(self) -> None:
        pygame.init()
        try:
            screen = pygame.Surface((960, 540))
            cam = pygame.Surface((640, 480))
            renderer = GameRenderer()
            pred = GesturePrediction("stable", "pointing", "pointing", None, 0.94, 1.0)
            rep = HoloReportScreen()

            renderer.render_hub(screen, cam, pred, (0.5, 0.82), "report", 30.0, "best_model.joblib")
            for tab in ("overview", "classes", "runtime"):
                rep.state.active_tab = tab
                renderer.render_report(
                    screen,
                    cam,
                    rep.data,
                    rep.state,
                    pred,
                    (0.3, 0.14),
                    30.0,
                    "best_model.joblib",
                )
        finally:
            pygame.quit()


if __name__ == "__main__":
    unittest.main()
