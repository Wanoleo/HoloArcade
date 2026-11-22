"""Model Evaluation Report loader, HTML exporter, and interactive screen state."""

from __future__ import annotations

import base64
import csv
from dataclasses import dataclass, field
import html
import json
from pathlib import Path
import time

from src.config import (
    DATASET_PROCESSED_DIR,
    DATASET_RAW_DIR,
    MODELS_DIR,
    PROJECT_ROOT,
    RESULTS_METRICS_DIR,
    RESULTS_PLOTS_DIR,
)
from src.game_logic import _is_confirm_gesture, _point_in_box


REPORT_DWELL_SECONDS = 0.65

# Normalized (0..1) UI hitboxes for the Report screen
TAB_BOXES: tuple[tuple[str, str, tuple[float, float, float, float]], ...] = (
    ("tab_overview", "1. OVERVIEW", (0.04, 0.115, 0.21, 0.065)),
    ("tab_classes", "2. PER-CLASS & CM", (0.265, 0.115, 0.23, 0.065)),
    ("tab_runtime", "3. RUNTIME & PIPELINE", (0.51, 0.115, 0.23, 0.065)),
)
EXPORT_HTML_BOX = (0.755, 0.115, 0.115, 0.065)
BACK_HUB_BOX = (0.88, 0.115, 0.085, 0.065)

MODEL_SELECT_BOXES: tuple[tuple[str, str, tuple[float, float, float, float]], ...] = (
    ("model_random_forest", "RANDOM FOREST", (0.04, 0.20, 0.17, 0.055)),
    ("model_svm", "SVM (RBF)", (0.22, 0.20, 0.14, 0.055)),
    ("model_knn", "KNN", (0.37, 0.20, 0.11, 0.055)),
)


@dataclass
class ReportData:
    """Aggregated ML evaluation, dataset, and runtime benchmark data."""

    selected_model: str = "random_forest"
    classes: list[str] = field(default_factory=list)
    comparison_rows: list[dict[str, str]] = field(default_factory=list)
    classification_reports: dict[str, dict] = field(default_factory=dict)
    runtime_benchmark: dict[str, object] = field(default_factory=dict)
    total_raw_samples: int = 0
    session_counts: dict[str, int] = field(default_factory=dict)
    handedness_counts: dict[str, int] = field(default_factory=dict)
    train_samples: int = 0
    test_samples: int = 0


def _count_csv_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        return max(0, sum(1 for _ in handle) - 1)


def load_report_data(
    metrics_dir: Path = RESULTS_METRICS_DIR,
    models_dir: Path = MODELS_DIR,
    raw_dir: Path = DATASET_RAW_DIR,
    processed_dir: Path = DATASET_PROCESSED_DIR,
) -> ReportData:
    """Load saved evaluation metrics, metadata, benchmark, and dataset counts."""
    data = ReportData()

    meta_path = models_dir / "model_metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        data.selected_model = str(meta.get("selected_model", "random_forest"))
        data.classes = list(meta.get("classes", []))

    comp_path = metrics_dir / "model_comparison.csv"
    if comp_path.exists():
        with comp_path.open("r", encoding="utf-8", newline="") as handle:
            data.comparison_rows = list(csv.DictReader(handle))

    rep_path = metrics_dir / "classification_reports.json"
    if rep_path.exists():
        data.classification_reports = json.loads(rep_path.read_text(encoding="utf-8"))

    bench_path = metrics_dir / "runtime_benchmark.json"
    if bench_path.exists():
        data.runtime_benchmark = json.loads(bench_path.read_text(encoding="utf-8"))

    if raw_dir.exists():
        for csv_path in sorted(raw_dir.glob("*.csv")):
            with csv_path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    data.total_raw_samples += 1
                    sid = row.get("session_id", "unknown")
                    hd = row.get("handedness", "Unknown")
                    data.session_counts[sid] = data.session_counts.get(sid, 0) + 1
                    data.handedness_counts[hd] = data.handedness_counts.get(hd, 0) + 1

    data.train_samples = _count_csv_rows(processed_dir / "train.csv")
    data.test_samples = _count_csv_rows(processed_dir / "test.csv")
    return data


def _img_data_uri(path: Path) -> str:
    if not path.exists():
        return ""
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def export_html_report(
    data: ReportData,
    output_path: Path = PROJECT_ROOT / "results" / "evaluation_report.html",
    plots_dir: Path = RESULTS_PLOTS_DIR,
) -> Path:
    """Export a self-contained HTML report with all metrics and embedded plots."""
    comp_rows_html = "\n".join(
        f"<tr><td>{html.escape(r.get('model', '').upper())}</td>"
        f"<td>{html.escape(r.get('accuracy', ''))}</td>"
        f"<td>{html.escape(r.get('precision_macro', ''))}</td>"
        f"<td>{html.escape(r.get('recall_macro', ''))}</td>"
        f"<td>{html.escape(r.get('f1_macro', ''))}</td>"
        f"<td>{html.escape(r.get('latency_ms_per_sample', ''))}</td></tr>"
        for r in data.comparison_rows
    )

    plots_html = ""
    for title, fname in (
        ("Model Comparison", "model_comparison.png"),
        ("Confusion Matrix — Random Forest", "cm_random_forest.png"),
        ("Confusion Matrix — SVM", "cm_svm.png"),
        ("Confusion Matrix — KNN", "cm_knn.png"),
    ):
        uri = _img_data_uri(plots_dir / fname)
        if uri:
            plots_html += f'<div class="card"><h3>{title}</h3><img src="{uri}" alt="{title}"/></div>\n'

    bench_items = "".join(
        f"<li><strong>{html.escape(str(k))}:</strong> {html.escape(str(v))}</li>"
        for k, v in data.runtime_benchmark.items()
    )

    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>HoloArcade — ML Model Evaluation Report</title>
<style>
  body {{ font-family: Segoe UI, Arial, sans-serif; background: #0b101a; color: #e8f0fa; margin: 28px; }}
  h1, h2 {{ color: #6ee7ff; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 16px; }}
  .card {{ background: #131c2e; border: 1px solid #2b4263; border-radius: 10px; padding: 16px; margin-bottom: 16px; }}
  table {{ width: 100%; border-collapse: collapse; margin-top: 8px; }}
  th, td {{ border-bottom: 1px solid #243652; padding: 8px 10px; text-align: left; }}
  th {{ color: #8be9ff; }}
  img {{ max-width: 100%; border-radius: 8px; background: #fff; }}
</style>
</head>
<body>
  <h1>HoloArcade — ML Model Evaluation Report</h1>
  <div class="card">
    <p><strong>Selected Best Model:</strong> {html.escape(data.selected_model.upper())} &nbsp;|&nbsp;
       <strong>Total Raw Samples:</strong> {data.total_raw_samples} &nbsp;|&nbsp;
       <strong>Train / Test Split:</strong> {data.train_samples} / {data.test_samples}</p>
  </div>
  <div class="card">
    <h2>Model Comparison</h2>
    <table>
      <tr><th>Model</th><th>Accuracy</th><th>Macro Precision</th><th>Macro Recall</th><th>Macro F1</th><th>Latency (ms)</th></tr>
      {comp_rows_html}
    </table>
  </div>
  <div class="card">
    <h2>Runtime Benchmark</h2>
    <ul>{bench_items}</ul>
  </div>
  <div class="grid">{plots_html}</div>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(doc, encoding="utf-8")
    return output_path


@dataclass
class ReportState:
    """Interactive UI state for the in-app Report & Evaluation dashboard."""

    active_tab: str = "overview"  # "overview" | "classes" | "runtime"
    selected_model: str = "random_forest"
    hovered_target: str | None = None
    dwell_target: str | None = None
    dwell_progress: float = 0.0
    dwell_locked: bool = False
    return_to_hub: bool = False
    status_message: str = "Pinch / Thumbs Up / Hold Point on tabs | M or Fist = Back to Hub"
    last_update_time: float = field(default_factory=time.monotonic)


class HoloReportScreen:
    """Drives the gesture-interactive Model Evaluation Report screen."""

    def __init__(self) -> None:
        self.data = load_report_data()
        self.state = ReportState(selected_model=self.data.selected_model or "random_forest")

    def refresh(self) -> None:
        """Reload metrics from disk and reset navigation state."""
        self.data = load_report_data()
        self.state = ReportState(selected_model=self.data.selected_model or "random_forest")

    def _hit_button(self, pointer_xy: tuple[float, float] | None) -> str | None:
        if pointer_xy is None:
            return None
        for btn_id, _, box in TAB_BOXES:
            if _point_in_box(pointer_xy, box):
                return btn_id
        if _point_in_box(pointer_xy, EXPORT_HTML_BOX):
            return "export_html"
        if _point_in_box(pointer_xy, BACK_HUB_BOX):
            return "back_hub"
        if self.state.active_tab == "classes":
            for btn_id, _, box in MODEL_SELECT_BOXES:
                if _point_in_box(pointer_xy, box):
                    return btn_id
        return None

    def _activate(self, btn_id: str) -> None:
        st = self.state
        if btn_id == "tab_overview":
            st.active_tab = "overview"
            st.status_message = "Viewing Model Comparison & Dataset Overview."
        elif btn_id == "tab_classes":
            st.active_tab = "classes"
            st.status_message = "Viewing Per-Class Metrics & Confusion Matrices."
        elif btn_id == "tab_runtime":
            st.active_tab = "runtime"
            st.status_message = "Viewing Runtime Benchmark & Feature Pipeline."
        elif btn_id.startswith("model_"):
            st.selected_model = btn_id[6:]
            st.status_message = f"Showing {st.selected_model.upper()} metrics & confusion matrix."
        elif btn_id == "export_html":
            out = export_html_report(self.data)
            st.status_message = f"Exported HTML report to {out.name}!"
        elif btn_id == "back_hub":
            st.return_to_hub = True

    def update(
        self,
        triggered_gesture: str | None,
        held_gesture: str | None,
        pointer_xy: tuple[float, float] | None,
        gesture_status: str,
        now: float | None = None,
        secondary_triggered: str | None = None,
    ) -> ReportState:
        """Update tab/button hover, dwell-to-select, and gesture triggers."""
        current_time = time.monotonic() if now is None else now
        st = self.state
        dt = max(0.0, min(0.1, current_time - st.last_update_time))
        st.last_update_time = current_time
        st.return_to_hub = False

        if triggered_gesture == "fist":
            st.return_to_hub = True
            return st

        if gesture_status == "no_hand" or pointer_xy is None:
            st.hovered_target = None
            st.dwell_target = None
            st.dwell_progress = 0.0
            st.dwell_locked = False
            return st

        hovered = self._hit_button(pointer_xy)
        st.hovered_target = hovered

        eff_trigger = triggered_gesture
        if eff_trigger is None and _is_confirm_gesture(secondary_triggered):
            eff_trigger = secondary_triggered

        if hovered is not None and held_gesture == "pointing":
            if hovered != st.dwell_target:
                st.dwell_target = hovered
                st.dwell_progress = 0.0
                st.dwell_locked = False
            elif not st.dwell_locked:
                st.dwell_progress = min(1.0, st.dwell_progress + dt / REPORT_DWELL_SECONDS)
                if st.dwell_progress >= 1.0:
                    st.dwell_locked = True
                    st.dwell_progress = 0.0
                    if eff_trigger is None:
                        eff_trigger = "pinch"
        else:
            st.dwell_target = None
            st.dwell_progress = 0.0
            st.dwell_locked = False

        if hovered is not None and _is_confirm_gesture(eff_trigger):
            self._activate(hovered)

        return st
