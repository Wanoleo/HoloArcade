"""Gesture-controlled Air-Drawing & Virtual Text Board (HoloBoard)."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import time

from src.game_logic import _is_confirm_gesture, _point_in_box


# Normalized (0..1) layout regions
TOOLBAR_BOX = (0.04, 0.115, 0.92, 0.065)
CANVAS_BOX = (0.04, 0.19, 0.92, 0.40)
TEXT_DISPLAY_BOX = (0.04, 0.605, 0.92, 0.075)
KEYBOARD_BOX = (0.04, 0.695, 0.92, 0.265)

BOARD_DWELL_SECONDS = 0.65

PEN_COLORS: tuple[tuple[str, tuple[int, int, int]], ...] = (
    ("cyan", (90, 240, 255)),
    ("emerald", (105, 250, 165)),
    ("amber", (255, 205, 80)),
    ("rose", (255, 110, 155)),
    ("white", (245, 248, 255)),
)

# Toolbar buttons: (id, label, box)
TOOLBAR_BUTTONS: tuple[tuple[str, str, tuple[float, float, float, float]], ...] = (
    ("color_0", "CYAN", (0.05, 0.122, 0.065, 0.050)),
    ("color_1", "GREEN", (0.122, 0.122, 0.065, 0.050)),
    ("color_2", "AMBER", (0.194, 0.122, 0.065, 0.050)),
    ("color_3", "ROSE", (0.266, 0.122, 0.065, 0.050)),
    ("color_4", "WHITE", (0.338, 0.122, 0.065, 0.050)),
    ("undo", "UNDO", (0.435, 0.122, 0.095, 0.050)),
    ("clear_draw", "CLEAR DRAW", (0.540, 0.122, 0.125, 0.050)),
    ("speak", "SPEAK", (0.675, 0.122, 0.105, 0.050)),
    ("clear_text", "CLEAR TEXT", (0.790, 0.122, 0.130, 0.050)),
)


def _build_keyboard_buttons() -> tuple[tuple[str, str, tuple[float, float, float, float]], ...]:
    """Generate normalized hitboxes for a 4-row QWERTY + digits + action keyboard."""
    rows: tuple[tuple[str, ...], ...] = (
        ("1", "2", "3", "4", "5", "6", "7", "8", "9", "0"),
        ("Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"),
        ("A", "S", "D", "F", "G", "H", "J", "K", "L", "BACK"),
        ("Z", "X", "C", "V", "B", "N", "M", "SPACE", "CLR"),
    )
    buttons: list[tuple[str, str, tuple[float, float, float, float]]] = []
    kb_x, kb_y, kb_w, kb_h = KEYBOARD_BOX
    row_h = (kb_h - 0.015) / len(rows)
    gap_x = 0.006
    gap_y = 0.008

    for r_idx, row in enumerate(rows):
        # Compute relative weights so SPACE is wider
        weights = [2.0 if k == "SPACE" else (1.3 if k in ("BACK", "CLR") else 1.0) for k in row]
        total_w = sum(weights)
        usable_w = kb_w - gap_x * (len(row) - 1)
        cur_x = kb_x
        by = kb_y + r_idx * row_h + gap_y * 0.5
        bh = row_h - gap_y
        for key, weight in zip(row, weights):
            bw = usable_w * (weight / total_w)
            buttons.append((f"key_{key}", key, (cur_x, by, bw, bh)))
            cur_x += bw + gap_x

    return tuple(buttons)


KEYBOARD_BUTTONS = _build_keyboard_buttons()


@dataclass
class BoardStroke:
    """One continuous air-drawn polyline stroke."""

    color: tuple[int, int, int]
    points: list[tuple[float, float]] = field(default_factory=list)
    width: int = 4


@dataclass
class BoardState:
    """Complete mutable state for HoloBoard (air-drawing + text board)."""

    strokes: list[BoardStroke] = field(default_factory=list)
    active_stroke: BoardStroke | None = None
    color_index: int = 0
    brush_width: int = 4

    text: str = ""
    pending_speech: str | None = None

    hovered_target: str | None = None
    dwell_target: str | None = None
    dwell_progress: float = 0.0
    dwell_locked: bool = False

    feedback_message: str = "Pinch inside Canvas to Air-Draw | Pinch or Hold Point on Keys to Type"
    feedback_kind: str = "info"
    last_update_time: float = field(default_factory=time.monotonic)


class HoloBoardGame:
    """Decoupled gesture-driven Air-Drawing & Virtual Keyboard state machine."""

    def __init__(self) -> None:
        self.state = BoardState()

    def restart(self) -> None:
        """Clear canvas and text board."""
        self.state = BoardState()

    def consume_speech(self) -> str | None:
        """Return and clear any pending TTS text requested by the SPEAK button."""
        txt = self.state.pending_speech
        self.state.pending_speech = None
        return txt

    def _hit_button(self, pointer_xy: tuple[float, float] | None) -> str | None:
        if pointer_xy is None:
            return None
        for btn_id, _, box in TOOLBAR_BUTTONS:
            if _point_in_box(pointer_xy, box):
                return btn_id
        for btn_id, _, box in KEYBOARD_BUTTONS:
            if _point_in_box(pointer_xy, box):
                return btn_id
        return None

    def _activate_button(self, btn_id: str) -> None:
        st = self.state
        if btn_id.startswith("color_"):
            idx = int(btn_id.split("_")[1])
            if 0 <= idx < len(PEN_COLORS):
                st.color_index = idx
                name, _ = PEN_COLORS[idx]
                st.feedback_message = f"Pen color set to {name.upper()}."
                st.feedback_kind = "info"
        elif btn_id == "undo":
            if st.strokes:
                st.strokes.pop()
                st.feedback_message = "Undid last stroke."
            else:
                st.feedback_message = "Nothing to undo."
            st.feedback_kind = "info"
        elif btn_id == "clear_draw":
            st.strokes.clear()
            st.active_stroke = None
            st.feedback_message = "Canvas cleared."
            st.feedback_kind = "info"
        elif btn_id == "speak":
            if st.text.strip():
                st.pending_speech = st.text.strip()
                st.feedback_message = f"Speaking: '{st.pending_speech}'"
                st.feedback_kind = "success"
            else:
                st.feedback_message = "Type some text first, then press SPEAK."
                st.feedback_kind = "warning"
        elif btn_id == "clear_text":
            st.text = ""
            st.feedback_message = "Text cleared."
            st.feedback_kind = "info"
        elif btn_id.startswith("key_"):
            key = btn_id[4:]
            if key == "BACK":
                st.text = st.text[:-1]
            elif key == "CLR":
                st.text = ""
            elif key == "SPACE":
                if len(st.text) < 80:
                    st.text += " "
            else:
                if len(st.text) < 80:
                    st.text += key
            st.feedback_message = f"Text: {st.text or '(empty)'}"
            st.feedback_kind = "info"

    def _finish_active_stroke(self) -> None:
        st = self.state
        if st.active_stroke is not None:
            if len(st.active_stroke.points) >= 1:
                st.strokes.append(st.active_stroke)
            st.active_stroke = None

    def update(
        self,
        triggered_gesture: str | None,
        held_gesture: str | None,
        pointer_xy: tuple[float, float] | None,
        gesture_status: str,
        now: float | None = None,
        secondary_triggered: str | None = None,
        secondary_held: str | None = None,
    ) -> BoardState:
        """Update air-drawing strokes and virtual keyboard state for one frame."""
        current_time = time.monotonic() if now is None else now
        st = self.state
        dt = max(0.0, min(0.1, current_time - st.last_update_time))
        st.last_update_time = current_time

        if gesture_status == "no_hand" or pointer_xy is None:
            self._finish_active_stroke()
            st.hovered_target = None
            st.dwell_target = None
            st.dwell_progress = 0.0
            st.dwell_locked = False
            return st

        in_canvas = _point_in_box(pointer_xy, CANVAS_BOX)
        drawing_held = held_gesture == "pinch" or (held_gesture == "pointing" and secondary_held == "pinch")

        if in_canvas and drawing_held:
            st.hovered_target = None
            st.dwell_target = None
            st.dwell_progress = 0.0
            st.dwell_locked = False
            if st.active_stroke is None:
                _, col = PEN_COLORS[st.color_index]
                st.active_stroke = BoardStroke(color=col, points=[pointer_xy], width=st.brush_width)
            else:
                last_pt = st.active_stroke.points[-1]
                if math.hypot(pointer_xy[0] - last_pt[0], pointer_xy[1] - last_pt[1]) >= 0.0025:
                    st.active_stroke.points.append(pointer_xy)
            st.feedback_message = "Air-Drawing... (release Pinch to lift pen)"
            st.feedback_kind = "info"
            return st

        # Pen lifted
        self._finish_active_stroke()

        # Check toolbar / keyboard hover
        hovered = self._hit_button(pointer_xy)
        st.hovered_target = hovered

        eff_trigger = triggered_gesture
        if eff_trigger is None and _is_confirm_gesture(secondary_triggered):
            eff_trigger = secondary_triggered

        # Dwell-to-select on buttons when holding Point steadily
        if hovered is not None and held_gesture == "pointing":
            if hovered != st.dwell_target:
                st.dwell_target = hovered
                st.dwell_progress = 0.0
                st.dwell_locked = False
            elif not st.dwell_locked:
                st.dwell_progress = min(1.0, st.dwell_progress + dt / BOARD_DWELL_SECONDS)
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
            self._activate_button(hovered)

        return st
