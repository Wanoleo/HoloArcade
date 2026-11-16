"""Independent room-exploration & narration-driven puzzle state machine for HoloEscape.

Decoupled from MediaPipe, OpenCV, scikit-learn, and TTS so gestures and audio
backends can be swapped without rewriting puzzle rules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import time


# First-Person POV hotspot boxes in normalized (0..1) screen coordinates
TERMINAL_BOX = (0.08, 0.30, 0.16, 0.42)
NPC_BOX = (0.24, 0.36, 0.14, 0.36)
SAFE_BOX = (0.43, 0.56, 0.22, 0.24)
EXIT_DOOR_BOX = (0.40, 0.36, 0.20, 0.22)

TERMINAL_POS = (0.16, 0.52)
NPC_POS = (0.31, 0.56)
SAFE_POS = (0.54, 0.68)
HOLO_CORE_START = (0.38, 0.68)
PEDESTAL_POS = (0.85, 0.52)
EXIT_DOOR_POS = (0.50, 0.47)
INTERACT_RADIUS = 0.14

# Overlay hitboxes when Safe (0-9 keypad), Exit Door, or NPC Dialogue panel is open
NPC_DIALOG_OPTIONS = (
    ("safe", "1. Ask about Cyber Vault Code", "Dr. Vance: 'Three power cells failed, seven alarms rang, two guards fled. Try 3 - 7 - 2!'"),
    ("core", "2. Ask about Holo Core", "Dr. Vance: 'Once the Vault opens, Pinch the glowing Holo Core and drag it into the Right Portal.'"),
    ("exit", "3. Ask about Exit Airlock", "Dr. Vance: 'After the portal aligns, enter seven pulses, two sparks, and four chimes (7 - 2 - 4).'"),
)
NPC_DIALOG_BOXES = (
    (0.24, 0.42, 0.52, 0.085),
    (0.24, 0.52, 0.52, 0.085),
    (0.24, 0.62, 0.52, 0.085),
)

CHEST_PAD_VALUES = (1, 2, 3, 4, 5, 6, 7, 8, 9, 0)
CHEST_PAD_BOXES = tuple(
    (0.22 + (i % 5) * 0.114, 0.40 + (i // 5) * 0.125, 0.100, 0.105)
    for i in range(10)
)
CHEST_CLEAR_BOX = (0.39, 0.67, 0.22, 0.08)

EXIT_PAD_VALUES = (2, 4, 7, 9)
EXIT_PAD_BOXES = (
    (0.28, 0.38, 0.09, 0.14),
    (0.40, 0.38, 0.09, 0.14),
    (0.52, 0.38, 0.09, 0.14),
    (0.64, 0.38, 0.09, 0.14),
)
RESTART_BOX = (0.39, 0.62, 0.22, 0.10)

NARRATION_STAGE_1 = (
    "Sector four log: Power failed at three past midnight. "
    "Seven alarms echoed down the corridor, and only two guards escaped. "
    "Walk to the wall safe and enter the numbers you heard."
)
NARRATION_STAGE_2 = (
    "The safe opens, powering the room scanner. "
    "Pinch the glowing Holo Core on the floor and carry it to the Scanner Pedestal."
)
NARRATION_STAGE_3 = (
    "Scanner aligned! Final transmission: seven pulses, two sparks, and four chimes "
    "will unlock the North Exit Door."
)
NARRATION_WIN = (
    "Airlock unsealed! You decoded the logs and escaped Sector four!"
)


def _point_in_box(
    point: tuple[float, float] | None,
    box: tuple[float, float, float, float],
) -> bool:
    if point is None:
        return False
    px, py = point
    bx, by, bw, bh = box
    return bx <= px <= (bx + bw) and by <= py <= (by + bh)


def _distance(a: tuple[float, float] | list[float], b: tuple[float, float] | list[float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def pov_camera_params(
    state: "GameState",
    pointer_xy: tuple[float, float] | None = None,
) -> tuple[float, float, float]:
    """Return (zoom, offset_x, offset_y) in normalized coords for First-Person POV."""
    depth_t = max(0.0, min(1.0, (0.86 - state.player_pos[1]) / 0.40))
    zoom = 1.05 + depth_t * 0.50
    look_pan_x = 0.0
    look_pan_y = 0.0
    if pointer_xy is not None and not (state.safe_panel_open or state.exit_panel_open):
        look_pan_x = (0.5 - pointer_xy[0]) * 0.045
        look_pan_y = (0.5 - pointer_xy[1]) * 0.03
    strafe_x = (0.50 - state.player_pos[0]) * 0.24
    strafe_y = (0.72 - state.player_pos[1]) * 0.08
    return zoom, strafe_x + look_pan_x, strafe_y + look_pan_y


def pov_project(
    world_xy: tuple[float, float] | list[float],
    state: "GameState",
    pointer_xy: tuple[float, float] | None = None,
) -> tuple[float, float]:
    """Project room-space normalized (x, y) to First-Person POV screen-space (x, y)."""
    zoom, off_x, off_y = pov_camera_params(state, pointer_xy)
    sx = 0.5 + (world_xy[0] - 0.5) * zoom + off_x
    sy = 0.5 + (world_xy[1] - 0.5) * zoom + off_y
    return sx, sy


def pov_unproject(
    screen_xy: tuple[float, float],
    state: "GameState",
    pointer_xy: tuple[float, float] | None = None,
) -> tuple[float, float]:
    """Convert First-Person POV screen-space (x, y) back to room-space (x, y)."""
    zoom, off_x, off_y = pov_camera_params(state, pointer_xy)
    wx = 0.5 + (screen_xy[0] - 0.5 - off_x) / zoom
    wy = 0.5 + (screen_xy[1] - 0.5 - off_y) / zoom
    return wx, wy


@dataclass
class GameState:
    """Complete mutable state for the HoloEscape room."""

    stage: str = "chest_lock"  # "chest_lock" | "mystery_object" | "final_exit" | "completed"
    menu_open: bool = False

    # Movable player inside the room (First-Person 3D POV position & yaw)
    player_pos: list[float] = field(default_factory=lambda: [0.50, 0.78])
    player_yaw: float = -math.pi / 2.0
    player_facing: str = "up"
    is_moving: bool = False
    nearby_station: str | None = None  # "terminal" | "safe" | "holo_core" | "pedestal" | "exit_door"

    # Stage 1: Narration-hidden Safe Code (3 - 7 - 2)
    dials: list[int] = field(default_factory=list)
    target_code: tuple[int, int, int] = (3, 7, 2)
    safe_panel_open: bool = False
    hovered_target: str | None = None
    selected_target: str | None = None

    # Stage 2: Mystery Holo-Core & Pedestal
    artifact_pos: list[float] = field(default_factory=lambda: [HOLO_CORE_START[0], HOLO_CORE_START[1]])
    artifact_grabbed: bool = False
    artifact_angle: int = 0

    # Stage 3: Final Exit Door (7 - 2 - 4)
    exit_panel_open: bool = False
    exit_target: tuple[int, int, int] = (7, 2, 4)
    exit_entered: list[int] = field(default_factory=list)

    # NPC interactive dialogue (Dr. Vance)
    npc_dialog_open: bool = False
    npc_topic: str = "greeting"
    npc_line: str = "Dr. Vance: 'Hey Agent! Aim at me + Thumbs Up (or press T) to talk for clues!'"

    # Narration & feedback (kept for compatibility, driven by NPC dialogue)
    narration_subtitle: str = "Dr. Vance: 'Talk to me for the Cyber Vault code, Holo Core, and Exit Airlock clues!'"
    pending_narration: str | None = None
    replay_requested: bool = False
    feedback_message: str = "Aim at Dr. Vance + Thumbs Up to talk, or open the Cyber Vault!"
    feedback_kind: str = "info"  # "info" | "success" | "warning" | "error"
    start_time: float = field(default_factory=time.monotonic)
    last_update_time: float = field(default_factory=time.monotonic)
    completion_seconds: float | None = None


class EscapeRoomGame:
    """Drives player movement, station interaction, and narration-based puzzles."""

    def __init__(self, move_speed: float = 0.38) -> None:
        self.move_speed = move_speed
        self.state = GameState()

    def restart(self) -> None:
        """Reset the level cleanly without restarting the application."""
        self.state = GameState()
        self.state.feedback_message = "Level restarted! Listen to the log and walk to the Safe."
        self.state.feedback_kind = "info"

    def consume_narration_events(self) -> tuple[str | None, bool]:
        """Return and clear (pending_narration, replay_requested) for the audio narrator."""
        pending = self.state.pending_narration
        replay = self.state.replay_requested
        self.state.pending_narration = None
        self.state.replay_requested = False
        return pending, replay

    def update(
        self,
        triggered_gesture: str | None,
        held_gesture: str | None,
        pointer_xy: tuple[float, float] | None,
        gesture_status: str,
        now: float | None = None,
        move_dir: tuple[float, float] = (0.0, 0.0),
        turn_dir: float = 0.0,
        interact_trigger: bool = False,
    ) -> GameState:
        """Advance First-Person 3D POV movement and puzzle state by one frame."""
        current_time = time.monotonic() if now is None else now
        st = self.state
        dt = max(0.0, min(0.1, current_time - st.last_update_time))
        st.last_update_time = current_time
        st.is_moving = False

        if gesture_status == "uncertain":
            st.feedback_message = "Aiming... hold a clear gesture to interact"
            st.feedback_kind = "info"

        if gesture_status == "no_hand":
            st.hovered_target = None
            if not (st.menu_open or st.safe_panel_open or st.exit_panel_open or st.stage == "completed"):
                if turn_dir != 0.0:
                    st.player_yaw += turn_dir * 2.1 * max(dt, 1.0 / 30.0)
                if move_dir != (0.0, 0.0):
                    self._move_player_by_dir(move_dir, dt)
                self._update_pov_hover(None)
                if interact_trigger:
                    triggered_gesture = "thumbs_up"
                    if st.stage == "chest_lock":
                        self._update_stage_1(triggered_gesture, None)
                    elif st.stage == "final_exit":
                        self._update_stage_3(triggered_gesture, None, current_time)
            return st

        # Global menu & narration replay controls
        if triggered_gesture == "open_palm":
            st.menu_open = True
            st.feedback_message = "Controls & Log opened (Fist to close)."
            st.feedback_kind = "info"
            return st

        if triggered_gesture == "fist":
            if st.menu_open:
                st.menu_open = False
                st.feedback_message = "Menu closed."
                st.feedback_kind = "info"
                return st
            if st.npc_dialog_open:
                st.npc_dialog_open = False
                st.hovered_target = None
                st.feedback_message = "Ended conversation with Dr. Vance."
                st.feedback_kind = "info"
                return st
            if st.safe_panel_open:
                if st.dials:
                    st.dials.clear()
                    st.feedback_message = "Cleared Chest code."
                else:
                    st.safe_panel_open = False
                    st.feedback_message = "Closed Chest keypad."
                st.feedback_kind = "info"
                return st
            if st.exit_panel_open:
                if st.exit_entered:
                    st.exit_entered.clear()
                    st.feedback_message = "Cleared exit keypad."
                else:
                    st.exit_panel_open = False
                    st.feedback_message = "Closed Exit keypad."
                st.feedback_kind = "info"
                return st
            if st.artifact_grabbed:
                st.artifact_grabbed = False
                st.feedback_message = "Dropped the Holo Core."
                st.feedback_kind = "info"
                return st

        if triggered_gesture == "two_fingers":
            if st.stage == "mystery_object" and st.artifact_grabbed:
                st.artifact_angle = (st.artifact_angle + 90) % 360
                st.feedback_message = f"Rotated Holo Core to {st.artifact_angle} deg."
            else:
                st.replay_requested = True
                st.feedback_message = "Replaying narration..."
            st.feedback_kind = "info"

        if st.menu_open:
            return st

        # First-Person POV turning & movement (WASD / Arrows / Point gesture)
        if not (st.safe_panel_open or st.exit_panel_open or st.npc_dialog_open or st.stage == "completed"):
            if move_dir != (0.0, 0.0):
                self._move_player_by_dir(move_dir, dt)
            elif held_gesture == "pointing" and pointer_xy is not None:
                target_world = pov_unproject(pointer_xy, st, pointer_xy)
                self._move_player_toward(target_world, dt)

        self._update_pov_hover(pointer_xy)
        eff_trigger = "thumbs_up" if interact_trigger else triggered_gesture

        if st.npc_dialog_open:
            self._update_npc_dialog(eff_trigger, pointer_xy)
            return st

        if eff_trigger == "thumbs_up" and st.nearby_station == "npc":
            st.npc_dialog_open = True
            st.selected_target = "npc_0"
            st.feedback_message = "Talking to Dr. Vance (Point + Thumbs Up on a question, Fist to close)."
            st.feedback_kind = "info"
            return st

        if st.stage == "chest_lock":
            self._update_stage_1(eff_trigger, pointer_xy)
        elif st.stage == "mystery_object":
            self._update_stage_2(held_gesture, pointer_xy)
        elif st.stage == "final_exit":
            self._update_stage_3(eff_trigger, pointer_xy, current_time)
        elif st.stage == "completed":
            self._update_completed(eff_trigger, pointer_xy)

        return st

    def _update_npc_dialog(
        self,
        triggered_gesture: str | None,
        pointer_xy: tuple[float, float] | None,
    ) -> None:
        st = self.state
        hovered: str | None = None
        for idx, box in enumerate(NPC_DIALOG_BOXES):
            if _point_in_box(pointer_xy, box):
                hovered = f"npc_{idx}"
                break
        st.hovered_target = hovered
        if hovered is not None:
            st.selected_target = hovered
        active = hovered or st.selected_target
        if triggered_gesture == "thumbs_up" and active is not None and active.startswith("npc_"):
            idx = int(active.split("_")[1])
            topic, _, reply = NPC_DIALOG_OPTIONS[idx]
            st.npc_topic = topic
            st.npc_line = reply
            st.narration_subtitle = reply
            st.feedback_message = reply
            st.feedback_kind = "success"

    def _update_pov_hover(self, pointer_xy: tuple[float, float] | None) -> None:
        st = self.state
        if st.safe_panel_open or st.exit_panel_open or st.npc_dialog_open or st.stage == "completed":
            return

        if pointer_xy is not None:
            npc_screen = pov_project(NPC_POS, st, pointer_xy)
            term_screen = pov_project(TERMINAL_POS, st, pointer_xy)
            safe_screen = pov_project(SAFE_POS, st, pointer_xy)
            exit_screen = pov_project(EXIT_DOOR_POS, st, pointer_xy)
            if _point_in_box(pointer_xy, NPC_BOX) or _distance(pointer_xy, npc_screen) <= 0.13:
                st.nearby_station = "npc"
                return
            if _point_in_box(pointer_xy, TERMINAL_BOX) or _distance(pointer_xy, term_screen) <= 0.14:
                st.nearby_station = "terminal"
                return
            if st.stage == "chest_lock" and (
                _point_in_box(pointer_xy, SAFE_BOX) or _distance(pointer_xy, safe_screen) <= 0.14
            ):
                st.nearby_station = "safe"
                return
            if st.stage == "final_exit" and (
                _point_in_box(pointer_xy, EXIT_DOOR_BOX) or _distance(pointer_xy, exit_screen) <= 0.13
            ):
                st.nearby_station = "exit_door"
                return

        # Proximity fallback when walking near a station
        stations = [("npc", NPC_POS), ("terminal", TERMINAL_POS)]
        if st.stage == "chest_lock":
            stations.append(("safe", SAFE_POS))
        elif st.stage == "final_exit":
            stations.append(("exit_door", EXIT_DOOR_POS))

        nearest: str | None = None
        best_d = 0.20
        for name, pos in stations:
            d = _distance(st.player_pos, pos)
            if d <= best_d:
                best_d = d
                nearest = name
        st.nearby_station = nearest

    def _move_player_by_dir(self, move_dir: tuple[float, float], dt: float) -> None:
        dx, dy = move_dir
        mag = math.hypot(dx, dy)
        if mag < 1e-6:
            return
        st = self.state
        step = self.move_speed * max(dt, 1.0 / 30.0)
        nx = st.player_pos[0] + (dx / mag) * step
        ny = st.player_pos[1] + (dy / mag) * step
        self._apply_player_move(nx, ny, dx, dy)

    def _move_player_toward(self, target_xy: tuple[float, float], dt: float) -> None:
        st = self.state
        tx = max(0.14, min(0.86, target_xy[0]))
        ty = max(0.46, min(0.86, target_xy[1]))
        dx = tx - st.player_pos[0]
        dy = ty - st.player_pos[1]
        dist = math.hypot(dx, dy)
        if dist < 0.02:
            return

        step = min(dist, self.move_speed * max(dt, 1.0 / 30.0))
        self._apply_player_move(
            st.player_pos[0] + (dx / dist) * step,
            st.player_pos[1] + (dy / dist) * step,
            dx,
            dy,
        )

    def _apply_player_move(self, nx: float, ny: float, dx: float, dy: float) -> None:
        st = self.state
        clamped_x = max(0.14, min(0.86, nx))
        clamped_y = max(0.46, min(0.86, ny))
        if abs(clamped_x - st.player_pos[0]) < 1e-5 and abs(clamped_y - st.player_pos[1]) < 1e-5:
            return
        st.player_pos[0] = clamped_x
        st.player_pos[1] = clamped_y
        st.is_moving = True

        if abs(dx) > abs(dy):
            st.player_facing = "right" if dx > 0 else "left"
        else:
            st.player_facing = "down" if dy > 0 else "up"

    def _update_nearby_station(self) -> None:
        st = self.state
        stations = [
            ("terminal", TERMINAL_POS),
            ("safe", SAFE_POS),
            ("pedestal", PEDESTAL_POS),
            ("exit_door", EXIT_DOOR_POS),
        ]
        if st.stage == "mystery_object" and not st.artifact_grabbed:
            stations.append(("holo_core", tuple(st.artifact_pos)))

        nearest_name: str | None = None
        nearest_dist = INTERACT_RADIUS
        for name, pos in stations:
            d = _distance(st.player_pos, pos)
            if d <= nearest_dist:
                nearest_dist = d
                nearest_name = name
        st.nearby_station = nearest_name

    def _update_stage_1(
        self,
        triggered_gesture: str | None,
        pointer_xy: tuple[float, float] | None,
    ) -> None:
        st = self.state
        if not st.safe_panel_open:
            if triggered_gesture == "thumbs_up":
                if st.nearby_station == "terminal":
                    st.replay_requested = True
                    st.feedback_message = "Terminal: Replaying Sector 4 audio log..."
                    st.feedback_kind = "info"
                elif st.nearby_station == "safe":
                    st.safe_panel_open = True
                    st.selected_target = "pad_2"  # Default highlight on '3'
                    st.feedback_message = "Point at digits 0-9 + Thumbs Up to enter the 3-digit code!"
                    st.feedback_kind = "info"
                else:
                    st.feedback_message = "Aim at the Left Cipher Wall or Center Chest + Thumbs Up!"
                    st.feedback_kind = "warning"
            return

        # Chest 0-9 keypad is open
        hovered: str | None = None
        if _point_in_box(pointer_xy, CHEST_CLEAR_BOX):
            hovered = "clear"
        else:
            for idx, box in enumerate(CHEST_PAD_BOXES):
                if _point_in_box(pointer_xy, box):
                    hovered = f"pad_{idx}"
                    break
        st.hovered_target = hovered
        if hovered is not None:
            st.selected_target = hovered
        active_target = hovered or st.selected_target

        if triggered_gesture == "thumbs_up":
            if active_target == "clear":
                st.dials.clear()
                st.feedback_message = "Cleared Chest code."
                st.feedback_kind = "info"
            elif active_target is not None and active_target.startswith("pad_"):
                idx = int(active_target.split("_")[1])
                digit = CHEST_PAD_VALUES[idx]
                st.dials.append(digit)
                st.feedback_message = f"Entered {digit} ({len(st.dials)}/3)."
                st.feedback_kind = "info"

                if len(st.dials) == len(st.target_code):
                    if tuple(st.dials) == st.target_code:
                        st.stage = "mystery_object"
                        st.safe_panel_open = False
                        st.hovered_target = None
                        st.selected_target = None
                        st.narration_subtitle = NARRATION_STAGE_2
                        st.pending_narration = NARRATION_STAGE_2
                        st.feedback_message = "Chest unlocked! Walk to the Holo Core and Pinch to grab it."
                        st.feedback_kind = "success"
                    else:
                        entered_str = "-".join(str(d) for d in st.dials)
                        st.dials.clear()
                        st.feedback_message = (
                            f"Wrong code {entered_str}! Cleared — use Two Fingers to replay narration."
                        )
                        st.feedback_kind = "error"

    def _update_stage_2(
        self,
        held_gesture: str | None,
        pointer_xy: tuple[float, float] | None,
    ) -> None:
        st = self.state
        if held_gesture == "pinch" and pointer_xy is not None:
            core_screen = pov_project(st.artifact_pos, st, pointer_xy)
            if st.artifact_grabbed or _distance(pointer_xy, core_screen) <= INTERACT_RADIUS:
                st.artifact_grabbed = True
                wx, wy = pov_unproject(pointer_xy, st, pointer_xy)
                st.artifact_pos[0] = max(0.10, min(0.92, wx))
                st.artifact_pos[1] = max(0.20, min(0.84, wy))
                st.feedback_message = "Holding Holo Core! Drag it into the Right Portal and release."
                st.feedback_kind = "info"
        else:
            if st.artifact_grabbed:
                st.artifact_grabbed = False

        if not st.artifact_grabbed and _distance(st.artifact_pos, PEDESTAL_POS) <= INTERACT_RADIUS:
            st.stage = "final_exit"
            st.narration_subtitle = NARRATION_STAGE_3
            st.pending_narration = NARRATION_STAGE_3
            st.feedback_message = "Portal aligned! Aim at the Back Exit Terminal + Thumbs Up."
            st.feedback_kind = "success"

    def _update_stage_3(
        self,
        triggered_gesture: str | None,
        pointer_xy: tuple[float, float] | None,
        current_time: float,
    ) -> None:
        st = self.state
        if not st.exit_panel_open:
            if triggered_gesture == "thumbs_up":
                if st.nearby_station == "exit_door":
                    st.exit_panel_open = True
                    st.selected_target = "exit_0"
                    st.feedback_message = "Exit Keypad open! Enter the 3 numbers from the narration."
                    st.feedback_kind = "info"
                else:
                    st.feedback_message = "Aim at the Back Wall Exit Terminal (under the clock) + Thumbs Up."
                    st.feedback_kind = "warning"
            return

        hovered: str | None = None
        for idx, box in enumerate(EXIT_PAD_BOXES):
            if _point_in_box(pointer_xy, box):
                hovered = f"exit_{idx}"
                break
        st.hovered_target = hovered
        if hovered is not None:
            st.selected_target = hovered
        active_target = hovered or st.selected_target

        if triggered_gesture == "thumbs_up" and active_target is not None and active_target.startswith("exit_"):
            idx = int(active_target.split("_")[1])
            digit = EXIT_PAD_VALUES[idx]
            st.exit_entered.append(digit)
            st.feedback_message = f"Entered {digit} ({len(st.exit_entered)}/3)."
            st.feedback_kind = "info"

            if len(st.exit_entered) == len(st.exit_target):
                if tuple(st.exit_entered) == st.exit_target:
                    st.stage = "completed"
                    st.exit_panel_open = False
                    st.hovered_target = None
                    st.selected_target = None
                    st.completion_seconds = round(current_time - st.start_time, 1)
                    st.narration_subtitle = NARRATION_WIN
                    st.pending_narration = NARRATION_WIN
                    st.feedback_message = "DOOR UNLOCKED! You escaped Sector 4!"
                    st.feedback_kind = "success"
                else:
                    st.exit_entered.clear()
                    st.feedback_message = "Wrong sequence! Two Fingers to replay narration."
                    st.feedback_kind = "error"

    def _update_completed(
        self,
        triggered_gesture: str | None,
        pointer_xy: tuple[float, float] | None,
    ) -> None:
        st = self.state
        hovered = "restart" if _point_in_box(pointer_xy, RESTART_BOX) else None
        st.hovered_target = hovered
        if hovered is not None:
            st.selected_target = hovered
        active_target = hovered or st.selected_target
        if triggered_gesture == "thumbs_up" and active_target == "restart":
            self.restart()
