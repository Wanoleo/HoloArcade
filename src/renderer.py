"""High-craft First-Person POV & Stadium 2D renderer for HoloArcade."""

from __future__ import annotations

import math
from pathlib import Path
import time

import pygame

from src.board_game import (
    CANVAS_BOX,
    KEYBOARD_BUTTONS,
    PEN_COLORS,
    TEXT_DISPLAY_BOX,
    TOOLBAR_BOX,
    TOOLBAR_BUTTONS,
    BoardState,
)
from src.config import PROJECT_ROOT, RESULTS_PLOTS_DIR
from src.report_view import (
    BACK_HUB_BOX,
    EXPORT_HTML_BOX,
    MODEL_SELECT_BOXES,
    TAB_BOXES,
    ReportData,
    ReportState,
)
from src.game_logic import (
    CHEST_CLEAR_BOX,
    CHEST_PAD_BOXES,
    CHEST_PAD_VALUES,
    EXIT_CLEAR_BOX,
    EXIT_DOOR_POS,
    EXIT_PAD_BOXES,
    EXIT_PAD_VALUES,
    NPC_DIALOG_BOXES,
    NPC_DIALOG_OPTIONS,
    NPC_POS,
    PEDESTAL_POS,
    RESTART_BOX,
    SAFE_POS,
    TERMINAL_POS,
    GameState,
    pov_camera_params,
    pov_project,
)
from src.gesture_recognition import GesturePrediction
from src.soccer_game import SoccerState


HUB_ESCAPE_BOX = (0.055, 0.24, 0.27, 0.49)
HUB_SOCCER_BOX = (0.365, 0.24, 0.27, 0.49)
HUB_BOARD_BOX = (0.675, 0.24, 0.27, 0.49)
HUB_REPORT_BOX = (0.29, 0.785, 0.42, 0.085)

ESCAPE_BG_PATH = PROJECT_ROOT / "assets" / "images" / "escape_room_bg.png"
SOCCER_BG_PATH = PROJECT_ROOT / "assets" / "images" / "soccer_pitch_bg.png"


def _to_px(pos: tuple[float, float] | list[float], width: int, height: int) -> tuple[int, int]:
    return int(pos[0] * width), int(pos[1] * height)


def _to_px_rect(box: tuple[float, float, float, float], width: int, height: int) -> pygame.Rect:
    bx, by, bw, bh = box
    return pygame.Rect(int(bx * width), int(by * height), int(bw * width), int(bh * height))


def _draw_glow(
    screen: pygame.Surface,
    center: tuple[int, int],
    radius: int,
    color: tuple[int, int, int],
    alpha: int = 75,
) -> None:
    """Draw a smooth multi-step radial glow without hard concentric rings."""
    steps = 14
    surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
    per_step_alpha = max(2, alpha // steps)
    for i in range(steps, 0, -1):
        r = max(2, int(radius * (i / steps)))
        pygame.draw.circle(surf, (*color, per_step_alpha), (radius, radius), r)
    screen.blit(surf, (center[0] - radius, center[1] - radius))


class GameRenderer:
    """Renders the Game Hub, First-Person POV HoloEscape, and 4v4 HoloSoccer."""

    def __init__(self) -> None:
        self._font_xs = pygame.font.SysFont("segoeui", 14)
        self._font_sm = pygame.font.SysFont("segoeui", 16, bold=True)
        self._font_md = pygame.font.SysFont("segoeui", 22, bold=True)
        self._font_lg = pygame.font.SysFont("segoeui", 32, bold=True)
        self._font_clock = pygame.font.SysFont("consolas", 24, bold=True)
        self._raw_bg_cache: dict[str, pygame.Surface] = {}
        self._bg_cache: dict[tuple[str, int, int], pygame.Surface] = {}

    def _get_background(self, path: Path, width: int, height: int) -> pygame.Surface | None:
        key = (str(path), width, height)
        if key in self._bg_cache:
            return self._bg_cache[key]
        raw = self._raw_bg_cache.get(str(path))
        if raw is None:
            if not path.exists():
                return None
            try:
                raw = pygame.image.load(str(path))
                self._raw_bg_cache[str(path)] = raw
            except pygame.error:
                return None
        if len(self._bg_cache) >= 24:
            self._bg_cache.clear()
        scaled = pygame.transform.smoothscale(raw, (width, height))
        self._bg_cache[key] = scaled
        return scaled

    # ------------------------------------------------------------------
    # Hub Screen (with Thumbnail Previews)
    # ------------------------------------------------------------------
    def render_hub(
        self,
        screen: pygame.Surface,
        camera_surface: pygame.Surface,
        prediction: GesturePrediction,
        pointer_xy: tuple[float, float] | None,
        hovered_card: str | None,
        fps: float,
        model_name: str,
        secondary_prediction: GesturePrediction | None = None,
    ) -> None:
        width, height = screen.get_size()
        bg = self._get_background(ESCAPE_BG_PATH, width, height)
        if bg is not None:
            screen.blit(bg, (0, 0))
            dim = pygame.Surface((width, height), pygame.SRCALPHA)
            dim.fill((6, 10, 18, 175))
            screen.blit(dim, (0, 0))
        else:
            screen.fill((12, 17, 28))

        title = self._font_lg.render("HOLOARCADE", True, (130, 235, 255))
        sub = self._font_sm.render(
            "Point at a card + Pinch or Thumbs Up to launch  (Keys: 1 / 2 / 3 / 4)",
            True,
            (215, 228, 242),
        )
        screen.blit(title, (width // 2 - title.get_width() // 2, 68))
        screen.blit(sub, (width // 2 - sub.get_width() // 2, 112))

        cards = [
            (
                "escape",
                HUB_ESCAPE_BOX,
                ESCAPE_BG_PATH,
                "1. HOLOESCAPE",
                "First-Person POV Escape",
                "Decode clues & unlock Sector 4",
                (85, 220, 255),
            ),
            (
                "soccer",
                HUB_SOCCER_BOX,
                SOCCER_BG_PATH,
                "2. HOLOSOCCER",
                "Gesture Stadium 4v4",
                "Point dribble, 2-Fingers pass, Fist shoot",
                (105, 250, 165),
            ),
            (
                "board",
                HUB_BOARD_BOX,
                None,
                "3. HOLOBOARD",
                "Air-Draw & Text Board",
                "Pinch to draw & type on virtual keys",
                (255, 200, 95),
            ),
        ]

        for key, box, thumb_path, heading, tagline, desc, accent in cards:
            rect = _to_px_rect(box, width, height)
            is_hov = hovered_card == key
            if is_hov:
                _draw_glow(screen, rect.center, max(rect.width, rect.height) // 2 + 24, accent, alpha=80)

            card_surf = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
            card_surf.fill((20, 30, 48, 228) if is_hov else (14, 21, 34, 215))
            screen.blit(card_surf, rect.topleft)

            # Embedded thumbnail preview inside the card
            thumb_w = rect.width - 28
            thumb_h = int(rect.height * 0.54)
            thumb = self._get_background(thumb_path, thumb_w, thumb_h) if thumb_path is not None else None
            thumb_rect = pygame.Rect(rect.x + 14, rect.y + 14, thumb_w, thumb_h)
            if thumb is not None:
                screen.blit(thumb, thumb_rect.topleft)
            else:
                pygame.draw.rect(screen, (14, 22, 36), thumb_rect, border_radius=8)
                # Stylized neon air-drawing preview inside the HoloBoard card
                cx, cy = thumb_rect.center
                pygame.draw.lines(
                    screen,
                    (90, 240, 255),
                    False,
                    [(cx - 55, cy + 12), (cx - 25, cy - 22), (cx + 10, cy + 14), (cx + 48, cy - 18)],
                    3,
                )
                kb_prev = self._font_md.render("A  B  C  ...", True, (255, 205, 95))
                screen.blit(kb_prev, (cx - kb_prev.get_width() // 2, thumb_rect.bottom - 34))
            pygame.draw.rect(screen, accent if is_hov else (70, 95, 130), thumb_rect, 2, border_radius=8)

            pygame.draw.rect(
                screen,
                accent if is_hov else (75, 105, 145),
                rect,
                4 if is_hov else 2,
                border_radius=14,
            )

            text_y = thumb_rect.bottom + 14
            h_surf = self._font_md.render(heading, True, (255, 255, 255))
            t_surf = self._font_sm.render(tagline, True, accent)
            d_surf = self._font_xs.render(desc, True, (215, 226, 240))
            screen.blit(h_surf, (rect.x + 18, text_y))
            screen.blit(t_surf, (rect.x + 18, text_y + 30))
            screen.blit(d_surf, (rect.x + 18, text_y + 56))

        # 4. Model Evaluation Report Button
        rep_rect = _to_px_rect(HUB_REPORT_BOX, width, height)
        rep_hov = hovered_card == "report"
        if rep_hov:
            _draw_glow(screen, rep_rect.center, rep_rect.width // 2, (165, 130, 255), alpha=65)
        rep_surf = pygame.Surface((rep_rect.width, rep_rect.height), pygame.SRCALPHA)
        rep_surf.fill((36, 30, 62, 232) if rep_hov else (22, 24, 42, 215))
        screen.blit(rep_surf, rep_rect.topleft)
        pygame.draw.rect(
            screen,
            (185, 155, 255) if rep_hov else (110, 125, 175),
            rep_rect,
            3 if rep_hov else 2,
            border_radius=12,
        )
        rep_lbl = self._font_md.render("4. MODEL EVALUATION & REPORT", True, (240, 235, 255))
        screen.blit(rep_lbl, (rep_rect.centerx - rep_lbl.get_width() // 2, rep_rect.centery - rep_lbl.get_height() // 2))

        self._draw_status_strip(
            screen,
            width,
            "Select a Game",
            prediction,
            fps,
            model_name,
            secondary_prediction=secondary_prediction,
        )
        self._draw_webcam_pip(screen, camera_surface, width, height)
        self._draw_pointer(screen, width, height, pointer_xy, prediction.smoothed_label)

    # ------------------------------------------------------------------
    # HoloEscape Room (First-Person POV)
    # ------------------------------------------------------------------
    def render(
        self,
        screen: pygame.Surface,
        camera_surface: pygame.Surface,
        state: GameState,
        prediction: GesturePrediction,
        pointer_xy: tuple[float, float] | None,
        fps: float,
        model_name: str,
        secondary_prediction: GesturePrediction | None = None,
    ) -> None:
        width, height = screen.get_size()
        t = time.monotonic()

        # First-Person POV camera transform on cinematic room (depth zoom + strafe + look pan + walking head-bob)
        zoom, off_x, off_y = pov_camera_params(state, pointer_xy)
        bob_x = int(math.cos(t * 6.0) * 3) if state.is_moving else 0
        bob_y = int(math.sin(t * 12.0) * 5) if state.is_moving else 0

        zoom_w = int(width * zoom)
        zoom_h = int(height * zoom)
        bg = self._get_background(ESCAPE_BG_PATH, zoom_w, zoom_h)
        if bg is not None:
            base_x = int((0.5 - 0.5 * zoom + off_x) * width) + bob_x
            base_y = int((0.5 - 0.5 * zoom + off_y) * height) + bob_y
            screen.blit(bg, (base_x, base_y))
        else:
            screen.fill((12, 18, 28))

        self._draw_wall_clock(screen, width, height, state, pointer_xy, bob_x, bob_y, zoom)
        self._draw_pov_props(screen, width, height, state, pointer_xy, bob_x, bob_y, zoom)
        self._draw_pov_atmosphere(screen, width, height, state, pointer_xy, bob_x, bob_y, zoom, t)
        self._draw_pov_hands(screen, width, height, state, prediction.smoothed_label, pointer_xy, t)
        self._draw_pov_visor_frame(screen, width, height, state)

        if state.npc_dialog_open:
            self._draw_npc_dialog(screen, width, height, state)
        elif state.safe_panel_open:
            self._draw_safe_panel(screen, width, height, state)
        elif state.exit_panel_open:
            self._draw_exit_panel(screen, width, height, state)
        elif state.stage == "completed":
            self._draw_completed(screen, width, height, state)

        if state.menu_open:
            self._draw_controls_menu(screen, width, height)

        stage_titles = {
            "chest_lock": "HoloEscape POV 1/3: Decode Narration & Open Chest",
            "mystery_object": "HoloEscape POV 2/3: Pinch Holo Core -> Right Portal",
            "final_exit": "HoloEscape POV 3/3: Unlock Back Exit Terminal",
            "completed": "HoloEscape: Sector 4 Escaped!",
        }
        self._draw_status_strip(
            screen,
            width,
            stage_titles.get(state.stage, state.stage),
            prediction,
            fps,
            model_name,
            secondary_prediction=secondary_prediction,
        )
        self._draw_webcam_pip(screen, camera_surface, width, height)
        self._draw_pov_reticle(
            screen,
            width,
            height,
            pointer_xy,
            prediction.smoothed_label,
            state.nearby_station,
            dwell_progress=state.dwell_progress,
        )
        self._draw_narration_and_feedback(screen, width, height, state)

    def _pov_px(
        self,
        world_xy: tuple[float, float] | list[float],
        width: int,
        height: int,
        state: GameState,
        pointer_xy: tuple[float, float] | None,
        bob_x: int,
        bob_y: int,
    ) -> tuple[int, int]:
        sx, sy = pov_project(world_xy, state, pointer_xy)
        return int(sx * width) + bob_x, int(sy * height) + bob_y

    def _draw_wall_clock(
        self,
        screen: pygame.Surface,
        width: int,
        height: int,
        state: GameState,
        pointer_xy: tuple[float, float] | None,
        bob_x: int,
        bob_y: int,
        zoom: float,
    ) -> None:
        elapsed = max(0, int(time.monotonic() - state.start_time))
        remaining = max(0, 45 * 60 - elapsed)
        mins, secs = divmod(remaining, 60)
        clock_str = f"{mins:02d}:{secs:02d}"

        # Centered inside the background's wall display bezel (y=0.473)
        cx, cy = self._pov_px((0.50, 0.473), width, height, state, pointer_xy, bob_x, bob_y)
        scale = zoom / 1.15
        panel_w = max(90, int(112 * scale))
        panel_h = max(30, int(36 * scale))
        panel_rect = pygame.Rect(cx - panel_w // 2, cy - panel_h // 2, panel_w, panel_h)

        backing = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        backing.fill((8, 16, 26, 220))
        screen.blit(backing, panel_rect.topleft)
        pygame.draw.rect(screen, (65, 195, 240), panel_rect, 1, border_radius=5)

        _draw_glow(screen, (cx, cy), max(32, int(44 * scale)), (70, 215, 255), alpha=45)
        clock_surf = self._font_clock.render(clock_str, True, (140, 248, 255))
        if abs(scale - 1.0) > 0.04:
            cw = max(20, int(clock_surf.get_width() * scale))
            ch = max(12, int(clock_surf.get_height() * scale))
            clock_surf = pygame.transform.smoothscale(clock_surf, (cw, ch))
        screen.blit(clock_surf, (cx - clock_surf.get_width() // 2, cy - clock_surf.get_height() // 2))

    def _draw_pov_props(
        self,
        screen: pygame.Surface,
        width: int,
        height: int,
        state: GameState,
        pointer_xy: tuple[float, float] | None,
        bob_x: int,
        bob_y: int,
        zoom: float,
    ) -> None:
        scale = zoom / 1.15

        # 1. Left Cipher Wall Hotspot indicator
        tx, ty = self._pov_px(TERMINAL_POS, width, height, state, pointer_xy, bob_x, bob_y)
        if state.nearby_station == "terminal":
            _draw_glow(screen, (tx, ty), int(55 * scale), (95, 255, 165), alpha=70)

        # 2. Sci-Fi Armored Cyber Vault (replaces flat cartoon chest)
        sx, sy = self._pov_px(SAFE_POS, width, height, state, pointer_xy, bob_x, bob_y)
        near_s = state.nearby_station == "safe"
        chest_open = state.stage != "chest_lock"
        vw = int(118 * scale)
        vh = int(56 * scale)

        # Soft ambient floor shadow
        shadow_surf = pygame.Surface((vw + 34, int(30 * scale)), pygame.SRCALPHA)
        pygame.draw.ellipse(shadow_surf, (4, 6, 10, 175), shadow_surf.get_rect())
        screen.blit(shadow_surf, (sx - (vw + 34) // 2, sy + vh // 2 - int(8 * scale)))

        led_col = (110, 255, 175) if chest_open else ((255, 215, 95) if near_s else (75, 215, 255))
        if near_s and not chest_open:
            _draw_glow(screen, (sx, sy), int(78 * scale), (255, 215, 95), alpha=70)

        # 3D perspective trapezoid lid
        lid_lift = int(12 * scale) if chest_open else 0
        top_w = int(vw * 0.80)
        lid_h = int(18 * scale)
        lid_top_y = sy - vh // 2 - lid_lift
        lid_pts = [
            (sx - top_w // 2, lid_top_y),
            (sx + top_w // 2, lid_top_y),
            (sx + vw // 2, lid_top_y + lid_h),
            (sx - vw // 2, lid_top_y + lid_h),
        ]
        pygame.draw.polygon(screen, (48, 60, 78), lid_pts)
        pygame.draw.polygon(screen, (115, 140, 172), lid_pts, 2)

        # Beveled front armored body
        body = pygame.Rect(sx - vw // 2, sy - vh // 2 + lid_h - int(2 * scale), vw, vh - lid_h + int(6 * scale))
        pygame.draw.rect(screen, (26, 34, 48), body, border_radius=6)
        inner = body.inflate(-int(16 * scale), -int(12 * scale))
        pygame.draw.rect(screen, (15, 20, 30), inner, border_radius=4)
        pygame.draw.rect(screen, (72, 92, 120), body, 2, border_radius=6)

        # Glowing LED seam strip + corner brackets
        pygame.draw.line(screen, led_col, (body.left + 6, body.top + 1), (body.right - 6, body.top + 1), 2)
        for bx in (body.left + int(14 * scale), body.right - int(18 * scale)):
            pygame.draw.rect(screen, (68, 86, 112), pygame.Rect(bx, body.top + 4, int(5 * scale), body.height - 8))

        # Center holographic biometric lock core
        core_y = body.centery
        _draw_glow(screen, (sx, core_y), int(18 * scale), led_col, alpha=85)
        pygame.draw.circle(screen, (14, 22, 34), (sx, core_y), max(6, int(9 * scale)))
        pygame.draw.circle(screen, led_col, (sx, core_y), max(6, int(9 * scale)), 2)
        pygame.draw.circle(screen, led_col, (sx, core_y), max(2, int(3 * scale)))

        # 3. Right Portal Glow
        px, py = self._pov_px(PEDESTAL_POS, width, height, state, pointer_xy, bob_x, bob_y)
        if state.stage == "mystery_object":
            _draw_glow(screen, (px, py), int(65 * scale), (130, 195, 255), alpha=70)

        # 4. Back Wall Exit Terminal Glow
        if state.stage in ("final_exit", "completed"):
            ex, ey = self._pov_px(EXIT_DOOR_POS, width, height, state, pointer_xy, bob_x, bob_y)
            col = (120, 255, 185) if state.stage == "completed" else (255, 195, 95)
            _draw_glow(screen, (ex, ey), int(58 * scale), col, alpha=80 if state.nearby_station == "exit_door" else 45)

        # 5. Holo-Core Crystal (Stage 2+)
        if state.stage in ("mystery_object", "final_exit", "completed"):
            ax, ay = self._pov_px(state.artifact_pos, width, height, state, pointer_xy, bob_x, bob_y)
            core_col = (255, 205, 80) if state.artifact_grabbed else (95, 220, 255)
            _draw_glow(screen, (ax, ay), int(42 * scale), core_col, alpha=95)
            r_x = int(15 * scale)
            r_y = int(20 * scale)
            diamond = [(ax, ay - r_y), (ax + r_x, ay), (ax, ay + r_y), (ax - r_x, ay)]
            pygame.draw.polygon(screen, core_col, diamond)
            pygame.draw.polygon(screen, (255, 255, 255), diamond, 2)

        # 6. Interactive NPC: Dr. Vance (Detailed Sci-Fi Engineer Character Model)
        nx, ny = self._pov_px(NPC_POS, width, height, state, pointer_xy, bob_x, bob_y)
        near_npc = state.nearby_station == "npc"
        npc_col = (120, 255, 190) if near_npc else (110, 210, 255)
        t_now = time.monotonic()
        idle_bob = int(math.sin(t_now * 2.5) * 3)
        arm_sway = int(math.cos(t_now * 2.0) * 2)

        sh_w, sh_h = int(60 * scale), int(18 * scale)
        sh_s = pygame.Surface((sh_w, sh_h), pygame.SRCALPHA)
        pygame.draw.ellipse(sh_s, (4, 6, 12, 175), sh_s.get_rect())
        screen.blit(sh_s, (nx - sh_w // 2, ny + int(42 * scale)))
        if near_npc:
            _draw_glow(screen, (nx, ny - int(6 * scale)), int(70 * scale), (110, 255, 190), alpha=80)

        # Legs & armored boots
        leg_w = max(6, int(9 * scale))
        leg_h = int(24 * scale)
        hip_y = ny + int(18 * scale) + idle_bob
        pygame.draw.rect(screen, (22, 30, 44), pygame.Rect(nx - int(11 * scale), hip_y, leg_w, leg_h), border_radius=3)
        pygame.draw.rect(screen, (22, 30, 44), pygame.Rect(nx + int(2 * scale), hip_y, leg_w, leg_h), border_radius=3)
        pygame.draw.rect(screen, (60, 80, 108), pygame.Rect(nx - int(12 * scale), hip_y + leg_h - int(6 * scale), leg_w + 3, int(7 * scale)), border_radius=3)
        pygame.draw.rect(screen, (60, 80, 108), pygame.Rect(nx + int(1 * scale), hip_y + leg_h - int(6 * scale), leg_w + 3, int(7 * scale)), border_radius=3)

        # Tactical lab coat torso + shoulder pads + glowing chest core
        torso_w, torso_h = int(38 * scale), int(46 * scale)
        torso = pygame.Rect(nx - torso_w // 2, ny - int(22 * scale) + idle_bob, torso_w, torso_h)
        pygame.draw.rect(screen, (32, 46, 68), torso, border_radius=9)
        pygame.draw.rect(screen, npc_col, torso, 2, border_radius=9)
        # Lab coat lapel V-trim & glowing badge
        pygame.draw.line(screen, (180, 225, 255), (nx, torso.top + int(4 * scale)), (nx, torso.bottom - int(6 * scale)), 2)
        pygame.draw.circle(screen, npc_col, (nx + int(8 * scale), torso.top + int(15 * scale)), max(3, int(4 * scale)))

        # Left & right arms (right arm holds a glowing holo-tablet)
        l_arm = pygame.Rect(torso.left - int(8 * scale), torso.top + int(6 * scale) + arm_sway, int(8 * scale), int(28 * scale))
        r_arm = pygame.Rect(torso.right, torso.top + int(6 * scale) - arm_sway, int(8 * scale), int(26 * scale))
        pygame.draw.rect(screen, (40, 56, 82), l_arm, border_radius=4)
        pygame.draw.rect(screen, (40, 56, 82), r_arm, border_radius=4)
        tablet = pygame.Rect(r_arm.left - int(6 * scale), r_arm.bottom - int(10 * scale), int(16 * scale), int(12 * scale))
        pygame.draw.rect(screen, (18, 42, 60), tablet, border_radius=3)
        pygame.draw.rect(screen, (95, 240, 255), tablet, 2, border_radius=3)

        # Head + hair + glowing AR visor
        hr = max(11, int(14 * scale))
        hy = torso.top - hr + int(2 * scale)
        pygame.draw.circle(screen, (224, 188, 162), (nx, hy), hr)
        # Hair / cap silhouette
        pygame.draw.arc(screen, (45, 52, 66), pygame.Rect(nx - hr - 1, hy - hr - 1, (hr + 1) * 2, hr * 2), 0.2, math.pi - 0.2, max(3, int(5 * scale)))
        # Glowing AR visor
        visor_rect = pygame.Rect(nx - hr + 2, hy - int(3 * scale), (hr - 2) * 2, max(5, int(6 * scale)))
        pygame.draw.rect(screen, (16, 36, 52), visor_rect, border_radius=3)
        pygame.draw.rect(screen, npc_col, visor_rect, 2, border_radius=3)

        tag = self._font_xs.render("DR. VANCE", True, npc_col)
        screen.blit(tag, (nx - tag.get_width() // 2, hy - hr - 22))
        if near_npc and not state.npc_dialog_open:
            prompt = self._font_xs.render("Pinch / Thumbs Up to Talk", True, (255, 240, 175))
            screen.blit(prompt, (nx - prompt.get_width() // 2, hy - hr - 38))

    def _draw_pov_atmosphere(
        self,
        screen: pygame.Surface,
        width: int,
        height: int,
        state: GameState,
        pointer_xy: tuple[float, float] | None,
        bob_x: int,
        bob_y: int,
        zoom: float,
        t: float,
    ) -> None:
        """Add 3D parallax holo-dust, animated cipher screen, and rotating portal rings."""
        scale = zoom / 1.15

        # Animated Left Cipher Wall waveform overlay
        tx, ty = self._pov_px(TERMINAL_POS, width, height, state, pointer_xy, bob_x, bob_y)
        wave_w, wave_h = int(66 * scale), int(44 * scale)
        wave_rect = pygame.Rect(tx - wave_w // 2, ty - wave_h // 2, wave_w, wave_h)
        wave_surf = pygame.Surface((wave_w, wave_h), pygame.SRCALPHA)
        wave_surf.fill((10, 35, 28, 120))
        for i in range(6):
            bar_x = 6 + i * max(6, wave_w // 7)
            bar_h = int((0.35 + 0.55 * abs(math.sin(t * 3.5 + i * 0.9))) * (wave_h - 12))
            pygame.draw.rect(wave_surf, (95, 255, 175, 190), pygame.Rect(bar_x, wave_h - 6 - bar_h, 4, bar_h))
        screen.blit(wave_surf, wave_rect.topleft)

        # Animated Right Energy Portal rings during Stage 2
        if state.stage == "mystery_object":
            px, py = self._pov_px(PEDESTAL_POS, width, height, state, pointer_xy, bob_x, bob_y)
            for ring_i in range(3):
                r = int((18 + ring_i * 12 + 4 * math.sin(t * 4.0 + ring_i)) * scale)
                pygame.draw.circle(screen, (125, 225, 255), (px, py), max(4, r), 2)

        # Floating 3D parallax dust motes
        _, off_x, off_y = pov_camera_params(state, pointer_xy)
        for idx in range(14):
            depth = 0.6 + (idx % 4) * 0.35
            fx = ((idx * 0.173 + t * 0.012 * depth + off_x * depth * 1.6) % 1.0) * width
            fy = ((idx * 0.291 + math.sin(t * 0.9 + idx) * 0.015 + off_y * depth * 1.4) % 0.85) * height + 50
            alpha = 90 + (idx % 3) * 40
            mote = pygame.Surface((6, 6), pygame.SRCALPHA)
            pygame.draw.circle(mote, (150, 235, 255, alpha), (3, 3), 2)
            screen.blit(mote, (int(fx), int(fy)))

    def _draw_pov_hands(
        self,
        screen: pygame.Surface,
        width: int,
        height: int,
        state: GameState,
        smoothed_label: str | None,
        pointer_xy: tuple[float, float] | None,
        t: float,
    ) -> None:
        """Draw First-Person cyber-gloved hands that swing as you walk and mirror gestures."""
        if state.safe_panel_open or state.exit_panel_open or state.menu_open or state.stage == "completed":
            return

        stride = math.sin(t * 10.0) * 14.0 if state.is_moving else math.sin(t * 2.0) * 2.5
        aim_dx = int((pointer_xy[0] - 0.5) * 70) if pointer_xy is not None else 0
        aim_dy = int((pointer_xy[1] - 0.5) * 45) if pointer_xy is not None else 0

        # Left forearm/glove (swings opposite phase during walking)
        lx = int(width * 0.19) + int(stride * 0.4)
        ly = height - 68 + int(abs(stride) * 0.5) - int(stride * 0.4)
        l_pts = [
            (lx - 48, height),
            (lx - 18, ly),
            (lx + 24, ly + 10),
            (lx + 10, height),
        ]
        pygame.draw.polygon(screen, (22, 30, 44), l_pts)
        pygame.draw.polygon(screen, (65, 165, 215), l_pts, 2)
        pygame.draw.circle(screen, (32, 44, 62), (lx + 2, ly + 4), 18)
        pygame.draw.circle(screen, (85, 210, 255), (lx + 2, ly + 4), 18, 2)

        # Right cyber-glove (tracks aim + gesture color)
        rx = int(width * 0.79) + aim_dx - int(stride * 0.4)
        ry = height - 74 + aim_dy + int(stride * 0.5)
        glow_col = (
            (255, 205, 80)
            if smoothed_label == "pinch"
            else ((120, 255, 175) if smoothed_label == "thumbs_up" else (85, 225, 255))
        )
        r_pts = [
            (rx - 14, height),
            (rx - 22, ry + 10),
            (rx + 18, ry),
            (rx + 52, height),
        ]
        pygame.draw.polygon(screen, (22, 30, 44), r_pts)
        pygame.draw.polygon(screen, glow_col, r_pts, 2)
        pygame.draw.circle(screen, (34, 46, 66), (rx - 2, ry + 4), 19)
        pygame.draw.circle(screen, glow_col, (rx - 2, ry + 4), 19, 2)
        if smoothed_label == "pointing":
            pygame.draw.line(screen, glow_col, (rx - 4, ry - 8), (rx - 16, ry - 34), 6)
            pygame.draw.circle(screen, (200, 250, 255), (rx - 16, ry - 34), 4)

    def _draw_pov_visor_frame(
        self,
        screen: pygame.Surface,
        width: int,
        height: int,
        state: GameState,
    ) -> None:
        """First-person AR visor brackets and compact room position radar."""
        col = (70, 180, 225)
        m = 18
        top = 64
        bot = height - 84
        arm = 28
        # Top-left & bottom-left brackets
        pygame.draw.line(screen, col, (m, top), (m + arm, top), 2)
        pygame.draw.line(screen, col, (m, top), (m, top + arm), 2)
        pygame.draw.line(screen, col, (m, bot), (m + arm, bot), 2)
        pygame.draw.line(screen, col, (m, bot), (m, bot - arm), 2)
        # Bottom-right bracket
        pygame.draw.line(screen, col, (width - m, bot), (width - m - arm, bot), 2)
        pygame.draw.line(screen, col, (width - m, bot), (width - m, bot - arm), 2)

        # Compact POV Room Radar (bottom-left above narration bar)
        rx, ry, rw, rh = 24, height - 166, 122, 74
        radar = pygame.Surface((rw, rh), pygame.SRCALPHA)
        radar.fill((10, 18, 30, 180))
        screen.blit(radar, (rx, ry))
        pygame.draw.rect(screen, (70, 165, 220), pygame.Rect(rx, ry, rw, rh), 1, border_radius=6)

        # Stations & NPC on radar
        for pos, scol in (
            (TERMINAL_POS, (95, 255, 165)),
            (NPC_POS, (255, 130, 235)),
            (SAFE_POS, (255, 215, 100)),
            (PEDESTAL_POS, (130, 195, 255)),
        ):
            sx = rx + int(pos[0] * rw)
            sy = ry + int(pos[1] * rh)
            pygame.draw.circle(screen, scol, (sx, sy), 3)

        # Player dot + 3D facing vector on radar
        px = rx + int(state.player_pos[0] * rw)
        py = ry + int(state.player_pos[1] * rh)
        fx = px + int(math.cos(state.player_yaw) * 14)
        fy = py + int(math.sin(state.player_yaw) * 14)
        pygame.draw.line(screen, (95, 235, 255), (px, py), (fx, fy), 2)
        pcol = (120, 255, 190) if state.is_moving else (255, 255, 255)
        pygame.draw.circle(screen, pcol, (px, py), 4)
        lbl_txt = "WALKING" if state.is_moving else "WASD/ARROWS/POINT"
        lbl = self._font_xs.render(lbl_txt, True, (170, 225, 250))
        screen.blit(lbl, (rx + 6, ry + 4))

    def _draw_pov_reticle(
        self,
        screen: pygame.Surface,
        width: int,
        height: int,
        pointer_xy: tuple[float, float] | None,
        smoothed_label: str | None,
        nearby_station: str | None,
        dwell_progress: float = 0.0,
    ) -> None:
        if pointer_xy is None:
            return
        px = int(max(0.0, min(1.0, pointer_xy[0])) * width)
        py = int(max(0.0, min(1.0, pointer_xy[1])) * height)
        is_pinch = smoothed_label == "pinch"
        color = (255, 205, 80) if is_pinch else ((130, 255, 180) if nearby_station else (90, 245, 255))

        # Crisp POV crosshair reticle
        pygame.draw.circle(screen, color, (px, py), 14, 2)
        pygame.draw.circle(screen, (255, 255, 255), (px, py), 3)
        for dx, dy in ((-20, 0), (20, 0), (0, -20), (0, 20)):
            pygame.draw.line(screen, color, (px + dx // 2, py + dy // 2), (px + dx, py + dy), 2)

        if dwell_progress > 0.02:
            arc_rect = pygame.Rect(px - 20, py - 20, 40, 40)
            end_angle = math.pi / 2.0 + dwell_progress * 2.0 * math.pi
            pygame.draw.arc(screen, (120, 255, 190), arc_rect, math.pi / 2.0, end_angle, 3)

        labels = {
            "npc": "TALK TO DR. VANCE (Pinch / Hold)",
            "terminal": "CIPHER LOG (Pinch / Hold)",
            "safe": "OPEN CHEST (Pinch / Hold)",
            "exit_door": "EXIT KEYPAD (Pinch / Hold)",
        }
        if nearby_station in labels:
            tag = self._font_xs.render(labels[nearby_station], True, (140, 255, 190))
            screen.blit(tag, (px + 20, py - 10))

    # ------------------------------------------------------------------
    # HoloSoccer Stadium (2.5D Perspective + Animated 4v4 Footballers)
    # ------------------------------------------------------------------
    def _soccer_proj(
        self,
        pos: tuple[float, float] | list[float],
        width: int,
        height: int,
        cam_pan_x: int,
    ) -> tuple[int, int, float]:
        """Project pitch coords into 2.5D perspective screen coords + depth scale."""
        px = int(pos[0] * width) + cam_pan_x
        py = int(pos[1] * height)
        depth_scale = 0.84 + max(0.0, min(1.0, float(pos[1]))) * 0.34
        return px, py, depth_scale

    def _draw_footballer(
        self,
        screen: pygame.Surface,
        pos: tuple[float, float] | list[float],
        vel: tuple[float, float] | list[float],
        width: int,
        height: int,
        cam_pan_x: int,
        kit_col: tuple[int, int, int],
        trim_col: tuple[int, int, int],
        number_str: str,
        t: float,
        is_controlled: bool = False,
        shot_flash: bool = False,
    ) -> None:
        px, py, sc = self._soccer_proj(pos, width, height, cam_pan_x)
        speed = math.hypot(vel[0], vel[1])
        stride = math.sin(t * 16.0) * 6.0 * sc if speed > 0.02 else 0.0

        # Active player tactical ring + shot glow
        if is_controlled:
            ring_r = int(23 * sc)
            pygame.draw.ellipse(
                screen,
                (255, 225, 90),
                pygame.Rect(px - ring_r, py + int(6 * sc) - ring_r // 2, ring_r * 2, ring_r),
                2,
            )
            if shot_flash:
                _draw_glow(screen, (px, py), int(50 * sc), (255, 210, 80), alpha=125)

        # Ground shadow
        sh_w, sh_h = int(28 * sc), int(11 * sc)
        shadow = pygame.Surface((sh_w, sh_h), pygame.SRCALPHA)
        pygame.draw.ellipse(shadow, (6, 16, 10, 165), shadow.get_rect())
        screen.blit(shadow, (px - sh_w // 2, py + int(10 * sc)))

        # Animated running legs / boots
        leg_w = max(4, int(5 * sc))
        leg_h = max(7, int(10 * sc))
        pygame.draw.rect(
            screen,
            (25, 28, 36),
            pygame.Rect(px - int(6 * sc), py + int(4 * sc) + int(stride * 0.5), leg_w, leg_h),
            border_radius=2,
        )
        pygame.draw.rect(
            screen,
            (25, 28, 36),
            pygame.Rect(px + int(2 * sc), py + int(4 * sc) - int(stride * 0.5), leg_w, leg_h),
            border_radius=2,
        )

        # Jersey torso
        tw, th = int(20 * sc), int(18 * sc)
        torso = pygame.Rect(px - tw // 2, py - int(9 * sc), tw, th)
        pygame.draw.rect(screen, kit_col, torso, border_radius=5)
        pygame.draw.rect(screen, trim_col, torso, 2, border_radius=5)

        # Head + hair
        hr = max(5, int(7 * sc))
        hx = px + (int(2 * sc) if vel[0] > 0.02 else (-int(2 * sc) if vel[0] < -0.02 else 0))
        hy = py - int(14 * sc)
        pygame.draw.circle(screen, (235, 198, 168), (hx, hy), hr)
        pygame.draw.arc(screen, (35, 28, 22), pygame.Rect(hx - hr, hy - hr, hr * 2, hr * 2), 0.2, 2.9, 3)

        # Jersey number
        num_surf = self._font_xs.render(number_str, True, (255, 255, 255))
        screen.blit(num_surf, (px - num_surf.get_width() // 2, torso.centery - num_surf.get_height() // 2))

        if is_controlled:
            you_lbl = self._font_xs.render("YOU", True, (255, 238, 130))
            screen.blit(you_lbl, (px - you_lbl.get_width() // 2, hy - hr - 16))

    def render_soccer(
        self,
        screen: pygame.Surface,
        camera_surface: pygame.Surface,
        state: SoccerState,
        prediction: GesturePrediction,
        pointer_xy: tuple[float, float] | None,
        fps: float,
        model_name: str,
        secondary_prediction: GesturePrediction | None = None,
    ) -> None:
        width, height = screen.get_size()
        t = time.monotonic()

        # Subtle dynamic stadium camera pan following the ball
        cam_pan_x = int((0.50 - state.ball_pos[0]) * width * 0.06)
        zoom_w = int(width * 1.08)
        zoom_h = int(height * 1.04)
        bg = self._get_background(SOCCER_BG_PATH, zoom_w, zoom_h)
        if bg is not None:
            screen.blit(bg, (-(zoom_w - width) // 2 + cam_pan_x, -(zoom_h - height) // 2))
        else:
            screen.fill((18, 55, 28))

        # Passing tactical laser line
        if state.pass_line is not None and state.pass_flash_timer > 0:
            p0 = self._soccer_proj(state.pass_line[0], width, height, cam_pan_x)[:2]
            p1 = self._soccer_proj(state.pass_line[1], width, height, cam_pan_x)[:2]
            pygame.draw.line(screen, (130, 255, 195), p0, p1, 3)

        # Depth-sort all players + keeper by Y so closer players overlap further ones naturally
        draw_list: list[tuple[float, str, int]] = [(state.keeper_pos[1], "keeper", 0)]
        for i, opp in enumerate(state.opponents):
            draw_list.append((opp.pos[1], "opp", i))
        for i, mate in enumerate(state.teammates):
            draw_list.append((mate.pos[1], "mate", i))
        draw_list.sort(key=lambda item: item[0])

        for _, kind, idx in draw_list:
            if kind == "keeper":
                self._draw_footballer(
                    screen,
                    state.keeper_pos,
                    (0.0, state.keeper_dir * 0.18),
                    width,
                    height,
                    cam_pan_x,
                    (205, 55, 45),
                    (255, 215, 95),
                    "GK",
                    t,
                )
            elif kind == "opp":
                opp = state.opponents[idx]
                self._draw_footballer(
                    screen,
                    opp.pos,
                    opp.vel,
                    width,
                    height,
                    cam_pan_x,
                    (220, 58, 58),
                    (255, 225, 225),
                    str(idx + 2),
                    t,
                )
            else:
                mate = state.teammates[idx]
                is_ctrl = idx == state.controlled_idx
                self._draw_footballer(
                    screen,
                    mate.pos,
                    mate.vel,
                    width,
                    height,
                    cam_pan_x,
                    (48, 175, 255) if is_ctrl else (35, 118, 210),
                    (255, 230, 110) if is_ctrl else (225, 240, 255),
                    str(idx + 1),
                    t,
                    is_controlled=is_ctrl,
                    shot_flash=state.shot_flash_timer > 0,
                )

        # Ball motion trail
        for ti, trail_pos in enumerate(state.ball_trail):
            tx, ty, _ = self._soccer_proj(trail_pos, width, height, cam_pan_x)
            tr = max(2, int(2 + ti * 0.6))
            trail_s = pygame.Surface((tr * 2, tr * 2), pygame.SRCALPHA)
            pygame.draw.circle(trail_s, (140, 245, 255, 25 + ti * 14), (tr, tr), tr)
            screen.blit(trail_s, (tx - tr, ty - tr))

        # 3D arcing & spinning soccer ball
        bx, by, bsc = self._soccer_proj(state.ball_pos, width, height, cam_pan_x)
        lift_px = int(state.ball_height * height * 0.45)
        b_rad = max(7, int((10 + state.ball_height * 22) * bsc))
        sh_w, sh_h = int(20 * bsc), int(8 * bsc)
        b_shadow = pygame.Surface((sh_w, sh_h), pygame.SRCALPHA)
        pygame.draw.ellipse(b_shadow, (6, 16, 10, 160), b_shadow.get_rect())
        screen.blit(b_shadow, (bx - sh_w // 2, by + int(6 * bsc)))

        ball_y = by - lift_px
        pygame.draw.circle(screen, (248, 250, 255), (bx, ball_y), b_rad)
        pygame.draw.circle(screen, (28, 34, 44), (bx, ball_y), b_rad, 2)
        # Spinning pentagon seams
        for p_i in range(3):
            ang = state.ball_spin + p_i * 2.09
            sx = bx + int(math.cos(ang) * b_rad * 0.48)
            sy = ball_y + int(math.sin(ang) * b_rad * 0.48)
            pygame.draw.circle(screen, (32, 38, 48), (sx, sy), max(2, b_rad // 4))

        if state.goal_banner_timer > 0:
            _draw_glow(screen, (width // 2, height // 2), 110, (255, 215, 70), alpha=105)
            g_lbl = self._font_lg.render("GOOOAL!!", True, (255, 235, 90))
            screen.blit(g_lbl, (width // 2 - g_lbl.get_width() // 2, height // 2 - 20))

        # Broadcast scoreboard pill
        sb_w, sb_h = 340, 36
        sb_rect = pygame.Rect(width // 2 - sb_w // 2, 54, sb_w, sb_h)
        sb_surf = pygame.Surface((sb_w, sb_h), pygame.SRCALPHA)
        sb_surf.fill((12, 18, 30, 220))
        screen.blit(sb_surf, sb_rect.topleft)
        pygame.draw.rect(screen, (95, 220, 255), sb_rect, 1, border_radius=8)
        sb_txt = self._font_sm.render(
            f"BLUE  {state.score_player}  -  {state.score_ai}  RED   |   {state.time_remaining:04.1f}s",
            True,
            (240, 248, 255),
        )
        screen.blit(sb_txt, (sb_rect.centerx - sb_txt.get_width() // 2, sb_rect.centery - sb_txt.get_height() // 2))

        self._draw_status_strip(
            screen,
            width,
            "HoloSoccer 4v4 (Point/WASD Move | 2-Fingers/E Pass | Fist/Space Shoot)",
            prediction,
            fps,
            model_name,
            secondary_prediction=secondary_prediction,
        )
        self._draw_webcam_pip(screen, camera_surface, width, height)
        self._draw_pointer(screen, width, height, pointer_xy, prediction.smoothed_label)

    # ------------------------------------------------------------------
    # HoloBoard (Air-Drawing & Gesture Text Board)
    # ------------------------------------------------------------------
    def render_board(
        self,
        screen: pygame.Surface,
        camera_surface: pygame.Surface,
        state: BoardState,
        prediction: GesturePrediction,
        pointer_xy: tuple[float, float] | None,
        fps: float,
        model_name: str,
        secondary_prediction: GesturePrediction | None = None,
    ) -> None:
        width, height = screen.get_size()
        screen.fill((11, 16, 26))

        # Toolbar strip
        tb_rect = _to_px_rect(TOOLBAR_BOX, width, height)
        pygame.draw.rect(screen, (18, 26, 40), tb_rect, border_radius=8)
        pygame.draw.rect(screen, (65, 95, 130), tb_rect, 1, border_radius=8)
        for btn_id, label, box in TOOLBAR_BUTTONS:
            rect = _to_px_rect(box, width, height)
            is_hov = state.hovered_target == btn_id
            if btn_id.startswith("color_"):
                idx = int(btn_id.split("_")[1])
                _, swatch = PEN_COLORS[idx]
                selected = state.color_index == idx
                pygame.draw.rect(screen, swatch, rect, border_radius=6)
                pygame.draw.rect(
                    screen,
                    (255, 255, 255) if (selected or is_hov) else (40, 50, 68),
                    rect,
                    3 if selected else 2,
                    border_radius=6,
                )
            else:
                bg_col = (52, 92, 138) if is_hov else (28, 42, 64)
                bd_col = (130, 240, 255) if is_hov else (85, 120, 160)
                pygame.draw.rect(screen, bg_col, rect, border_radius=6)
                pygame.draw.rect(screen, bd_col, rect, 2, border_radius=6)
                txt = self._font_xs.render(label, True, (240, 248, 255))
                screen.blit(txt, (rect.centerx - txt.get_width() // 2, rect.centery - txt.get_height() // 2))

        # Air-Drawing Canvas
        c_rect = _to_px_rect(CANVAS_BOX, width, height)
        pygame.draw.rect(screen, (14, 21, 33), c_rect, border_radius=10)
        # Subtle grid lines
        for gx in range(c_rect.left + 40, c_rect.right, 40):
            pygame.draw.line(screen, (22, 33, 50), (gx, c_rect.top), (gx, c_rect.bottom), 1)
        for gy in range(c_rect.top + 40, c_rect.bottom, 40):
            pygame.draw.line(screen, (22, 33, 50), (c_rect.left, gy), (c_rect.right, gy), 1)
        pygame.draw.rect(
            screen,
            (105, 225, 255) if state.active_stroke is not None else (75, 115, 155),
            c_rect,
            2,
            border_radius=10,
        )

        # Render completed + active air-drawn strokes
        all_strokes = [*state.strokes]
        if state.active_stroke is not None:
            all_strokes.append(state.active_stroke)
        for stroke in all_strokes:
            if not stroke.points:
                continue
            pts_px = [_to_px(p, width, height) for p in stroke.points]
            if len(pts_px) == 1:
                pygame.draw.circle(screen, stroke.color, pts_px[0], stroke.width)
            else:
                pygame.draw.lines(screen, stroke.color, False, pts_px, stroke.width * 2)
                for pt in pts_px:
                    pygame.draw.circle(screen, stroke.color, pt, stroke.width)

        # Typed Text Display Bar
        td_rect = _to_px_rect(TEXT_DISPLAY_BOX, width, height)
        pygame.draw.rect(screen, (16, 25, 40), td_rect, border_radius=8)
        pygame.draw.rect(screen, (110, 215, 255), td_rect, 2, border_radius=8)
        cursor_ch = "_" if int(time.monotonic() * 2) % 2 == 0 else " "
        disp_str = (state.text + cursor_ch) if state.text else f"Typed text appears here...{cursor_ch}"
        disp_col = (245, 250, 255) if state.text else (130, 155, 185)
        disp_surf = self._font_md.render(disp_str, True, disp_col)
        screen.blit(disp_surf, (td_rect.left + 14, td_rect.centery - disp_surf.get_height() // 2))

        # Virtual Gesture Keyboard
        for btn_id, label, box in KEYBOARD_BUTTONS:
            rect = _to_px_rect(box, width, height)
            is_hov = state.hovered_target == btn_id
            is_action = label in ("BACK", "CLR", "SPACE")
            bg_col = (60, 115, 175) if is_hov else ((38, 52, 76) if is_action else (24, 36, 56))
            bd_col = (130, 245, 255) if is_hov else (85, 115, 155)
            pygame.draw.rect(screen, bg_col, rect, border_radius=6)
            pygame.draw.rect(screen, bd_col, rect, 3 if is_hov else 1, border_radius=6)
            k_surf = self._font_sm.render(label, True, (255, 255, 255))
            screen.blit(k_surf, (rect.centerx - k_surf.get_width() // 2, rect.centery - k_surf.get_height() // 2))

        self._draw_status_strip(
            screen,
            width,
            "HoloBoard (Pinch to Draw | Pinch / Hold Point on Keys to Type)",
            prediction,
            fps,
            model_name,
            secondary_prediction=secondary_prediction,
        )
        self._draw_webcam_pip(screen, camera_surface, width, height)
        self._draw_pointer(screen, width, height, pointer_xy, prediction.smoothed_label)

    # ------------------------------------------------------------------
    # Model Evaluation & Report Dashboard
    # ------------------------------------------------------------------
    def render_report(
        self,
        screen: pygame.Surface,
        camera_surface: pygame.Surface,
        data: ReportData,
        state: ReportState,
        prediction: GesturePrediction,
        pointer_xy: tuple[float, float] | None,
        fps: float,
        model_name: str,
        secondary_prediction: GesturePrediction | None = None,
    ) -> None:
        width, height = screen.get_size()
        screen.fill((11, 16, 26))

        # Top tab bar + Export HTML + Back
        tab_map = {"tab_overview": "overview", "tab_classes": "classes", "tab_runtime": "runtime"}
        for btn_id, label, box in TAB_BOXES:
            rect = _to_px_rect(box, width, height)
            is_active = state.active_tab == tab_map[btn_id]
            is_hov = state.hovered_target == btn_id
            bg = (45, 95, 145) if is_active else ((34, 62, 98) if is_hov else (20, 30, 48))
            bd = (130, 245, 255) if (is_active or is_hov) else (75, 105, 145)
            pygame.draw.rect(screen, bg, rect, border_radius=8)
            pygame.draw.rect(screen, bd, rect, 3 if is_active else 2, border_radius=8)
            txt = self._font_xs.render(label, True, (255, 255, 255))
            screen.blit(txt, (rect.centerx - txt.get_width() // 2, rect.centery - txt.get_height() // 2))

        for btn_id, label, box, base_col in (
            ("export_html", "EXPORT HTML", EXPORT_HTML_BOX, (36, 92, 72)),
            ("back_hub", "BACK", BACK_HUB_BOX, (105, 48, 56)),
        ):
            rect = _to_px_rect(box, width, height)
            is_hov = state.hovered_target == btn_id
            pygame.draw.rect(screen, tuple(min(255, c + 30) for c in base_col) if is_hov else base_col, rect, border_radius=8)
            pygame.draw.rect(screen, (200, 245, 255) if is_hov else (130, 160, 195), rect, 2, border_radius=8)
            txt = self._font_xs.render(label, True, (255, 255, 255))
            screen.blit(txt, (rect.centerx - txt.get_width() // 2, rect.centery - txt.get_height() // 2))

        if state.active_tab == "overview":
            self._draw_report_overview(screen, width, height, data)
        elif state.active_tab == "classes":
            self._draw_report_classes(screen, width, height, data, state)
        else:
            self._draw_report_runtime(screen, width, height, data)

        # Bottom status bar
        foot = pygame.Rect(int(width * 0.04), int(height * 0.90), int(width * 0.72), int(height * 0.065))
        pygame.draw.rect(screen, (16, 24, 38), foot, border_radius=8)
        pygame.draw.rect(screen, (75, 110, 150), foot, 1, border_radius=8)
        f_txt = self._font_xs.render(state.status_message, True, (215, 235, 250))
        screen.blit(f_txt, (foot.left + 12, foot.centery - f_txt.get_height() // 2))

        self._draw_status_strip(
            screen,
            width,
            "Model Evaluation Report (Pinch / Hold Point on Tabs | M or Fist = Hub)",
            prediction,
            fps,
            model_name,
            secondary_prediction=secondary_prediction,
        )
        self._draw_webcam_pip(screen, camera_surface, width, height)
        self._draw_pointer(screen, width, height, pointer_xy, prediction.smoothed_label)
        if pointer_xy is not None and state.dwell_progress > 0.02:
            px, py = _to_px(pointer_xy, width, height)
            arc_rect = pygame.Rect(px - 18, py - 18, 36, 36)
            pygame.draw.arc(screen, (120, 255, 190), arc_rect, math.pi / 2.0, math.pi / 2.0 + state.dwell_progress * 2.0 * math.pi, 3)

    def _draw_table_row(
        self,
        screen: pygame.Surface,
        rect: pygame.Rect,
        y: int,
        cols: list[tuple[str, float, str]],
        color: tuple[int, int, int],
        stripe: bool = False,
        row_h: int = 26,
    ) -> int:
        """Draw a crisp column-aligned table row inside `rect` and return next y."""
        if stripe:
            pygame.draw.rect(
                screen,
                (22, 33, 50),
                pygame.Rect(rect.left + 10, y - 3, rect.width - 20, row_h),
                border_radius=4,
            )
        inner_w = rect.width - 28
        for text, frac, align in cols:
            surf = self._font_xs.render(text, True, color)
            x = rect.left + 14 + int(inner_w * frac)
            if align == "right":
                x -= surf.get_width()
            screen.blit(surf, (x, y))
        return y + row_h

    def _draw_report_overview(self, screen: pygame.Surface, width: int, height: int, data: ReportData) -> None:
        # Left card: Dataset & Model Comparison table
        left = pygame.Rect(int(width * 0.04), int(height * 0.20), int(width * 0.50), int(height * 0.67))
        pygame.draw.rect(screen, (16, 24, 38), left, border_radius=10)
        pygame.draw.rect(screen, (75, 115, 160), left, 1, border_radius=10)

        y = left.top + 14
        screen.blit(self._font_md.render("DATASET & MODEL COMPARISON", True, (130, 235, 255)), (left.left + 14, y))
        y += 32
        summary_lines = [
            f"Selected Best Model : {data.selected_model.upper()} (models/best_model.joblib)",
            f"Raw Dataset Samples : {data.total_raw_samples} across {len(data.session_counts)} sessions",
            f"Train / Test Split  : {data.train_samples} train / {data.test_samples} test (session-grouped)",
            f"Handedness Balance  : " + ", ".join(f"{k}={v}" for k, v in sorted(data.handedness_counts.items())),
        ]
        for line in summary_lines:
            screen.blit(self._font_xs.render(line, True, (220, 235, 248)), (left.left + 14, y))
            y += 23

        y += 16
        col_specs = [0.00, 0.42, 0.56, 0.70, 0.84, 0.99]
        y = self._draw_table_row(
            screen,
            left,
            y,
            [
                ("MODEL", col_specs[0], "left"),
                ("ACC", col_specs[1], "right"),
                ("PREC", col_specs[2], "right"),
                ("REC", col_specs[3], "right"),
                ("F1", col_specs[4], "right"),
                ("MS", col_specs[5], "right"),
            ],
            color=(130, 245, 200),
            row_h=24,
        )
        pygame.draw.line(screen, (60, 88, 125), (left.left + 12, y - 4), (left.right - 12, y - 4), 1)
        for r_idx, row in enumerate(data.comparison_rows):
            m = row.get("model", "").upper()
            is_best = row.get("model", "") == data.selected_model
            col = (255, 230, 120) if is_best else (230, 240, 250)
            y = self._draw_table_row(
                screen,
                left,
                y,
                [
                    (m, col_specs[0], "left"),
                    (f"{float(row.get('accuracy', 0)):.2%}", col_specs[1], "right"),
                    (f"{float(row.get('precision_macro', 0)):.4f}", col_specs[2], "right"),
                    (f"{float(row.get('recall_macro', 0)):.4f}", col_specs[3], "right"),
                    (f"{float(row.get('f1_macro', 0)):.4f}", col_specs[4], "right"),
                    (f"{float(row.get('latency_ms_per_sample', 0)):.4f}", col_specs[5], "right"),
                ],
                color=col,
                stripe=(r_idx % 2 == 0),
                row_h=28,
            )

        # Right card: Model Comparison Plot
        right = pygame.Rect(int(width * 0.56), int(height * 0.20), int(width * 0.40), int(height * 0.67))
        pygame.draw.rect(screen, (16, 24, 38), right, border_radius=10)
        pygame.draw.rect(screen, (75, 115, 160), right, 1, border_radius=10)
        plot = self._get_background(RESULTS_PLOTS_DIR / "model_comparison.png", right.width - 16, right.height - 44)
        screen.blit(self._font_sm.render("ACCURACY / F1 / RECALL CHART", True, (130, 235, 255)), (right.left + 12, right.top + 10))
        if plot is not None:
            screen.blit(plot, (right.left + 8, right.top + 36))

    def _draw_report_classes(
        self,
        screen: pygame.Surface,
        width: int,
        height: int,
        data: ReportData,
        state: ReportState,
    ) -> None:
        for btn_id, label, box in MODEL_SELECT_BOXES:
            rect = _to_px_rect(box, width, height)
            mname = btn_id[6:]
            is_sel = state.selected_model == mname
            is_hov = state.hovered_target == btn_id
            bg = (55, 115, 165) if is_sel else ((38, 70, 108) if is_hov else (22, 34, 52))
            bd = (130, 255, 195) if is_sel else (85, 115, 155)
            pygame.draw.rect(screen, bg, rect, border_radius=7)
            pygame.draw.rect(screen, bd, rect, 2, border_radius=7)
            txt = self._font_xs.render(label, True, (255, 255, 255))
            screen.blit(txt, (rect.centerx - txt.get_width() // 2, rect.centery - txt.get_height() // 2))

        # Left table: per-class precision/recall/f1/support
        left = pygame.Rect(int(width * 0.04), int(height * 0.275), int(width * 0.49), int(height * 0.595))
        pygame.draw.rect(screen, (16, 24, 38), left, border_radius=10)
        pygame.draw.rect(screen, (75, 115, 160), left, 1, border_radius=10)
        rep = data.classification_reports.get(state.selected_model, {})
        y = left.top + 14
        screen.blit(
            self._font_sm.render(f"PER-CLASS METRICS — {state.selected_model.upper()}", True, (130, 235, 255)),
            (left.left + 14, y),
        )
        y += 30
        col_specs = [0.00, 0.42, 0.60, 0.78, 0.99]
        y = self._draw_table_row(
            screen,
            left,
            y,
            [
                ("GESTURE", col_specs[0], "left"),
                ("PREC", col_specs[1], "right"),
                ("REC", col_specs[2], "right"),
                ("F1", col_specs[3], "right"),
                ("SUPP", col_specs[4], "right"),
            ],
            color=(130, 245, 200),
            row_h=24,
        )
        pygame.draw.line(screen, (60, 88, 125), (left.left + 12, y - 4), (left.right - 12, y - 4), 1)
        classes = data.classes or ("fist", "open_palm", "pinch", "pointing", "thumbs_up", "two_fingers")
        for r_idx, cls_name in enumerate(classes):
            m = rep.get(cls_name, {})
            y = self._draw_table_row(
                screen,
                left,
                y,
                [
                    (cls_name, col_specs[0], "left"),
                    (f"{m.get('precision', 0):.4f}", col_specs[1], "right"),
                    (f"{m.get('recall', 0):.4f}", col_specs[2], "right"),
                    (f"{m.get('f1-score', 0):.4f}", col_specs[3], "right"),
                    (f"{int(m.get('support', 0))}", col_specs[4], "right"),
                ],
                color=(230, 240, 250),
                stripe=(r_idx % 2 == 0),
                row_h=27,
            )

        macro = rep.get("macro avg", {})
        if macro:
            y += 4
            pygame.draw.line(screen, (60, 88, 125), (left.left + 12, y - 2), (left.right - 12, y - 2), 1)
            y = self._draw_table_row(
                screen,
                left,
                y + 4,
                [
                    (f"MACRO AVG (Acc {rep.get('accuracy', 0):.2%})", col_specs[0], "left"),
                    (f"{macro.get('precision', 0):.4f}", col_specs[1], "right"),
                    (f"{macro.get('recall', 0):.4f}", col_specs[2], "right"),
                    (f"{macro.get('f1-score', 0):.4f}", col_specs[3], "right"),
                    (f"{int(macro.get('support', 0))}", col_specs[4], "right"),
                ],
                color=(255, 230, 120),
                row_h=26,
            )

        # Right card: Confusion matrix PNG
        right = pygame.Rect(int(width * 0.55), int(height * 0.20), int(width * 0.41), int(height * 0.67))
        pygame.draw.rect(screen, (16, 24, 38), right, border_radius=10)
        pygame.draw.rect(screen, (75, 115, 160), right, 1, border_radius=10)
        cm_path = RESULTS_PLOTS_DIR / f"cm_{state.selected_model}.png"
        cm_img = self._get_background(cm_path, right.width - 16, right.height - 40)
        screen.blit(self._font_sm.render(f"CONFUSION MATRIX ({state.selected_model.upper()})", True, (130, 235, 255)), (right.left + 12, right.top + 10))
        if cm_img is not None:
            screen.blit(cm_img, (right.left + 8, right.top + 34))

    def _draw_report_runtime(self, screen: pygame.Surface, width: int, height: int, data: ReportData) -> None:
        left = pygame.Rect(int(width * 0.04), int(height * 0.20), int(width * 0.45), int(height * 0.67))
        right = pygame.Rect(int(width * 0.51), int(height * 0.20), int(width * 0.45), int(height * 0.67))
        for box in (left, right):
            pygame.draw.rect(screen, (16, 24, 38), box, border_radius=10)
            pygame.draw.rect(screen, (75, 115, 160), box, 1, border_radius=10)

        y = left.top + 14
        screen.blit(self._font_md.render("RUNTIME BENCHMARK", True, (130, 235, 255)), (left.left + 14, y))
        y += 36
        for k, v in data.runtime_benchmark.items():
            screen.blit(self._font_sm.render(f"{k}: {v}", True, (225, 238, 250)), (left.left + 14, y))
            y += 28

        y2 = right.top + 14
        screen.blit(self._font_md.render("ML PIPELINE & SESSIONS", True, (130, 235, 255)), (right.left + 14, y2))
        y2 += 34
        pipeline_lines = [
            "Landmarks : 21 MediaPipe (x, y, z) = 63 features",
            "Norm      : Wrist-centered, max-Euclidean scaled",
            "Geom      : +15 invariant distances & curl ratios (78 total)",
            "RF Config : 200 trees, max_depth=18, balanced",
            "SVM Config: RBF kernel, C=5.0 + StandardScaler",
            "KNN Config: k=7, distance-weighted + StandardScaler",
            "Live Gate : 3-frame vote, min_votes=2, 0.35s cooldown",
            "Sessions  : " + ", ".join(f"{k}:{v}" for k, v in sorted(data.session_counts.items())),
        ]
        for line in pipeline_lines:
            screen.blit(self._font_xs.render(line, True, (220, 235, 248)), (right.left + 14, y2))
            y2 += 24

    # ------------------------------------------------------------------
    # Shared HUD & Overlays
    # ------------------------------------------------------------------
    def _draw_npc_dialog(self, screen: pygame.Surface, width: int, height: int, state: GameState) -> None:
        panel = pygame.Surface((int(width * 0.64), int(height * 0.56)), pygame.SRCALPHA)
        panel.fill((12, 20, 34, 242))
        px = (width - panel.get_width()) // 2
        py = int(height * 0.22)
        screen.blit(panel, (px, py))
        pygame.draw.rect(screen, (110, 255, 190), pygame.Rect(px, py, panel.get_width(), panel.get_height()), 2, border_radius=12)

        title = self._font_md.render("DR. VANCE — SECTOR 4 ENGINEER (Pinch/Hold to Ask | Fist=Close)", True, (130, 255, 200))
        screen.blit(title, (px + 74, py + 14))

        # Character portrait badge on top-left of dialogue panel
        p_cx, p_cy = px + 38, py + 44
        pygame.draw.circle(screen, (22, 36, 56), (p_cx, p_cy), 24)
        pygame.draw.circle(screen, (110, 255, 190), (p_cx, p_cy), 24, 2)
        pygame.draw.circle(screen, (224, 188, 162), (p_cx, p_cy - 2), 11)
        pygame.draw.rect(screen, (16, 36, 52), pygame.Rect(p_cx - 9, p_cy - 5, 18, 6), border_radius=2)
        pygame.draw.rect(screen, (110, 255, 190), pygame.Rect(p_cx - 9, p_cy - 5, 18, 6), 1, border_radius=2)

        reply_surf = self._font_sm.render(state.npc_line, True, (235, 245, 255))
        screen.blit(reply_surf, (px + 74, py + 46))

        active = state.hovered_target or state.selected_target
        for idx, box in enumerate(NPC_DIALOG_BOXES):
            rect = _to_px_rect(box, width, height)
            sel = active == f"npc_{idx}"
            pygame.draw.rect(screen, (42, 92, 130) if sel else (24, 38, 58), rect, border_radius=10)
            pygame.draw.rect(screen, (120, 255, 195) if sel else (110, 145, 185), rect, 3 if sel else 2, border_radius=10)
            lbl = self._font_sm.render(NPC_DIALOG_OPTIONS[idx][1], True, (255, 255, 255))
            screen.blit(lbl, (rect.centerx - lbl.get_width() // 2, rect.centery - lbl.get_height() // 2))

    def _draw_safe_panel(self, screen: pygame.Surface, width: int, height: int, state: GameState) -> None:
        panel = pygame.Surface((int(width * 0.64), int(height * 0.56)), pygame.SRCALPHA)
        panel.fill((14, 20, 32, 240))
        px = (width - panel.get_width()) // 2
        py = int(height * 0.22)
        screen.blit(panel, (px, py))
        pygame.draw.rect(screen, (235, 195, 90), pygame.Rect(px, py, panel.get_width(), panel.get_height()), 2, border_radius=12)

        title = self._font_md.render("CHEST KEYPAD (Pinch / Thumbs Up / Hold Point | Fist=Close)", True, (255, 235, 175))
        screen.blit(title, (width // 2 - title.get_width() // 2, py + 12))

        slots = [str(state.dials[i]) if i < len(state.dials) else "_" for i in range(3)]
        code_disp = self._font_lg.render(f"[ {slots[0]} ]   [ {slots[1]} ]   [ {slots[2]} ]", True, (130, 245, 255))
        screen.blit(code_disp, (width // 2 - code_disp.get_width() // 2, py + 44))

        active = state.hovered_target or state.selected_target
        for idx, box in enumerate(CHEST_PAD_BOXES):
            rect = _to_px_rect(box, width, height)
            sel = active == f"pad_{idx}"
            pygame.draw.rect(screen, (55, 105, 165) if sel else (28, 42, 65), rect, border_radius=10)
            pygame.draw.rect(screen, (120, 245, 255) if sel else (150, 170, 200), rect, 4 if sel else 2, border_radius=10)
            d_surf = self._font_md.render(str(CHEST_PAD_VALUES[idx]), True, (255, 255, 255))
            screen.blit(d_surf, (rect.centerx - d_surf.get_width() // 2, rect.centery - d_surf.get_height() // 2))

        clr_rect = _to_px_rect(CHEST_CLEAR_BOX, width, height)
        sel_c = active == "clear"
        pygame.draw.rect(screen, (170, 70, 75) if sel_c else (115, 45, 50), clr_rect, border_radius=10)
        pygame.draw.rect(screen, (255, 180, 180), clr_rect, 3 if sel_c else 2, border_radius=10)
        clbl = self._font_sm.render("CLEAR CODE", True, (255, 255, 255))
        screen.blit(clbl, (clr_rect.centerx - clbl.get_width() // 2, clr_rect.centery - clbl.get_height() // 2))

    def _draw_exit_panel(self, screen: pygame.Surface, width: int, height: int, state: GameState) -> None:
        panel = pygame.Surface((int(width * 0.60), int(height * 0.43)), pygame.SRCALPHA)
        panel.fill((14, 20, 32, 238))
        px = (width - panel.get_width()) // 2
        py = int(height * 0.25)
        screen.blit(panel, (px, py))
        pygame.draw.rect(screen, (130, 235, 255), pygame.Rect(px, py, panel.get_width(), panel.get_height()), 2, border_radius=12)

        entered = " - ".join(str(d) for d in state.exit_entered) or "_ - _ - _"
        title = self._font_md.render(f"EXIT KEYPAD | Entered: {entered}", True, (240, 248, 255))
        screen.blit(title, (width // 2 - title.get_width() // 2, py + 16))

        active = state.hovered_target or state.selected_target
        for idx, box in enumerate(EXIT_PAD_BOXES):
            rect = _to_px_rect(box, width, height)
            sel = active == f"exit_{idx}"
            pygame.draw.rect(screen, (70, 125, 200) if sel else (30, 50, 85), rect, border_radius=10)
            pygame.draw.rect(screen, (140, 245, 255) if sel else (150, 180, 210), rect, 4 if sel else 2, border_radius=10)
            val = self._font_lg.render(str(EXIT_PAD_VALUES[idx]), True, (255, 255, 255))
            screen.blit(val, (rect.centerx - val.get_width() // 2, rect.centery - val.get_height() // 2))

        clr_rect = _to_px_rect(EXIT_CLEAR_BOX, width, height)
        sel_c = active == "exit_clear"
        pygame.draw.rect(screen, (170, 70, 75) if sel_c else (115, 45, 50), clr_rect, border_radius=10)
        pygame.draw.rect(screen, (255, 180, 180), clr_rect, 3 if sel_c else 2, border_radius=10)
        clbl = self._font_sm.render("CLEAR CODE", True, (255, 255, 255))
        screen.blit(clbl, (clr_rect.centerx - clbl.get_width() // 2, clr_rect.centery - clbl.get_height() // 2))

    def _draw_completed(self, screen: pygame.Surface, width: int, height: int, state: GameState) -> None:
        overlay = pygame.Surface((width, height), pygame.SRCALPHA)
        overlay.fill((10, 30, 25, 195))
        screen.blit(overlay, (0, 0))

        title = self._font_lg.render("SECTOR 4 ESCAPED!", True, (130, 255, 185))
        time_txt = self._font_md.render(f"Clear Time: {state.completion_seconds or 0.0:.1f}s", True, (235, 245, 250))
        screen.blit(title, (width // 2 - title.get_width() // 2, int(height * 0.34)))
        screen.blit(time_txt, (width // 2 - time_txt.get_width() // 2, int(height * 0.44)))

        restart_rect = _to_px_rect(RESTART_BOX, width, height)
        sel = (state.hovered_target or state.selected_target) == "restart"
        pygame.draw.rect(screen, (55, 170, 115) if sel else (35, 120, 85), restart_rect, border_radius=10)
        pygame.draw.rect(screen, (210, 255, 230), restart_rect, 3 if sel else 2, border_radius=10)
        lbl = self._font_md.render("PLAY AGAIN (R)", True, (255, 255, 255))
        screen.blit(lbl, (restart_rect.centerx - lbl.get_width() // 2, restart_rect.centery - lbl.get_height() // 2))

    def _draw_controls_menu(self, screen: pygame.Surface, width: int, height: int) -> None:
        panel = pygame.Surface((int(width * 0.74), int(height * 0.56)), pygame.SRCALPHA)
        panel.fill((15, 22, 34, 240))
        px = (width - panel.get_width()) // 2
        py = (height - panel.get_height()) // 2
        screen.blit(panel, (px, py))
        pygame.draw.rect(screen, (110, 210, 255), pygame.Rect(px, py, panel.get_width(), panel.get_height()), 2, border_radius=12)

        title = self._font_md.render("GESTURE CONTROLS (Fist to close | M = Game Hub)", True, (130, 230, 255))
        screen.blit(title, (px + 22, py + 16))
        custom_map = {
            "open_palm": "Open this menu",
            "fist": "Close menu / Shoot (Soccer) / Clear",
            "pointing": "Aim POV reticle / Move & Dribble (Soccer)",
            "thumbs_up": "Interact / Confirm",
            "pinch": "Grab & drag Holo Core",
            "two_fingers": "Replay narration / Pass (Soccer)",
        }
        y = py + 58
        for gesture, desc in custom_map.items():
            line = self._font_sm.render(f"{gesture:12s}  —  {desc}", True, (235, 240, 245))
            screen.blit(line, (px + 26, y))
            y += 32

    def _draw_status_strip(
        self,
        screen: pygame.Surface,
        width: int,
        title: str,
        prediction: GesturePrediction,
        fps: float,
        model_name: str,
        secondary_prediction: GesturePrediction | None = None,
    ) -> None:
        hud = pygame.Surface((width, 56), pygame.SRCALPHA)
        hud.fill((10, 15, 24, 215))
        screen.blit(hud, (0, 0))
        top_text = f"{title}   |   {model_name}   |   FPS: {fps:4.1f}   |   F11=Fullscreen  M=Hub  R=Restart"
        screen.blit(self._font_sm.render(top_text, True, (235, 242, 250)), (16, 6))

        if prediction.status == "no_hand":
            st_txt, col = "Hand 1: None", (255, 200, 110)
        elif prediction.status == "uncertain":
            st_txt, col = f"Hand 1: Uncertain ({prediction.confidence:.2f})", (255, 170, 100)
        elif prediction.status == "stabilizing":
            st_txt, col = f"Hand 1: Stabilizing [{prediction.raw_label}]", (170, 215, 255)
        else:
            st_txt, col = f"Hand 1: {prediction.smoothed_label} ({prediction.confidence:.2f})", (120, 240, 175)
        h1_surf = self._font_xs.render(st_txt, True, col)
        screen.blit(h1_surf, (16, 31))

        if secondary_prediction is not None and secondary_prediction.status != "no_hand":
            if secondary_prediction.status == "stable":
                h2_txt = f"  |  Hand 2: {secondary_prediction.smoothed_label} ({secondary_prediction.confidence:.2f})"
                h2_col = (255, 205, 105)
            else:
                h2_txt = f"  |  Hand 2: {secondary_prediction.status}"
                h2_col = (200, 185, 140)
            screen.blit(self._font_xs.render(h2_txt, True, h2_col), (16 + h1_surf.get_width(), 31))

    def _draw_webcam_pip(
        self,
        screen: pygame.Surface,
        camera_surface: pygame.Surface,
        width: int,
        height: int,
    ) -> None:
        pip_w, pip_h = 192, 108
        pip = pygame.transform.smoothscale(camera_surface, (pip_w, pip_h))
        px = width - pip_w - 14
        py = 64
        screen.blit(pip, (px, py))
        pygame.draw.rect(screen, (90, 200, 255), pygame.Rect(px, py, pip_w, pip_h), 2, border_radius=6)

    def _draw_pointer(
        self,
        screen: pygame.Surface,
        width: int,
        height: int,
        pointer_xy: tuple[float, float] | None,
        smoothed_label: str | None,
    ) -> None:
        if pointer_xy is None:
            return
        px = int(max(0.0, min(1.0, pointer_xy[0])) * width)
        py = int(max(0.0, min(1.0, pointer_xy[1])) * height)
        color = (255, 205, 80) if smoothed_label == "pinch" else (90, 245, 255)
        pygame.draw.circle(screen, color, (px, py), 11, 2)
        pygame.draw.circle(screen, (255, 255, 255), (px, py), 3)

    def _draw_narration_and_feedback(
        self,
        screen: pygame.Surface,
        width: int,
        height: int,
        state: GameState,
    ) -> None:
        bar_h = 76
        bar = pygame.Surface((width, bar_h), pygame.SRCALPHA)
        bar.fill((10, 15, 24, 225))
        screen.blit(bar, (0, height - bar_h))

        subtitle = f"NARRATOR: \"{state.narration_subtitle}\""
        if len(subtitle) > 106:
            subtitle = subtitle[:103] + "...\""
        screen.blit(self._font_sm.render(subtitle, True, (255, 235, 165)), (16, height - 66))

        colors = {
            "info": (215, 230, 245),
            "success": (120, 245, 175),
            "warning": (255, 205, 110),
            "error": (255, 120, 120),
        }
        color = colors.get(state.feedback_kind, (215, 230, 245))
        screen.blit(self._font_xs.render(state.feedback_message, True, color), (16, height - 34))
