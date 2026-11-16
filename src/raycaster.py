"""Real-time NumPy-accelerated 3D Raycasting engine for HoloEscape."""

from __future__ import annotations

import math
import numpy as np
import pygame


# 10x10 Sector 4 Chamber:
# 0 = empty, 1 = titanium wall, 2 = cipher terminal wall, 3 = exit airlock, 4 = portal wall
ROOM_MAP = np.array(
    [
        [1, 1, 1, 1, 3, 3, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [2, 0, 0, 0, 0, 0, 0, 0, 0, 4],
        [2, 0, 0, 0, 0, 0, 0, 0, 0, 4],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 1, 0, 0, 1, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ],
    dtype=np.int32,
)

MAP_H, MAP_W = ROOM_MAP.shape

# World positions (grid coordinates) of interactive 3D stations
WORLD_TERMINAL = (1.6, 4.0)
WORLD_SAFE = (5.0, 4.8)
WORLD_HOLO_CORE = (3.4, 5.2)
WORLD_PORTAL = (8.3, 4.0)
WORLD_EXIT = (5.0, 1.3)


def pos_norm_to_world(pos_norm: tuple[float, float] | list[float]) -> tuple[float, float]:
    """Convert normalized (0..1, 0..1) room coordinates to 3D grid coordinates."""
    wx = 1.2 + max(0.0, min(1.0, float(pos_norm[0]))) * (MAP_W - 2.4)
    wy = 1.2 + max(0.0, min(1.0, float(pos_norm[1]))) * (MAP_H - 2.4)
    return wx, wy


def world_to_pos_norm(wx: float, wy: float) -> tuple[float, float]:
    """Convert 3D grid coordinates back to normalized (0..1, 0..1) coordinates."""
    nx = (wx - 1.2) / (MAP_W - 2.4)
    ny = (wy - 1.2) / (MAP_H - 2.4)
    return max(0.0, min(1.0, nx)), max(0.0, min(1.0, ny))


def is_walkable(wx: float, wy: float, margin: float = 0.28) -> bool:
    """Return True if a circle of radius `margin` around (wx, wy) is free of walls."""
    for ox in (-margin, margin):
        for oy in (-margin, margin):
            gx = int(wx + ox)
            gy = int(wy + oy)
            if gx < 0 or gx >= MAP_W or gy < 0 or gy >= MAP_H:
                return False
            if ROOM_MAP[gy, gx] != 0:
                return False
    return True


class RaycastRenderer:
    """Fast vectorized 3D raycaster with shaded walls, floor/ceiling, and 3D sprites."""

    def __init__(self, ray_w: int = 320, ray_h: int = 180) -> None:
        self.ray_w = ray_w
        self.ray_h = ray_h
        self.fov = math.radians(66.0)
        self._col_offsets = np.linspace(-0.5, 0.5, ray_w, dtype=np.float32) * self.fov

        # Precompute ceiling & floor vertical gradients
        y_idx = np.arange(ray_h, dtype=np.float32)
        half = ray_h / 2.0
        ceil_t = np.clip((half - y_idx[: int(half)]) / half, 0.0, 1.0)[:, None]
        floor_t = np.clip((y_idx[int(half) :] - half) / half, 0.0, 1.0)[:, None]

        self._ceil_bg = np.zeros((int(half), ray_w, 3), dtype=np.uint8)
        self._ceil_bg[:, :, 0] = (10 + 12 * ceil_t).astype(np.uint8)
        self._ceil_bg[:, :, 1] = (14 + 18 * ceil_t).astype(np.uint8)
        self._ceil_bg[:, :, 2] = (24 + 28 * ceil_t).astype(np.uint8)

        self._floor_bg = np.zeros((ray_h - int(half), ray_w, 3), dtype=np.uint8)
        self._floor_bg[:, :, 0] = (14 + 22 * floor_t).astype(np.uint8)
        self._floor_bg[:, :, 1] = (18 + 26 * floor_t).astype(np.uint8)
        self._floor_bg[:, :, 2] = (26 + 34 * floor_t).astype(np.uint8)

        self._wall_colors = {
            1: np.array([58, 76, 102], dtype=np.float32),
            2: np.array([38, 175, 130], dtype=np.float32),
            3: np.array([215, 120, 55], dtype=np.float32),
            4: np.array([105, 145, 245], dtype=np.float32),
        }

    def render_3d_view(
        self,
        screen: pygame.Surface,
        wx: float,
        wy: float,
        yaw: float,
        bob_px: int,
        stage: str,
        artifact_world: tuple[float, float],
        artifact_grabbed: bool,
        t: float,
    ) -> np.ndarray:
        """Render the 3D raycasted chamber onto `screen` and return the 1D Z-buffer."""
        w, h = screen.get_size()
        rw, rh = self.ray_w, self.ray_h
        half = rh // 2

        frame = np.empty((rh, rw, 3), dtype=np.uint8)
        frame[:half] = self._ceil_bg
        frame[half:] = self._floor_bg

        ray_angles = yaw + self._col_offsets
        cos_a = np.cos(ray_angles)
        sin_a = np.sin(ray_angles)
        # Fix fisheye distortion
        cos_rel = np.cos(self._col_offsets)

        z_buf = np.full(rw, 20.0, dtype=np.float32)
        wall_ids = np.ones(rw, dtype=np.int32)
        hit_u = np.zeros(rw, dtype=np.float32)
        side_shade = np.ones(rw, dtype=np.float32)

        # Vectorized ray march (step 0.045 up to 14 units, then 1 refinement)
        step_size = 0.045
        max_steps = 300
        active = np.ones(rw, dtype=bool)
        dists = np.zeros(rw, dtype=np.float32)

        for s in range(1, max_steps):
            if not np.any(active):
                break
            d = s * step_size
            rx = wx + cos_a[active] * d
            ry = wy + sin_a[active] * d
            gx = np.clip(rx.astype(np.int32), 0, MAP_W - 1)
            gy = np.clip(ry.astype(np.int32), 0, MAP_H - 1)
            cell = ROOM_MAP[gy, gx]
            hit_mask = cell != 0
            if np.any(hit_mask):
                active_idx = np.flatnonzero(active)
                hit_cols = active_idx[hit_mask]
                dists[hit_cols] = d
                wall_ids[hit_cols] = cell[hit_mask]
                hx = rx[hit_mask] - np.floor(rx[hit_mask])
                hy = ry[hit_mask] - np.floor(ry[hit_mask])
                # Determine which face was hit for panel seams & shading
                near_x = np.minimum(hx, 1.0 - hx)
                near_y = np.minimum(hy, 1.0 - hy)
                x_face = near_x < near_y
                hit_u[hit_cols] = np.where(x_face, hy, hx)
                side_shade[hit_cols] = np.where(x_face, 0.78, 1.0)
                active[hit_cols] = False

        perp_dist = np.maximum(0.15, dists * cos_rel)
        z_buf[:] = perp_dist

        wall_h = np.minimum(rh * 2, (rh * 0.92 / perp_dist).astype(np.int32))
        y0 = np.clip(half - wall_h // 2 + bob_px // 4, 0, rh - 1)
        y1 = np.clip(half + wall_h // 2 + bob_px // 4, 0, rh)

        # Draw vertical wall columns with distance fog & sci-fi panel seam lines
        for col in range(rw):
            top = int(y0[col])
            bot = int(y1[col])
            if bot <= top:
                continue
            wid = int(wall_ids[col])
            if wid == 3 and stage == "completed":
                base = np.array([75, 235, 165], dtype=np.float32)
            else:
                base = self._wall_colors.get(wid, self._wall_colors[1])
            fog = max(0.22, min(1.0, 1.15 - perp_dist[col] * 0.085))
            u = float(hit_u[col])
            seam = 0.65 if (u < 0.04 or u > 0.96 or abs(u - 0.5) < 0.02) else 1.0
            rgb = np.clip(base * side_shade[col] * fog * seam, 0, 255).astype(np.uint8)
            frame[top:bot, col] = rgb
            # Glowing neon trim strip along top/bottom of wall
            if bot - top > 10:
                trim_col = (95, 235, 255) if wid in (1, 4) else ((110, 255, 175) if wid == 2 else (255, 185, 85))
                frame[top : min(bot, top + 2), col] = trim_col

        surf = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        scaled = pygame.transform.scale(surf, (w, h))
        screen.blit(scaled, (0, 0))

        # Render 3D Billboard Props sorted back-to-front with Z-buffer check
        sprites: list[tuple[float, str, tuple[float, float]]] = [
            (math.hypot(WORLD_SAFE[0] - wx, WORLD_SAFE[1] - wy), "safe", WORLD_SAFE),
            (math.hypot(WORLD_TERMINAL[0] - wx, WORLD_TERMINAL[1] - wy), "terminal", WORLD_TERMINAL),
            (math.hypot(WORLD_PORTAL[0] - wx, WORLD_PORTAL[1] - wy), "portal", WORLD_PORTAL),
            (math.hypot(WORLD_EXIT[0] - wx, WORLD_EXIT[1] - wy), "exit", WORLD_EXIT),
        ]
        if stage in ("mystery_object", "final_exit", "completed") and not artifact_grabbed:
            sprites.append(
                (math.hypot(artifact_world[0] - wx, artifact_world[1] - wy), "holo_core", artifact_world)
            )
        sprites.sort(key=lambda item: item[0], reverse=True)

        for dist, kind, (sx, sy) in sprites:
            if dist < 0.25:
                continue
            ang = math.atan2(sy - wy, sx - wx) - yaw
            while ang > math.pi:
                ang -= 2.0 * math.pi
            while ang < -math.pi:
                ang += 2.0 * math.pi
            if abs(ang) > self.fov * 0.72:
                continue

            screen_x = int((0.5 + ang / self.fov) * w)
            col_idx = max(0, min(rw - 1, int((0.5 + ang / self.fov) * rw)))
            if dist > z_buf[col_idx] + 0.35:
                continue

            sp_h = int((h * 0.55) / max(0.4, dist))
            sp_y = h // 2 + int((h * 0.14) / max(0.4, dist)) + bob_px
            self._draw_3d_prop(screen, kind, screen_x, sp_y, sp_h, stage, t)

        return z_buf

    def _draw_3d_prop(
        self,
        screen: pygame.Surface,
        kind: str,
        cx: int,
        cy: int,
        size: int,
        stage: str,
        t: float,
    ) -> None:
        if kind == "safe":
            vw = max(36, int(size * 1.15))
            vh = max(24, int(size * 0.58))
            chest_open = stage != "chest_lock"
            led = (110, 255, 175) if chest_open else (75, 225, 255)
            body = pygame.Rect(cx - vw // 2, cy - vh // 2, vw, vh)
            pygame.draw.rect(screen, (26, 36, 52), body, border_radius=6)
            pygame.draw.rect(screen, led, body, 2, border_radius=6)
            pygame.draw.circle(screen, led, body.center, max(4, vh // 5))
        elif kind == "terminal":
            tw, th = max(26, int(size * 0.7)), max(34, int(size * 0.9))
            rect = pygame.Rect(cx - tw // 2, cy - th // 2 - size // 4, tw, th)
            pygame.draw.rect(screen, (14, 42, 34), rect, border_radius=5)
            pygame.draw.rect(screen, (95, 255, 175), rect, 2, border_radius=5)
        elif kind == "portal":
            pr = max(16, int(size * 0.55))
            py = cy - size // 5
            col = (120, 245, 255) if stage == "mystery_object" else (90, 145, 225)
            pygame.draw.circle(screen, col, (cx, py), pr, 3)
            pygame.draw.circle(screen, (190, 245, 255), (cx, py), max(4, int(pr * (0.55 + 0.1 * math.sin(t * 4)))), 2)
        elif kind == "holo_core":
            bob = int(math.sin(t * 4.5) * 6)
            r = max(10, int(size * 0.28))
            py = cy - size // 5 + bob
            pts = [(cx, py - r), (cx + int(r * 0.75), py), (cx, py + r), (cx - int(r * 0.75), py)]
            pygame.draw.polygon(screen, (95, 225, 255), pts)
            pygame.draw.polygon(screen, (255, 255, 255), pts, 2)
        elif kind == "exit":
            ew, eh = max(32, int(size * 0.85)), max(44, int(size * 1.15))
            rect = pygame.Rect(cx - ew // 2, cy - eh // 2 - size // 5, ew, eh)
            col = (110, 255, 175) if stage == "completed" else (255, 185, 85)
            pygame.draw.rect(screen, (24, 28, 38), rect, border_radius=4)
            pygame.draw.rect(screen, col, rect, 2, border_radius=4)
