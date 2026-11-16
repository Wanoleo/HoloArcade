"""Gesture-controlled 4v4 Football (HoloSoccer) with passing, 3 teammates, and 3 opponents."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import time


PITCH_LEFT = 0.18
PITCH_RIGHT = 0.82
PITCH_TOP = 0.18
PITCH_BOTTOM = 0.82
GOAL_Y_MIN = 0.43
GOAL_Y_MAX = 0.57


@dataclass
class FieldPlayer:
    """Represents one outfield player on either team."""

    pos: list[float]
    home_pos: tuple[float, float]
    vel: list[float] = field(default_factory=lambda: [0.0, 0.0])


def _default_teammates() -> list[FieldPlayer]:
    # Index 0 = initial controlled striker; 1..3 = 3 AI teammates
    return [
        FieldPlayer(pos=[0.38, 0.50], home_pos=(0.38, 0.50)),
        FieldPlayer(pos=[0.46, 0.28], home_pos=(0.48, 0.28)),
        FieldPlayer(pos=[0.46, 0.72], home_pos=(0.48, 0.72)),
        FieldPlayer(pos=[0.27, 0.50], home_pos=(0.27, 0.50)),
    ]


def _default_opponents() -> list[FieldPlayer]:
    # 3 AI outfield opponents (+ 1 AI goalkeeper)
    return [
        FieldPlayer(pos=[0.62, 0.50], home_pos=(0.60, 0.50)),
        FieldPlayer(pos=[0.66, 0.30], home_pos=(0.66, 0.30)),
        FieldPlayer(pos=[0.66, 0.70], home_pos=(0.66, 0.70)),
    ]


@dataclass
class SoccerState:
    """Complete mutable state for a 4v4 HoloSoccer match."""

    teammates: list[FieldPlayer] = field(default_factory=_default_teammates)
    opponents: list[FieldPlayer] = field(default_factory=_default_opponents)
    controlled_idx: int = 0

    ball_pos: list[float] = field(default_factory=lambda: [0.46, 0.50])
    ball_vel: list[float] = field(default_factory=lambda: [0.0, 0.0])
    ball_height: float = 0.0
    ball_vz: float = 0.0
    ball_spin: float = 0.0
    ball_trail: list[tuple[float, float]] = field(default_factory=list)
    keeper_pos: list[float] = field(default_factory=lambda: [0.80, 0.50])
    keeper_dir: float = 1.0

    score_player: int = 0
    score_ai: int = 0
    time_remaining: float = 60.0
    match_over: bool = False
    goal_banner_timer: float = 0.0
    shot_flash_timer: float = 0.0
    pass_line: tuple[tuple[float, float], tuple[float, float]] | None = None
    pass_flash_timer: float = 0.0

    feedback_message: str = "Point: Move/Dribble | Two Fingers: Pass | Fist: Shoot!"
    feedback_kind: str = "info"
    last_update_time: float = field(default_factory=time.monotonic)


class HoloSoccerGame:
    """Decoupled 4v4 football physics, AI, passing, and shooting driven by gestures."""

    def __init__(self, move_speed: float = 0.095) -> None:
        self.move_speed = move_speed
        self.state = SoccerState()

    @property
    def player_pos(self) -> list[float]:
        return self.state.teammates[self.state.controlled_idx].pos

    def restart(self) -> None:
        """Reset scores, timer, and team formations for a fresh 60s match."""
        self.state = SoccerState()

    def _reset_kickoff(self) -> None:
        st = self.state
        st.teammates = _default_teammates()
        st.opponents = _default_opponents()
        st.controlled_idx = 0
        st.ball_pos = [0.46, 0.50]
        st.ball_vel = [0.0, 0.0]
        st.ball_height = 0.0
        st.ball_vz = 0.0
        st.ball_trail.clear()

    def _select_pass_target(self, pointer_xy: tuple[float, float] | None) -> int:
        st = self.state
        candidates = [i for i in range(len(st.teammates)) if i != st.controlled_idx]
        if pointer_xy is not None:
            return min(
                candidates,
                key=lambda i: math.hypot(
                    st.teammates[i].pos[0] - pointer_xy[0],
                    st.teammates[i].pos[1] - pointer_xy[1],
                ),
            )
        cur = st.teammates[st.controlled_idx].pos
        return min(
            candidates,
            key=lambda i: math.hypot(
                st.teammates[i].pos[0] - cur[0],
                st.teammates[i].pos[1] - cur[1],
            ),
        )

    def update(
        self,
        triggered_gesture: str | None,
        held_gesture: str | None,
        pointer_xy: tuple[float, float] | None,
        gesture_status: str,
        now: float | None = None,
        move_dir: tuple[float, float] = (0.0, 0.0),
        kick_trigger: str | None = None,
    ) -> SoccerState:
        """Advance match physics, teammate/opponent AI, passing, and shooting."""
        current_time = time.monotonic() if now is None else now
        st = self.state
        dt = max(0.0, min(0.08, current_time - st.last_update_time))
        st.last_update_time = current_time

        if st.match_over:
            if triggered_gesture == "thumbs_up":
                self.restart()
            return st

        st.time_remaining = max(0.0, st.time_remaining - dt)
        if st.time_remaining <= 0.0:
            st.match_over = True
            st.feedback_message = (
                f"FULL TIME! Final Score: You {st.score_player} - {st.score_ai} Red (Thumbs Up / R to replay)"
            )
            st.feedback_kind = "success" if st.score_player >= st.score_ai else "info"
            return st

        if st.goal_banner_timer > 0.0:
            st.goal_banner_timer = max(0.0, st.goal_banner_timer - dt)
        if st.shot_flash_timer > 0.0:
            st.shot_flash_timer = max(0.0, st.shot_flash_timer - dt)
        if st.pass_flash_timer > 0.0:
            st.pass_flash_timer = max(0.0, st.pass_flash_timer - dt)
            if st.pass_flash_timer == 0.0:
                st.pass_line = None

        active = st.teammates[st.controlled_idx]

        # 1. Controlled player movement via pointing or WASD/Arrows
        active.vel = [0.0, 0.0]
        if move_dir != (0.0, 0.0):
            mag = max(1e-6, math.hypot(move_dir[0], move_dir[1]))
            vx = (move_dir[0] / mag) * self.move_speed
            vy = (move_dir[1] / mag) * self.move_speed
            active.vel = [vx, vy]
            active.pos[0] = max(PITCH_LEFT + 0.02, min(PITCH_RIGHT - 0.02, active.pos[0] + vx * dt))
            active.pos[1] = max(PITCH_TOP + 0.02, min(PITCH_BOTTOM - 0.02, active.pos[1] + vy * dt))
        elif gesture_status not in ("no_hand", "uncertain") and held_gesture == "pointing" and pointer_xy is not None:
            tx = max(PITCH_LEFT + 0.02, min(PITCH_RIGHT - 0.02, pointer_xy[0]))
            ty = max(PITCH_TOP + 0.02, min(PITCH_BOTTOM - 0.02, pointer_xy[1]))
            dx = tx - active.pos[0]
            dy = ty - active.pos[1]
            dist = math.hypot(dx, dy)
            if dist > 0.015:
                vx = (dx / dist) * self.move_speed
                vy = (dy / dist) * self.move_speed
                active.vel = [vx, vy]
                active.pos[0] += vx * dt
                active.pos[1] += vy * dt

        # 2. Two Fingers (or E) = Pass to nearest/aimed teammate & switch control!
        if triggered_gesture == "two_fingers" or kick_trigger == "pass":
            dist_to_ball = math.hypot(st.ball_pos[0] - active.pos[0], st.ball_pos[1] - active.pos[1])
            target_idx = self._select_pass_target(pointer_xy)
            if dist_to_ball <= 0.11:
                target_pos = st.teammates[target_idx].pos
                pdx = target_pos[0] - st.ball_pos[0]
                pdy = target_pos[1] - st.ball_pos[1]
                pdist = max(1e-4, math.hypot(pdx, pdy))
                pass_speed = 0.36
                st.ball_vel[0] = (pdx / pdist) * pass_speed
                st.ball_vel[1] = (pdy / pdist) * pass_speed
                st.ball_vz = 0.24
                st.pass_line = (tuple(st.ball_pos), tuple(target_pos))
                st.pass_flash_timer = 0.35
                st.controlled_idx = target_idx
                st.feedback_message = "PASS! Switched to receiver."
                st.feedback_kind = "success"
            else:
                st.controlled_idx = target_idx
                st.feedback_message = "Switched to teammate!"
                st.feedback_kind = "info"

        # 3. Fist (or Space) = Power Shot!
        if triggered_gesture == "fist" or kick_trigger == "shoot":
            active = st.teammates[st.controlled_idx]
            dist_to_ball = math.hypot(st.ball_pos[0] - active.pos[0], st.ball_pos[1] - active.pos[1])
            if dist_to_ball <= 0.095:
                target_x = 0.86
                target_y = 0.50
                if pointer_xy is not None and pointer_xy[0] > active.pos[0]:
                    target_y = max(GOAL_Y_MIN + 0.01, min(GOAL_Y_MAX - 0.01, pointer_xy[1]))
                sdx = target_x - st.ball_pos[0]
                sdy = target_y - st.ball_pos[1]
                sdist = max(1e-4, math.hypot(sdx, sdy))
                shot_speed = 0.55
                st.ball_vel[0] = (sdx / sdist) * shot_speed
                st.ball_vel[1] = (sdy / sdist) * shot_speed
                st.ball_vz = 0.36
                st.shot_flash_timer = 0.25
                st.feedback_message = "POWER SHOT!"
                st.feedback_kind = "success"
            else:
                st.feedback_message = "Get closer to the ball before Fist-shooting!"
                st.feedback_kind = "warning"

        # 4. Update 3 AI Teammates (support positions shifting with ball)
        for idx, mate in enumerate(st.teammates):
            if idx == st.controlled_idx:
                continue
            goal_tx = max(
                PITCH_LEFT + 0.03,
                min(PITCH_RIGHT - 0.05, mate.home_pos[0] + 0.35 * (st.ball_pos[0] - 0.50)),
            )
            goal_ty = max(
                PITCH_TOP + 0.03,
                min(PITCH_BOTTOM - 0.03, mate.home_pos[1] + 0.25 * (st.ball_pos[1] - 0.50)),
            )
            self._move_unit_toward(mate, (goal_tx, goal_ty), 0.06, dt)

        # 5. Update 3 AI Opponents (closest presses the ball, others hold formation)
        chaser_idx = min(
            range(len(st.opponents)),
            key=lambda i: math.hypot(
                st.opponents[i].pos[0] - st.ball_pos[0],
                st.opponents[i].pos[1] - st.ball_pos[1],
            ),
        )
        for idx, opp in enumerate(st.opponents):
            if idx == chaser_idx:
                self._move_unit_toward(opp, (st.ball_pos[0], st.ball_pos[1]), 0.055, dt)
            else:
                tx = max(PITCH_LEFT + 0.06, min(PITCH_RIGHT - 0.04, opp.home_pos[0] + 0.25 * (st.ball_pos[0] - 0.50)))
                ty = max(PITCH_TOP + 0.03, min(PITCH_BOTTOM - 0.03, opp.home_pos[1] + 0.25 * (st.ball_pos[1] - 0.50)))
                self._move_unit_toward(opp, (tx, ty), 0.05, dt)

            # Opponent ball kick / tackle
            odist = math.hypot(st.ball_pos[0] - opp.pos[0], st.ball_pos[1] - opp.pos[1])
            if odist < 0.034 and st.shot_flash_timer <= 0.0 and st.pass_flash_timer <= 0.0:
                kdx = PITCH_LEFT - st.ball_pos[0]
                kdy = 0.50 - st.ball_pos[1]
                kdist = max(1e-4, math.hypot(kdx, kdy))
                st.ball_vel[0] = (kdx / kdist) * 0.30
                st.ball_vel[1] = (kdy / kdist) * 0.30
                st.feedback_message = "Opponent intercepted! Chase it down!"
                st.feedback_kind = "warning"

        # 6. Controlled magnet-dribble for active player
        active = st.teammates[st.controlled_idx]
        bdx = st.ball_pos[0] - active.pos[0]
        bdy = st.ball_pos[1] - active.pos[1]
        bdist = math.hypot(bdx, bdy)
        if 1e-5 < bdist < 0.040 and st.shot_flash_timer <= 0.0 and st.pass_flash_timer <= 0.0:
            nx, ny = bdx / bdist, bdy / bdist
            st.ball_pos[0] = active.pos[0] + nx * 0.036
            st.ball_pos[1] = active.pos[1] + ny * 0.036
            st.ball_vel[0] = active.vel[0] * 1.02 + nx * 0.04
            st.ball_vel[1] = active.vel[1] * 1.02 + ny * 0.04

        # 7. AI Goalkeeper patrol & save
        keeper_speed = 0.07
        st.keeper_pos[1] += st.keeper_dir * keeper_speed * dt
        if st.keeper_pos[1] >= GOAL_Y_MAX - 0.015:
            st.keeper_pos[1] = GOAL_Y_MAX - 0.015
            st.keeper_dir = -1.0
        elif st.keeper_pos[1] <= GOAL_Y_MIN + 0.015:
            st.keeper_pos[1] = GOAL_Y_MIN + 0.015
            st.keeper_dir = 1.0

        kdist = math.hypot(st.ball_pos[0] - st.keeper_pos[0], st.ball_pos[1] - st.keeper_pos[1])
        if kdist < 0.038 and st.ball_vel[0] > 0:
            st.ball_vel[0] = -abs(st.ball_vel[0]) * 0.75
            st.ball_vel[1] += st.keeper_dir * 0.15
            st.feedback_message = "Keeper saved it! Rebound!"
            st.feedback_kind = "warning"

        # 8. Ball physics, 3D arc, spin, trail & pitch bounds
        st.ball_pos[0] += st.ball_vel[0] * dt
        st.ball_pos[1] += st.ball_vel[1] * dt
        ball_speed = math.hypot(st.ball_vel[0], st.ball_vel[1])
        st.ball_spin = (st.ball_spin + ball_speed * dt * 16.0) % (2.0 * math.pi)
        if ball_speed > 0.22:
            st.ball_trail.append((st.ball_pos[0], st.ball_pos[1]))
            if len(st.ball_trail) > 10:
                st.ball_trail.pop(0)
        elif st.ball_trail:
            st.ball_trail.pop(0)

        if st.ball_height > 0.0 or st.ball_vz > 0.0:
            st.ball_height += st.ball_vz * dt
            st.ball_vz -= 1.65 * dt
            if st.ball_height <= 0.0:
                st.ball_height = 0.0
                st.ball_vz = -st.ball_vz * 0.45 if abs(st.ball_vz) > 0.12 else 0.0

        friction = max(0.0, 1.0 - 2.2 * dt)
        st.ball_vel[0] *= friction
        st.ball_vel[1] *= friction

        if st.ball_pos[1] <= PITCH_TOP:
            st.ball_pos[1] = PITCH_TOP
            st.ball_vel[1] = abs(st.ball_vel[1]) * 0.8
        elif st.ball_pos[1] >= PITCH_BOTTOM:
            st.ball_pos[1] = PITCH_BOTTOM
            st.ball_vel[1] = -abs(st.ball_vel[1]) * 0.8

        in_goal_y = GOAL_Y_MIN <= st.ball_pos[1] <= GOAL_Y_MAX
        if st.ball_pos[0] >= PITCH_RIGHT:
            if in_goal_y:
                st.score_player += 1
                st.goal_banner_timer = 1.6
                st.feedback_message = f"GOOOAL!! Score: {st.score_player} - {st.score_ai}"
                st.feedback_kind = "success"
                self._reset_kickoff()
            else:
                st.ball_pos[0] = PITCH_RIGHT
                st.ball_vel[0] = -abs(st.ball_vel[0]) * 0.8
        elif st.ball_pos[0] <= PITCH_LEFT:
            if in_goal_y:
                st.score_ai += 1
                st.feedback_message = "Red Team scored! Push forward!"
                st.feedback_kind = "error"
                self._reset_kickoff()
            else:
                st.ball_pos[0] = PITCH_LEFT
                st.ball_vel[0] = abs(st.ball_vel[0]) * 0.8

        return st

    def _move_unit_toward(
        self,
        unit: FieldPlayer,
        target: tuple[float, float],
        speed: float,
        dt: float,
    ) -> None:
        dx = target[0] - unit.pos[0]
        dy = target[1] - unit.pos[1]
        dist = math.hypot(dx, dy)
        if dist < 0.015:
            unit.vel = [0.0, 0.0]
            return
        vx = (dx / dist) * speed
        vy = (dy / dist) * speed
        unit.vel = [vx, vy]
        unit.pos[0] += vx * dt
        unit.pos[1] += vy * dt
