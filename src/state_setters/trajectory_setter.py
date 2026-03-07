# src/state_setters/trajectory_setter.py
"""
Generic trajectory-based state setter for mechanic training.

Loads pre-extracted physics trajectories from .npy files, identifies the
"mechanic moment" (e.g., a pinch, a flip reset) via peak ball speed, and
sets game state to a frame from the trajectory based on a curriculum
difficulty slider.

Designed to be reused for ANY mechanic — just point `data_dir` at a
different directory of .npy files and set `mechanic_name` accordingly.

.npy format (per frame, 28 floats):
    [0:3]   Ball position (x, y, z)
    [3:6]   Ball linear velocity (vx, vy, vz)
    [6:9]   Ball angular velocity (wx, wy, wz)
    [9:12]  Car position (x, y, z)
    [12:15] Car linear velocity (vx, vy, vz)
    [15:18] Car angular velocity (wx, wy, wz)
    [18:22] Car quaternion (w, x, y, z)
    [22:25] Car Euler angles (pitch, yaw, roll)
    [25]    Boost amount (0-100)
    [26]    On-ground flag (0 or 1)
    [27]    Elapsed time (seconds)
"""
from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Optional

import numpy as np

from rlgym.rocket_league import common_values


# ─────────────────────────────────────────────────────────────────── #
#  Column index constants — edit here if your .npy layout ever changes
# ─────────────────────────────────────────────────────────────────── #
COL_BALL_POS       = slice(0, 3)
COL_BALL_LIN_VEL   = slice(3, 6)
COL_BALL_ANG_VEL   = slice(6, 9)
COL_CAR_POS        = slice(9, 12)
COL_CAR_LIN_VEL    = slice(12, 15)
COL_CAR_ANG_VEL    = slice(15, 18)
COL_CAR_QUAT       = slice(18, 22)
COL_CAR_EULER      = slice(22, 25)
COL_CAR_BOOST      = 25
COL_CAR_ON_GROUND  = 26
COL_ELAPSED_TIME   = 27

FRAME_WIDTH = 28


class MechanicTrajectorySetter:
    """
    RLGym v2 state mutator that replays extracted mechanic trajectories.

    Parameters
    ----------
    data_dir : str
        Path to the directory containing .npy trajectory files.
    mechanic_name : str
        Human-readable ID for the mechanic (e.g., "kuxir", "ceiling_shot").
        Written to ``shared_info["setter_type"]`` so downstream reward
        functions and callbacks can distinguish episode types.
    fps : int
        Assumed frame rate of the extracted replays (default 30).
    pre_mechanic_seconds : float
        How many seconds of trajectory to keep before the mechanic moment
        (default 1.5 → ~45 frames at 30 FPS).
    noise_scales : dict, optional
        Override the maximum per-axis noise for each quantity.
        Defaults are tuned for Kuxir pinch but work as a starting point
        for other mechanics too.
    """

    def __init__(
        self,
        data_dir: str,
        mechanic_name: str = "kuxir",
        fps: int = 30,
        pre_mechanic_seconds: float = 1.5,
        noise_scales: Optional[dict] = None,
    ):
        self.mechanic_name = mechanic_name
        self.fps = fps
        self.pre_mechanic_seconds = pre_mechanic_seconds

        # ── Curriculum sliders (externally adjustable) ──────────────
        self.difficulty = 0.0    # 0.0 = last frame before mechanic, 1.0 = earliest frame
        self.noise_amount = 0.0  # 0.0 = no noise, 1.0 = full noise scales

        # ── Default noise scales (max noise when noise_amount == 1.0) ──
        defaults = {
            "car_lin_vel": np.array([300.0, 300.0, 200.0], dtype=np.float32),
            "car_ang_vel": np.array([2.0, 2.0, 2.0], dtype=np.float32),
            "car_euler":   np.array([0.3, 0.3, 0.3], dtype=np.float32),
        }
        if noise_scales:
            defaults.update(noise_scales)
        self.noise_scales = defaults

        # ── Load and process trajectories ───────────────────────────
        self.trajectories: list[np.ndarray] = []
        self._load_trajectories(data_dir)

    # ─────────────────── Data Loading & Slicing ─────────────────── #

    def _load_trajectories(self, data_dir: str) -> None:
        """Load all .npy files and slice to the usable pre-mechanic window."""
        data_path = Path(data_dir)
        if not data_path.is_dir():
            raise FileNotFoundError(f"Trajectory directory not found: {data_dir}")

        npy_files = sorted(data_path.glob("*.npy"))
        if not npy_files:
            raise FileNotFoundError(f"No .npy files found in: {data_dir}")

        keep_frames = int(self.fps * self.pre_mechanic_seconds)

        for fpath in npy_files:
            raw = np.load(str(fpath))  # shape: (N, 28)
            if raw.ndim != 2 or raw.shape[1] != FRAME_WIDTH:
                print(f"  WARNING: Skipping {fpath.name} — unexpected shape {raw.shape}")
                continue

            # Find the mechanic moment: peak ball speed
            ball_speeds = np.linalg.norm(raw[:, COL_BALL_LIN_VEL], axis=1)
            mechanic_frame = int(np.argmax(ball_speeds))

            # Slice: keep `keep_frames` frames immediately BEFORE the mechanic
            start = max(0, mechanic_frame - keep_frames)
            end = mechanic_frame  # exclusive — don't include the mechanic frame itself

            if end - start < 5:
                print(f"  WARNING: Skipping {fpath.name} — only {end - start} usable frames")
                continue

            self.trajectories.append(raw[start:end].copy())

        if not self.trajectories:
            raise RuntimeError(
                f"No valid trajectories loaded from {data_dir}. "
                f"Check that .npy files contain the expected (N, {FRAME_WIDTH}) format."
            )

        # Print summary
        shapes = [t.shape[0] for t in self.trajectories]
        print(f"  [{self.mechanic_name}] Loaded {len(self.trajectories)} trajectories "
              f"(frames per traj: min={min(shapes)}, max={max(shapes)}, "
              f"target={keep_frames})")

    # ─────────────────── State Application ──────────────────────── #

    def apply(self, state, shared_info: Optional[dict] = None) -> None:
        """
        RLGym v2 state mutator interface.

        Randomly selects a trajectory, picks a frame based on ``self.difficulty``,
        applies physics, adds noise scaled by ``self.noise_amount``, and
        optionally mirrors across the X axis.
        """
        rng = np.random

        # ── 1. Select trajectory and frame ──────────────────────────
        traj_idx = rng.randint(0, len(self.trajectories))
        traj = self.trajectories[traj_idx]
        num_frames = traj.shape[0]

        # difficulty=0.0 → near the end but with a buffer (a few frames before contact)
        # difficulty=1.0 → first frame (farthest from contact)
        # The buffer ensures the bot always has time to position itself,
        # even at the easiest difficulty.
        min_buffer = min(5, num_frames - 1)  # ~0.17s at 30 fps
        easiest_frame = num_frames - 1 - min_buffer
        frame_idx = int((1.0 - self.difficulty) * easiest_frame)
        frame_idx = np.clip(frame_idx, 0, num_frames - 1)
        frame = traj[frame_idx]

        # ── 2. Extract physics from the frame ───────────────────────
        b_pos = frame[COL_BALL_POS].copy()
        b_vel = frame[COL_BALL_LIN_VEL].copy()
        b_ang = frame[COL_BALL_ANG_VEL].copy()

        c_pos   = frame[COL_CAR_POS].copy()
        c_vel   = frame[COL_CAR_LIN_VEL].copy()
        c_ang   = frame[COL_CAR_ANG_VEL].copy()
        c_euler = frame[COL_CAR_EULER].copy()
        c_boost = float(frame[COL_CAR_BOOST])
        c_ground = float(frame[COL_CAR_ON_GROUND])

        # ── 3. Apply noise to CAR only (no positional noise → no wall clipping)
        if self.noise_amount > 0.0:
            amt = self.noise_amount
            c_vel += rng.uniform(
                -self.noise_scales["car_lin_vel"] * amt,
                 self.noise_scales["car_lin_vel"] * amt,
            )
            c_ang += rng.uniform(
                -self.noise_scales["car_ang_vel"] * amt,
                 self.noise_scales["car_ang_vel"] * amt,
            )
            c_euler += rng.uniform(
                -self.noise_scales["car_euler"] * amt,
                 self.noise_scales["car_euler"] * amt,
            )

        # ── 4. Mirror across X axis with 50% probability ───────────
        if rng.random() > 0.5:
            b_pos[0] *= -1.0
            b_vel[0] *= -1.0
            b_ang[1] *= -1.0  # angular velocity Y component flips
            b_ang[2] *= -1.0  # angular velocity Z component flips

            c_pos[0] *= -1.0
            c_vel[0] *= -1.0
            c_ang[1] *= -1.0
            c_ang[2] *= -1.0

            # Yaw mirror: new_yaw = pi - old_yaw, normalized to [-pi, pi]
            c_yaw = c_euler[1]
            new_yaw = math.pi - c_yaw
            if new_yaw > math.pi:
                new_yaw -= 2 * math.pi
            if new_yaw < -math.pi:
                new_yaw += 2 * math.pi
            c_euler[1] = new_yaw
            c_euler[2] *= -1.0  # Roll inverts

        # ── 5. Apply physics to the game state ──────────────────────
        state.ball.position = b_pos.astype(np.float32)
        state.ball.linear_velocity = b_vel.astype(np.float32)
        state.ball.angular_velocity = b_ang.astype(np.float32)

        cars_list = list(state.cars.items())
        agent_set = False
        for cid, car in cars_list:
            if not agent_set:
                # First car = the learning agent: set from trajectory
                car.physics.position = c_pos.astype(np.float32)
                car.physics.linear_velocity = c_vel.astype(np.float32)
                car.physics.angular_velocity = c_ang.astype(np.float32)
                car.physics.euler_angles = c_euler.astype(np.float32)
                car.boost_amount = np.clip(c_boost, 0.0, 100.0)

                # Set ground/air state from replay data
                if c_ground > 0.5:
                    car.on_ground = True
                    car.has_jumped = False
                    car.has_double_jumped = False
                    car.has_flipped = False
                else:
                    # Mid-air: grant infinite flip timer
                    car.on_ground = False
                    car.has_jumped = False
                    car.has_double_jumped = False
                    car.has_flipped = False
                    car.air_time_since_jump = 0.0

                car.hitbox_type = common_values.DOMINUS
                agent_set = True
            else:
                # Opponent car: spawn at center field, stationary, irrelevant
                car.physics.position = np.array([0.0, 0.0, 17.01], dtype=np.float32)
                car.physics.linear_velocity = np.zeros(3, dtype=np.float32)
                car.physics.angular_velocity = np.zeros(3, dtype=np.float32)
                car.physics.euler_angles = np.array([0.0, -math.pi / 2, 0.0], dtype=np.float32)
                car.boost_amount = 0.0
                car.on_ground = True
                car.has_jumped = False
                car.has_double_jumped = False
                car.has_flipped = False
                car.hitbox_type = common_values.DOMINUS

        # ── 6. Tag the episode type in shared_info ──────────────────
        if shared_info is not None:
            shared_info["setter_type"] = self.mechanic_name
