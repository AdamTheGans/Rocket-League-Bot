# src/state_setters/trajectory_setter.py
"""
Generic trajectory-based state setter for mechanic training.

Loads pre-extracted physics trajectories from .npy files, identifies the
"mechanic moment" (e.g., a pinch, a flip reset) via peak ball speed, and
sets game state to a frame from the trajectory based on a curriculum
difficulty slider.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np
from rlgym.rocket_league import common_values

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
    def __init__(
        self,
        data_dir: str,
        mechanic_name: str = "kuxir",
        fps: int = 30,
        pre_mechanic_seconds: float = 1.5,
        noise_scales: Optional[dict] = None,
        shared_curriculum = None,  # Pass the shared multiprocessing curriculum here
    ):
        self.mechanic_name = mechanic_name
        self.fps = fps
        self.pre_mechanic_seconds = pre_mechanic_seconds
        self.shared_curriculum = shared_curriculum

        defaults = {
            "car_lin_vel": np.array([300.0, 300.0, 200.0], dtype=np.float32),
            "car_ang_vel": np.array([2.0, 2.0, 2.0], dtype=np.float32),
            "car_euler":   np.array([0.3, 0.3, 0.3], dtype=np.float32),
        }
        if noise_scales:
            defaults.update(noise_scales)
        self.noise_scales = defaults

        self.trajectories: list[np.ndarray] = []
        self._load_trajectories(data_dir)

    def _load_trajectories(self, data_dir: str) -> None:
        """Load .npy files and standardize to Blue team."""
        data_path = Path(data_dir)
        if not data_path.is_dir():
            raise FileNotFoundError(f"Trajectory directory not found: {data_dir}")

        npy_files = sorted(data_path.glob("*.npy"))
        keep_frames = int(self.fps * self.pre_mechanic_seconds)

        for fpath in npy_files:
            raw = np.load(str(fpath))
            if raw.ndim != 2 or raw.shape[1] != FRAME_WIDTH:
                continue

            ball_speeds = np.linalg.norm(raw[:, COL_BALL_LIN_VEL], axis=1)
            mechanic_frame = int(np.argmax(ball_speeds))
            start = max(0, mechanic_frame - keep_frames)
            end = mechanic_frame 

            if end - start < 5:
                continue

            sliced_traj = raw[start:end].copy()

            # STANDARDIZE TO BLUE TEAM
            ball_vy_at_mechanic = raw[mechanic_frame, 4]
            if ball_vy_at_mechanic < 0:
                sliced_traj[:, [0, 1, 9, 10]] *= -1.0
                sliced_traj[:, [3, 4, 12, 13]] *= -1.0
                sliced_traj[:, [6, 7, 15, 16]] *= -1.0
                sliced_traj[:, 23] += math.pi
                sliced_traj[:, 23] = (sliced_traj[:, 23] + math.pi) % (2 * math.pi) - math.pi

            self.trajectories.append(sliced_traj)

    def apply(self, state, shared_info: Optional[dict] = None) -> None:
        rng = np.random

        # Read live curriculum values from shared memory
        difficulty = self.shared_curriculum.difficulty.value if self.shared_curriculum else 0.5
        noise_amount = self.shared_curriculum.noise_amount.value if self.shared_curriculum else 0.0

        traj_idx = rng.randint(0, len(self.trajectories))
        traj = self.trajectories[traj_idx]
        num_frames = traj.shape[0]

        min_buffer = min(5, num_frames - 1)
        easiest_frame = num_frames - 1 - min_buffer
        frame_idx = int((1.0 - difficulty) * easiest_frame)
        frame_idx = np.clip(frame_idx, 0, num_frames - 1)
        frame = traj[frame_idx]

        b_pos = frame[COL_BALL_POS].copy()
        b_vel = frame[COL_BALL_LIN_VEL].copy()
        b_ang = frame[COL_BALL_ANG_VEL].copy()

        c_pos   = frame[COL_CAR_POS].copy()
        c_vel   = frame[COL_CAR_LIN_VEL].copy()
        c_ang   = frame[COL_CAR_ANG_VEL].copy()
        c_euler = frame[COL_CAR_EULER].copy()
        c_boost = float(frame[COL_CAR_BOOST])
        c_ground = float(frame[COL_CAR_ON_GROUND])

        if noise_amount > 0.0:
            c_vel += rng.uniform(-self.noise_scales["car_lin_vel"] * noise_amount, self.noise_scales["car_lin_vel"] * noise_amount)
            c_ang += rng.uniform(-self.noise_scales["car_ang_vel"] * noise_amount, self.noise_scales["car_ang_vel"] * noise_amount)
            c_euler += rng.uniform(-self.noise_scales["car_euler"] * noise_amount, self.noise_scales["car_euler"] * noise_amount)

        # Mirror across X axis with 50% probability (Left vs Right wall)
        if rng.random() > 0.5:
            b_pos[0] *= -1.0
            b_vel[0] *= -1.0
            b_ang[1] *= -1.0
            b_ang[2] *= -1.0
            c_pos[0] *= -1.0
            c_vel[0] *= -1.0
            c_ang[1] *= -1.0
            c_ang[2] *= -1.0
            
            c_yaw = c_euler[1]
            new_yaw = math.pi - c_yaw
            if new_yaw > math.pi: new_yaw -= 2 * math.pi
            if new_yaw < -math.pi: new_yaw += 2 * math.pi
            c_euler[1] = new_yaw
            c_euler[2] *= -1.0

        state.ball.position = b_pos.astype(np.float32)
        state.ball.linear_velocity = b_vel.astype(np.float32)
        state.ball.angular_velocity = b_ang.astype(np.float32)

        cars_list = list(state.cars.items())
        agent_set = False
        
        for cid, car in cars_list:
            if not agent_set:
                # ── The Learning Bot ──
                car.physics.position = c_pos.astype(np.float32)
                car.physics.linear_velocity = c_vel.astype(np.float32)
                car.physics.angular_velocity = c_ang.astype(np.float32)
                car.physics.euler_angles = c_euler.astype(np.float32)
                car.boost_amount = np.clip(c_boost, 0.0, 100.0)

                car.on_ground = bool(c_ground > 0.5)
                car.has_jumped = False
                car.has_double_jumped = False
                car.has_flipped = False
                car.air_time_since_jump = 0.0 if not car.on_ground else 0.0
                
                car.hitbox_type = common_values.OCTANE # Fixed to Octane
                agent_set = True
            else:
                # ── The Ghost Opponent ──
                # Teleported out of bounds, zero velocity, demoed flag set.
                car.physics.position = np.array([0.0, 0.0, -1000.0], dtype=np.float32)
                car.physics.linear_velocity = np.zeros(3, dtype=np.float32)
                car.physics.angular_velocity = np.zeros(3, dtype=np.float32)
                car.physics.euler_angles = np.zeros(3, dtype=np.float32)
                car.boost_amount = 0.0
                car.on_ground = True
                car.is_demoed = True # RocketSim ignores physics for demoed cars
                car.hitbox_type = common_values.OCTANE

        if shared_info is not None:
            shared_info["setter_type"] = self.mechanic_name