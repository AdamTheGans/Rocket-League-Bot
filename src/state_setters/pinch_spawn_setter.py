# src/state_setters/pinch_spawn_setter.py
"""
RLGym v2 state mutator for the pinch specialist.

Three backward-chaining stages:
  Stage 1 (micro): ball flush on wall, car 50-250uu away, pre-aligned
  Stage 2 (approach): car 600-1500uu from ball, moderate yaw noise
  Stage 3 (live-ish): car 800-2500uu, full random yaw, broader ball positions
"""
from __future__ import annotations

import math
import numpy as np

from rlgym.rocket_league import common_values

# Field constants
_SIDE_WALL_X = common_values.SIDE_WALL_X        # 4096
_BACK_WALL_Y = common_values.BACK_WALL_Y        # 5120
_BALL_RADIUS  = common_values.BALL_RADIUS        # 91.25
_BALL_REST_Z  = common_values.BALL_RESTING_HEIGHT  # 93.15
_CAR_REST_Z   = 17.01  # Dominus resting height (hitbox center)

# Ball X when "flush" on the wall = wall - ball_radius
_FLUSH_X = _SIDE_WALL_X - _BALL_RADIUS  # ~4004.75


class PinchSpawnMutator:
    """
    Spawn distribution for wall/corner pinch training.

    Parameters
    ----------
    stage : int
        Training stage (1, 2, or 3).  Controls tightness of spawns.
    """

    # ── Per-stage configs ──
    _CONFIGS = {
        1: dict(
            ball_x_range=(3900.0, _FLUSH_X),      # nearly on wall
            ball_y_range=(1000.0, 4500.0),         # offensive half
            ball_z_range=(_BALL_REST_Z, 150.0),
            ball_vel_max=100.0,
            ball_wall_vel=(100.0, 300.0),          # slight velocity toward wall
            car_dist_range=(150.0, 500.0),         # micro-skill: enough for 3-5 decisions
            car_yaw_noise=0.15,                    # tightly pre-aligned
            car_speed_range=(500.0, 1200.0),
            boost_range=(40.0, 100.0),
            corner_prob=0.05,
            timeout_seconds=2.0,
        ),
        2: dict(
            ball_x_range=(3200.0, _FLUSH_X),
            ball_y_range=(200.0, 4800.0),
            ball_z_range=(_BALL_REST_Z, 300.0),
            ball_vel_max=400.0,
            car_dist_range=(600.0, 1500.0),
            car_yaw_noise=0.5,
            car_speed_range=(0.0, 1400.0),
            boost_range=(20.0, 100.0),
            corner_prob=0.15,
            timeout_seconds=4.0,
        ),
        3: dict(
            ball_x_range=(2400.0, _FLUSH_X),
            ball_y_range=(-500.0, 4800.0),
            ball_z_range=(_BALL_REST_Z, 400.0),
            ball_vel_max=700.0,
            car_dist_range=(800.0, 2500.0),
            car_yaw_noise=math.pi,               # full random
            car_speed_range=(0.0, 1600.0),
            boost_range=(10.0, 100.0),
            corner_prob=0.30,
            timeout_seconds=6.0,
        ),
    }

    def __init__(self, stage: int = 1):
        if stage not in self._CONFIGS:
            raise ValueError(f"stage must be 1, 2, or 3, got {stage}")
        self.stage = stage
        self.cfg = self._CONFIGS[stage]

    @property
    def timeout_seconds(self) -> float:
        """Episode length for this stage (used by env factory)."""
        return self.cfg["timeout_seconds"]

    def apply(self, state, shared_info=None):
        rng = np.random

        cfg = self.cfg
        is_corner = (rng.random() < cfg["corner_prob"])

        # ── Ball position ──
        ball_x_abs = rng.uniform(*cfg["ball_x_range"])
        if is_corner:
            # Corner: push Y toward back wall, keep X high
            ball_y = rng.uniform(3500.0, min(4800.0, cfg["ball_y_range"][1]))
            ball_x_abs = max(ball_x_abs, 3000.0)
        else:
            ball_y = rng.uniform(*cfg["ball_y_range"])

        # Mirror left/right randomly
        wall_sign = 1.0 if rng.random() < 0.5 else -1.0
        ball_x = wall_sign * ball_x_abs

        ball_z = rng.uniform(*cfg["ball_z_range"])

        state.ball.position = np.array(
            [float(ball_x), float(ball_y), float(ball_z)], dtype=np.float32
        )

        # Ball velocity: small random + optional push toward wall
        v_max = cfg["ball_vel_max"]
        ball_vx = rng.uniform(-v_max, v_max)
        ball_vy = rng.uniform(-v_max, v_max)
        ball_vz = rng.uniform(-v_max * 0.3, v_max * 0.3)

        # Add slight velocity toward the wall for more natural pinch setups
        wall_vel_cfg = cfg.get("ball_wall_vel")
        if wall_vel_cfg is not None:
            toward_wall_speed = rng.uniform(*wall_vel_cfg)
            ball_vx += wall_sign * toward_wall_speed

        state.ball.linear_velocity = np.array(
            [float(ball_vx), float(ball_vy), float(ball_vz)],
            dtype=np.float32,
        )
        state.ball.angular_velocity = np.array(
            [rng.uniform(-1.0, 1.0),
             rng.uniform(-1.0, 1.0),
             rng.uniform(-1.0, 1.0)],
            dtype=np.float32,
        )

        # ── Cars ──
        for _cid, car in state.cars.items():
            # Pinch point = point on wall closest to ball
            pinch_x = wall_sign * _SIDE_WALL_X
            pinch_y = float(ball_y)

            # Angle from ball to pinch point (world frame)
            angle_to_pinch = math.atan2(
                pinch_y - float(ball_y),
                pinch_x - float(ball_x)
            )

            # Place car offset from ball, roughly opposite the wall
            dist = rng.uniform(*cfg["car_dist_range"])
            
            # We want the car to be "behind" the ball relative to the target goal
            # Target goal is y = -5120 (Orange) or y = 5120 (Blue)
            # Since we assume the agent is attacking the opponent's goal, the car 
            # needs to spawn slightly closer to its OWN goal than the ball is.
            # If car is Blue, it attacks +Y, so it should spawn at a lesser Y than ball.
            target_goal_y = -_BACK_WALL_Y if car.is_orange else _BACK_WALL_Y
            
            # Vector from ball towards center field, but tilted backwards
            center_x_dir = -wall_sign 
            back_y_dir = -1.0 if target_goal_y > 0 else 1.0 # Away from target goal
            
            base_angle = math.atan2(back_y_dir * 0.5, center_x_dir)
            offset_angle = base_angle + rng.uniform(-0.6, 0.6)

            car_x = float(ball_x) + dist * math.cos(offset_angle)
            car_y = float(ball_y) + dist * math.sin(offset_angle)

            # Clamp to field
            car_x = np.clip(car_x, -(_SIDE_WALL_X - 100), _SIDE_WALL_X - 100)
            car_y = np.clip(car_y, -(_BACK_WALL_Y - 200), _BACK_WALL_Y - 200)

            car.physics.position = np.array(
                [float(car_x), float(car_y), _CAR_REST_Z], dtype=np.float32
            )

            # Yaw: point toward ball/pinch-point + noise
            yaw_to_ball = math.atan2(
                float(ball_y) - float(car_y),
                float(ball_x) - float(car_x),
            )
            yaw = yaw_to_ball + rng.uniform(
                -cfg["car_yaw_noise"], cfg["car_yaw_noise"]
            )
            car.physics.euler_angles = np.array(
                [0.0, float(yaw), 0.0], dtype=np.float32
            )

            # Initial car speed in the facing direction
            speed = rng.uniform(*cfg["car_speed_range"])
            car.physics.linear_velocity = np.array(
                [speed * math.cos(yaw),
                 speed * math.sin(yaw),
                 0.0],
                dtype=np.float32,
            )
            car.physics.angular_velocity = np.zeros(3, dtype=np.float32)

            # Boost
            car.boost_amount = float(rng.uniform(*cfg["boost_range"]))

            # Hitbox — Dominus (project-wide standard)
            car.hitbox_type = common_values.DOMINUS
