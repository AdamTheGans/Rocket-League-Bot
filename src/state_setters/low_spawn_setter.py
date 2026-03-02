# src/state_setters/low_spawn_setter.py
from __future__ import annotations

import math
import numpy as np

from rlgym.rocket_league import common_values


class LowGroundSpawnMutator:
    """
    RLGym v2 state mutator for the grounded-strike specialist.

    Curriculum:
      - easy_prob fraction: car placed behind ball, roughly facing orange goal (+Y)
      - remainder: wide near-ground spawns with random yaw

    Apply signature matches RLGym v2: apply(state, shared_info).
    """

    def __init__(
        self,
        easy_prob: float = 0.70,
        boost_min: float = 40.0,   # 0-100 scale (matches Car.boost_amount)
        boost_max: float = 100.0,
        ball_z: float = 93.15,     # BALL_RESTING_HEIGHT from common_values
        car_z: float = 15.65,      # Dominus resting height (hitbox center ≈ 31.30/2)
    ):
        self.easy_prob = float(easy_prob)
        self.boost_min = float(boost_min)
        self.boost_max = float(boost_max)
        self.ball_z = float(ball_z)
        self.car_z = float(car_z)

    def apply(self, state, shared_info=None):
        easy = (np.random.random() < self.easy_prob)

        # --- Ball ---
        if easy:
            ball_x = np.random.uniform(-1400, 1400)
            ball_y = np.random.uniform(-800, 2200)
        else:
            ball_x = np.random.uniform(-3000, 3000)
            ball_y = np.random.uniform(-4200, 4200)

        state.ball.position = np.array(
            [float(ball_x), float(ball_y), self.ball_z], dtype=np.float32
        )
        state.ball.linear_velocity = np.zeros(3, dtype=np.float32)
        state.ball.angular_velocity = np.zeros(3, dtype=np.float32)

        # --- Cars ---
        for _cid, car in state.cars.items():
            if easy:
                # Car behind ball, roughly facing orange goal (+Y)
                back_dist = np.random.uniform(900, 1500)
                car_x = ball_x + np.random.uniform(-250, 250)
                car_y = ball_y - back_dist + np.random.uniform(-250, 250)
                yaw = np.random.uniform(
                    np.pi / 2 - 0.35, np.pi / 2 + 0.35
                )  # ~facing +Y (toward orange goal)
            else:
                # Random placement near the ball
                car_x = np.clip(
                    ball_x + np.random.uniform(-1800, 1800), -4000, 4000
                )
                car_y = np.clip(
                    ball_y + np.random.uniform(-1800, 1800), -5000, 5000
                )
                yaw = np.random.uniform(-math.pi, math.pi)

            # Position & velocities
            car.physics.position = np.array(
                [float(car_x), float(car_y), self.car_z], dtype=np.float32
            )
            car.physics.linear_velocity = np.zeros(3, dtype=np.float32)
            car.physics.angular_velocity = np.zeros(3, dtype=np.float32)

            # Rotation — use euler_angles (pitch, yaw, roll) which is the
            # correct PhysicsObject API.  Setting it clears cached quaternion
            # and rotation_mtx so the engine picks up the new value.
            car.physics.euler_angles = np.array(
                [0.0, float(yaw), 0.0], dtype=np.float32
            )

            # Boost (Car.boost_amount is on a 0-100 scale)
            car.boost_amount = float(
                np.random.uniform(self.boost_min, self.boost_max)
            )

            # Hitbox — Dominus for better pinches and power shots
            car.hitbox_type = common_values.DOMINUS