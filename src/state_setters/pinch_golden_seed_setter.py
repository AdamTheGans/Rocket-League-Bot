# src/state_setters/pinch_golden_seed_setter.py
"""
RLGym v2 state mutator for the Golden Seed Pinch.

Initializes the specific baseline discovered for the Left Wall Kuxir Pinch,
and provides domain randomization bounds and the "Flip Timer Fix".
"""
from __future__ import annotations

import math
import numpy as np

from rlgym.rocket_league import common_values


class PinchGoldenSeedSetter:
    """
    Spawns the car and ball using the manually discovered "Golden Seed" parameters.
    
    Includes a `randomize` parameter for Domain Randomization (Curriculum) to prevent 
    policy overfitting. Adds the Critical Flip Timer fix to grant an infinite dodge timer.
    """

    def __init__(self, randomize: bool = True):
        self.randomize = randomize
        
        # --- The Baseline State (The Golden Seed) ---
        # Discovered values for the Left Wall (Targeting Orange Goal +Y)
        self.ball_pos = np.array([-3982.31, -2398.29, 178.15], dtype=np.float32)
        self.ball_vel = np.array([-380.37, 84.31, 720.48], dtype=np.float32)
        
        self.car_pos = np.array([-3686.92, -2466.33, 222.20], dtype=np.float32)
        self.car_vel = np.array([-2100.00, 500.00, 25.00], dtype=np.float32)
        self.car_euler = np.array([-0.52, 2.99, 1.86], dtype=np.float32)
        
        # --- Domain Randomization Bounds ---
        # NOTE to user: Adjust these values to widen or narrow the variance.
        self.pos_noise_uu = 15.0       # +/- 15 uu for car and ball positions
        self.vel_noise_pct = 0.03      # +/- 3% for car and ball velocities
        self.euler_noise_rad = 0.05    # +/- 5 degrees for pitch, yaw, roll

    def apply(self, state, shared_info=None):
        """
        Applies the state mutator to the state_wrapper.
        Note: The method signature requires 'state' and optionally 'shared_info'.
        """
        rng = np.random
        
        # Copy baseline arrays
        b_pos = self.ball_pos.copy()
        b_vel = self.ball_vel.copy()
        c_pos = self.car_pos.copy()
        c_vel = self.car_vel.copy()
        c_euler = self.car_euler.copy()

        # Apply Domain Randomization curriculum if requested
        if self.randomize:
            # Add uniform noise to positions
            b_pos += rng.uniform(-self.pos_noise_uu, self.pos_noise_uu, size=3)
            c_pos += rng.uniform(-self.pos_noise_uu, self.pos_noise_uu, size=3)
            
            # Add percentage-based noise to velocities
            b_vel_scales = rng.uniform(1.0 - self.vel_noise_pct, 1.0 + self.vel_noise_pct, size=3)
            c_vel_scales = rng.uniform(1.0 - self.vel_noise_pct, 1.0 + self.vel_noise_pct, size=3)
            b_vel *= b_vel_scales
            c_vel *= c_vel_scales
            
            # Add uniform noise to euler angles
            c_euler += rng.uniform(-self.euler_noise_rad, self.euler_noise_rad, size=3)

        # Set Ball state
        state.ball.position = b_pos
        state.ball.linear_velocity = b_vel
        state.ball.angular_velocity = np.zeros(3, dtype=np.float32)

        # Set Car state
        for _cid, car in state.cars.items():
            car.physics.position = c_pos
            car.physics.linear_velocity = c_vel
            car.physics.angular_velocity = np.zeros(3, dtype=np.float32)
            
            # Euler array [pitch, yaw, roll]
            car.physics.euler_angles = c_euler
            
            # The Flip Timer Fix (Critical Mechanic)
            # This mid-air state mimics falling from the ceiling to grant infinite dodge.
            car.has_jumped = False
            car.has_double_jumped = False
            car.has_flipped = False
            car.on_ground = False
            # car.has_flip = True
            # car.can_flip = True
            car.air_time_since_jump = 0.0
            
            # Boost
            car.boost_amount = 100.0
            car.hitbox_type = common_values.DOMINUS
