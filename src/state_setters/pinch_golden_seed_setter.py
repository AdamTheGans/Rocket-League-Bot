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

    def __init__(self, randomize: bool = True, difficulty_level: int = 1):
        self.randomize = randomize
        
        # --- The Baseline State (The Golden Seed) ---
        # Discovered values for the Left Wall (Targeting Orange Goal +Y)
        self.ball_pos = np.array([-3982.31, -2398.29, 178.15], dtype=np.float32)
        self.ball_vel = np.array([-380.37, 84.31, 720.48], dtype=np.float32)
        
        self.car_pos = np.array([-3686.92, -2466.33, 222.20], dtype=np.float32)
        self.car_vel = np.array([-2100.00, 500.00, 25.00], dtype=np.float32)
        self.car_euler = np.array([0.52, 2.99, 1.86], dtype=np.float32) # Inverted Pitch to point nose UP instead of down
        
        # --- Domain Randomization Bounds ---
        if difficulty_level == 1:
            # Provide independent XYZ noise vectors to prevent physics clipping and grant more variety
            self.pos_noise_car = np.array([15.0, 15.0, 15.0], dtype=np.float32)
            # Avoid randomizing ball X position (it's already resting on the wall)
            self.pos_noise_ball = np.array([0.0, 15.0, 15.0], dtype=np.float32)
        
            self.vel_noise_car = np.array([100.0, 100.0, 50.0], dtype=np.float32)
            # Avoid randomizing ball X velocity (keep it glued to the wall)
            self.vel_noise_ball = np.array([0.0, 25.0, 75.0], dtype=np.float32)

            self.euler_noise_rad = 0.1    # +/- 10 degrees for pitch, yaw, roll
            self.y_slide_uu = 500.0       # Max distance to slide up/down the wall

        elif difficulty_level == 2:
            # Provide independent XYZ noise vectors to prevent physics clipping and grant more variety
            self.pos_noise_car = np.array([25.0, 25.0, 25.0], dtype=np.float32)
            # Avoid randomizing ball X position (it's already resting on the wall)
            self.pos_noise_ball = np.array([0.0, 25.0, 25.0], dtype=np.float32)
        
            self.vel_noise_car = np.array([200.0, 200.0, 100.0], dtype=np.float32)
            # Avoid randomizing ball X velocity (keep it glued to the wall)
            self.vel_noise_ball = np.array([0.0, 50.0, 150.0], dtype=np.float32)

            self.euler_noise_rad = 0.2    # +/- 20 degrees for pitch, yaw, roll
            self.y_slide_uu = 1000.0       # Max distance to slide up/down the wall
        
        elif difficulty_level == 3:
            # Provide independent XYZ noise vectors to prevent physics clipping and grant more variety
            self.pos_noise_car = np.array([35.0, 35.0, 35.0], dtype=np.float32)
            # Avoid randomizing ball X position (it's already resting on the wall)
            self.pos_noise_ball = np.array([0.0, 35.0, 35.0], dtype=np.float32)
        
            self.vel_noise_car = np.array([250.0, 250.0, 150.0], dtype=np.float32)
            # Avoid randomizing ball X velocity (keep it glued to the wall)
            self.vel_noise_ball = np.array([0.0, 75.0, 225.0], dtype=np.float32)

            self.euler_noise_rad = 0.35    # +/- 35 degrees for pitch, yaw, roll
            self.y_slide_uu = 1500.0       # Max distance to slide up/down the wall

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
            # Independent Position Noise
            b_pos += rng.uniform(-self.pos_noise_ball, self.pos_noise_ball, size=3)
            c_pos += rng.uniform(-self.pos_noise_car, self.pos_noise_car, size=3)
            
            # Independent Velocity Noise
            b_vel += rng.uniform(-self.vel_noise_ball, self.vel_noise_ball, size=3)
            c_vel += rng.uniform(-self.vel_noise_car, self.vel_noise_car, size=3)
            
            # Add uniform noise to euler angles
            c_euler += rng.uniform(-self.euler_noise_rad, self.euler_noise_rad, size=3)

            # Slide the entire setup along the Y-axis (wall parallel)
            y_offset = rng.uniform(-self.y_slide_uu, self.y_slide_uu)
            b_pos[1] += y_offset
            c_pos[1] += y_offset

            # 50% chance to mirror to the opposite wall (+X -> -X)
            if rng.random() > 0.5:
                # Invert X for positions and velocities
                b_pos[0] *= -1.0
                b_vel[0] *= -1.0
                c_pos[0] *= -1.0
                c_vel[0] *= -1.0
                
                # Invert Euler angles for X-mirroring (Pitch, Yaw, Roll)
                # Pitch stays the same relative to the ground.
                # Yaw inverts (Z-axis rotation): if angle is theta, mirrored is pi - theta, but in RLGym's [-pi, pi] range
                # wait, actually if we negate X, a vector (x, y) -> (-x, y). 
                # The yaw rotation matrix handles this, so we must manually calculate:
                # new_yaw = math.pi - old_yaw (normalized)
                c_yaw = c_euler[1]
                nyaw = math.pi - c_yaw
                if nyaw > math.pi: nyaw -= 2 * math.pi
                if nyaw < -math.pi: nyaw += 2 * math.pi
                c_euler[1] = nyaw
                
                # Roll inverts to maintain the ceiling/wall orientation
                c_euler[2] *= -1.0

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
