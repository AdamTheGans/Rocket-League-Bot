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

    def __init__(self, randomize: bool = True, stage: float = 1.0, difficulty_level: int = 1):
        self.randomize = randomize
        self.stage = stage
        
        # --- The Baseline State (The Golden Seed) ---
        # Discovered values for the Left Wall (Targeting Orange Goal +Y)
        self.ball_pos = np.array([-3982.31, -2398.29, 178.15], dtype=np.float32)
        self.ball_vel = np.array([-380.37, 84.31, 720.48], dtype=np.float32)
        
        self.car_pos = np.array([-3686.92, -2466.33, 222.20], dtype=np.float32)
        self.car_vel = np.array([-2100.00, 500.00, 25.00], dtype=np.float32)
        self.car_euler = np.array([0.52, 2.99, 1.86], dtype=np.float32) # Inverted Pitch to point nose UP instead of down
        
        # Pre-calculate ball trajectory to allow dynamic rewinding along the wall geometry
        import RocketSim as rsim
        from rlgym.rocket_league.sim import RocketSimEngine
        
        # Instantiate RocketSimEngine which safely initializes the rsim collision meshes globally
        temp_engine = RocketSimEngine()
        temp_arena = temp_engine._arena
        b_state = temp_arena.ball.get_state()
        b_state.pos = rsim.Vec(-3913.5, -2500.0, 93.15)
        b_state.vel = rsim.Vec(-1500.0, 150.0, 0.0)
        temp_arena.ball.set_state(b_state)
        
        self.ball_traj_pos = []
        self.ball_traj_vel = []
        
        for _ in range(300):
            temp_arena.step(1)
            bs = temp_arena.ball.get_state()
            self.ball_traj_pos.append(np.array([bs.pos.x, bs.pos.y, bs.pos.z], dtype=np.float32))
            self.ball_traj_vel.append(np.array([bs.vel.x, bs.vel.y, bs.vel.z], dtype=np.float32))

        # Find the tick where Z is closest to self.ball_pos[2]
        z_diffs = [abs(p[2] - self.ball_pos[2]) for p in self.ball_traj_pos]
        self.golden_tick = int(np.argmin(z_diffs))
        
        # Calculate the Y offset between the simulated trajectory and our baseline ball_pos
        self.golden_traj_y = self.ball_traj_pos[self.golden_tick][1]
        
        # --- Domain Randomization Bounds ---
        if self.stage == 1:
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
                self.y_slide_uu = 800.0       # Max distance to slide up/down the wall
            
            elif difficulty_level == 3:
                # Provide independent XYZ noise vectors to prevent physics clipping and grant more variety
                self.pos_noise_car = np.array([35.0, 35.0, 35.0], dtype=np.float32)
                # Avoid randomizing ball X position (it's already resting on the wall)
                self.pos_noise_ball = np.array([0.0, 35.0, 35.0], dtype=np.float32)
            
                self.vel_noise_car = np.array([250.0, 250.0, 150.0], dtype=np.float32)
                # Avoid randomizing ball X velocity (keep it glued to the wall)
                self.vel_noise_ball = np.array([0.0, 75.0, 225.0], dtype=np.float32)

                self.euler_noise_rad = 0.35    # +/- 35 degrees for pitch, yaw, roll
                self.y_slide_uu = 1200.0       # Max distance to slide up/down the wall

        elif self.stage == 1.5:
            # Stage 1.5: The Bridge - Minimal Noise
            self.pos_noise_car = np.array([10.0, 10.0, 10.0], dtype=np.float32)
            self.pos_noise_ball = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        
            self.vel_noise_car = np.array([20.0, 20.0, 20.0], dtype=np.float32)
            self.vel_noise_ball = np.array([0.0, 0.0, 0.0], dtype=np.float32)

            self.euler_noise_rad = 0.05   # +/- 0.05 radians
            self.y_slide_uu = 200.0       # Max distance to slide up/down the wall

        elif self.stage == 2:
            # Stage 2: The Approach
            self.pos_noise_car = np.array([100.0, 100.0, 100.0], dtype=np.float32)
            # Avoid randomizing ball XYZ deeply to prevent physics clipping since it stays on the wall
            self.pos_noise_ball = np.array([0.0, 35.0, 35.0], dtype=np.float32)
        
            self.vel_noise_car = np.array([200.0, 200.0, 200.0], dtype=np.float32)
            # Avoid randomizing ball X velocity (keep it glued to the wall)
            self.vel_noise_ball = np.array([0.0, 75.0, 225.0], dtype=np.float32)

            self.euler_noise_rad = 0.5    # +/- 0.5 radians (~28 degrees) for pitch, yaw, roll
            self.y_slide_uu = 1200.0      # Max distance to slide up/down the wall

        elif self.stage == 3:
            # Stage 3: The Ground-to-Wall Approach
            # Both car and ball are custom randomized in apply(), so zero all noises here
            self.pos_noise_car = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            self.pos_noise_ball = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            self.vel_noise_car = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            self.vel_noise_ball = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            self.euler_noise_rad = 0.0
            self.y_slide_uu = 800.0          # Max distance to slide up/down the wall

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
            
            # Dynamic Rewind for Stage 1.5
            if self.stage == 1.5:
                # Randomize flight time between 0.15 and 0.3 seconds
                time_to_impact = rng.uniform(0.15, 0.3)
                additional_rewind = time_to_impact - 0.15
                c_pos -= c_vel * additional_rewind

            # Dynamic Rewind for Stage 2
            elif self.stage == 2:
                # Randomize flight time between 0.3 seconds and 0.6 seconds
                time_to_impact = rng.uniform(0.3, 0.6)
                # The baseline car pos is already 0.15s away from the impact point.
                additional_rewind = time_to_impact - 0.15
                
                # Spawn the car closer and higher to counteract gravity and acceleration delay
                car_rewind = additional_rewind * 0.8
                c_pos -= c_vel * car_rewind
                c_pos[2] += 325.0 * (car_rewind ** 2)
                
                # Rewind ball using pre-calculated trajectory
                # We need to go back by additional_rewind seconds.
                # However, the user noted the car consistently misses the ball (arrives too late).
                # To compensate, we rewind the ball *further* back in its timeline (e.g. 1.5x)
                # so it takes longer to climb the wall, giving the car a larger buffer.
                ball_rewind_factor = 1.35
                rewind_ticks = int(additional_rewind * ball_rewind_factor * 120)
                target_tick = max(0, self.golden_tick - rewind_ticks)
                
                b_pos = self.ball_traj_pos[target_tick].copy()
                b_vel = self.ball_traj_vel[target_tick].copy()
                
                # Align the trajectory's Y-coordinate to our baseline Y-coordinate
                y_shift = self.ball_pos[1] - self.golden_traj_y
                b_pos[1] += y_shift
                
            elif self.stage == 3:
                # Stage 3 Custom Floor Spawn Logic
                # Left wall is negative X (-4096). Middle is 0.
                c_pos[0] = rng.uniform(-1500.0, -500.0)
                c_pos[1] = self.ball_pos[1]
                c_pos[2] = 17.01
                
                # Ball is almost just in front of the player
                b_pos[0] = c_pos[0] - rng.uniform(400.0, 1000.0)
                b_pos[1] = self.ball_pos[1] + rng.uniform(-50.0, 50.0)
                b_pos[2] = 93.15  # ball radius, resting on the floor
                
                # Ball Velocity: heading to the wall at 1500-2000 uu/s
                b_speed = rng.uniform(1500.0, 2000.0)
                b_vel[0] = -b_speed
                b_vel[1] = rng.uniform(-100.0, 100.0)
                b_vel[2] = 0.0
                
                # Yaw: Point roughly towards the ball
                dx = b_pos[0] - c_pos[0]
                dy = b_pos[1] - c_pos[1]
                target_yaw = math.atan2(dy, dx)
                c_euler[1] = target_yaw + rng.uniform(-0.1, 0.1)
                
                # Pitch and Roll
                c_euler[0] = 0.0
                c_euler[2] = 0.0
                
                # Car Velocity driving behind the ball at 1000-1800 uu/s
                speed = rng.uniform(1000.0, 1500.0)
                c_vel[0] = speed * math.cos(c_euler[1])
                c_vel[1] = speed * math.sin(c_euler[1])
                c_vel[2] = 0.0
            
            # Independent Position Noise
            b_pos += rng.uniform(-self.pos_noise_ball, self.pos_noise_ball, size=3)
            c_pos += rng.uniform(-self.pos_noise_car, self.pos_noise_car, size=3)
            
            # Independent Velocity Noise
            b_vel += rng.uniform(-self.vel_noise_ball, self.vel_noise_ball, size=3)
            c_vel += rng.uniform(-self.vel_noise_car, self.vel_noise_car, size=3)
            
            # Add uniform noise to euler angles
            c_euler += rng.uniform(-self.euler_noise_rad, self.euler_noise_rad, size=3)
            
            # For Stage 2, 30% of the time spawn the car flat (wheels facing the ground)
            # This forces the bot to learn how to intentionally roll its car to align with the wall
            if self.stage == 2 and rng.random() < 0.30:
                # Override the roll component (index 2) to ~0.0 radians
                c_euler[2] = rng.uniform(-0.1, 0.1)

            # Slide the entire setup along the Y-axis (wall parallel)
            y_offset = rng.uniform(-self.y_slide_uu, self.y_slide_uu)
            b_pos[1] += y_offset
            c_pos[1] += y_offset
            
            # Prevent going out of bounds (Y limits roughly +/- 5120)
            b_pos[1] = np.clip(b_pos[1], -4900.0, 4900.0)
            c_pos[1] = np.clip(c_pos[1], -4900.0, 4900.0)

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
            if self.stage == 3:
                # Stage 3 starts on the floor normally
                car.on_ground = True
                car.has_jumped = False
                car.has_double_jumped = False
                car.has_flipped = False
            else:
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
