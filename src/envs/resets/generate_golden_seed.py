# src/envs/resets/generate_golden_seed.py
import math
import time
import numpy as np
import random

import RocketSim as rsim

# Physics Constants
SIDE_WALL_X = -4096.0
BALL_RADIUS = 91.25

def main():
    # RocketSim requires collision meshes. rlgym-v2 provides them.
    # The safest way is to just let RocketSimEngine initialize it implicitly or pull its path.
    import os
    # The safest way to init RocketSim meshes in rlgym-v2 is to just instantiate RocketSimEngine once.
    # It automatically handles finding the meshes and calling rsim.init().
    from rlgym.rocket_league.sim import RocketSimEngine
    _dummy_engine = RocketSimEngine()
    
    # Now rsim is initialized properly. We can just create our Arena.
    arena = rsim.Arena(rsim.GameMode.SOCCAR)

    # Spawn ball near the left wall (-X), closer to own net (-Y)
    ball_state = arena.ball.get_state()
    ball_state.pos = rsim.Vec(-3000.0, -2500.0, 93.15)
    # Moving fast towards the negative X wall, along the ground
    ball_state.vel = rsim.Vec(-1500.0, 150.0, 0.0)
    impact_vel = rsim.Vec(0.0, 700.0, 800.0)
    V = 2300.0 # Car speed at impact vs ball (slightly faster to catch it)
    T = 0.15 # 0.15s flight time
    arena.ball.set_state(ball_state)
    
    # Add a dominus car to the arena
    car = arena.add_car(rsim.Team.BLUE, rsim.CarConfig(rsim.CarConfig.DOMINUS))

    print("Rolling simulation until ball is flush on the wall...")

    # Wait for the ball to roll up the wall curve and be on the flat wall part
    ticks = 0
    buffer_frames = int(0.1 * arena.tick_rate)  # 0.1s rollback buffer (~12 ticks)
    ball_history = []
    
    while True:
        arena.step(1)
        ticks += 1
        
        ball_state = arena.ball.get_state()
        ball_x = ball_state.pos.x
        ball_z = ball_state.pos.z
        
        # Save a copy of the state into the buffer
        # (Need to manually copy values since rsim.BallState is a reference to the C++ object)
        state_copy = rsim.BallState()
        state_copy.pos = rsim.Vec(ball_state.pos.x, ball_state.pos.y, ball_state.pos.z)
        state_copy.vel = rsim.Vec(ball_state.vel.x, ball_state.vel.y, ball_state.vel.z)
        state_copy.ang_vel = rsim.Vec(ball_state.ang_vel.x, ball_state.ang_vel.y, ball_state.ang_vel.z)
        ball_history.append(state_copy)
        
        if len(ball_history) > buffer_frames + 1:
            ball_history.pop(0)

        if ticks % 120 == 0:
            print(f"Tick {ticks}: Ball pos: [{ball_x:.1f}, {ball_state.pos.y:.1f}, {ball_z:.1f}] | Vel: [{ball_state.vel.x:.1f}, {ball_state.vel.y:.1f}, {ball_state.vel.z:.1f}]")

        # Wait until the ball rolls up the wall to the desired height
        if ball_z > 240.0 and ball_x <= SIDE_WALL_X + BALL_RADIUS + 5.0:
            print(f"Found flush wall contact at tick {ticks}!")
            break
            
        if ticks > 1200:
            print("Ball never reached the wall. Bailing out.")
            break

    # We found the perfect moment.
    # We rollback the ball state by T=0.15s (the buffer limit) to ensure it's still rolling up the wall
    # when the car spawns, so we don't cheat by placing it statically flat on the wall.
    
    impact_ball_state = ball_history[-1]
    
    # We need the ball state from T seconds ago for the environment spawn
    spawn_ball_state = ball_history[0] if len(ball_history) > buffer_frames else ball_history[-1]
    
    arena.ball.set_state(spawn_ball_state)
    
    # We base the car's intercept math on the actual target IMPACT location 0.15s from now
    impact_pos = impact_ball_state.pos
    impact_vel = impact_ball_state.vel
    spawn_pos = spawn_ball_state.pos

    # Kuxir Pinch Pre-dodge Setup
    V = 2100.0
    T = 0.15
    
    yaw = math.pi - math.radians(20)
    pitch = math.radians(10)
    roll = -math.radians(95)
    
    car_vel_x = impact_vel.x + V * math.cos(yaw)
    car_vel_y = impact_vel.y + V * math.sin(yaw)
    car_vel_z = impact_vel.z
    
    # Cap speed physically to let RocketSim simulate correctly if vector math overshoots limit
    speed = math.sqrt(car_vel_x**2 + car_vel_y**2 + car_vel_z**2)
    if speed > 2300.0:
        scale = 2300.0 / speed
        car_vel_x *= scale
    from rlgym.rocket_league.math import euler_to_rotation
    import itertools
    
    print("\n--- Optimizing Golden Seed Offsets ---")
    best_speed = 0
    best_speed_goal = 0
    total_scores = 0
    best_params = None
    best_params_goal = None
    
    closest_miss_dist = 999999.0
    closest_miss_params = None
    
    # Base configuration
    base_yaw = math.pi - math.radians(20)
    base_pitch = math.radians(10)
    base_roll = -math.radians(95)
    
    # Grid search ranges
    # Be careful not to make the search space too massive or it will take minutes
    # We expand X up to -120 to allow the grid search to test very deep wall-clipping impacts
    # Those generate the highest velocity pinches in the physics engine
    x_offs = np.arange(-120, 70, 10)
    y_offs = np.arange(-100, 100, 10)
    z_offs = np.arange(-100, 100, 10)
    delays = [2, 3, 4, 5, 6, 7]
    yaw_offs = np.arange(-0.4, 0.4, 0.05)
    pitch_offs = np.arange(-0.4, 0.4, 0.05)  
    roll_offs = np.arange(-0.4, 0.4, 0.05)
    
    dodge_pitchs = [-1.0] # Always front-flip component
    dodge_rolls = [-1.0, 0.0, 1.0] # Left diagonal, straight front, right diagonal

    total_combinations = len(x_offs) * len(y_offs) * len(z_offs) * len(delays) * len(yaw_offs) * len(pitch_offs) * len(roll_offs) * len(dodge_pitchs) * len(dodge_rolls)
    sample = 1000000
    
    print(f"Testing {sample} random samples out of {total_combinations} possible combinations...")
    
    for idx in range(sample):
        try:
            x_off = random.choice(x_offs)
            y_off = random.choice(y_offs)
            z_off = random.choice(z_offs)
            delay = random.choice(delays)
            yaw_off = random.choice(yaw_offs)
            pitch_off = random.choice(pitch_offs)
            roll_off = random.choice(roll_offs)
            dodge_pitch = random.choice(dodge_pitchs)
            dodge_roll = random.choice(dodge_rolls)
            
            if idx % 5000 == 0 and idx > 0:
                print(f"  Processed {idx}/{sample}... (Goals found: {total_scores})")
                
            # Reset ball state
            arena.ball.set_state(spawn_ball_state)
            
            # Calculate modified car orientation and velocity
            yaw = base_yaw + yaw_off
            pitch = base_pitch + pitch_off
            roll = base_roll + roll_off
            
            c_vel_x = impact_vel.x + V * math.cos(yaw)
            c_vel_y = impact_vel.y + V * math.sin(yaw)
            c_vel_z = impact_vel.z
            
            spd = math.sqrt(c_vel_x**2 + c_vel_y**2 + c_vel_z**2)
            if spd > 2300.0:
                scale = 2300.0 / spd
                c_vel_x *= scale
                c_vel_y *= scale
                c_vel_z *= scale
                
            car_x = (impact_pos.x + x_off) - (c_vel_x * T)
            car_y = (impact_pos.y + y_off) - (c_vel_y * T)
            car_z = (impact_pos.z + z_off) - (c_vel_z * T)
            
            car_state = car.get_state()
            car_state.pos = rsim.Vec(car_x, car_y, car_z)
            car_state.vel = rsim.Vec(c_vel_x, c_vel_y, c_vel_z)
            rot_array = euler_to_rotation(np.array([pitch, yaw, roll], dtype=np.float32))
            car_state.rot_mat = rsim.RotMat(*rot_array.flatten())
            car_state.boost = 100.0
            car.set_state(car_state)
            
            max_v = 0
            scored = False
            
            # Simulate out looking for a goal
            for tick in range(300):  # Simulate up to 2.5 seconds
                controls = rsim.CarControls()
                controls.throttle = 1.0
                controls.boost = True
                if tick >= delay and tick < delay + 5:
                    controls.jump = True
                    controls.pitch = -1.0
                    # For Kuxir, a slight diagonal component helps grab the ball
                    controls.roll = 1.0
                car.set_controls(controls)
                arena.step(1)
                
                bs = arena.ball.get_state()
                bv = bs.vel
                bp = bs.pos
                
                spd_ball = math.sqrt(bv.x**2 + bv.y**2 + bv.z**2)
                if tick < 60 and spd_ball > max_v and bv.z >= 0:
                    max_v = spd_ball
                    
                # Track closest distance to center of orange goal while ball is moving
                dist_to_goal = math.sqrt(bp.x**2 + (bp.y - 5120.0)**2 + (bp.z - 320.0)**2)
                if dist_to_goal < closest_miss_dist:
                    # print(f"New closest miss: {dist_to_goal:.2f} uu")
                    closest_miss_dist = dist_to_goal
                    closest_miss_params = (x_off, y_off, z_off, delay, yaw_off, pitch_off, roll_off, c_vel_x, c_vel_y, c_vel_z, dodge_pitch, dodge_roll)
                    
                # Check if ball enters the blue goal (since we spawn at -2500 Y, the target should be +Y orange goal)
                # Orange Goal bounds: Y > 5120.0, abs(X) < 892.755, Z < 642.775
                if bp.y > 5120.0 and abs(bp.x) < 892.0 and bp.z < 642.0:
                    scored = True
                    break
                    
                # Early exit if the ball is bouncing backward towards our own side
                if tick > 120 and bv.y < 0.0:
                    break
                # Or if it's moving way too slow to reach the goal
                if tick > 120 and spd_ball < 500.0:
                    break
            
            if scored:
                if max_v > best_speed_goal:
                    best_speed_goal = max_v  
                total_scores += 1
                best_params_goal = (x_off, y_off, z_off, delay, yaw_off, pitch_off, roll_off, c_vel_x, c_vel_y, c_vel_z, dodge_pitch, dodge_roll)
            
            if max_v > best_speed and bv.z >= 0:
                best_speed = max_v
                best_params = (x_off, y_off, z_off, delay, yaw_off, pitch_off, roll_off, c_vel_x, c_vel_y, c_vel_z, dodge_pitch, dodge_roll)
                
        except KeyboardInterrupt:
            print(f"\n--- PAUSED at {idx}/{sample} ---")
            print(f"Best Scoring Speed so far: {best_speed_goal:.2f} uu/s ({total_scores} scores found)")
            print(f"Best Non-Scoring Speed so far: {best_speed:.2f} uu/s")
            res = input("Press ENTER to continue searching, or type 'q' to quit and use best parameters: ")
            if res.lower().strip() == 'q':
                break

    if best_params is None:
        print("Optimization complete! NO parameter combination successfully interacted with the ball upwards.")
        print("Defaulting to a known setup.")
        best_params = (0, 0, 0, 4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 1.0)
    elif best_params_goal is None:
        if closest_miss_params is not None:
            print(f"\nOptimization complete, no goals scored! BUT found a close miss (distance to goal center: {closest_miss_dist:.2f} uu)")
            print(f"-> Optimal X Off: {closest_miss_params[0]}, Y Off: {closest_miss_params[1]}, Z Off: {closest_miss_params[2]}, Delay: {closest_miss_params[3]}")
            print(f"-> Optimal Yaw Offset: {closest_miss_params[4]:.3f}, Pitch Offset: {closest_miss_params[5]:.3f}, Roll Offset: {closest_miss_params[6]:.3f}")
            print(f"-> Dodge Pitch: {closest_miss_params[10]:.1f}, Dodge Roll: {closest_miss_params[11]:.1f}")
            print(f"\nOptimization complete, no goals scored! Best Non-Scoring Speed: {best_speed:.2f} uu/s")
            print(f"-> Optimal X Off: {best_params[0]}, Y Off: {best_params[1]}, Z Off: {best_params[2]}, Delay: {best_params[3]}")
            print(f"-> Optimal Yaw Offset: {best_params[4]:.3f}, Pitch Offset: {best_params[5]:.3f}, Roll Offset: {best_params[6]:.3f}")
            print(f"-> Dodge Pitch: {best_params[10]:.1f}, Dodge Roll: {best_params[11]:.1f}")
            best_p = closest_miss_params
        else:
            print(f"\nOptimization complete, no goals scored! Best Non-Scoring Speed: {best_speed:.2f} uu/s")
            print(f"-> Optimal X Off: {best_params[0]}, Y Off: {best_params[1]}, Z Off: {best_params[2]}, Delay: {best_params[3]}")
            print(f"-> Optimal Yaw Offset: {best_params[4]:.3f}, Pitch Offset: {best_params[5]:.3f}, Roll Offset: {best_params[6]:.3f}")
            print(f"-> Dodge Pitch: {best_params[10]:.1f}, Dodge Roll: {best_params[11]:.1f}")
            best_p = best_params
    else:
        print(f"Optimization complete! Best Scoring Speed: {best_speed_goal:.2f} uu/s")
        print(f"-> Optimal X Off: {best_params_goal[0]}, Y Off: {best_params_goal[1]}, Z Off: {best_params_goal[2]}, Delay: {best_params_goal[3]}")
        print(f"-> Optimal Yaw Offset: {best_params_goal[4]:.3f}, Pitch Offset: {best_params_goal[5]:.3f}, Roll Offset: {best_params_goal[6]:.3f}")
        print(f"Best Speed overall: {best_speed:.2f} uu/s")
        print(f"-> Optimal X Off: {best_params[0]}, Y Off: {best_params[1]}, Z Off: {best_params[2]}, Delay: {best_params[3]}")
        print(f"-> Optimal Yaw Offset: {best_params[4]:.3f}, Pitch Offset: {best_params[5]:.3f}, Roll Offset: {best_params[6]:.3f}")
        
    # Set best parameters! Prefer goal scoring ones
    best_p = best_params_goal if best_params_goal is not None else best_p
    
    # Now that we found the best params, set them up for the Golden Seed and Visualizer
    best_x, best_y, best_z, best_delay, best_yaw_off, best_pitch_off, best_roll_off, best_cvx, best_cvy, best_cvz, best_dp, best_dr = best_p
    
    # Reconstruct optimal car state
    yaw = base_yaw + best_yaw_off
    pitch = base_pitch + best_pitch_off
    roll = base_roll + best_roll_off
    
    if best_cvx == 0.0 and best_cvy == 0.0 and best_cvz == 0.0:
        c_vel_x = impact_vel.x + V * math.cos(yaw)
        c_vel_y = impact_vel.y + V * math.sin(yaw)
        c_vel_z = impact_vel.z
        spd = math.sqrt(c_vel_x**2 + c_vel_y**2 + c_vel_z**2)
        if spd > 2300.0:
            scale = 2300.0 / spd
            c_vel_x *= scale
            c_vel_y *= scale
            c_vel_z *= scale
    else:
        c_vel_x, c_vel_y, c_vel_z = best_cvx, best_cvy, best_cvz
        
    arena.ball.set_state(spawn_ball_state)
    
    impact_x = impact_pos.x + best_x
    impact_y = impact_pos.y + best_y
    impact_z = impact_pos.z + best_z

    car_x = impact_x - (c_vel_x * T)
    car_y = impact_y - (c_vel_y * T)
    car_z = impact_z - (c_vel_z * T)
    
    car_state = car.get_state()
    car_state.pos = rsim.Vec(car_x, car_y, car_z)
    car_state.vel = rsim.Vec(c_vel_x, c_vel_y, c_vel_z)
    rot_array = euler_to_rotation(np.array([pitch, yaw, roll], dtype=np.float32))
    car_state.rot_mat = rsim.RotMat(*rot_array.flatten())
    car_state.boost = 100.0
    car.set_state(car_state)

    print("\n--- Golden Seed Generated ---")
    print("Ball Spawn State:")
    print(f"  Pos: [{spawn_pos.x:.2f}, {spawn_pos.y:.2f}, {spawn_pos.z:.2f}]")
    print(f"  Vel: [{spawn_ball_state.vel.x:.2f}, {spawn_ball_state.vel.y:.2f}, {spawn_ball_state.vel.z:.2f}]")
    print("\nImpact Target State:")
    print(f"  Pos: [{impact_pos.x:.2f}, {impact_pos.y:.2f}, {impact_pos.z:.2f}]")
    
    print("\nCar State:")
    print(f"  Pos: [{car_x:.2f}, {car_y:.2f}, {car_z:.2f}]")
    print(f"  Vel: [{car_state.vel.x:.2f}, {car_state.vel.y:.2f}, {car_state.vel.z:.2f}]")
    print(f"  Euler (Pitch/Yaw/Roll): [{pitch:.2f}, {yaw:.2f}, {roll:.2f}]")
    
    try:
        import rlviser_py as vis
        import time
        
        print("\nLaunching RLViser to view the generated state for 10 seconds...")
        print("NOTE: If Windows Firewall prompts you, you MUST allow it for private networks.")
        print("rlviser.exe requires local UDP access to stream the physics data.")
        # Set boost pad locations
        vis.set_boost_pad_locations([pad.get_pos().as_tuple() for pad in arena.get_boost_pads()])
        
        # Controls are applied dynamically in the main loop below.
        
        # Sleep to let rlviser.exe fully initialize its UDP server and bind ports appropriately.
        # This prevents the "memory allocation of 72057... bytes failed" Rust panic on boot.
        time.sleep(2.0)
        
        print("Showing generated state statically for 15 seconds...")
        pad_states = [pad.get_state().is_active for pad in arena.get_boost_pads()]
        b_state = arena.ball.get_state()
        car_data = [
            (c.id, c.team, c.get_config(), c.get_state())
            for c in arena.get_cars()
        ]
        
        pause_ticks = int(15.0 * arena.tick_rate)
        for _ in range(pause_ticks):
            vis.render(0, arena.tick_rate, rsim.GameMode.SOCCAR, pad_states, b_state, car_data)
            time.sleep(1.0 / arena.tick_rate)
            
        TIME = 10.0
        steps = 0
        start_time = time.time()
        
        for _ in range(round(TIME * arena.tick_rate)):
            arena.step(1)

            # Render the current game state
            pad_states = [pad.get_state().is_active for pad in arena.get_boost_pads()]
            b_state = arena.ball.get_state()
            car_data = [
                (c.id, c.team, c.get_config(), c.get_state())
                for c in arena.get_cars()
            ]

            vis.render(steps, arena.tick_rate, rsim.GameMode.SOCCAR, pad_states, b_state, car_data)

            # sleep to simulate running real time
            time.sleep(max(0, start_time + steps / arena.tick_rate - time.time()))
            steps += 1
        
        print("Exiting visualizer...")
        vis.quit()
        
    except Exception as e:
        print("\nGolden seed parameters logged above.")
        print("rlviser render skipped:", e)
    
if __name__ == "__main__":
    main()
