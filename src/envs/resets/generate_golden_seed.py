# src/envs/resets/generate_golden_seed.py
import math
import time
import numpy as np

import RocketSim as rsim

# Physics Constants
SIDE_WALL_X = 4096.0
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

    # Spawn ball near the right wall (+X), closer to own net (-Y)
    ball_state = arena.ball.get_state()
    ball_state.pos = rsim.Vec(3800.0, -2500.0, 150.0)
    # Moving up and fast towards the positive X wall, and slowly forward +Y
    ball_state.vel = rsim.Vec(2000.0, 150.0, 1500.0) 
    arena.ball.set_state(ball_state)
    
    # Add a dominus car to the arena
    car = arena.add_car(rsim.Team.BLUE, rsim.CarConfig(rsim.CarConfig.DOMINUS))

    print("Rolling simulation until ball is flush on the wall...")

    # Wait for the ball to roll up the wall curve and be on the flat wall part
    ticks = 0
    while True:
        arena.step(1)
        ticks += 1
        
        ball_state = arena.ball.get_state()
        ball_x = ball_state.pos.x
        ball_z = ball_state.pos.z
        
        if ticks % 120 == 0:
            print(f"Tick {ticks}: Ball pos: [{ball_x:.1f}, {ball_state.pos.y:.1f}, {ball_z:.1f}] | Vel: [{ball_state.vel.x:.1f}, {ball_state.vel.y:.1f}, {ball_state.vel.z:.1f}]")

        # Check if the ball is on the flat part of the side wall (right wall, so positive X)
        if ball_z > 300.0 and ball_x >= SIDE_WALL_X - BALL_RADIUS - 5.0:
            print(f"Found flush wall contact at tick {ticks}!")
            break
            
        if ticks > 1200:
            print("Ball never reached the wall. Bailing out.")
            break

    # We found the perfect moment. Now let's place the car.
    # The user requested placing it slightly off-surface/with initial Z velocity 
    # to mimic a jump already in progress.
    
    ball_state = arena.ball.get_state()
    
    # Force perfectly flush on the right wall
    ball_state.pos.x = SIDE_WALL_X - BALL_RADIUS
    arena.ball.set_state(ball_state)
    
    ball_pos = ball_state.pos
    ball_vel = ball_state.vel

    # Position car ~200 uu towards center field (-X) and ~200 uu behind the ball (-Y)
    # This gives it a 0.2s run-up to intercept the ball.
    car_x = ball_pos.x - 200.0  
    car_y = ball_pos.y - 200.0 
    car_z = ball_pos.z         

    # Orient the car to face the ball/goal
    # Yaw: facing the ball center (-45 degrees / 315 degrees). 
    # Since +X is Forward, and +Y is Left in Unreal/RLGym coords:
    # A positive Y value means left. The ball is at +X, +Y (+200, +200).
    # Wait: The ball is at X=4000, Y=-2400.
    # The car is at X=3800, Y=-2600.
    # So relative to car, ball is +200 X, +200 Y.
    # In Rocket League, `Yaw = math.atan2(dy, dx)`. 
    # But if it spawned facing 90 deg LEFT of the ball, we need to shift Yaw right by 90 deg.
    # math.pi / 4 (45deg) - math.pi / 2 (90deg) = -math.pi / 4 (-45deg or 315deg).
    pitch = 0.0
    yaw = -math.pi / 4.0   
    roll = 0.0 

    car_state = car.get_state()
    car_state.pos = rsim.Vec(car_x, car_y, car_z)
    
    # car_vel_x = ball_vel_x + (dx / 0.2) = ball_vel_x + 1000.0
    # car_vel_y = ball_vel_y + (dy / 0.2) = ball_vel_y + 1000.0
    # car_vel_z = ball_vel_z (level flight)
    car_state.vel = rsim.Vec(ball_vel.x + 1000.0, ball_vel.y + 1000.0, ball_vel.z) 

    from rlgym.rocket_league.math import euler_to_rotation
    
    # car_state.rot_mat expects a list of floats or a RotMat
    # The rlgym math utils return a flat array or a numpy matrix type.
    rot_array = euler_to_rotation(np.array([pitch, yaw, roll], dtype=np.float32))
    
    # Rsim python bindings for RotMat:
    rot = rsim.RotMat(*rot_array.flatten())
    car_state.rot_mat = rot
    
    car.set_state(car_state)

    print("\n--- Golden Seed Generated ---")
    print("Ball State:")
    print(f"  Pos: [{ball_pos.x:.2f}, {ball_pos.y:.2f}, {ball_pos.z:.2f}]")
    print(f"  Vel: [{ball_vel.x:.2f}, {ball_vel.y:.2f}, {ball_vel.z:.2f}]")
    
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
        
        # We apply a default forward drive so we can see the car attempt to hit the ball
        car.set_controls(rsim.CarControls(throttle=1.0, boost=True))
        
        # Sleep to let rlviser.exe fully initialize its UDP server and bind ports appropriately.
        # This prevents the "memory allocation of 72057... bytes failed" Rust panic on boot.
        time.sleep(2.0)
        
        print("Pausing for 10 seconds so you can position the camera...")
        pad_states = [pad.get_state().is_active for pad in arena.get_boost_pads()]
        b_state = arena.ball.get_state()
        car_data = [
            (c.id, c.team, c.get_config(), c.get_state())
            for c in arena.get_cars()
        ]
        pause_ticks = int(10.0 * arena.tick_rate)
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
