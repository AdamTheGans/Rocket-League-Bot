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
    # Moving up and fast towards the positive X wall, and forward +Y
    ball_state.vel = rsim.Vec(2000.0, 1000.0, 1500.0) 
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

    # Position car ~20-30 uu behind the ball on the Y axis, and slightly off the wall on the X axis.
    # Since ball_vel is going +Y, the car should be at a lesser Y.
    car_x = ball_pos.x - 40.0  # slightly off the right wall (further negative X)
    car_y = ball_pos.y - 50.0  # behind the ball in its path
    car_z = ball_pos.z         # level with the ball

    # Orient the car to face the ball/goal
    # Yaw: facing the +Y direction (since we're attacking +Y)
    # Roll: Right wall surface normal points in -X. So the roof should face -X, wheels face +X.
    # A roll of -90 degrees (-pi/2) usually puts wheels towards +X.
    
    pitch = 0.0
    yaw = math.pi / 2.0  # facing +Y
    roll = -math.pi / 2.0 # roll left, so right wheels hit the right wall

    car_state = car.get_state()
    car_state.pos = rsim.Vec(car_x, car_y, car_z)
    # Moving forward faster than the ball to pinch it, and matching Z-velocity exactly!
    car_state.vel = rsim.Vec(400.0, ball_vel.y + 1000.0, ball_vel.z) # 400.0 X to push hard into the wall

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
        
        print("\nLaunching RLViser to view the generated state for 25 seconds...")
        print("NOTE: If Windows Firewall prompts you, you MUST allow it for private networks.")
        print("rlviser.exe requires local UDP access to stream the physics data.")
        # Set boost pad locations
        vis.set_boost_pad_locations([pad.get_pos().as_tuple() for pad in arena.get_boost_pads()])
        
        # We apply a default forward drive so we can see the car attempt to hit the ball
        car.set_controls(rsim.CarControls(throttle=1.0, boost=True))
        
        # Sleep to let rlviser.exe fully initialize its UDP server and bind ports appropriately.
        # This prevents the "memory allocation of 72057... bytes failed" Rust panic on boot.
        time.sleep(2.0)
        
        TIME = 25.0
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
