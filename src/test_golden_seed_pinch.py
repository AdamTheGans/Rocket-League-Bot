import os
import numpy as np
import RocketSim as rsim
from rlgym.rocket_league.sim import RocketSimEngine
from rlgym.rocket_league.math import euler_to_rotation
import math

from eval_specialist_1 import _save_topdown_gif_labeled

def get_state_dict(arena):
    car = arena.get_cars()[0]
    car_state = car.get_state()
    ball_state = arena.ball.get_state()
    
    fwd = car_state.rot_mat.forward
    yaw = math.atan2(fwd.y, fwd.x)
    
    return {
        "car_pos": [car_state.pos.x, car_state.pos.y, car_state.pos.z],
        "ball_pos": [ball_state.pos.x, ball_state.pos.y, ball_state.pos.z],
        "boost": car_state.boost,
        "yaw": yaw
    }

def main():
    print("Initializing RocketSim...")
    _dummy = RocketSimEngine()
    arena = rsim.Arena(rsim.GameMode.SOCCAR)
    
    car = arena.add_car(rsim.Team.BLUE, rsim.CarConfig(rsim.CarConfig.DOMINUS))
    
    pitch, yaw, roll = 0.0, math.pi / 2.0, -math.pi / 2.0
    rot_array = euler_to_rotation(np.array([pitch, yaw, roll], dtype=np.float32))
    
    frames = []
    
    # Let's do 10 random attempts near the same spot
    num_attempts = 10
    
    import rlviser_py as vis
    import time
    vis.set_boost_pad_locations([pad.get_pos().as_tuple() for pad in arena.get_boost_pads()])
    print("Waiting 5.0s for RLViser to boot...")
    time.sleep(5.0)
    
    total_steps = 0
    
    for i in range(num_attempts):
        delay = np.random.randint(2, 7)
        y_slide = np.random.uniform(-1000.0, 1000.0)
        
        print(f"Simulating attempt {i+1}/{num_attempts} (Jump delay: {delay} ticks, Y-Offset: {y_slide:.1f})")
        
        # ── Setup Golden Seed ──
        ball_state = arena.ball.get_state()
        ball_state.pos = rsim.Vec(4004.75, -2391.86 + y_slide, 308.11)
        ball_state.vel = rsim.Vec(1993.41, 996.71, 1424.75)
        arena.ball.set_state(ball_state)
        
        car_state = car.get_state()
        car_state.pos = rsim.Vec(3964.75, -2441.86 + y_slide, 308.11)
        # Apply a slight inward velocity to mimic the jump/dodge approach
        car_state.vel = rsim.Vec(400.00, 1996.71, 1424.75)
        car_state.rot_mat = rsim.RotMat(*rot_array.flatten())
        car_state.boost = 100.0
        car.set_state(car_state)
        
        # Freeze on the last attempt so the user can position the camera
        if i == num_attempts - 1:
            print("Final attempt: Pausing for 15 seconds so you can position the camera!")
            pad_states = [pad.get_state().is_active for pad in arena.get_boost_pads()]
            b_state = arena.ball.get_state()
            car_data = [(c.id, c.team, c.get_config(), c.get_state()) for c in arena.get_cars()]
            
            # We must repeatedly send the state to rlviser, or it will despawn them
            pause_ticks = int(15.0 * arena.tick_rate)
            for _ in range(pause_ticks):
                vis.render(total_steps, arena.tick_rate, rsim.GameMode.SOCCAR, pad_states, b_state, car_data)
                time.sleep(1.0 / arena.tick_rate)
                total_steps += 1
        
        ep_frames = []
        
        # 120 ticks = 1 second. We simulate for 60 ticks (0.5s) to see the pinch
        for tick in range(60):
            controls = rsim.CarControls()
            controls.throttle = 1.0
            controls.boost = True
            
            # Hardcoded dodge logic (Right dodge into the wall)
            if tick == delay:
                controls.jump = True
            elif tick > delay and tick < delay + 4:
                controls.jump = True # hold jump
            elif tick == delay + 4:
                controls.jump = False # release before second jump
            elif tick == delay + 5:
                # Second jump (dodge)
                controls.jump = True
                controls.steer = 1.0  # Right (into the right wall)
                controls.pitch = 0.0
                controls.roll = 1.0   # Right roll
            elif tick > delay + 5 and tick < delay + 10:
                controls.jump = True
                controls.steer = 1.0
                controls.pitch = 0.0
                controls.roll = 1.0
            
            car.set_controls(controls)
            arena.step(1)
            
            pad_states = [pad.get_state().is_active for pad in arena.get_boost_pads()]
            b_state = arena.ball.get_state()
            car_data = [(c.id, c.team, c.get_config(), c.get_state()) for c in arena.get_cars()]
            vis.render(total_steps, arena.tick_rate, rsim.GameMode.SOCCAR, pad_states, b_state, car_data)
            time.sleep(1.0 / arena.tick_rate)
            total_steps += 1
            
            ep_frames.append(get_state_dict(arena))
            
        frames.append(ep_frames)
        time.sleep(0.5) # Slight pause between variation previews

    vis.quit()

    out_path = os.path.join("checkpoints", "golden_seed_tests.gif")
    print("Generating GIF using top-down evaluation tool...")
    _save_topdown_gif_labeled(
        frames,
        out_path,
        title="Golden Seed Variations",
        attack_orange=True
    )
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
