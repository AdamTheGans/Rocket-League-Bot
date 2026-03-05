import time
import math
import numpy as np
import rlviser_py as vis
import RocketSim as rsim
from envs.pinch import build_env

def main():
    print("Initializing RLGym Env...")
    env = build_env(render=False, tick_skip=8, stage=2, difficulty_level=1)
    
    rlgym_env = getattr(env, "rlgym_env", env.unwrapped.rlgym_env if hasattr(env, "unwrapped") and hasattr(env.unwrapped, "rlgym_env") else env)
    engine = rlgym_env.transition_engine
    arena = engine._arena
    
    print("\nLaunching RLViser...")
    vis.set_boost_pad_locations([pad.get_pos().as_tuple() for pad in arena.get_boost_pads()])
    time.sleep(2.0)
    
    for i in range(10):
        print(f"\n--- Showing Stage 2 Random Spawn {i+1}/10 ---")
        
        # Reset the environment to generate a new Stage 2 seed
        env.reset()
        
        # Pull initial render states
        pad_states = [pad.get_state().is_active for pad in arena.get_boost_pads()]
        b_state = arena.ball.get_state()
        try:
            car_data = [(c.id, c.team, c.get_config(), c.get_state()) for c in arena.get_cars()]
        except:
            car_data = []

        # Freeze for 5 seconds to let the user observe the spawn points before the physics run
        print("Freezing for 5 seconds...")
        freeze_seconds = 5.0
        start_time = time.time()
        for i in range(int(freeze_seconds * 120)):
            vis.render(0, 120, rsim.GameMode.SOCCAR, pad_states, b_state, car_data)
            target_time = start_time + (i / 120.0)
            now = time.time()
            if target_time > now:
                time.sleep(target_time - now)
        
        # Simulate physics for 3 seconds so we can see the ballistic arc
        # We step the underlying arena 1 tick at a time for 120fps smooth viewing
        
        # Calculate car speed
        try:
            car = arena.get_cars()[0]
            c_vel = car.get_state().vel
            speed = math.sqrt(c_vel.x**2 + c_vel.y**2 + c_vel.z**2)
            print(f"Car Initial Speed: {speed:.1f} uu/s (Supersonic is 2200 uu/s)")
        except:
            pass

        print("Simulating physics for 3 seconds...")
        sim_seconds = 3.0
        start_time = time.time()
        for tick in range(int(sim_seconds * 120)):
            controls = rsim.CarControls()
            controls.boost = True
            controls.throttle = 1.0
            
            for car in arena.get_cars():
                car.set_controls(controls)
                
            arena.step(1)
            
            pad_states = [pad.get_state().is_active for pad in arena.get_boost_pads()]
            b_state = arena.ball.get_state()
            try:
                car_data = [(c.id, c.team, c.get_config(), c.get_state()) for c in arena.get_cars()]
            except:
                car_data = []
            vis.render(0, 120, rsim.GameMode.SOCCAR, pad_states, b_state, car_data)
            
            # Sleep to match real time
            target_time = start_time + (tick / 120.0)
            now = time.time()
            if target_time > now:
                time.sleep(target_time - now)
            
    print("\nVisualizations complete!")
    vis.quit()

if __name__ == "__main__":
    main()
