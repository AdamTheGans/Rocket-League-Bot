# src/verify_pinch_env.py
"""
Quick sanity test for the pinch specialist environment across all 3 stages.
Verifies: env builds, resets produce valid observations, ball spawns near wall,
steps execute without error, rewards are finite.
"""
import sys; sys.path.insert(0, 'src')
import numpy as np

from rlgym.rocket_league import common_values


def test_stage(stage: int):
    print(f"\n{'='*40}")
    print(f"  STAGE {stage} -- Testing pinch environment")
    print(f"{'='*40}")

    from envs.pinch import build_env
    env = build_env(render=False, tick_skip=8, stage=stage)
    print(f"  Obs space: {env.observation_space.shape}")
    print(f"  Act space: {env.action_space.n}")

    # Test resets
    print(f"\n  --- 5 resets ---")
    rlgym_env = env.rlgym_env
    wall_violations = 0

    for i in range(5):
        obs = env.reset()
        state = rlgym_env.state
        car_ids = list(state.cars.keys())
        car = state.cars[car_ids[0]]

        ball_x = float(state.ball.position[0])
        ball_y = float(state.ball.position[1])
        ball_z = float(state.ball.position[2])
        car_x = float(car.physics.position[0])
        car_y = float(car.physics.position[1])
        car_yaw = float(car.physics.euler_angles[1])
        boost = float(car.boost_amount)

        ball_wall_dist = common_values.SIDE_WALL_X - abs(ball_x)
        car_ball_dist = float(np.linalg.norm(
            np.array(car.physics.position[:2]) - np.array(state.ball.position[:2])
        ))

        print(f"  Reset {i}: ball=({ball_x:.0f},{ball_y:.0f},{ball_z:.0f}) "
              f"wall_dist={ball_wall_dist:.0f} "
              f"car_ball={car_ball_dist:.0f} "
              f"yaw={car_yaw:.2f} boost={boost:.0f}")

        # Stage 1: ball should be very close to wall
        if stage == 1 and ball_wall_dist > 300:
            wall_violations += 1
            print(f"    WARNING: ball too far from wall for stage 1 (dist={ball_wall_dist:.0f})")

    if wall_violations > 2:
        print(f"  FAIL: Too many wall violations ({wall_violations}/5)")
        return False

    # Test stepping
    print(f"\n  --- 10 random steps ---")
    obs = env.reset()
    for i in range(10):
        action = np.array([[np.random.randint(0, 90)]], dtype=np.int32)
        result = env.step(action)
        if len(result) == 5:
            obs, rew, done, trunc, info = result
        else:
            obs, rew, done, info = result
            trunc = False

        if i == 0:
            rew_f = float(rew[0]) if isinstance(rew, (list, np.ndarray)) else float(rew)
            print(f"  Step 0: obs={obs.shape} rew={rew_f:.4f} done={done} trunc={trunc}")

        # Check reward is finite
        rew_val = float(rew) if not isinstance(rew, (list, tuple)) else float(rew[0])
        if not np.isfinite(rew_val):
            print(f"  FAIL: Non-finite reward at step {i}: {rew_val}")
            return False

    # Test reward function builds
    from rewards.pinch_reward import build_pinch_reward
    reward = build_pinch_reward(stage)
    print(f"  Reward function: {type(reward).__name__}")

    print(f"  Stage {stage}: PASSED")
    env.close()
    return True


if __name__ == "__main__":
    all_passed = True
    for stage in [1, 2, 3]:
        try:
            if not test_stage(stage):
                all_passed = False
        except Exception as e:
            print(f"\n  Stage {stage}: FAILED with error: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False

    print(f"\n{'='*40}")
    if all_passed:
        print("=== ALL PINCH ENV TESTS PASSED ===")
    else:
        print("=== SOME TESTS FAILED ===")
    print(f"{'='*40}")
