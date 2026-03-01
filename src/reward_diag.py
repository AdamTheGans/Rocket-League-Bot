"""Diagnostic: check if rewards flow correctly through the env."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
import numpy as np

from envs.grounded_strike import build_env

env = build_env(render=False, tick_skip=8, episode_seconds=12.0)

lines = []
def log(msg):
    print(msg)
    lines.append(msg)

log('=== Reward Diagnostic: Running 5 episodes ===')
log('')

all_returns = []
for ep in range(5):
    obs = env.reset()
    ep_reward = 0.0
    ep_steps = 0
    step_rewards = []

    while True:
        action = np.array([[np.random.randint(0, 90)]], dtype=np.int32)
        result = env.step(action)
        obs, rew, done, trunc, info = result
        if isinstance(rew, (list, np.ndarray)):
            r = float(rew[0]) if hasattr(rew, '__len__') and len(rew) > 0 else float(rew)
        else:
            r = float(rew)
        ep_reward += r
        ep_steps += 1
        step_rewards.append(r)
        if done or trunc:
            break

    scored = bool(done and not trunc)
    rewards_arr = np.array(step_rewards)
    all_returns.append(ep_reward)

    log(f'Episode {ep}: steps={ep_steps} scored={scored} total_return={ep_reward:.4f}')
    log(f'  Per-step: mean={rewards_arr.mean():.6f} std={rewards_arr.std():.6f} min={rewards_arr.min():.6f} max={rewards_arr.max():.6f}')
    log(f'  Non-zero: {np.count_nonzero(rewards_arr)}/{len(rewards_arr)}')
    log('')

returns = np.array(all_returns)
log(f'Return variance: mean={returns.mean():.4f} std={returns.std():.4f}')
log(f'Returns: {[f"{r:.4f}" for r in all_returns]}')

# Test constant forward driving
log('')
log('=== Constant action 0 for 20 steps ===')
obs = env.reset()
state = env.rlgym_env.state
car = state.cars[list(state.cars.keys())[0]]
log(f'  Start: car_pos={car.physics.position} forward={car.physics.forward} ball_pos={state.ball.position}')

for i in range(20):
    action = np.array([[0]], dtype=np.int32)
    obs, rew, done, trunc, info = env.step(action)
    r = float(rew[0]) if isinstance(rew, (list, np.ndarray)) else float(rew)
    state = env.rlgym_env.state
    car = state.cars[list(state.cars.keys())[0]]
    car_speed = np.linalg.norm(car.physics.linear_velocity)
    ball_dist = np.linalg.norm(state.ball.position - car.physics.position)
    log(f'  Step {i:2d}: rew={r:>8.4f} speed={car_speed:>7.1f} ball_dist={ball_dist:>7.1f}')

env.close()

# Write to file
with open('checkpoints/reward_diag.txt', 'w') as f:
    f.write('\n'.join(lines))
log('Saved to checkpoints/reward_diag.txt')
