import sys; sys.path.insert(0, 'src')
import numpy as np

# Test 1: Build env and step
print('=== Test 1: Build env ===')
from envs.grounded_strike import build_env
env = build_env(render=False, tick_skip=8, episode_seconds=12.0)
print(f'Obs space: {env.observation_space.shape}')
print(f'Act space: {env.action_space.n}')

# Test 2: Reset and check state
print('\n=== Test 2: Reset + inspect state ===')
obs = env.reset()
print(f'Obs shape: {obs.shape}, dtype: {obs.dtype}')
print(f'Obs range: [{obs.min():.3f}, {obs.max():.3f}]')

# Check car state
rlgym_env = env.rlgym_env
state = rlgym_env.state
car_ids = list(state.cars.keys())
car = state.cars[car_ids[0]]
print(f'Car pos: {car.physics.position}')
print(f'Car euler: {car.physics.euler_angles}')
print(f'Car forward: {car.physics.forward}')
print(f'Car boost: {car.boost_amount}')
print(f'Ball pos: {state.ball.position}')

# Test 3: Step with random action
print('\n=== Test 3: 10 random steps ===')
for i in range(10):
    action = np.array([[np.random.randint(0, 90)]], dtype=np.int32)
    result = env.step(action)
    if len(result) == 5:
        obs, rew, done, trunc, info = result
    else:
        obs, rew, done, info = result
        trunc = False
    if i == 0:
        print(f'Step 0: obs={obs.shape}, rew={rew}, done={done}, trunc={trunc}')

# Test 4: Multiple resets to check spawning works
print('\n=== Test 4: 5 resets (check spawn variety) ===')
for i in range(5):
    env.reset()
    state = rlgym_env.state
    car = state.cars[list(state.cars.keys())[0]]
    print(f'  Reset {i}: car_y={car.physics.position[1]:.0f} yaw={car.physics.euler_angles[1]:.2f} boost={car.boost_amount:.0f}')

print('\n=== ALL TESTS PASSED ===')
env.close()