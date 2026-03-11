# src/visualize_mixed_ppo.py
"""
Watch PPO checkpoints play in a mixed environment (Kuxir setups + Self-Play) via RLViser.

Usage:
    cd src
    python visualize_mixed_ppo.py \
        --checkpoint ../checkpoints/kuxir_curriculum/kuxir_ppo_3200000_steps.zip \
        --opponent student_bc \
        --opponent_checkpoint ../checkpoints/nexto_distill/student_policy.pt \
        --difficulty 0.2 --noise 0.075 \
        --episodes 10 --viser
"""
from __future__ import annotations

import argparse
import os
import time
from typing import Optional

import numpy as np
import torch
from stable_baselines3 import PPO

from rlgym.api import RLGym
from rlgym.rocket_league.sim import RocketSimEngine
from rlgym.rocket_league.obs_builders import DefaultObs
from rlgym.rocket_league.action_parsers import LookupTableAction, RepeatAction
from rlgym.rocket_league.done_conditions import GoalCondition, TimeoutCondition, AnyCondition
from rlgym.rocket_league.reward_functions import CombinedReward
from rlgym.rocket_league.state_mutators import MutatorSequence, FixedTeamSizeMutator, KickoffMutator
from rlgym.rocket_league import common_values

# ===================================================================== #
#  TODO: Adjust these imports to match your project structure!
# ===================================================================== #
try:
    from state_setters.mixed_state_setter import MixedStateSetter
    from state_setters.trajectory_setter import MechanicTrajectorySetter
    from rewards.mixed_reward import build_mixed_reward
    from envs.mixed_training_env import CurriculumRewardWrapper
except ImportError:
    print("WARNING: Could not import seters. Run from src/ dir.")
    MixedStateSetter = type("MixedStateSetter", (object,), {})
    MechanicTrajectorySetter = type("MechanicTrajectorySetter", (object,), {})
    build_mixed_reward = lambda **kwargs: None
    CurriculumRewardWrapper = type("CurriculumRewardWrapper", (object,), {})

class MockCurriculum:
    class MockValue:
        def __init__(self, val): self.value = val
    def __init__(self, diff, noise):
        self.difficulty = self.MockValue(diff)
        self.noise_amount = self.MockValue(noise)
        self.outcomes_queue = type("MockQueue", (object,), {"put": lambda self, x: None})()

# We will skip importing nexto_distill if it breaks, as we are mainly using "idle" to test spawns
try:
    from nexto_distill.eval_imitation import _load_student
    from nexto_distill.teacher_nexto import NextoTeacher
except ImportError:
    pass


# ===================================================================== #
#  Policy & Opponent loaders
# ===================================================================== #

def _load_ppo_policy(checkpoint: str, device: str):
    """Load SB3 PPO model or rlgym-ppo model."""
    if not checkpoint or not os.path.exists(checkpoint):
        print(f"  [WARN] Checkpoint not found: {checkpoint}. Returning random fallback policy.")
        return lambda obs, gs, aid: np.random.randint(0, 90) # Lookup table has ~90 actions
        
    print(f"  Loading checkpoint: {checkpoint}")
    
    # 1. Try rlgym-ppo first
    ppo_pt_path = os.path.join(checkpoint, "PPO_POLICY.pt")
    if os.path.isfile(ppo_pt_path):
        print(f"  Detected rlgym-ppo checkpoint. Loading PPO_POLICY.pt...")
        try:
            from rlgym_ppo.ppo.discrete_policy import DiscreteFF
            state_dict = torch.load(ppo_pt_path, map_location=device)
            
            # Infer observation space from first layer
            first_layer_key = next((k for k in state_dict.keys() if 'weight' in k and '0' in k), None)
            if first_layer_key and len(state_dict[first_layer_key].shape) == 2:
                obs_size = state_dict[first_layer_key].shape[1]
            else:
                obs_size = 92
                
            model = DiscreteFF(
                input_shape=int(obs_size),
                n_actions=90,
                layer_sizes=[2048, 1024, 1024, 512],
                device=device
            )
            model.load_state_dict(state_dict)
            model.eval()
            
            def policy_fn(obs_dict, game_state, blue_agent):
                obs = obs_dict[blue_agent].astype(np.float32)
                t = torch.from_numpy(obs).unsqueeze(0).to(device)
                with torch.no_grad():
                    action, _ = model.get_action(t, deterministic=True)
                return int(action.item())
                
            return policy_fn
        except Exception as e:
            print(f"  [Error loading rlgym-ppo model] {e}. Using random fallback.")
            return lambda obs, gs, aid: np.random.randint(0, 90)

    # 2. Try SB3 fallback
    try:
        model = PPO.load(checkpoint, device=device)
        def policy_fn(obs_dict, game_state, blue_agent):
            obs = obs_dict[blue_agent].astype(np.float32)
            action, _ = model.predict(obs, deterministic=True)
            return int(action)
        return policy_fn
    except Exception as e:
        print(f"  [Error loading SB3 model] {e}. Using random fallback.")
        return lambda obs, gs, aid: np.random.randint(0, 90)

def _make_opponent(opponent_name: str, checkpoint: Optional[str], device: str, tick_skip: int):
    """Load the opponent (Student BC, Teacher Nexto, or Idle)."""
    if opponent_name == "idle":
        print("  Opponent: Idle (action 0)")
        return (lambda obs, gs, aid: 0), None

    elif opponent_name == "teacher_nexto":
        opp_teacher = NextoTeacher(device=device, tick_skip=tick_skip)
        print(f"  Opponent: Teacher Nexto")
        return (lambda obs, gs, aid: opp_teacher.act(gs, player_index=1)), opp_teacher

    elif opponent_name == "student_bc":
        if not checkpoint or not os.path.isfile(checkpoint):
            raise FileNotFoundError(f"Opponent checkpoint required/not found: {checkpoint}")
        dev = torch.device(device)
        opp_model, opp_meta = _load_student(checkpoint, device)
        print(f"  Opponent: Student BC (params={opp_meta.get('total_params', '?'):,})")

        def opp_fn(obs_dict, game_state, agent_id):
            obs = obs_dict[agent_id]
            with torch.no_grad():
                t = torch.from_numpy(obs).float().unsqueeze(0).to(dev)
                logits = opp_model(t)
                return int(logits.argmax(dim=-1).item())

        return opp_fn, None

    raise ValueError(f"Unknown opponent: {opponent_name}")


# ===================================================================== #
#  RLViser setup
# ===================================================================== #

def _setup_rlviser(env):
    import rlviser_py as vis
    import RocketSim as rsim

    arena = env.transition_engine._arena
    vis.set_boost_pad_locations([pad.get_pos().as_tuple() for pad in arena.get_boost_pads()])
    time.sleep(2.0)
    print("  RLViser initialized")
    return vis, arena, rsim

def _render_frame(vis, arena, rsim):
    pad_states = [p.get_state().is_active for p in arena.get_boost_pads()]
    b_state = arena.ball.get_state()
    try:
        car_data = [(c.id, c.team, c.get_config(), c.get_state()) for c in arena.get_cars()]
    except Exception:
        car_data = []
    vis.render(0, 120, rsim.GameMode.SOCCAR, pad_states, b_state, car_data)


# ===================================================================== #
#  Main Loop
# ===================================================================== #

def visualize(args):
    print("=" * 60)
    print(f"  MIXED PPO VISUALIZATION")
    print("=" * 60)

    # 1. Setup Policies
    policy_fn = _load_ppo_policy(args.checkpoint, args.device)
    opp_fn, opp_teacher = _make_opponent(args.opponent, args.opponent_checkpoint, args.device, args.tick_skip)

    # 2. Setup Mixed Environment
    mock_curriculum = MockCurriculum(args.difficulty, args.noise)
    
    kuxir_setter = MechanicTrajectorySetter(
        data_dir="../extracted_mechanics",
        mechanic_name="kuxir",
        fps=30,
        pre_mechanic_seconds=1.5,
        curriculum=mock_curriculum
    )
    normal_setter = KickoffMutator()
    mixed_setter = MixedStateSetter(
        setters=[normal_setter, kuxir_setter],
        probabilities=[0.0, 1.0], # FORCE KUXIR SPAWNS 100% OF THE TIME for debugging
        names=["normal", "kuxir"],
    )

    base_reward = build_mixed_reward(mechanic_name="kuxir")
    reward_fn = CurriculumRewardWrapper(base_reward, "kuxir", mock_curriculum)

    env = RLGym(
        state_mutator=MutatorSequence(
            FixedTeamSizeMutator(blue_size=1, orange_size=1),
            mixed_setter
        ),
        obs_builder=DefaultObs(
            zero_padding=None,
            pos_coef=np.asarray([1/common_values.SIDE_WALL_X, 1/common_values.BACK_NET_Y, 1/common_values.CEILING_Z], dtype=np.float32),
            ang_coef=1/np.pi,
            lin_vel_coef=1/common_values.CAR_MAX_SPEED,
            ang_vel_coef=1/common_values.CAR_MAX_ANG_VEL,
            boost_coef=1/100.0,
        ),
        action_parser=RepeatAction(LookupTableAction(), repeats=args.tick_skip),
        reward_fn=reward_fn, 
        termination_cond=GoalCondition(),
        truncation_cond=AnyCondition(TimeoutCondition(timeout_seconds=args.episode_seconds)),
        transition_engine=RocketSimEngine(),
    )

    # 3. RLViser
    vis = arena = rsim = None
    if args.viser:
        vis, arena, rsim = _setup_rlviser(env)

    step_dt = args.tick_skip / 120.0 / args.speed

    # 4. Episode Loop
    for ep in range(1, args.episodes + 1):
        obs_dict = env.reset()
        game_state = env.state

        if opp_teacher is not None:
            opp_teacher.reset(game_state)
            opp_teacher.reset_scores()

        blue_agent = orange_agent = None
        for agent_id, car in game_state.cars.items():
            if car.team_num == 0: blue_agent = agent_id
            else: orange_agent = agent_id

        # Determine what type of spawn we just got
        current_setter = mixed_setter.last_setter_name or "normal"
        print(f"\n--- Episode {ep}/{args.episodes} | Spawn: [{current_setter.upper()}] ---")

        ep_steps = 0
        done = False
        
        while not done:
            frame_start = time.time()

            # Blue (PPO) Action
            action_idx = policy_fn(obs_dict, game_state, blue_agent)

            # Orange (Opponent) Action
            if current_setter != "normal":
                opp_action = 0  # Force idle during mechanic setups
            else:
                # SELF-PLAY: Use the exact same PPO policy for the Orange car
                opp_action = policy_fn(obs_dict, game_state, orange_agent)

            actions = {
                blue_agent: np.array([action_idx]),
                orange_agent: np.array([opp_action]),
            }
            obs_dict, _, terminated_dict, truncated_dict = env.step(actions)
            game_state = env.state
            ep_steps += 1

            if opp_teacher is not None:
                opp_teacher.update_score(game_state)

            if args.viser:
                _render_frame(vis, arena, rsim)
                elapsed = time.time() - frame_start
                sleep_time = step_dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

            for agent_id in terminated_dict:
                if terminated_dict[agent_id] or truncated_dict[agent_id]:
                    done = True
                    break
        
        # End of episode stats
        ball_speed = np.linalg.norm(game_state.ball.linear_velocity)
        ball_y = game_state.ball.position[1]
        scored = any(terminated_dict.values())
        scorer = "BLUE" if (scored and ball_y > 0) else "ORANGE" if scored else "NOBODY"

        print(f"Result: {scorer} scored | Length: {ep_steps/(120/args.tick_skip):.1f}s | Final Ball Speed: {ball_speed:.0f} uu/s")

    if vis is not None:
        vis.quit()
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="none", help="Path to PPO .zip")
    parser.add_argument("--opponent", type=str, default="student_bc", choices=["idle", "teacher_nexto", "student_bc"])
    parser.add_argument("--opponent_checkpoint", type=str, default=None)
    parser.add_argument("--difficulty", type=float, default=0.2)
    parser.add_argument("--noise", type=float, default=0.05)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--viser", action="store_true")
    parser.add_argument("--tick_skip", type=int, default=8)
    parser.add_argument("--episode_seconds", type=float, default=15.0)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cpu")
    
    args = parser.parse_args()
    visualize(args)