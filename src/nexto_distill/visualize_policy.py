# src/nexto_distill/visualize_policy.py
"""
Watch policies play 1v1 in RLViser 3D.

Supports: Teacher Nexto, Student BC, Random, and PPO checkpoint (stub).

Usage:
    cd src
    python -m nexto_distill.visualize_policy --policy student_bc \
        --checkpoint ../checkpoints/nexto_distill/student_policy.pt \
        --opponent lazy --episodes 5 --viser
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Optional

import numpy as np
import torch

from nexto_distill.generate_dataset import (
    build_distill_env,
    LazyChaserOpponent,
    FreeplayMutator,
)
from nexto_distill.eval_imitation import _load_student
from nexto_distill.teacher_nexto import NextoTeacher
from rlgym.rocket_league.action_parsers import LookupTableAction, RepeatAction
from rlgym.rocket_league.state_mutators import (
    MutatorSequence,
    FixedTeamSizeMutator,
    KickoffMutator,
)


# ===================================================================== #
#  Policy loaders
# ===================================================================== #

def _load_policy(policy_name: str, checkpoint: Optional[str], device: str,
                 tick_skip: int):
    """
    Return a callable  policy_fn(obs_dict, game_state, blue_agent) -> int
    and a display name string.
    """
    if policy_name == "teacher_nexto":
        teacher = NextoTeacher(device=device, tick_skip=tick_skip)
        print(f"  Loaded Teacher Nexto ({teacher.num_actions} actions)")

        def policy_fn(obs_dict, game_state, blue_agent):
            return teacher.act(game_state, player_index=0)

        return policy_fn, "Teacher Nexto", teacher

    elif policy_name == "student_bc":
        if not checkpoint:
            raise ValueError("--checkpoint is required for --policy student_bc")
        if not os.path.isfile(checkpoint):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        dev = torch.device(device)
        model, meta = _load_student(checkpoint, device)
        print(f"  Loaded Student BC: layers={meta['layer_sizes']}, "
              f"params={meta['total_params']:,}")

        def policy_fn(obs_dict, game_state, blue_agent):
            obs = obs_dict[blue_agent]
            with torch.no_grad():
                t = torch.from_numpy(obs).float().unsqueeze(0).to(dev)
                logits = model(t)
                return int(logits.argmax(dim=-1).item())

        return policy_fn, "Student BC", None

    elif policy_name == "random":
        num_actions = len(LookupTableAction.make_lookup_table())
        rng = np.random.RandomState(42)
        print(f"  Random policy ({num_actions} actions)")

        def policy_fn(obs_dict, game_state, blue_agent):
            return int(rng.randint(0, num_actions))

        return policy_fn, "Random", None

    elif policy_name == "ppo_checkpoint":
        raise NotImplementedError(
            "PPO checkpoint loading is not yet implemented. "
            "Once the rlgym-ppo actor format is determined, add loading "
            "logic here. Expected: --checkpoint <path_to_ppo_actor.pt>"
        )

    else:
        raise ValueError(f"Unknown policy: {policy_name}")


# ===================================================================== #
#  Opponent factory
# ===================================================================== #

def _make_opponent(opponent_name: str, seed: int, device: str = "cpu",
                   tick_skip: int = 8, checkpoint: Optional[str] = None):
    """
    Return (opp_fn, opp_teacher_or_None).

    opp_fn signature:  (obs_dict, game_state, orange_agent) -> int
    opp_teacher is set only when opponent is teacher_nexto (needs reset).
    """
    if opponent_name == "lazy":
        opp = LazyChaserOpponent(seed=seed)
        print("  Opponent: LazyChaserOpponent")
        return (lambda obs_dict, game_state, agent_id: opp.act()), None

    elif opponent_name == "idle":
        print("  Opponent: Idle (action 0)")
        return (lambda obs_dict, game_state, agent_id: 0), None

    elif opponent_name == "teacher_nexto":
        opp_teacher = NextoTeacher(device=device, tick_skip=tick_skip)
        print(f"  Opponent: Teacher Nexto ({opp_teacher.num_actions} actions)")

        def opp_fn(obs_dict, game_state, agent_id):
            return opp_teacher.act(game_state, player_index=1)

        return opp_fn, opp_teacher

    elif opponent_name == "student_bc":
        if not checkpoint:
            raise ValueError(
                "--opponent_checkpoint is required for --opponent student_bc"
            )
        if not os.path.isfile(checkpoint):
            raise FileNotFoundError(f"Opponent checkpoint not found: {checkpoint}")
        dev = torch.device(device)
        opp_model, opp_meta = _load_student(checkpoint, device)
        print(f"  Opponent: Student BC: layers={opp_meta['layer_sizes']}, "
              f"params={opp_meta['total_params']:,}")

        def opp_fn(obs_dict, game_state, agent_id):
            obs = obs_dict[agent_id]
            with torch.no_grad():
                t = torch.from_numpy(obs).float().unsqueeze(0).to(dev)
                logits = opp_model(t)
                return int(logits.argmax(dim=-1).item())

        return opp_fn, None

    elif opponent_name == "random":
        num_actions = len(LookupTableAction.make_lookup_table())
        rng = np.random.RandomState(seed)
        print(f"  Opponent: Random ({num_actions} actions)")
        return (lambda obs_dict, game_state, agent_id: int(rng.randint(0, num_actions))), None

    else:
        raise ValueError(f"Unknown opponent: {opponent_name}")


# ===================================================================== #
#  RLViser setup
# ===================================================================== #

def _setup_rlviser(env):
    """Initialize RLViser and return (vis, arena, rsim) or raise ImportError."""
    import rlviser_py as vis
    import RocketSim as rsim

    engine = env.transition_engine
    arena = engine._arena

    vis.set_boost_pad_locations(
        [pad.get_pos().as_tuple() for pad in arena.get_boost_pads()]
    )
    time.sleep(2.0)
    print("  RLViser initialized")
    return vis, arena, rsim


def _render_frame(vis, arena, rsim):
    """Render one frame to RLViser."""
    pad_states = [p.get_state().is_active for p in arena.get_boost_pads()]
    b_state = arena.ball.get_state()
    try:
        car_data = [
            (c.id, c.team, c.get_config(), c.get_state())
            for c in arena.get_cars()
        ]
    except Exception:
        car_data = []
    vis.render(0, 120, rsim.GameMode.SOCCAR, pad_states, b_state, car_data)


# ===================================================================== #
#  Main visualization loop
# ===================================================================== #

def _build_env(tick_skip: int, episode_seconds: float, seed: int,
               kickoff: bool = False):
    """Build env with either FreeplayMutator or KickoffMutator."""
    if not kickoff:
        return build_distill_env(
            tick_skip=tick_skip,
            episode_seconds=episode_seconds,
            freeplay_seed=seed,
        )

    # Kickoff mode: use KickoffMutator for proper game spawns
    from rlgym.api import RLGym
    from rlgym.rocket_league.sim import RocketSimEngine
    from rlgym.rocket_league.obs_builders import DefaultObs
    from rlgym.rocket_league.done_conditions import (
        GoalCondition, TimeoutCondition, AnyCondition,
    )
    from rlgym.rocket_league.reward_functions import CombinedReward
    from rlgym.rocket_league import common_values

    return RLGym(
        state_mutator=MutatorSequence(
            FixedTeamSizeMutator(blue_size=1, orange_size=1),
            KickoffMutator(),
        ),
        obs_builder=DefaultObs(
            zero_padding=None,
            pos_coef=np.asarray([
                1 / common_values.SIDE_WALL_X,
                1 / common_values.BACK_NET_Y,
                1 / common_values.CEILING_Z,
            ], dtype=np.float32),
            ang_coef=1 / np.pi,
            lin_vel_coef=1 / common_values.CAR_MAX_SPEED,
            ang_vel_coef=1 / common_values.CAR_MAX_ANG_VEL,
            boost_coef=1 / 100.0,
        ),
        action_parser=RepeatAction(LookupTableAction(), repeats=tick_skip),
        reward_fn=CombinedReward(),
        termination_cond=GoalCondition(),
        truncation_cond=AnyCondition(
            TimeoutCondition(timeout_seconds=episode_seconds),
        ),
        transition_engine=RocketSimEngine(),
    )


def visualize(
    policy_name: str,
    checkpoint: Optional[str],
    opponent_name: str,
    opponent_checkpoint: Optional[str],
    episodes: int,
    use_viser: bool,
    tick_skip: int,
    episode_seconds: float,
    seed: int,
    device: str,
    speed: float,
    kickoff: bool = False,
):
    print("=" * 60)
    print("  POLICY VISUALIZATION")
    print("=" * 60)
    if kickoff:
        print("  Mode: Kickoff (proper game spawns)")
    else:
        print("  Mode: Freeplay (random spawns)")

    # Load policy
    policy_fn, policy_label, teacher = _load_policy(
        policy_name, checkpoint, device, tick_skip
    )

    # Build opponent
    opp_fn, opp_teacher = _make_opponent(
        opponent_name, seed + 1, device, tick_skip, opponent_checkpoint
    )

    # Build env
    env = _build_env(tick_skip, episode_seconds, seed, kickoff)

    # RLViser (optional)
    vis = arena = rsim = None
    if use_viser:
        try:
            vis, arena, rsim = _setup_rlviser(env)
        except ImportError:
            print("  WARNING: rlviser_py not installed — running headless")
            use_viser = False

    # Time per decision step for real-time playback
    step_dt = tick_skip / 120.0 / speed

    # ── Episode loop ──
    all_results = []
    for ep in range(1, episodes + 1):
        obs_dict = env.reset()
        game_state = env.state

        # Reset teacher obs builder if using teacher policy
        if teacher is not None:
            teacher.reset(game_state)
            teacher.reset_scores()
        if opp_teacher is not None:
            opp_teacher.reset(game_state)
            opp_teacher.reset_scores()

        # Identify agents
        blue_agent = orange_agent = None
        for agent_id in sorted(game_state.cars.keys()):
            car = game_state.cars[agent_id]
            if car.team_num == 0:
                blue_agent = agent_id
            else:
                orange_agent = agent_id

        # Per-episode metrics
        ep_steps = 0
        ep_touches = 0
        ball_speed_sum = 0.0
        dist_to_ball_sum = 0.0
        goals_blue = 0
        goals_orange = 0
        scored = False
        ep_start = time.time()

        done = False
        while not done:
            frame_start = time.time()

            # Blue action
            action_idx = policy_fn(obs_dict, game_state, blue_agent)

            # Orange action
            opp_action = opp_fn(obs_dict, game_state, orange_agent)

            # Step
            actions = {
                blue_agent: np.array([action_idx]),
                orange_agent: np.array([opp_action]),
            }
            obs_dict, reward_dict, terminated_dict, truncated_dict = env.step(actions)
            game_state = env.state

            # Update teacher scores if applicable
            if teacher is not None:
                teacher.update_score(game_state)
            if opp_teacher is not None:
                opp_teacher.update_score(game_state)

            ep_steps += 1

            # Collect metrics
            ball_vel = game_state.ball.linear_velocity
            ball_speed = float(np.linalg.norm(ball_vel))
            ball_speed_sum += ball_speed

            blue_car = game_state.cars[blue_agent]
            dist = float(np.linalg.norm(
                blue_car.physics.position - game_state.ball.position
            ))
            dist_to_ball_sum += dist

            if blue_car.ball_touches > 0:
                ep_touches += blue_car.ball_touches

            # RLViser rendering
            if use_viser:
                _render_frame(vis, arena, rsim)
                elapsed = time.time() - frame_start
                sleep_time = step_dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

            # Check done
            for agent_id in terminated_dict:
                if terminated_dict[agent_id] or truncated_dict[agent_id]:
                    done = True
                    if terminated_dict[agent_id]:
                        scored = True
                    break

        # Determine who scored (ball y position heuristic)
        if scored:
            ball_y = game_state.ball.position[1]
            if ball_y > 0:
                goals_blue += 1
            else:
                goals_orange += 1

        ep_elapsed = time.time() - ep_start
        avg_ball_speed = ball_speed_sum / max(ep_steps, 1)
        avg_dist = dist_to_ball_sum / max(ep_steps, 1)
        sim_time = ep_steps * tick_skip / 120.0

        result = {
            "episode": ep,
            "steps": ep_steps,
            "sim_time": sim_time,
            "wall_time": ep_elapsed,
            "goals_blue": goals_blue,
            "goals_orange": goals_orange,
            "touches": ep_touches,
            "avg_ball_speed": avg_ball_speed,
            "avg_dist_to_ball": avg_dist,
        }
        all_results.append(result)

        print(
            f"Episode {ep}/{episodes}: "
            f"steps={ep_steps} | "
            f"time={sim_time:.1f}s | "
            f"goals_blue={goals_blue} goals_orange={goals_orange} | "
            f"touches={ep_touches} | "
            f"avg_ball_speed={avg_ball_speed:.0f} uu/s | "
            f"avg_dist_to_ball={avg_dist:.0f} uu"
        )

    # ── Summary ──
    total_eps = len(all_results)
    total_steps = sum(r["steps"] for r in all_results)
    total_blue = sum(r["goals_blue"] for r in all_results)
    total_orange = sum(r["goals_orange"] for r in all_results)
    total_touches = sum(r["touches"] for r in all_results)
    avg_speed = sum(r["avg_ball_speed"] for r in all_results) / max(total_eps, 1)
    avg_dist = sum(r["avg_dist_to_ball"] for r in all_results) / max(total_eps, 1)
    draws = sum(1 for r in all_results if r["goals_blue"] == 0 and r["goals_orange"] == 0)

    print(f"\n{'=' * 60}")
    print(f"  SUMMARY — {policy_label} vs {opponent_name} ({total_eps} episodes)")
    print(f"    Blue Wins:            {total_blue}  ({total_blue/max(total_eps,1):.0%})")
    print(f"    Orange Wins:          {total_orange}  ({total_orange/max(total_eps,1):.0%})")
    print(f"    Draws (timeout):      {draws}  ({draws/max(total_eps,1):.0%})")
    print(f"    Total Steps:          {total_steps:,}")
    print(f"    Total Touches:        {total_touches}")
    print(f"    Avg Ball Speed:       {avg_speed:.0f} uu/s")
    print(f"    Avg Dist to Ball:     {avg_dist:.0f} uu")
    print(f"    Avg Steps/Episode:    {total_steps / max(total_eps, 1):.0f}")
    print(f"{'=' * 60}")

    if vis is not None:
        vis.quit()

    env.close()


# ===================================================================== #
#  CLI
# ===================================================================== #

def main():
    parser = argparse.ArgumentParser(
        description="Watch policies play 1v1 in RLViser 3D.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Random policy, headless
  python -m nexto_distill.visualize_policy --policy random --episodes 3

  # Student BC with RLViser
  python -m nexto_distill.visualize_policy --policy student_bc \\
      --checkpoint ../checkpoints/nexto_distill/student_policy.pt \\
      --viser --episodes 5

  # Teacher Nexto, slow-motion
  python -m nexto_distill.visualize_policy --policy teacher_nexto \\
      --viser --speed 0.5 --episodes 3

   # Student vs lazy chaser
  python -m nexto_distill.visualize_policy --policy student_bc \\
      --checkpoint ../checkpoints/nexto_distill/student_policy.pt \\
      --opponent lazy --viser --episodes 5

  # Student vs Teacher Nexto
  python -m nexto_distill.visualize_policy --policy student_bc \\
      --checkpoint ../checkpoints/nexto_distill/student_policy.pt \\
      --opponent teacher_nexto --viser --episodes 5

  # Student vs Student (self-play)
  python -m nexto_distill.visualize_policy --policy student_bc \\
      --checkpoint ../checkpoints/nexto_distill/student_policy.pt \\
      --opponent student_bc --opponent_checkpoint ../checkpoints/nexto_distill/student_policy.pt \\
      --viser --episodes 5
""",
    )
    parser.add_argument(
        "--policy", type=str, required=True,
        choices=["teacher_nexto", "student_bc", "random", "ppo_checkpoint"],
        help="Policy to control blue car",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to .pt checkpoint (required for student_bc, ppo_checkpoint)",
    )
    parser.add_argument(
        "--opponent", type=str, default="lazy",
        choices=["lazy", "idle", "teacher_nexto", "student_bc", "random"],
        help="Orange car behavior (default: lazy)",
    )
    parser.add_argument(
        "--opponent_checkpoint", type=str, default=None,
        help="Path to .pt checkpoint for --opponent student_bc",
    )
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument(
        "--viser", action="store_true",
        help="Launch RLViser 3D visualization",
    )
    parser.add_argument("--tick_skip", type=int, default=8)
    parser.add_argument("--episode_seconds", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", type=str, default="cpu", choices=["cpu", "cuda"],
    )
    parser.add_argument(
        "--speed", type=float, default=1.0,
        help="Playback speed multiplier (0.5 = slow-mo, 2.0 = fast)",
    )
    parser.add_argument(
        "--kickoff", action="store_true",
        help="Use kickoff spawns instead of random freeplay spawns",
    )

    args = parser.parse_args()

    visualize(
        policy_name=args.policy,
        checkpoint=args.checkpoint,
        opponent_name=args.opponent,
        opponent_checkpoint=args.opponent_checkpoint,
        episodes=args.episodes,
        use_viser=args.viser,
        tick_skip=args.tick_skip,
        episode_seconds=args.episode_seconds,
        seed=args.seed,
        device=args.device,
        speed=args.speed,
        kickoff=args.kickoff,
    )


if __name__ == "__main__":
    main()
