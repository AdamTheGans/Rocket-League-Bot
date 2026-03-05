# src/nexto_distill/generate_dataset.py
"""
Generate (student_obs, teacher_action, teacher_logits) datasets by
rolling out Nexto as the teacher in a RocketSim 1v1 environment.

Usage:
    cd src
    python -m nexto_distill.generate_dataset \
        --out_dir ../data/nexto_distill/shards \
        --num_steps 1000000 \
        --shard_size 50000 \
        --seed 42
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
from typing import Optional

import numpy as np
import torch

# ── rlgym v2 imports ──
from rlgym.api import RLGym
from rlgym.rocket_league.sim import RocketSimEngine
from rlgym.rocket_league.action_parsers import LookupTableAction, RepeatAction
from rlgym.rocket_league.done_conditions import (
    GoalCondition,
    TimeoutCondition,
    AnyCondition,
)
from rlgym.rocket_league.obs_builders import DefaultObs
from rlgym.rocket_league.state_mutators import (
    MutatorSequence,
    FixedTeamSizeMutator,
    KickoffMutator,
)
from rlgym.rocket_league import common_values

from nexto_distill.teacher_nexto import NextoTeacher


# ===================================================================== #
#  Randomized freeplay state mutator
# ===================================================================== #
from rlgym.api import StateMutator
from rlgym.rocket_league.api import GameState as V2GameState


class FreeplayMutator(StateMutator):
    """
    Randomize car + ball positions for diverse training states.
    Applied after FixedTeamSizeMutator sets teams.
    """

    def __init__(self, seed: Optional[int] = None):
        super().__init__()
        self._rng = np.random.RandomState(seed)

    def apply(self, state: V2GameState, shared_info: dict) -> None:
        # Random ball position on the field
        bx = self._rng.uniform(-3500, 3500)
        by = self._rng.uniform(-4500, 4500)
        bz = self._rng.uniform(93, 600)
        state.ball.position = np.array([bx, by, bz], dtype=np.float32)

        # Random ball velocity
        bvx = self._rng.uniform(-1500, 1500)
        bvy = self._rng.uniform(-1500, 1500)
        bvz = self._rng.uniform(-300, 300)
        state.ball.linear_velocity = np.array([bvx, bvy, bvz], dtype=np.float32)
        state.ball.angular_velocity = np.zeros(3, dtype=np.float32)

        for agent_id, car in state.cars.items():
            # Random car position
            cx = self._rng.uniform(-3800, 3800)
            cy = self._rng.uniform(-4800, 4800)
            cz = 17.01  # on ground
            car.physics.position = np.array([cx, cy, cz], dtype=np.float32)

            # Random yaw
            yaw = self._rng.uniform(-np.pi, np.pi)
            car.physics.euler_angles = np.array([0.0, yaw, 0.0], dtype=np.float32)

            # Small random velocity
            cvx = self._rng.uniform(-500, 500)
            cvy = self._rng.uniform(-500, 500)
            car.physics.linear_velocity = np.array([cvx, cvy, 0.0], dtype=np.float32)
            car.physics.angular_velocity = np.zeros(3, dtype=np.float32)

            # Random boost
            car.boost_amount = self._rng.uniform(0, 100)

            # Reset flip/jump state
            car.has_jumped = False
            car.has_double_jumped = False
            car.has_flipped = False
            car.is_jumping = False
            car.is_holding_jump = False


# ===================================================================== #
#  "Lazy chaser" opponent action provider
# ===================================================================== #

class LazyChaserOpponent:
    """
    Simple opponent that biases toward ball-chasing with some randomness.
    Provides valid action indices from the 90-action LUT.

    Behavior: 80% chance to drive forward (with or without boost),
    occasionally steers toward ball (crude), 20% random action.
    """

    def __init__(self, seed: Optional[int] = None):
        self._rng = np.random.RandomState(seed)
        self._lut = LookupTableAction.make_lookup_table()

        # Pre-compute useful action indices
        # Ground actions where throttle=1, steer=0, boost=0, handbrake=0
        self._forward_actions = []
        self._forward_boost_actions = []
        for i, a in enumerate(self._lut):
            if a[0] == 1 and a[1] == 0 and a[5] == 0 and a[7] == 0:
                if a[6] == 0:
                    self._forward_actions.append(i)
                else:
                    self._forward_boost_actions.append(i)

        self._all_indices = list(range(len(self._lut)))

    def act(self) -> int:
        """Return an action index."""
        r = self._rng.random()
        if r < 0.4:
            # Drive forward
            return self._rng.choice(self._forward_actions)
        elif r < 0.6:
            # Drive forward with boost
            return self._rng.choice(self._forward_boost_actions)
        else:
            # Random action
            return self._rng.choice(self._all_indices)


# ===================================================================== #
#  Environment builder
# ===================================================================== #

def build_distill_env(
    tick_skip: int = 8,
    episode_seconds: float = 30.0,
    freeplay_seed: Optional[int] = None,
):
    """
    Build a 1v1 RLGym v2 + RocketSim env for distillation data collection.

    Returns the raw RLGym env (NOT wrapped in RLGymV2GymWrapper)
    because we need direct access to game states.
    """
    action_parser = RepeatAction(LookupTableAction(), repeats=tick_skip)

    termination_condition = GoalCondition()
    truncation_condition = AnyCondition(
        TimeoutCondition(timeout_seconds=episode_seconds),
    )

    # Student obs: DefaultObs (same format as PPO training)
    obs_builder = DefaultObs(
        zero_padding=None,
        pos_coef=np.asarray(
            [
                1 / common_values.SIDE_WALL_X,
                1 / common_values.BACK_NET_Y,
                1 / common_values.CEILING_Z,
            ],
            dtype=np.float32,
        ),
        ang_coef=1 / np.pi,
        lin_vel_coef=1 / common_values.CAR_MAX_SPEED,
        ang_vel_coef=1 / common_values.CAR_MAX_ANG_VEL,
        boost_coef=1 / 100.0,
    )

    state_mutator = MutatorSequence(
        FixedTeamSizeMutator(blue_size=1, orange_size=1),
        FreeplayMutator(seed=freeplay_seed),
    )

    # No reward needed — we're just collecting data
    from rlgym.rocket_league.reward_functions import CombinedReward
    reward_fn = CombinedReward()

    env = RLGym(
        state_mutator=state_mutator,
        obs_builder=obs_builder,
        action_parser=action_parser,
        reward_fn=reward_fn,
        termination_cond=termination_condition,
        truncation_cond=truncation_condition,
        transition_engine=RocketSimEngine(),
    )

    return env


# ===================================================================== #
#  Shard writer
# ===================================================================== #

class ShardWriter:
    """Buffers data and writes .npz shards when full."""

    def __init__(self, out_dir: str, shard_size: int):
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir
        self.shard_size = shard_size
        self._shard_idx = 0
        self._buffers = {
            "obs": [],
            "actions": [],
            "logits": [],
            "episode_ids": [],
            "timesteps": [],
        }
        self._count = 0

    def add(
        self,
        obs: np.ndarray,
        action: int,
        logits: np.ndarray,
        episode_id: int,
        timestep: int,
    ):
        self._buffers["obs"].append(obs)
        self._buffers["actions"].append(action)
        self._buffers["logits"].append(logits)
        self._buffers["episode_ids"].append(episode_id)
        self._buffers["timesteps"].append(timestep)
        self._count += 1

        if self._count >= self.shard_size:
            self.flush()

    def flush(self):
        if self._count == 0:
            return
        path = os.path.join(self.out_dir, f"shard_{self._shard_idx:05d}.npz")
        np.savez_compressed(
            path,
            obs=np.array(self._buffers["obs"], dtype=np.float32),
            actions=np.array(self._buffers["actions"], dtype=np.int64),
            logits=np.array(self._buffers["logits"], dtype=np.float32),
            episode_ids=np.array(self._buffers["episode_ids"], dtype=np.int64),
            timesteps=np.array(self._buffers["timesteps"], dtype=np.int64),
        )
        print(f"  Shard {self._shard_idx:05d} saved ({self._count} steps) → {path}")
        self._shard_idx += 1
        for k in self._buffers:
            self._buffers[k] = []
        self._count = 0

    @property
    def total_shards(self) -> int:
        return self._shard_idx


# ===================================================================== #
#  Action distribution sanity report
# ===================================================================== #

def _print_action_report(action_counts: Counter, total: int, lut: np.ndarray):
    """Print top actions and entropy to catch bad env setups."""
    print("\n==== ACTION DISTRIBUTION REPORT ====")

    # Entropy
    probs = np.array([action_counts.get(i, 0) / total for i in range(len(lut))])
    probs = probs[probs > 0]
    entropy = -np.sum(probs * np.log(probs))
    max_entropy = np.log(len(lut))
    print(f"  Entropy: {entropy:.3f} / {max_entropy:.3f} "
          f"({entropy/max_entropy*100:.1f}% of max)")

    # Top 10 actions
    print(f"  Top 10 actions (of {len(lut)} total):")
    for action_idx, count in action_counts.most_common(10):
        pct = count / total * 100
        a = lut[action_idx]
        print(f"    [{action_idx:3d}] {pct:5.1f}%  "
              f"thr={a[0]:+.0f} str={a[1]:+.0f} "
              f"pitch={a[2]:+.0f} yaw={a[3]:+.0f} roll={a[4]:+.0f} "
              f"jmp={a[5]:.0f} bst={a[6]:.0f} hb={a[7]:.0f}")

    # Actions never used
    unused = sum(1 for i in range(len(lut)) if action_counts.get(i, 0) == 0)
    print(f"  Unused actions: {unused} / {len(lut)}")
    print("====================================\n")


# ===================================================================== #
#  Main generation loop
# ===================================================================== #

def generate_dataset(
    out_dir: str,
    num_steps: int,
    shard_size: int = 50_000,
    seed: int = 42,
    device: str = "cpu",
    tick_skip: int = 8,
    episode_seconds: float = 30.0,
    model_path: Optional[str] = None,
    report_every: int = 10_000,
):
    """
    Roll out Nexto teacher in RocketSim and collect
    (student_obs, teacher_action, teacher_logits) data.
    """
    print("=" * 60)
    print("  NEXTO DISTILLATION — DATASET GENERATION")
    print("=" * 60)
    print(f"  Output dir:      {out_dir}")
    print(f"  Num steps:       {num_steps:,}")
    print(f"  Shard size:      {shard_size:,}")
    print(f"  Seed:            {seed}")
    print(f"  Device:          {device}")
    print(f"  Tick skip:       {tick_skip}")
    print(f"  Episode seconds: {episode_seconds}")
    print()

    # Build env
    env = build_distill_env(
        tick_skip=tick_skip,
        episode_seconds=episode_seconds,
        freeplay_seed=seed,
    )

    # Build teacher
    teacher_kwargs = {"device": device, "tick_skip": tick_skip}
    if model_path:
        teacher_kwargs["model_path"] = model_path
    teacher = NextoTeacher(**teacher_kwargs)

    # Build opponent
    opponent = LazyChaserOpponent(seed=seed + 1)

    # Shard writer
    writer = ShardWriter(out_dir, shard_size)

    # LUT for reports
    lut = LookupTableAction.make_lookup_table()
    action_counts: Counter = Counter()

    # Compute LUT hash for metadata
    lut_hash = hashlib.sha256(lut.tobytes()).hexdigest()[:16]

    # ── Main loop ──
    episode_id = 0
    timestep_in_episode = 0
    total_steps = 0
    total_episodes = 0
    t0 = time.time()

    # Initial reset
    obs_dict = env.reset()
    game_state = env.state

    teacher.reset_scores()
    teacher.reset(game_state)

    # Identify blue and orange agents
    blue_agent = None
    orange_agent = None
    for agent_id in sorted(game_state.cars.keys()):
        car = game_state.cars[agent_id]
        if car.team_num == 0:
            blue_agent = agent_id
        else:
            orange_agent = agent_id

    if blue_agent is None or orange_agent is None:
        raise RuntimeError(
            f"Expected 1 blue + 1 orange car, got agents: {list(game_state.cars.keys())}"
        )

    print(f"Agents: blue={blue_agent}, orange={orange_agent}")
    print(f"Student obs dim: {obs_dict[blue_agent].shape[0]}")
    print()

    while total_steps < num_steps:
        # Get student obs for blue player
        student_obs = obs_dict[blue_agent]

        # Get teacher action + logits for blue player
        teacher_action = teacher.act(game_state, player_index=0)
        teacher_logits = teacher.get_logits(game_state, player_index=0)

        # Record
        writer.add(
            obs=student_obs,
            action=teacher_action,
            logits=teacher_logits,
            episode_id=episode_id,
            timestep=timestep_in_episode,
        )
        action_counts[teacher_action] += 1

        # Opponent action
        opp_action = opponent.act()

        # Build action dict for env.step()
        actions = {
            blue_agent: np.array([teacher_action]),
            orange_agent: np.array([opp_action]),
        }

        # Step env
        obs_dict, reward_dict, terminated_dict, truncated_dict = env.step(actions)
        game_state = env.state

        # Track scores from goal events
        teacher.update_score(game_state)

        total_steps += 1
        timestep_in_episode += 1

        # Check for episode end
        # In rlgym v2, terminated/truncated are dicts keyed by agent_id
        done = False
        for agent_id in terminated_dict:
            if terminated_dict[agent_id] or truncated_dict[agent_id]:
                done = True
                break

        if done:
            total_episodes += 1
            episode_id += 1
            timestep_in_episode = 0

            # Reset env
            obs_dict = env.reset()
            game_state = env.state
            teacher.reset(game_state)

            # Re-identify agents (shouldn't change, but be safe)
            for agent_id in sorted(game_state.cars.keys()):
                car = game_state.cars[agent_id]
                if car.team_num == 0:
                    blue_agent = agent_id
                else:
                    orange_agent = agent_id

        # Progress
        if total_steps % report_every == 0:
            elapsed = time.time() - t0
            sps = total_steps / elapsed if elapsed > 0 else 0
            print(
                f"[{total_steps:>10,} / {num_steps:,}] "
                f"episodes: {total_episodes} | "
                f"speed: {sps:,.0f} steps/s | "
                f"elapsed: {elapsed:.1f}s"
            )

    # Flush remaining data
    writer.flush()

    elapsed = time.time() - t0
    sps = total_steps / elapsed if elapsed > 0 else 0

    # Action distribution report
    _print_action_report(action_counts, total_steps, lut)

    # Save metadata
    meta = {
        "total_steps": total_steps,
        "total_episodes": total_episodes,
        "total_shards": writer.total_shards,
        "shard_size": shard_size,
        "seed": seed,
        "tick_skip": tick_skip,
        "episode_seconds": episode_seconds,
        "device": device,
        "lut_hash": lut_hash,
        "num_actions": len(lut),
        "student_obs_dim": obs_dict[blue_agent].shape[0],
        "elapsed_seconds": elapsed,
        "steps_per_second": sps,
    }
    meta_path = os.path.join(out_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata saved: {meta_path}")

    print(f"\n{'=' * 60}")
    print(f"  DONE — {total_steps:,} steps, {total_episodes} episodes")
    print(f"  {writer.total_shards} shards in {out_dir}")
    print(f"  {sps:,.0f} steps/s ({elapsed:.1f}s total)")
    print(f"{'=' * 60}")

    env.close()


# ===================================================================== #
#  CLI
# ===================================================================== #

def main():
    parser = argparse.ArgumentParser(
        description="Generate distillation dataset from Nexto teacher."
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=os.path.join("..", "data", "nexto_distill", "shards"),
        help="Output directory for shard files",
    )
    parser.add_argument("--num_steps", type=int, default=1_000_000)
    parser.add_argument("--shard_size", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--tick_skip", type=int, default=8)
    parser.add_argument("--episode_seconds", type=float, default=30.0)
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to nexto-model.pt (auto-detected if not given)")
    parser.add_argument("--report_every", type=int, default=10_000)

    args = parser.parse_args()

    generate_dataset(
        out_dir=args.out_dir,
        num_steps=args.num_steps,
        shard_size=args.shard_size,
        seed=args.seed,
        device=args.device,
        tick_skip=args.tick_skip,
        episode_seconds=args.episode_seconds,
        model_path=args.model_path,
        report_every=args.report_every,
    )


if __name__ == "__main__":
    main()
