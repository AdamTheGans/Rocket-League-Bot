# src/nexto_distill/dagger_collect.py
"""
DAgger data collection: roll out the STUDENT policy in RocketSim,
but label each state with the TEACHER's action and logits.

This produces on-policy (student distribution) training data that
addresses the covariate shift problem in standard behavior cloning.

Usage:
    cd src
    python -m nexto_distill.dagger_collect \
        --checkpoint ../checkpoints/nexto_distill/student_policy.pt \
        --out_dir ../data/nexto_distill/dagger_shards \
        --num_steps 500000 \
        --shard_size 50000
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter
from typing import Optional

import numpy as np
import torch

from rlgym.rocket_league.action_parsers import LookupTableAction

from nexto_distill.student_policy import StudentPolicy
from nexto_distill.teacher_nexto import NextoTeacher
from nexto_distill.generate_dataset import (
    build_distill_env,
    LazyChaserOpponent,
    ShardWriter,
    _print_action_report,
)


def _load_student(checkpoint_path: str, device: str = "cpu"):
    ckpt_dir = os.path.dirname(checkpoint_path)
    meta_path = os.path.join(ckpt_dir, "metadata.json")
    with open(meta_path) as f:
        meta = json.load(f)
    model = StudentPolicy(
        obs_dim=meta["obs_dim"],
        num_actions=meta["num_actions"],
        layer_sizes=meta["layer_sizes"],
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model, meta


def dagger_collect(
    checkpoint_path: str,
    out_dir: str,
    num_steps: int = 500_000,
    shard_size: int = 50_000,
    seed: int = 42,
    tick_skip: int = 8,
    episode_seconds: float = 30.0,
    model_path: Optional[str] = None,
    report_every: int = 10_000,
):
    print("=" * 60)
    print("  DAgger COLLECTION — Student drives, Teacher labels")
    print("=" * 60)
    print(f"  Checkpoint:      {checkpoint_path}")
    print(f"  Output dir:      {out_dir}")
    print(f"  Num steps:       {num_steps:,}")
    print(f"  Shard size:      {shard_size:,}")
    print()

    # Load student
    model, meta = _load_student(checkpoint_path)
    print(f"  Student: {meta['layer_sizes']}, {meta['total_params']:,} params")

    # Load teacher
    teacher_kwargs = {"tick_skip": tick_skip}
    if model_path:
        teacher_kwargs["model_path"] = model_path
    teacher = NextoTeacher(**teacher_kwargs)
    print(f"  Teacher: {teacher.num_actions} actions")

    # Build env
    env = build_distill_env(
        tick_skip=tick_skip,
        episode_seconds=episode_seconds,
        freeplay_seed=seed + 200,  # Different seed from original data
    )

    # Opponent
    opponent = LazyChaserOpponent(seed=seed + 201)

    # Writer
    writer = ShardWriter(out_dir, shard_size)
    lut = LookupTableAction.make_lookup_table()
    action_counts: Counter = Counter()
    student_action_counts: Counter = Counter()
    lut_hash = hashlib.sha256(lut.tobytes()).hexdigest()[:16]

    # Stats
    episode_id = 0
    timestep_in_episode = 0
    total_steps = 0
    total_episodes = 0
    total_agree = 0
    t0 = time.time()

    # Initial reset
    obs_dict = env.reset()
    game_state = env.state
    teacher.reset_scores()
    teacher.reset(game_state)

    blue_agent = orange_agent = None
    for aid in sorted(game_state.cars.keys()):
        car = game_state.cars[aid]
        if car.team_num == 0:
            blue_agent = aid
        else:
            orange_agent = aid

    print(f"  Agents: blue={blue_agent}, orange={orange_agent}")
    print()

    while total_steps < num_steps:
        student_obs = obs_dict[blue_agent]

        # Student chooses action (this is who drives)
        with torch.no_grad():
            logits = model(torch.from_numpy(student_obs).float().unsqueeze(0))
            student_action = int(logits.argmax(dim=-1).item())

        # Teacher labels the same state
        teacher_action = teacher.act(game_state, player_index=0)
        teacher_logits = teacher.get_logits(game_state, player_index=0)

        # Track agreement
        if student_action == teacher_action:
            total_agree += 1

        # Record: student obs + TEACHER labels (this is the DAgger data)
        writer.add(
            obs=student_obs,
            action=teacher_action,      # Teacher's action as label
            logits=teacher_logits,       # Teacher's logits
            episode_id=episode_id,
            timestep=timestep_in_episode,
        )
        action_counts[teacher_action] += 1
        student_action_counts[student_action] += 1

        # Student drives the car
        opp_action = opponent.act()
        actions = {
            blue_agent: np.array([student_action]),
            orange_agent: np.array([opp_action]),
        }

        obs_dict, _, term, trunc = env.step(actions)
        game_state = env.state
        teacher.update_score(game_state)

        total_steps += 1
        timestep_in_episode += 1

        done = False
        for aid in term:
            if term[aid] or trunc[aid]:
                done = True
                break

        if done:
            total_episodes += 1
            episode_id += 1
            timestep_in_episode = 0
            obs_dict = env.reset()
            game_state = env.state
            teacher.reset(game_state)
            for aid in sorted(game_state.cars.keys()):
                car = game_state.cars[aid]
                if car.team_num == 0:
                    blue_agent = aid
                else:
                    orange_agent = aid

        if total_steps % report_every == 0:
            elapsed = time.time() - t0
            sps = total_steps / elapsed if elapsed > 0 else 0
            agree_pct = total_agree / total_steps
            print(f"[{total_steps:>10,} / {num_steps:,}] "
                  f"eps: {total_episodes} | "
                  f"agree: {agree_pct:.1%} | "
                  f"speed: {sps:,.0f} steps/s | "
                  f"elapsed: {elapsed:.1f}s")

    writer.flush()
    elapsed = time.time() - t0

    print(f"\n  DAgger collection agreement: {total_agree/total_steps:.2%}")
    print(f"\n  Teacher labels on student states:")
    _print_action_report(action_counts, total_steps, lut)

    # Save metadata
    meta_out = {
        "type": "dagger",
        "total_steps": total_steps,
        "total_episodes": total_episodes,
        "total_shards": writer.total_shards,
        "shard_size": shard_size,
        "seed": seed,
        "tick_skip": tick_skip,
        "episode_seconds": episode_seconds,
        "lut_hash": lut_hash,
        "num_actions": len(lut),
        "student_obs_dim": obs_dict[blue_agent].shape[0],
        "online_agreement": total_agree / total_steps,
        "elapsed_seconds": elapsed,
        "source_checkpoint": checkpoint_path,
    }
    meta_path = os.path.join(out_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta_out, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"  DONE — {total_steps:,} steps, {total_episodes} episodes")
    print(f"  {writer.total_shards} shards in {out_dir}")
    print(f"{'=' * 60}")

    env.close()


def main():
    parser = argparse.ArgumentParser(description="DAgger data collection.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--out_dir", type=str,
                        default=os.path.join("..", "data", "nexto_distill", "dagger_shards"))
    parser.add_argument("--num_steps", type=int, default=500_000)
    parser.add_argument("--shard_size", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tick_skip", type=int, default=8)
    parser.add_argument("--episode_seconds", type=float, default=30.0)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--report_every", type=int, default=10_000)
    args = parser.parse_args()

    dagger_collect(
        checkpoint_path=args.checkpoint,
        out_dir=args.out_dir,
        num_steps=args.num_steps,
        shard_size=args.shard_size,
        seed=args.seed,
        tick_skip=args.tick_skip,
        episode_seconds=args.episode_seconds,
        model_path=args.model_path,
        report_every=args.report_every,
    )


if __name__ == "__main__":
    main()
