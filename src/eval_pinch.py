# src/eval_pinch.py
"""
Evaluation script for the wall/corner pinch specialist.

Usage:
    python src/eval_pinch.py --stage 1
    python src/eval_pinch.py --stage 2 --episodes 100
"""
from __future__ import annotations

import argparse
import os
import json
import time
import collections
from dataclasses import dataclass
from typing import Any, Optional, List, Tuple

import numpy as np
import torch

from envs.pinch import build_env
from rlgym.rocket_league import common_values


# ── Reuse helpers from eval_specialist_1 ──
from eval_specialist_1 import (
    _load_obs_stats,
    _standardize_obs,
    _find_latest_policy_path,
    _build_policy_from_state_dict,
    _load_policy,
    _reset_env,
    _to_action_for_env,
    _step_env,
    _unwrap_bool,
    _unwrap_float,
    _get_state_for_debug,
    _save_topdown_gif_labeled,
    EpisodeResult,
)


def _summarize_pinch(results: List[EpisodeResult], spike_maxes: List[float]) -> dict:
    eps = len(results)
    goal_rate = sum(r.scored for r in results) / eps if eps else 0.0
    timeout_rate = sum(r.timeout for r in results) / eps if eps else 0.0
    ttg = np.array([r.seconds for r in results if r.scored], dtype=np.float32)
    returns = np.array([r.total_reward for r in results], dtype=np.float32)

    # Speed spike metrics
    spike_arr = np.array(spike_maxes, dtype=np.float32) if spike_maxes else np.array([], dtype=np.float32)
    spike_rate = float(np.mean(spike_arr > 1500)) if spike_arr.size else 0.0

    return {
        "episodes": eps,
        "goal_rate": goal_rate,
        "timeout_rate": timeout_rate,
        "median_time_to_goal": float(np.median(ttg)) if ttg.size else float("inf"),
        "mean_time_to_goal": float(np.mean(ttg)) if ttg.size else float("inf"),
        "avg_return": float(np.mean(returns)) if eps else float("nan"),
        "spike_rate_1500": spike_rate,
        "median_max_goalward_spike": float(np.median(spike_arr)) if spike_arr.size else 0.0,
        "mean_max_goalward_spike": float(np.mean(spike_arr)) if spike_arr.size else 0.0,
    }


def evaluate_pinch(
    checkpoints_root: str,
    stage: int = 1,
    n_episodes: int = 200,
    render: bool = False,
    tick_skip: int = 8,
    deterministic: bool = True,
    print_every: int = 25,
    record_gif_episodes: int = 2,
    gif_out_path: str = os.path.join("checkpoints", "pinch_eval.gif"),
):
    env = build_env(render=render, tick_skip=tick_skip, stage=stage)

    policy_path = _find_latest_policy_path(checkpoints_root)
    print(f"Loading policy: {policy_path}")
    policy = _load_policy(policy_path, env)

    obs_mean, obs_std = _load_obs_stats(policy_path)
    if obs_mean is not None:
        print(f"Loaded obs standardization stats (dim={obs_mean.shape[0]})")
    else:
        print("WARNING: No obs standardization stats found")

    sec_per_step = float(tick_skip) / 120.0

    results: List[EpisodeResult] = []
    spike_maxes: List[float] = []
    captured: List[List[dict]] = []

    obs = _reset_env(env)
    ep_states: List[dict] = []
    if len(captured) < record_gif_episodes:
        st = _get_state_for_debug(env)
        if st is not None:
            ep_states.append(st)

    ep_steps = 0
    ep_return = 0.0
    ep_max_gw_spike = 0.0
    prev_goalward = 0.0

    t0 = time.time()
    while len(results) < n_episodes:
        policy_obs = _standardize_obs(obs, obs_mean, obs_std) if obs_mean is not None else obs

        if hasattr(policy, "get_action"):
            with torch.no_grad():
                action, _ = policy.get_action(policy_obs, deterministic=deterministic)
        else:
            with torch.no_grad():
                logits = policy(torch.from_numpy(policy_obs).float().unsqueeze(0))
                action = int(torch.argmax(logits, dim=-1).item())

        action_arr = _to_action_for_env(action)
        obs, r, terminated, truncated, info = _step_env(env, action_arr)

        ep_steps += 1
        ep_return += r

        # Track goalward speed spike
        try:
            rlgym_env = getattr(env, "rlgym_env", None)
            if rlgym_env and hasattr(rlgym_env, "state"):
                state = rlgym_env.state
                ball_vel = np.asarray(state.ball.linear_velocity, dtype=np.float32)
                ball_pos = np.asarray(state.ball.position, dtype=np.float32)
                goal = np.array([0, common_values.BACK_NET_Y, 0], dtype=np.float32)
                diff = goal - ball_pos
                dist = float(np.linalg.norm(diff))
                if dist > 1e-6:
                    goal_dir = diff / dist
                else:
                    goal_dir = np.array([0, 1, 0], dtype=np.float32)
                curr_goalward = float(np.dot(ball_vel, goal_dir))
                delta = curr_goalward - prev_goalward
                if delta > 0 and curr_goalward > 0:
                    ep_max_gw_spike = max(ep_max_gw_spike, delta)
                prev_goalward = curr_goalward
        except Exception:
            pass

        if len(captured) < record_gif_episodes:
            st = _get_state_for_debug(env)
            if st is not None:
                ep_states.append(st)

        done = terminated or truncated
        if done:
            scored = bool(terminated and not truncated)
            results.append(EpisodeResult(
                scored=scored,
                timeout=bool(truncated),
                steps=ep_steps,
                seconds=ep_steps * sec_per_step,
                total_reward=ep_return,
            ))
            spike_maxes.append(ep_max_gw_spike)

            if len(captured) < record_gif_episodes:
                captured.append(ep_states)

            if print_every and len(results) % print_every == 0:
                s = _summarize_pinch(results[-print_every:], spike_maxes[-print_every:])
                print(
                    f"[last {print_every}] "
                    f"goal_rate={s['goal_rate']:.2%} "
                    f"spike_rate={s['spike_rate_1500']:.2%} "
                    f"med_spike={s['median_max_goalward_spike']:.0f} "
                    f"med_ttg={s['median_time_to_goal']:.2f}s "
                    f"eps={len(results)}/{n_episodes} "
                    f"elapsed={time.time()-t0:.1f}s"
                )

            obs = _reset_env(env)
            ep_steps = 0
            ep_return = 0.0
            ep_max_gw_spike = 0.0
            prev_goalward = 0.0
            ep_states = []

            if len(captured) < record_gif_episodes:
                st = _get_state_for_debug(env)
                if st is not None:
                    ep_states.append(st)

    s = _summarize_pinch(results, spike_maxes)
    print("\n==== PINCH EVAL SUMMARY ====")
    for k, v in s.items():
        if isinstance(v, float):
            if np.isinf(v):
                print(f"  {k}: inf")
            elif "rate" in k:
                print(f"  {k}: {v:.2%}")
            else:
                print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    if record_gif_episodes > 0 and captured:
        try:
            _save_topdown_gif_labeled(
                captured, gif_out_path, fps=20,
                title=f"Pinch Eval Stage {stage} (Top-Down)",
                attack_orange=True,
            )
        except Exception as e:
            print(f"Could not save GIF: {e}")


def _discover_pinch_root(stage: int) -> str:
    """Find checkpoint folder for the given pinch stage."""
    checkpoints_dir = "checkpoints"
    base_name = f"pinch_stage{stage}"
    if not os.path.isdir(checkpoints_dir):
        raise FileNotFoundError(f"No '{checkpoints_dir}' directory found.")

    matches = []
    for name in sorted(os.listdir(checkpoints_dir)):
        full = os.path.join(checkpoints_dir, name)
        if os.path.isdir(full) and name.startswith(base_name):
            matches.append(full)

    if not matches:
        raise FileNotFoundError(
            f"No checkpoint folders matching '{base_name}*' in {checkpoints_dir}/.\n"
            f"Contents: {os.listdir(checkpoints_dir)}"
        )

    if len(matches) == 1:
        return matches[0]

    print(f"\nFound {len(matches)} checkpoint runs for stage {stage}:")
    for i, m in enumerate(matches):
        subs = [s for s in os.listdir(m) if os.path.isdir(os.path.join(m, s))]
        latest = max(subs) if subs else "empty"
        print(f"  [{i+1}] {m}  ({len(subs)} checkpoints, latest: {latest})")

    while True:
        try:
            choice = input(f"\nWhich run? [1-{len(matches)}]: ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(matches):
                return matches[idx]
        except (ValueError, EOFError):
            pass
        print("Invalid choice, try again.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate pinch specialist.")
    parser.add_argument("--stage", type=int, required=True, choices=[1, 2, 3])
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--gif-episodes", type=int, default=2)
    args = parser.parse_args()

    root = _discover_pinch_root(args.stage)
    evaluate_pinch(
        root,
        stage=args.stage,
        n_episodes=args.episodes,
        render=False,
        record_gif_episodes=args.gif_episodes,
        gif_out_path=os.path.join("checkpoints", "pinch_eval.gif"),
    )
