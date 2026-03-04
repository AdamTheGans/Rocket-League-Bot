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


def visualize_pinch(checkpoints_root: str, stage: int = 1):
    import math
    import rlviser_py as vis
    import RocketSim as rsim

    env = build_env(render=False, tick_skip=8, stage=stage)

    policy_path = _find_latest_policy_path(checkpoints_root)
    print(f"Loading policy: {policy_path}")
    policy = _load_policy(policy_path, env)

    obs_mean, obs_std = _load_obs_stats(policy_path)
    if obs_mean is not None:
        print(f"Loaded obs standardization stats (dim={obs_mean.shape[0]})")
    else:
        print("WARNING: No obs standardization stats found")

    # Get underlying arena
    rlgym_env = getattr(env, "rlgym_env", env.unwrapped.rlgym_env if hasattr(env, "unwrapped") and hasattr(env.unwrapped, "rlgym_env") else env)
    engine = rlgym_env.transition_engine
    arena = engine._arena

    print("\nLaunching RLViser to view the agent in action...")
    print("NOTE: If Windows Firewall prompts you, you MUST allow it for private networks.")
    print("rlviser.exe requires local UDP access to stream the physics data.")
    
    vis.set_boost_pad_locations([pad.get_pos().as_tuple() for pad in arena.get_boost_pads()])
    time.sleep(2.0)

    scenarios_needed = {"right": 2, "left": 2}
    scenarios_run = 0
    first_scenario = True

    while scenarios_needed["right"] > 0 or scenarios_needed["left"] > 0:
        obs = _reset_env(env)
        
        ball_x = arena.ball.get_state().pos.x
        side = "right" if ball_x > 0 else "left"
        
        if scenarios_needed[side] <= 0:
            continue
            
        scenarios_needed[side] -= 1
        scenarios_run += 1
        
        print(f"\n--- Scenario {scenarios_run}/4: {side.capitalize()} Wall ---")
        
        delay_seconds = 7 if first_scenario else 4
        first_scenario = False
        
        print(f"Showing starting state statically for {delay_seconds} seconds...")
        pause_ticks = int(delay_seconds * 120)
        start_time = time.time()
        for i in range(pause_ticks):
            pad_states = [pad.get_state().is_active for pad in arena.get_boost_pads()]
            b_state = arena.ball.get_state()
            try:
                car_data = [(c.id, c.team, c.get_config(), c.get_state()) for c in arena.get_cars()]
            except:
                car_data = []
            vis.render(0, 120, rsim.GameMode.SOCCAR, pad_states, b_state, car_data)
            
            target_time = start_time + (i / 120.0)
            now = time.time()
            if target_time > now:
                time.sleep(target_time - now)
            
        original_engine_step = engine.step
        
        start_time = time.time()
        global_steps = 0
        
        def hooked_engine_step(actions, shared_info):
            nonlocal global_steps
            if len(engine._cars) == 0:
                steps = 1
            else:
                action = next(iter(actions.values()))
                steps = action.shape[0]

            for step in range(steps):
                if engine._rlbot_delay:
                    engine._arena.step(1)

                for agent_id, action in actions.items():
                    controls = rsim.CarControls()
                    controls.throttle = action[step, 0]
                    controls.steer = action[step, 1]
                    controls.pitch = action[step, 2]
                    controls.yaw = action[step, 3]
                    controls.roll = action[step, 4]
                    controls.jump = bool(action[step, 5])
                    controls.boost = bool(action[step, 6])
                    controls.handbrake = bool(action[step, 7])

                    engine._cars[agent_id].set_controls(controls)

                if not engine._rlbot_delay:
                    engine._arena.step(1)

                engine._tick_count += 1
                
                pad_states = [pad.get_state().is_active for pad in arena.get_boost_pads()]
                b_state = arena.ball.get_state()
                try:
                    car_data = [(c.id, c.team, c.get_config(), c.get_state()) for c in arena.get_cars()]
                except:
                    car_data = []
                    
                vis.render(0, 120, rsim.GameMode.SOCCAR, pad_states, b_state, car_data)
                
                # Dynamic sleep to correct for python execution overhead
                target_time = start_time + (global_steps / 120.0)
                now = time.time()
                if target_time > now:
                    time.sleep(target_time - now)
                global_steps += 1

            return engine._get_state()
                
        engine.step = hooked_engine_step
        
        ep_max_speed = 0.0
        ep_max_gw_spike = 0.0
        prev_goalward = 0.0
        
        state = arena.ball.get_state()
        ball_pos = np.asarray(state.pos.as_tuple(), dtype=np.float32)
        goal = np.array([0, common_values.BACK_NET_Y, 0], dtype=np.float32)
        diff = goal - ball_pos
        dist = float(np.linalg.norm(diff))
        goal_dir = diff / dist if dist > 1e-6 else np.array([0, 1, 0], dtype=np.float32)
        prev_goalward = float(np.dot(np.asarray(state.vel.as_tuple(), dtype=np.float32), goal_dir))
        
        done = False
        steps = 0
        max_duration_seconds = 7.0
        max_steps = int(max_duration_seconds * 120 / 8)
        
        print("Agent is playing...")
        while not done and steps < max_steps:
            policy_obs = _standardize_obs(obs, obs_mean, obs_std) if obs_mean is not None else obs

            if hasattr(policy, "get_action"):
                with torch.no_grad():
                    action, _ = policy.get_action(policy_obs, deterministic=True)
            else:
                with torch.no_grad():
                    logits = policy(torch.from_numpy(policy_obs).float().unsqueeze(0))
                    action = int(torch.argmax(logits, dim=-1).item())

            action_arr = _to_action_for_env(action)
            obs, r, terminated, truncated, info = _step_env(env, action_arr)
            
            steps += 1
            done = terminated or truncated
            
            # Use manual physics interrogation just like pinch_reward.py for accurate metrics
            state = arena.ball.get_state()
            vel_arr = np.asarray(state.vel.as_tuple(), dtype=np.float32)
            pos_arr = np.asarray(state.pos.as_tuple(), dtype=np.float32)
            
            raw_spd = math.sqrt(vel_arr[0]**2 + vel_arr[1]**2 + vel_arr[2]**2)
            ep_max_speed = max(ep_max_speed, raw_spd)
            
            diff = goal - pos_arr
            dist = float(np.linalg.norm(diff))
            goal_dir = diff / dist if dist > 1e-6 else np.array([0, 1, 0], dtype=np.float32)
            
            curr_goalward = float(np.dot(vel_arr, goal_dir))
            
            delta = curr_goalward - prev_goalward
            if delta > 0:
                ball_z_vel = vel_arr[2]
                if 300.0 <= ball_z_vel <= 550.0:
                    multiplier = 1.0
                else:
                    z_diff = min(abs(ball_z_vel - 300.0), abs(ball_z_vel - 550.0))
                    multiplier = max(0.1, 1.0 - (z_diff / 500.0))
                
                gw_spike = delta * multiplier
                # Convert back up to raw UU diff to match how training logs it implicitly during backward pass summation
                ep_max_gw_spike = max(ep_max_gw_spike, gw_spike)
            
            prev_goalward = curr_goalward

        engine.step = original_engine_step
        
        remaining_ticks = int(max_duration_seconds * 120) - (steps * 8)
        if remaining_ticks > 0:
            print(f"Goal scored/episode ended early! Letting physics play out remaining {remaining_ticks/120.0:.2f}s...")
            for _ in range(remaining_ticks):
                arena.step(1)
                pad_states = [pad.get_state().is_active for pad in arena.get_boost_pads()]
                b_state = arena.ball.get_state()
                try:
                    car_data = [(c.id, c.team, c.get_config(), c.get_state()) for c in arena.get_cars()]
                except:
                    car_data = []
                vis.render(0, 120, rsim.GameMode.SOCCAR, pad_states, b_state, car_data)
                
                target_time = start_time + (global_steps / 120.0)
                now = time.time()
                if target_time > now:
                    time.sleep(target_time - now)
                global_steps += 1

        print(f"Attempt {scenarios_run} Top Speed: {ep_max_speed:.2f} | FilteredGoalward: {ep_max_gw_spike:.2f}")

    print("\nVisualizations complete!")
    vis.quit()


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
    parser.add_argument("--visualize", action="store_true", help="Launch RLViser and playback 4 scenarios (7s and 4s startup delays)")
    args = parser.parse_args()

    root = _discover_pinch_root(args.stage)
    
    if args.visualize:
        print("\n=== STARTING 3D VISUALIZATION ===")
        visualize_pinch(root, stage=args.stage)
    else:
        evaluate_pinch(
            root,
            stage=args.stage,
            n_episodes=args.episodes,
            render=False,
            record_gif_episodes=args.gif_episodes,
            gif_out_path=os.path.join("checkpoints", "pinch_eval.gif"),
        )
