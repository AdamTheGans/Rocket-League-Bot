# src/nexto_distill/diagnostics.py
"""
Deep diagnostic analysis of student imitation quality.

Runs 5 checks:
1. Teacher baseline metrics (teacher plays, report touches/distance/speed)
2. Agreement on teacher trajectories (off-policy: student labels teacher's states)
3. Action distribution comparison (teacher vs student rollouts)
4. Confusion hotspot analysis (per-action accuracy on val set)
5. Recommendation (go/no-go for PPO, or DAgger)

Usage:
    cd src
    python -m nexto_distill.diagnostics \
        --checkpoint ../checkpoints/nexto_distill/student_policy.pt \
        --data_dir ../data/nexto_distill/shards \
        --episodes 50 \
        --on_policy_steps 200000
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time
from collections import Counter
from typing import Optional, List

import numpy as np
import torch
import torch.nn.functional as F

from nexto_distill.student_policy import StudentPolicy
from nexto_distill.teacher_nexto import NextoTeacher
from nexto_distill.generate_dataset import build_distill_env, LazyChaserOpponent
from nexto_distill.train_bc import (
    DistillationDataset,
    _discover_episode_ids,
    _split_episode_ids,
)
from rlgym.rocket_league.action_parsers import LookupTableAction


LUT = LookupTableAction.make_lookup_table()


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


def _action_label(idx: int) -> str:
    a = LUT[idx]
    return (f"[{idx:2d}] thr={a[0]:+.0f} str={a[1]:+.0f} "
            f"p={a[2]:+.0f} y={a[3]:+.0f} r={a[4]:+.0f} "
            f"j={a[5]:.0f} b={a[6]:.0f} hb={a[7]:.0f}")


def _print_action_dist(counts: Counter, total: int, label: str, top_n: int = 15):
    probs = np.array([counts.get(i, 0) / total for i in range(len(LUT))])
    probs_nonzero = probs[probs > 0]
    entropy = -np.sum(probs_nonzero * np.log(probs_nonzero))
    max_entropy = np.log(len(LUT))

    # Count "drive forward" type actions (throttle=1, no jump, ground-ish)
    fwd_frac = sum(counts.get(i, 0) for i in range(len(LUT))
                   if LUT[i][0] == 1 and LUT[i][5] == 0) / total

    print(f"\n  [{label}] Action Distribution:")
    print(f"    Entropy: {entropy:.3f} / {max_entropy:.3f} ({entropy/max_entropy*100:.1f}%)")
    print(f"    Drive-forward fraction: {fwd_frac:.1%}")
    print(f"    Top {top_n} actions:")
    for idx, count in counts.most_common(top_n):
        pct = count / total * 100
        print(f"      {_action_label(idx)}  {pct:5.1f}%")
    unused = sum(1 for i in range(len(LUT)) if counts.get(i, 0) == 0)
    print(f"    Unused: {unused}/{len(LUT)}")


# ===================================================================== #
#  CHECK 1: Teacher Baseline
# ===================================================================== #

def check_teacher_baseline(
    teacher: NextoTeacher,
    episodes: int = 50,
    seed: int = 42,
    tick_skip: int = 8,
    episode_seconds: float = 30.0,
):
    print("\n" + "=" * 60)
    print("  CHECK 1: TEACHER BASELINE METRICS")
    print("=" * 60)

    env = build_distill_env(tick_skip=tick_skip, episode_seconds=episode_seconds,
                            freeplay_seed=seed)
    opponent = LazyChaserOpponent(seed=seed + 1)

    total_steps = 0
    total_touches = 0
    ball_speed_sum = 0.0
    dist_sum = 0.0
    action_counts: Counter = Counter()
    ep_count = 0
    t0 = time.time()

    while ep_count < episodes:
        obs_dict = env.reset()
        game_state = env.state
        teacher.reset(game_state)

        blue_agent = orange_agent = None
        for aid in sorted(game_state.cars.keys()):
            car = game_state.cars[aid]
            if car.team_num == 0:
                blue_agent = aid
            else:
                orange_agent = aid

        done = False
        while not done:
            teacher_action = teacher.act(game_state, player_index=0)
            action_counts[teacher_action] += 1

            opp_action = opponent.act()
            actions = {
                blue_agent: np.array([teacher_action]),
                orange_agent: np.array([opp_action]),
            }
            obs_dict, _, term, trunc = env.step(actions)
            game_state = env.state
            teacher.update_score(game_state)

            blue_car = game_state.cars[blue_agent]
            ball_speed = float(np.linalg.norm(game_state.ball.linear_velocity))
            ball_speed_sum += ball_speed
            dist = float(np.linalg.norm(blue_car.physics.position - game_state.ball.position))
            dist_sum += dist
            if blue_car.ball_touches > 0:
                total_touches += blue_car.ball_touches
            total_steps += 1

            for aid in term:
                if term[aid] or trunc[aid]:
                    done = True
                    break

        ep_count += 1

    elapsed = time.time() - t0
    print(f"\n  Teacher Baseline ({episodes} episodes, {total_steps:,} steps):")
    print(f"    Total Touches:        {total_touches}")
    print(f"    Touches/Episode:      {total_touches/episodes:.1f}")
    print(f"    Avg Ball Speed:       {ball_speed_sum/total_steps:.0f} uu/s")
    print(f"    Avg Dist to Ball:     {dist_sum/total_steps:.0f} uu")
    print(f"    Avg Steps/Episode:    {total_steps/episodes:.1f}")
    print(f"    Elapsed:              {elapsed:.1f}s")

    _print_action_dist(action_counts, total_steps, "TEACHER", top_n=15)
    env.close()

    return {
        "touches": total_touches,
        "touches_per_ep": total_touches / episodes,
        "avg_ball_speed": ball_speed_sum / total_steps,
        "avg_dist": dist_sum / total_steps,
        "avg_steps_per_ep": total_steps / episodes,
        "action_counts": action_counts,
        "total_steps": total_steps,
    }


# ===================================================================== #
#  CHECK 2: Agreement on Teacher Trajectories
# ===================================================================== #

def check_on_policy_agreement(
    teacher: NextoTeacher,
    model: StudentPolicy,
    steps: int = 200_000,
    seed: int = 42,
    tick_skip: int = 8,
    episode_seconds: float = 30.0,
):
    print("\n" + "=" * 60)
    print("  CHECK 2: AGREEMENT ON TEACHER TRAJECTORIES")
    print("  (Student labels teacher's states — isolates covariate shift)")
    print("=" * 60)

    env = build_distill_env(tick_skip=tick_skip, episode_seconds=episode_seconds,
                            freeplay_seed=seed + 100)
    opponent = LazyChaserOpponent(seed=seed + 101)

    top1_agree = 0
    top5_agree = 0
    total = 0
    t0 = time.time()

    obs_dict = env.reset()
    game_state = env.state
    teacher.reset(game_state)

    blue_agent = orange_agent = None
    for aid in sorted(game_state.cars.keys()):
        car = game_state.cars[aid]
        if car.team_num == 0:
            blue_agent = aid
        else:
            orange_agent = aid

    while total < steps:
        student_obs = obs_dict[blue_agent]
        teacher_action = teacher.act(game_state, player_index=0)

        with torch.no_grad():
            logits = model(torch.from_numpy(student_obs).float().unsqueeze(0))
            student_action = int(logits.argmax(dim=-1).item())
            _, top5 = logits.topk(5, dim=-1)
            top5_list = top5.squeeze().tolist()

        if student_action == teacher_action:
            top1_agree += 1
        if teacher_action in top5_list:
            top5_agree += 1
        total += 1

        opp_action = opponent.act()
        actions = {
            blue_agent: np.array([teacher_action]),  # Teacher drives
            orange_agent: np.array([opp_action]),
        }
        obs_dict, _, term, trunc = env.step(actions)
        game_state = env.state
        teacher.update_score(game_state)

        done = False
        for aid in term:
            if term[aid] or trunc[aid]:
                done = True
                break

        if done:
            obs_dict = env.reset()
            game_state = env.state
            teacher.reset(game_state)
            for aid in sorted(game_state.cars.keys()):
                car = game_state.cars[aid]
                if car.team_num == 0:
                    blue_agent = aid
                else:
                    orange_agent = aid

        if total % 50_000 == 0:
            elapsed = time.time() - t0
            print(f"    [{total:>8,}/{steps:,}] "
                  f"top1={top1_agree/total:.2%} "
                  f"top5={top5_agree/total:.2%} "
                  f"elapsed={elapsed:.1f}s")

    elapsed = time.time() - t0
    top1_rate = top1_agree / total
    top5_rate = top5_agree / total

    print(f"\n  On-Policy Agreement ({total:,} steps):")
    print(f"    Top-1 Agreement:  {top1_rate:.2%}")
    print(f"    Top-5 Agreement:  {top5_rate:.2%}")
    print(f"    Elapsed:          {elapsed:.1f}s")

    env.close()
    return {"top1": top1_rate, "top5": top5_rate}


# ===================================================================== #
#  CHECK 3: Student Action Distribution (online rollout)
# ===================================================================== #

def check_student_action_dist(
    model: StudentPolicy,
    episodes: int = 50,
    seed: int = 42,
    tick_skip: int = 8,
    episode_seconds: float = 30.0,
):
    print("\n" + "=" * 60)
    print("  CHECK 3: STUDENT ACTION DISTRIBUTION (online rollout)")
    print("=" * 60)

    env = build_distill_env(tick_skip=tick_skip, episode_seconds=episode_seconds,
                            freeplay_seed=seed)

    total_steps = 0
    action_counts: Counter = Counter()
    ep_count = 0

    while ep_count < episodes:
        obs_dict = env.reset()
        game_state = env.state

        blue_agent = orange_agent = None
        for aid in sorted(game_state.cars.keys()):
            car = game_state.cars[aid]
            if car.team_num == 0:
                blue_agent = aid
            else:
                orange_agent = aid

        done = False
        while not done:
            student_obs = obs_dict[blue_agent]
            with torch.no_grad():
                logits = model(torch.from_numpy(student_obs).float().unsqueeze(0))
                action = int(logits.argmax(dim=-1).item())
            action_counts[action] += 1

            actions = {
                blue_agent: np.array([action]),
                orange_agent: np.array([0]),
            }
            obs_dict, _, term, trunc = env.step(actions)
            game_state = env.state
            total_steps += 1

            for aid in term:
                if term[aid] or trunc[aid]:
                    done = True
                    break
        ep_count += 1

    _print_action_dist(action_counts, total_steps, "STUDENT", top_n=15)
    env.close()

    return {"action_counts": action_counts, "total_steps": total_steps}


# ===================================================================== #
#  CHECK 4: Confusion Hotspots
# ===================================================================== #

def check_confusion_hotspots(
    model: StudentPolicy,
    data_dir: str,
    val_fraction: float = 0.1,
    seed: int = 42,
    device: str = "cpu",
):
    print("\n" + "=" * 60)
    print("  CHECK 4: CONFUSION HOTSPOTS (val set)")
    print("=" * 60)

    shard_paths = sorted(glob.glob(os.path.join(data_dir, "shard_*.npz")))
    all_eps = _discover_episode_ids(shard_paths)
    _, val_eps = _split_episode_ids(all_eps, val_fraction, seed)
    val_ds = DistillationDataset(shard_paths, val_eps)

    # Collect predictions
    all_teacher = []
    all_student = []

    loader = torch.utils.data.DataLoader(val_ds, batch_size=4096, shuffle=False)
    with torch.no_grad():
        for obs, actions, _ in loader:
            logits = model(obs)
            preds = logits.argmax(dim=-1)
            all_teacher.append(actions.numpy())
            all_student.append(preds.numpy())

    teacher_actions = np.concatenate(all_teacher)
    student_actions = np.concatenate(all_student)

    # Per-action accuracy for top teacher actions
    teacher_counts = Counter(teacher_actions.tolist())
    top_actions = teacher_counts.most_common(15)

    print(f"\n  Per-Action Accuracy (top 15 teacher actions, {len(teacher_actions):,} samples):")
    print(f"  {'Action':<52s} {'Count':>6s} {'Freq':>6s} {'Acc':>6s} {'Top Confusion':>30s}")
    print(f"  {'-'*100}")

    for action_idx, count in top_actions:
        mask = teacher_actions == action_idx
        correct = (student_actions[mask] == action_idx).sum()
        acc = correct / count if count > 0 else 0

        # What does the student predict instead?
        wrong_mask = mask & (student_actions != action_idx)
        if wrong_mask.sum() > 0:
            wrong_preds = Counter(student_actions[wrong_mask].tolist())
            top_wrong = wrong_preds.most_common(1)[0]
            conf_str = f"→ [{top_wrong[0]:2d}] ({top_wrong[1]/count:.0%})"
        else:
            conf_str = "—"

        freq = count / len(teacher_actions)
        print(f"  {_action_label(action_idx):<52s} {count:>6d} {freq:>5.1%} {acc:>5.1%} {conf_str:>30s}")

    # Overall stats
    overall_acc = (teacher_actions == student_actions).mean()
    print(f"\n  Overall accuracy: {overall_acc:.2%}")

    # Most confused pairs
    print(f"\n  Top 10 confusion pairs (teacher → student):")
    confusion_pairs = Counter()
    wrong = teacher_actions != student_actions
    for t, s in zip(teacher_actions[wrong], student_actions[wrong]):
        confusion_pairs[(t, s)] += 1
    for (t, s), count in confusion_pairs.most_common(10):
        print(f"    [{t:2d}] → [{s:2d}]  {count:>5d} times  "
              f"({count/len(teacher_actions):.1%} of all)")


# ===================================================================== #
#  MAIN
# ===================================================================== #

def main():
    parser = argparse.ArgumentParser(description="Deep diagnostics for distillation quality.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str,
                        default=os.path.join("..", "data", "nexto_distill", "shards"))
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--on_policy_steps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    model, meta = _load_student(args.checkpoint, args.device)
    print(f"Student: {meta['layer_sizes']}, {meta['total_params']:,} params")

    teacher = NextoTeacher()
    print(f"Teacher: {teacher.num_actions} actions")

    # 1. Teacher baseline
    teacher_stats = check_teacher_baseline(teacher, episodes=args.episodes, seed=args.seed)

    # 2. On-policy agreement
    on_policy = check_on_policy_agreement(
        teacher, model, steps=args.on_policy_steps, seed=args.seed
    )

    # 3. Student action distribution
    student_stats = check_student_action_dist(model, episodes=args.episodes, seed=args.seed)

    # 4. Confusion hotspots
    check_confusion_hotspots(model, args.data_dir, seed=args.seed, device=args.device)

    # 5. Summary + Recommendation
    print("\n" + "=" * 60)
    print("  SUMMARY & RECOMMENDATION")
    print("=" * 60)

    print(f"\n  Teacher Baseline:")
    print(f"    Touches/ep: {teacher_stats['touches_per_ep']:.1f}")
    print(f"    Avg dist:   {teacher_stats['avg_dist']:.0f} uu")
    print(f"    Avg speed:  {teacher_stats['avg_ball_speed']:.0f} uu/s")

    print(f"\n  On-Policy Agreement (teacher drives, student labels):")
    print(f"    Top-1: {on_policy['top1']:.2%}")
    print(f"    Top-5: {on_policy['top5']:.2%}")

    print(f"\n  Online Agreement (from previous eval): ~23.7%")
    print(f"  Covariate Shift Gap: {on_policy['top1']:.2%} → ~23.7%")

    if on_policy["top1"] > 0.35:
        print(f"\n  ⚡ DIAGNOSIS: Significant covariate shift detected.")
        print(f"     On-policy accuracy is {on_policy['top1']:.0%}, but online drops to ~24%.")
        print(f"     → RECOMMENDATION: (C) Run DAgger iteration.")
        print(f"       Collect 500K-1M steps of STUDENT rollouts labeled by TEACHER,")
        print(f"       mix with original data, retrain, and re-evaluate.")
    elif on_policy["top1"] > 0.28:
        print(f"\n  → RECOMMENDATION: (A) Proceed to PPO fine-tuning.")
        print(f"     The student has learned reasonable behavioral priors.")
    else:
        print(f"\n  → RECOMMENDATION: (B) Collect more teacher data.")
        print(f"     On-policy accuracy is low, suggesting data quality/quantity issues.")

    print()


if __name__ == "__main__":
    main()
