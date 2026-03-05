# src/nexto_distill/eval_imitation.py
"""
Evaluate student imitation quality in two modes:

1. **Offline** — accuracy / loss on held-out validation shards.
2. **Online**  — roll out student in RocketSim and compare actions
   with the teacher on the same states.

Usage:
    cd src
    # Offline evaluation
    python -m nexto_distill.eval_imitation \
        --checkpoint ../checkpoints/nexto_distill/student_policy.pt \
        --mode offline --data_dir ../data/nexto_distill/shards

    # Online evaluation
    python -m nexto_distill.eval_imitation \
        --checkpoint ../checkpoints/nexto_distill/student_policy.pt \
        --mode online --episodes 100

    # Online with RLViser visualization
    python -m nexto_distill.eval_imitation \
        --checkpoint ../checkpoints/nexto_distill/student_policy.pt \
        --mode online --episodes 5 --visualize
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from nexto_distill.student_policy import StudentPolicy
from nexto_distill.teacher_nexto import NextoTeacher
from nexto_distill.train_bc import (
    DistillationDataset,
    _discover_episode_ids,
    _split_episode_ids,
)
from nexto_distill.generate_dataset import build_distill_env


# ===================================================================== #
#  Load student checkpoint
# ===================================================================== #

def _load_student(checkpoint_path: str, device: str = "cpu"):
    """Load student policy from checkpoint + metadata."""
    ckpt_dir = os.path.dirname(checkpoint_path)
    meta_path = os.path.join(ckpt_dir, "metadata.json")

    if not os.path.isfile(meta_path):
        raise FileNotFoundError(
            f"metadata.json not found next to {checkpoint_path}. "
            "Cannot infer model architecture."
        )

    with open(meta_path) as f:
        meta = json.load(f)

    model = StudentPolicy(
        obs_dim=meta["obs_dim"],
        num_actions=meta["num_actions"],
        layer_sizes=meta["layer_sizes"],
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    model.to(device)

    return model, meta


# ===================================================================== #
#  Offline evaluation
# ===================================================================== #

def eval_offline(
    checkpoint_path: str,
    data_dir: str,
    val_fraction: float = 0.1,
    batch_size: int = 4096,
    seed: int = 42,
    device: str = "cpu",
):
    print("=" * 60)
    print("  OFFLINE EVALUATION")
    print("=" * 60)

    dev = torch.device(device)
    model, meta = _load_student(checkpoint_path, device)
    print(f"  Student: obs_dim={meta['obs_dim']}, "
          f"layers={meta['layer_sizes']}, "
          f"params={meta['total_params']:,}")

    shard_paths = sorted(glob.glob(os.path.join(data_dir, "shard_*.npz")))
    if not shard_paths:
        raise FileNotFoundError(f"No shards in {data_dir}")
    print(f"  Shards: {len(shard_paths)}")

    # Episode-level val split (same split as training)
    all_episodes = _discover_episode_ids(shard_paths)
    _, val_episodes = _split_episode_ids(all_episodes, val_fraction, seed)
    val_ds = DistillationDataset(shard_paths, val_episodes)
    print(f"  Val samples: {len(val_ds):,} from {len(val_episodes)} episodes")

    if len(val_ds) == 0:
        print("  ERROR: No validation samples!")
        return

    loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
    )

    ce_loss_fn = torch.nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_correct5 = 0
    total_kl = 0.0
    total_entropy = 0.0
    total = 0

    with torch.no_grad():
        for obs, actions, logits_teacher in loader:
            obs = obs.to(dev)
            actions = actions.to(dev)
            logits_teacher = logits_teacher.to(dev)

            logits_student = model(obs)

            # CE loss
            loss = ce_loss_fn(logits_student, actions)
            total_loss += loss.item() * obs.size(0)

            # Top-1
            preds = logits_student.argmax(dim=-1)
            total_correct += (preds == actions).sum().item()

            # Top-5
            _, top5 = logits_student.topk(5, dim=-1)
            total_correct5 += (top5 == actions.unsqueeze(-1)).any(dim=-1).sum().item()

            # KL divergence
            log_probs = F.log_softmax(logits_student, dim=-1)
            probs_teacher = F.softmax(logits_teacher, dim=-1)
            kl = F.kl_div(log_probs, probs_teacher, reduction="sum", log_target=False)
            total_kl += kl.item()

            # Student entropy
            probs = F.softmax(logits_student, dim=-1)
            ent = -(probs * (probs + 1e-8).log()).sum(dim=-1)
            total_entropy += ent.sum().item()

            total += obs.size(0)

    print(f"\n  Results ({total:,} samples):")
    print(f"    CE Loss:    {total_loss / total:.4f}")
    print(f"    Top-1 Acc:  {total_correct / total:.2%}")
    print(f"    Top-5 Acc:  {total_correct5 / total:.2%}")
    print(f"    KL Div:     {total_kl / total:.4f}")
    print(f"    Entropy:    {total_entropy / total:.3f}")
    print()


# ===================================================================== #
#  Online evaluation
# ===================================================================== #

def eval_online(
    checkpoint_path: str,
    episodes: int = 100,
    seed: int = 42,
    device: str = "cpu",
    tick_skip: int = 8,
    episode_seconds: float = 30.0,
    model_path: Optional[str] = None,
    visualize: bool = False,
    print_every: int = 10,
):
    print("=" * 60)
    print("  ONLINE EVALUATION — Student vs Teacher Agreement")
    print("=" * 60)

    dev = torch.device(device)
    model, meta = _load_student(checkpoint_path, device)
    print(f"  Student loaded: {meta['layer_sizes']}, {meta['total_params']:,} params")

    # Build teacher
    teacher_kwargs = {"device": device, "tick_skip": tick_skip}
    if model_path:
        teacher_kwargs["model_path"] = model_path
    teacher = NextoTeacher(**teacher_kwargs)
    print(f"  Teacher loaded: {teacher.num_actions} actions")

    # Build env
    env = build_distill_env(
        tick_skip=tick_skip,
        episode_seconds=episode_seconds,
        freeplay_seed=seed,
    )

    # Optional RLViser setup
    arena = None
    vis = None
    if visualize:
        try:
            import rlviser_py as _vis
            import RocketSim as rsim

            rlgym_env = env
            engine = rlgym_env.transition_engine
            arena = engine._arena
            vis = _vis
            vis.set_boost_pad_locations(
                [pad.get_pos().as_tuple() for pad in arena.get_boost_pads()]
            )
            time.sleep(2.0)
            print("  RLViser initialized for visualization")
        except ImportError:
            print("  WARNING: rlviser_py not available, skipping visualization")
            visualize = False

    # ── Rollout loop ──
    total_agreement = 0
    total_steps = 0
    total_touches = 0
    ball_speed_sum = 0.0
    dist_to_ball_sum = 0.0
    ep_count = 0

    t0 = time.time()

    while ep_count < episodes:
        obs_dict = env.reset()
        game_state = env.state

        teacher.reset(game_state)

        # Identify agents
        blue_agent = None
        orange_agent = None
        for agent_id in sorted(game_state.cars.keys()):
            car = game_state.cars[agent_id]
            if car.team_num == 0:
                blue_agent = agent_id
            else:
                orange_agent = agent_id

        done = False
        ep_steps = 0
        ep_agree = 0

        while not done:
            student_obs = obs_dict[blue_agent]

            # Student action
            with torch.no_grad():
                obs_tensor = torch.from_numpy(student_obs).float().unsqueeze(0).to(dev)
                logits = model(obs_tensor)
                student_action = int(logits.argmax(dim=-1).item())

            # Teacher action on same state
            teacher_action = teacher.act(game_state, player_index=0)

            # Agreement
            if student_action == teacher_action:
                ep_agree += 1

            # Use student's action to drive the car (that's who we're evaluating)
            opp_action = 0  # idle opponent for eval
            actions = {
                blue_agent: np.array([student_action]),
                orange_agent: np.array([opp_action]),
            }

            obs_dict, reward_dict, terminated_dict, truncated_dict = env.step(actions)
            game_state = env.state
            teacher.update_score(game_state)

            # Metrics
            ball_vel = game_state.ball.linear_velocity
            ball_speed = float(np.linalg.norm(ball_vel))
            ball_speed_sum += ball_speed

            blue_car = game_state.cars[blue_agent]
            dist = float(np.linalg.norm(
                blue_car.physics.position - game_state.ball.position
            ))
            dist_to_ball_sum += dist

            if blue_car.ball_touches > 0:
                total_touches += blue_car.ball_touches

            ep_steps += 1

            # RLViser rendering
            if visualize and arena is not None:
                import RocketSim as rsim
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
                time.sleep(1 / 120.0 * tick_skip)

            # Check done
            for agent_id in terminated_dict:
                if terminated_dict[agent_id] or truncated_dict[agent_id]:
                    done = True
                    break

        total_agreement += ep_agree
        total_steps += ep_steps
        ep_count += 1

        if print_every and ep_count % print_every == 0:
            agree_rate = total_agreement / total_steps if total_steps > 0 else 0
            avg_speed = ball_speed_sum / total_steps if total_steps > 0 else 0
            avg_dist = dist_to_ball_sum / total_steps if total_steps > 0 else 0
            print(
                f"  [{ep_count:>4d}/{episodes}] "
                f"agree={agree_rate:.2%} "
                f"touches={total_touches} "
                f"avg_ball_spd={avg_speed:.0f} "
                f"avg_dist={avg_dist:.0f} "
                f"elapsed={time.time() - t0:.1f}s"
            )

    # Final report
    elapsed = time.time() - t0
    agree_rate = total_agreement / total_steps if total_steps > 0 else 0
    avg_speed = ball_speed_sum / total_steps if total_steps > 0 else 0
    avg_dist = dist_to_ball_sum / total_steps if total_steps > 0 else 0

    print(f"\n{'=' * 60}")
    print(f"  ONLINE EVAL RESULTS ({episodes} episodes, {total_steps:,} steps)")
    print(f"    Action Agreement Rate:     {agree_rate:.2%}")
    print(f"    Total Ball Touches:        {total_touches}")
    print(f"    Avg Ball Speed:            {avg_speed:.0f} uu/s")
    print(f"    Avg Distance to Ball:      {avg_dist:.0f} uu")
    print(f"    Avg Steps per Episode:     {total_steps / episodes:.1f}")
    print(f"    Elapsed Time:              {elapsed:.1f}s")
    print(f"{'=' * 60}")

    if visualize and vis is not None:
        vis.quit()

    env.close()


# ===================================================================== #
#  CLI
# ===================================================================== #

def main():
    parser = argparse.ArgumentParser(description="Evaluate student imitation quality.")
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to student_policy.pt",
    )
    parser.add_argument(
        "--mode", type=str, default="online",
        choices=["offline", "online", "both"],
    )
    parser.add_argument(
        "--data_dir", type=str,
        default=os.path.join("..", "data", "nexto_distill", "shards"),
        help="Shard directory (for offline mode)",
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--tick_skip", type=int, default=8)
    parser.add_argument("--episode_seconds", type=float, default=30.0)
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to nexto-model.pt (auto-detected if not given)")
    parser.add_argument("--visualize", action="store_true",
                        help="Enable RLViser visualization (online mode only)")
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=4096)

    args = parser.parse_args()

    if args.mode in ("offline", "both"):
        eval_offline(
            checkpoint_path=args.checkpoint,
            data_dir=args.data_dir,
            val_fraction=args.val_fraction,
            batch_size=args.batch_size,
            seed=args.seed,
            device=args.device,
        )

    if args.mode in ("online", "both"):
        eval_online(
            checkpoint_path=args.checkpoint,
            episodes=args.episodes,
            seed=args.seed,
            device=args.device,
            tick_skip=args.tick_skip,
            episode_seconds=args.episode_seconds,
            model_path=args.model_path,
            visualize=args.visualize,
        )


if __name__ == "__main__":
    main()
