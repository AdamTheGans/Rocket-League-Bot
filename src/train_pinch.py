# src/train_pinch.py
"""
Training script for the wall/corner pinch specialist.

Usage:
    python src/train_pinch.py --stage 1                        # Stage 1 micro-skill
    python src/train_pinch.py --stage 2 --resume-from checkpoints/pinch_stage1
    python src/train_pinch.py --stage 3 --gpu --n-proc 24 --seed 42
"""
from __future__ import annotations

import argparse
import os
import sys

from rlgym_ppo import Learner

from envs.pinch import build_env
from metrics.pinch_metrics import PinchLogger

# Module-level stage variable -- set by main() before Learner is created.
# Must be module-level so env_factory can be pickled for multiprocessing.
_STAGE: int = 1


def env_factory():
    """Build the pinch env. Uses module-level _STAGE (pickle-safe)."""
    return build_env(render=False, tick_skip=8, stage=_STAGE)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the pinch specialist (backward-chaining stages 1-3)."
    )
    parser.add_argument(
        "--stage", type=int, required=True, choices=[1, 2, 3],
        help="Training stage: 1=micro-skill, 2=approach, 3=live-ish",
    )
    parser.add_argument(
        "--gpu", action="store_true",
        help="Use GPU-optimized config (larger batch, more procs).",
    )
    parser.add_argument(
        "--n-proc", type=int, default=20,
        help="Number of parallel env processes (default: 20).",
    )
    parser.add_argument(
        "--seed", type=int, default=123,
        help="Random seed for reproducibility (default: 123).",
    )
    parser.add_argument(
        "--resume-from", type=str, default=None,
        help="Path to checkpoint folder from a previous stage to fine-tune.",
    )
    return parser.parse_args()


def _find_latest_checkpoint_in(folder: str) -> str | None:
    """
    Given a base checkpoint folder (e.g. 'checkpoints/pinch_stage1'), find
    the latest timestamped run folder (rlgym-ppo appends a unix timestamp).
    Returns the run folder path, or None if nothing found.
    """
    if not os.path.isdir(folder):
        # Maybe the user gave us the exact run folder already
        parent = os.path.dirname(folder)
        if os.path.isdir(parent):
            folder = parent
        else:
            return None

    # Look for timestamped sub-folders in the parent directory
    parent_dir = os.path.dirname(folder)
    base_name = os.path.basename(folder)

    if not os.path.isdir(parent_dir):
        return None

    # Find all folders starting with our base name
    candidates = []
    for name in os.listdir(parent_dir):
        full = os.path.join(parent_dir, name)
        if os.path.isdir(full) and name.startswith(base_name):
            candidates.append(full)

    if not candidates:
        return None

    # Return the one with the highest unix timestamp (or just the latest modified)
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def main():
    global _STAGE

    args = parse_args()
    _STAGE = args.stage
    stage = args.stage

    os.makedirs("checkpoints", exist_ok=True)

    # -- Monkey-patch: fix rlgym-ppo kbhit crash on Windows --
    try:
        import rlgym_ppo.util.kbhit as _kbhit
        _original_getch = _kbhit.KBHit.getch

        def _safe_getch(self):
            try:
                return _original_getch(self)
            except UnicodeDecodeError:
                return ""

        _kbhit.KBHit.getch = _safe_getch
    except Exception:
        pass

    # -- Hyperparameters --
    n_proc = args.n_proc
    min_inference_size = max(1, int(round(n_proc * 0.9)))

    is_resume = args.resume_from is not None
    resume_path = args.resume_from

    from metrics.pinch_metrics import PinchLogger, StageCompleteException

    while stage <= 3:
        _STAGE = stage
        lr = 5e-5 if is_resume else 1e-4
        ent_coef = 0.008 if is_resume else 0.01

        checkpoint_folder = os.path.join("checkpoints", f"pinch_stage{stage}")

        if args.gpu:
            batch_size = 100_000
            ts_per_iter = 100_000
            exp_buffer = 300_000
            minibatch = 50_000
        else:
            batch_size = 50_000
            ts_per_iter = 50_000
            exp_buffer = 150_000
            minibatch = 50_000

        # Timestep limits per stage (can be exceeded -- just a default)
        timestep_limits = {1: 100_000_000, 2: 200_000_000, 3: 500_000_000}

        csv_path = os.path.join("checkpoints", f"pinch_stage{stage}_metrics.csv")

        print(f"\n{'='*50}")
        print(f"  PINCH SPECIALIST -- Stage {stage}")
        print(f"  {'GPU mode' if args.gpu else 'CPU mode'}")
        print(f"  n_proc={n_proc}  batch={batch_size}  lr={lr}")
        print(f"  Checkpoint folder: {checkpoint_folder}")
        if is_resume:
            print(f"  Resuming from: {resume_path}")
        print(f"  Seed: {args.seed}")
        print(f"{'='*50}\n")

        # Determine checkpoint_load_folder:
        if is_resume:
            # Cross-stage resume: find the actual run folder
            resolved = _find_latest_checkpoint_in(resume_path)
            if resolved is None:
                print(f"WARNING: Could not find checkpoint run in {resume_path}")
                print("Starting fresh instead.")
                load_folder = None
            else:
                load_folder = resolved
                print(f"Resolved resume path: {load_folder}")
        else:
            # Fresh start or same-stage auto-resume via "latest"
            load_folder = "latest"

        learner = Learner(
            env_create_function=env_factory,
            n_proc=n_proc,
            min_inference_size=min_inference_size,
            metrics_logger=PinchLogger(
                csv_path=csv_path,
                tick_skip=8,
                timeout_seconds={1: 2.0, 2: 4.0, 3: 6.0}[stage],
                stage=stage,
            ),
            random_seed=args.seed,
            timestep_limit=timestep_limits.get(stage, 200_000_000),
            save_every_ts=1_000_000,
            checkpoints_save_folder=checkpoint_folder,
            checkpoint_load_folder=load_folder,
            policy_layer_sizes=[1024, 1024, 512, 512],
            critic_layer_sizes=[1024, 1024, 512, 512],
            ppo_batch_size=batch_size,
            ts_per_iteration=ts_per_iter,
            exp_buffer_size=exp_buffer,
            ppo_minibatch_size=minibatch,
            ppo_epochs=2,
            policy_lr=lr,
            critic_lr=lr,
            ppo_ent_coef=ent_coef,
            standardize_returns=True,
            standardize_obs=False,
        )

        try:
            learner.learn()
            # If learn exits normally without an exception, the user quit or timestep limit reached
            break
        except StageCompleteException as e:
            print(f"\n{'='*60}")
            print(f"  🎉 STAGE {stage} MASTERED! Transitioning to Stage {stage + 1}")
            print(f"  Criteria met at {learner.agent.cumulative_timesteps} timesteps.")
            print(f"{'='*60}\n")
            
            # Save final checkpoint for this stage
            learner.save(learner.agent.cumulative_timesteps)
            
            # Setup for next stage auto-resume
            is_resume = True
            resume_path = checkpoint_folder
            stage += 1


if __name__ == "__main__":
    main()
