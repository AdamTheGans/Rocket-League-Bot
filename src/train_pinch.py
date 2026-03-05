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
_DIFFICULTY: int = 1


def env_factory():
    """Build the pinch env. Uses module-level _STAGE and _DIFFICULTY (pickle-safe)."""
    return build_env(render=False, tick_skip=8, stage=_STAGE, difficulty_level=_DIFFICULTY)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the pinch specialist (backward-chaining stages 1-3)."
    )
    parser.add_argument(
        "--stage", type=int, required=True, choices=[1, 2, 3],
        help="Training stage: 1=micro-skill, 2=approach, 3=live-ish",
    )
    parser.add_argument(
        "--difficulty", type=int, default=1, choices=[1, 2, 3],
        help="Difficulty level for Domain Randomization in Stage 1 (default: 1).",
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


def _find_highest_timestep_checkpoint(base_folder: str) -> str | None:
    """
    Given a base checkpoint folder (e.g. 'checkpoints/pinch_stage1'), 
    finds the timestamped run folder that contains the highest timestep 
    sub-folder inside its 'checkpoints' directory.
    """
    if not os.path.isdir(base_folder):
        return None

    highest_ts = -1
    best_run_folder = None

    # rlgym-ppo structures it as: base_folder / <unix_timestamp> / checkpoints / <timestep> / ...
    for run_name in os.listdir(base_folder):
        run_path = os.path.join(base_folder, run_name)
        if not os.path.isdir(run_path):
            continue
            
        checkpoints_dir = os.path.join(run_path, "checkpoints")
        if not os.path.isdir(checkpoints_dir):
            continue
            
        for ts_name in os.listdir(checkpoints_dir):
            if ts_name.isdigit():
                ts = int(ts_name)
                if ts > highest_ts:
                    highest_ts = ts
                    best_run_folder = run_path
                    
    return best_run_folder


def main():
    global _STAGE, _DIFFICULTY

    args = parse_args()
    _STAGE = args.stage
    _DIFFICULTY = args.difficulty
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

    from metrics.pinch_metrics import PinchLogger

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
        if stage == 1:
            print(f"  Difficulty level: {_DIFFICULTY}")
        print(f"  {'GPU mode' if args.gpu else 'CPU mode'}")
        print(f"  n_proc={n_proc}  batch={batch_size}  lr={lr}")
        print(f"  Checkpoint folder: {checkpoint_folder}")
        if is_resume:
            print(f"  Resuming from: {resume_path}")
        print(f"  Seed: {args.seed}")
        print(f"{'='*50}\n")

        # Auto-resolve the best checkpoint folder for the current stage if we aren't given a cross-stage resume path
        if not is_resume:
            resume_path = checkpoint_folder

        resolved = _find_highest_timestep_checkpoint(resume_path)
        if resolved is None:
            print(f"WARNING: Could not find any valid checkpoints in {resume_path}")
            print("Starting fresh instead.")
            load_folder = None
        else:
            load_folder = resolved
            print(f"Resolved resume path with highest timestep: {load_folder}")

        if stage == 1:
            ep_secs = float(_DIFFICULTY + 1.0)
        elif stage == 2:
            ep_secs = 4.0
        else:
            ep_secs = 6.0

        learner = Learner(
            env_create_function=env_factory,
            n_proc=n_proc,
            min_inference_size=min_inference_size,
            metrics_logger=PinchLogger(
                csv_path=csv_path,
                tick_skip=8,
                timeout_seconds=ep_secs,
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

        # Run indefinitely until user terminates
        learner.learn()

if __name__ == "__main__":
    main()
