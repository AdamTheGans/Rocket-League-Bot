# src/train_mechanic_curriculum.py
"""
Multi-task curriculum training script using rlgym-ppo.
"""
import os
import argparse
from typing import Tuple

from rlgym_ppo import Learner
from envs.mixed_training_env import build_env
from curriculum.curriculum_callback import SharedCurriculum, CurriculumMetricsLogger
import torch
import RocketSim as rs

import multiprocessing as mp
import functools

# Limit PyTorch to 1 OpenMP thread so it doesn't fight the RocketSim workers for CPU
# on high-core machines!
torch.set_num_threads(1)

student_path = "../NOT_WORKING/data/nexto_distill/nexto_checkpoints/student_policy.pt"

# 1. Define a standard function at the top level of the module.
# Pickle can handle this because it's globally accessible.
def env_factory(shared_curriculum, tick_skip=8):
    from envs.mixed_training_env import build_env
    
    return build_env(
        mechanic_name="kuxir",
        data_dir="../extracted_mechanics",
        curriculum=shared_curriculum,
        tick_skip=tick_skip
    )


def train(
    n_proc: int = 4,
    total_timesteps: int = 50_000_000,
    gpu: bool = False,
    batch_size: int = 50_000,
    save_dir: str = "../checkpoints",
    run_name: str = "kuxir_ppo_run",
    critic_warmup_steps: int = 0,
    init_difficulty: float = 0.1,
    init_noise: float = 0.025,
    min_difficulty: float = 0.0,
    min_noise: float = 0.0,
    tick_skip: int = 8,
    load_checkpoint: str = None,
    wandb_id: str = None,
):
    device = "cuda" if gpu else "cpu"
    checkpoint_dir = os.path.join(save_dir, run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f" MECHANIC CURRICULUM TRAINING (RLGYM-PPO)")
    print(f"{'='*60}")
    print(f" Workers:         {n_proc}")
    print(f" Device:          {device}")
    print(f" Batch size:      {batch_size}")
    print(f"{'='*60}\n")

    # Create the Manager in the main process
    manager = mp.Manager()
    shared_curriculum = SharedCurriculum(manager, init_difficulty, init_noise, min_difficulty, min_noise)
    logger = CurriculumMetricsLogger(shared_curriculum, mechanic_name="kuxir")

    # Initialize the rlgym-ppo Learner
    learner = Learner(
        env_create_function=functools.partial(env_factory, shared_curriculum, tick_skip=tick_skip),
        metrics_logger=logger,
        n_proc=n_proc,
        min_inference_size=max(1, n_proc // 2),
        ppo_batch_size=batch_size,
        ts_per_iteration=batch_size,
        exp_buffer_size=batch_size * 2,
        ppo_minibatch_size=10_000,
        ppo_ent_coef=0.005,
        ppo_epochs=3,
        policy_layer_sizes=[2048, 1024, 1024, 512],
        critic_layer_sizes=[2048, 1024, 1024, 512],
        device=device,
        log_to_wandb=True,
        wandb_run_name=run_name,
        checkpoint_load_folder=load_checkpoint if load_checkpoint else None,
        render=False,
    )

    # Load the pretrained Nexto weights into the learner's agent (only if starting from scratch)
    if not load_checkpoint and os.path.exists(student_path):
        print(f" -> Load distillation candidate: {student_path}...")
        try:
            checkpoint = torch.load(student_path, map_location=device, weights_only=True)
            learner.agent.policy.load_state_dict(checkpoint)
            print(" -> Student weights loaded. Bot is starting with Nexto-level basic car control!")
        except Exception as e:
            print(f" -> Could not load distillation checkpoint ({e}). Proceeding from scratch.")
    else:
        print(" -> No student policy found. Starting from scratch.")

    # Variables for our custom training loop
    obs_count = 0
    actor_frozen = False

    # Critic Warmup Logic
    if critic_warmup_steps > 0:
        print(f" [Critic Warmup] Freezing actor for {critic_warmup_steps} steps...")
        for param in learner.agent.policy.parameters():
            param.requires_grad = False
        actor_frozen = True

    # Main Training Loop
    while obs_count < total_timesteps:
        learner.learn()
        obs_count += batch_size
        
        # Check if we need to unfreeze the actor
        if actor_frozen and obs_count >= critic_warmup_steps:
            print(f"\n [Critic Warmup] Unfreezing actor at step {obs_count}!")
            for param in learner.agent.policy.parameters():
                param.requires_grad = True
            actor_frozen = False

        # --- CURRICULUM LOGIC GOES HERE ---
        # Example: if obs_count % 1_000_000 == 0:
        #     increase_mechanic_difficulty()

        # Save Checkpoint
        if obs_count % 5_000_000 == 0:
            learner.save(os.path.join(checkpoint_dir, f"step_{obs_count}"))

    learner.save(os.path.join(checkpoint_dir, "final_model"))
    learner.cleanup()
    print("Training complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-proc", type=int, default=16)
    parser.add_argument("--total-timesteps", type=int, default=50_000_000)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--init-difficulty", type=float, default=0.1)
    parser.add_argument("--init-noise", type=float, default=0.025)
    parser.add_argument("--min-difficulty", type=float, default=0.0)
    parser.add_argument("--min-noise", type=float, default=0.0)
    parser.add_argument("--tick-skip", type=int, default=8)
    parser.add_argument("--load-checkpoint", type=str, default=None, help="Path to checkpoint folder to resume from")
    parser.add_argument("--wandb-id", type=str, default=None, help="Wandb run ID to resume logging into")
    args = parser.parse_args()

    train(
        n_proc=args.n_proc, 
        total_timesteps=args.total_timesteps, 
        gpu=args.gpu,
        init_difficulty=args.init_difficulty,
        init_noise=args.init_noise,
        min_difficulty=args.min_difficulty,
        min_noise=args.min_noise,
        tick_skip=args.tick_skip,
        load_checkpoint=args.load_checkpoint,
        wandb_id=args.wandb_id
    )