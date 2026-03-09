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


def train(
    n_proc: int = 4,
    total_timesteps: int = 50_000_000,
    gpu: bool = False,
    batch_size: int = 100_000,
    save_dir: str = "../checkpoints",
    run_name: str = "kuxir_ppo_run",
    critic_warmup_steps: int = 150_000,
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

    shared_curriculum = SharedCurriculum()
    logger = CurriculumMetricsLogger(shared_curriculum, mechanic_name="kuxir")

    env_factory = lambda: build_env(
        mechanic_name="kuxir",
        data_dir="../extracted_mechanics",
        curriculum=shared_curriculum
    )

    # Initialize the rlgym-ppo Learner
    learner = Learner(
        env_create_function=env_factory,
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
        log_to_wandb=False,
        render=False,
    )

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
        obs_count += learner.ts_per_iteration
        
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
    args = parser.parse_args()

    train(n_proc=args.n_proc, total_timesteps=args.total_timesteps, gpu=args.gpu)