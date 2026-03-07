# src/train_mechanic_curriculum.py
"""
Multi-task curriculum training script for mechanic specialists.

Loads a pretrained student policy, wraps it in an SB3-compatible
ActorCriticPolicy, and trains with PPO on a mix of mechanic-specific
setups (from replay trajectories) and normal gameplay (kickoffs).

Usage:
    cd src
    python train_mechanic_curriculum.py \\
        --mechanic kuxir \\
        --data-dir ../extracted_mechanics \\
        --checkpoint ../checkpoints/nexto_stuff/dagger_2/student_policy.pt \\
        --n-proc 4 \\
        --total-timesteps 50_000_000

    # GPU mode with more workers:
    python train_mechanic_curriculum.py \\
        --mechanic kuxir --gpu --n-proc 16 --mechanic-prob 0.4
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Type

import numpy as np
import torch
import torch.nn as nn

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList

from envs.mixed_training_env import MixedEnvFactory
from curriculum.curriculum_callback import CurriculumCallback


# ─────────────────────────────────────────────────────────────── #
#  Custom SB3 Feature Extractor (matches StudentPolicy arch)
# ─────────────────────────────────────────────────────────────── #

class StudentFeatureExtractor(BaseFeaturesExtractor):
    """
    SB3 feature extractor that matches the pretrained StudentPolicy's
    MLP architecture (with LayerNorm + ReLU).

    The extracted features are the output of the last hidden layer.
    SB3's ActorCriticPolicy will add its own action_net and value_net
    linear heads on top.

    Parameters
    ----------
    observation_space : gym.spaces.Box
        The observation space (obs_dim is inferred from it).
    layer_sizes : list[int]
        Hidden layer sizes matching the student architecture.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        layer_sizes: List[int] = None,
    ):
        if layer_sizes is None:
            layer_sizes = [2048, 1024, 1024, 512]

        # The features_dim is the size of the last hidden layer
        features_dim = layer_sizes[-1]
        super().__init__(observation_space, features_dim=features_dim)

        # Build the MLP with LayerNorm (same as StudentPolicy)
        layers: list[nn.Module] = []
        in_dim = int(np.prod(observation_space.shape))
        for hidden_dim in layer_sizes:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim

        self.net = nn.Sequential(*layers)
        self.layer_sizes = list(layer_sizes)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations)


def load_pretrained_weights(
    model: PPO,
    checkpoint_path: str,
    metadata_path: Optional[str] = None,
    device: str = "cpu",
) -> None:
    """
    Load pretrained StudentPolicy weights into the SB3 PPO model's
    feature extractor (actor side).

    The StudentPolicy has layers:
        Linear -> LayerNorm -> ReLU -> ... -> Linear (final logits)

    Our StudentFeatureExtractor has the hidden layers (all but the final
    Linear). We load weights for those layers.

    The final logits layer of the StudentPolicy corresponds to SB3's
    ``action_net`` — we load that too.

    Parameters
    ----------
    model : PPO
        The SB3 PPO model with StudentFeatureExtractor as feature extractor.
    checkpoint_path : str
        Path to ``student_policy.pt``.
    metadata_path : str, optional
        Path to ``metadata.json`` (defaults to same dir as checkpoint).
    device : str
        PyTorch device.
    """
    if metadata_path is None:
        metadata_path = os.path.join(os.path.dirname(checkpoint_path), "metadata.json")

    # Load student state dict
    student_sd = torch.load(checkpoint_path, map_location=device)

    # ── Map student keys to feature extractor ───────────────────
    # The student's nn.Sequential uses indices where ReLU modules
    # (indices 2, 5, 8, 11) have no parameters.  So keys are:
    #   net.0 (Linear), net.1 (LN), net.3 (Linear), net.4 (LN), ...
    # Our feature extractor has the SAME nn.Sequential structure
    # (same indices), so keys match directly for all hidden layers.
    # The student's final layer (e.g., net.12) is the logits head
    # which maps to SB3's action_net.

    # Target the Actor's dedicated feature extractor (pi)
    fe = model.policy.pi_features_extractor
    fe_sd = fe.state_dict()

    # Direct key matching: copy any student key that exists in the FE
    new_fe_sd = {}
    for key in fe_sd:
        if key in student_sd:
            if fe_sd[key].shape == student_sd[key].shape:
                new_fe_sd[key] = student_sd[key]
            else:
                print(f"  WARNING: Shape mismatch for {key}: "
                      f"FE={fe_sd[key].shape} vs Student={student_sd[key].shape}")

    loaded = fe.load_state_dict(new_fe_sd, strict=False)
    matched = len(new_fe_sd)
    total = len(fe_sd)
    print(f"  Loaded {matched}/{total} feature extractor parameters from {checkpoint_path}")
    if loaded.missing_keys:
        print(f"  Missing FE keys: {loaded.missing_keys}")

    # ── Load final logits Linear into SB3's action_net ──────────
    # Find the highest-indexed Linear in the student (the logits head)
    # Student keys not in FE are candidates for the action head
    action_keys = {k: v for k, v in student_sd.items() if k not in fe_sd}
    if action_keys:
        action_sd = {}
        for key, val in action_keys.items():
            # Rename e.g. "net.12.weight" → "weight"
            simple_key = key.split(".")[-1]
            action_sd[simple_key] = val

        try:
            model.policy.action_net.load_state_dict(action_sd, strict=True)
            print(f"  Loaded final logits layer into action_net "
                  f"(keys: {list(action_keys.keys())})")
        except Exception as e:
            print(f"  WARNING: Could not load action_net weights: {e}")


# ─────────────────────────────────────────────────────────────── #
#  Critic Warmup Callback
# ─────────────────────────────────────────────────────────────── #

class CriticWarmupCallback(BaseCallback):
    """
    Freeze the actor (features_extractor + action_net) for the first
    ``warmup_steps`` timesteps so only the value head trains.

    This prevents catastrophic forgetting of pretrained actor weights
    while the critic learns reasonable value estimates from scratch.

    After warmup, all parameters are unfrozen and normal PPO resumes.

    Parameters
    ----------
    warmup_steps : int
        Number of timesteps to keep the actor frozen (default 150_000).
    verbose : int
        Verbosity (0 = silent, 1 = print events).
    """

    def __init__(self, warmup_steps: int = 150_000, verbose: int = 1):
        super().__init__(verbose=verbose)
        self.warmup_steps = warmup_steps
        self._frozen = False
        self._unfrozen = False

    def _on_training_start(self) -> None:
        """Freeze actor parameters at the start of training."""
        if self.warmup_steps <= 0:
            return

        policy = self.model.policy
        frozen_count = 0

        # Freeze features_extractor (the pretrained MLP hidden layers)
        # Freeze/Unfreeze pi_features_extractor (the pretrained MLP hidden layers)
        for param in policy.pi_features_extractor.parameters():
            param.requires_grad = False
            frozen_count += param.numel()

        # Freeze action_net (the pretrained logits head)
        for param in policy.action_net.parameters():
            param.requires_grad = False
            frozen_count += param.numel()

        # Freeze action_dist (log_std etc. if it exists)
        if hasattr(policy, 'action_dist') and hasattr(policy.action_dist, 'parameters'):
            for param in policy.action_dist.parameters():
                param.requires_grad = False
                frozen_count += param.numel()

        self._frozen = True

        # Count trainable (value head) params
        trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)

        if self.verbose:
            print(f"\n  [CriticWarmup] ❄️  Actor FROZEN for {self.warmup_steps:,} steps")
            print(f"  [CriticWarmup]    Frozen params:    {frozen_count:,}")
            print(f"  [CriticWarmup]    Trainable params:  {trainable:,} (value head only)\n")

    def _on_step(self) -> bool:
        """Check if warmup is done and unfreeze actor."""
        if self._frozen and not self._unfrozen and self.num_timesteps >= self.warmup_steps:
            self._unfreeze_actor()

        # Log warmup status to TensorBoard
        if self.logger is not None:
            self.logger.record("warmup/actor_frozen", int(self._frozen and not self._unfrozen))

        return True

    def _unfreeze_actor(self) -> None:
        """Unfreeze all actor parameters."""
        policy = self.model.policy

        # Freeze/Unfreeze pi_features_extractor (the pretrained MLP hidden layers)
        for param in policy.pi_features_extractor.parameters():
            param.requires_grad = True
        for param in policy.action_net.parameters():
            param.requires_grad = True
        if hasattr(policy, 'action_dist') and hasattr(policy.action_dist, 'parameters'):
            for param in policy.action_dist.parameters():
                param.requires_grad = True

        self._unfrozen = True
        trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)

        if self.verbose:
            print(f"\n  [CriticWarmup] 🔥 Actor UNFROZEN at step {self.num_timesteps:,}")
            print(f"  [CriticWarmup]    All {trainable:,} params now trainable\n")


# ─────────────────────────────────────────────────────────────── #
#  Main training function
# ─────────────────────────────────────────────────────────────── #

def train(
    mechanic_name: str = "kuxir",
    data_dir: str = "../extracted_mechanics",
    checkpoint_path: Optional[str] = None,
    n_proc: int = 4,
    total_timesteps: int = 50_000_000,
    mechanic_prob: float = 0.4,
    tick_skip: int = 8,
    episode_seconds: float = 15.0,
    gpu: bool = False,
    lr: float = 5e-5,
    batch_size: int = 4096,
    n_epochs: int = 3,
    clip_range: float = 0.08,
    ent_coef: float = 0.005,
    critic_warmup_steps: int = 150_000,
    save_freq: int = 100_000,
    save_dir: str = "../checkpoints",
    seed: int = 42,
    opponent_checkpoint: Optional[str] = None,
    eval_interval: int = 5000,
    layer_sizes: Optional[List[int]] = None,
    resume_from: Optional[str] = None,
):
    """
    Main training loop using SB3 PPO with curriculum learning.

    Parameters
    ----------
    mechanic_name : str
        Mechanic identifier (e.g., "kuxir").
    data_dir : str
        Path to .npy trajectory directory.
    checkpoint_path : str, optional
        Path to pretrained student_policy.pt for weight initialization.
    n_proc : int
        Number of parallel environments.
    total_timesteps : int
        Total training timesteps.
    mechanic_prob : float
        Probability of mechanic episodes (default 0.4).
    tick_skip : int
        Action repeat (default 8).
    episode_seconds : float
        Episode timeout (default 15.0).
    gpu : bool
        Whether to use GPU.
    lr : float
        Learning rate (default 3e-4).
    batch_size : int
        PPO batch size.
    n_epochs : int
        PPO epochs per update.
    ent_coef : float
        Entropy coefficient.
    clip_range : float
        PPO clip range (default 0.08 for fine-tuning).
    save_freq : int
        Save checkpoint every N timesteps.
    save_dir : str
        Directory for saving checkpoints.
    seed : int
        Random seed.
    opponent_checkpoint : str, optional
        Path to frozen opponent checkpoint.
    eval_interval : int
        Curriculum evaluation interval in steps.
    layer_sizes : list[int], optional
        Hidden layer sizes for the feature extractor.
    resume_from : str, optional
        Path to a saved SB3 model to resume training from.
    critic_warmup_steps : int
        Freeze actor parameters for this many timesteps while the
        value head trains from scratch (default 150_000).
    """
    if layer_sizes is None:
        layer_sizes = [2048, 1024, 1024, 512]

    device = "cuda" if gpu and torch.cuda.is_available() else "cpu"

    # ── Check metadata for obs_dim / num_actions ────────────────────
    obs_size = 92
    num_actions = 90
    if checkpoint_path:
        meta_path = os.path.join(os.path.dirname(checkpoint_path), "metadata.json")
        if os.path.isfile(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            obs_size = meta.get("obs_dim", 92)
            num_actions = meta.get("num_actions", 90)
            if "layer_sizes" in meta:
                layer_sizes = meta["layer_sizes"]
            print(f"  Loaded metadata: obs_dim={obs_size}, num_actions={num_actions}, "
                  f"layers={layer_sizes}")

    # ── Print training banner ───────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  MECHANIC CURRICULUM TRAINING — {mechanic_name.upper()}")
    print(f"{'='*60}")
    print(f"  Mechanic:            {mechanic_name}")
    print(f"  Data dir:            {data_dir}")
    print(f"  Mechanic prob:       {mechanic_prob:.0%}")
    print(f"  Device:              {device}")
    print(f"  Workers:             {n_proc}")
    print(f"  Total timesteps:     {total_timesteps:,}")
    print(f"  Batch size:          {batch_size}")
    print(f"  Learning rate:       {lr}")
    print(f"  Clip range:          {clip_range}")
    print(f"  Entropy coef:        {ent_coef}")
    print(f"  Episode timeout:     {episode_seconds}s")
    print(f"  Checkpoint:          {checkpoint_path or 'None (train from scratch)'}")
    print(f"  Critic warmup:       {critic_warmup_steps:,} steps")
    print(f"  Resume from:         {resume_from or 'None'}")
    print(f"  Save directory:      {save_dir}")
    print(f"  Seed:                {seed}")
    print(f"{'='*60}\n")

    # ── Create vectorized environments ──────────────────────────────
    env_kwargs = dict(
        mechanic_name=mechanic_name,
        data_dir=data_dir,
        mechanic_prob=mechanic_prob,
        tick_skip=tick_skip,
        episode_seconds=episode_seconds,
        obs_size=obs_size,
        num_actions=num_actions,
        opponent_checkpoint=opponent_checkpoint,
        opponent_device="cpu",  # Opponent always on CPU to save GPU memory
    )

    print(f"  Creating {n_proc} parallel environments...")
    if n_proc > 1:
        vec_env = SubprocVecEnv(
            [MixedEnvFactory(**env_kwargs) for _ in range(n_proc)]
        )
    else:
        vec_env = DummyVecEnv(
            [MixedEnvFactory(**env_kwargs)]
        )

    # ── Create or resume the PPO model ──────────────────────────────
    run_name = f"{mechanic_name}_curriculum"
    tensorboard_log = os.path.join(save_dir, f"{run_name}_tb")

    # Custom policy kwargs to use our StudentFeatureExtractor
    policy_kwargs = dict(
        features_extractor_class=StudentFeatureExtractor,
        features_extractor_kwargs=dict(layer_sizes=layer_sizes),
        net_arch=dict(pi=[], vf=[]), 
        share_features_extractor=False, # <--- SEPARATES THE BRAINS
    )

    if resume_from and os.path.isfile(resume_from):
        print(f"  Resuming from: {resume_from}")
        model = PPO.load(
            resume_from,
            env=vec_env,
            device=device,
            tensorboard_log=tensorboard_log,
        )
    else:
        model = PPO(
            policy=ActorCriticPolicy,
            env=vec_env,
            learning_rate=lr,
            n_steps=2048,
            batch_size=batch_size,
            n_epochs=n_epochs,
            clip_range=clip_range,
            ent_coef=ent_coef,
            verbose=1,
            device=device,
            seed=seed,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
        )

        # Load pretrained weights into the model
        if checkpoint_path and os.path.isfile(checkpoint_path):
            print(f"\n  Loading pretrained weights from: {checkpoint_path}")
            load_pretrained_weights(model, checkpoint_path, device=device)
        else:
            print(f"  No pretrained weights — training from scratch")

    # ── Build callbacks ─────────────────────────────────────────────
    # The CurriculumCallback syncs difficulty/noise to all workers
    # via VecEnv.set_attr().  Each worker's SelfPlayEnv exposes
    # difficulty/noise_amount properties that drill into the
    # MechanicTrajectorySetter.

    curriculum_cb = CurriculumCallback(
        mechanic_name=mechanic_name,
        eval_interval=eval_interval,
        verbose=1,
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=max(save_freq // n_proc, 1),
        save_path=os.path.join(save_dir, run_name),
        name_prefix=f"{mechanic_name}_ppo",
        verbose=1,
    )

    callbacks = CallbackList([curriculum_cb, checkpoint_cb])

    # Add critic warmup if loading pretrained weights
    if critic_warmup_steps > 0 and checkpoint_path and not resume_from:
        warmup_cb = CriticWarmupCallback(
            warmup_steps=critic_warmup_steps,
            verbose=1,
        )
        callbacks = CallbackList([warmup_cb, curriculum_cb, checkpoint_cb])

    # ── Train ───────────────────────────────────────────────────────
    print(f"\n  Starting training...")
    print(f"  TensorBoard: tensorboard --logdir={tensorboard_log}\n")

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            tb_log_name=run_name,
        )
    except KeyboardInterrupt:
        print(f"\n  Training interrupted by user.")

    # ── Save final model ────────────────────────────────────────────
    final_path = os.path.join(save_dir, run_name, f"{mechanic_name}_final.zip")
    model.save(final_path)
    print(f"  Final model saved to: {final_path}")

    vec_env.close()
    print(f"\n  Training complete!")


# ─────────────────────────────────────────────────────────────── #
#  CLI
# ─────────────────────────────────────────────────────────────── #

def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-task curriculum training for mechanic specialists.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train kuxir pinch with 4 workers:
  python train_mechanic_curriculum.py --mechanic kuxir --n-proc 4

  # GPU mode, load pretrained student:
  python train_mechanic_curriculum.py --mechanic kuxir --gpu --n-proc 16 \\
      --checkpoint ../checkpoints/nexto_stuff/dagger_2/student_policy.pt

  # Resume training:
  python train_mechanic_curriculum.py --mechanic kuxir \\
      --resume-from ../checkpoints/kuxir_curriculum/kuxir_final.zip
""",
    )
    parser.add_argument("--mechanic", type=str, default="kuxir",
                        help="Mechanic name (default: kuxir)")
    parser.add_argument("--data-dir", type=str, default="../extracted_mechanics",
                        help="Path to .npy trajectory directory")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to pretrained student_policy.pt")
    parser.add_argument("--opponent-checkpoint", type=str, default=None,
                        help="Path to frozen opponent checkpoint")
    parser.add_argument("--n-proc", type=int, default=4,
                        help="Number of parallel environments (default: 4)")
    parser.add_argument("--total-timesteps", type=int, default=50_000_000,
                        help="Total training timesteps (default: 50M)")
    parser.add_argument("--mechanic-prob", type=float, default=0.4,
                        help="Probability of mechanic episodes (default: 0.4)")
    parser.add_argument("--tick-skip", type=int, default=8,
                        help="Action repeat frames (default: 8)")
    parser.add_argument("--episode-seconds", type=float, default=15.0,
                        help="Episode timeout in seconds (default: 15)")
    parser.add_argument("--gpu", action="store_true",
                        help="Use GPU for training")
    parser.add_argument("--lr", type=float, default=5e-5,
                        help="Learning rate (default: 5e-5)")
    parser.add_argument("--batch-size", type=int, default=8192,
                        help="PPO batch size (default: 8192)")
    parser.add_argument("--n-epochs", type=int, default=3,
                        help="PPO epochs per update (default: 3)")
    parser.add_argument("--clip-range", type=float, default=0.08,
                        help="PPO clip range (default: 0.08)")
    parser.add_argument("--ent-coef", type=float, default=0.005,
                        help="Entropy coefficient (default: 0.005)")
    parser.add_argument("--save-freq", type=int, default=100_000,
                        help="Save checkpoint every N timesteps (default: 100k)")
    parser.add_argument("--save-dir", type=str, default="../checkpoints",
                        help="Save directory for checkpoints")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--eval-interval", type=int, default=10000,
                        help="Curriculum evaluation interval (default: 10k)")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Path to a saved SB3 model.zip to resume from")
    parser.add_argument("--critic-warmup-steps", type=int, default=150_000,
                        help="Freeze actor for N steps while critic warms up (default: 150k)")

    return parser.parse_args()


def main():
    args = parse_args()

    train(
        mechanic_name=args.mechanic,
        data_dir=args.data_dir,
        checkpoint_path=args.checkpoint,
        n_proc=args.n_proc,
        total_timesteps=args.total_timesteps,
        mechanic_prob=args.mechanic_prob,
        tick_skip=args.tick_skip,
        episode_seconds=args.episode_seconds,
        gpu=args.gpu,
        lr=args.lr,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        save_freq=args.save_freq,
        save_dir=args.save_dir,
        seed=args.seed,
        opponent_checkpoint=args.opponent_checkpoint,
        eval_interval=args.eval_interval,
        resume_from=args.resume_from,
        critic_warmup_steps=args.critic_warmup_steps,
    )


if __name__ == "__main__":
    main()
