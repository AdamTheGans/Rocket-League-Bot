# src/envs/mixed_training_env.py
"""
Environment factory for mixed mechanic + normal training.

Creates a 1v1 RLGym env that alternates between mechanic-specific resets
(from extracted replay trajectories) and normal gameplay resets (kickoffs).
Wraps the result in a SelfPlayEnv for single-agent SB3 compatibility.

The factory function ``make_mixed_env`` returns a ready-to-use gym.Env
that can be passed directly to SB3's ``SubprocVecEnv`` or ``DummyVecEnv``.
"""
from __future__ import annotations

import os
import sys
from typing import Optional

import numpy as np

from rlgym.api import RLGym
from rlgym.rocket_league.sim import RocketSimEngine
from rlgym.rocket_league.action_parsers import LookupTableAction, RepeatAction
from rlgym.rocket_league.done_conditions import GoalCondition, TimeoutCondition, AnyCondition
from rlgym.rocket_league.obs_builders import DefaultObs
from rlgym.rocket_league.state_mutators import (
    MutatorSequence,
    FixedTeamSizeMutator,
    KickoffMutator,
)
from rlgym.rocket_league import common_values

from state_setters.trajectory_setter import MechanicTrajectorySetter
from state_setters.mixed_state_setter import MixedStateSetter
from rewards.mixed_reward import build_mixed_reward
from wrappers.self_play_env import SelfPlayEnv, make_idle_opponent, make_frozen_opponent


def make_mixed_env(
    mechanic_name: str = "kuxir",
    data_dir: str = "../extracted_mechanics",
    mechanic_prob: float = 0.4,
    tick_skip: int = 8,
    episode_seconds: float = 15.0,
    obs_size: int = 92,
    num_actions: int = 90,
    opponent_checkpoint: Optional[str] = None,
    opponent_device: str = "cpu",
    fps: int = 30,
    pre_mechanic_seconds: float = 1.5,
) -> SelfPlayEnv:
    """
    Build a 1v1 mixed-training environment wrapped for SB3.

    Parameters
    ----------
    mechanic_name : str
        Identifier for the mechanic (e.g., "kuxir", "ceiling_shot").
    data_dir : str
        Path to the directory containing .npy trajectory files.
    mechanic_prob : float
        Probability of selecting the mechanic setter (0.0 to 1.0).
    tick_skip : int
        Action repeat frames (default 8).
    episode_seconds : float
        Episode timeout in seconds.
    obs_size : int
        Observation dimension (must match the pretrained policy).
    num_actions : int
        Number of discrete actions (default 90 for LookupTableAction).
    opponent_checkpoint : str, optional
        Path to frozen opponent checkpoint. If None, uses idle opponent.
    opponent_device : str
        PyTorch device for the opponent model.
    fps : int
        Frame rate of extracted replays (default 30).
    pre_mechanic_seconds : float
        Seconds of pre-mechanic trajectory to keep (default 1.5).

    Returns
    -------
    SelfPlayEnv
        A gym.Env exposing single-agent interface for SB3.
    """
    # ── 1. Build state setters ──────────────────────────────────────
    mechanic_setter = MechanicTrajectorySetter(
        data_dir=data_dir,
        mechanic_name=mechanic_name,
        fps=fps,
        pre_mechanic_seconds=pre_mechanic_seconds,
    )

    normal_setter = KickoffMutator()

    mixed_setter = MixedStateSetter(
        setters=[normal_setter, mechanic_setter],
        probabilities=[1.0 - mechanic_prob, mechanic_prob],
        names=["normal", mechanic_name],
    )

    # ── 2. Build the rlgym environment ──────────────────────────────
    obs_builder = DefaultObs(
        zero_padding=None,
        pos_coef=np.asarray(
            [
                1 / common_values.SIDE_WALL_X,
                1 / common_values.BACK_NET_Y,
                1 / common_values.CEILING_Z,
            ],
            dtype=np.float32,
        ),
        ang_coef=1 / np.pi,
        lin_vel_coef=1 / common_values.CAR_MAX_SPEED,
        ang_vel_coef=1 / common_values.CAR_MAX_ANG_VEL,
        boost_coef=1 / 100.0,
    )

    # State mutator: fix team sizes first, then apply our mixed setter
    state_mutator = MutatorSequence(
        FixedTeamSizeMutator(blue_size=1, orange_size=1),
        mixed_setter,
    )

    rlgym_env = RLGym(
        state_mutator=state_mutator,
        obs_builder=obs_builder,
        action_parser=RepeatAction(LookupTableAction(), repeats=tick_skip),
        reward_fn=build_mixed_reward(mechanic_name=mechanic_name),
        termination_cond=GoalCondition(),
        truncation_cond=AnyCondition(
            TimeoutCondition(timeout_seconds=episode_seconds),
        ),
        transition_engine=RocketSimEngine(),
    )

    # ── 3. Build the opponent ───────────────────────────────────────
    if opponent_checkpoint and os.path.isfile(opponent_checkpoint):
        opponent_fn = make_frozen_opponent(
            checkpoint_path=opponent_checkpoint,
            device=opponent_device,
        )
    else:
        opponent_fn = make_idle_opponent()

    # ── 4. Wrap in SelfPlayEnv ──────────────────────────────────────
    env = SelfPlayEnv(
        rlgym_env=rlgym_env,
        opponent_fn=opponent_fn,
        obs_size=obs_size,
        num_actions=num_actions,
        mechanic_name=mechanic_name,
        mechanic_setter=mechanic_setter,
        mixed_setter=mixed_setter,
    )

    return env


class MixedEnvFactory:
    """
    Pickle-safe callable for creating mixed training envs.

    Use with SB3's ``SubprocVecEnv``:
    >>> vec_env = SubprocVecEnv([MixedEnvFactory(...) for _ in range(n_proc)])
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self) -> SelfPlayEnv:
        return make_mixed_env(**self.kwargs)
