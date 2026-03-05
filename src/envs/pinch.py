# src/envs/pinch.py
"""
RLGym v2 + RocketSim 1v0 environment for the wall/corner pinch specialist.
"""
from __future__ import annotations

import numpy as np

from rlgym.api import RLGym
from rlgym.rocket_league.sim import RocketSimEngine
from rlgym.rocket_league.action_parsers import LookupTableAction, RepeatAction
from rlgym.rocket_league.done_conditions import GoalCondition, TimeoutCondition, AnyCondition
from rlgym.rocket_league.obs_builders import DefaultObs
from rlgym.rocket_league.state_mutators import MutatorSequence, FixedTeamSizeMutator
from rlgym.rocket_league import common_values

from rlgym_ppo.util import RLGymV2GymWrapper

from rewards.pinch_reward import build_pinch_reward, build_golden_seed_reward
from state_setters.pinch_golden_seed_setter import PinchGoldenSeedSetter


def build_env(render: bool = False, tick_skip: int = 8, stage: int = 1, difficulty_level: int = 1):
    """
    Build the pinch specialist environment.

    Parameters
    ----------
    render : bool
        Enable RLViser rendering.
    tick_skip : int
        Action repeat frames (default 8, the RLGym standard).
    stage : int
        Training stage (1, 2, or 3). Controls spawn distribution and timeout.
    difficulty_level : int
        Difficulty level for the Stage 1 Golden Seed Setting.
    """
    spawn_mutator = PinchGoldenSeedSetter(randomize=True, stage=stage, difficulty_level=difficulty_level)
    
    if stage == 1:
        episode_seconds = float(difficulty_level + 1.0)
    elif stage == 2:
        episode_seconds = 10.0
    else:
        episode_seconds = 10.0

    action_parser = RepeatAction(LookupTableAction(), repeats=int(tick_skip))

    termination_condition = GoalCondition()
    truncation_condition = AnyCondition(
        TimeoutCondition(timeout_seconds=float(episode_seconds))
    )

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

    state_mutator = MutatorSequence(
        FixedTeamSizeMutator(blue_size=1, orange_size=0),
        spawn_mutator,
    )

    rlgym_env = RLGym(
        state_mutator=state_mutator,
        obs_builder=obs_builder,
        action_parser=action_parser,
        reward_fn=build_pinch_reward(stage),
        termination_cond=termination_condition,
        truncation_cond=truncation_condition,
        transition_engine=RocketSimEngine(),
    )

    return RLGymV2GymWrapper(rlgym_env)
