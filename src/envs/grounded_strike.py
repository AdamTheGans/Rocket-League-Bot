# src/envs/grounded_strike.py
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

from rewards.strike_reward import build_strike_reward
from state_setters.low_spawn_setter import LowGroundSpawnMutator


def build_env(render: bool = False, tick_skip: int = 8, episode_seconds: float = 12.0):
    """
    RLGym v2 + RocketSim 1v0 environment for the grounded strike specialist.

    Uses tick_skip=8 (the RLGym standard) for action repeat.
    """
    spawn_opponents = False
    blue_team_size = 1
    orange_team_size = 0 if not spawn_opponents else 1

    action_parser = RepeatAction(LookupTableAction(), repeats=int(tick_skip))

    # Terminate on goal; truncate on timeout
    termination_condition = GoalCondition()
    truncation_condition = AnyCondition(
        TimeoutCondition(timeout_seconds=float(episode_seconds))
    )

    # Observation builder — identical to the official RLGym examples
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

    # Ensure correct team sizes, then apply our custom spawn mutator
    state_mutator = MutatorSequence(
        FixedTeamSizeMutator(blue_size=blue_team_size, orange_size=orange_team_size),
        LowGroundSpawnMutator(easy_prob=0.70, boost_min=40.0, boost_max=100.0),
    )

    rlgym_env = RLGym(
        state_mutator=state_mutator,
        obs_builder=obs_builder,
        action_parser=action_parser,
        reward_fn=build_strike_reward(),
        termination_cond=termination_condition,
        truncation_cond=truncation_condition,
        transition_engine=RocketSimEngine(),
    )

    return RLGymV2GymWrapper(rlgym_env)