# src/envs/mixed_training_env.py
"""
Environment factory for mixed mechanic + normal training.

Creates a native 1v1 RLGym env that alternates between mechanic-specific resets
(from extracted replay trajectories) and normal gameplay resets (kickoffs).
"""
from __future__ import annotations

import numpy as np
import rocketsim as rs  # Added for the Oracle

from rlgym.api import RLGym
from rlgym.rocket_league.sim import RocketSimEngine
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.action_parsers import LookupTableAction, RepeatAction
from rlgym.rocket_league.done_conditions import GoalCondition, AnyCondition
from rlgym.rocket_league.obs_builders import DefaultObs
from rlgym.rocket_league.state_mutators import (
    MutatorSequence,
    FixedTeamSizeMutator,
    KickoffMutator,
)
from rlgym.rocket_league import common_values
from rlgym.api import DoneCondition

from state_setters.trajectory_setter import MechanicTrajectorySetter
from state_setters.mixed_state_setter import MixedStateSetter
from rewards.mixed_reward import build_mixed_reward


class DynamicTimeoutCondition(DoneCondition):
    """Gives 15s for normal play, but only 2.5s for mechanics to force speed."""
    def __init__(self, normal_seconds: float = 15.0, mechanic_seconds: float = 2.5):
        super().__init__()
        self.normal_timeout = normal_seconds * 120
        self.mechanic_timeout = mechanic_seconds * 120
        self.steps = 0
        self.current_limit = self.normal_timeout

    def reset(self, initial_state: GameState, *args, **kwargs) -> None:
        self.steps = 0
        # If the ball starts on the wall (X > 3000), it's a Kuxir setup
        if abs(initial_state.ball.position[0]) > 3000:
            self.current_limit = self.mechanic_timeout
        else:
            self.current_limit = self.normal_timeout

    def step(self, state: GameState, *args, **kwargs) -> bool:
        self.steps += state.tick_count - state.previous_tick_count
        return self.steps >= self.current_limit


class FastForwardGoalCondition(DoneCondition):
    """
    The Oracle: Peeks into the future using a ghost RocketSim arena.
    Terminates the episode early if the ball is mathematically guaranteed to go in.
    """
    def __init__(self, min_speed=1500.0, max_seconds=2.5):
        super().__init__()
        self.min_speed = min_speed
        self.max_steps = int(max_seconds * 120)  # 120 ticks per second
        
        # Initialize a hidden, car-less arena purely for ball physics
        self.oracle = rs.Arena(rs.GameMode.SOCCAR)

    def reset(self, initial_state: GameState, shared_info: dict, *args, **kwargs) -> None:
        shared_info["fast_forward_scorer"] = None

    def step(self, state: GameState, shared_info: dict, *args, **kwargs) -> bool:
        ball_vel = state.ball.linear_velocity
        ball_speed = np.linalg.norm(ball_vel)

        # 1. Cheap check: Is the ball moving fast enough to be worth simulating?
        if ball_speed < self.min_speed:
            return False

        # 2. Setup the Oracle (sync ball state only)
        ball_state = self.oracle.ball.get_state()
        ball_state.pos = rs.Vec3(state.ball.position[0], state.ball.position[1], state.ball.position[2])
        ball_state.vel = rs.Vec3(ball_vel[0], ball_vel[1], ball_vel[2])
        ball_state.ang_vel = rs.Vec3(state.ball.angular_velocity[0], state.ball.angular_velocity[1], state.ball.angular_velocity[2])
        self.oracle.ball.set_state(ball_state)

        # 3. Step into the future
        for _ in range(self.max_steps):
            self.oracle.step(1)
            
            # 4. Check for a goal
            current_y = self.oracle.ball.get_state().pos.y
            if current_y > common_values.BACK_NET_Y:
                shared_info["fast_forward_scorer"] = 0  # Blue scored
                return True
            elif current_y < -common_values.BACK_NET_Y:
                shared_info["fast_forward_scorer"] = 1  # Orange scored
                return True

        return False


# Add this new class above build_env
class CurriculumDoneWrapper(DoneCondition):
    """
    Wraps your existing done conditions to evaluate if the episode was a success,
    writing the result to shared_info for the MetricsLogger to read.
    """
    def __init__(self, base_condition: DoneCondition, mechanic_name: str):
        super().__init__()
        self.base_condition = base_condition
        self.mechanic_name = mechanic_name
        self.success_key = f"{mechanic_name}_success"

    def reset(self, initial_state: GameState, shared_info: dict, *args, **kwargs) -> None:
        self.base_condition.reset(initial_state, shared_info, *args, **kwargs)
        # Default to False at the start of the episode
        shared_info[self.success_key] = False

    def step(self, state: GameState, shared_info: dict, *args, **kwargs) -> bool:
        done = self.base_condition.step(state, shared_info, *args, **kwargs)
        
        if done:
            ball_y = state.ball.position[1]
            ff_scorer = shared_info.get("fast_forward_scorer", None)
            
            # Did Blue score? (Assuming the bot is playing Blue for mechanic setups)
            blue_scored = (ball_y > common_values.BACK_NET_Y - 200) or (ff_scorer == 0)
            if blue_scored:
                shared_info[self.success_key] = True
                
        return done


def build_env(
    mechanic_name: str = "kuxir",
    data_dir: str = "../extracted_mechanics",
    mechanic_prob: float = 0.4,
    tick_skip: int = 8,
    fps: int = 30,
    pre_mechanic_seconds: float = 1.5,
    curriculum=None,
) -> RLGym:
    """
    Build a native 1v1 mixed-training environment for rlgym-ppo.
    """
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

    mechanic_setter = MechanicTrajectorySetter(
        data_dir=data_dir,
        mechanic_name=mechanic_name,
        fps=fps,
        pre_mechanic_seconds=pre_mechanic_seconds,
        curriculum=curriculum
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
        FixedTeamSizeMutator(blue_size=1, orange_size=1),
        mixed_setter,
    )

    base_termination = AnyCondition(
        GoalCondition(),
        FastForwardGoalCondition(min_speed=1500.0, max_seconds=2.5)
    )

    env = RLGym(
        state_mutator=state_mutator,
        obs_builder=obs_builder,
        action_parser=RepeatAction(LookupTableAction(), repeats=tick_skip),
        reward_fn=build_mixed_reward(mechanic_name=mechanic_name),
        termination_cond=CurriculumDoneWrapper(base_termination, mechanic_name),
        truncation_cond=DynamicTimeoutCondition(15.0, 2.5),
        transition_engine=RocketSimEngine(),
    )

    return env