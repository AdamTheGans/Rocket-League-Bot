# src/envs/mixed_training_env.py
"""
Environment factory for mixed mechanic + normal training.

Creates a native 1v1 RLGym env that alternates between mechanic-specific resets
(from extracted replay trajectories) and normal gameplay resets (kickoffs).
"""
from __future__ import annotations

import numpy as np
import RocketSim as rs  # Added for the Oracle

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
from rlgym.api import DoneCondition, RewardFunction

from state_setters.trajectory_setter import MechanicTrajectorySetter
from state_setters.mixed_state_setter import MixedStateSetter
from rewards.mixed_reward import build_mixed_reward
from rlgym_ppo.util import RLGymV2GymWrapper

class OracleTimeoutCondition(DoneCondition):
    """
    The Oracle: Replaces the standard timeout.
    At exactly the timeout limit (e.g., 2.5s), it freezes the game state,
    puts the ball into a headless RocketSim Arena, and fast-forwards for
    up to 3.5 seconds to see if the current trajectory will score.
    
    If it scores, it logs it in shared_info so Dense goal rewards can trigger.
    """
    def __init__(self, mechanic_name: str, normal_seconds: float = 15.0, mechanic_seconds: float = 2.5, oracle_seconds: float = 3.5):
        super().__init__()
        self.mechanic_name = mechanic_name
        self.normal_timeout = normal_seconds * 120
        self.mechanic_timeout = mechanic_seconds * 120
        self.oracle_ticks = int(oracle_seconds * 120)
        self.steps = 0
        self.current_limit = self.normal_timeout
        
        # We delay creating the Arena until it's needed so RocketSimEngine 
        # has time to extract the collision meshes first!
        self.oracle = None

    def reset(self, agents, initial_state: GameState, shared_info: dict, *args, **kwargs) -> None:
        self.steps = 0
        self.last_tick = initial_state.tick_count
        shared_info["oracle_scorer"] = None
        
        # Check if the episode is a mechanic spawn via shared_info
        if shared_info.get("setter_type") == self.mechanic_name:
            self.current_limit = self.mechanic_timeout
        else:
            self.current_limit = self.normal_timeout

    def is_done(self, agents, state: GameState, shared_info: dict, *args, **kwargs) -> dict:
        self.steps += state.tick_count - self.last_tick
        self.last_tick = state.tick_count
        
        is_timeout = self.steps >= self.current_limit
        
        # If we timed out AND this is a short mechanic episode, trigger the Oracle!
        if is_timeout and self.current_limit == self.mechanic_timeout:
            # Lazy initialization of the Ghost Arena
            if self.oracle is None:
                self.oracle = rs.Arena(rs.GameMode.SOCCAR)
                
            # Setup the Oracle State
            ball_state = rs.BallState()
            
            # Copy position
            ball_state.pos = rs.Vec(
                state.ball.position[0],
                state.ball.position[1],
                state.ball.position[2]
            )
            
            # Copy velocity
            ball_state.vel = rs.Vec(
                state.ball.linear_velocity[0],
                state.ball.linear_velocity[1],
                state.ball.linear_velocity[2]
            )
            
            # Copy angular velocity
            ball_state.ang_vel = rs.Vec(
                state.ball.angular_velocity[0],
                state.ball.angular_velocity[1],
                state.ball.angular_velocity[2]
            )
            
            # Apply state to the internal RocketSim engine ball
            self.oracle.ball.set_state(ball_state)
            
            self.oracle.set_goal_score_callback(lambda arena, team, *args, **kwargs: setattr(self, '_oracle_scored_team', team))
            self._oracle_scored_team = None
            
            # Fast forward!
            for _ in range(self.oracle_ticks):
                self.oracle.step(1)
                
                # If a goal was scored, break early and record it
                if self._oracle_scored_team is not None:
                    shared_info["oracle_scorer"] = int(self._oracle_scored_team)
                    break
                    
        return {agent: is_timeout for agent in agents}


# Add this new class above build_env
class CurriculumRewardWrapper(RewardFunction):
    """
    Wraps the reward function to capture BOTH terminations (goals) and truncations (timeouts),
    extracting metrics from shared_info before the RL agent steps.
    """
    def __init__(self, base_reward: RewardFunction, mechanic_name: str, curriculum):
        super().__init__()
        self.base_reward = base_reward
        self.mechanic_name = mechanic_name
        self.curriculum = curriculum
        self.steps = 0

    def reset(self, agents, initial_state: GameState, shared_info: dict, *args, **kwargs) -> None:
        self.base_reward.reset(agents, initial_state, shared_info, *args, **kwargs)
        shared_info["max_ball_speed"] = 0.0
        self.steps = 0
        self.episode_sub_rewards = {agent: {} for agent in agents}

    def get_rewards(self, agents, state: GameState, is_terminated: dict, is_truncated: dict, shared_info: dict, *args, **kwargs) -> dict:
        rewards = self.base_reward.get_rewards(agents, state, is_terminated, is_truncated, shared_info, *args, **kwargs)
        self.steps += 1
        
        # Accumulate sub-rewards from the active CombinedRewardWrapper
        if hasattr(self.base_reward, "last_rewards"):
            last_rews = self.base_reward.last_rewards
            for agent in agents:
                for r_name, r_val in last_rews.get(agent, {}).items():
                    self.episode_sub_rewards[agent][r_name] = self.episode_sub_rewards[agent].get(r_name, 0.0) + r_val
        
        ball_speed = float(np.linalg.norm(state.ball.linear_velocity))
        if ball_speed > shared_info.get("max_ball_speed", 0.0):
            shared_info["max_ball_speed"] = ball_speed
            
        done = any(is_terminated.values()) or any(is_truncated.values())
        if done and self.curriculum is not None:
            stype = shared_info.get("setter_type", "normal")
            
            if stype == self.mechanic_name:
                # Did Blue score? Check physically OR via oracle
                ball_y = state.ball.position[1]
                physical_blue_scored = (ball_y > common_values.BACK_NET_Y - 200)
                oracle_blue_scored = (shared_info.get("oracle_scorer") == rs.Team.BLUE)
                
                # Assume agent[0] is Blue for the sub_rewards
                metrics = {
                    "setter_type": stype,
                    "success": physical_blue_scored or oracle_blue_scored,
                    "max_ball_speed": shared_info["max_ball_speed"],
                    "ball_speed_at_end": ball_speed,
                    "episode_length": self.steps,
                    "sub_rewards": self.episode_sub_rewards.get(agents[0], {})
                }
            else:
                metrics = {
                    "setter_type": stype,
                    "episode_length": self.steps,
                    "sub_rewards": self.episode_sub_rewards.get(agents[0], {})
                }
            
            self.curriculum.outcomes_queue.put(metrics)
                
        return rewards


def build_env(
    mechanic_name: str = "kuxir",
    data_dir: str = "../extracted_mechanics",
    mechanic_prob: float = 1.0,
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
        curriculum=curriculum
    )

    normal_setter = KickoffMutator()

    mixed_setter = MixedStateSetter(
        setters=[normal_setter, mechanic_setter],
        probabilities=[1.0 - mechanic_prob, mechanic_prob],
        names=["normal", mechanic_name],
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

    base_termination = GoalCondition()

    base_reward = build_mixed_reward(mechanic_name=mechanic_name)
    reward_fn = CurriculumRewardWrapper(base_reward, mechanic_name, curriculum)

    env = RLGym(
        state_mutator=state_mutator,
        obs_builder=obs_builder,
        action_parser=RepeatAction(LookupTableAction(), repeats=tick_skip),
        reward_fn=reward_fn,
        termination_cond=base_termination,
        truncation_cond=OracleTimeoutCondition(mechanic_name, 15.0, 2.5, 3.5),
        transition_engine=RocketSimEngine(),
    )

    return RLGymV2GymWrapper(env)