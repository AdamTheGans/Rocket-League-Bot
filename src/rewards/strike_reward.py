# src/rewards/strike_reward.py
from __future__ import annotations

import numpy as np
from typing import Any, Dict, List

from rlgym.api import AgentID, RewardFunction
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league import common_values

from rlgym.rocket_league.reward_functions import CombinedReward, GoalReward, TouchReward


class TimePenalty(RewardFunction[AgentID, GameState, float]):
    """Return +1 each step; we weight it negative to penalize time."""
    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(
        self,
        agents: List[AgentID],
        state: GameState,
        is_terminated: Dict[AgentID, bool],
        is_truncated: Dict[AgentID, bool],
        shared_info: Dict[str, Any],
    ) -> Dict[AgentID, float]:
        return {a: 1.0 for a in agents}


class BallVelocityToGoalReward(RewardFunction[AgentID, GameState, float]):
    """
    Ball velocity component toward the opponent's goal.
    Normalized by BALL_MAX_SPEED, clamped to [0, 1].
    Self-limiting: only fires when ball is moving, and fast-moving
    balls score quickly (ending the episode with GoalReward).
    """
    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(
        self,
        agents: List[AgentID],
        state: GameState,
        is_terminated: Dict[AgentID, bool],
        is_truncated: Dict[AgentID, bool],
        shared_info: Dict[str, Any],
    ) -> Dict[AgentID, float]:
        rewards = {}
        for agent in agents:
            car = state.cars[agent]
            ball = state.ball
            goal_y = -common_values.BACK_NET_Y if car.is_orange else common_values.BACK_NET_Y
            pos_diff = np.array([0, goal_y, 0], dtype=np.float32) - ball.position
            dist = np.linalg.norm(pos_diff)
            if dist < 1e-6:
                rewards[agent] = 0.0
                continue
            dir_to_goal = pos_diff / dist
            vel_toward_goal = float(np.dot(ball.linear_velocity, dir_to_goal))
            rewards[agent] = max(vel_toward_goal / common_values.BALL_MAX_SPEED, 0.0)
        return rewards


class SpeedTowardBallReward(RewardFunction[AgentID, GameState, float]):
    """
    Car velocity component toward the ball, clamped to [0, 1].
    Kept very light — just enough to guide exploration, not enough
    to be worth gaming.
    """
    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(
        self,
        agents: List[AgentID],
        state: GameState,
        is_terminated: Dict[AgentID, bool],
        is_truncated: Dict[AgentID, bool],
        shared_info: Dict[str, Any],
    ) -> Dict[AgentID, float]:
        rewards = {}
        for agent in agents:
            car = state.cars[agent]
            car_physics = car.inverted_physics if car.is_orange else car.physics
            ball_physics = state.inverted_ball if car.is_orange else state.ball
            pos_diff = ball_physics.position - car_physics.position
            dist = np.linalg.norm(pos_diff)
            if dist < 1e-6:
                rewards[agent] = 0.0
                continue
            dir_to_ball = pos_diff / dist
            speed_toward = float(np.dot(car_physics.linear_velocity, dir_to_ball))
            rewards[agent] = max(speed_toward / common_values.CAR_MAX_SPEED, 0.0)
        return rewards


def build_strike_reward():
    """
    Event-dominant reward for the grounded-strike specialist.

    Design principle: no amount of per-step shaping exploitation should
    outweigh a single touch or goal. This prevents the agent from
    finding degenerate strategies (driving past ball, reversing while
    facing ball, etc.).

    Per-episode budget at max shaping (all 180 steps):
      SpeedTowardBall:  0.1 * 180 = 18     ← light guide
      BallVelToGoal:    1.0 * 180 = 180    ← only fires when ball moves
      TimePenalty:     -0.01 * 180 = -1.8   ← urgency

    Event rewards (one-time bursts):
      TouchReward:      10 per touch        ← worth 100 steps of shaping
      GoalReward:       100 per goal        ← worth 10 touches
    """
    return CombinedReward(
        (GoalReward(), 100.0),                   # scoring is king
        (TouchReward(), 10.0),                    # each touch > 50 steps of shaping
        (BallVelocityToGoalReward(), 0.1),        # light; self-limiting (ball → goal)
        (SpeedTowardBallReward(), 0.1),           # whisper-quiet guide
        (TimePenalty(), -0.01),                    # moderate urgency
    )