# src/rewards/strike_reward.py
"""
Power-shot focused rewards for the grounded-strike specialist.

Design evolution:
  v1: SpeedTowardBall only → drove past ball to goal
  v2: FaceBall + CloseToBall → backed away while facing ball
  v3: Event-dominant (Touch=10) → dribble exploit (15K touches/iter)
  v4: Option B — BallGoalProgress + minimal Touch + strong TimePenalty
"""
from __future__ import annotations

import numpy as np
from typing import Any, Dict, List

from rlgym.api import AgentID, RewardFunction
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league import common_values

from rlgym.rocket_league.reward_functions import CombinedReward, TouchReward


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


class BallGoalProgressReward(RewardFunction[AgentID, GameState, float]):
    """
    Rewards decrease in ball-to-goal distance between steps.

    reward = (prev_dist - curr_dist) / NORMALIZER

    Positive when ball moves closer to goal, negative when it moves away.
    Normalized by BACK_NET_Y (~5120) so max per-step reward ≈ 0.1–0.3
    at typical ball speeds. NOT clamped — the agent is penalized for
    hitting the ball away from goal.

    This is the key "alignment" reward: unlike BallVelocityToGoal (which
    rewards ball speed), this directly rewards PROGRESS regardless of
    speed, making every step's contribution clear.
    """
    NORMALIZER = common_values.BACK_NET_Y  # ~5120 uu

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        self._prev_dist: Dict[AgentID, float] = {}
        ball = initial_state.ball
        for agent in agents:
            car = initial_state.cars[agent]
            goal_y = -common_values.BACK_NET_Y if car.is_orange else common_values.BACK_NET_Y
            goal = np.array([0, goal_y, 0], dtype=np.float32)
            self._prev_dist[agent] = float(np.linalg.norm(goal - ball.position))

    def get_rewards(
        self,
        agents: List[AgentID],
        state: GameState,
        is_terminated: Dict[AgentID, bool],
        is_truncated: Dict[AgentID, bool],
        shared_info: Dict[str, Any],
    ) -> Dict[AgentID, float]:
        rewards = {}
        ball = state.ball
        for agent in agents:
            car = state.cars[agent]
            goal_y = -common_values.BACK_NET_Y if car.is_orange else common_values.BACK_NET_Y
            goal = np.array([0, goal_y, 0], dtype=np.float32)
            curr_dist = float(np.linalg.norm(goal - ball.position))

            prev = self._prev_dist.get(agent, curr_dist)
            delta = prev - curr_dist  # positive = ball moved closer to goal
            rewards[agent] = delta / self.NORMALIZER

            self._prev_dist[agent] = curr_dist
        return rewards


class QuickGoalReward(RewardFunction[AgentID, GameState, float]):
    """
    Goal reward with a speed bonus: faster goals get more reward.

    reward = base + bonus * (1 - steps_elapsed / max_steps)

    A goal at step 10/180 gets: base + bonus * 0.94 ≈ base + bonus
    A goal at step 170/180 gets: base + bonus * 0.056 ≈ base
    No goal gets: 0

    This directly incentivizes fast, decisive scoring over dithering.
    """
    def __init__(self, base: float = 1.0, bonus: float = 1.0):
        super().__init__()
        self.base = base
        self.bonus = bonus
        self._step_count: Dict[AgentID, int] = {}
        self._max_steps: int = 180  # 12s * 15 steps/s (tick_skip=8)

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        for agent in agents:
            self._step_count[agent] = 0

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
            self._step_count[agent] = self._step_count.get(agent, 0) + 1

            if is_terminated.get(agent, False) and getattr(state, "goal_scored", False):
                frac_remaining = max(0, 1.0 - self._step_count[agent] / self._max_steps)
                rewards[agent] = self.base + self.bonus * frac_remaining
            else:
                rewards[agent] = 0.0
        return rewards


def build_strike_reward():
    """
    Option B: Power-shot focused reward with BallGoalProgress.

    Design:
      - QuickGoalReward (100 + 50 speed bonus): fast scoring is king
      - BallGoalProgress (0.4): dense signal rewarding ball approaching goal
      - BallVelocityToGoal (0.5): rewards hard hits toward goal
      - TouchReward (0.1): exploration only
      - SpeedTowardBall (0.1): gentle guide toward ball
      - TimePenalty (-0.05): strong urgency

    Per-episode budget (180 steps, no goal):
      BallGoalProgress: ≈ ±5 (depends on ball movement)
      BallVelToGoal:    0.5 * 180 = 90 max (only when ball flying)
      SpeedTowardBall:  0.1 * 180 = 18 max
      TouchReward:      0.1 * N_touches ≈ 1-2
      TimePenalty:     -0.05 * 180 = -9.0

    With goal (at step 50): +100 base +36 speed bonus = 136
    """
    return CombinedReward(
        (QuickGoalReward(base=1.0, bonus=0.5), 100.0),  # 100-150 depending on speed
        (TouchReward(), 0.1),                             # exploration only
        (BallGoalProgressReward(), 0.4),                  # dense aligned signal
        (BallVelocityToGoalReward(), 0.5),                # reward ball flying to goal
        (SpeedTowardBallReward(), 0.1),                   # gentle guide
        (TimePenalty(), -0.05),                            # strong urgency
    )