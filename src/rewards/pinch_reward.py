# src/rewards/pinch_reward.py
"""
Pinch-specific reward function for wall/corner pinch specialist.

Key design: GoalwardSpeedSpikeReward measures the *increase in goalward ball
speed* between steps, not raw speed delta.  This specifically rewards pinch
contacts that redirect the ball toward the opponent's goal.
"""
from __future__ import annotations

import numpy as np
from typing import Any, Dict, List

from rlgym.api import AgentID, RewardFunction
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league import common_values

from rlgym.rocket_league.reward_functions import CombinedReward, TouchReward

# Import shared components from the strike reward module
from rewards.strike_reward import (
    QuickGoalReward,
    BallVelocityToGoalReward,
    TimePenalty,
)


# ─── Pinch-specific reward components ───────────────────────────────────────


class GoalwardSpeedSpikeReward(RewardFunction[AgentID, GameState, float]):
    """
    Reward positive increases in the ball's goalward velocity component.

    Each step computes:
        curr_goalward = ball_vel · goal_dir
        delta = curr_goalward - prev_goalward
        reward = max(0, delta) / BALL_MAX_SPEED   (if curr_goalward > 0)

    This specifically rewards pinch contacts that send the ball *toward* the
    goal, ignoring sideways or backwards speed spikes.  The curr_goalward > 0
    gate ensures we don't reward minor rebounds in the wrong direction.
    """

    def __init__(self):
        super().__init__()
        self._prev_goalward: Dict[AgentID, float] = {}

    def reset(
        self,
        agents: List[AgentID],
        initial_state: GameState,
        shared_info: Dict[str, Any],
    ) -> None:
        ball = initial_state.ball
        for agent in agents:
            car = initial_state.cars[agent]
            goal_dir = self._goal_direction(ball.position, car.is_orange)
            goalward = float(np.dot(ball.linear_velocity, goal_dir))
            self._prev_goalward[agent] = goalward

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
            goal_dir = self._goal_direction(ball.position, car.is_orange)
            curr_goalward = float(np.dot(ball.linear_velocity, goal_dir))

            prev = self._prev_goalward.get(agent, curr_goalward)
            delta = curr_goalward - prev

            # Only reward positive goalward increases when ball is moving goalward
            if delta > 0 and curr_goalward > 0:
                rewards[agent] = delta / common_values.BALL_MAX_SPEED
            else:
                rewards[agent] = 0.0

            self._prev_goalward[agent] = curr_goalward
        return rewards

    @staticmethod
    def _goal_direction(ball_pos: np.ndarray, is_orange: bool) -> np.ndarray:
        """Unit vector from ball toward the opponent's goal."""
        goal_y = -common_values.BACK_NET_Y if is_orange else common_values.BACK_NET_Y
        goal = np.array([0.0, goal_y, 0.0], dtype=np.float32)
        diff = goal - ball_pos
        dist = np.linalg.norm(diff)
        if dist < 1e-6:
            return np.array([0.0, 1.0, 0.0], dtype=np.float32)
        return diff / dist


class BallWallProximityReward(RewardFunction[AgentID, GameState, float]):
    """
    Reward ball being close to a side wall (where pinches happen).

    reward = 1.0 - (dist_to_nearest_wall / SIDE_WALL_X), clamped [0, 1]
    Only active in offensive half (ball_y > 0 for blue, ball_y < 0 for orange).
    """

    def reset(self, agents: List[AgentID], initial_state: GameState,
              shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(
        self,
        agents: List[AgentID],
        state: GameState,
        is_terminated: Dict[AgentID, bool],
        is_truncated: Dict[AgentID, bool],
        shared_info: Dict[str, Any],
    ) -> Dict[AgentID, float]:
        ball_x = float(state.ball.position[0])
        ball_y = float(state.ball.position[1])
        dist_to_wall = common_values.SIDE_WALL_X - abs(ball_x)
        proximity = max(0.0, 1.0 - dist_to_wall / common_values.SIDE_WALL_X)

        rewards = {}
        for agent in agents:
            car = state.cars[agent]
            # Only reward in offensive half
            in_offensive = (ball_y > 0 and not car.is_orange) or \
                           (ball_y < 0 and car.is_orange)
            rewards[agent] = proximity if in_offensive else 0.0
        return rewards


class ApproachPinchPointReward(RewardFunction[AgentID, GameState, float]):
    """
    Reward car velocity toward the estimated pinch point (wall contact zone).

    Pinch point = projection of ball onto the nearest side wall.
    reward = max(0, car_vel · dir_to_pinch) / CAR_MAX_SPEED, clamped [0, 1]
    """

    def reset(self, agents: List[AgentID], initial_state: GameState,
              shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(
        self,
        agents: List[AgentID],
        state: GameState,
        is_terminated: Dict[AgentID, bool],
        is_truncated: Dict[AgentID, bool],
        shared_info: Dict[str, Any],
    ) -> Dict[AgentID, float]:
        ball_pos = state.ball.position
        ball_x = float(ball_pos[0])

        # Nearest side wall
        if ball_x >= 0:
            wall_x = common_values.SIDE_WALL_X
        else:
            wall_x = -common_values.SIDE_WALL_X

        # Pinch point = ball projected onto wall plane
        pinch_point = np.array(
            [float(wall_x), float(ball_pos[1]), float(ball_pos[2])],
            dtype=np.float32,
        )

        rewards = {}
        for agent in agents:
            car = state.cars[agent]
            car_pos = car.physics.position
            car_vel = car.physics.linear_velocity

            diff = pinch_point - car_pos
            dist = float(np.linalg.norm(diff))
            if dist < 1e-6:
                rewards[agent] = 0.0
                continue

            dir_to_pinch = diff / dist
            speed_toward = float(np.dot(car_vel, dir_to_pinch))
            rewards[agent] = max(0.0, speed_toward / common_values.CAR_MAX_SPEED)
        return rewards


# ─── Factory ─────────────────────────────────────────────────────────────────


def build_pinch_reward(stage: int = 1) -> CombinedReward:
    """
    Build stage-dependent pinch reward.

    Stage 1: heavy spike weight, small dense shaping
    Stage 2: approach shaping added
    Stage 3: full shaping, emphasis on goalward ball velocity
    """
    if stage == 1:
        return CombinedReward(
            (QuickGoalReward(base=1.0, bonus=0.5), 100.0),
            (GoalwardSpeedSpikeReward(),             8.0),
            (BallVelocityToGoalReward(),             0.3),
            (BallWallProximityReward(),               0.05),
            (ApproachPinchPointReward(),              0.05),
            (TouchReward(),                           0.1),
            (TimePenalty(),                           -0.03),
        )
    elif stage == 2:
        return CombinedReward(
            (QuickGoalReward(base=1.0, bonus=0.5), 100.0),
            (GoalwardSpeedSpikeReward(),             4.0),
            (BallVelocityToGoalReward(),             0.4),
            (BallWallProximityReward(),               0.1),
            (ApproachPinchPointReward(),              0.2),
            (TouchReward(),                           0.05),
            (TimePenalty(),                           -0.04),
        )
    elif stage == 3:
        return CombinedReward(
            (QuickGoalReward(base=1.0, bonus=0.5), 100.0),
            (GoalwardSpeedSpikeReward(),             3.0),
            (BallVelocityToGoalReward(),             0.5),
            (BallWallProximityReward(),               0.1),
            (ApproachPinchPointReward(),              0.3),
            (TouchReward(),                           0.05),
            (TimePenalty(),                           -0.05),
        )
    else:
        raise ValueError(f"stage must be 1, 2, or 3, got {stage}")
