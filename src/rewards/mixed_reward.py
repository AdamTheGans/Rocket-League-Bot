# src/rewards/mixed_reward.py
"""
Reward function that dispatches to mechanic-specific or normal rewards
based on the episode's setter type.

Uses a dictionary mapping mechanic names to reward function builders,
making it easy to extend for new trickshots without modifying this file.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import numpy as np

from rlgym.api import AgentID
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league import common_values


class MixedRewardFunction:
    """
    Dispatches reward computation based on ``shared_info["setter_type"]``.

    Parameters
    ----------
    mechanic_rewards : dict[str, RewardFunction]
        Maps mechanic names (e.g., ``"kuxir"``) to rlgym RewardFunction objects.
    default_reward : RewardFunction
        The reward function used for "normal" (non-mechanic) episodes.
    """

    def __init__(
        self,
        mechanic_rewards: Dict[str, Any],
        default_reward: Any,
    ):
        self.mechanic_rewards = dict(mechanic_rewards)
        self.default_reward = default_reward
        self._current_setter_type: str = "normal"
        self._active_reward = default_reward

    def reset(
        self,
        agents: List[AgentID],
        initial_state: GameState,
        shared_info: Dict[str, Any],
    ) -> None:
        """Select the appropriate reward function based on setter_type."""
        self._current_setter_type = shared_info.get("setter_type", "normal")

        # Pick the right reward function
        if self._current_setter_type in self.mechanic_rewards:
            self._active_reward = self.mechanic_rewards[self._current_setter_type]
        else:
            self._active_reward = self.default_reward

        # Reset the selected reward function
        self._active_reward.reset(agents, initial_state, shared_info)

    def get_rewards(
        self,
        agents: List[AgentID],
        state: GameState,
        is_terminated: Dict[AgentID, bool],
        is_truncated: Dict[AgentID, bool],
        shared_info: Dict[str, Any],
    ) -> Dict[AgentID, float]:
        """Delegate to the active reward function."""
        return self._active_reward.get_rewards(
            agents, state, is_terminated, is_truncated, shared_info
        )


# ─────────────────────────────────────────────────────────────────── #
#  Reward building blocks
# ─────────────────────────────────────────────────────────────────── #

class SimpleGoalReward:
    """
    Sparse: +goal_value for scoring, -goal_value for being scored on.
    """

    def __init__(self, goal_value: float = 10.0):
        self.goal_value = goal_value

    def reset(self, agents, initial_state, shared_info):
        pass

    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info):
        rewards = {}
        for agent_id in agents:
            car = state.cars[agent_id]
            if any(is_terminated.values()):
                ball_y = state.ball.position[1]
                if car.team_num == 0:  # Blue
                    rewards[agent_id] = self.goal_value if ball_y > 0 else -self.goal_value
                else:  # Orange
                    rewards[agent_id] = self.goal_value if ball_y < 0 else -self.goal_value
            else:
                rewards[agent_id] = 0.0
        return rewards


class SparseGoalSpeedReward:
    """
    Sparse: Massive payout for scoring, heavily scaled by ball speed.
    Fixes the velocity integration trap by only rewarding speed ONCE upon scoring.
    """

    def __init__(self, base_goal_value: float = 20.0, speed_multiplier: float = 30.0):
        self.base_goal_value = base_goal_value
        self.speed_multiplier = speed_multiplier

    def reset(self, agents, initial_state, shared_info):
        pass

    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info):
        rewards = {agent_id: 0.0 for agent_id in agents}
        
        if any(is_terminated.values()):
            ball_y = state.ball.position[1]
            # Calculate ball speed ratio (0.0 to 1.0)
            ball_speed = float(np.linalg.norm(state.ball.linear_velocity))
            speed_ratio = ball_speed / common_values.BALL_MAX_SPEED
            
            bonus = self.base_goal_value + (speed_ratio * self.speed_multiplier)

            for agent_id in agents:
                car = state.cars[agent_id]
                # Blue scored
                if car.team_num == 0 and ball_y > 0:
                    rewards[agent_id] = bonus
                # Orange scored
                elif car.team_num == 1 and ball_y < 0:
                    rewards[agent_id] = bonus
                # Scored upon
                else:
                    rewards[agent_id] = -self.base_goal_value
        return rewards


class VelocityTowardBallReward:
    """
    Dense: reward the agent for moving toward the ball.

    Encourages the bot to approach the ball during both normal play
    and mechanic setups.
    """

    def __init__(self, weight: float = 0.005):
        self.weight = weight

    def reset(self, agents, initial_state, shared_info):
        pass

    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info):
        rewards = {}
        ball_pos = state.ball.position
        for agent_id in agents:
            car = state.cars[agent_id]
            car_pos = car.physics.position
            car_vel = car.physics.linear_velocity

            to_ball = ball_pos - car_pos
            dist = np.linalg.norm(to_ball)
            if dist > 1e-6:
                to_ball_dir = to_ball / dist
                speed_toward = float(np.dot(car_vel, to_ball_dir))
                # Normalize by max car speed so reward is ~0-1 range
                rewards[agent_id] = max(0.0, speed_toward / common_values.CAR_MAX_SPEED) * self.weight
            else:
                rewards[agent_id] = 0.0
        return rewards


class BallVelocityToGoalReward:
    """
    Dense: reward absolute ball velocity toward the opponent's goal.
    (Kept for normal play, removed from Kuxir to prevent integration trap).
    """

    def __init__(self, weight: float = 0.01):
        self.weight = weight

    def reset(self, agents, initial_state, shared_info):
        pass

    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info):
        rewards = {}
        ball_vel = state.ball.linear_velocity
        ball_pos = state.ball.position

        for agent_id in agents:
            car = state.cars[agent_id]
            if car.team_num == 0:  # Blue attacks +Y
                goal = np.array([0.0, common_values.BACK_NET_Y, 0.0], dtype=np.float32)
            else:  # Orange attacks -Y
                goal = np.array([0.0, -common_values.BACK_NET_Y, 0.0], dtype=np.float32)

            diff = goal - ball_pos
            dist = float(np.linalg.norm(diff))
            if dist > 1e-6:
                goal_dir = diff / dist
                goalward_speed = float(np.dot(ball_vel, goal_dir))
                # Normalize by max ball speed so reward is ~0-1 range
                rewards[agent_id] = max(0.0, goalward_speed / common_values.BALL_MAX_SPEED) * self.weight
            else:
                rewards[agent_id] = 0.0
        return rewards


class TouchBallReward:
    """
    Sparse-ish: +touch_value each time the ball is touched.
    """

    # Fix: Changed touch_distance to 170.0 to match actual hitbox collision radii
    def __init__(self, touch_value: float = 1.0, touch_distance: float = 170.0):
        self.touch_value = touch_value
        self.touch_distance = touch_distance
        self._gave_touch: Dict[AgentID, bool] = {}

    def reset(self, agents, initial_state, shared_info):
        self._gave_touch = {aid: False for aid in agents}

    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info):
        rewards = {}
        ball_pos = state.ball.position
        for agent_id in agents:
            car = state.cars[agent_id]
            dist = float(np.linalg.norm(ball_pos - car.physics.position))
            if dist < self.touch_distance and not self._gave_touch.get(agent_id, False):
                rewards[agent_id] = self.touch_value
                self._gave_touch[agent_id] = True
            else:
                rewards[agent_id] = 0.0
        return rewards


class TickPenalty:
    """
    Dense: Applies a constant negative reward every step.
    Forces the bot to complete the episode (score) as fast as possible.
    """
    def __init__(self, penalty: float = 0.02):
        self.penalty = penalty

    def reset(self, agents, initial_state, shared_info):
        pass

    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info):
        return {agent_id: -self.penalty for agent_id in agents}


class TimeoutPenalty:
    """
    Sparse: -penalty_value when the episode is truncated (timeout).
    """

    def __init__(self, penalty_value: float = 1.0):
        self.penalty_value = penalty_value

    def reset(self, agents, initial_state, shared_info):
        pass

    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info):
        rewards = {}
        timed_out = any(is_truncated.values())
        for agent_id in agents:
            rewards[agent_id] = -self.penalty_value if timed_out else 0.0
        return rewards


class CombinedRewardWrapper:
    """
    Combine multiple reward functions with weights.
    Sums the rewards from all component functions.
    """

    def __init__(self, *reward_weight_pairs):
        self.components = list(reward_weight_pairs)

    def reset(self, agents, initial_state, shared_info):
        for reward_fn, _ in self.components:
            reward_fn.reset(agents, initial_state, shared_info)

    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info):
        combined = {agent_id: 0.0 for agent_id in agents}
        for reward_fn, weight in self.components:
            sub_rewards = reward_fn.get_rewards(agents, state, is_terminated, is_truncated, shared_info)
            for agent_id in agents:
                combined[agent_id] += sub_rewards.get(agent_id, 0.0) * weight
        return combined


# ─────────────────────────────────────────────────────────────────── #
#  Factory functions
# ─────────────────────────────────────────────────────────────────── #

def build_normal_reward() -> CombinedRewardWrapper:
    """Build the default reward function for normal self-play episodes."""
    return CombinedRewardWrapper(
        (SimpleGoalReward(goal_value=10.0), 1.0),
        (VelocityTowardBallReward(weight=0.005), 1.0),
        (BallVelocityToGoalReward(weight=0.01), 1.0),
    )


def build_kuxir_reward() -> CombinedRewardWrapper:
    """
    Build the reward function for Kuxir pinch mechanic episodes.

    Uses a time bleed to encourage speed, and a massive jackpot for high-velocity goals.
    """
    return CombinedRewardWrapper(
        # Dense approach: minor breadcrumbs to keep it engaged
        (VelocityTowardBallReward(weight=0.01), 1.0),
        
        # Dense penalty: Bleed reward every step so it wants to score FAST
        (TickPenalty(penalty=0.02), 1.0),
        
        # Sparse contact: Must physically hit the ball for a bonus
        (TouchBallReward(touch_value=2.0, touch_distance=170.0), 1.0),
        
        # Sparse jackpot: Replaces simple goal + dense velocity
        (SparseGoalSpeedReward(base_goal_value=20.0, speed_multiplier=30.0), 1.0),
        
        # Timeout penalty - FIXED keyword argument here
        (TimeoutPenalty(penalty_value=1.0), 1.0),
    )


def build_mixed_reward(mechanic_name: str = "kuxir") -> MixedRewardFunction:
    """
    Build the full mixed reward function with mechanic dispatch.
    """
    return MixedRewardFunction(
        mechanic_rewards={mechanic_name: build_kuxir_reward()},
        default_reward=build_normal_reward(),
    )