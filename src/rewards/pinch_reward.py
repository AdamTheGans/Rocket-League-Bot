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

# Module-level variable to store reward breakdowns inside the env process.
# This bypasses RLGym's GameState serialization which strips dynamic attributes.
GLOBAL_REWARD_BREAKDOWN: Dict[str, float] = {}

# ─── Pinch-specific reward components ───────────────────────────────────────

class LoggingCombinedReward(CombinedReward):
    """
    A CombinedReward that intercepts the computed per-component rewards
    and saves them out to the GameState object so the metrics logger can
    read them before the state is serialized into shared memory.
    """
    def __init__(self, *rewards_and_weights):
        super().__init__(*rewards_and_weights)
        self.reward_names = []
        for r, w in rewards_and_weights:
            # e.g., "GoalwardSpeedSpikeReward" -> "GoalwardSpeedSpike"
            self.reward_names.append(r.__class__.__name__.replace("Reward", ""))

    def get_rewards(
        self,
        agents: List[AgentID],
        state: GameState,
        is_terminated: Dict[AgentID, bool],
        is_truncated: Dict[AgentID, bool],
        shared_info: Dict[str, Any],
    ) -> Dict[AgentID, float]:
        combined_rewards = {agent: 0.0 for agent in agents}
        
        # We will store the breakdown here, averaged across all agents for this step
        breakdown = {name: 0.0 for name in self.reward_names}
        
        for name, reward_fn, weight in zip(self.reward_names, self.reward_fns, self.weights):
            rewards = reward_fn.get_rewards(agents, state, is_terminated, is_truncated, shared_info)
            for agent, reward in rewards.items():
                val = reward * weight
                combined_rewards[agent] += val
                breakdown[name] += val
                
        # Average the breakdown by number of agents
        # Average the breakdown by number of agents
        if agents:
            for name in breakdown:
                breakdown[name] /= len(agents)
                
        # Store in global state to bypass rlgym process serialization wiping out attributes
        global GLOBAL_REWARD_BREAKDOWN
        GLOBAL_REWARD_BREAKDOWN.clear()
        GLOBAL_REWARD_BREAKDOWN.update(breakdown)

        return combined_rewards


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


class ZFilteredGoalwardSpikeReward(RewardFunction[AgentID, GameState, float]):
    """
    Rewards positive increases in the ball's goalward velocity component, capped to
    the absolute maximum speed registered in the episode to prevent repetitive acceleration farming.
    
    Additionally, applies a Z-velocity penalty multiplier:
    - Target flat range: 200 to 550 uu/s (multiplier = 1.0)
    - Linearly scales down the multiplier based on distance from that target range (max drop to 0.1).
    """

    def __init__(self, target_z_min: float = 200.0, target_z_max: float = 550.0, max_z_diff: float = 500.0):
        super().__init__()
        self._max_goalward: Dict[AgentID, float] = {}
        self.target_z_min = target_z_min
        self.target_z_max = target_z_max
        self.max_z_diff = max_z_diff

    def reset(
        self,
        agents: List[AgentID],
        initial_state: GameState,
        shared_info: Dict[str, Any],
    ) -> None:
        ball = initial_state.ball
        for agent in agents:
            car = initial_state.cars[agent]
            goal_dir = GoalwardSpeedSpikeReward._goal_direction(ball.position, car.is_orange)
            goalward = float(np.dot(ball.linear_velocity, goal_dir))
            # Start tracking the max from the spawn state
            self._max_goalward[agent] = max(0.0, goalward)

    def get_rewards(
        self,
        agents: List[AgentID],
        state: GameState,
        is_terminated: Dict[AgentID, bool],
        is_truncated: Dict[AgentID, bool],
        shared_info: Dict[str, Any],
    ) -> Dict[AgentID, float]:
        rewards = {agent: 0.0 for agent in agents}
        ball = state.ball
        
        for agent in agents:
            car = state.cars[agent]
            goal_dir = GoalwardSpeedSpikeReward._goal_direction(ball.position, car.is_orange)
            curr_goalward = float(np.dot(ball.linear_velocity, goal_dir))

            current_max = self._max_goalward.get(agent, 0.0)
            delta = curr_goalward - current_max

            if delta > 0:
                # We reached a new top speed pointing at the net
                self._max_goalward[agent] = curr_goalward
                base_reward = delta / common_values.BALL_MAX_SPEED
                
                # Apply Z-Velocity Filter
                ball_z_vel = ball.linear_velocity[2]
                if self.target_z_min <= ball_z_vel <= self.target_z_max:
                    multiplier = 1.0
                else:
                    z_diff = min(abs(ball_z_vel - self.target_z_min), abs(ball_z_vel - self.target_z_max))
                    multiplier = max(0.1, 1.0 - (z_diff / self.max_z_diff))
                    
                rewards[agent] = base_reward * multiplier

        return rewards


class LatchGoalwardSpeedSpikeReward(RewardFunction[AgentID, GameState, float]):
    """
    Grants a massive one-time reward if the ball velocity exceeds a threshold
    in the goalward direction, and the car has recently touched it.
    """
    def __init__(self, spike_threshold: float = 2500.0, latch_reward: float = 50.0):
        super().__init__()
        self.spike_threshold = spike_threshold
        self.latch_reward = latch_reward
        self._latched: Dict[AgentID, bool] = {}
        self._last_touch_ticks: Dict[AgentID, int] = {}
        
    def reset(
        self,
        agents: List[AgentID],
        initial_state: GameState,
        shared_info: Dict[str, Any],
    ) -> None:
        for agent in agents:
            self._latched[agent] = False
            self._last_touch_ticks[agent] = -100

    def get_rewards(
        self,
        agents: List[AgentID],
        state: GameState,
        is_terminated: Dict[AgentID, bool],
        is_truncated: Dict[AgentID, bool],
        shared_info: Dict[str, Any],
    ) -> Dict[AgentID, float]:
        rewards = {agent: 0.0 for agent in agents}
        ball = state.ball
        
        # In RocketSim/RLGym v2, game_state usually has a tick_count mapped to `state.tick_count`
        # But if it's not present, we can just increment a counter per step.
        current_tick = getattr(state, 'tick_count', 0)
        
        for agent in agents:
            if self._latched.get(agent, False):
                continue
                
            car = state.cars[agent]
            
            # Update last touch tick
            if car.ball_touches > 0:
                self._last_touch_ticks[agent] = current_tick
                
            # Check if touch occurred within the last 2 ticks
            ticks_since_touch = current_tick - self._last_touch_ticks.get(agent, -100)
            touched_recently = (0 <= ticks_since_touch <= 2)
            
            if not touched_recently:
                continue
                
            goal_y = -common_values.BACK_NET_Y if car.is_orange else common_values.BACK_NET_Y
            goal_pos = np.array([0.0, goal_y, 0.0], dtype=np.float32)
            diff = goal_pos - ball.position
            dist = np.linalg.norm(diff)
            
            if dist > 1e-6:
                goal_dir = diff / dist
            else:
                goal_dir = np.array([0.0, 1.0, 0.0], dtype=np.float32)
                
            curr_goalward = float(np.dot(ball.linear_velocity, goal_dir))
            
            # Additional validation: make sure it's actually moving in the absolute Y direction of the goal
            absolute_goalward = ball.linear_velocity[1] < 0 if car.is_orange else ball.linear_velocity[1] > 0
            
            if curr_goalward >= self.spike_threshold and absolute_goalward:
                rewards[agent] = self.latch_reward
                self._latched[agent] = True
        return rewards


class TouchReward(RewardFunction[AgentID, GameState, float]):
    """
    Rewards the agent exactly once per episode for touching the ball.
    Prevents exploitation where the agent dribbles or rubs against the ball.
    """
    def __init__(self):
        super().__init__()
        self._has_touched: Dict[AgentID, bool] = {}

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        for agent in agents:
            self._has_touched[agent] = False

    def get_rewards(
        self,
        agents: List[AgentID],
        state: GameState,
        is_terminated: Dict[AgentID, bool],
        is_truncated: Dict[AgentID, bool],
        shared_info: Dict[str, Any],
    ) -> Dict[AgentID, float]:
        rewards = {agent: 0.0 for agent in agents}
        for agent in agents:
            if not self._has_touched.get(agent, False):
                car = state.cars[agent]
                if car.ball_touches > 0:
                    rewards[agent] = 1.0
                    self._has_touched[agent] = True
        return rewards


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


def build_pinch_reward(stage: float = 1.0) -> CombinedReward:
    """
    Build stage-dependent pinch reward.

    Stage 1/1.5: No goals rewarded. Only heavy spike weight and shaping.
    Stage 2: Massive spike weight (150) equivalent to a goal (100).
    Stage 3: Full shaping, emphasis on goalward ball velocity
    """
    if stage == 1 or stage == 1.5:
        return LoggingCombinedReward(
            (QuickGoalReward(base=1.0, bonus=0.5), 100.0),
            (ZFilteredGoalwardSpikeReward(),       150.0),
            (TouchReward(),                          5.0),
            (TimePenalty(),                          0.0),
        )
    elif stage == 2:
        return LoggingCombinedReward(
            (QuickGoalReward(base=1.0, bonus=0.5), 500.0),
            (ZFilteredGoalwardSpikeReward(),       150.0),
            (TouchReward(),                          1.5),
            (TimePenalty(),                         -0.05),
        )
    elif stage == 3:
        return LoggingCombinedReward(
            (QuickGoalReward(base=1.0, bonus=0.5), 500.0),
            (ZFilteredGoalwardSpikeReward(),       150.0),
            (TouchReward(),                          1.5),
            (ApproachPinchPointReward(),             0.15),
            (TimePenalty(),                         -0.05),
        )
    else:
        raise ValueError(f"stage must be 1, 1.5, 2, or 3, got {stage}")


def build_golden_seed_reward() -> CombinedReward:
    """
    Reward function tailored strictly for the Golden Seed initialization.
    
    Design:
      - QuickGoalReward (100): Massive reward for scoring, with speed bonus.
      - GoalwardSpeedSpikeReward (150.0): Huge reward for the actual pinch impact.
        Properly aligned towards the opponent's goal.
      - TouchReward (5.0): Dense reward for just finding the ball and dodging into it.
      - TimePenalty (-0.05): General urgency to act fast.
    """
    return LoggingCombinedReward(
        (QuickGoalReward(base=1.0, bonus=0.5), 100.0),
        (ZFilteredGoalwardSpikeReward(),       150.0),
        (TouchReward(),                          5.0),
        (TimePenalty(),                          0.0),
    )
