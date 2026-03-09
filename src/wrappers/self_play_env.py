# src/wrappers/self_play_env.py
"""
Single-agent gym.Env wrapper for 1v1 rlgym environments.

Manages the opponent's actions internally via a frozen policy, exposing
only the Blue (team 0) agent's observations / rewards / done signals to
SB3.  Also injects mechanic success info into the ``info`` dict for
the CurriculumCallback.
"""
from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch

from rlgym.rocket_league import common_values


class SelfPlayEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        rlgym_env,
        opponent_fn: Callable,
        obs_size: int = 92,
        num_actions: int = 90,
        mechanic_name: Optional[str] = None,
        mechanic_setter=None,
        mixed_setter=None,
        goal_reward_bonus: float = 0.0,
        idle_opponent_during_mechanic: bool = True,
    ):
        super().__init__()
        self.env = rlgym_env
        self.opponent_fn = opponent_fn
        self.mechanic_name = mechanic_name
        self._mechanic_setter = mechanic_setter
        self._mixed_setter = mixed_setter
        self.goal_reward_bonus = goal_reward_bonus
        self.idle_opponent_during_mechanic = idle_opponent_during_mechanic

        # Gym spaces
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(num_actions)

        # Agent IDs
        self.blue_agent = None
        self.orange_agent = None

        # Cached obs_dict
        self._last_obs_dict: Optional[Dict] = None

        # Episode tracking
        self._current_setter_type: str = "normal"
        self._episode_scored: bool = False
        self._max_ball_speed: float = 0.0
        self._episode_steps: int = 0
        
        # New Metric Tracking
        self._initial_tick_count: int = 0
        self._time_to_goal: float = 0.0

    # ─────────────── Curriculum properties ──────────────────────── #

    @property
    def difficulty(self) -> float:
        if self._mechanic_setter is not None:
            return self._mechanic_setter.difficulty
        return 0.0

    @difficulty.setter
    def difficulty(self, value: float) -> None:
        if self._mechanic_setter is not None:
            self._mechanic_setter.difficulty = float(value)

    @property
    def noise_amount(self) -> float:
        if self._mechanic_setter is not None:
            return self._mechanic_setter.noise_amount
        return 0.0

    @noise_amount.setter
    def noise_amount(self, value: float) -> None:
        if self._mechanic_setter is not None:
            self._mechanic_setter.noise_amount = float(value)

    # ─────────────────── Core gym interface ─────────────────────── #

    def reset(self, *, seed=None, options=None) -> Tuple[np.ndarray, dict]:
        obs_dict = self.env.reset()
        game_state = self.env.state

        self.blue_agent = None
        self.orange_agent = None
        for aid in sorted(game_state.cars.keys()):
            car = game_state.cars[aid]
            if car.team_num == 0:
                self.blue_agent = aid
            else:
                self.orange_agent = aid

        if self.blue_agent is None:
            raise RuntimeError("No Blue (team 0) agent found in the environment.")

        self._last_obs_dict = obs_dict

        if self._mixed_setter is not None:
            self._current_setter_type = self._mixed_setter.last_setter_name or "normal"
        else:
            self._current_setter_type = "normal"
            
        # Reset episode trackers
        self._episode_scored = False
        self._max_ball_speed = 0.0
        self._episode_steps = 0
        self._time_to_goal = 0.0
        self._initial_tick_count = game_state.tick_count

        obs = obs_dict[self.blue_agent]
        info = {"setter_type": self._current_setter_type}
        return obs.astype(np.float32), info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        opp_action = 0
        if self.orange_agent is not None and self._last_obs_dict is not None:
            opp_obs = self._last_obs_dict.get(self.orange_agent)
            if opp_obs is not None:
                if self.idle_opponent_during_mechanic and self._current_setter_type != "normal":
                    opp_action = 0
                else:
                    opp_action = self.opponent_fn(opp_obs)

        actions = {}
        if self.blue_agent is not None:
            actions[self.blue_agent] = np.array([action])
        if self.orange_agent is not None:
            actions[self.orange_agent] = np.array([opp_action])

        obs_dict, reward_dict, terminated_dict, truncated_dict = self.env.step(actions)

        self._last_obs_dict = obs_dict

        obs = obs_dict[self.blue_agent].astype(np.float32)
        reward = float(reward_dict.get(self.blue_agent, 0.0))

        terminated = bool(terminated_dict.get(self.blue_agent, False))
        truncated = bool(truncated_dict.get(self.blue_agent, False))

        ball_vel = self.env.state.ball.linear_velocity
        ball_speed = float(np.linalg.norm(ball_vel))
        self._max_ball_speed = max(self._max_ball_speed, ball_speed)
        self._episode_steps += 1

        # Check if Blue scored
        if terminated:
            ball_y = self.env.state.ball.position[1]
            if ball_y > 0:
                self._episode_scored = True
                reward += self.goal_reward_bonus
                
                # Calculate exact physics time to goal
                ticks_elapsed = self.env.state.tick_count - self._initial_tick_count
                self._time_to_goal = ticks_elapsed / 120.0 

        info: Dict[str, Any] = {
            "setter_type": self._current_setter_type,
        }
        
        # --- REWARD COMPONENTS LOGGING ---
        # If your reward function has a 'last_rewards' attribute, extract it here
        if hasattr(self.env.reward_fn, "last_rewards"):
            # Ensure we only grab the Blue agent's rewards
            if self.blue_agent in self.env.reward_fn.last_rewards:
                info["reward_components"] = self.env.reward_fn.last_rewards[self.blue_agent]

        done = terminated or truncated
        if done:
            prefix = self._current_setter_type
            info[f"{prefix}_episode_length"] = self._episode_steps
            info[f"{prefix}_max_ball_speed"] = self._max_ball_speed
            info[f"{prefix}_ball_speed_at_end"] = ball_speed

            if self.mechanic_name is not None:
                if self._current_setter_type == self.mechanic_name:
                    info[f"{self.mechanic_name}_success"] = self._episode_scored
                    
                    # Only log time_to_goal if they actually scored
                    if self._episode_scored:
                        info[f"{self.mechanic_name}_time_to_goal"] = self._time_to_goal

        return obs, reward, terminated, truncated, info

    def close(self):
        if hasattr(self.env, 'close'):
            self.env.close()

    def render(self):
        pass

# ─────────────────────────────────────────────────────────────────── #
#  Opponent factory helpers (Unchanged, included for completeness)
# ─────────────────────────────────────────────────────────────────── #

def make_frozen_opponent(
    checkpoint_path: str,
    metadata_path: Optional[str] = None,
    device: str = "cpu",
) -> Callable:
    import sys
    src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    from nexto_distill.student_policy import StudentPolicy

    if metadata_path is None:
        metadata_path = os.path.join(os.path.dirname(checkpoint_path), "metadata.json")

    with open(metadata_path) as f:
        meta = json.load(f)

    model = StudentPolicy(
        obs_dim=meta["obs_dim"],
        num_actions=meta["num_actions"],
        layer_sizes=meta["layer_sizes"],
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    dev = torch.device(device)
    model = model.to(dev)

    def opponent_fn(obs: np.ndarray) -> int:
        with torch.no_grad():
            t = torch.from_numpy(obs).float().unsqueeze(0).to(dev)
            logits = model(t)
            return int(logits.argmax(dim=-1).item())

    return opponent_fn

def make_idle_opponent() -> Callable:
    def opponent_fn(obs: np.ndarray) -> int:
        return 0
    return opponent_fn