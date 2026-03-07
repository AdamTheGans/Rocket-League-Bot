# src/wrappers/self_play_env.py
"""
Single-agent gym.Env wrapper for 1v1 rlgym environments.

Manages the opponent's actions internally via a frozen policy, exposing
only the Blue (team 0) agent's observations / rewards / done signals to
SB3.  Also injects mechanic success info into the ``info`` dict for
the CurriculumCallback.

The wrapper exposes ``difficulty`` and ``noise_amount`` properties that
drill down into the ``MechanicTrajectorySetter`` inside the env's
``MixedStateSetter``.  This allows SB3's ``VecEnv.set_attr()`` to
propagate curriculum changes to all worker processes.
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
    """
    Wraps a multi-agent RLGym v2 env into a single-agent gym interface.

    Parameters
    ----------
    rlgym_env : RLGym
        The raw RLGym environment (NOT wrapped in RLGymV2GymWrapper).
        Must be a 1v1 setup (blue_size=1, orange_size=1).
    opponent_fn : callable
        ``opponent_fn(obs: np.ndarray) -> int``
        Takes the opponent's observation vector and returns an action index.
    obs_size : int
        Expected observation dimension (used to define the gym obs space).
    num_actions : int
        Number of discrete actions (default 90 for LookupTableAction).
    mechanic_name : str, optional
        If set, the wrapper will emit ``{mechanic_name}_success`` in the
        info dict when the episode ends and the setter_type matched.
    mechanic_setter : MechanicTrajectorySetter, optional
        Reference to the trajectory setter for exposing curriculum properties.
    mixed_setter : MixedStateSetter, optional
        Reference to the mixed setter for reading which setter was used
        after each reset (needed because rlgym's shared_info is transient).
    goal_reward_bonus : float
        Extra reward added when the Blue agent scores a goal (default 0).
    """

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
    ):
        super().__init__()
        self.env = rlgym_env
        self.opponent_fn = opponent_fn
        self.mechanic_name = mechanic_name
        self._mechanic_setter = mechanic_setter
        self._mixed_setter = mixed_setter
        self.goal_reward_bonus = goal_reward_bonus

        # Gym spaces
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(num_actions)

        # Agent IDs (discovered on first reset)
        self.blue_agent = None
        self.orange_agent = None

        # Cached obs_dict from last reset/step — needed for opponent action
        self._last_obs_dict: Optional[Dict] = None

        # Episode tracking
        self._current_setter_type: str = "normal"
        self._episode_scored: bool = False
        self._max_ball_speed: float = 0.0
        self._episode_steps: int = 0

    # ─────────────── Curriculum properties (for set_attr sync) ──── #

    @property
    def difficulty(self) -> float:
        """Current curriculum difficulty (0.0–1.0)."""
        if self._mechanic_setter is not None:
            return self._mechanic_setter.difficulty
        return 0.0

    @difficulty.setter
    def difficulty(self, value: float) -> None:
        """Set curriculum difficulty across this worker's setter."""
        if self._mechanic_setter is not None:
            self._mechanic_setter.difficulty = float(value)

    @property
    def noise_amount(self) -> float:
        """Current curriculum noise amount (0.0–1.0)."""
        if self._mechanic_setter is not None:
            return self._mechanic_setter.noise_amount
        return 0.0

    @noise_amount.setter
    def noise_amount(self, value: float) -> None:
        """Set curriculum noise across this worker's setter."""
        if self._mechanic_setter is not None:
            self._mechanic_setter.noise_amount = float(value)

    # ─────────────────── Core gym interface ─────────────────────── #

    def reset(self, *, seed=None, options=None) -> Tuple[np.ndarray, dict]:
        """Reset the rlgym env and return Blue agent's observation."""
        obs_dict = self.env.reset()
        game_state = self.env.state

        # Discover agent IDs
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

        # Cache obs_dict so opponent can use it on the next step
        self._last_obs_dict = obs_dict

        # Read setter type from MixedStateSetter (shared_info is transient
        # in rlgym v2 and not stored on the env object)
        if self._mixed_setter is not None:
            self._current_setter_type = self._mixed_setter.last_setter_name or "normal"
        else:
            self._current_setter_type = "normal"
        self._episode_scored = False
        self._max_ball_speed = 0.0
        self._episode_steps = 0

        obs = obs_dict[self.blue_agent]
        info = {"setter_type": self._current_setter_type}
        return obs.astype(np.float32), info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Step the env with Blue's action and the opponent's response.

        Returns (obs, reward, terminated, truncated, info) in gym v0.26+ format.
        """
        # ── Get opponent action from cached obs ─────────────────────
        opp_action = 0
        if self.orange_agent is not None and self._last_obs_dict is not None:
            opp_obs = self._last_obs_dict.get(self.orange_agent)
            if opp_obs is not None:
                opp_action = self.opponent_fn(opp_obs)

        # ── Pack actions for both agents ────────────────────────────
        actions = {}
        if self.blue_agent is not None:
            actions[self.blue_agent] = np.array([action])
        if self.orange_agent is not None:
            actions[self.orange_agent] = np.array([opp_action])

        # ── Step the environment ────────────────────────────────────
        obs_dict, reward_dict, terminated_dict, truncated_dict = self.env.step(actions)

        # Cache obs_dict for the next step's opponent action
        self._last_obs_dict = obs_dict

        # ── Extract Blue agent's data ───────────────────────────────
        obs = obs_dict[self.blue_agent].astype(np.float32)
        reward = float(reward_dict.get(self.blue_agent, 0.0))

        terminated = bool(terminated_dict.get(self.blue_agent, False))
        truncated = bool(truncated_dict.get(self.blue_agent, False))

        # Track ball speed for telemetry
        ball_vel = self.env.state.ball.linear_velocity
        ball_speed = float(np.linalg.norm(ball_vel))
        self._max_ball_speed = max(self._max_ball_speed, ball_speed)
        self._episode_steps += 1

        # Check if Blue scored (goal terminated the episode, ball in orange half)
        if terminated:
            ball_y = self.env.state.ball.position[1]
            if ball_y > 0:
                self._episode_scored = True
                reward += self.goal_reward_bonus

        # Build info dict
        info: Dict[str, Any] = {
            "setter_type": self._current_setter_type,
        }

        # Emit telemetry and mechanic success on episode end
        done = terminated or truncated
        if done:
            prefix = self._current_setter_type  # "kuxir" or "normal"
            info[f"{prefix}_episode_length"] = self._episode_steps
            info[f"{prefix}_max_ball_speed"] = self._max_ball_speed
            info[f"{prefix}_ball_speed_at_end"] = ball_speed

            if self.mechanic_name is not None:
                if self._current_setter_type == self.mechanic_name:
                    info[f"{self.mechanic_name}_success"] = self._episode_scored

        return obs, reward, terminated, truncated, info

    def close(self):
        """Close the underlying rlgym env."""
        if hasattr(self.env, 'close'):
            self.env.close()

    def render(self):
        """Rendering is not supported in training mode."""
        pass


# ─────────────────────────────────────────────────────────────────── #
#  Opponent factory helpers
# ─────────────────────────────────────────────────────────────────── #

def make_frozen_opponent(
    checkpoint_path: str,
    metadata_path: Optional[str] = None,
    device: str = "cpu",
) -> Callable:
    """
    Load a frozen StudentPolicy as an opponent.

    The returned callable takes an observation vector (np.ndarray of shape
    ``(obs_dim,)``) and returns a discrete action index.

    Parameters
    ----------
    checkpoint_path : str
        Path to the ``student_policy.pt`` file.
    metadata_path : str, optional
        Path to ``metadata.json`` (default: same directory as checkpoint).
    device : str
        PyTorch device (default "cpu").

    Returns
    -------
    opponent_fn : callable
        ``(obs: np.ndarray) -> int``
    """
    import sys
    # Ensure src/nexto_distill is importable
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

    print(f"  [Opponent] Loaded frozen StudentPolicy: "
          f"layers={meta['layer_sizes']}, params={meta.get('total_params', '?'):,}")

    def opponent_fn(obs: np.ndarray) -> int:
        """
        Run the frozen student policy on the opponent's observation.

        Parameters
        ----------
        obs : np.ndarray
            Observation vector of shape (obs_dim,) from DefaultObs,
            computed by RLGym for the Orange agent.

        Returns
        -------
        int
            Discrete action index.
        """
        with torch.no_grad():
            t = torch.from_numpy(obs).float().unsqueeze(0).to(dev)
            logits = model(t)
            return int(logits.argmax(dim=-1).item())

    return opponent_fn


def make_idle_opponent() -> Callable:
    """Return an opponent that always picks action 0 (idle)."""
    def opponent_fn(obs: np.ndarray) -> int:
        return 0
    return opponent_fn
