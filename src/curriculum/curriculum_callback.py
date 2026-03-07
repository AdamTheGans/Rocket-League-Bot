# src/curriculum/curriculum_callback.py
"""
SB3 callback that implements curriculum learning for mechanic training.

Monitors the success rate for episodes that used the mechanic setter and
adaptively adjusts the trajectory setter's ``difficulty`` and ``noise_amount``
sliders across ALL worker processes via ``VecEnv.set_attr()``.

Also collects per-episode telemetry (ball speed, episode length) and logs
everything to TensorBoard.

Fully generic — works with any ``mechanic_name``.
"""
from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional

import numpy as np

from stable_baselines3.common.callbacks import BaseCallback


class CurriculumCallback(BaseCallback):
    """
    Adaptive curriculum callback with TensorBoard telemetry.

    Parameters
    ----------
    mechanic_name : str
        The mechanic identifier (e.g., "kuxir").
    eval_interval : int
        Check success rate every this many training steps.
    window_size : int
        Rolling window of recent mechanic episodes to evaluate.
    promote_threshold : float
        If success rate > this, increase difficulty/noise (default 0.80).
    demote_threshold : float
        If success rate < this, decrease difficulty/noise (default 0.40).
    noise_step : float
        How much to increment ``noise_amount`` on promotion (default 0.05).
    difficulty_step : float
        How much to increment ``difficulty`` on promotion (default 0.02).
    noise_gate : float
        Only start increasing ``difficulty`` once ``noise_amount`` exceeds
        this threshold (default 0.3).
    noise_cap : float
        Maximum value for ``noise_amount`` (default 1.0).
    difficulty_cap : float
        Maximum value for ``difficulty`` (default 1.0).
    verbose : int
        Verbosity level (0 = silent, 1 = print updates).
    """

    def __init__(
        self,
        mechanic_name: str = "kuxir",
        eval_interval: int = 5000,
        window_size: int = 100,
        promote_threshold: float = 0.80,
        demote_threshold: float = 0.40,
        noise_step: float = 0.05,
        difficulty_step: float = 0.02,
        noise_gate: float = 0.3,
        noise_cap: float = 1.0,
        difficulty_cap: float = 1.0,
        verbose: int = 1,
    ):
        super().__init__(verbose=verbose)
        self.eval_interval = eval_interval
        self.window_size = window_size
        self.promote_threshold = promote_threshold
        self.demote_threshold = demote_threshold
        self.noise_step = noise_step
        self.difficulty_step = difficulty_step
        self.noise_gate = noise_gate
        self.noise_cap = noise_cap
        self.difficulty_cap = difficulty_cap

        # Derive the info key from the mechanic name
        self.mechanic_name = mechanic_name
        self.success_key = f"{self.mechanic_name}_success"

        # Rolling buffer of mechanic episode outcomes (True/False)
        self._mechanic_outcomes: deque = deque(maxlen=window_size)
        self._last_eval_step: int = 0

        # Current slider values (authoritative source of truth;
        # synced to workers via set_attr)
        self._difficulty: float = 0.1
        self._noise_amount: float = 0.05

        # ── Telemetry buffers ───────────────────────────────────────
        # Collect per-episode metrics between evaluations, then average
        self._mechanic_max_speeds: List[float] = []
        self._mechanic_end_speeds: List[float] = []
        self._mechanic_ep_lengths: List[int] = []
        self._normal_ep_lengths: List[int] = []
        self._mechanic_goals: int = 0
        self._mechanic_episodes: int = 0

    def _on_step(self) -> bool:
        """Called after every environment step by SB3."""
        mn = self.mechanic_name

        # ── Collect episode-end data from info buffers ──────────────
        infos = self.locals.get("infos", [])
        for info in infos:
            # Mechanic outcomes (for curriculum)
            if self.success_key in info:
                scored = bool(info[self.success_key])
                self._mechanic_outcomes.append(scored)
                self._mechanic_episodes += 1
                if scored:
                    self._mechanic_goals += 1

            # Mechanic telemetry
            if f"{mn}_max_ball_speed" in info:
                self._mechanic_max_speeds.append(float(info[f"{mn}_max_ball_speed"]))
            if f"{mn}_ball_speed_at_end" in info:
                self._mechanic_end_speeds.append(float(info[f"{mn}_ball_speed_at_end"]))
            if f"{mn}_episode_length" in info:
                self._mechanic_ep_lengths.append(int(info[f"{mn}_episode_length"]))

            # Normal play telemetry
            if "normal_episode_length" in info:
                self._normal_ep_lengths.append(int(info["normal_episode_length"]))

        # ── Periodic evaluation ─────────────────────────────────────
        if self.num_timesteps - self._last_eval_step >= self.eval_interval:
            self._last_eval_step = self.num_timesteps
            self._evaluate_and_adjust()
            self._log_telemetry()

        return True  # Continue training

    def _sync_to_workers(self) -> None:
        """Push current difficulty/noise values to all VecEnv workers."""
        try:
            self.training_env.set_attr("difficulty", self._difficulty)
            self.training_env.set_attr("noise_amount", self._noise_amount)
        except Exception as e:
            if self.verbose:
                print(f"  [Curriculum] WARNING: set_attr failed: {e}")

    def _log_telemetry(self) -> None:
        """Log telemetry metrics to TensorBoard and clear buffers."""
        if self.logger is None:
            return

        mn = self.mechanic_name

        # Mechanic ball speed metrics
        if self._mechanic_max_speeds:
            self.logger.record(
                f"{mn}/max_ball_speed",
                np.mean(self._mechanic_max_speeds),
            )
        if self._mechanic_end_speeds:
            self.logger.record(
                f"{mn}/avg_ball_speed_at_end",
                np.mean(self._mechanic_end_speeds),
            )

        # Goal rate (cumulative)
        if self._mechanic_episodes > 0:
            self.logger.record(
                f"{mn}/goal_rate",
                self._mechanic_goals / self._mechanic_episodes,
            )
            self.logger.record(
                f"{mn}/total_episodes",
                self._mechanic_episodes,
            )

        # Episode lengths
        if self._mechanic_ep_lengths:
            self.logger.record(
                f"global/episode_length_{mn}",
                np.mean(self._mechanic_ep_lengths),
            )
        if self._normal_ep_lengths:
            self.logger.record(
                f"global/episode_length_normal",
                np.mean(self._normal_ep_lengths),
            )

        # Clear per-interval buffers (keep cumulative counters)
        self._mechanic_max_speeds.clear()
        self._mechanic_end_speeds.clear()
        self._mechanic_ep_lengths.clear()
        self._normal_ep_lengths.clear()

    def _evaluate_and_adjust(self) -> None:
        """Check success rate and adjust curriculum sliders."""
        if len(self._mechanic_outcomes) < 10:
            # Not enough data yet
            if self.verbose:
                print(f"  [Curriculum] Step {self.num_timesteps}: "
                      f"Only {len(self._mechanic_outcomes)} mechanic episodes "
                      f"collected, waiting for more data...")
            return

        success_rate = np.mean(list(self._mechanic_outcomes))
        old_noise = self._noise_amount
        old_diff = self._difficulty

        if success_rate > self.promote_threshold:
            # ── Promote: increase noise, then difficulty ────────────
            self._noise_amount = min(
                self.noise_cap,
                old_noise + self.noise_step,
            )
            if old_noise >= self.noise_gate:
                self._difficulty = min(
                    self.difficulty_cap,
                    old_diff + self.difficulty_step,
                )

        elif success_rate < self.demote_threshold:
            # ── Demote: decrease noise and difficulty ────────────────
            self._noise_amount = max(
                0.05,
                old_noise - self.noise_step * 0.5,
            )
            self._difficulty = max(
                0.1,
                old_diff - self.difficulty_step * 0.5,
            )

        # ── Sync updated values to all worker envs ──────────────────
        if self._noise_amount != old_noise or self._difficulty != old_diff:
            self._sync_to_workers()

        # ── Log to TensorBoard ──────────────────────────────────────
        if self.logger is not None:
            self.logger.record(f"curriculum/{self.mechanic_name}_success_rate", success_rate)
            self.logger.record(f"curriculum/{self.mechanic_name}_noise", self._noise_amount)
            self.logger.record(f"curriculum/{self.mechanic_name}_difficulty", self._difficulty)
            self.logger.record(f"curriculum/{self.mechanic_name}_window_size", len(self._mechanic_outcomes))

        if self.verbose:
            noise_delta = self._noise_amount - old_noise
            diff_delta = self._difficulty - old_diff
            arrow = "↑" if noise_delta > 0 else ("↓" if noise_delta < 0 else "→")
            print(
                f"  [Curriculum] Step {self.num_timesteps}: "
                f"success={success_rate:.1%} {arrow} | "
                f"noise={self._noise_amount:.3f} "
                f"(Δ{noise_delta:+.3f}) | "
                f"difficulty={self._difficulty:.3f} "
                f"(Δ{diff_delta:+.3f}) | "
                f"window={len(self._mechanic_outcomes)}"
            )
