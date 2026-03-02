# src/metrics/pinch_metrics.py
"""
Custom MetricsLogger for the Pinch specialist.

Tracks pinch-relevant signals: goalward speed spikes, ball-wall distance,
and standard metrics (goals, touches, speeds, boost).
"""
from __future__ import annotations

import csv
import os
import numpy as np
from rlgym_ppo.util import MetricsLogger
from rlgym.rocket_league import common_values


class PinchLogger(MetricsLogger):
    """
    Per-step metrics (flat float array):
      [0] goal_scored
      [1] ball_speed         — norm of ball velocity (uu/s)
      [2] goalward_speed     — ball_vel · goal_dir (uu/s, can be negative)
      [3] ball_wall_dist     — distance from ball to nearest side wall (uu)
      [4] car_speed          — norm of car velocity (uu/s)
      [5] boost              — car boost_amount (0-100)
      [6] ball_touched       — 1.0 if touched
    """

    CSV_HEADER = [
        "timesteps", "goals", "ball_touches",
        "avg_ball_speed", "max_goalward_spike", "avg_goalward_speed",
        "avg_wall_dist", "avg_car_speed", "avg_boost", "steps",
    ]

    def __init__(self, csv_path: str = "checkpoints/pinch_metrics.csv",
                 tick_skip: int = 8, timeout_seconds: float = 2.0):
        super().__init__()
        self.csv_path = csv_path
        self.tick_skip = tick_skip
        self.timeout_seconds = timeout_seconds
        self._prev_goalward: float = 0.0
        # Seconds per decision step
        self._sec_per_step = tick_skip / 120.0
        # Max steps per episode (for estimating episode count)
        self._max_steps_per_ep = int(timeout_seconds / self._sec_per_step)

    def _collect_metrics(self, game_state) -> list:
        goal_scored_flag = float(getattr(game_state, "goal_scored", False))

        ball_vel = np.asarray(game_state.ball.linear_velocity, dtype=np.float32)
        ball_speed = float(np.linalg.norm(ball_vel))
        ball_pos = np.asarray(game_state.ball.position, dtype=np.float32)

        # Goalward speed (toward orange goal +Y for blue)
        goal_y = common_values.BACK_NET_Y
        goal_pos = np.array([0.0, goal_y, 0.0], dtype=np.float32)
        diff = goal_pos - ball_pos
        dist_to_goal = float(np.linalg.norm(diff))
        if dist_to_goal > 1e-6:
            goal_dir = diff / dist_to_goal
        else:
            goal_dir = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        goalward_speed = float(np.dot(ball_vel, goal_dir))

        # Ball-wall distance
        ball_wall_dist = common_values.SIDE_WALL_X - abs(float(ball_pos[0]))

        # Car data
        car_ids = list(game_state.cars.keys())
        if car_ids:
            car = game_state.cars[car_ids[0]]
            car_vel = np.asarray(car.physics.linear_velocity, dtype=np.float32)
            car_speed = float(np.linalg.norm(car_vel))
            boost = float(car.boost_amount)
            touched = float(car.ball_touches > 0)
        else:
            car_speed = 0.0
            boost = 0.0
            touched = 0.0

        return [np.array([
            goal_scored_flag, ball_speed, goalward_speed,
            ball_wall_dist, car_speed, boost, touched,
        ])]

    def report_metrics(self, collected_metrics, wandb_run, cumulative_timesteps):
        """Override to always print, even without wandb."""
        all_reports = []
        for serialized_metrics in collected_metrics:
            metrics_arrays = []
            i = 0
            while i < len(serialized_metrics):
                n_shape = int(serialized_metrics[i])
                n_values = 1
                shape = []
                i += 1
                for arg in serialized_metrics[i:i + n_shape]:
                    n_values *= int(arg)
                    shape.append(int(arg))
                n_values = int(n_values)
                metric = serialized_metrics[i + n_shape:i + n_shape + n_values]
                metrics_arrays.append(metric)
                i = i + n_shape + n_values
            all_reports.append(metrics_arrays)

        self._report_metrics(all_reports, wandb_run, cumulative_timesteps)

    def _report_metrics(self, collected_metrics, wandb_run, cumulative_timesteps):
        goals = []
        ball_speeds = []
        goalward_speeds = []
        wall_dists = []
        car_speeds = []
        boosts = []
        touches = []
        goalward_spikes = []
        prev_gw = 0.0

        for step_arrays in collected_metrics:
            arr = step_arrays[0]
            if len(arr) >= 7:
                goals.append(arr[0])
                ball_speeds.append(arr[1])
                goalward_speeds.append(arr[2])
                wall_dists.append(arr[3])
                car_speeds.append(arr[4])
                boosts.append(arr[5])
                touches.append(arr[6])
                # Track goalward speed deltas for spike detection
                gw = arr[2]
                delta = gw - prev_gw
                if delta > 0 and gw > 0:
                    goalward_spikes.append(delta)
                prev_gw = gw

        n_steps = len(goals)
        if n_steps == 0:
            return

        total_goals = int(sum(goals))
        total_touches = int(sum(touches))
        avg_ball_speed = float(np.mean(ball_speeds))
        max_gw_spike = float(max(goalward_spikes)) if goalward_spikes else 0.0
        avg_goalward = float(np.mean(goalward_speeds))
        avg_wall_dist = float(np.mean(wall_dists))
        avg_car_speed = float(np.mean(car_speeds))
        avg_boost = float(np.mean(boosts))

        # Episode estimate: steps / max_steps_per_episode
        est_episodes = max(1, n_steps // max(1, self._max_steps_per_ep))
        goal_rate = total_goals / est_episodes * 100
        touches_per_ep = total_touches / est_episodes

        # Tiered spike tracking (goalward spikes at kph thresholds)
        # 50 kph ~ 1400 uu/s, 75 kph ~ 2100, 100 kph ~ 2800, 125 kph ~ 3500
        spike_tiers = [
            ("50kph",  1400.0),
            ("75kph",  2100.0),
            ("100kph", 2800.0),
            ("125kph", 3500.0),
        ]
        spike_counts = {}
        for label, thresh in spike_tiers:
            count = sum(1 for s in goalward_spikes if s > thresh)
            spike_counts[label] = (count, count / est_episodes * 100)

        # Cumulative simulated gameplay time
        sim_seconds = cumulative_timesteps * self._sec_per_step
        sim_hours = sim_seconds / 3600
        sim_days = sim_hours / 24
        sim_years = sim_days / 365.25
        if sim_years >= 1.0:
            sim_time_str = f"{sim_years:.1f} years ({sim_days:.0f} days)"
        elif sim_days >= 1.0:
            sim_time_str = f"{sim_days:.1f} days ({sim_hours:.0f} hrs)"
        else:
            sim_time_str = f"{sim_hours:.1f} hours"

        print(f"\n{'='*8} PINCH METRICS {'='*8}")
        print(f"  Episodes (est):           ~{est_episodes}")
        print(f"  Goals this iteration:     {total_goals}  ({goal_rate:.1f}% goal rate)")
        print(f"  Ball touches:             {total_touches}  ({touches_per_ep:.1f}/ep)")
        # Spike tiers
        tier_parts = []
        for label, (count, pct) in spike_counts.items():
            tier_parts.append(f"{label}:{count}({pct:.1f}%)")
        print(f"  Goalward spikes:          {' | '.join(tier_parts)}")
        print(f"  Avg ball speed:           {avg_ball_speed:.0f} uu/s")
        print(f"  Max goalward spike:       {max_gw_spike:.0f} uu/s ({max_gw_spike * 0.036:.0f} kph)")
        print(f"  Avg goalward speed:       {avg_goalward:.0f} uu/s")
        print(f"  Avg ball-wall dist:       {avg_wall_dist:.0f} uu")
        print(f"  Avg car speed:            {avg_car_speed:.0f} uu/s")
        print(f"  Avg boost:                {avg_boost:.1f} / 100")
        print(f"  Steps this iteration:     {n_steps}")
        print(f"  Simulated gameplay:       {sim_time_str}")
        print(f"{'='*32}\n")

        self._append_csv(
            cumulative_timesteps, total_goals, total_touches,
            avg_ball_speed, max_gw_spike, avg_goalward,
            avg_wall_dist, avg_car_speed, avg_boost, n_steps,
        )

        if wandb_run is not None:
            log_dict = {
                "pinch/goals": total_goals,
                "pinch/goal_rate": goal_rate,
                "pinch/ball_touches": total_touches,
                "pinch/touches_per_ep": touches_per_ep,
                "pinch/avg_ball_speed": avg_ball_speed,
                "pinch/max_goalward_spike": max_gw_spike,
                "pinch/avg_goalward_speed": avg_goalward,
                "pinch/avg_wall_dist": avg_wall_dist,
                "pinch/avg_car_speed": avg_car_speed,
                "pinch/avg_boost": avg_boost,
                "pinch/sim_hours": sim_hours,
            }
            for label, (count, pct) in spike_counts.items():
                log_dict[f"pinch/spikes_{label}"] = count
                log_dict[f"pinch/spikes_{label}_pct"] = pct
            wandb_run.log(log_dict)

    def _append_csv(self, timesteps, goals, touches, ball_speed, max_spike,
                    avg_goalward, wall_dist, car_speed, boost, steps):
        file_exists = os.path.isfile(self.csv_path)
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)

        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(self.CSV_HEADER)
            writer.writerow([
                timesteps, goals, touches,
                f"{ball_speed:.1f}", f"{max_spike:.1f}", f"{avg_goalward:.1f}",
                f"{wall_dist:.1f}", f"{car_speed:.1f}", f"{boost:.1f}", steps,
            ])
