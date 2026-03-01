# src/metrics/strike_metrics.py
"""
Custom MetricsLogger for the Grounded Strike specialist.

Collects per-step metrics from the RLGym v2 GameState and reports
aggregated stats (goal count, ball touches, speeds, boost) at each
training iteration — printed to console and optionally logged to wandb.
"""
from __future__ import annotations

import numpy as np
from rlgym_ppo.util import MetricsLogger


class GroundedStrikeLogger(MetricsLogger):
    """
    Metrics collected per timestep (as a flat float array):
      [0] goal_scored  — 1.0 if a goal was scored this step, else 0.0
      [1] ball_speed   — norm of ball linear velocity (uu/s)
      [2] car_speed    — norm of car linear velocity (uu/s)
      [3] boost        — car boost_amount (0-100)
      [4] ball_touched — 1.0 if car touched ball, else 0.0
    """

    def _collect_metrics(self, game_state) -> list:
        """
        Called every timestep inside the env subprocess.
        `game_state` is the RLGym v2 GameState from the wrapper.
        Must return a list of numpy arrays.
        """
        goal = float(getattr(game_state, "goal_scored", False))

        ball_vel = np.asarray(game_state.ball.linear_velocity, dtype=np.float32)
        ball_speed = float(np.linalg.norm(ball_vel))

        # Get first car's data
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

        return [np.array([goal, ball_speed, car_speed, boost, touched])]

    def report_metrics(self, collected_metrics, wandb_run, cumulative_timesteps):
        """
        Override the PUBLIC report_metrics (not just _report_metrics) because
        the base class returns early when wandb_run is None, which means our
        console output would never be printed without wandb.
        """
        # Deserialize using the parent's logic
        all_reports = []
        for serialized_metrics in collected_metrics:
            metrics_arrays = []
            i = 0
            while i < len(serialized_metrics):
                n_shape = int(serialized_metrics[i])
                n_values_in_metric = 1
                shape = []
                i += 1
                for arg in serialized_metrics[i:i + n_shape]:
                    n_values_in_metric *= int(arg)
                    shape.append(int(arg))
                n_values_in_metric = int(n_values_in_metric)
                metric = serialized_metrics[i + n_shape:i + n_shape + n_values_in_metric]
                metrics_arrays.append(metric)
                i = i + n_shape + n_values_in_metric
            all_reports.append(metrics_arrays)

        self._report_metrics(all_reports, wandb_run, cumulative_timesteps)

    def _report_metrics(self, collected_metrics, wandb_run, cumulative_timesteps):
        """
        Called once per training iteration with all collected metric arrays.
        `collected_metrics` is a list of lists — one inner list per timestep,
        each inner list contains the arrays returned by _collect_metrics.
        """
        goals = []
        ball_speeds = []
        car_speeds = []
        boosts = []
        touches = []

        for step_arrays in collected_metrics:
            arr = step_arrays[0]
            if len(arr) >= 5:
                goals.append(arr[0])
                ball_speeds.append(arr[1])
                car_speeds.append(arr[2])
                boosts.append(arr[3])
                touches.append(arr[4])

        n_steps = len(goals)
        if n_steps == 0:
            return

        total_goals = sum(goals)
        total_touches = sum(touches)
        avg_ball_speed = np.mean(ball_speeds)
        avg_car_speed = np.mean(car_speeds)
        avg_boost = np.mean(boosts)

        # Print to console (always, regardless of wandb)
        print(f"\n{'='*8} STRIKE METRICS {'='*8}")
        print(f"  Goals this iteration:    {int(total_goals)}")
        print(f"  Ball touches:            {int(total_touches)}")
        print(f"  Avg ball speed:          {avg_ball_speed:.0f} uu/s")
        print(f"  Avg car speed:           {avg_car_speed:.0f} uu/s")
        print(f"  Avg boost:               {avg_boost:.1f} / 100")
        print(f"  Steps this iteration:    {n_steps}")
        print(f"{'='*32}\n")

        # Log to wandb if available
        if wandb_run is not None:
            wandb_run.log({
                "strike/goals": int(total_goals),
                "strike/ball_touches": int(total_touches),
                "strike/avg_ball_speed": avg_ball_speed,
                "strike/avg_car_speed": avg_car_speed,
                "strike/avg_boost": avg_boost,
            })
