# src/metrics/pinch_metrics.py
"""
Custom MetricsLogger for the Pinch specialist.

Tracks pinch-relevant signals: goalward speed spikes, ball-wall distance,
and standard metrics (goals, touches, speeds, boost).
"""
from __future__ import annotations

import collections
import csv
import os
import numpy as np
import numpy as np
from rlgym_ppo.util import MetricsLogger
from rlgym_ppo.util.rlgym_v2_gym_wrapper import RLGymV2GymWrapper

from rewards.pinch_reward import GLOBAL_REWARD_BREAKDOWN
from rewards.pinch_reward import GLOBAL_REWARD_BREAKDOWN

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
        "timesteps", "episodes", "goals", "goal_rate_pct",
        "ball_touches", "touches_per_ep",
        "spikes_50kph", "spikes_75kph", "spikes_100kph", "spikes_125kph",
        "spikes_50kph_pct", "spikes_75kph_pct", "spikes_100kph_pct", "spikes_125kph_pct",
        "avg_ball_speed", "max_goalward_spike", "max_spike_kph", "avg_goalward_speed",
        "avg_wall_dist", "avg_car_speed", "avg_boost", "steps", "sim_hours",
    ]

    TRACKED_REWARDS = [
        "QuickGoal", "GoalwardSpeedSpike", "ZFilteredGoalwardSpike", "LatchGoalwardSpeedSpike",
        "BallVelocityToGoal", "BallWallProximity", "ApproachPinchPoint", "Touch", "TimePenalty"
    ]

    def __init__(self, csv_path: str = "checkpoints/pinch_metrics.csv",
                 tick_skip: int = 4, timeout_seconds: float = 2.0, stage: int = 1):
        super().__init__()
        self.csv_path = csv_path
        self.tick_skip = tick_skip
        self.timeout_seconds = timeout_seconds
        self.stage = stage
        self._prev_gw = None
        # Seconds per decision step
        self._sec_per_step = tick_skip / 120.0
        # Max steps per episode (for estimating episode count)
        self._max_steps_per_ep = int(timeout_seconds / self._sec_per_step)

        # Moving averages for auto-progression
        self.recent_goal_rates = collections.deque(maxlen=10)
        self.recent_spike_50 = collections.deque(maxlen=10)
        self.recent_spike_75 = collections.deque(maxlen=10)
        self.recent_spike_100 = collections.deque(maxlen=10)
        self.recent_spike_125 = collections.deque(maxlen=10)

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
            touched = 0.0

        gw = goalward_speed
        if self._prev_gw is None or getattr(game_state, 'tick_count', 1) == 0:
            delta = 0.0
        else:
            delta = max(0.0, gw - self._prev_gw)
        self._prev_gw = gw

        reward_vals = [float(GLOBAL_REWARD_BREAKDOWN.get(name, 0.0)) for name in self.TRACKED_REWARDS]

        return [np.array([
            goal_scored_flag, ball_speed, goalward_speed,
            ball_wall_dist, car_speed, boost, touched, delta,
            *reward_vals
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

        component_rewards = {name: [] for name in self.TRACKED_REWARDS}

        for step_arrays in collected_metrics:
            arr = step_arrays[0]
            if len(arr) >= 8:
                goals.append(arr[0])
                ball_speeds.append(arr[1])
                goalward_speeds.append(arr[2])
                wall_dists.append(arr[3])
                car_speeds.append(arr[4])
                boosts.append(arr[5])
                touches.append(arr[6])
                goalward_spikes.append(arr[7])
                
            if len(arr) >= 8 + len(self.TRACKED_REWARDS):
                for i, name in enumerate(self.TRACKED_REWARDS):
                    component_rewards[name].append(arr[8 + i])

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
        # Calculate max spike per episode bucket to prevent 1000% metrics from sustained speeding
        steps_per_ep = max(1, n_steps // est_episodes)
        ep_max_spikes = []
        for i in range(est_episodes):
            start_idx = i * steps_per_ep
            end_idx = min((i + 1) * steps_per_ep, len(goalward_spikes))
            ep_range = goalward_spikes[start_idx:end_idx]
            if ep_range:
                ep_max_spikes.append(max(ep_range))
            
        spike_tiers = [
            ("50kph",  1400.0),
            ("75kph",  2100.0),
            ("100kph", 2800.0),
            ("125kph", 3500.0),
        ]
        spike_counts = {}
        for label, thresh in spike_tiers:
            count = sum(1 for max_s in ep_max_spikes if max_s > thresh)
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
        print(f"{'='*32}")
        
        if component_rewards[self.TRACKED_REWARDS[0]]:
            print(f"  {'='*7} AVG REWARDS (PER STEP) {'='*7}")
            for name in self.TRACKED_REWARDS:
                avg_val = float(np.mean(component_rewards[name]))
                print(f"  {name:25s} {avg_val:.4f}")
        print(f"{'='*32}\n")

        self._append_csv(
            cumulative_timesteps, est_episodes, total_goals, goal_rate,
            total_touches, touches_per_ep, spike_counts,
            avg_ball_speed, max_gw_spike, avg_goalward,
            avg_wall_dist, avg_car_speed, avg_boost, n_steps, sim_hours,
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
                
            if component_rewards[self.TRACKED_REWARDS[0]]:
                for name in self.TRACKED_REWARDS:
                    log_dict[f"reward/{name}"] = float(np.mean(component_rewards[name]))
                    
            wandb_run.log(log_dict)

        # Update moving averages and check auto-progression
        self.recent_goal_rates.append(goal_rate)
        self.recent_spike_50.append(spike_counts["50kph"][1])
        self.recent_spike_75.append(spike_counts["75kph"][1])
        self.recent_spike_100.append(spike_counts["100kph"][1])
        self.recent_spike_125.append(spike_counts["125kph"][1])
        
        if len(self.recent_goal_rates) == 10:
            avg_goal_rate = sum(self.recent_goal_rates) / 10
            avg_50 = sum(self.recent_spike_50) / 10
            avg_75 = sum(self.recent_spike_75) / 10
            avg_100 = sum(self.recent_spike_100) / 10
            avg_125 = sum(self.recent_spike_125) / 10
            
            if self.stage == 1:
                if avg_50 > 50.0 and avg_75 > 15.0:
                    print(f"\n{'='*60}")
                    print(f"  🎉 STAGE {self.stage} MASTERED! Model meets mastery criteria!")
                    print(f"{'='*60}\n")
            elif self.stage == 2:
                if avg_50 > 95.0 and avg_75 > 75.0 and avg_100 > 50.0 and avg_125 > 25.0 and avg_goal_rate > 50.0:
                    print(f"\n{'='*60}")
                    print(f"  🎉 STAGE {self.stage} MASTERED! Model meets mastery criteria!")
                    print(f"{'='*60}\n")

    def _append_csv(self, timesteps, episodes, goals, goal_rate,
                    touches, touches_per_ep, spike_counts,
                    ball_speed, max_spike, avg_goalward,
                    wall_dist, car_speed, boost, steps, sim_hours):
        file_exists = os.path.isfile(self.csv_path)
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)

        # Extract spike tier data in order
        tier_counts = [spike_counts[l][0] for l in ["50kph", "75kph", "100kph", "125kph"]]
        tier_pcts = [f"{spike_counts[l][1]:.1f}" for l in ["50kph", "75kph", "100kph", "125kph"]]
        max_spike_kph = max_spike * 0.036

        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(self.CSV_HEADER)
            writer.writerow([
                timesteps, episodes, goals, f"{goal_rate:.1f}",
                touches, f"{touches_per_ep:.1f}",
                *tier_counts, *tier_pcts,
                f"{ball_speed:.1f}", f"{max_spike:.1f}", f"{max_spike_kph:.1f}",
                f"{avg_goalward:.1f}",
                f"{wall_dist:.1f}", f"{car_speed:.1f}", f"{boost:.1f}",
                steps, f"{sim_hours:.2f}",
            ])
