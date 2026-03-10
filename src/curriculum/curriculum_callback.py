import queue
import numpy as np
import multiprocessing as mp
from collections import deque
from rlgym_ppo.util import MetricsLogger
from multiprocessing.managers import SyncManager

class SharedCurriculum:
    """Holds shared memory proxy variables that all CPU workers can safely read/write."""
    def __init__(self, manager: SyncManager):
        # Using Manager proxies allows these to be safely pickled for Windows workers
        self.difficulty = manager.Value('d', 0.3)
        self.noise_amount = manager.Value('d', 0.1)
        
        # A thread-safe queue to pass success/failure booleans from workers to the main thread
        self.outcomes_queue = manager.Queue()

class CurriculumMetricsLogger(MetricsLogger):
    """
    rlgym-ppo MetricsLogger that handles curriculum progression and logging.
    """
    def __init__(
        self, 
        shared_curriculum: SharedCurriculum,
        mechanic_name: str = "kuxir",
        eval_interval_episodes: int = 500,
    ):
        super().__init__()
        self.shared = shared_curriculum
        self.mechanic_name = mechanic_name
        self.eval_interval = eval_interval_episodes
        
        # Curriculum hyperparameters
        self.promote_threshold = 0.80
        self.demote_threshold = 0.40
        self.noise_step = 0.05
        self.difficulty_step = 0.02
        self.noise_gate = 0.3
        
        # Local state (only updated in the main process)
        self.outcomes = deque(maxlen=self.eval_interval)
        self.episodes_since_eval = 0

    def _collect_metrics(self, game_state):
        """
        REQUIRED BY RLGYM-PPO.
        Runs in the worker processes on EVERY step.
        We return an empty list because we are tracking successes via the Queue instead.
        """
        return []

    def _report_metrics(self, collected_metrics, wandb_run, cumulative_timesteps):
        """
        REQUIRED BY RLGYM-PPO.
        Runs in the main process at the end of every batch.
        """
        # 1. Drain the queue to get all outcomes sent by the workers during this batch
        while not self.shared.outcomes_queue.empty():
            try:
                # get_nowait() is non-blocking
                scored = bool(self.shared.outcomes_queue.get_nowait())
                self.outcomes.append(scored)
                self.episodes_since_eval += 1
                
                # Every N mechanic episodes, evaluate curriculum
                if self.episodes_since_eval >= self.eval_interval:
                    self._evaluate_curriculum()
                    self.episodes_since_eval = 0
            except queue.Empty:
                break

        # 2. Package the metrics for rlgym-ppo to log to Wandb/Console
        log_dict = {
            f"Curriculum/Difficulty": self.shared.difficulty.value,
            f"Curriculum/Noise": self.shared.noise_amount.value,
        }
        
        # Log success rate if we have data
        if len(self.outcomes) > 0:
            log_dict[f"Curriculum/Success_Rate"] = float(np.mean(self.outcomes))
            
        # Natively log to wandb if enabled
        if wandb_run is not None:
            wandb_run.log(log_dict, step=cumulative_timesteps)
            
        # Add this print statement!
        print(f"\n[CUSTOM METRICS] {log_dict}\n")
        
        # Returning the dict prints it nicely to the terminal!
        return log_dict

    def _evaluate_curriculum(self):
        # Your exact curriculum math!
        success_rate = np.mean(self.outcomes)
        old_noise = self.shared.noise_amount.value
        old_diff = self.shared.difficulty.value

        if success_rate > self.promote_threshold:
            new_noise = min(1.0, old_noise + self.noise_step)
            self.shared.noise_amount.value = new_noise
            
            if new_noise >= self.noise_gate:
                self.shared.difficulty.value = min(1.0, old_diff + self.difficulty_step)
                
        elif success_rate < self.demote_threshold:
            self.shared.noise_amount.value = max(0.1, old_noise - self.noise_step * 0.5)
            self.shared.difficulty.value = max(0.3, old_diff - self.difficulty_step * 0.5)

        print(f"\n[Curriculum] Eval! Success: {success_rate:.1%} | Noise: {self.shared.noise_amount.value:.3f} | Diff: {self.shared.difficulty.value:.3f}\n")