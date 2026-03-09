import numpy as np
import multiprocessing as mp
from collections import deque
from rlgym_ppo.util import MetricsLogger

class SharedCurriculum:
    """Holds shared memory variables that all CPU workers can read."""
    def __init__(self):
        self.difficulty = mp.Value('d', 0.3)
        self.noise_amount = mp.Value('d', 0.1)

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
        
        self.outcomes = deque(maxlen=self.eval_interval)
        self.episodes_since_eval = 0

    def collect_episode_metrics(self, agents, state, shared_info):
        """Called by rlgym-ppo at the end of an episode."""
        
        # Check if this was a mechanic episode and if we got a success
        success_key = f"{self.mechanic_name}_success"
        
        # Only log mechanic outcomes if it was actually a mechanic setup 
        # (Assuming mixed_training_env sets "setter_type" in shared_info)
        if shared_info.get("setter_type") == self.mechanic_name:
            if success_key in shared_info:
                scored = bool(shared_info[success_key])
                self.outcomes.append(scored)
                self.episodes_since_eval += 1
                
                # Every N mechanic episodes, evaluate curriculum
                if self.episodes_since_eval >= self.eval_interval:
                    self._evaluate_curriculum()
                    self.episodes_since_eval = 0
                
        # Return metrics for rlgym-ppo to log to Wandb/Tensorboard
        log_dict = {
            f"Curriculum/Difficulty": self.shared.difficulty.value,
            f"Curriculum/Noise": self.shared.noise_amount.value,
        }
        
        # Log success rate if we have data
        if len(self.outcomes) > 0:
            log_dict[f"Curriculum/Success_Rate"] = np.mean(self.outcomes)
            
        return log_dict

    def _evaluate_curriculum(self):
        # ... (Keep your exact evaluate logic here, it is completely fine!) ...
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