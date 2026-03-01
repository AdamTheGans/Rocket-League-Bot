# src/train_specialist_1.py
from __future__ import annotations

import os
from rlgym_ppo import Learner

from envs.grounded_strike import build_env
from metrics.strike_metrics import GroundedStrikeLogger


def env_factory():
    # render False during training; tick_skip=8 is the default in build_env
    return build_env(render=False, tick_skip=8, episode_seconds=12.0)


if __name__ == "__main__":
    os.makedirs("checkpoints", exist_ok=True)

    # ── Monkey-patch: fix rlgym-ppo kbhit crash on Windows ──
    # msvcrt.getch() returns b'\xe0' for special keys (arrows, F-keys),
    # which can't be decoded as UTF-8. The library doesn't handle this.
    import rlgym_ppo.util.kbhit as _kbhit
    _original_getch = _kbhit.KBHit.getch

    def _safe_getch(self):
        try:
            return _original_getch(self)
        except UnicodeDecodeError:
            return ""

    _kbhit.KBHit.getch = _safe_getch

    # n_proc=4 for 6-core laptop (leaves 2 cores for OS + main process)
    n_proc = 32
    min_inference_size = max(1, int(round(n_proc * 0.9)))

    learner = Learner(
        env_factory,
        n_proc=n_proc,
        min_inference_size=min_inference_size,
        metrics_logger=GroundedStrikeLogger(),
        timestep_limit=200_000_000,
        save_every_ts=500_000,
        checkpoints_save_folder=os.path.join("checkpoints", "grounded_strike"),
        # Network — same arch, weights loaded from 100M checkpoint
        policy_layer_sizes=[512, 256, 256],
        critic_layer_sizes=[512, 256, 256],
        # PPO — standard settings
        ppo_batch_size=50_000,
        ts_per_iteration=50_000,
        exp_buffer_size=150_000,   # 3x batch size
        ppo_minibatch_size=50_000,
        ppo_epochs=2,
        # ── Resume tuning ──
        # Lower LR (was 1e-4): prevents large updates from destabilizing
        # the value function while it adapts to the new reward scale
        policy_lr=5e-5,
        critic_lr=5e-5,
        # Higher entropy (was 0.005): helps escape the dribble basin
        # by encouraging exploration of new behaviors (power shots)
        ppo_ent_coef=0.01,
        standardize_returns=True,
        standardize_obs=False,
    )

    learner.learn()