# src/train_specialist_1_gpu.py
"""
Optimized training config for 16-core + RTX 3080 + 32GB RAM.
Use this instead of train_specialist_1.py on the friend's PC.
"""
from __future__ import annotations

import os
from rlgym_ppo import Learner

from envs.grounded_strike import build_env
from metrics.strike_metrics import GroundedStrikeLogger


def env_factory():
    return build_env(render=False, tick_skip=8, episode_seconds=12.0)


if __name__ == "__main__":
    os.makedirs("checkpoints", exist_ok=True)

    # ── Monkey-patch: fix rlgym-ppo kbhit crash on Windows ──
    try:
        import rlgym_ppo.util.kbhit as _kbhit
        _original_getch = _kbhit.KBHit.getch

        def _safe_getch(self):
            try:
                return _original_getch(self)
            except UnicodeDecodeError:
                return ""

        _kbhit.KBHit.getch = _safe_getch
    except Exception:
        pass  # Linux doesn't use msvcrt, no patch needed

    # 16 physical cores — use 20 env processes
    n_proc = 20
    min_inference_size = max(1, int(round(n_proc * 0.9)))

    learner = Learner(
        env_factory,
        n_proc=n_proc,
        min_inference_size=min_inference_size,
        metrics_logger=GroundedStrikeLogger(),
        timestep_limit=500_000_000,            # 500M steps (overnight run)
        save_every_ts=1_000_000,               # save every 1M
        checkpoints_save_folder=os.path.join("checkpoints", "grounded_strike"),
        # Bigger network — 3080 handles inference easily
        policy_layer_sizes=[512, 512, 256],
        critic_layer_sizes=[512, 512, 256],
        # PPO — larger batches for more stable gradients
        ppo_batch_size=100_000,
        ts_per_iteration=100_000,
        exp_buffer_size=300_000,               # 3x batch
        ppo_minibatch_size=50_000,             # 2 minibatches per epoch
        ppo_epochs=2,
        policy_lr=1e-4,
        critic_lr=1e-4,
        ppo_ent_coef=0.005,
        standardize_returns=True,
        standardize_obs=False,
    )

    learner.learn()
