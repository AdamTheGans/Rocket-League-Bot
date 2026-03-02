# Training Guide -- Pinch Specialist v0 (Kuxir / Corner Wall Pinch)

This guide walks through training the pinch specialist from scratch using **backward chaining**: learn the contact micro-skill first, then extend to approach, then to live-ish game states.

---

## Prerequisites

Same environment as the grounded-strike specialist. If you already ran `README.md` setup, you're good.

```bash
# Verify environment
python src/verify_pinch_env.py
# Expected: === ALL PINCH ENV TESTS PASSED ===
```

---

## Stage 1: Micro-Skill (Near-Contact Pinch)

**Goal:** Learn to execute the pinch contact itself -- ball is flush on the wall, car is 50-250 uu away and pre-aligned. Agent only needs to learn flip/boost timing for a goalward speed spike.

**Spawn:** Ball nearly touching the wall (gap <= ball radius), car 50-250 uu from ball, aggressively pre-aligned (+-0.15 rad yaw noise), 5% corner probability. 2s episodes.

### Train

```powershell
# Default (20 procs)
python src\train_pinch.py --stage 1

# GPU
python src\train_pinch.py --stage 1 --gpu

# Custom proc count
python src\train_pinch.py --stage 1 --n-proc 8 --seed 42
```

### What to Watch For

| Metric | Bad Sign | Good Sign |
|---|---|---|
| Goals per iteration | 0 for 10+ iterations | Steadily climbing from ~5M steps |
| Max goalward spike | Near 0 | >1500 uu/s regularly |
| Avg goalward speed | Negative or near 0 | Positive and growing |
| Avg ball-wall dist | Increasing (running away) | Staying low (~100-200 uu) |

### When to Move On

- **Goal rate > 15%** on eval
- **Median max goalward spike > 2000 uu/s** in metrics
- **~20-50M steps** typical (a few hours on GPU)

### Evaluate

```powershell
python src\eval_pinch.py --stage 1
```

---

## Stage 2: Approach (1-2 Seconds Pre-Contact)

**Goal:** Learn to drive toward the ball-wall contact point and execute the pinch from farther away, with more noise in angle and speed.

**Spawn:** Car 600-1500 uu from ball, +-0.5 rad yaw deviation, 15% corner probability, episodes up to 4s.

### Train

```powershell
# Start fresh
python src\train_pinch.py --stage 2

# Resume from Stage 1 checkpoint (recommended)
python src\train_pinch.py --stage 2 --resume-from checkpoints\pinch_stage1

# GPU variant
python src\train_pinch.py --stage 2 --gpu --resume-from checkpoints\pinch_stage1
```

> **TIP:** Resuming from Stage 1 gives the agent a head start -- it already knows the contact timing. The Stage 2 reward adds approach shaping to help it learn the drive-in.

### What to Watch For

- Early on (first 5M steps): goal rate will drop from Stage 1 because spawns are harder. This is expected.
- Agent should learn to drive toward the pinch point within ~10M steps.
- Watch for "just ramming the ball without wall contact" -- if avg wall distance stays high, the ball proximity shaping may need tuning.

### When to Move On

- **Goal rate > 10%** on eval
- Agent consistently approaches wall before contact (visible in GIF)
- **~30-80M steps**

---

## Stage 3: Live-ish Pinch-Ready States

**Goal:** Self-initiate a wall pinch from "normal-ish" game states -- ball in offensive half near a wall, car not pre-aligned.

**Spawn:** Car 800-2500 uu from ball, full random yaw, 30% corner probability, episodes up to 6s.

### Train

```powershell
# Resume from Stage 2 (strongly recommended)
python src\train_pinch.py --stage 3 --resume-from checkpoints\pinch_stage2

# GPU
python src\train_pinch.py --stage 3 --gpu --resume-from checkpoints\pinch_stage2
```

### What to Watch For

- Goal rate will drop significantly when first moving to Stage 3 (wider spawns).
- Look for the agent developing a "read" -- turning toward the wall when ball is close to it.
- **Failure mode:** agent ignores the ball and drives to the wall. If this happens, increase the approach reward weight or add a speed-toward-ball term.

### When to Move On

- **Goal rate > 5%** on eval with Stage 3 resets
- Agent can score from various angles and positions
- **~50-100M+ steps** (this stage requires the most training)

---

## Key Hyperparameters

| Parameter | Fresh Start | Resuming from Previous Stage |
|---|---|---|
| `policy_lr` | `1e-4` | `5e-5` |
| `critic_lr` | `1e-4` | `5e-5` |
| `ppo_ent_coef` | `0.01` | `0.008` |
| `batch_size` (CPU) | `50,000` | `50,000` |
| `batch_size` (GPU) | `100,000` | `100,000` |

---

## Checkpoint Management

Checkpoints are saved per-stage:
```
checkpoints/
  pinch_stage1/           # Stage 1 checkpoints
  pinch_stage2/           # Stage 2 checkpoints
  pinch_stage3/           # Stage 3 checkpoints
```

To continue training on a different machine, copy the entire stage folder.

> **IMPORTANT:** All stages use the same network architecture (`[512, 256, 256]`) so checkpoints are compatible when resuming from a previous stage.

---

## Troubleshooting

| Problem | Likely Cause | Fix |
|---|---|---|
| 0 goals for 20+ iters | Sparse reward too far away | Verify Stage 1 spawns are tight. Check `verify_pinch_env.py` output. |
| Ball flies toward own goal | Reward not penalizing backwards hits | Increase `BallVelocityToGoalReward` weight. |
| Agent dribbles instead of pinching | Touch reward too high | Reduce TouchReward, increase GoalwardSpeedSpikeReward weight. |
| Agent drives past ball into wall | Not enough guidance toward ball | Add SpeedTowardBallReward to the reward config. |
| High speed spikes but no goals | Good execution, aim is off | May need more training or goalward reward tuning. |
| Loss spikes after stage transition | LR too high for fine-tuning | Use `--resume-from` (auto-sets lr=5e-5). |

---

## Full Pipeline Summary

```
Stage 1 (micro) --> Stage 2 (approach) --> Stage 3 (live)
    20-50M steps       30-80M steps          50-100M+ steps
    2s episodes        4s episodes           6s episodes
    50-250uu offset    600-1500uu offset     800-2500uu offset
    5% corner          15% corner            30% corner
```

After Stage 3, the policy is ready for integration testing with the viability model (Phase B from PLAN.md).
