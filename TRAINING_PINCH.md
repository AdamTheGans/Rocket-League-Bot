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

## Automated Stage Progression & Metrics

The `train_pinch.py` script employs an automated curriculum system. It evaluates a 10-iteration rolling average of performance metrics mapping goal rates and specifically thresholded speed spikes (e.g., % of 50kph/75kph hits computed per-environment step to prevent false positives).
- When thresholds are met, the script raises a `StageCompleteException` (inheriting from `BaseException` to bypass standard PPO crash handlers) which cleanly exits training, saves a checkpoint, and instantaneously instantiates the next stage's `Learner`.
- The logger also includes a precise `AVERAGE REWARDS (PER STEP)` display in the iteration report, generated using a global `GLOBAL_REWARD_BREAKDOWN` dictionary bypassing serialization constraints.

---

## Reward Computation & Breakdown (Per-Step)

The `GLOBAL_REWARD_BREAKDOWN` dictionary is used to extract the raw, per-step reward component values natively from the isolated Rocket League environments. It averages out the component metrics per step, taking the mathematical delta *per-environment* to precisely quantify goalward speed spikes without artificial variance. 

### Stage Weightings:
| Component | Stage 1 | Stage 2 | Stage 3 | Description |
|---|---|---|---|---|
| **QuickGoal** | *Disabled* | `100.0` | `100.0` | Sparse reward for actually scoring. Base: 1.0, Bonus: +0.5. |
| **GoalwardSpeedSpike** | `15.0` | `150.0` | `3.0` | Dense reward for positive *increases* (delta) in goalward ball velocity. |
| **BallVelocityToGoal** | *Disabled* | `0.4` | `0.5` | Dense shaping for keeping the ball moving toward the goal continuously. |
| **ApproachPinchPoint** | `0.05` | `0.2` | `0.3` | Shaping to guide the car exactly to the projected contact point on the wall. |
| **BallWallProximity** | `0.05` | `0.1` | `0.1` | Shaping to keep ball near the wall (prevents runaway play). |
| **Touch** | `0.1` | `0.05` | `0.05` | Flat reward for touching the ball. Heavily down-scaled to prevent dribbling exploits. |
| **TimePenalty** | `-0.03` | `-0.04` | `-0.05` | Constant negative step penalty to encourage immediate execution. |

---

## Stage 1: Micro-Skill (Near-Contact Pinch)

**Goal:** Learn to execute the pinch contact itself -- ball is perfectly flush on the side wall, car is positioned linearly to match Z-velocity for a clean intercept. Agent only needs to learn flip/boost timing for a goalward speed spike.

**Spawn:** Ball perfectly flush on the side wall (right wall, +X), car precisely mapped to intercept trajectory with matching Z-velocity. Spawn includes a random +/- 1000.0 uu sliding Y-variance, and a 50% chance to mirror across the X-axis to the left wall to prevent positional overfitting. 2s episodes.

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

### When to Move On (Automated)

The script will automatically transition to Stage 2 when the 10-iteration rolling average meets:
- **50kph Spikes > 50.0%**
- **75kph Spikes > 15.0%**

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

### When to Move On (Automated)

The script will automatically transition to Stage 3 when the 10-iteration rolling average meets:
- **50kph Spikes > 95.0%**
- **75kph Spikes > 75.0%**
- **100kph Spikes > 50.0%**
- **125kph Spikes > 25.0%**
- **Goal rate > 50.0%**

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

Stage 3 natively runs indefinitely as it represents live-ish "mastery". The training loop will gracefully hit the timestep limit (200M+) unless manually stopped.

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

> **IMPORTANT:** All stages use the same scaled-up network architecture (`[1024, 1024, 512, 512]`) so checkpoints are compatible when resuming from a previous stage seamlessly.

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
