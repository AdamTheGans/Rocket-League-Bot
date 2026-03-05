# Training Guide -- Pinch Specialist v0 (Upright Aerial Wall Pinch)

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
- When thresholds are met, the script continuously prints a **Stage Mastery Banner** to the console to alert the user that the stage's goals are fully learned. Training continues infinitely until manually stopped with `CTRL+C`, allowing the user to seamlessly graduate the checkpoints at their discretion.
- The logger also includes a precise `AVERAGE REWARDS (PER STEP)` display in the iteration report, generated using a global `GLOBAL_REWARD_BREAKDOWN` dictionary bypassing serialization constraints.

---

## Reward Computation & Breakdown (Per-Step)

The `GLOBAL_REWARD_BREAKDOWN` dictionary is used to extract the raw, per-step reward component values natively from the isolated Rocket League environments. It averages out the component metrics per step, taking the mathematical delta *per-environment* to precisely quantify goalward speed spikes without artificial variance. 

### Stage Weightings:
| Component | Stage 1 | Stage 2 | Stage 3 | Description |
|---|---|---|---|---|
| **QuickGoal** | *Disabled* | `100.0` | `100.0` | Sparse reward for actually scoring. Base: 1.0, Bonus: +0.5. |
| **ZFilteredGoalwardSpike** | `150.0` | `150.0` | `3.0` | Dense reward tracking latch max goalward ball velocity, penalized linearly if out of `[300, 550] uu/s` Z-velocity bounds. |
| **BallVelocityToGoal** | *Disabled* | `0.4` | `0.5` | Dense shaping for keeping the ball moving toward the goal continuously. |
| **ApproachPinchPoint** | `0.05` | `0.2` | `0.3` | Shaping to guide the car exactly to the projected contact point on the wall. |
| **BallWallProximity** | `0.05` | `0.1` | `0.1` | Shaping to keep ball near the wall (prevents runaway play). |
| **Touch** | `5.0` | `0.05` | `0.05` | Flat reward for touching the ball. Strictly granted *exactly once* per episode to prevent dribbling exploits. |
| **TimePenalty** | `0.0` | `-0.04` | `-0.05` | Constant negative step penalty (disabled in Stage 1). |

---

## Stage 1: Micro-Skill (Near-Contact Pinch)

**Goal:** Learn to execute the pinch contact itself -- ball is rolling up the side wall, car is positioned in an upright aerial trajectory directly intercepting the ball. Agent only needs to learn front/diagonal dodge timing for a goalward speed spike.

**Spawn:** Ball rolling up the *Right Wall* (+X) with low forward momentum, car precisely mapped to intercept trajectory. Spawn includes fully independent `[X, Y, Z]` noise vectors for tracking positions and velocities to cleanly prevent trajectory overfitting. It applies a random sliding Y-variance up to 3000uu and a 50% chance to mirror across the X-axis for left wall training. The mathematical intercept strictly forces the car deep into the wall's geometric bounding box (X offset up to -120.0uu) and exactly aligns the center-of-mass Z-coordinate to the ball's center, explicitly forcing RocketSim's impulse collision engine to calculate a high-velocity pinch on the next discrete frame. A dedicated `generate_golden_seed.py` random grid search runs to refine these impact permutations alongside dodge angles, and `test_golden_seed_pinch.py` is used to visualize the exact seed and impact physics. 2s episodes.

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

### When to Move On (Manual)

The script will automatically print a mastery banner when the 10-iteration rolling average meets:
- **50kph Spikes > 50.0%**
- **75kph Spikes > 15.0%**

### Evaluate

```powershell
python src\eval_pinch.py --stage 1
```

---

## Stage 2: Approach (1-2 Seconds Pre-Contact)

**Goal:** Learn to drive toward the ball-wall contact point and execute the pinch from farther away, with more noise in angle and speed.

**Spawn:** `time_to_impact` is drawn from a uniform `[0.3, 0.6]` second distribution. The ball is precisely back-propagated along its curved wall-roll trajectory using `RocketSimEngine`. The car is pulled backwards sequentially along its velocity vector by `time_to_impact * 0.8`, with a geometric gravity multiplier applied to its Z-coordinate to perfectly mimic its fall-curve over that duration. 

**30% Flat Spawns:** To organically force the bot to learn mid-air rotations rather than relying exclusively on its golden seed upright-roll geometry, 30% of Stage 2 spawns natively override the car's Euler roll to `0.0` (with tiny noise), spawning the car completely flat and forcing an intentional mechanical correction. Episodes up to 4s.

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

### When to Move On (Manual)

The script will automatically print a mastery banner when the 10-iteration rolling average meets:
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
  pinch_stage1-17726.../  # Run folder with timestamp
    checkpoints/
      7001994/...         # Highest timestep save
```

To continue training on a different machine, copy the entire stage's subfolders. The internal script's runtime path-resolution natively supports dynamic `--resume-from` paths on all 3 tiers of depth:
- `--resume-from checkpoints/pinch_stage1` (Auto-scans and matches the prefix with the highest global timestep run folder)
- `--resume-from checkpoints/pinch_stage1-1772.../` (Descends into the precise run folder to pick the highest internal timestep)
- `--resume-from checkpoints/pinch_stage1-1772.../7001994/` (Strict targeted load)

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
