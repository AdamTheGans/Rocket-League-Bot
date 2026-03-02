# Rocket League Freestyle Agent — Technical Plan

## 0. Project goal

Build an offline Rocket League bot that prioritizes **style-first scoring**:
- It attempts flashy “cool goals” whenever they are *actually viable*.
- Otherwise it plays safe defense/recovery so it doesn’t constantly concede.

Core architecture (long-term):
- **Backbone policy**: full-gameplay PPO (defense/rotation/recoveries/50s).
- **Specialist policies**: mechanic experts (ground strike, air dribble, pinch, ceiling plays, etc.).
- **Viability + risk model**: estimates `P(score | state, skill, params)` and `P(concede | ...)` to:
  - choose which skill to attempt,
  - decide when to bail out mid-attempt.

Training stack:
- Train primarily in **RocketSim via RLGym** for speed.
- PPO engine: **rlgym-ppo**.
- Optional references/baselines: public self-play PPO agents (e.g., CanoPy) for configs/ideas. CanoPy is PPO via rlgym-ppo, uses `DefaultObs`, lookup-table actions, action repeat 8, and combined reward components like speed-toward-ball and ball-velocity-toward-goal. (2v2).  

## 1. Key design principles

1) **Robustness > flashiness early**
   Specialists must work from a broad distribution of entry states, or the high-level controller will almost never call them.

2) **Train by curriculum + backward chaining**
   For thin-manifold mechanics (pinches, flip resets), train “last 10–20 frames” first, then move resets earlier.

3) **Always include a termination/bail-out mechanism**
   Each specialist must expose a clean “abort/terminate” signal so the agent can hand back control.

4) **Evaluation is a first-class feature**
   Every milestone includes:
   - held-out reset seeds,
   - success rate,
   - median time-to-goal,
   - concede risk (when defenders exist),
   - regression tests.

## 2. Potential Repository structure

Rocket-League-Bot/
  PLAN.md
  README.md
  pyproject.toml (or requirements.txt)
  src/
    envs/
      make_env.py
      obs.py
      actions.py
      rewards/
        grounded_strike.py
        pinch.py
        shared.py
      resets/
        grounded_strike.py
        pinch.py
      terminations.py
    train/
      train_grounded_strike.py
      train_backbone_selfplay.py
      train_pinch.py
      train_viability.py
    eval/
      eval_skill.py
      eval_backbone.py
      eval_matchups.py
    runtime/
      selector.py              # chooses skill / params
      bail_out.py              # mid-attempt abort logic
      policy_wrapper.py        # loads models, action repeat, etc.
    util/
      seeding.py
      checkpoints.py
      logging.py
      metrics.py

## 3. Milestones and phases

### Phase A — Pipeline + Specialist #1 (Grounded Strike)
Objective: learn to score **as fast as possible** from broad near-ground spawns (1v0).

A1) Environment + tooling
- Install:
  - `pip install rlgym[rl-sim]`
  - `pip install git+https://github.com/AechPro/rlgym-ppo`
  - Optional: `pip install rlgym[rl-rlviser]`
- Implement env factory: RocketSim, 1 agent, 0 opponents, standard goal.

A2) Reset distribution (near-ground)
- Randomize car pose (x,y,yaw), boost amount.
- Randomize ball (x,y,z small), small velocities/spin.
- Curriculum: start easy (ball in front), then widen offsets, angles, distances.

A3) Action + observation
- Start with a simple, learnable action interface:
  - lookup-table/discrete actions + action repeat (common in community PPO setups).
- Observations:
  - Default normalized kinematics (positions/velocities/angles/boost).

A4) Reward shaping ("score fast" without hacking)
- Large sparse: goal.
- Dense: ball velocity toward opponent goal, player velocity toward ball, touch quality.
- Time pressure: per-step penalty or explicit time-to-goal objective.
- Terminate:
  - goal,
  - timeout (10–20s),
  - optional fail conditions (own-goal / ball far behind).

A5) Evaluation harness
Report:
- goal rate on held-out seeds,
- median time-to-goal,
- failure modes (stalls, misses, own-goals).
Gate for success:
- high goal rate across broad spawns,
- stable time-to-goal improvements,
- no degenerate reward hacking.

Deliverable: `grounded_strike.pt` (or equivalent checkpoint) + eval report JSON.

### Phase B — Viability model v0 (for Specialist #1)
Objective: predict when to attempt a strike (and when to abort).

B1) Generate rollouts
- Sample diverse resets.
- Run the specialist and collect:
  - did score within T seconds,
  - did lose possession,
  - (later) did concede.

B2) Train a predictor
- Inputs: compact state features (car/ball relative pose, velocities, boost).
- Outputs:
  - `P(score in T)`,
  - `P(failure in T)` (lose possession / concede risk when applicable).

B3) Add runtime selector v0
- If `P(score)` above threshold -> call specialist.
- Else -> fallback behavior (initially scripted defense/recovery).

Deliverable: `viability_grounded_strike.pt` + runtime selector demo.

### Phase C — Backbone gameplay PPO (1v1 self-play)
Objective: a stable default controller that defends and doesn’t throw.

C1) Train 1v1 self-play PPO in RocketSim
- Opponent sampling / past-policy pool.
- Reward: balanced for defense, possession, and scoring (avoid pure ball-chasing).
- Eval: ELO vs policy pool, concede rate, kickoff stability.

C2) Optional: compare configs to public agents (e.g., CanoPy) for hparams/obs/action choices.

Deliverable: `backbone_selfplay.pt` + matchup evals.

### Phase D — Add “signature weapons” specialists
Order (recommended):
D1) Pop → Air Dribble specialist
D2) Pinch specialist v0 (kuxir/corner pinch first)
D3) Ceiling plays (ceiling shot / ceiling double tap)
D4) Flip reset family

For each specialist:
- Train no-defender -> weak defender -> learned defender.
- Add/extend viability + bail-out.
- Ensure robust entry state distribution.

### Phase E — Unified Style Agent
Objective: “go for cool shots when viable, otherwise defend.”

E1) Decision policy
- Option A: Mixture-of-Experts gating (soft blend of experts).
- Option B: Options framework (hard skill switching with termination).

E2) Online bail-out
- Recompute viability during attempt.
- Abort to backbone if success probability drops or concede risk rises.

E3) Final evaluation
- Style metrics:
  - percent of possessions resulting in an attempt,
  - diversity of shots,
  - highlight score (custom).
- Competitive metrics:
  - winrate vs bot pool,
  - concede rate,
  - average boost waste.

## 4. “Best bot” strategy
To credibly pursue top-tier performance:
- Keep backbone training strong and continuously evaluated.
- Add one differentiating specialist (pinch) early, but only deploy when viability is high.
- Avoid brittle “always go for it” behavior by enforcing concede-risk thresholds.



## Plan for Specialist: Pinch v0 (Kuxir-style wall/corner pinch)

### Goal
Train a specialist that can produce a **goalward speed spike** via a wall/corner pinch:
- reliable from a controlled but non-trivial distribution of “pinch-ready” states
- later extendable to coarse aim (left/center/right) and defender pressure
- includes a clean **abort/terminate** rule so the runtime can bail out safely

### Why this approach
Pinches are rare under random exploration. We make them learnable via:
1) **Backward chaining** (train near-contact first)
2) **Reset engineering** (pinch-ready distributions)
3) **Directional reward** (speed spike must be goalward, not just fast)

---

## Stage 0: Setup
- Fix car hitbox for all future training (recommended: Plank/Paladin if pinch-focused).
- Use RocketSim via RLGym for speed.
- Use rlgym-ppo (same pipeline as Grounded Strike).
- Action space: keep the same lookup-table/discrete actions + action repeat as before.

---

## Stage 1: Micro-skill (near-contact pinch)
### Reset distribution (tight)
Spawn states ~10–20 frames before the pinch contact.
Example (conceptual):
- Ball: near side wall in offensive half, low-to-mid height (tuned), low random vel/spin.
- Car: already near wall, oriented toward pinch point, moderate forward speed, small yaw/pitch noise.
- Start with “easy” geometry: ball close to wall and close to car contact path.

### Reward (anti-hacking)
Let g be unit vector toward opponent goal.
- r_goal: big on scoring
- r_spike: reward positive **increase** in (ball_vel · g) right after contact (speed spike, goalward)
- r_align: small shaping for keeping ball motion in a forward cone (ball_vel · g > 0)
- r_bad: penalty if ball_vel · g < 0 (danger clear toward own half)
- r_time: small per-step penalty (encourage immediate execution)

### Termination
- goal
- timeout (short, e.g., 1–2 seconds sim time)
- optionally terminate if ball becomes clearly non-goalward for too long

### Success metrics
- % episodes with a speed spike above threshold AND goalward
- median goalward speed after contact
- % actual goals within T

---

## Stage 2: Approach skill (1–2 seconds pre-contact)
### Reset distribution (medium)
Spawn earlier:
- Car a bit farther from wall
- Ball still pinch-ready but more variance
- Add more noise to yaw/speed and ball position
Objective: learn the approach + jump/flip timing + contact.

### Reward
Same core, but add gentle shaping:
- speed toward pinch point
- maintain controllable approach (avoid chaotic jumps)
Keep spike-to-goalward as the “main” signal.

### Termination
- goal
- timeout 2–4 seconds
- abort condition if setup is clearly lost

---

## Stage 3: Live-ish pinch-ready states (broad)
### Reset distribution (broad but still “pinchable”)
- Ball in offensive half near side wall/corner with wider ranges
- Car on ground, not pre-aligned; add more random yaw and offset
- Ball velocities broader, include slight rolls up wall

Goal: the policy can self-initiate the pinch from “normal-ish” possession states near wall.

### Reward
Same, plus optional penalty for overcommits:
- if car ends far out of position with ball going the wrong way, penalize.

---

## Stage 4 (later): Coarse aim + pressure
- Add a discrete target label: left/center/right.
- Condition the policy on target.
- Reward spike toward target direction.
- Add weak defender or concede-risk shaping.

---

## Runtime integration v0
- Use a simple viability model:
  P(score in T | state, pinch_skill)
  P(concede in T | ...)
- Attempt pinch only if:
  P(score) - λ * P(concede) > threshold
- During attempt, recompute every N frames:
  if metric drops -> bail out to backbone / recovery.