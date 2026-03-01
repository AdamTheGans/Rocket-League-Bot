## Long-horizon plan (full project)

### Phase 0 — Tooling + baselines

**Goal:** prove we can train in sim, checkpoint, evaluate, and later run in-game.

* Train in RocketSim via RLGym (fast). RLGym’s quickstart explicitly supports RocketSim and training with rlgym-ppo.
* Use rlgym-ppo as my PPO engine (simple learner interface).
* Optional: set up RLViser to visualize training (RLGym recommends `rlgym[rl-rlviser]`).

### Phase 1 — Specialist 1: **Grounded Strike** (In Progress)

**Goal:** from many randomized *near-ground* starts, score as quickly as possible (no defender initially).

Deliverables:

* Robust reset distribution
* Reward shaping that produces fast, decisive shooting (not dithering)
* Evaluation harness (success rate, time-to-goal, generalization across spawn bins)

### Phase 2 — “Style pipeline” scaffold (even before full game bot)

**Goal:** make “fun-first” decision logic work without needing a perfect main bot.

* Train a **viability predictor** for Specialist 1:

  * inputs: game state
  * outputs: $P(\text{score within }T), P(\text{lose possession})$
* Build a trivial “bot brain”:

  * if $P(\text{score})$ high → use specialist
  * else → basic defend/recover (scripted or a weak learned policy)

This gets us playable behavior quickly.

### Phase 3 — Main gameplay policy (defense/rotation backbone)

**Goal:** a stable “default” policy.

Options:

* Train our own 1v1 self-play PPO as the backbone, **or**
* Start from a public model as a baseline (more below)

A public option we can inspect/download is **CanoPy** (PPO, trained in RLGym + RLBot v5, but it’s **2v2**).
Even if we don’t *use* it as our 1v1 backbone, it’s a good artifact to study for settings (obs/action/reward/hparams).

### Phase 4 — Add flashy specialists (one by one)

Suggested order (highest “fun per pain”):

1. pop → **air dribble**
2. wall setup → aerial touch chain
3. ceiling shot variants
4. flip-reset chains (later)

For each specialist:

* Phase A: no defender (mechanics)
* Phase B: weak defender/noise
* Phase C: learned defender/self-play pressure
* Add/extend viability + bail-out heads

### Phase 5 — Unified “Style Agent”

**Goal:** “go for cool shots whenever plausible; otherwise defend.”

Implementation patterns:

* Mixture-of-Experts (soft gating) or Options (hard switching)
* Shared viability/risk predictor used both:

  * for *selection* (which shot now)
  * for *bail-out* mid-attempt