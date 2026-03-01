# Rocket League Bot — Setup Guide

## What This Is

A Rocket League bot trained with reinforcement learning (PPO) using RLGym v2 + RocketSim.
Currently training the "Grounded Strike" specialist — a 1v0 agent that learns to score from near-ground spawns.

---

## Quick Setup (Windows)

### 1. Install Python 3.10+

Download from [python.org](https://www.python.org/downloads/).
Make sure to check **"Add Python to PATH"** during installation.

### 2. Clone / Copy the Project

Copy this entire folder to the target machine.

### 3. Create Virtual Environment

Open PowerShell in the project root:

```powershell
cd C:\path\to\Rocket-League-Bot
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 4. Install PyTorch with CUDA

**For GPU machines (RTX 3080, etc.):**

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

**For CPU-only machines:**

```powershell
pip install torch
```

### 5. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 6. Verify Installation

```powershell
python src\verify_env.py
```

You should see `=== ALL TESTS PASSED ===` at the end.

---

## Training

### On the GPU Machine (16 cores + 3080)

```powershell
.venv\Scripts\Activate.ps1
python src\train_specialist_1_gpu.py
```

This uses the optimized config:
- `n_proc=20` (20 parallel RocketSim environments)
- `[512, 512, 256]` network (GPU handles this easily)
- `100,000` batch size
- `500M` timestep limit (~6-8 hours on 16-core + 3080)

### On the Laptop (6 cores, CPU only)

```powershell
.venv\Scripts\Activate.ps1
python src\train_specialist_1.py
```

### Keyboard Controls During Training

- `p` — Pause (any key to resume)
- `c` — Save checkpoint
- `q` — Save checkpoint and quit

### Resuming Training

The learner auto-resumes from the latest checkpoint in `checkpoints/grounded_strike/`.
Just run the same command again.

---

## Evaluation

```powershell
.venv\Scripts\Activate.ps1
python src\eval_specialist_1.py
```

This auto-finds the latest checkpoint, runs 200 episodes, and saves a
top-down debug GIF to `checkpoints/eval_labeled.gif`.

The GIF shows car position, ball position, facing direction, boost level,
and height (z-coordinate) for both car and ball.

---

## Project Structure

```
Rocket-League-Bot/
├── src/
│   ├── envs/
│   │   └── grounded_strike.py      # 1v0 environment setup
│   ├── rewards/
│   │   └── strike_reward.py        # Reward function
│   ├── state_setters/
│   │   └── low_spawn_setter.py     # Spawn positions/rotations
│   ├── metrics/
│   │   └── strike_metrics.py       # Custom training metrics
│   ├── train_specialist_1.py       # Training config (laptop/CPU)
│   ├── train_specialist_1_gpu.py   # Training config (GPU machine)
│   ├── eval_specialist_1.py        # Evaluation + GIF generation
│   └── verify_env.py               # Quick sanity test
├── checkpoints/                    # Saved model checkpoints
├── requirements.txt
└── README.md
```

---

## Transferring Checkpoints

To continue training on a different machine, copy the entire
`checkpoints/grounded_strike/` folder. The learner will auto-detect
and resume from the latest checkpoint.

**Note:** If switching between different `policy_layer_sizes` configs
(e.g., `[512,256,256]` on laptop vs `[512,512,256]` on GPU machine),
you must start training fresh — checkpoints are not compatible across
different network architectures.