# Rocket League Bot — Setup Guide

## What This Is

A Rocket League bot trained with reinforcement learning (PPO) using RLGym v2 + RocketSim.
Currently training two specialists:
- **Grounded Strike**: A 1v0 agent that learns to score from near-ground spawns.
- **Pinch Specialist**: A 1v0 agent that masters the side wall Kuxir pinch through 3 automated curriculum stages.

---

## Quick Setup

### 1. Install Python 3.10+

**Windows:** Download from [python.org](https://www.python.org/downloads/). Check **"Add Python to PATH"** during install.

**Linux (Ubuntu):**
```bash
sudo apt update && sudo apt install python3 python3-venv python3-pip
```

### 2. Clone / Copy the Project

Copy this entire folder to the target machine.

### 3. Create Virtual Environment

**Windows (PowerShell):**
```powershell
cd C:\path\to\Rocket-League-Bot
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Linux:**
```bash
cd /path/to/Rocket-League-Bot
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install PyTorch with CUDA

**GPU machines:**
```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

**CPU-only machines:**
```bash
pip install torch
```

### 5. Install RLGym + RocketSim + PPO engine

```bash
pip install rlgym[rl-sim,rl-rlviser]
pip install rlviser-py
pip install git+https://github.com/AechPro/rlgym-ppo
```

### 6. Install remaining dependencies

```bash
pip install -r requirements.txt
```

### 6. Verify Installation

**Windows:**
```powershell
python src\verify_env.py
```

**Linux:**
```bash
python src/verify_env.py
```

You should see `=== ALL TESTS PASSED ===` at the end.

---

## Training

### On the GPU Machine

**Windows:**
```powershell
.venv\Scripts\Activate.ps1
python src\train_specialist_1_gpu.py
```

**Linux:**
```bash
source .venv/bin/activate
python src/train_specialist_1_gpu.py
```

This uses the optimized config:
- `n_proc=20` (20 parallel RocketSim environments)
- `[512, 512, 256]` network (GPU handles this easily)
- `100,000` batch size
- `500M` timestep limit (~6-8 hours on GPU)

### On the Laptop (6 cores, CPU only)

**Windows:**
```powershell
.venv\Scripts\Activate.ps1
python src\train_specialist_1.py
```

**Linux:**
```bash
source .venv/bin/activate
python src/train_specialist_1.py
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

**Windows:**
```powershell
.venv\Scripts\Activate.ps1
python src\eval_specialist_1.py
```

**Linux:**
```bash
source .venv/bin/activate
python src/eval_specialist_1.py
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
│   │   ├── grounded_strike.py      # 1v0 strike environment setup
│   │   ├── pinch.py                # 1v0 pinch environment setup
│   │   └── resets/
│   │       └── generate_golden_seed.py # Generates precise pinch setups and visualizes them using rlviser
│   ├── rewards/
│   │   ├── strike_reward.py        # Strike reward function
│   │   └── pinch_reward.py         # Pinch reward function
│   ├── state_setters/
│   │   ├── low_spawn_setter.py     # Strike spawn positions
│   │   └── pinch_spawn_setter.py   # Pinch spawn positions / offset geometry
│   ├── metrics/
│   │   ├── strike_metrics.py       # Custom strike metrics
│   │   └── pinch_metrics.py        # Custom pinch metrics with auto-progression
│   ├── train_specialist_1.py       # Training config (laptop/CPU)
│   ├── train_specialist_1_gpu.py   # Training config (GPU machine)
│   ├── eval_specialist_1.py        # Strike Evaluation + GIF generation
│   ├── train_pinch.py              # Pinch training config (supports --stage auto-progression)
│   ├── eval_pinch.py               # Pinch Evaluation + GIF generation
│   ├── test_golden_seed_pinch.py   # Renders generated pinch simulations natively in 3D RLViser
│   ├── verify_env.py               # Quick strike sanity test
│   └── verify_pinch_env.py         # Quick pinch sanity test
├── checkpoints/                    # Saved model checkpoints
├── requirements.txt
├── PLAN.md                         # Full project roadmap
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