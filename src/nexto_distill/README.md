# Nexto Distillation Pipeline

Distill the pre-trained **Nexto** bot into a flat MLP student policy via behavior cloning. The student can then be used as initialization for PPO fine-tuning.

## Prerequisites

- Python venv with rlgym, rlgym-ppo, torch, numpy installed
- `nexto/nexto-model.pt` present in the repo root

## Pipeline

### Step 1: Generate Dataset

Roll out Nexto as teacher in a 1v1 RocketSim environment and collect `(student_obs, teacher_action, teacher_logits)` tuples.

```bash
cd src
python -m nexto_distill.generate_dataset \
    --out_dir ../data/nexto_distill/shards \
    --num_steps 1000000 \
    --shard_size 50000 \
    --seed 42 \
    --device cpu
```

**Key flags:**
- `--num_steps` — total environment steps to collect
- `--shard_size` — steps per `.npz` shard file
- `--device cpu/cuda` — where to run Nexto inference
- `--episode_seconds 30` — episode timeout
- `--report_every 10000` — progress print interval

Output: `data/nexto_distill/shards/shard_XXXXX.npz` + `metadata.json`

### Step 2: Train Student (Behavior Cloning)

```bash
cd src
python -m nexto_distill.train_bc \
    --data_dir ../data/nexto_distill/shards \
    --layers 2048 1024 1024 512 \
    --lr 3e-4 \
    --epochs 50 \
    --batch_size 4096 \
    --checkpoint_dir ../checkpoints/nexto_distill \
    --seed 42
```

**Key flags:**
- `--layers` — hidden layer sizes for the MLP
- `--kl_weight 1.0` — optional KL divergence to teacher logits
- `--amp` — enable mixed precision training
- `--patience 10` — early stopping patience (0=disabled)
- `--device cuda` — GPU training

Output: `checkpoints/nexto_distill/student_policy.pt` + `metadata.json`

### Step 3: Evaluate

**Offline** (accuracy on held-out validation data):
```bash
cd src
python -m nexto_distill.eval_imitation \
    --checkpoint ../checkpoints/nexto_distill/student_policy.pt \
    --mode offline \
    --data_dir ../data/nexto_distill/shards
```

**Online** (student plays in RocketSim, compared to teacher):
```bash
cd src
python -m nexto_distill.eval_imitation \
    --checkpoint ../checkpoints/nexto_distill/student_policy.pt \
    --mode online \
    --episodes 100
```

**With RLViser visualization:**
```bash
cd src
python -m nexto_distill.eval_imitation \
    --checkpoint ../checkpoints/nexto_distill/student_policy.pt \
    --mode online \
    --episodes 5 \
    --visualize
```

## Architecture

```
nexto_distill/
├── __init__.py
├── compat_adapter.py      # rlgym v2 GameState → Nexto encoded format
├── teacher_nexto.py        # Loads Nexto model, exposes act()/get_logits()
├── student_policy.py       # Configurable MLP (obs → action logits)
├── generate_dataset.py     # RocketSim rollout + shard saving
├── train_bc.py             # Behavior cloning trainer
├── eval_imitation.py       # Offline + online evaluation
└── README.md
```

## Troubleshooting

- **"Nexto model not found"**: Ensure `nexto/nexto-model.pt` exists
- **Low entropy in action report**: Teacher is stuck in degenerate behavior — check spawn randomization
- **Import errors**: Run from `src/` directory and ensure venv is activated
- **OOM during training**: Reduce `--batch_size` or use `--amp`
