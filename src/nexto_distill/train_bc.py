# src/nexto_distill/train_bc.py
"""
Behavior-cloning trainer: teaches the student MLP to imitate Nexto
by minimizing cross-entropy loss on (obs, teacher_action) pairs,
with optional KL divergence to teacher logits.

Usage:
    cd src
    python -m nexto_distill.train_bc \
        --data_dir ../data/nexto_distill/shards \
        --epochs 50 \
        --batch_size 4096 \
        --lr 3e-4 \
        --checkpoint_dir ../checkpoints/nexto_distill
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import glob
import time
from typing import Optional, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from nexto_distill.student_policy import StudentPolicy


# ===================================================================== #
#  Dataset
# ===================================================================== #

class DistillationDataset(Dataset):
    """
    Loads obs/actions/logits from .npz shards.
    Supports episode-level train/val splitting.
    """

    def __init__(
        self,
        shard_paths: List[str],
        episode_ids_to_keep: Optional[set] = None,
    ):
        all_obs = []
        all_actions = []
        all_logits = []

        for path in shard_paths:
            data = np.load(path)
            obs = data["obs"]
            actions = data["actions"]
            logits = data["logits"]
            episode_ids = data["episode_ids"]

            if episode_ids_to_keep is not None:
                mask = np.isin(episode_ids, list(episode_ids_to_keep))
                obs = obs[mask]
                actions = actions[mask]
                logits = logits[mask]

            if len(obs) > 0:
                all_obs.append(obs)
                all_actions.append(actions)
                all_logits.append(logits)

        if all_obs:
            self.obs = np.concatenate(all_obs, axis=0)
            self.actions = np.concatenate(all_actions, axis=0)
            self.logits = np.concatenate(all_logits, axis=0)
        else:
            self.obs = np.zeros((0, 1), dtype=np.float32)
            self.actions = np.zeros((0,), dtype=np.int64)
            self.logits = np.zeros((0, 1), dtype=np.float32)

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.obs[idx]).float(),
            torch.tensor(self.actions[idx], dtype=torch.long),
            torch.from_numpy(self.logits[idx]).float(),
        )


def _discover_episode_ids(shard_paths: List[str]) -> set:
    """Collect all unique episode IDs across shards."""
    all_ids = set()
    for path in shard_paths:
        data = np.load(path)
        all_ids.update(data["episode_ids"].tolist())
    return all_ids


def _split_episode_ids(
    all_ids: set, val_fraction: float = 0.1, seed: int = 42
) -> Tuple[set, set]:
    """Split episode IDs into train and val sets."""
    rng = np.random.RandomState(seed)
    ids_sorted = sorted(all_ids)
    rng.shuffle(ids_sorted)
    n_val = max(1, int(len(ids_sorted) * val_fraction))
    val_ids = set(ids_sorted[:n_val])
    train_ids = set(ids_sorted[n_val:])
    return train_ids, val_ids


# ===================================================================== #
#  Training loop
# ===================================================================== #

def train_bc(
    data_dir: str,
    checkpoint_dir: str,
    layer_sizes: List[int],
    lr: float = 3e-4,
    epochs: int = 50,
    batch_size: int = 4096,
    kl_weight: float = 0.0,
    val_fraction: float = 0.1,
    seed: int = 42,
    device: str = "cpu",
    amp: bool = False,
    patience: int = 0,
    num_workers: int = 0,
    extra_data_dirs: Optional[List[str]] = None,
):
    print("=" * 60)
    print("  NEXTO DISTILLATION — BEHAVIOR CLONING TRAINING")
    print("=" * 60)

    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device(device)

    # ── Load shard paths ──
    shard_paths = sorted(glob.glob(os.path.join(data_dir, "shard_*.npz")))
    if not shard_paths:
        raise FileNotFoundError(f"No shard_*.npz files found in {data_dir}")
    print(f"  Found {len(shard_paths)} shard files in {data_dir}")

    # ── Load metadata ──
    meta_path = os.path.join(data_dir, "metadata.json")
    if os.path.isfile(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        obs_dim = meta["student_obs_dim"]
        num_actions = meta["num_actions"]
        lut_hash = meta.get("lut_hash", "unknown")
        print(f"  Metadata: obs_dim={obs_dim}, num_actions={num_actions}, lut_hash={lut_hash}")
    else:
        # Infer from data
        sample = np.load(shard_paths[0])
        obs_dim = sample["obs"].shape[1]
        num_actions = sample["logits"].shape[1]
        lut_hash = "unknown"
        print(f"  Inferred: obs_dim={obs_dim}, num_actions={num_actions}")

    # ── Split by episode ──
    all_episodes = _discover_episode_ids(shard_paths)
    train_episodes, val_episodes = _split_episode_ids(all_episodes, val_fraction, seed)
    print(f"  Episodes: {len(all_episodes)} total, "
          f"{len(train_episodes)} train, {len(val_episodes)} val")

    train_ds = DistillationDataset(shard_paths, train_episodes)
    val_ds = DistillationDataset(shard_paths, val_episodes)
    print(f"  Primary:  {len(train_ds):,} train, {len(val_ds):,} val")

    # ── Load extra data dirs (e.g., DAgger shards) — train only ──
    if extra_data_dirs:
        for extra_dir in extra_data_dirs:
            extra_shards = sorted(glob.glob(os.path.join(extra_dir, "shard_*.npz")))
            if extra_shards:
                extra_ds = DistillationDataset(extra_shards)  # No episode filtering
                print(f"  Extra ({extra_dir}): {len(extra_ds):,} samples added to train")
                # Merge into train_ds
                train_ds.obs = np.concatenate([train_ds.obs, extra_ds.obs], axis=0)
                train_ds.actions = np.concatenate([train_ds.actions, extra_ds.actions], axis=0)
                train_ds.logits = np.concatenate([train_ds.logits, extra_ds.logits], axis=0)

    print(f"  Total:    {len(train_ds):,} train, {len(val_ds):,} val")

    if len(train_ds) == 0:
        raise RuntimeError("No training samples! Check your data_dir and shard files.")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device == "cuda"),
    )

    # ── Build student ──
    print(f"  Student layers: {layer_sizes}")
    model = StudentPolicy(obs_dim, num_actions, layer_sizes).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler(enabled=amp)
    ce_loss_fn = nn.CrossEntropyLoss()

    print(f"\n  lr={lr}, batch_size={batch_size}, kl_weight={kl_weight}, AMP={amp}")
    print(f"  Patience={patience} (0=disabled)")
    print()

    os.makedirs(checkpoint_dir, exist_ok=True)
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # ── Train ──
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        for obs, actions, logits_teacher in train_loader:
            obs = obs.to(dev)
            actions = actions.to(dev)
            logits_teacher = logits_teacher.to(dev)

            with torch.amp.autocast(device_type=device, enabled=amp):
                logits_student = model(obs)

                # Cross-entropy to teacher's argmax
                loss = ce_loss_fn(logits_student, actions)

                # Optional KL divergence
                if kl_weight > 0:
                    log_probs_student = F.log_softmax(logits_student, dim=-1)
                    probs_teacher = F.softmax(logits_teacher, dim=-1)
                    kl = F.kl_div(
                        log_probs_student, probs_teacher,
                        reduction="batchmean", log_target=False,
                    )
                    loss = loss + kl_weight * kl

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss_sum += loss.item() * obs.size(0)
            preds = logits_student.argmax(dim=-1)
            train_correct += (preds == actions).sum().item()
            train_total += obs.size(0)

        scheduler.step()

        train_loss = train_loss_sum / train_total
        train_acc = train_correct / train_total

        # ── Validate ──
        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_correct_top5 = 0
        val_total = 0
        val_entropy_sum = 0.0

        with torch.no_grad():
            for obs, actions, logits_teacher in val_loader:
                obs = obs.to(dev)
                actions = actions.to(dev)
                logits_teacher = logits_teacher.to(dev)

                logits_student = model(obs)

                loss = ce_loss_fn(logits_student, actions)
                if kl_weight > 0:
                    log_probs_student = F.log_softmax(logits_student, dim=-1)
                    probs_teacher = F.softmax(logits_teacher, dim=-1)
                    kl = F.kl_div(
                        log_probs_student, probs_teacher,
                        reduction="batchmean", log_target=False,
                    )
                    loss = loss + kl_weight * kl

                val_loss_sum += loss.item() * obs.size(0)

                preds = logits_student.argmax(dim=-1)
                val_correct += (preds == actions).sum().item()

                # Top-5 accuracy
                _, top5 = logits_student.topk(5, dim=-1)
                val_correct_top5 += (top5 == actions.unsqueeze(-1)).any(dim=-1).sum().item()

                # Entropy of student distribution
                probs = F.softmax(logits_student, dim=-1)
                ent = -(probs * (probs + 1e-8).log()).sum(dim=-1)
                val_entropy_sum += ent.sum().item()

                val_total += obs.size(0)

        val_loss = val_loss_sum / val_total if val_total > 0 else float("inf")
        val_acc = val_correct / val_total if val_total > 0 else 0.0
        val_acc5 = val_correct_top5 / val_total if val_total > 0 else 0.0
        val_entropy = val_entropy_sum / val_total if val_total > 0 else 0.0

        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.2%} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.2%} val_acc5={val_acc5:.2%} "
            f"ent={val_entropy:.3f} | "
            f"lr={lr_now:.2e} | {elapsed:.1f}s"
        )

        # ── Checkpointing ──
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            # Save best model
            ckpt_path = os.path.join(checkpoint_dir, "student_policy.pt")
            torch.save(model.state_dict(), ckpt_path)

            # Save metadata
            ckpt_meta = {
                "obs_dim": obs_dim,
                "num_actions": num_actions,
                "layer_sizes": layer_sizes,
                "lut_hash": lut_hash,
                "best_val_loss": best_val_loss,
                "best_val_acc": val_acc,
                "best_val_acc5": val_acc5,
                "best_epoch": epoch,
                "total_params": n_params,
                "lr": lr,
                "kl_weight": kl_weight,
                "seed": seed,
            }
            meta_out = os.path.join(checkpoint_dir, "metadata.json")
            with open(meta_out, "w") as f:
                json.dump(ckpt_meta, f, indent=2)

            print(f"  ✓ New best val_loss={val_loss:.4f} — saved to {ckpt_path}")
        else:
            patience_counter += 1
            if patience > 0 and patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch} (patience={patience})")
                break

    print(f"\n{'=' * 60}")
    print(f"  TRAINING COMPLETE")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Checkpoint: {os.path.join(checkpoint_dir, 'student_policy.pt')}")
    print(f"{'=' * 60}")


# ===================================================================== #
#  CLI
# ===================================================================== #

def main():
    parser = argparse.ArgumentParser(description="Train student via behavior cloning.")
    parser.add_argument(
        "--data_dir", type=str,
        default=os.path.join("..", "data", "nexto_distill", "shards"),
    )
    parser.add_argument(
        "--checkpoint_dir", type=str,
        default=os.path.join("..", "checkpoints", "nexto_distill"),
    )
    parser.add_argument("--layers", type=int, nargs="+", default=[2048, 1024, 1024, 512])
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--kl_weight", type=float, default=0.0,
                        help="Weight for KL divergence to teacher logits (0=CE only)")
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--amp", action="store_true", help="Use mixed precision")
    parser.add_argument("--patience", type=int, default=0,
                        help="Early stopping patience (0=disabled)")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--extra_data_dirs", type=str, nargs="*", default=None,
                        help="Extra shard directories (e.g. DAgger data) to add to training")

    args = parser.parse_args()

    train_bc(
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        layer_sizes=args.layers,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        kl_weight=args.kl_weight,
        val_fraction=args.val_fraction,
        seed=args.seed,
        device=args.device,
        amp=args.amp,
        patience=args.patience,
        num_workers=args.num_workers,
        extra_data_dirs=args.extra_data_dirs,
    )


if __name__ == "__main__":
    main()
