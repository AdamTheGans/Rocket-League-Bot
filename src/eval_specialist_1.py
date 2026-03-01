# src/eval_specialist_1.py
from __future__ import annotations

import os
import json
import time
import collections
from dataclasses import dataclass
from typing import Any, Optional, List, Tuple

import numpy as np
import torch

from envs.grounded_strike import build_env


# -----------------------------
# Observation standardization
# -----------------------------
def _load_obs_stats(policy_path: str):
    """
    Load obs_running_stats (mean, var) from the checkpoint's BOOK_KEEPING_VARS.json.
    Returns (mean, std) numpy arrays, or (None, None) if not found.
    """
    step_dir = os.path.dirname(policy_path)
    bk_path = os.path.join(step_dir, "BOOK_KEEPING_VARS.json")
    if not os.path.isfile(bk_path):
        return None, None
    try:
        with open(bk_path, "r", encoding="utf-8") as f:
            bk = json.load(f)
        obs_stats = bk.get("obs_running_stats")
        if obs_stats is None:
            return None, None
        mean = np.asarray(obs_stats["mean"], dtype=np.float32)
        var = np.asarray(obs_stats["var"], dtype=np.float32)
        std = np.sqrt(var)
        # Avoid division by zero: clamp std to a small minimum
        std = np.maximum(std, 1e-6)
        return mean, std
    except Exception as e:
        print(f"Warning: could not load obs stats: {e}")
        return None, None


def _standardize_obs(obs: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Apply the same standardization as BatchedAgentManager."""
    return np.clip((obs - mean) / std, -5.0, 5.0).astype(np.float32)


# -----------------------------
# Checkpoint finding
# -----------------------------
def _find_latest_policy_path(root: str) -> str:
    if os.path.isfile(root) and root.lower().endswith(".pt"):
        return root
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Checkpoint folder not found: {root}")

    candidates = [root]
    for name in os.listdir(root):
        full = os.path.join(root, name)
        if os.path.isdir(full):
            candidates.append(full)

    best_policy: Optional[str] = None
    best_mtime = -1.0

    for run in candidates:
        try:
            subs = os.listdir(run)
        except OSError:
            continue
        for sub in subs:
            sub_path = os.path.join(run, sub)
            if not os.path.isdir(sub_path):
                continue
            policy_path = os.path.join(sub_path, "PPO_POLICY.pt")
            if os.path.isfile(policy_path):
                mtime = os.path.getmtime(policy_path)
                if mtime > best_mtime:
                    best_mtime = mtime
                    best_policy = policy_path

    if best_policy is None:
        raise FileNotFoundError(f"No PPO_POLICY.pt found under: {root}")
    return best_policy


# -----------------------------
# Policy loading / reconstruction
# -----------------------------
def _build_policy_from_state_dict(policy_state, env, policy_path: str | None = None):
    import inspect

    obs_dim = int(env.observation_space.shape[0])
    act_dim = int(env.action_space.n)

    # Auto-detect layer sizes from the state_dict weights
    layer_sizes = [256, 256, 256]  # fallback default

    # Try to infer from state_dict: look for 'model.N.weight' keys
    weight_keys = sorted(
        [k for k in policy_state.keys() if k.startswith("model.") and k.endswith(".weight")],
        key=lambda k: int(k.split(".")[1]),
    )
    if len(weight_keys) >= 2:
        # All hidden layer weights except the last one (which is the output layer)
        hidden_weights = weight_keys[:-1]
        layer_sizes = [policy_state[k].shape[0] for k in hidden_weights]

    DiscreteFF = None
    for import_path in (
        "rlgym_ppo.policy.discrete_policy",
        "rlgym_ppo.ppo.discrete_policy",
        "rlgym_ppo.policy.discrete_ff_policy",
    ):
        try:
            mod = __import__(import_path, fromlist=["DiscreteFF"])
            DiscreteFF = getattr(mod, "DiscreteFF", None)
            if DiscreteFF is not None:
                break
        except Exception:
            pass

    if DiscreteFF is None:
        raise ImportError("Could not import DiscreteFF from rlgym-ppo.")

    sig = inspect.signature(DiscreteFF.__init__)
    params = list(sig.parameters.keys())

    args = [obs_dim, act_dim]
    kwargs = {}

    if "layer_sizes" in params:
        kwargs["layer_sizes"] = layer_sizes
    elif len(params) >= 4:
        args.append(layer_sizes)

    if "device" in params:
        kwargs["device"] = "cpu"

    policy = DiscreteFF(*args, **kwargs)
    policy.load_state_dict(policy_state)
    policy.eval()
    for p in policy.parameters():
        p.requires_grad_(False)
    return policy


def _load_policy(policy_path: str, env):
    obj = torch.load(policy_path, map_location="cpu")

    if isinstance(obj, (dict, collections.OrderedDict)) and all(isinstance(k, str) for k in obj.keys()):
        for key in ("state_dict", "policy_state_dict", "model_state_dict"):
            if key in obj and isinstance(obj[key], (dict, collections.OrderedDict)):
                return _build_policy_from_state_dict(obj[key], env, policy_path=policy_path)
        if any(torch.is_tensor(v) for v in obj.values()):
            return _build_policy_from_state_dict(obj, env, policy_path=policy_path)

    if hasattr(obj, "eval"):
        obj.eval()
        try:
            for p in obj.parameters():
                p.requires_grad_(False)
        except Exception:
            pass
    return obj


# -----------------------------
# Env IO helpers
# -----------------------------
def _reset_env(env) -> np.ndarray:
    out = env.reset()
    obs = out[0] if isinstance(out, tuple) else out
    if isinstance(obs, torch.Tensor):
        obs = obs.detach().cpu().numpy()
    return np.asarray(obs, dtype=np.float32)


def _to_action_for_env(action: Any) -> np.ndarray:
    # wrapper expects array-like and indexes actions[i]; for 1 agent use shape (1,1)
    if isinstance(action, torch.Tensor):
        action = action.detach().cpu().numpy()
    if isinstance(action, np.ndarray):
        a_int = int(action.reshape(-1)[0].item()) if action.size else 0
    else:
        a_int = int(action)
    return np.asarray([[a_int]], dtype=np.int32)


def _unwrap_bool(x: Any) -> bool:
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (list, tuple)) and len(x) > 0:
        return _unwrap_bool(x[0])
    if isinstance(x, np.ndarray):
        return bool(x.reshape(-1)[0]) if x.size else False
    return bool(x)


def _unwrap_float(x: Any) -> float:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return float(x.reshape(-1)[0].item()) if x.size else 0.0
    if isinstance(x, (list, tuple)):
        return float(x[0]) if len(x) else 0.0
    return float(x)


def _step_env(env, action_arr: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
    step_out = env.step(action_arr)
    if isinstance(step_out, tuple) and len(step_out) == 5:
        obs, reward, terminated, truncated, info = step_out
    else:
        obs, reward, done, info = step_out
        terminated, truncated = False, done

    if isinstance(obs, torch.Tensor):
        obs = obs.detach().cpu().numpy()
    obs = np.asarray(obs, dtype=np.float32)

    r = _unwrap_float(reward)
    term = _unwrap_bool(terminated)
    trunc = _unwrap_bool(truncated)
    return obs, r, term, trunc, info


# -----------------------------
# State extraction (robust)
# -----------------------------
def _try_get_yaw_from_car(car) -> Optional[float]:
    """
    Best-effort yaw extraction. Different versions expose different fields.
    Returns yaw radians in world frame if available.
    """
    # Common: car.physics.rotation or car.physics.yaw
    phy = getattr(car, "physics", None)
    if phy is None:
        return None
    for attr in ("yaw", "rotation_yaw"):
        if hasattr(phy, attr):
            try:
                return float(getattr(phy, attr))
            except Exception:
                pass

    rot = getattr(phy, "rotation", None)
    # Sometimes rotation is (pitch,yaw,roll)
    if rot is not None:
        try:
            arr = np.asarray(rot, dtype=np.float32).reshape(-1)
            if arr.size >= 2:
                return float(arr[1])
        except Exception:
            pass
    return None


def _get_state_for_debug(env) -> Optional[dict]:
    """
    Pull car pose + ball pose + boost for top-down gif.
    Works with cars being dict keyed by agent id.
    """
    rlgym_env = getattr(env, "rlgym_env", None)
    if rlgym_env is None:
        return None
    state = getattr(rlgym_env, "state", None)
    if state is None:
        return None

    try:
        car_ids = list(state.cars.keys())
        if not car_ids:
            return None
        car = state.cars[car_ids[0]]

        car_pos = np.asarray(car.physics.position, dtype=np.float32).reshape(-1)[:3]
        ball_pos = np.asarray(state.ball.position, dtype=np.float32).reshape(-1)[:3]

        boost = None
        for attr in ("boost_amount", "boost"):
            if hasattr(car, attr):
                try:
                    boost = float(getattr(car, attr))
                    break
                except Exception:
                    pass

        yaw = _try_get_yaw_from_car(car)

        return {
            "car_pos": car_pos,
            "ball_pos": ball_pos,
            "boost": boost,
            "yaw": yaw,
        }
    except Exception:
        return None


# -----------------------------
# Metrics
# -----------------------------
@dataclass
class EpisodeResult:
    scored: bool
    timeout: bool
    steps: int
    seconds: float
    total_reward: float


def _summarize(results: List[EpisodeResult]) -> dict:
    eps = len(results)
    goal_rate = sum(r.scored for r in results) / eps if eps else 0.0
    timeout_rate = sum(r.timeout for r in results) / eps if eps else 0.0
    secs = np.array([r.seconds for r in results], dtype=np.float32) if eps else np.array([], dtype=np.float32)
    returns = np.array([r.total_reward for r in results], dtype=np.float32) if eps else np.array([], dtype=np.float32)
    ttg = np.array([r.seconds for r in results if r.scored], dtype=np.float32)

    return {
        "episodes": eps,
        "goal_rate": goal_rate,
        "timeout_rate": timeout_rate,
        "median_ep_seconds": float(np.median(secs)) if eps else float("nan"),
        "mean_ep_seconds": float(np.mean(secs)) if eps else float("nan"),
        "avg_return": float(np.mean(returns)) if eps else float("nan"),
        "median_time_to_goal": float(np.median(ttg)) if ttg.size else float("inf"),
        "mean_time_to_goal": float(np.mean(ttg)) if ttg.size else float("inf"),
    }


# -----------------------------
# Pillow top-down GIF (labeled + arrow + boost)
# -----------------------------
def _save_topdown_gif_labeled(
    episodes_states: List[List[dict]],
    out_path: str,
    fps: int = 20,
    title: str = "Grounded Strike Eval",
    attack_orange: bool = True,
):
    from PIL import Image, ImageDraw, ImageFont

    # Flatten + validate
    frames_raw: List[dict] = []
    for ep in episodes_states:
        if ep:
            frames_raw.extend(ep)
    if not frames_raw:
        raise ValueError("No frames captured.")

    good = []
    for f in frames_raw:
        try:
            car = np.asarray(f["car_pos"], dtype=np.float32).reshape(-1)
            ball = np.asarray(f["ball_pos"], dtype=np.float32).reshape(-1)
            if car.size < 2 or ball.size < 2:
                continue
            yaw = f.get("yaw", None)
            boost = f.get("boost", None)
            car_z = float(car[2]) if car.size >= 3 else None
            ball_z = float(ball[2]) if ball.size >= 3 else None
            good.append((float(car[0]), float(car[1]), float(ball[0]), float(ball[1]), yaw, boost, car_z, ball_z))
        except Exception:
            continue

    if len(good) < 2:
        raise ValueError("Not enough valid frames to animate.")

    # Field bounds (approx)
    X_MIN, X_MAX = -4200.0, 4200.0
    Y_MIN, Y_MAX = -5200.0, 5200.0

    W, H = 720, 1080
    PAD = 40

    def world_to_px(x: float, y: float) -> tuple[int, int]:
        u = (x - X_MIN) / (X_MAX - X_MIN)
        v = (y - Y_MIN) / (Y_MAX - Y_MIN)
        px = int(PAD + u * (W - 2 * PAD))
        py = int(PAD + (1.0 - v) * (H - 2 * PAD))
        return px, py

    # Optional font (falls back if unavailable)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
        font_big = ImageFont.truetype("arial.ttf", 24)
    except Exception:
        font = None
        font_big = None

    img_frames: List[Image.Image] = []
    for t, (cx, cy, bx, by, yaw, boost, car_z, ball_z) in enumerate(good):
        img = Image.new("RGB", (W, H), (14, 16, 22))
        draw = ImageDraw.Draw(img)

        # Field outline
        draw.rectangle([PAD, PAD, W - PAD, H - PAD], outline=(190, 190, 190), width=2)

        # Midline
        mid_y = world_to_px(0.0, 0.0)[1]
        draw.line([PAD, mid_y, W - PAD, mid_y], fill=(90, 90, 90), width=2)

        # Goal boxes & labels
        # Orange goal at +Y, Blue goal at -Y
        top_line = world_to_px(0.0, 5120.0)[1]
        bot_line = world_to_px(0.0, -5120.0)[1]

        # draw "goals" as thick bands
        draw.rectangle([PAD, top_line - 8, W - PAD, top_line + 8], fill=(140, 90, 20))
        draw.rectangle([PAD, bot_line - 8, W - PAD, bot_line + 8], fill=(30, 90, 140))

        if font_big:
            draw.text((PAD + 8, PAD + 6), title, fill=(240, 240, 240), font=font_big)
        else:
            draw.text((PAD + 8, PAD + 6), title, fill=(240, 240, 240))

        target_txt = "Target: ORANGE goal (+Y)" if attack_orange else "Target: BLUE goal (-Y)"
        if font:
            draw.text((PAD + 8, PAD + 38), target_txt, fill=(220, 220, 220), font=font)
        else:
            draw.text((PAD + 8, PAD + 38), target_txt, fill=(220, 220, 220))

        # Draw ball and car
        bx_px, by_px = world_to_px(bx, by)
        cx_px, cy_px = world_to_px(cx, cy)

        r_ball = 10
        r_car = 10
        draw.ellipse([bx_px - r_ball, by_px - r_ball, bx_px + r_ball, by_px + r_ball], fill=(255, 170, 40))
        draw.ellipse([cx_px - r_car, cy_px - r_car, cx_px + r_car, cy_px + r_car], fill=(80, 170, 255))

        # Labels with height
        if font:
            ball_label = f"BALL  z={ball_z:.0f}" if ball_z is not None else "BALL"
            car_label = f"CAR  z={car_z:.0f}" if car_z is not None else "CAR"
            draw.text((bx_px + 12, by_px - 10), ball_label, fill=(255, 210, 120), font=font)
            draw.text((cx_px + 12, cy_px - 10), car_label, fill=(160, 210, 255), font=font)

        # Facing arrow
        if yaw is not None and np.isfinite(yaw):
            # In RL coords, yaw=0 usually points +X; in Rocket League, forward vector depends on convention.
            # We'll assume yaw=0 -> +X. Arrow will be "good enough" to debug turning/wall hugging.
            L = 35
            dx = float(np.cos(yaw)) * L
            dy = float(np.sin(yaw)) * L
            end = (int(cx_px + dx), int(cy_px - dy))  # invert y because image coords
            draw.line([cx_px, cy_px, end[0], end[1]], fill=(220, 220, 220), width=3)
            # Arrow head
            draw.ellipse([end[0] - 4, end[1] - 4, end[0] + 4, end[1] + 4], fill=(220, 220, 220))

        # Boost bar
        if boost is not None and np.isfinite(boost):
            # normalize if needed (some versions are 0..100, others 0..1)
            b = float(boost)
            if b > 1.5:
                b = b / 100.0
            b = max(0.0, min(1.0, b))
            bar_x0, bar_y0 = PAD + 8, H - PAD + 8
            bar_w, bar_h = 240, 16
            draw.rectangle([bar_x0, bar_y0, bar_x0 + bar_w, bar_y0 + bar_h], outline=(220, 220, 220), width=2)
            draw.rectangle([bar_x0, bar_y0, bar_x0 + int(bar_w * b), bar_y0 + bar_h], fill=(100, 220, 120))
            if font:
                draw.text((bar_x0 + bar_w + 10, bar_y0 - 2), f"BOOST {int(b*100)}%", fill=(230, 230, 230), font=font)

        # Timestamp-ish
        if font:
            draw.text((W - PAD - 160, PAD + 38), f"frame {t}", fill=(210, 210, 210), font=font)

        img_frames.append(img)

    duration_ms = int(1000 / fps)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img_frames[0].save(
        out_path,
        save_all=True,
        append_images=img_frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )
    print(f"Saved GIF: {out_path} (frames={len(img_frames)})")


# -----------------------------
# Main eval
# -----------------------------
def evaluate(
    checkpoints_root: str,
    n_episodes: int = 200,
    render: bool = False,      # RLViser is optional and often not viable on Windows
    tick_skip: int = 8,
    episode_seconds: float = 12.0,
    deterministic: bool = True,
    print_every_episodes: int = 25,
    record_gif_episodes: int = 2,
    gif_out_path: str = os.path.join("checkpoints", "eval_labeled.gif"),
):
    env = build_env(render=render, tick_skip=tick_skip, episode_seconds=episode_seconds)

    policy_path = _find_latest_policy_path(checkpoints_root)
    print(f"Loading policy: {policy_path}")

    policy = _load_policy(policy_path, env)

    # Load observation standardization stats from checkpoint
    obs_mean, obs_std = _load_obs_stats(policy_path)
    if obs_mean is not None:
        print(f"Loaded obs standardization stats (dim={obs_mean.shape[0]})")
    else:
        print("WARNING: No obs standardization stats found — feeding raw observations")

    print("Evaluation running.")

    sec_per_step = float(tick_skip) / 120.0

    results: List[EpisodeResult] = []
    captured: List[List[dict]] = []

    obs = _reset_env(env)

    ep_states: List[dict] = []
    if len(captured) < record_gif_episodes:
        st = _get_state_for_debug(env)
        if st is not None:
            ep_states.append(st)

    ep_steps = 0
    ep_return = 0.0

    t0 = time.time()
    while len(results) < n_episodes:
        # Standardize obs before inference (matching training pipeline)
        policy_obs = _standardize_obs(obs, obs_mean, obs_std) if obs_mean is not None else obs

        if hasattr(policy, "get_action"):
            with torch.no_grad():
                action, _ = policy.get_action(policy_obs, deterministic=deterministic)
        else:
            with torch.no_grad():
                logits = policy(torch.from_numpy(policy_obs).float().unsqueeze(0))
                action = int(torch.argmax(logits, dim=-1).item())

        action_arr = _to_action_for_env(action)
        obs, r, terminated, truncated, info = _step_env(env, action_arr)

        ep_steps += 1
        ep_return += r

        if len(captured) < record_gif_episodes:
            st = _get_state_for_debug(env)
            if st is not None:
                ep_states.append(st)

        done = terminated or truncated
        if done:
            scored = bool(terminated and not truncated)
            timeout = bool(truncated)
            results.append(
                EpisodeResult(
                    scored=scored,
                    timeout=timeout,
                    steps=ep_steps,
                    seconds=ep_steps * sec_per_step,
                    total_reward=ep_return,
                )
            )

            if len(captured) < record_gif_episodes:
                captured.append(ep_states)

            if print_every_episodes and len(results) % print_every_episodes == 0:
                s = _summarize(results[-print_every_episodes:])
                print(
                    f"[last {print_every_episodes}] "
                    f"goal_rate={s['goal_rate']:.2%} timeout_rate={s['timeout_rate']:.2%} "
                    f"median_ttg={s['median_time_to_goal']:.2f}s "
                    f"median_ep={s['median_ep_seconds']:.2f}s "
                    f"avg_return={s['avg_return']:.3f} "
                    f"episodes={len(results)}/{n_episodes} "
                    f"elapsed={time.time()-t0:.1f}s"
                )

            obs = _reset_env(env)
            ep_steps = 0
            ep_return = 0.0
            ep_states = []

            if len(captured) < record_gif_episodes:
                st = _get_state_for_debug(env)
                if st is not None:
                    ep_states.append(st)

    s = _summarize(results)
    print("\n==== FINAL EVAL SUMMARY ====")
    for k, v in s.items():
        if isinstance(v, float):
            if np.isinf(v):
                print(f"{k}: inf")
            else:
                print(f"{k}: {v:.4f}" if "rate" not in k else f"{k}: {v:.2%}")
        else:
            print(f"{k}: {v}")

    # Save labeled GIF
    if record_gif_episodes > 0 and captured:
        try:
            _save_topdown_gif_labeled(
                captured,
                gif_out_path,
                fps=20,
                title="Grounded Strike (Top-Down Debug)",
                attack_orange=True,
            )
        except Exception as e:
            print(f"Could not save labeled GIF: {e}")


def _discover_checkpoint_root(base_name: str = "grounded_strike") -> str:
    """
    Find the checkpoint root folder, handling rlgym-ppo's timestamped
    folder names (e.g. 'grounded_strike-1772320959983582600').

    If multiple matching runs exist, prompts the user to choose.
    """
    checkpoints_dir = "checkpoints"
    if not os.path.isdir(checkpoints_dir):
        raise FileNotFoundError(f"No '{checkpoints_dir}' directory found.")

    # Find all folders starting with the base name
    matches = []
    for name in sorted(os.listdir(checkpoints_dir)):
        full = os.path.join(checkpoints_dir, name)
        if os.path.isdir(full) and name.startswith(base_name):
            matches.append(full)

    if not matches:
        raise FileNotFoundError(
            f"No checkpoint folders matching '{base_name}*' found in {checkpoints_dir}/.\n"
            f"Contents: {os.listdir(checkpoints_dir)}"
        )

    if len(matches) == 1:
        return matches[0]

    # Multiple runs — prompt user
    print(f"\nFound {len(matches)} checkpoint runs:")
    for i, m in enumerate(matches):
        # Count sub-checkpoints and find latest
        subs = [s for s in os.listdir(m) if os.path.isdir(os.path.join(m, s))]
        num_subs = len(subs)
        latest = max(subs) if subs else "empty"
        print(f"  [{i + 1}] {m}  ({num_subs} checkpoints, latest: {latest})")

    while True:
        try:
            choice = input(f"\nWhich run to evaluate? [1-{len(matches)}]: ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(matches):
                return matches[idx]
        except (ValueError, EOFError):
            pass
        print("Invalid choice, try again.")


if __name__ == "__main__":
    root = _discover_checkpoint_root("grounded_strike")
    evaluate(
        root,
        n_episodes=200,
        render=False,
        record_gif_episodes=2,
        gif_out_path=os.path.join("checkpoints", "eval_labeled.gif"),
    )