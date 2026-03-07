# src/replay_pipeline/visualize_pinch.py
"""
Kuxir Pinch Visualization Tool
===============================
Reads a .npy file produced by extract_pinches.py and renders an animated
2D (or 3D) matplotlib plot showing:

  - Rocket League pitch outline (walls, goals, center circle)
  - Ball trajectory (blue → red gradient)
  - Car trajectory (green → yellow gradient)
  - Pinch moment marker (star)
  - Animated playback with trailing paths

Usage:
    python src/replay_pipeline/visualize_pinch.py extracted_mechanics/some_replay.npy
    python src/replay_pipeline/visualize_pinch.py extracted_mechanics/some_replay.npy --3d
    python src/replay_pipeline/visualize_pinch.py extracted_mechanics/some_replay.npy --speed 2.0

Per-frame layout (28 floats):
    0-2   Ball pos (x,y,z)        9-11  Car pos (x,y,z)
    3-5   Ball vel (vx,vy,vz)     12-14 Car vel (vx,vy,vz)
    6-8   Ball angvel              15-17 Car angvel
                                  18-21 Car quat (w,x,y,z)
                                  22-24 Car euler (p,y,r)
    25    Car boost               26    Car on_ground
    27    Elapsed time (s)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch
from matplotlib.collections import LineCollection

# ─────────────────────────────────────────────────────────────────────────────
# Rocket League field dimensions (Unreal Units)
# ─────────────────────────────────────────────────────────────────────────────
FIELD_HALF_X = 4096.0   # Side walls at ±4096
FIELD_HALF_Y = 5120.0   # Backboard / goal at ±5120
GOAL_HALF_W  = 893.0    # Goal opening half-width
GOAL_DEPTH   = 880.0    # Goal depth behind the backboard
CENTER_RADIUS = 1024.0  # Center circle radius


def draw_pitch_2d(ax):
    """Draw a top-down 2D Rocket League pitch outline."""
    # Field boundary
    ax.plot(
        [-FIELD_HALF_X, FIELD_HALF_X, FIELD_HALF_X, -FIELD_HALF_X, -FIELD_HALF_X],
        [-FIELD_HALF_Y, -FIELD_HALF_Y, FIELD_HALF_Y, FIELD_HALF_Y, -FIELD_HALF_Y],
        color="white", linewidth=1.5, zorder=1
    )

    # Center line
    ax.plot([-FIELD_HALF_X, FIELD_HALF_X], [0, 0],
            color="white", linewidth=0.8, linestyle="--", alpha=0.4, zorder=1)

    # Center circle
    circle = Circle((0, 0), CENTER_RADIUS, fill=False,
                     edgecolor="white", linewidth=0.8, alpha=0.4, zorder=1)
    ax.add_patch(circle)

    # Goals (simple rectangles behind the backboard)
    for y_sign in [-1, 1]:
        color = "#3B82F6" if y_sign < 0 else "#F97316"  # Blue vs Orange
        goal_y = y_sign * FIELD_HALF_Y
        goal_rect = Rectangle(
            (-GOAL_HALF_W, goal_y if y_sign > 0 else goal_y - GOAL_DEPTH),
            2 * GOAL_HALF_W, GOAL_DEPTH,
            linewidth=1.5, edgecolor=color, facecolor=color, alpha=0.15, zorder=1
        )
        ax.add_patch(goal_rect)

    ax.set_facecolor("#1a1a2e")
    ax.set_aspect("equal")
    ax.set_xlim(-FIELD_HALF_X - 500, FIELD_HALF_X + 500)
    ax.set_ylim(-FIELD_HALF_Y - 500, FIELD_HALF_Y + 500)
    ax.tick_params(colors="gray", labelsize=7)
    for spine in ax.spines.values():
        spine.set_color("gray")


def draw_pitch_3d(ax):
    """Draw a simplified 3D pitch outline."""
    # Floor rectangle
    xs = [-FIELD_HALF_X, FIELD_HALF_X, FIELD_HALF_X, -FIELD_HALF_X, -FIELD_HALF_X]
    ys = [-FIELD_HALF_Y, -FIELD_HALF_Y, FIELD_HALF_Y, FIELD_HALF_Y, -FIELD_HALF_Y]
    zs = [0, 0, 0, 0, 0]
    ax.plot(xs, ys, zs, color="white", linewidth=1.0, alpha=0.5)

    # Side walls (partial, just corners)
    wall_h = 2044.0
    for x in [-FIELD_HALF_X, FIELD_HALF_X]:
        for y in [-FIELD_HALF_Y, FIELD_HALF_Y]:
            ax.plot([x, x], [y, y], [0, wall_h],
                    color="white", linewidth=0.5, alpha=0.3)

    ax.set_xlim(-FIELD_HALF_X - 500, FIELD_HALF_X + 500)
    ax.set_ylim(-FIELD_HALF_Y - 500, FIELD_HALF_Y + 500)
    ax.set_zlim(0, 2500)
    ax.set_facecolor("#1a1a2e")
    ax.tick_params(colors="gray", labelsize=6)


def find_pinch_frame(data: np.ndarray) -> int:
    """
    Find the pinch moment in the extracted data by looking for the
    sharpest ball velocity spike.
    """
    speeds = np.linalg.norm(data[:, 3:6], axis=1)
    if len(speeds) < 6:
        return len(speeds) // 2

    # Compute speed delta over a 5-frame lookback
    deltas = np.zeros_like(speeds)
    for i in range(5, len(speeds)):
        deltas[i] = speeds[i] - speeds[i - 5]

    return int(np.argmax(deltas))


def visualize_static_2d(data: np.ndarray, title: str = ""):
    """Static 2D top-down plot with full trajectories."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 10), dpi=100)
    fig.patch.set_facecolor("#0f0f23")
    draw_pitch_2d(ax)

    ball_x, ball_y = data[:, 0], data[:, 1]
    car_x, car_y = data[:, 9], data[:, 10]
    pinch_idx = find_pinch_frame(data)

    # Ball trajectory: blue → red gradient
    ball_points = np.column_stack([ball_x, ball_y]).reshape(-1, 1, 2)
    ball_segments = np.concatenate([ball_points[:-1], ball_points[1:]], axis=1)
    ball_colors = plt.cm.coolwarm(np.linspace(0, 1, len(ball_segments)))
    ball_lc = LineCollection(ball_segments, colors=ball_colors, linewidths=2.5, zorder=3)
    ax.add_collection(ball_lc)

    # Car trajectory: green → yellow gradient
    car_points = np.column_stack([car_x, car_y]).reshape(-1, 1, 2)
    car_segments = np.concatenate([car_points[:-1], car_points[1:]], axis=1)
    car_colors = plt.cm.summer(np.linspace(0, 1, len(car_segments)))
    car_lc = LineCollection(car_segments, colors=car_colors, linewidths=2.0, zorder=3)
    ax.add_collection(car_lc)

    # Start markers
    ax.plot(ball_x[0], ball_y[0], "o", color="#60A5FA", markersize=10,
            zorder=5, label="Ball start")
    ax.plot(car_x[0], car_y[0], "s", color="#34D399", markersize=10,
            zorder=5, label="Car start")

    # Pinch moment marker
    ax.plot(ball_x[pinch_idx], ball_y[pinch_idx], "*", color="#FBBF24",
            markersize=20, zorder=6, label=f"Pinch (frame {pinch_idx})")

    # End markers
    ax.plot(ball_x[-1], ball_y[-1], "o", color="#EF4444", markersize=8,
            zorder=5, label="Ball end")
    ax.plot(car_x[-1], car_y[-1], "s", color="#F59E0B", markersize=8,
            zorder=5, label="Car end")

    # Info text
    ball_speed_at_pinch = np.linalg.norm(data[pinch_idx, 3:6])
    time_span = data[-1, 27] - data[0, 27]
    info = (f"Frames: {len(data)} | Span: {time_span:.2f}s | "
            f"Pinch speed: {ball_speed_at_pinch:.0f} uu/s")
    ax.set_title(f"{title}\n{info}", color="white", fontsize=11, pad=10)

    ax.legend(loc="upper left", fontsize=8, facecolor="#1a1a2e",
              edgecolor="gray", labelcolor="white")

    plt.tight_layout()
    plt.show()


def visualize_animated_2d(data: np.ndarray, speed: float = 1.0, title: str = ""):
    """Animated 2D playback with trailing paths."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 10), dpi=100)
    fig.patch.set_facecolor("#0f0f23")
    draw_pitch_2d(ax)

    pinch_idx = find_pinch_frame(data)
    n_frames = len(data)

    # Compute inter-frame time deltas for animation timing
    times = data[:, 27]
    avg_dt = np.mean(np.diff(times)) if len(times) > 1 else 0.033

    # Elements to animate
    ball_trail, = ax.plot([], [], "-", color="#60A5FA", linewidth=1.5, alpha=0.6, zorder=3)
    car_trail, = ax.plot([], [], "-", color="#34D399", linewidth=1.5, alpha=0.6, zorder=3)
    ball_dot, = ax.plot([], [], "o", color="#FBBF24", markersize=12, zorder=5)
    car_dot, = ax.plot([], [], "s", color="#34D399", markersize=10, zorder=5)
    pinch_marker, = ax.plot([], [], "*", color="#FBBF24", markersize=25, zorder=6)

    time_text = ax.text(0.02, 0.98, "", transform=ax.transAxes,
                        color="white", fontsize=10, va="top",
                        fontfamily="monospace",
                        bbox=dict(boxstyle="round,pad=0.3",
                                  facecolor="#1a1a2e", edgecolor="gray", alpha=0.9))

    ball_speed_at_pinch = np.linalg.norm(data[pinch_idx, 3:6])
    ax.set_title(f"{title}\nPinch speed: {ball_speed_at_pinch:.0f} uu/s",
                 color="white", fontsize=11, pad=10)

    def init():
        ball_trail.set_data([], [])
        car_trail.set_data([], [])
        ball_dot.set_data([], [])
        car_dot.set_data([], [])
        pinch_marker.set_data([], [])
        time_text.set_text("")
        return ball_trail, car_trail, ball_dot, car_dot, pinch_marker, time_text

    def update(frame_idx):
        i = frame_idx

        # Trails (all frames up to current)
        ball_trail.set_data(data[:i+1, 0], data[:i+1, 1])
        car_trail.set_data(data[:i+1, 9], data[:i+1, 10])

        # Current positions
        ball_dot.set_data([data[i, 0]], [data[i, 1]])
        car_dot.set_data([data[i, 9]], [data[i, 10]])

        # Show pinch marker once we reach the pinch frame
        if i >= pinch_idx:
            pinch_marker.set_data([data[pinch_idx, 0]], [data[pinch_idx, 1]])
        else:
            pinch_marker.set_data([], [])

        # Info text
        ball_speed = np.linalg.norm(data[i, 3:6])
        car_speed = np.linalg.norm(data[i, 12:15])
        elapsed = data[i, 27]
        txt = (f"t={elapsed:5.2f}s  frame {i:3d}/{n_frames-1}\n"
               f"ball: {ball_speed:6.0f} uu/s\n"
               f"car:  {car_speed:6.0f} uu/s\n"
               f"boost: {data[i, 25]:5.1f}")
        if i == pinch_idx:
            txt += "\n★ PINCH ★"
        time_text.set_text(txt)

        return ball_trail, car_trail, ball_dot, car_dot, pinch_marker, time_text

    # Animation interval: try to approximate real-time, scaled by speed factor
    interval_ms = max(10, int(avg_dt * 1000 / speed))
    anim = animation.FuncAnimation(
        fig, update, frames=n_frames, init_func=init,
        interval=interval_ms, blit=True, repeat=True
    )

    plt.tight_layout()
    plt.show()


def visualize_3d(data: np.ndarray, title: str = ""):
    """Static 3D plot showing ball and car trajectories with height."""
    fig = plt.figure(figsize=(10, 8), dpi=100)
    fig.patch.set_facecolor("#0f0f23")
    ax = fig.add_subplot(111, projection="3d")
    draw_pitch_3d(ax)

    pinch_idx = find_pinch_frame(data)

    # Ball trajectory
    n = len(data)
    ball_colors = plt.cm.coolwarm(np.linspace(0, 1, n))
    ax.scatter(data[:, 0], data[:, 1], data[:, 2],
               c=ball_colors, s=15, alpha=0.8, label="Ball", zorder=3)

    # Car trajectory
    car_colors = plt.cm.summer(np.linspace(0, 1, n))
    ax.scatter(data[:, 9], data[:, 10], data[:, 11],
               c=car_colors, s=10, alpha=0.7, label="Car", zorder=3)

    # Pinch marker
    ax.scatter([data[pinch_idx, 0]], [data[pinch_idx, 1]], [data[pinch_idx, 2]],
               color="#FBBF24", s=200, marker="*", zorder=6, label="Pinch")

    # Start markers
    ax.scatter([data[0, 0]], [data[0, 1]], [data[0, 2]],
               color="#60A5FA", s=80, marker="o", zorder=5)
    ax.scatter([data[0, 9]], [data[0, 10]], [data[0, 11]],
               color="#34D399", s=80, marker="s", zorder=5)

    ball_speed_at_pinch = np.linalg.norm(data[pinch_idx, 3:6])
    time_span = data[-1, 27] - data[0, 27]
    ax.set_title(f"{title}\nFrames: {n} | Span: {time_span:.2f}s | "
                 f"Pinch speed: {ball_speed_at_pinch:.0f} uu/s",
                 color="white", fontsize=10, pad=10)

    ax.set_xlabel("X", color="gray", fontsize=8)
    ax.set_ylabel("Y", color="gray", fontsize=8)
    ax.set_zlabel("Z", color="gray", fontsize=8)
    ax.legend(loc="upper left", fontsize=8, facecolor="#1a1a2e",
              edgecolor="gray", labelcolor="white")

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize an extracted Kuxir pinch .npy file"
    )
    parser.add_argument("npy_file", type=str,
                        help="Path to the .npy file from extract_pinches.py")
    parser.add_argument("--3d", dest="three_d", action="store_true",
                        help="Show 3D view instead of 2D")
    parser.add_argument("--static", action="store_true",
                        help="Show static plot instead of animation")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Animation playback speed multiplier (default: 1.0)")
    args = parser.parse_args()

    npy_path = Path(args.npy_file)
    if not npy_path.exists():
        print(f"File not found: {npy_path}")
        sys.exit(1)

    data = np.load(str(npy_path))
    print(f"Loaded {npy_path.name}: shape={data.shape}")

    if data.shape[0] == 0:
        print("Empty data, nothing to visualize.")
        sys.exit(1)

    if data.shape[1] != 28:
        print(f"Expected 28 columns, got {data.shape[1]}. Wrong format?")
        sys.exit(1)

    title = npy_path.stem

    # Quick summary
    pinch_idx = find_pinch_frame(data)
    ball_speed = np.linalg.norm(data[pinch_idx, 3:6])
    print(f"  Pinch frame: {pinch_idx}/{len(data)-1}")
    print(f"  Ball speed at pinch: {ball_speed:.0f} uu/s")
    print(f"  Time span: {data[0, 27]:.2f}s → {data[-1, 27]:.2f}s")
    print(f"  Ball start pos: ({data[0, 0]:.0f}, {data[0, 1]:.0f}, {data[0, 2]:.0f})")
    print(f"  Ball pinch pos: ({data[pinch_idx, 0]:.0f}, {data[pinch_idx, 1]:.0f}, {data[pinch_idx, 2]:.0f})")

    if args.three_d:
        visualize_3d(data, title=title)
    elif args.static:
        visualize_static_2d(data, title=title)
    else:
        visualize_animated_2d(data, speed=args.speed, title=title)


if __name__ == "__main__":
    main()
