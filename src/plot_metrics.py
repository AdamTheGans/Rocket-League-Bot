"""Plot training metrics from the CSV log."""
import os
import sys
import csv
import matplotlib.pyplot as plt

def plot_metrics(csv_path: str = "checkpoints/metrics.csv"):
    if not os.path.isfile(csv_path):
        print(f"No metrics file found at: {csv_path}")
        print("Run training first — metrics are logged every iteration.")
        sys.exit(1)

    # Read CSV
    timesteps, goals, touches, ball_speed, car_speed, boost = [], [], [], [], [], []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            timesteps.append(int(row["timesteps"]))
            goals.append(int(row["goals"]))
            touches.append(int(row["ball_touches"]))
            ball_speed.append(float(row["avg_ball_speed"]))
            car_speed.append(float(row["avg_car_speed"]))
            boost.append(float(row["avg_boost"]))

    if not timesteps:
        print("CSV is empty.")
        sys.exit(1)

    # Convert timesteps to millions for readability
    ts_m = [t / 1_000_000 for t in timesteps]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Grounded Strike Training Metrics", fontsize=16, fontweight="bold")

    # Goals per iteration
    ax = axes[0, 0]
    ax.plot(ts_m, goals, color="#e74c3c", alpha=0.4, linewidth=0.8)
    # Rolling average
    window = min(20, len(goals) // 3 + 1)
    if window > 1:
        import numpy as np
        goals_smooth = np.convolve(goals, np.ones(window)/window, mode="valid")
        ax.plot(ts_m[window-1:], goals_smooth, color="#e74c3c", linewidth=2, label=f"{window}-iter avg")
        ax.legend()
    ax.set_title("Goals per Iteration")
    ax.set_xlabel("Timesteps (M)")
    ax.set_ylabel("Goals")
    ax.grid(alpha=0.3)

    # Ball touches per iteration
    ax = axes[0, 1]
    ax.plot(ts_m, touches, color="#3498db", alpha=0.4, linewidth=0.8)
    if window > 1:
        import numpy as np
        touches_smooth = np.convolve(touches, np.ones(window)/window, mode="valid")
        ax.plot(ts_m[window-1:], touches_smooth, color="#3498db", linewidth=2, label=f"{window}-iter avg")
        ax.legend()
    ax.set_title("Ball Touches per Iteration")
    ax.set_xlabel("Timesteps (M)")
    ax.set_ylabel("Touches")
    ax.grid(alpha=0.3)

    # Avg ball speed
    ax = axes[1, 0]
    ax.plot(ts_m, ball_speed, color="#f39c12", alpha=0.4, linewidth=0.8)
    if window > 1:
        speed_smooth = np.convolve(ball_speed, np.ones(window)/window, mode="valid")
        ax.plot(ts_m[window-1:], speed_smooth, color="#f39c12", linewidth=2, label=f"{window}-iter avg")
        ax.legend()
    ax.set_title("Avg Ball Speed (uu/s)")
    ax.set_xlabel("Timesteps (M)")
    ax.set_ylabel("Speed")
    ax.grid(alpha=0.3)

    # Avg car speed
    ax = axes[1, 1]
    ax.plot(ts_m, car_speed, color="#2ecc71", alpha=0.4, linewidth=0.8)
    if window > 1:
        car_smooth = np.convolve(car_speed, np.ones(window)/window, mode="valid")
        ax.plot(ts_m[window-1:], car_smooth, color="#2ecc71", linewidth=2, label=f"{window}-iter avg")
        ax.legend()
    ax.set_title("Avg Car Speed (uu/s)")
    ax.set_xlabel("Timesteps (M)")
    ax.set_ylabel("Speed")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out_path = "checkpoints/training_curves.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved training curves to: {out_path}")
    plt.show()


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/metrics.csv"
    plot_metrics(path)
