# src/replay_pipeline/extract_pinches.py
"""
Kuxir Pinch Extraction Pipeline
================================
Parses .replay files, detects Kuxir pinch events via velocity-spike heuristics,
extracts a 5-second window (4s before, 1s after the pinch), isolates the scorer's
car + ball into a clean 1v1 numpy array, and saves as .npy for use in a StateSetter.

Usage (simple — one replay at a time):
    python src/replay_pipeline/extract_pinches.py --replay 9CD67A5C.replay --time 140

Usage (batch — process all replays with TIMESTAMP_HINTS dict):
    python src/replay_pipeline/extract_pinches.py

Output:
    extracted_mechanics/<replay_stem>.npy   (shape: [N_frames, 28])
"""
from __future__ import annotations

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from typing import Optional

from rlgym_tools.rocket_league.replays.parsed_replay import ParsedReplay
from rlgym_tools.rocket_league.replays.convert import replay_to_rlgym
from rlgym_tools.rocket_league.replays.replay_frame import ReplayFrame

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION: Map each replay filename -> approximate game-clock seconds
# of the Kuxir pinch goal.  The game clock counts DOWN from 300 (5:00).
#
# Example: a goal at "4:05 remaining" is clock=245.
# The pipeline finds the nearest goal in metadata within ±15s of this value.
#
# To add new replays, just append entries here.
# ─────────────────────────────────────────────────────────────────────────────
TIMESTAMP_HINTS: dict[str, float | list[float]] = {
    # Fill these in with the approximate game-clock time of the Kuxir pinch.
    # Use a list if the replay contains multiple Kuxir pinch goals.
    #
    # "0A8EA1A44064FBD41B56AF93901B1FE5.replay": 180,
    # "6E50C2AD4D235CC7BCD5DDAAD7B90DE4.replay": [210, 45],
}

# ─────────────────────────────────────────────────────────────────────────────
# PINCH DETECTION TUNING CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
SPIKE_LOOKBACK_FRAMES  = 5       # Compare speed[i] vs speed[i-k] over this window
SPIKE_THRESHOLD        = 1200.0  # Minimum delta (uu/s) to qualify as a pinch spike
MIN_PEAK_SPEED         = 2000.0  # Ball must reach this speed at the spike frame
SEARCH_WINDOW_FRAMES   = 300     # Walk back at most this many frames from the goal
WALL_X_MIN             = 3200.0  # abs(ball.x) must exceed this (near side wall)
BALL_Z_MAX             = 1500.0  # ball.z must be below this (mid-to-low height)
GOAL_Y_MAX             = 4800.0  # abs(ball.y) must be below this (far from backboard)
TIMESTAMP_TOLERANCE    = 15.0    # ±seconds to match a timestamp hint to a goal

# ─────────────────────────────────────────────────────────────────────────────
# DIRECTORIES
# ─────────────────────────────────────────────────────────────────────────────
REPLAY_DIR     = Path("replays/kuxirs")
OUTPUT_DIR     = Path("extracted_mechanics")

# ─────────────────────────────────────────────────────────────────────────────
# Per-frame packing layout (28 floats)
# ─────────────────────────────────────────────────────────────────────────────
# Index  Field
# 0-2    Ball position          (x, y, z)
# 3-5    Ball linear velocity   (vx, vy, vz)
# 6-8    Ball angular velocity  (wx, wy, wz)
# 9-11   Car position           (x, y, z)
# 12-14  Car linear velocity    (vx, vy, vz)
# 15-17  Car angular velocity   (wx, wy, wz)
# 18-21  Car quaternion          (w, x, y, z)
# 22-24  Car euler angles        (pitch, yaw, roll)
# 25     Car boost amount        [0..100]
# 26     Car on ground            (0 or 1)
# 27     Elapsed time (seconds)   relative to window start
FRAME_WIDTH = 28


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def build_name_to_uid(metadata: dict) -> dict[str, int]:
    """Map player display names to their integer unique_ids."""
    return {p["name"]: int(p["unique_id"]) for p in metadata["players"]}


def find_goal_by_timestamp(
    metadata: dict,
    timestamp_hint: float,
    tolerance: float = TIMESTAMP_TOLERANCE,
) -> Optional[dict]:
    """
    Find the goal in metadata['game']['goals'] whose game-clock time is closest
    to `timestamp_hint`, within ±tolerance seconds.

    The metadata goals list doesn't store game-clock time directly; it stores
    `frame`.  We'll return the raw goal dict and let the caller match it to the
    gameplay-period frames where we *can* read the scoreboard timer.

    Strategy: return ALL goals as candidates; the caller filters by scoreboard
    timer once frames are loaded.
    """
    goals = metadata["game"].get("goals", [])
    if not goals:
        return None
    # Return the full list; the caller will match by scoreboard timer
    return goals


def match_goal_to_period_frame(
    period_frames: list[ReplayFrame],
    goal_frame_number: int,
) -> bool:
    """
    Check whether a metadata goal (by frame number) falls within a given
    gameplay period's frame range.  The period's last frame should be near
    the goal frame.
    """
    if not period_frames:
        return False
    # The last frame's tick_count in a goal-ending period is close to the goal frame.
    # But tick_count is in ticks, not frame numbers.  Instead, check goal_scored flag.
    return period_frames[-1].state.goal_scored


def detect_pinch_moment(
    frames: list[ReplayFrame],
    timestamp_hint: Optional[float] = None,
) -> Optional[int]:
    """
    Walk backward from the end of a gameplay period to find the pinch moment.

    Returns the index into `frames` of the detected pinch, or None.
    """
    if len(frames) < SPIKE_LOOKBACK_FRAMES + 1:
        return None

    # ── Step 1: Compute ball speed at every frame ──
    speeds = np.array([
        np.linalg.norm(f.state.ball.linear_velocity)
        for f in frames
    ], dtype=np.float64)

    # ── Step 2: Restrict search to SEARCH_WINDOW_FRAMES before the end ──
    end_idx = len(frames) - 1
    start_idx = max(SPIKE_LOOKBACK_FRAMES, end_idx - SEARCH_WINDOW_FRAMES)

    # ── Step 3: If timestamp hint given, further restrict by scoreboard timer ──
    if timestamp_hint is not None:
        # Find frames where scoreboard timer is within ±TIMESTAMP_TOLERANCE
        valid_mask = np.zeros(len(frames), dtype=bool)
        for i, f in enumerate(frames):
            timer = f.scoreboard.game_timer_seconds
            if abs(timer - timestamp_hint) <= TIMESTAMP_TOLERANCE:
                valid_mask[i] = True
        # Restrict the search range to valid frames
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) == 0:
            return None  # No frames near the timestamp
        start_idx = max(start_idx, int(valid_indices[0]))
        end_idx = min(end_idx, int(valid_indices[-1]))

    # ── Step 4: Find the frame with the sharpest velocity spike ──
    best_idx = None
    best_delta = -1.0

    for i in range(end_idx, start_idx - 1, -1):
        lookback_idx = max(0, i - SPIKE_LOOKBACK_FRAMES)
        speed_delta = speeds[i] - speeds[lookback_idx]

        if (speed_delta > SPIKE_THRESHOLD
                and speeds[i] > MIN_PEAK_SPEED
                and speed_delta > best_delta):
            # ── Step 5: Geometric validation ──
            ball_pos = frames[i].state.ball.position
            if (abs(ball_pos[0]) > WALL_X_MIN        # Near side wall
                    and ball_pos[2] < BALL_Z_MAX      # Mid-to-low height
                    and abs(ball_pos[1]) < GOAL_Y_MAX  # Far from backboard
            ):
                best_delta = speed_delta
                best_idx = i

    return best_idx


def identify_scorer_uid(
    metadata: dict,
    goal_entry: dict,
) -> Optional[int]:
    """
    Map a goal's player_name to their integer unique_id using the players list.
    """
    name_to_uid = build_name_to_uid(metadata)
    scorer_name = goal_entry.get("player_name")
    if scorer_name and scorer_name in name_to_uid:
        return name_to_uid[scorer_name]
    # Fuzzy fallback: partial match
    for name, uid in name_to_uid.items():
        if scorer_name and scorer_name.lower() in name.lower():
            return uid
    return None


def pack_frame(state, car, t_elapsed: float) -> np.ndarray:
    """
    Pack a single frame's ball + car physics into a flat 28-float array.
    """
    row = np.zeros(FRAME_WIDTH, dtype=np.float32)

    # Ball (indices 0-8)
    row[0:3]  = state.ball.position
    row[3:6]  = state.ball.linear_velocity
    row[6:9]  = state.ball.angular_velocity

    # Car physics (indices 9-17)
    row[9:12]  = car.physics.position
    row[12:15] = car.physics.linear_velocity
    row[15:18] = car.physics.angular_velocity

    # Car quaternion (indices 18-21)
    row[18:22] = car.physics.quaternion

    # Car euler angles (indices 22-24): pitch, yaw, roll
    row[22:25] = car.physics.euler_angles

    # Boost (index 25)
    row[25] = car.boost_amount

    # On ground (index 26): wheels_with_contact can be a tuple of bools or an int
    try:
        on_ground = any(car.wheels_with_contact)
    except TypeError:
        on_ground = bool(car.wheels_with_contact)
    row[26] = 1.0 if on_ground else 0.0

    # Elapsed time (index 27)
    row[27] = t_elapsed

    return row


def extract_window(
    frames: list[ReplayFrame],
    pinch_idx: int,
    scorer_uid: int,
    seconds_before: float = 2.5,
    seconds_after: float = 0.5,
) -> np.ndarray:
    """
    Slice frames around the pinch moment and pack into an (N, 28) array.

    Parameters
    ----------
    frames : list of ReplayFrame
        Complete gameplay period frames.
    pinch_idx : int
        Index of the detected pinch moment.
    scorer_uid : int
        The unique_id of the scoring player (key into state.cars).
    seconds_before / seconds_after : float
        Window bounds relative to the pinch moment.

    Returns
    -------
    np.ndarray of shape (N, 28)
    """
    from rlgym.rocket_league.common_values import TICKS_PER_SECOND

    # Get the tick_count at the pinch frame
    pinch_tick = frames[pinch_idx].state.tick_count

    # Compute tick bounds
    start_tick = pinch_tick - seconds_before * TICKS_PER_SECOND
    end_tick = pinch_tick + seconds_after * TICKS_PER_SECOND

    # Collect frames within the time window
    packed_rows = []
    t0 = None  # Will be set to the first frame's tick for elapsed time

    for frame in frames:
        tick = frame.state.tick_count
        if tick < start_tick:
            continue
        if tick > end_tick:
            break

        # Check that the scorer's car exists in this frame
        if scorer_uid not in frame.state.cars:
            continue

        if t0 is None:
            t0 = tick

        car = frame.state.cars[scorer_uid]
        t_elapsed = (tick - t0) / TICKS_PER_SECOND
        packed_rows.append(pack_frame(frame.state, car, t_elapsed))

    if not packed_rows:
        return np.empty((0, FRAME_WIDTH), dtype=np.float32)

    return np.stack(packed_rows, axis=0)


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def process_single_replay(
    replay_path: Path,
    timestamp_hints: list[float],
    output_dir: Path,
) -> list[Path]:
    """
    Full pipeline for one .replay file.  Returns list of saved .npy paths.
    """
    stem = replay_path.stem
    print(f"\n{'='*70}")
    print(f"  Processing: {replay_path.name}")
    print(f"{'='*70}")

    # ── 1. Parse the replay ──
    print("  [1/5] Parsing replay with carball...")
    parsed = ParsedReplay.load(str(replay_path))
    metadata = parsed.metadata
    goals = metadata["game"].get("goals", [])
    name_to_uid = build_name_to_uid(metadata)

    print(f"        Found {len(goals)} goals, {len(metadata['players'])} players")
    for g in goals:
        print(f"        - Frame {g['frame']:>6d}  {g['player_name']}")

    # ── 2. Convert to rlgym frames, grouped by gameplay period ──
    print("  [2/5] Converting to rlgym frames...")
    all_periods: list[list[ReplayFrame]] = []
    current_period: list[ReplayFrame] = []

    for frame in replay_to_rlgym(parsed, interpolation="none"):
        current_period.append(frame)
        # A period ends when goal_scored is True on the last frame,
        # but replay_to_rlgym yields all frames for a period before moving
        # to the next.  We detect the boundary by checking if the next frame
        # has a much lower tick_count (new period) or if goal_scored is True.
        if frame.state.goal_scored:
            all_periods.append(current_period)
            current_period = []

    # Don't forget the last period (might not end in a goal)
    if current_period:
        all_periods.append(current_period)

    print(f"        Found {len(all_periods)} gameplay periods "
          f"({sum(1 for p in all_periods if p and p[-1].state.goal_scored)} ending in goals)")

    # ── 3. For each timestamp hint, find the matching goal ──
    saved_paths: list[Path] = []

    for hint_idx, hint_time in enumerate(timestamp_hints):
        print(f"\n  [3/5] Searching for pinch near game-clock = {hint_time:.0f}s...")

        # Find the gameplay period whose scoreboard timer near the end is
        # closest to the hint_time, and that ends in a goal.
        best_period_idx = None
        best_goal_entry = None
        best_timer_diff = float("inf")

        for p_idx, period in enumerate(all_periods):
            if not period or not period[-1].state.goal_scored:
                continue

            # Check scoreboard timer near the end of the period
            goal_timer = period[-1].scoreboard.game_timer_seconds

            # Special case: NaN hint_time means "match OT goals"
            if np.isnan(hint_time):
                if np.isnan(goal_timer) or np.isinf(goal_timer):
                    diff = 0.0
                else:
                    continue  # Skip non-OT periods
            else:
                if np.isnan(goal_timer) or np.isinf(goal_timer):
                    continue  # Skip OT periods when looking for a timed goal
                diff = abs(goal_timer - hint_time)

            if diff < best_timer_diff and diff <= TIMESTAMP_TOLERANCE:
                best_timer_diff = diff
                best_period_idx = p_idx

                # Match this period's goal to the metadata goals list
                # by finding the goal with the closest frame number
                period_end_tick = period[-1].state.tick_count
                for g in goals:
                    if best_goal_entry is None:
                        best_goal_entry = g
                    # Just pick any goal — we refine below

        # If scoreboard matching failed, try matching goal metadata frame numbers
        # to periods.  Walk through goals and see which goal's timer is closest.
        if best_period_idx is None:
            print(f"        ⚠ Could not match by scoreboard timer.")
            print(f"        Trying to match by period index / goal order...")

            # Try matching goals to periods in order
            goal_period_map = []
            for p_idx, period in enumerate(all_periods):
                if period and period[-1].state.goal_scored:
                    goal_period_map.append(p_idx)

            # goals and goal-ending periods should be in the same order
            for g_idx, g in enumerate(goals):
                if g_idx < len(goal_period_map):
                    p_idx = goal_period_map[g_idx]
                    period = all_periods[p_idx]
                    # Check if the scoreboard timer is close enough
                    goal_timer = period[-1].scoreboard.game_timer_seconds
                    diff = abs(goal_timer - hint_time)
                    if diff < best_timer_diff:
                        best_timer_diff = diff
                        best_period_idx = p_idx
                        best_goal_entry = g

        if best_period_idx is None:
            print(f"        ✗ No goal found near timestamp {hint_time:.0f}s (tolerance ±{TIMESTAMP_TOLERANCE}s)")
            continue

        period = all_periods[best_period_idx]
        goal_timer = period[-1].scoreboard.game_timer_seconds
        print(f"        ✓ Matched to period {best_period_idx} "
              f"(goal at game-clock {goal_timer:.1f}s, Δ={best_timer_diff:.1f}s)")

        # The goal entry tells us the scorer
        # Match the goal that corresponds to this period by checking frame ordering
        goal_ending_periods = [
            (p_idx, p) for p_idx, p in enumerate(all_periods)
            if p and p[-1].state.goal_scored
        ]
        period_order = [p_idx for p_idx, _ in goal_ending_periods]
        if best_period_idx in period_order:
            goal_index_in_order = period_order.index(best_period_idx)
            if goal_index_in_order < len(goals):
                best_goal_entry = goals[goal_index_in_order]

        if best_goal_entry is None:
            print(f"        ✗ Could not determine goal entry from metadata")
            continue

        scorer_uid = identify_scorer_uid(metadata, best_goal_entry)
        if scorer_uid is None:
            print(f"        ✗ Could not find UID for scorer '{best_goal_entry.get('player_name')}'")
            continue

        print(f"        Scorer: {best_goal_entry['player_name']} (UID: {scorer_uid})")

        # ── 4. Detect the pinch moment within this period ──
        print(f"  [4/5] Detecting pinch moment (velocity spike heuristic)...")
        pinch_idx = detect_pinch_moment(period, timestamp_hint=hint_time)

        if pinch_idx is None:
            # Fall back: use the goal frame itself minus a small offset
            print(f"        ⚠ No velocity spike detected. Using goal frame - 10 as fallback.")
            pinch_idx = max(0, len(period) - 10)

        # Print detection diagnostics
        ball_speed = np.linalg.norm(period[pinch_idx].state.ball.linear_velocity)
        ball_pos = period[pinch_idx].state.ball.position
        print(f"        Pinch at frame index {pinch_idx}/{len(period)-1}")
        print(f"        Ball speed: {ball_speed:.0f} uu/s")
        print(f"        Ball pos:   ({ball_pos[0]:.0f}, {ball_pos[1]:.0f}, {ball_pos[2]:.0f})")

        # ── 5. Extract the 5-second window and save ──
        print(f"  [5/5] Extracting window & saving...")
        data = extract_window(period, pinch_idx, scorer_uid,
                              seconds_before=4.0, seconds_after=1.0)

        if data.shape[0] == 0:
            print(f"        ✗ Empty extraction (scorer UID {scorer_uid} not in frames?)")
            continue

        # Save
        output_dir.mkdir(parents=True, exist_ok=True)
        suffix = f"_hint{hint_idx}" if len(timestamp_hints) > 1 else ""
        out_path = output_dir / f"{stem}{suffix}.npy"
        np.save(str(out_path), data)
        saved_paths.append(out_path)

        print(f"        ✓ Saved {out_path.name}  shape={data.shape}")
        print(f"        Time span: {data[0, 27]:.2f}s → {data[-1, 27]:.2f}s "
              f"({data.shape[0]} frames)")

    return saved_paths


def parse_time(time_str: str) -> float:
    """
    Parse a game-clock time string.  Accepts:
      - "2:40" -> 160.0  (M:SS format, what you see on screen)
      - "160"  -> 160.0  (raw seconds remaining)
      - "OT"   -> NaN    (overtime goal)
    """
    time_str = time_str.strip()
    if time_str.upper() == "OT":
        return float("nan")
    if ":" in time_str:
        parts = time_str.split(":")
        minutes = int(parts[0])
        seconds = float(parts[1])
        return minutes * 60.0 + seconds
    return float(time_str)


def list_goals_for_replay(replay_path: Path):
    """Parse a replay and print every goal with its game-clock time."""
    print(f"\n{'='*70}")
    print(f"  Goals in: {replay_path.name}")
    print(f"{'='*70}")

    print("  Parsing replay...")
    parsed = ParsedReplay.load(str(replay_path))
    metadata = parsed.metadata
    goals = metadata["game"].get("goals", [])

    if not goals:
        print("  No goals found in this replay.")
        return

    # Convert frames to get scoreboard timers
    print("  Converting to rlgym frames (this may take a moment)...")
    all_periods: list[list[ReplayFrame]] = []
    current_period: list[ReplayFrame] = []

    for frame in replay_to_rlgym(parsed, interpolation="none"):
        current_period.append(frame)
        if frame.state.goal_scored:
            all_periods.append(current_period)
            current_period = []
    if current_period:
        all_periods.append(current_period)

    # Map goal-ending periods to goals (they should be in the same order)
    goal_periods = [p for p in all_periods if p and p[-1].state.goal_scored]

    print(f"\n  {'#':<4} {'Scorer':<20} {'Team':<8} {'Clock':<10} {'Use with --time'}")
    print(f"  {'─'*4} {'─'*20} {'─'*8} {'─'*10} {'─'*15}")

    for i, g in enumerate(goals):
        team = "Orange" if g.get("is_orange") else "Blue"
        scorer = g.get("player_name", "???")

        # Get game clock from the corresponding period
        if i < len(goal_periods):
            period = goal_periods[i]
            timer = period[-1].scoreboard.game_timer_seconds
            if np.isnan(timer) or np.isinf(timer):
                clock_str = "OT"
                time_arg = "OT"
            else:
                minutes = int(timer // 60)
                seconds = timer % 60
                clock_str = f"{minutes}:{seconds:04.1f}"
                time_arg = f"{minutes}:{int(seconds):02d}"
        else:
            clock_str = "???"
            time_arg = "???"

        print(f"  {i+1:<4} {scorer:<20} {team:<8} {clock_str:<10} {time_arg}")

    print(f"\n  Usage:")
    print(f"    python src/replay_pipeline/extract_pinches.py --replay {replay_path.name} --time 2:40")
    print(f"    (Replace 2:40 with the Clock value of the pinch goal above)")


def main():
    parser = argparse.ArgumentParser(
        description="Extract Kuxir pinch windows from .replay files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all goals in a replay (with game-clock times):
  python src/replay_pipeline/extract_pinches.py --replay FILE.replay --list-goals

  # Extract pinch at game-clock 2:40 (what you see on screen):
  python src/replay_pipeline/extract_pinches.py --replay FILE.replay --time 2:40

  # Same thing with raw seconds:
  python src/replay_pipeline/extract_pinches.py --replay FILE.replay --time 160
"""
    )
    parser.add_argument("--replay-dir", type=str, default=str(REPLAY_DIR),
                        help="Directory containing .replay files")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR),
                        help="Output directory for .npy files")
    parser.add_argument("--timestamps", type=str, default=None,
                        help='JSON dict of filename->timestamp (batch mode)')

    # ── Simple mode: one replay at a time (PowerShell friendly) ──
    parser.add_argument("--replay", type=str, default=None,
                        help="Single replay filename (e.g. 9CD67A5C.replay)")
    parser.add_argument("--time", type=str, default=None,
                        help="Game clock when the pinch goal scores (e.g. 2:40 or 160)")
    parser.add_argument("--list-goals", action="store_true",
                        help="List all goals in the replay with their game-clock times, then exit")

    args = parser.parse_args()

    replay_dir = Path(args.replay_dir)
    output_dir = Path(args.output_dir)

    # ── List-goals mode ──
    if args.list_goals:
        if not args.replay:
            parser.error("--list-goals requires --replay")
        replay_path = replay_dir / args.replay
        if not replay_path.exists():
            print(f"File not found: {replay_path}")
            print(f"Available replays in {replay_dir}:")
            for rf in sorted(replay_dir.glob("*.replay")):
                print(f"    {rf.name}")
            sys.exit(1)
        list_goals_for_replay(replay_path)
        return

    # ── Build the hints dict ──
    hints = dict(TIMESTAMP_HINTS)

    # Simple mode: --replay + --time override everything else
    if args.replay and args.time is not None:
        hints[args.replay] = parse_time(args.time)
    elif args.replay and args.time is None:
        parser.error("--replay requires --time (or use --list-goals to see goals)")
    elif args.time is not None and not args.replay:
        parser.error("--time requires --replay (replay filename)")

    # Batch mode: --timestamps JSON merges with built-in hints
    if args.timestamps:
        cli_hints = json.loads(args.timestamps)
        hints.update(cli_hints)

    # Find all .replay files
    replay_files = sorted(replay_dir.glob("*.replay"))
    if not replay_files:
        print(f"No .replay files found in {replay_dir}")
        sys.exit(1)

    print(f"Found {len(replay_files)} replay files in {replay_dir}")

    # Filter to only replays that have timestamp hints
    replays_with_hints = []
    replays_without_hints = []
    for rf in replay_files:
        if rf.name in hints:
            replays_with_hints.append(rf)
        else:
            replays_without_hints.append(rf)

    if not replays_with_hints:
        print(f"\n✗ No replays matched the provided hints.")
        if args.replay:
            print(f"  Looked for '{args.replay}' in {replay_dir}")
            print(f"  Available files:")
            for rf in replay_files:
                print(f"    {rf.name}")
        else:
            print(f"\nUsage:")
            print(f"  python src/replay_pipeline/extract_pinches.py --replay FILENAME.replay --time 2:40")
            print(f"\n  Tip: use --list-goals to see all goals with their clock times:")
            print(f"  python src/replay_pipeline/extract_pinches.py --replay FILENAME.replay --list-goals")
        sys.exit(1)

    if replays_without_hints and not args.replay:
        print(f"\n⚠ {len(replays_without_hints)} replays have no timestamp hints (skipped):")
        for rf in replays_without_hints:
            print(f"    {rf.name}")
        print(f"\nTo process them:")
        print(f"  python src/replay_pipeline/extract_pinches.py --replay FILENAME.replay --time M:SS\n")

    all_saved: list[Path] = []
    for rf in replays_with_hints:
        hint_val = hints[rf.name]
        # Normalize to list (single timestamp or list of timestamps)
        if isinstance(hint_val, (int, float)):
            hint_list = [float(hint_val)]
        else:
            hint_list = [float(h) for h in hint_val]

        saved = process_single_replay(rf, hint_list, output_dir)
        all_saved.extend(saved)

    # ── Summary ──
    print(f"\n{'='*70}")
    print(f"  PIPELINE COMPLETE")
    print(f"{'='*70}")
    print(f"  Processed: {len(replays_with_hints)} replays")
    print(f"  Extracted: {len(all_saved)} .npy files → {output_dir}/")
    for p in all_saved:
        data = np.load(str(p))
        print(f"    {p.name:50s}  shape={data.shape}")

    if all_saved:
        print(f"\n  Visualize with:")
        print(f"    python src/replay_pipeline/visualize_pinch.py {all_saved[0]}")


if __name__ == "__main__":
    main()


