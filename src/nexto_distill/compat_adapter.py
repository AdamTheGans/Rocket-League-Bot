# src/nexto_distill/compat_adapter.py
"""
Bridge between rlgym v2 GameState and the encoded flat array that
Nexto's NextoObsBuilder.batched_build_obs() expects.

We reimplement the data packing from nexto/nexto_obs.py::encode_gamestate()
using rlgym v2 types directly, avoiding any dependency on rlgym_compat.
"""
from __future__ import annotations

import numpy as np

from rlgym.rocket_league.api import GameState, Car, PhysicsObject


# --------------------------------------------------------------------------- #
# Quaternion from rotation matrix  (matches nexto_obs.rotation_to_quaternion)
# --------------------------------------------------------------------------- #

def _rotation_to_quaternion(m: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion [w, x, y, z].

    Reproduces the *exact* convention used by Nexto's encode_gamestate().
    The returned quaternion is negated to match Nexto's convention.
    """
    trace = np.trace(m)
    q = np.zeros(4)

    if trace > 0:
        s = (trace + 1) ** 0.5
        q[0] = s * 0.5
        s = 0.5 / s
        q[1] = (m[2, 1] - m[1, 2]) * s
        q[2] = (m[0, 2] - m[2, 0]) * s
        q[3] = (m[1, 0] - m[0, 1]) * s
    else:
        if m[0, 0] >= m[1, 1] and m[0, 0] >= m[2, 2]:
            s = (1 + m[0, 0] - m[1, 1] - m[2, 2]) ** 0.5
            inv_s = 0.5 / s
            q[1] = 0.5 * s
            q[2] = (m[1, 0] + m[0, 1]) * inv_s
            q[3] = (m[2, 0] + m[0, 2]) * inv_s
            q[0] = (m[2, 1] - m[1, 2]) * inv_s
        elif m[1, 1] > m[2, 2]:
            s = (1 + m[1, 1] - m[0, 0] - m[2, 2]) ** 0.5
            inv_s = 0.5 / s
            q[1] = (m[0, 1] + m[1, 0]) * inv_s
            q[2] = 0.5 * s
            q[3] = (m[1, 2] + m[2, 1]) * inv_s
            q[0] = (m[0, 2] - m[2, 0]) * inv_s
        else:
            s = (1 + m[2, 2] - m[0, 0] - m[1, 1]) ** 0.5
            inv_s = 0.5 / s
            q[1] = (m[0, 2] + m[2, 0]) * inv_s
            q[2] = (m[1, 2] + m[2, 1]) * inv_s
            q[3] = 0.5 * s
            q[0] = (m[1, 0] - m[0, 1]) * inv_s

    return -q  # Nexto convention: negate


def _encode_physics(physics: PhysicsObject) -> list:
    """Encode a PhysicsObject into [pos(3), quat(4), lin_vel(3), ang_vel(3)] = 13 values."""
    vals = []
    vals.extend(physics.position.tolist())
    vals.extend(_rotation_to_quaternion(physics.rotation_mtx).tolist())
    vals.extend(physics.linear_velocity.tolist())
    vals.extend(physics.angular_velocity.tolist())
    return vals


def _encode_ball(physics: PhysicsObject) -> list:
    """Encode ball data: [pos(3), lin_vel(3), ang_vel(3)] = 9 values."""
    vals = []
    vals.extend(physics.position.tolist())
    vals.extend(physics.linear_velocity.tolist())
    vals.extend(physics.angular_velocity.tolist())
    return vals


INV_VEC = np.array([-1, -1, 1], dtype=np.float64)


def encode_v2_state(
    state: GameState,
    blue_score: int = 0,
    orange_score: int = 0,
) -> np.ndarray:
    """
    Convert an rlgym v2 GameState into the flat encoded array that
    ``NextoObsBuilder.batched_build_obs()`` expects.

    This replicates the layout of ``nexto_obs.encode_gamestate()``:
      [0]        = 0 (unused)
      [1]        = blue_score
      [2]        = orange_score
      [3:37]     = boost_pads (34 bools, 1=active 0=inactive)
      [37:46]    = ball  (pos3 + lin_vel3 + ang_vel3)
      [46:55]    = inverted_ball
      [55+ ...]  = per-player blocks (48 values each)

    Parameters
    ----------
    state : rlgym v2 GameState
    blue_score, orange_score : int
        Cumulative scores to embed. Maintained externally.

    Returns
    -------
    np.ndarray  –  1-D float64 array, length = 55 + 48*n_players
    """
    vals: list = [0, blue_score, orange_score]

    # Boost pads: timer==0 means active (available)
    boost_pads = (state.boost_pad_timers == 0).astype(float).tolist()
    vals.extend(boost_pads)

    # Ball  (normal + inverted)
    vals.extend(_encode_ball(state.ball))
    vals.extend(_encode_ball(state.inverted_ball))

    # Players — must be ordered: sorted by agent_id for determinism
    for agent_id in sorted(state.cars.keys()):
        car: Car = state.cars[agent_id]

        # car_id (int)  and team_num
        # Nexto uses p.car_id (an int). In v2 agent_id is typically (str, int).
        # We'll use the integer index from the agent_id tuple.
        if isinstance(agent_id, tuple):
            car_id_int = agent_id[1]
        elif isinstance(agent_id, int):
            car_id_int = agent_id
        else:
            car_id_int = 0

        vals.extend([car_id_int, car.team_num])

        # car_data (normal physics) — 13 values
        vals.extend(_encode_physics(car.physics))

        # inverted_car_data — 13 values
        vals.extend(_encode_physics(car.inverted_physics))

        # Tertiary info — 10 values matching encode_gamestate layout:
        # [0,0,0,0,0, is_demoed, on_ground, ball_touched, has_flip, boost_amount]
        vals.extend([
            0, 0, 0, 0, 0,
            float(car.is_demoed),
            float(car.on_ground),
            float(car.ball_touches > 0),
            float(car.has_flip),
            car.boost_amount / 100.0,  # Nexto expects [0,1]
        ])

    return np.array(vals, dtype=np.float64)
