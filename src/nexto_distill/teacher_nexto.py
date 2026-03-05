# src/nexto_distill/teacher_nexto.py
"""
Teacher wrapper that loads the Nexto TorchScript model and exposes
a simple API for querying actions / logits from an rlgym v2 GameState.

Uses the compat adapter to convert v2 GameState → encoded array,
then feeds through Nexto's NextoObsBuilder → model.
"""
from __future__ import annotations

import os
import sys
import types
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# rlgym_compat shim: inject minimal stub modules so that nexto_obs.py can be
# imported without installing the legacy rlgym_compat package.
# We only need the BLUE_TEAM/ORANGE_TEAM constants and dummy class stubs.
# --------------------------------------------------------------------------- #
if "rlgym_compat" not in sys.modules:
    _compat = types.ModuleType("rlgym_compat")
    sys.modules["rlgym_compat"] = _compat

    _cv = types.ModuleType("rlgym_compat.common_values")
    _cv.BLUE_TEAM = 0
    _cv.ORANGE_TEAM = 1
    sys.modules["rlgym_compat.common_values"] = _cv
    _compat.common_values = _cv

    _gs = types.ModuleType("rlgym_compat.game_state")

    class _DummyGameState:
        """Stub — not used in our code path (we feed encoded arrays directly)."""
        pass

    class _DummyPlayerData:
        """Stub — not used in our code path."""
        pass

    _gs.GameState = _DummyGameState
    _gs.PlayerData = _DummyPlayerData
    sys.modules["rlgym_compat.game_state"] = _gs
    _compat.game_state = _gs

# Add nexto/ to path so we can import its modules directly
_NEXTO_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "nexto")
)
if _NEXTO_DIR not in sys.path:
    sys.path.insert(0, _NEXTO_DIR)

from nexto_obs import NextoObsBuilder  # type: ignore  # noqa: E402
from agent import Agent as NextoAgent  # type: ignore  # noqa: E402  (only for make_lookup_table)

from nexto_distill.compat_adapter import encode_v2_state


class NextoTeacher:
    """
    Wraps the pre-trained Nexto TorchScript model for use as a
    distillation teacher inside an rlgym v2 environment.

    Parameters
    ----------
    model_path : str
        Path to ``nexto-model.pt`` (TorchScript).
    device : str
        ``"cpu"`` or ``"cuda"``.
    tick_skip : int
        Must match the environment's action repeat (default 8).
    n_players : int or None
        Number of players expected in the game. Passed to NextoObsBuilder.
        None = auto-detect from first state.
    """

    def __init__(
        self,
        model_path: str = os.path.join(_NEXTO_DIR, "nexto-model.pt"),
        device: str = "cpu",
        tick_skip: int = 8,
        n_players: Optional[int] = None,
    ):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"Nexto model not found at {model_path!r}. "
                "Ensure nexto/nexto-model.pt exists."
            )

        self.device = torch.device(device)
        self.tick_skip = tick_skip

        # Load TorchScript model (always load to CPU first, then move —
        # torch.jit.load with map_location=cuda can fail for CPU-traced models)
        with open(model_path, "rb") as f:
            self.model = torch.jit.load(f, map_location="cpu")
        if self.device.type != "cpu":
            self.model = self.model.to(self.device)
        self.model.eval()

        # Build the action lookup table (identical to rlgym v2 LookupTableAction)
        self._lookup_table = NextoAgent.make_lookup_table()
        self.num_actions = len(self._lookup_table)

        # Obs builder — re-created on reset to handle player count changes
        self._n_players = n_players
        self._obs_builder: Optional[NextoObsBuilder] = None

        # Previous action per player (keyed by sorted index)
        self._prev_actions: dict[int, np.ndarray] = {}

        # Cumulative scores (tracked from goal events)
        self.blue_score: int = 0
        self.orange_score: int = 0

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def reset(self, game_state) -> None:
        """
        Call on every episode reset to reinitialize obs builder state
        and clear previous actions.

        Parameters
        ----------
        game_state : rlgym v2 GameState
        """
        n_players = len(game_state.cars)
        self._obs_builder = NextoObsBuilder(
            field_info=None,
            n_players=n_players,
            tick_skip=self.tick_skip,
        )

        # Build the encoded state just to trigger _reset
        encoded = encode_v2_state(
            game_state,
            blue_score=self.blue_score,
            orange_score=self.orange_score,
        )
        encoded_batch = np.expand_dims(encoded, axis=0)

        # Dummy GameState-like wrapper for _reset (needs .players and .boost_pads)
        class _DummyState:
            def __init__(self, n_players, n_boosts):
                self.players = [None] * n_players
                self.boost_pads = np.zeros(n_boosts)

        n_boosts = len(game_state.boost_pad_timers)
        dummy = _DummyState(n_players, n_boosts)
        self._obs_builder._reset(dummy)

        # Reset previous actions (8-dim zero vector per player)
        self._prev_actions = {}
        for i, agent_id in enumerate(sorted(game_state.cars.keys())):
            self._prev_actions[i] = np.zeros(8, dtype=np.float32)

    def act(
        self,
        game_state,
        player_index: int = 0,
    ) -> int:
        """
        Return the teacher's chosen action index (argmax over logits)
        for the specified player.

        Parameters
        ----------
        game_state : rlgym v2 GameState
        player_index : int
            Which player in the sorted-cars list to get the action for.
            Default 0 = first (blue) player.

        Returns
        -------
        int – action index into the 90-action lookup table
        """
        logits = self.get_logits(game_state, player_index)
        action_idx = int(np.argmax(logits))

        # Update prev_actions with the chosen action's 8-dim vector
        self._prev_actions[player_index] = self._lookup_table[action_idx].astype(
            np.float32
        )

        return action_idx

    def get_logits(
        self,
        game_state,
        player_index: int = 0,
    ) -> np.ndarray:
        """
        Return raw logits from the teacher for the specified player.

        Parameters
        ----------
        game_state : rlgym v2 GameState
        player_index : int

        Returns
        -------
        np.ndarray – shape (num_actions,) float32 logits
        """
        if self._obs_builder is None:
            raise RuntimeError(
                "Teacher not initialized. Call teacher.reset(game_state) first."
            )

        # 1. Encode v2 state → flat array
        encoded = encode_v2_state(
            game_state,
            blue_score=self.blue_score,
            orange_score=self.orange_score,
        )
        encoded_batch = np.expand_dims(encoded, axis=0)

        # 2. Build obs via NextoObsBuilder
        all_obs = self._obs_builder.batched_build_obs(encoded_batch)

        # 3. Inject previous actions for the target player
        prev_act = self._prev_actions.get(
            player_index, np.zeros(8, dtype=np.float32)
        )
        self._obs_builder.add_actions(all_obs, prev_act, player_index)

        # 4. Get the obs tuple for target player
        q, kv, mask = all_obs[player_index]

        # 5. Convert to tensors and run model
        state_tuple = (
            torch.from_numpy(q).float().to(self.device),
            torch.from_numpy(kv).float().to(self.device),
            torch.from_numpy(mask).float().to(self.device),
        )

        with torch.no_grad():
            out, _weights = self.model(state_tuple)

        # Process logits (same padding logic as nexto/agent.py)
        out = (out,)
        max_shape = max(o.shape[-1] for o in out)
        logits = torch.stack(
            [
                l if l.shape[-1] == max_shape
                else F.pad(l, pad=(0, max_shape - l.shape[-1]), value=float("-inf"))
                for l in out
            ],
            dim=1,
        )

        # logits shape: (1, 1, num_actions) → squeeze to (num_actions,)
        logits_np = logits.squeeze().cpu().numpy()
        return logits_np

    def update_score(self, game_state) -> None:
        """
        Call after each step to detect and track goals.
        Uses v2 GameState.goal_scored flag.
        """
        if game_state.goal_scored:
            # Determine which team scored from ball y position
            # ball y > 0 → scored in orange goal → blue scored
            if game_state.ball.position[1] > 0:
                self.blue_score += 1
            else:
                self.orange_score += 1

    def reset_scores(self) -> None:
        """Reset score counters (call at the start of a new match)."""
        self.blue_score = 0
        self.orange_score = 0


# --------------------------------------------------------------------- #
# Quick smoke-test
# --------------------------------------------------------------------- #
if __name__ == "__main__":
    teacher = NextoTeacher()
    print(f"Teacher loaded successfully!")
    print(f"  LUT size: {teacher.num_actions}")
    print(f"  Device:   {teacher.device}")
    print(f"  Model:    {type(teacher.model).__name__}")
