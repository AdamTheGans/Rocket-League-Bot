# src/state_setters/mixed_state_setter.py
"""
Probabilistic state setter that randomly selects from a pool of setters.

Used for multi-task training: e.g., 60% normal kickoffs + 40% mechanic
trajectories.  Tracks which setter was chosen so downstream rewards and
callbacks can branch on episode type.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np


class MixedStateSetter:
    """
    RLGym v2 state mutator that probabilistically delegates to child setters.

    Parameters
    ----------
    setters : list
        State setter objects, each with an ``apply(state, shared_info)`` method.
    probabilities : list[float]
        Selection probability for each setter. Must sum to ~1.0.
    names : list[str], optional
        Human-readable names for each setter. If not provided, uses the
        class name of each setter.

    Attributes
    ----------
    last_setter_used : object or None
        Reference to the setter that was last selected.
    last_setter_name : str
        Name of the setter that was last selected.

    Example
    -------
    >>> from rlgym.rocket_league.state_mutators import KickoffMutator
    >>> from state_setters.trajectory_setter import MechanicTrajectorySetter
    >>>
    >>> mixed = MixedStateSetter(
    ...     setters=[KickoffMutator(), MechanicTrajectorySetter("../extracted_mechanics")],
    ...     probabilities=[0.6, 0.4],
    ...     names=["normal", "kuxir"],
    ... )
    """

    def __init__(
        self,
        setters: List,
        probabilities: List[float],
        names: Optional[List[str]] = None,
    ):
        if len(setters) != len(probabilities):
            raise ValueError(
                f"setters ({len(setters)}) and probabilities ({len(probabilities)}) "
                f"must have the same length."
            )

        probs = np.array(probabilities, dtype=np.float64)
        if abs(probs.sum() - 1.0) > 1e-4:
            raise ValueError(
                f"Probabilities must sum to ~1.0, got {probs.sum():.6f}. "
                f"Values: {probabilities}"
            )
        # Normalize to fix any floating-point drift
        self.probabilities = probs / probs.sum()

        self.setters = list(setters)
        self.names = (
            list(names) if names
            else [type(s).__name__ for s in setters]
        )

        # ── Tracking ────────────────────────────────────────────────
        self.last_setter_used = None
        self.last_setter_name: str = ""

    def apply(self, state, shared_info: Optional[dict] = None) -> None:
        """
        Select a setter at random (weighted by probabilities) and apply it.

        Writes the setter's name to ``shared_info["setter_type"]`` if the
        chosen setter didn't set it already.
        """
        idx = int(np.random.choice(len(self.setters), p=self.probabilities))

        setter = self.setters[idx]
        name = self.names[idx]

        # Apply the chosen setter
        if shared_info is None:
            shared_info = {}
        setter.apply(state, shared_info)

        # Track which setter was used
        self.last_setter_used = setter
        self.last_setter_name = name

        # Ensure setter_type is tagged (the setter might have set it already)
        if "setter_type" not in shared_info:
            shared_info["setter_type"] = name
