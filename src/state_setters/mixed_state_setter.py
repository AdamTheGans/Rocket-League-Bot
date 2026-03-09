# src/state_setters/mixed_state_setter.py
"""
Probabilistic state setter that randomly selects from a pool of setters.
"""
from __future__ import annotations

from typing import List, Optional
import numpy as np

class MixedStateSetter:
    def __init__(
        self,
        setters: List,
        probabilities: List[float],
        names: Optional[List[str]] = None,
    ):
        if len(setters) != len(probabilities):
            raise ValueError("setters and probabilities must have the same length.")

        probs = np.array(probabilities, dtype=np.float64)
        self.probabilities = probs / probs.sum()
        self.setters = list(setters)
        self.names = list(names) if names else [type(s).__name__ for s in setters]

        self.last_setter_used = None
        self.last_setter_name: str = ""

    def apply(self, state, shared_info: Optional[dict] = None) -> None:
        idx = int(np.random.choice(len(self.setters), p=self.probabilities))
        setter = self.setters[idx]
        name = self.names[idx]

        if shared_info is None:
            shared_info = {}
            
        setter.apply(state, shared_info)

        self.last_setter_used = setter
        self.last_setter_name = name

        # Only set the name if the child setter didn't already override it
        if "setter_type" not in shared_info:
            shared_info["setter_type"] = name