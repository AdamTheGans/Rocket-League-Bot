# src/nexto_distill/student_policy.py
"""
Configurable MLP student policy for behavior-cloning from Nexto.

Outputs logits over the 90-action discrete action space (same LUT
as Nexto and rlgym's LookupTableAction).
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class StudentPolicy(nn.Module):
    """
    Flat MLP that maps an observation vector to action logits.

    Parameters
    ----------
    obs_dim : int
        Dimension of the input observation.
    num_actions : int
        Number of discrete actions (90 for the standard LUT).
    layer_sizes : list[int]
        Hidden layer sizes.  Default ``[2048, 1024, 1024, 512]``.
    """

    def __init__(
        self,
        obs_dim: int,
        num_actions: int = 90,
        layer_sizes: List[int] | None = None,
    ):
        super().__init__()
        if layer_sizes is None:
            layer_sizes = [2048, 1024, 1024, 512]

        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.layer_sizes = list(layer_sizes)

        layers: list[nn.Module] = []
        in_dim = obs_dim
        for hidden_dim in layer_sizes:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim

        # Final projection to action logits (no activation)
        layers.append(nn.Linear(in_dim, num_actions))

        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        obs : Tensor, shape ``(batch, obs_dim)``

        Returns
        -------
        logits : Tensor, shape ``(batch, num_actions)``
        """
        return self.net(obs)
