# src/envs/terminations.py
"""
Custom termination conditions for the Rocket League environments.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List

from rlgym.api import AgentID, DoneCondition
from rlgym.rocket_league.api import GameState


class EarlyWhiffCondition(DoneCondition[AgentID, GameState]):
    """
    Terminates the episode early to save compute if:
    1. The car has not touched the ball within `max_whiff_ticks`.
    2. The ball hits the ground (Z < ground_z_threshold) and is not moving fast.
    """
    def __init__(self, max_whiff_ticks: int = 120, ground_z_threshold: float = 100.0, min_speed_threshold: float = 1500.0):
        super().__init__()
        self.max_whiff_ticks = max_whiff_ticks
        self.ground_z_threshold = ground_z_threshold
        self.min_speed_threshold = min_speed_threshold
        
        self.current_tick = 0

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        self.current_tick = 0

    def is_done(
        self,
        agents: List[AgentID],
        state: GameState,
        shared_info: Dict[str, Any],
    ) -> Dict[AgentID, bool]:
        self.current_tick += 1
        
        # 1. Check if ANY car has touched the ball.
        #    If not touched within max_whiff_ticks, terminate to save compute.
        total_touches = sum(car.ball_touches for car in state.cars.values())
        if self.current_tick >= self.max_whiff_ticks and total_touches == 0:
            return {agent: True for agent in agents}
            
        # 2. Check if the ball hit the ground prematurely.
        #    A proper pinch will have the ball way above the ground or moving extremely fast.
        ball_z = state.ball.position[2]
        vel = state.ball.linear_velocity
        ball_speed = math.sqrt(vel[0]**2 + vel[1]**2 + vel[2]**2)
        
        if ball_z < self.ground_z_threshold and ball_speed < self.min_speed_threshold:
            return {agent: True for agent in agents}
            
        return {agent: False for agent in agents}
