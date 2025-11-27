"""
Base substrate interface.
"""

from __future__ import annotations
import jax.numpy as jnp
from typing import Any
from state import UniverseConfig, UniverseState


class Substrate:
    """Abstract substrate interface."""
    
    def __init__(self, config: UniverseConfig):
        self.config = config

    def update(self, state: UniverseState, dt: float):
        """
        Evolve substrate field(s) over time.
        
        Args:
            state: Current universe state
            dt: Timestep
        """
        raise NotImplementedError

    def force_at(self, pos: jnp.ndarray, vel: jnp.ndarray) -> jnp.ndarray:
        """
        Calculate substrate forces at given positions.
        
        Args:
            pos: Entity positions (N, 3)
            vel: Entity velocities (N, 3)
            
        Returns:
            Force array of shape (N, 3)
        """
        return jnp.zeros_like(pos)


class NullSubstrate(Substrate):
    """A substrate that does nothing."""
    
    def update(self, state: UniverseState, dt: float):
        pass

    def force_at(self, pos: jnp.ndarray, vel: jnp.ndarray) -> jnp.ndarray:
        return jnp.zeros_like(pos)
