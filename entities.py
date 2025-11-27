"""
Entity management for CosmoSim.

This module handles the lifecycle of entities (particles, massive bodies)
in the simulation. It provides JAX-compatible functions to spawn and
despawn entities within the fixed-size arrays of UniverseState.
"""

import jax
import jax.numpy as jnp
from state import UniverseState, UniverseConfig
from typing import Tuple


def allocate_entities(cfg: UniverseConfig) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Allocate zero-initialized entity arrays consistent with UniverseConfig.

    Provides a consistent way to allocate entity arrays with the correct shapes
    and dtypes based on the simulation configuration.

    Args:
        cfg: Universe configuration containing max_entities and dim

    Returns:
        Tuple of (pos, vel, mass, type_id, active) where:
            - pos:     (max_entities, dim) - positions
            - vel:     (max_entities, dim) - velocities
            - mass:    (max_entities,) - masses
            - type_id: (max_entities,) - entity type identifiers
            - active:  (max_entities,) - boolean activation mask (as int)

    Example:
        >>> cfg = UniverseConfig(max_entities=10, dim=2, ...)
        >>> pos, vel, mass, type_id, active = allocate_entities(cfg)
        >>> # pos.shape == (10, 2), all zeros
    """
    pos = jnp.zeros((cfg.max_entities, cfg.dim))
    vel = jnp.zeros((cfg.max_entities, cfg.dim))
    mass = jnp.zeros((cfg.max_entities,))
    type_id = jnp.zeros((cfg.max_entities,), dtype=int)
    active = jnp.zeros((cfg.max_entities,), dtype=int)

    return pos, vel, mass, type_id, active


def spawn_entity(state: UniverseState, position: jnp.ndarray, velocity: jnp.ndarray, 
                 mass: float, ent_type: int) -> UniverseState:
    """Spawn a new entity in the first available slot.
    
    Args:
        state: Current universe state
        position: Position vector (dim,)
        velocity: Velocity vector (dim,)
        mass: Mass scalar
        ent_type: Entity type integer
        
    Returns:
        Updated state with new entity, or original state if full.
    """
    # Validate dimensionality
    if position.shape[-1] != state.entity_pos.shape[-1] or velocity.shape[-1] != state.entity_vel.shape[-1]:
        raise ValueError("spawn_entity: position/velocity dimensionality does not match UniverseConfig.dim")

    # Find the first inactive index
    # argmin on boolean array returns index of first False (0)
    # If all are True (1), it returns 0. So we must check if any are False.
    idx = jnp.argmin(state.entity_active)
    is_full = jnp.all(state.entity_active)
    
    # Create the updated state eagerly (JAX lazy evaluation handles this efficiently)
    # We use .at[idx].set() for functional updates
    new_active = state.entity_active.at[idx].set(True)
    new_pos = state.entity_pos.at[idx].set(position)
    new_vel = state.entity_vel.at[idx].set(velocity)
    new_mass = state.entity_mass.at[idx].set(mass)
    new_type = state.entity_type.at[idx].set(ent_type)
    
    new_state = state.replace(
        entity_active=new_active,
        entity_pos=new_pos,
        entity_vel=new_vel,
        entity_mass=new_mass,
        entity_type=new_type
    )
    
    # Return new_state if not full, otherwise return original state
    return jax.lax.cond(
        is_full,
        lambda _: state,
        lambda _: new_state,
        operand=None
    )


def despawn_entity(state: UniverseState, index: int) -> UniverseState:
    """Despawn an entity by marking it as inactive.
    
    Args:
        state: Current universe state
        index: Index of the entity to despawn
        
    Returns:
        Updated state with entity_active[index] = False
    """
    # Simply set active to False. Data remains but is ignored by physics.
    new_active = state.entity_active.at[index].set(False)
    return state.replace(entity_active=new_active)
