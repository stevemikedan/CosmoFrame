"""
Bulk Ring Scenario - Demonstrates bulk entity allocation.

A ring of equal-mass bodies arranged in a circle with tangential velocities.
This scenario uses allocate_entities() for direct array initialization instead
of spawn_entity() loops.
"""

from __future__ import annotations

import jax.numpy as jnp
from state import UniverseConfig, UniverseState, initialize_state
from entities import allocate_entities
from kernel import step_simulation


def build_config() -> UniverseConfig:
    """
    Create configuration for a ring of orbiting bodies.
    """
    return UniverseConfig(
        physics_mode=0,      # Newtonian gravity
        radius=20.0,
        max_entities=128,
        max_nodes=1,
        dt=0.05,
        c=1.0,
        G=1.0,
        dim=2,
        topology_type=0,     # flat
        bounds=20.0,
    )


def build_initial_state(cfg: UniverseConfig) -> UniverseState:
    """
    Build initial state using bulk allocation.
    
    Creates a ring of bodies with tangential velocities using allocate_entities()
    for efficient direct array initialization.
    """
    # Allocate entity arrays
    pos, vel, mass, type_id, active = allocate_entities(cfg)
    
    # Number of active bodies (max 64 for visual clarity)
    N = min(64, cfg.max_entities)
    
    # Generate ring positions
    angles = jnp.linspace(0.0, 2.0 * jnp.pi, N, endpoint=False)
    ring_radius = 8.0  # orbital radius
    
    pos_xy = jnp.stack(
        [ring_radius * jnp.cos(angles), ring_radius * jnp.sin(angles)],
        axis=1,
    )
    
    # Generate tangential velocities
    speed = 0.8
    vel_xy = jnp.stack(
        [-speed * jnp.sin(angles), speed * jnp.cos(angles)],
        axis=1,
    )
    
    # Fill arrays using bulk updates
    pos = pos.at[:N].set(pos_xy)
    vel = vel.at[:N].set(vel_xy)
    mass = mass.at[:N].set(jnp.ones(N) * 1.0)
    type_id = type_id.at[:N].set(1)
    active = active.at[:N].set(1)
    
    # Create base state and replace entity fields
    base_state = initialize_state(cfg)
    state = base_state.replace(
        entity_pos=pos,
        entity_vel=vel,
        entity_mass=mass,
        entity_type=type_id,
        entity_active=active,
    )
    
    state.scenario_name = "bulk_ring"
    return state


def run(cfg: UniverseConfig, state: UniverseState, steps: int = 300) -> UniverseState:
    """
    Standard simulation loop.
    """
    for _ in range(steps):
        state = step_simulation(state, cfg)
    return state
