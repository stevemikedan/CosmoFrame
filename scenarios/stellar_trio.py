"""
Stellar Trio – 3-body orbital dance.
Fully compatible with CosmoSim’s UniverseState and UniverseConfig.
"""

from __future__ import annotations

import jax.numpy as jnp

from state import UniverseConfig, UniverseState, initialize_state
from entities import spawn_entity
from kernel import step_simulation

SCENARIO_PARAMS = {
    "dim": {"type": "int", "default": 2, "allowed": [2, 3]}
}

SCENARIO_PRESETS = {
    "stable_figure8": {
        "dim": 2
    },
    "chaotic_3d": {
        "dim": 3,
    },
    "stable_plane_3d": {
        "dim": 3,
    },
    "inverted_orbits": {
        "dim": 3,
    }
}

def build_config(params: dict | None = None):
    """
    Create a 3-body configuration.
    """
    p = params or {}
    radius = p.get('radius', 20.0)
    dt = p.get('dt', 0.001)
    G = p.get('G', 1.0)
    c = p.get('c', 1.0)
    topology_type = p.get('topology_type', 0)
    physics_mode = p.get('physics_mode', 0)
    dim = p.get('dim', 2)

    return UniverseConfig(
        physics_mode=physics_mode,        # 0 = VECTOR (Newtonian gravity)
        radius=radius,
        max_entities=3,
        max_nodes=1,
        dt=dt,
        c=c,
        G=G,
        dim=dim,
        topology_type=topology_type,       # flat
        bounds=radius,
    )


def build_initial_state(cfg: UniverseConfig, params: dict | None = None) -> UniverseState:
    """
    Build the initial UniverseState with three orbiting stars.
    """
    state = initialize_state(cfg)
    
    if cfg.dim == 3:
        # 3D Initialization
        # Base figure-8 in XY plane
        pos = jnp.array([
            [0.97000436, -0.24308753, 0.0],
            [-0.97000436, 0.24308753, 0.0],
            [0.0, 0.0, 0.0]
        ])
        vel = jnp.array([
            [0.4662036850, 0.4323657300, 0.0],
            [0.4662036850, 0.4323657300, 0.0],
            [-2*0.4662036850, -2*0.4323657300, 0.0]
        ])
        
        # Apply simple Z perturbation for "chaotic_3d" or "inverted_orbits" if implied
        # Since we don't have explicit preset name here easily without passing it,
        # we'll just add a tiny Z offset to make it 3D-valid but mostly planar,
        # unless user hacks it.
        # Ideally we'd check params['preset'] if passed, but PSS passes merged params.
        # Let's add a small Z velocity to one star to induce 3D chaos if dim=3
        vel = vel.at[0, 2].set(0.1) 
        
    else:
        # 2D Initialization (Figure-8 solution)
        pos = jnp.array([
            [0.97000436, -0.24308753],
            [-0.97000436, 0.24308753],
            [0.0, 0.0]
        ])
        vel = jnp.array([
            [0.4662036850, 0.4323657300],
            [0.4662036850, 0.4323657300],
            [-2*0.4662036850, -2*0.4323657300]
        ])
    
    masses = jnp.array([1.0, 1.0, 1.0]) # Figure-8 requires equal masses
    
    # Spawn all three stars
    for i in range(3):
        state = spawn_entity(state, pos[i], vel[i], masses[i], 1)
    
    state.scenario_name = "stellar_trio"
    return state


def run(cfg: UniverseConfig, state: UniverseState, steps: int = 300):
    """
    Standard simulation loop.
    """
    for _ in range(steps):
        state = step_simulation(state, cfg)

    return state
