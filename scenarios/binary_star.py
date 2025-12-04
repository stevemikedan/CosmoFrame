"""
Binary Star System
Two stars orbiting each other.
"""
import jax
import jax.numpy as jnp
from state import UniverseConfig, UniverseState, initialize_state
from entities import spawn_entity
from kernel import step_simulation

SCENARIO_PARAMS = {
    "mass": {"type": "float", "default": 5.0, "min": 0.1, "max": 100.0},
    "separation": {"type": "float", "default": 10.0, "min": 1.0, "max": 100.0},
    "G": {"type": "float", "default": 2.0, "min": 0.1, "max": 100.0},
    "dt": {"type": "float", "default": 0.05, "min": 0.001, "max": 1.0},
    "dim": {"type": "int", "default": 2, "allowed": [2, 3]}
}

SCENARIO_PRESETS = {
    "wide_binary": {
        "separation": 20.0,
        "dt": 0.05,
    },
    "tight_binary": {
        "separation": 5.0,
        "dt": 0.01,
        "G": 5.0
    },
    "eccentric_binary": {
        "separation": 15.0,
        # Eccentricity handled in logic via velocity factor if we added a param for it
        # For now, just a named preset that might be tweaked manually or we add 'eccentricity' param later.
    }
}

def build_config(params: dict | None = None) -> UniverseConfig:
    p = params or {}
    dt = p.get('dt', 0.05)
    G = p.get('G', 2.0)
    dim = p.get('dim', 2)
    
    return UniverseConfig(
        topology_type=0,
        physics_mode=0,
        radius=30.0,
        max_entities=2,
        max_nodes=1,
        dt=dt,
        c=1.0,
        G=G,
        dim=dim
    )

def build_initial_state(config: UniverseConfig, params: dict | None = None) -> UniverseState:
    state = initialize_state(config)
    p = params or {}
    
    mass = p.get('mass', 5.0)
    sep = p.get('separation', 10.0)
    
    half_sep = sep / 2.0
    
    # Calculate orbital velocity for circular orbit: v = sqrt(G*M / (4*R)) ? 
    # For two equal masses M separated by r=2R, force is F = G*M^2 / r^2.
    # Centripetal force F = M*v^2 / R.
    # G*M^2 / (2R)^2 = M*v^2 / R
    # G*M / 4R^2 = v^2 / R
    # v = sqrt(G*M / 4R)
    
    # R = half_sep
    # v = sqrt(config.G * mass / (4 * half_sep))
    
    v_circ = jnp.sqrt(config.G * mass / (4 * half_sep))
    
    if config.dim == 3:
        pos1 = jnp.array([-half_sep, 0.0, 0.0])
        pos2 = jnp.array([ half_sep, 0.0, 0.0])
        vel1 = jnp.array([0.0, -v_circ, 0.0])
        vel2 = jnp.array([0.0,  v_circ, 0.0])
    else:
        pos1 = jnp.array([-half_sep, 0.0])
        pos2 = jnp.array([ half_sep, 0.0])
        vel1 = jnp.array([0.0, -v_circ])
        vel2 = jnp.array([0.0,  v_circ])
        
    state = spawn_entity(state, pos1, vel1, mass, 1)
    state = spawn_entity(state, pos2, vel2, mass, 1)
    
    state.scenario_name = "binary_star"
    return state

def run(config, state, steps=300):
    for _ in range(steps):
        state = step_simulation(state, config)
    return state
