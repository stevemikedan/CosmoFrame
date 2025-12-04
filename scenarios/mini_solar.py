"""
Mini Solar System
A central sun with orbiting planets.
"""
import jax
import jax.numpy as jnp
from state import UniverseConfig, UniverseState, initialize_state
from entities import spawn_entity
from kernel import step_simulation

SCENARIO_PARAMS = {
    "sun_mass": {"type": "float", "default": 20.0, "min": 1.0, "max": 1000.0},
    "num_planets": {"type": "int", "default": 3, "min": 1, "max": 10},
    "G": {"type": "float", "default": 1.0, "min": 0.1, "max": 100.0},
    "dt": {"type": "float", "default": 0.03, "min": 0.001, "max": 1.0},
    "dim": {"type": "int", "default": 2, "allowed": [2, 3]}
}

SCENARIO_PRESETS = {
    "three_planet": {
        "num_planets": 3
    },
    "compact_system": {
        "num_planets": 5,
        "G": 2.0, # Stronger gravity for faster orbits
    },
    "outer_belt": {
        "num_planets": 8,
        "sun_mass": 50.0,
    }
}

def build_config(params: dict | None = None) -> UniverseConfig:
    p = params or {}
    dt = p.get('dt', 0.03)
    G = p.get('G', 1.0)
    dim = p.get('dim', 2)
    n_planets = p.get('num_planets', 3)
    
    return UniverseConfig(
        topology_type=0,
        physics_mode=0,
        radius=50.0,
        max_entities=1 + n_planets,
        max_nodes=1,
        dt=dt,
        c=1.0,
        G=G,
        dim=dim
    )

def build_initial_state(config: UniverseConfig, params: dict | None = None) -> UniverseState:
    state = initialize_state(config)
    p = params or {}
    
    sun_mass = p.get('sun_mass', 20.0)
    n_planets = p.get('num_planets', 3)
    
    # Sun at origin
    if config.dim == 3:
        sun_pos = jnp.zeros(3)
        sun_vel = jnp.zeros(3)
    else:
        sun_pos = jnp.zeros(2)
        sun_vel = jnp.zeros(2)
        
    state = spawn_entity(state, sun_pos, sun_vel, sun_mass, 1)
    
    # Planets
    # Generate orbits based on distance
    # v = sqrt(G*M / r)
    
    for i in range(n_planets):
        dist = 4.0 * (i + 1)
        mass = 1.0 + (i % 2) * 0.5 # Alternating mass slightly
        
        v_orb = jnp.sqrt(config.G * sun_mass / dist)
        
        if config.dim == 3:
            # Planar orbits in XY plane for simplicity, maybe slight inclination later
            pos = jnp.array([float(dist), 0.0, 0.0])
            vel = jnp.array([0.0, float(v_orb), 0.0])
        else:
            pos = jnp.array([float(dist), 0.0])
            vel = jnp.array([0.0, float(v_orb)])
            
        state = spawn_entity(state, pos, vel, mass, 1)
    
    state.scenario_name = "mini_solar"
    return state

def run(config, state, steps=300):
    for _ in range(steps):
        state = step_simulation(state, config)
    return state
