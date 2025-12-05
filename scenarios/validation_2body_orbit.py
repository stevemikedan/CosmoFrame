"""
2-Body Circular Orbit Validation Scenario

Tests Velocity Verlet integrator stability and energy conservation.
Developer scenario for PS2.4 validation.
"""
import jax.numpy as jnp
from state import UniverseConfig, UniverseState, initialize_state
from entities import spawn_entity
from kernel import step_simulation

DEVELOPER_SCENARIO = True

SCENARIO_PARAMS = {
    "separation": {"type": "float", "default": 2.0, "min": 0.5, "max": 10.0},
    "mass": {"type": "float", "default": 1.0, "min": 0.1, "max": 10.0},
    "G": {"type": "float", "default": 1.0, "min": 0.1, "max": 10.0},
    "dt": {"type": "float", "default": 0.01, "min": 0.001, "max": 0.1},
}

def build_config(params: dict | None = None) -> UniverseConfig:
    p = params or {}
    dt = p.get('dt', 0.01)
    G = p.get('G', 1.0)
    
    return UniverseConfig(
        topology_type=0,  # Flat
        physics_mode=0,
        radius=50.0,
        max_entities=2,
        max_nodes=1,
        dt=dt,
        c=1.0,
        G=G,
        dim=2,
        enable_diagnostics=True
    )

def build_initial_state(config: UniverseConfig, params: dict | None = None) -> UniverseState:
    state = initialize_state(config)
    p = params or {}
    
    mass = p.get('mass', 1.0)
    sep = p.get('separation', 2.0)
    
    # Circular orbit setup
    # Two equal masses separated by 'sep', orbiting their center of mass
    half_sep = sep / 2.0
    
    # Orbital velocity for circular orbit: v = sqrt(G*M / (4*R))
    # where R = half_sep
    v_circ = jnp.sqrt(config.G * mass / (4.0 * half_sep))
    
    pos1 = jnp.array([-half_sep, 0.0])
    pos2 = jnp.array([half_sep, 0.0])
    vel1 = jnp.array([0.0, -v_circ])
    vel2 = jnp.array([0.0, v_circ])
    
    state = spawn_entity(state, pos1, vel1, mass, 1)
    state = spawn_entity(state, pos2, vel2, mass, 1)
    
    state.scenario_name = "2body_orbit"
    return state

def run(config, state, steps=300):
    """Run 2-body orbit with diagnostics monitoring."""
    print(f"[2BODY_ORBIT] Initial E = {float(state.total_energy):.6f}")
    
    for i in range(steps):
        state = step_simulation(state, config)
        
        # Print diagnostics every 100 steps
        if (i + 1) % 100 == 0:
            print(f"[2BODY_ORBIT] Step {i+1}: E = {float(state.total_energy):.6f}, "
                  f"KE = {float(state.kinetic_energy):.6f}, "
                  f"PE = {float(state.potential_energy):.6f}")
    
    print(f"[2BODY_ORBIT] Final E = {float(state.total_energy):.6f}")
    return state
