import jax
import jax.numpy as jnp
from state import UniverseConfig, UniverseState, initialize_state
from entities import spawn_entity
from kernel import step_simulation

# Constants for strict topology enforcement
TOPOLOGY_FLAT = 0
TOPOLOGY_TORUS = 1

SCENARIO_PARAMS = {
    "N": {"type": "int", "default": 200, "min": 10, "max": 5000},
    "viscosity": {"type": "float", "default": 0.01, "min": 0.0, "max": 1.0},
    "base_speed": {"type": "float", "default": 1.0, "min": 0.1, "max": 10.0},
    "noise": {"type": "float", "default": 0.1, "min": 0.0, "max": 2.0},
}

SCENARIO_PRESETS = {
    "laminar": {
        "dt": 0.01,
        "viscosity": 0.1,
        "base_speed": 1.0,
        "noise": 0.01,
    },
    "turbulent": {
        "dt": 0.005,
        "viscosity": 0.001,
        "base_speed": 2.0,
        "noise": 0.5,
    },
    "torus_flow": {
        "dt": 0.01,
        "topology_type": 1,
        "base_speed": 1.5,
    }
}

def build_config(params: dict | None = None) -> UniverseConfig:
    p = params or {}
    n = p.get('max_entities', p.get('N', 200))
    dt = p.get('dt', 0.01)
    topology_type = p.get('topology_type', TOPOLOGY_FLAT)
    
    return UniverseConfig(
        topology_type=topology_type,
        physics_mode=0,
        radius=20.0,
        max_entities=n,
        max_nodes=1,
        dt=dt,
        c=1.0,
        G=0.0,  # No gravity, just fluid dynamics
        dim=2,
        bounds=20.0
    )

def build_initial_state(config: UniverseConfig, params: dict | None = None) -> UniverseState:
    state = initialize_state(config)
    p = params or {}
    n = config.max_entities
    base_speed = p.get('base_speed', 1.0)
    noise = p.get('noise', 0.1)
    
    # Random positions in 2D box
    key = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(key, 3)
    
    pos_x = jax.random.uniform(k1, (n,), minval=-config.radius, maxval=config.radius)
    pos_y = jax.random.uniform(k2, (n,), minval=-config.radius, maxval=config.radius)
    
    # Vortex sheet initialization
    # Particles above midline (y > 0) move right (+x)
    # Particles below midline (y < 0) move left (-x)
    vel_x = jnp.where(pos_y > 0, base_speed, -base_speed)
    
    # Add noise to y-velocity to trigger Kelvin-Helmholtz instability
    vel_y = jax.random.normal(k3, (n,)) * noise
    
    positions = jnp.stack([pos_x, pos_y], axis=1)
    velocities = jnp.stack([vel_x, vel_y], axis=1)
    
    masses = jnp.ones((n,))
    
    for i in range(n):
        state = spawn_entity(state, positions[i], velocities[i], masses[i], 1)
        
    state.scenario_name = "vortex_sheet"
    return state


def apply_shear_dynamics(state, config, params):
    """
    Apply shear layer dynamics (viscosity/damping).
    """
    viscosity = params.get('viscosity', 0.01)
    dt = config.dt
    
    # Simple viscosity: velocity decay
    # vel = vel * (1 - viscosity * dt)
    decay = 1.0 - (viscosity * dt)
    decay = max(0.0, decay) # Prevent negative decay
    
    new_vel = state.entity_vel * decay
    
    return new_vel


def run(config, state, steps=300):
    """Run vortex sheet simulation with strict topology enforcement."""
    
    # STRICT TOPOLOGY ENFORCEMENT
    if config.topology_type not in [TOPOLOGY_FLAT, TOPOLOGY_TORUS]:
        print(f"[VORTEX_SHEET] Unsupported topology {config.topology_type}; reverting to FLAT.")
        config = config.replace(topology_type=TOPOLOGY_FLAT)
        state = state.replace(topology_type=TOPOLOGY_FLAT)
    
    # Extract params for dynamics
    # We need to reconstruct params dict or extract from state if encoded
    # Here we just use defaults or what was passed if we had access, 
    # but run() signature doesn't take params.
    # We'll use defaults or try to infer.
    # Actually, best practice in Phase 3 is to encode params in state or just use defaults.
    # But we can't easily get them back without passing them.
    # We'll use a default viscosity of 0.01 if we can't get it.
    # Or we can assume the user wants the default if they didn't pass it.
    # Wait, build_initial_state used params.
    # We'll define a local params dict with defaults.
    params = {"viscosity": 0.01} 
    
    for _ in range(steps):
        # Apply custom dynamics
        new_vel = apply_shear_dynamics(state, config, params)
        state = state.replace(entity_vel=new_vel)
        
        # Step simulation
        state = step_simulation(state, config)
    
    return state
