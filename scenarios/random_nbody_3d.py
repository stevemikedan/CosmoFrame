"""
Flagship 3D N-Body Simulation
Advanced 3D distributions and dynamics.
"""
import jax
import jax.numpy as jnp
from state import UniverseConfig, UniverseState, initialize_state
from entities import spawn_entity
from kernel import step_simulation

SCENARIO_PARAMS = {
    "N": {"type": "int", "default": 300, "min": 10, "max": 10000},
    "radius": {"type": "float", "default": 20.0, "min": 1.0, "max": 200.0},
    "distribution": {"type": "str", "default": "spherical", "allowed": ["spherical", "gaussian", "disc"]},
    "rotation": {"type": "float", "default": 0.0, "min": 0.0, "max": 5.0}, # Angular velocity factor
    "expansion": {"type": "float", "default": 0.0, "min": -5.0, "max": 5.0}, # Radial velocity factor
    "G": {"type": "float", "default": 1.0, "min": 0.1, "max": 100.0},
    "dt": {"type": "float", "default": 0.01, "min": 0.001, "max": 1.0},
    "dim": {"type": "int", "default": 3, "allowed": [3]} # Strictly 3D
}

SCENARIO_PRESETS = {
    "galaxy_disc": {
        "distribution": "disc",
        "N": 500,
        "rotation": 1.0,
        "radius": 30.0,
        "G": 2.0
    },
    "globular_cluster": {
        "distribution": "gaussian",
        "N": 400,
        "radius": 15.0,
        "rotation": 0.2, # Slight rotation
    },
    "expanding_cloud": {
        "distribution": "spherical",
        "N": 300,
        "expansion": 2.0,
        "G": 0.5, # Low gravity
    },
    "high_energy_chaos": {
        "distribution": "spherical",
        "N": 200,
        "radius": 10.0,
        "expansion": 0.0,
        "rotation": 0.0,
        "G": 10.0, # Strong gravity
        "dt": 0.005
    }
}

def build_config(params: dict | None = None) -> UniverseConfig:
    p = params or {}
    dt = p.get('dt', 0.01)
    G = p.get('G', 1.0)
    radius = p.get('radius', 20.0)
    n = p.get('N', 300)
    
    return UniverseConfig(
        topology_type=0,
        physics_mode=0,
        radius=radius,
        max_entities=n,
        max_nodes=1,
        dt=dt,
        c=1.0,
        G=G,
        dim=3 # Enforce 3D
    )

def build_initial_state(config: UniverseConfig, params: dict | None = None) -> UniverseState:
    state = initialize_state(config)
    p = params or {}
    
    n = config.max_entities
    radius = config.radius
    dist_type = p.get('distribution', 'spherical')
    rotation = p.get('rotation', 0.0)
    expansion = p.get('expansion', 0.0)
    
    key = jax.random.PRNGKey(42)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    
    if dist_type == "spherical":
        # Uniform in sphere
        # r ~ U[0, 1]^(1/3) for uniform volume
        u = jax.random.uniform(k1, (n,))
        r = radius * jnp.cbrt(u)
        
        # Random direction
        # z ~ U[-1, 1], theta ~ U[0, 2pi]
        z = jax.random.uniform(k2, (n,), minval=-1.0, maxval=1.0)
        theta = jax.random.uniform(k3, (n,), minval=0, maxval=2*jnp.pi)
        r_xy = jnp.sqrt(1.0 - z**2)
        x = r_xy * jnp.cos(theta)
        y = r_xy * jnp.sin(theta)
        
        pos = jnp.stack([x, y, z], axis=1) * r[:, None]
        
    elif dist_type == "gaussian":
        # Gaussian cloud
        pos = jax.random.normal(k1, (n, 3)) * (radius / 3.0) # 3 sigma ~ radius
        
    elif dist_type == "disc":
        # Disc in XY plane with some thickness
        # r ~ uniform or gaussian? Let's do uniform disc
        r = radius * jnp.sqrt(jax.random.uniform(k1, (n,)))
        theta = jax.random.uniform(k2, (n,), minval=0, maxval=2*jnp.pi)
        
        x = r * jnp.cos(theta)
        y = r * jnp.sin(theta)
        z = jax.random.normal(k3, (n,)) * (radius * 0.05) # Thin disc
        
        pos = jnp.stack([x, y, z], axis=1)
        
    else:
        # Fallback
        pos = jax.random.uniform(k1, (n, 3), minval=-radius, maxval=radius)

    # Velocity Initialization
    vel = jnp.zeros((n, 3))
    
    # 1. Expansion (Radial velocity)
    if expansion != 0.0:
        # v_rad = expansion * (r / radius)
        norms = jnp.linalg.norm(pos, axis=1, keepdims=True) + 1e-6
        dirs = pos / norms
        vel += dirs * expansion * (norms / radius)
        
    # 2. Rotation (Tangential velocity)
    if rotation != 0.0:
        # Rotate around Z axis
        # v = omega x r
        # omega = [0, 0, rotation]
        # v_x = -omega * y
        # v_y = omega * x
        v_rot_x = -rotation * pos[:, 1]
        v_rot_y = rotation * pos[:, 0]
        v_rot_z = jnp.zeros(n)
        
        vel_rot = jnp.stack([v_rot_x, v_rot_y, v_rot_z], axis=1)
        
        # Scale rotation by distance? Keplerian?
        # For solid body: v ~ r (implemented above)
        # For Keplerian: v ~ 1/sqrt(r)
        # Let's stick to solid body-ish or constant V for simplicity in this demo,
        # or just the simple cross product which is solid body.
        vel += vel_rot

    # Add small random thermal noise
    vel += jax.random.normal(k4, (n, 3)) * 0.1
    
    masses = jnp.ones((n,))
    
    for i in range(n):
        state = spawn_entity(state, pos[i], vel[i], masses[i], 1)
        
    state.scenario_name = "random_nbody_3d"
    return state

def run(config, state, steps=300):
    for _ in range(steps):
        state = step_simulation(state, config)
    return state
