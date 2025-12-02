
import jax
import jax.numpy as jnp
import numpy as np
from state import UniverseConfig
from physics_utils import compute_gravity_forces

def debug_gravity():
    print("Setting up debug scenario...")
    
    # Create two particles very close together
    pos = jnp.array([
        [0.0, 0.0, 0.0],
        [0.001, 0.0, 0.0]
    ])
    mass = jnp.array([1.0, 1.0])
    active = jnp.array([True, True])
    
    config = UniverseConfig(
        physics_mode=0,
        radius=10.0,
        max_entities=2,
        max_nodes=1,
        dt=0.1,
        c=1.0,
        G=1.0,
        gravity_softening=0.05,
        topology_type=0  # FLAT
    )
    
    print(f"Config topology: {config.topology_type}")
    
    print("Calling compute_gravity_forces...")
    total_force = compute_gravity_forces(pos, mass, active, config)
    print(f"Total force:\n{total_force}")
    
    norm = jnp.linalg.norm(total_force[0])
    print(f"Force magnitude on p0: {norm}")

if __name__ == "__main__":
    debug_gravity()
