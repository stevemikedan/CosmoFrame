
import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import jax
import jax.numpy as jnp
import kernel
from state import UniverseConfig, initialize_state

print("Debug: Starting...")
cfg = UniverseConfig(
    physics_mode=0,
    radius=10.0,
    max_entities=2,
    max_nodes=1,
    dt=0.1,
    c=1.0,
    G=1.0,
    enable_diagnostics=True,
    topology_type=0
)
state = initialize_state(cfg)
print("Debug: State initialized.")

# Force entity positions to something non-zero to test gravity
state = state.replace(
    entity_pos=jnp.array([[1.0, 0.0], [-1.0, 0.0]]),
    entity_mass=jnp.array([1.0, 1.0]),
    entity_active=jnp.array([True, True], dtype=bool)
)

print("Debug: Running step 1...")
state = kernel.step_simulation(state, cfg)
# Block to ensure completion
state.entity_pos.block_until_ready()
print("Debug: Step 1 complete.")
print(f"Pos: {state.entity_pos}")

print("Debug: Running step 2...")
state = kernel.step_simulation(state, cfg)
state.entity_pos.block_until_ready()
print("Debug: Step 2 complete.")
print(f"Pos: {state.entity_pos}")
