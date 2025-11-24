import jax
import jax.numpy as jnp

from state import UniverseConfig, initialize_state
from entities import spawn_entity
from kernel import step_simulation


def manual_test():
    print("Running manual physics test...")

    cfg = UniverseConfig(
        topology_type=0,
        physics_mode=0,
        radius=10.0,
        max_entities=3,
        max_nodes=1,
        dt=0.1,
        c=1.0,
        G=1.0,
    )

    state = initialize_state(cfg)

    # simple 3-body line
    state = spawn_entity(state, jnp.array([-2.0, 0.0]), jnp.array([0.0, 0.0]), 2.0, 1)
    state = spawn_entity(state, jnp.array([ 0.0, 0.0]), jnp.array([0.0, 0.0]), 1.0, 1)
    state = spawn_entity(state, jnp.array([ 2.0, 0.0]), jnp.array([0.0, 0.0]), 2.0, 1)

    step_fn = jax.jit(step_simulation)

    for i in range(20):
        state = step_fn(state, cfg)
        print(f"Step {i:02d}: {state.entity_pos}")


if __name__ == "__main__":
    manual_test()
