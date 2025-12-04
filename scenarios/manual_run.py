import jax
import jax.numpy as jnp

from state import UniverseConfig, initialize_state
from entities import spawn_entity
from kernel import step_simulation

DEVELOPER_SCENARIO = True

def build_config() -> UniverseConfig:
    return UniverseConfig(
        topology_type=0,
        physics_mode=0,
        radius=10.0,
        max_entities=3,
        max_nodes=1,
        dt=0.1,
        c=1.0,
        G=1.0,
        dim=2
    )


def build_initial_state(config: UniverseConfig):
    state = initialize_state(config)
    # simple 3-body line
    # Ensure dim-safe initialization
    if config.dim == 3:
        pos_a = jnp.array([-2.0, 0.0, 0.0])
        pos_b = jnp.array([ 0.0, 0.0, 0.0])
        pos_c = jnp.array([ 2.0, 0.0, 0.0])
        vel = jnp.zeros(3)
    else:
        pos_a = jnp.array([-2.0, 0.0])
        pos_b = jnp.array([ 0.0, 0.0])
        pos_c = jnp.array([ 2.0, 0.0])
        vel = jnp.zeros(2)

    state = spawn_entity(state, pos_a, vel, 2.0, 1)
    state = spawn_entity(state, pos_b, vel, 1.0, 1)
    state = spawn_entity(state, pos_c, vel, 2.0, 1)
    state.scenario_name = "manual_run"
    return state


def run(config: UniverseConfig, state):
    print("Running manual physics test...")
    
    step_fn = jax.jit(step_simulation)

    for i in range(20):
        state = step_fn(state, config)
        print(f"Step {i:02d}: {state.entity_pos}")
    return state


if __name__ == "__main__":
    cfg = build_config()
    state = build_initial_state(cfg)
    run(cfg, state)
