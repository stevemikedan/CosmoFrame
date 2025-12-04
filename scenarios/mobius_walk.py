import jax
import jax.numpy as jnp
from state import UniverseConfig, UniverseState, initialize_state
from entities import spawn_entity
from kernel import step_simulation

# Möbius strip domain parameters
STRIP_LENGTH = 20.0  # L: u ∈ [0, L]
STRIP_WIDTH = 5.0    # W: v ∈ [-W, W]

SCENARIO_PARAMS = {
    "speed": {"type": "float", "default": 1.0, "min": 0.1, "max": 10.0}
}

SCENARIO_PRESETS = {
    "slow": {
        "dt": 0.1,
        "speed": 1.0,
    },
    "fast": {
        "dt": 0.02,
        "speed": 5.0,
    },
    "wrap-test": {
        "speed": 2.0,
    }
}

def build_config(params: dict | None = None) -> UniverseConfig:
    p = params or {}
    dt = p.get('dt', 0.1)
    
    return UniverseConfig(
        topology_type=0,  # Flat - we handle Möbius wrapping manually
        physics_mode=0,
        radius=max(STRIP_LENGTH, STRIP_WIDTH * 2),
        max_entities=1,
        max_nodes=1,
        dt=dt,
        c=1.0,
        G=0.0,  # No gravity, just walking
        dim=2,
        bounds=max(STRIP_LENGTH, STRIP_WIDTH * 2)
    )

def build_initial_state(config: UniverseConfig, params: dict | None = None) -> UniverseState:
    state = initialize_state(config)
    p = params or {}
    speed = p.get('speed', 1.0)
    
    # Initialize walker in (u, v) coordinates
    # Start at center of strip: u = L/2, v = 0
    # Map to (x, y): x = u, y = v
    initial_u = STRIP_LENGTH / 2.0
    initial_v = 0.0
    
    # Initial velocity: moving in +u direction
    state = spawn_entity(
        state,
        jnp.array([initial_u, initial_v]),  # (x, y) = (u, v)
        jnp.array([speed, 0.0]),             # Moving along the strip
        1.0,
        1
    )
        
    state.scenario_name = "mobius_walk"
    return state


def apply_mobius_wrapping(u, v, L, W):
    """
    Apply Möbius strip boundary conditions.
    
    The essential Möbius topology rule:
    - When crossing u = L: wrap to u = 0, INVERT v
    - When crossing u = 0: wrap to u = L, INVERT v
    
    This creates the non-orientable surface behavior.
    """
    # Count how many times we've wrapped (to track inversions)
    # Each wrap inverts v
    
    # Handle u > L (crossed right edge)
    wrapped_u = jnp.where(u > L, u - L, u)
    wrapped_v = jnp.where(u > L, -v, v)
    
    # Handle u < 0 (crossed left edge)
    wrapped_u = jnp.where(wrapped_u < 0, wrapped_u + L, wrapped_u)
    wrapped_v = jnp.where(u < 0, -wrapped_v, wrapped_v)  # Use original u for condition
    
    # Clamp v to strip width (soft boundary or reflect)
    # For now, just clamp to stay on strip
    wrapped_v = jnp.clip(wrapped_v, -W, W)
    
    return wrapped_u, wrapped_v


def apply_mobius_motion(state, config, params):
    """
    Apply Möbius strip walker dynamics in (u, v) space.
    
    Motion:
    - u increases by speed * dt (walking along the strip)
    - v remains constant (or can have small perturbation)
    
    Then apply Möbius wrapping rules.
    """
    speed = params.get('speed', 1.0)
    dt = config.dt
    
    pos = state.entity_pos
    vel = state.entity_vel
    active = state.entity_active
    
    # Current (u, v) coordinates (stored as x, y)
    u = pos[:, 0]
    v = pos[:, 1]
    
    # Update u based on velocity in u-direction
    # The walker moves along the strip with speed
    new_u = u + vel[:, 0] * dt
    new_v = v + vel[:, 1] * dt  # Usually 0, but allow v-drift
    
    # Apply Möbius boundary wrapping
    wrapped_u, wrapped_v = apply_mobius_wrapping(new_u, new_v, STRIP_LENGTH, STRIP_WIDTH)
    
    # Build new position array
    new_pos = jnp.stack([wrapped_u, wrapped_v], axis=1)
    
    # Only apply to active entities
    new_pos = jnp.where(active[:, None] > 0, new_pos, pos)
    
    # Velocity remains constant (drift along strip)
    # Could add v-inversion on wrap, but keeping simple for now
    
    return new_pos, vel


def run(config, state, steps=300):
    """Run Möbius strip walk simulation with topology-correct dynamics."""
    # Extract speed from initial velocity
    initial_speed = float(jnp.abs(state.entity_vel[0, 0]))
    params = {'speed': initial_speed if initial_speed > 0 else 1.0}
    
    for _ in range(steps):
        # Apply Möbius motion and wrapping BEFORE physics step
        new_pos, new_vel = apply_mobius_motion(state, config, params)
        state = state.replace(entity_pos=new_pos, entity_vel=new_vel)
        
        # Standard physics step (minimal effect since G=0)
        state = step_simulation(state, config)
    
    return state
