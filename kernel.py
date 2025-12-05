"""Physics kernel implementations for CosmoSim."""

import jax
import jax.numpy as jnp
from state import UniverseConfig, UniverseState
from entities import spawn_entity, despawn_entity
from topology import enforce_boundaries, apply_topology
from physics.integrator import velocity_verlet
from physics.forces import compute_forces

def compute_diagnostics(state: UniverseState, config: UniverseConfig) -> UniverseState:
    """
    Compute and update diagnostics fields in the state.
    
    Calculates:
    - Kinetic Energy: 0.5 * m * v^2
    - Potential Energy: -0.5 * sum(G * m_i * m_j / r_ij)
    - Total Energy: KE + PE
    - Momentum: sum(m * v)
    - Center of Mass: sum(m * r) / sum(m)
    - Energy Drift: (E - E0) / E0 (requires storing E0, for now just store E)
    """
    pos = state.entity_pos
    vel = state.entity_vel
    mass = state.entity_mass
    active = state.entity_active
    
    # Mask inactive entities
    # active is boolean, cast to float for multiplication
    active_f = active.astype(jnp.float32)
    mass_active = mass * active_f
    
    # 1. Kinetic Energy
    # KE = 0.5 * sum(m * |v|^2)
    v_sq = jnp.sum(vel**2, axis=-1)
    ke = 0.5 * jnp.sum(mass_active * v_sq)
    
    # 2. Potential Energy
    # PE = -0.5 * sum_{i!=j} (G * m_i * m_j / r_ij)
    # We need pairwise distances again.
    # Note: This duplicates distance calc from forces. 
    # In a highly optimized engine, we'd return PE from compute_forces or share the dist matrix.
    # For now, recomputing is cleaner for modularity.
    from environment.topology_math import compute_distance
    
    p1 = pos[:, None, :]
    p2 = pos[None, :, :]
    dist = compute_distance(p1, p2, config.topology_type, config)
    
    # Avoid division by zero
    eps = getattr(config, 'gravity_softening', 1e-12)
    inv_dist = 1.0 / (dist + eps)
    
    # Pairwise potential
    # U_ij = -G * m_i * m_j / r_ij
    m_i = mass_active[:, None]
    m_j = mass_active[None, :]
    
    pot_matrix = -config.G * m_i * m_j * inv_dist
    
    # Mask self-interactions (diagonal)
    # dist is 0 on diagonal, but eps handles singularity.
    # However, self-potential should be 0.
    # We can mask diagonal.
    eye_mask = jnp.eye(config.max_entities, dtype=bool)
    pot_matrix = jnp.where(eye_mask, 0.0, pot_matrix)
    
    # Sum and divide by 2 (double counting)
    pe = 0.5 * jnp.sum(pot_matrix)
    
    # 3. Total Energy
    total_e = ke + pe
    
    # 4. Momentum
    # P = sum(m * v)
    mom = jnp.sum(mass_active[:, None] * vel, axis=0)
    
    # 5. Center of Mass
    # R_cm = sum(m * r) / sum(m)
    total_mass = jnp.sum(mass_active) + 1e-12
    com = jnp.sum(mass_active[:, None] * pos, axis=0) / total_mass
    
    # 6. Energy Drift
    # Ideally we compare to initial energy.
    # Storing initial energy in state is tricky if state is immutable/stateless between runs.
    # For now, we just store 0.0 or calculate relative change if we had E0.
    # We'll leave drift as 0.0 for this step, or maybe user wants step-to-step drift?
    # Usually drift is (E(t) - E(0)) / E(0).
    # Without E(0) in state, we can't compute cumulative drift.
    # We'll just store 0.0 for now, or maybe we can repurpose a field?
    # Let's just store 0.0.
    drift = 0.0
    
    return state.replace(
        kinetic_energy=jnp.array(ke),
        potential_energy=jnp.array(pe),
        total_energy=jnp.array(total_e),
        energy_drift=jnp.array(drift),
        momentum=mom,
        center_of_mass=com,
        dt_actual=jnp.array(config.dt)
    )

def update_vector_physics(state: UniverseState, config: UniverseConfig) -> UniverseState:
    """Update physics for VECTOR mode using Velocity Verlet."""
    # Integrate
    state = velocity_verlet(state, config, compute_forces)
    
    # Apply topology boundaries (wrapping)
    # Note: velocity_verlet updates pos, but doesn't wrap.
    # We wrap explicitly.
    # However, integrator uses force_fn which uses topology-aware displacement.
    # So particles can go arbitrarily far in "unwrapped" space during the step?
    # No, we should wrap after integration to keep coordinates bounded.
    
    # Use legacy enforce_boundaries or new topology logic?
    # PS2.4 says "Replace legacy displacement logic with topology_math functions."
    # But for wrapping, we still need a wrap function.
    # topology_math.py doesn't have wrap_position (it was in Topology classes).
    # We should probably use `topology.apply_topology` or `enforce_boundaries`.
    # Let's stick to `apply_topology` from `topology.py` which calls `enforce_boundaries`.
    # Wait, `topology.py` imports `enforce_boundaries`.
    
    new_pos = state.entity_pos
    new_vel = state.entity_vel
    
    final_pos, final_vel = apply_topology(new_pos, new_vel, config)
    
    state = state.replace(entity_pos=final_pos, entity_vel=final_vel)
    
    return state

def update_lattice_physics(state: UniverseState, config: UniverseConfig) -> UniverseState:
    """Update physics for LATTICE mode (no-op placeholder)."""
    return state

def dispatch_physics(state: UniverseState, config: UniverseConfig) -> UniverseState:
    """Dispatch to the appropriate physics kernel based on physics_mode."""
    def vector_branch(state):
        return update_vector_physics(state, config)

    def lattice_branch(state):
        return update_lattice_physics(state, config)

    def reserved_branch(state):
        return state

    branches = [
        vector_branch,   # mode 0: VECTOR
        lattice_branch,  # mode 1: LATTICE
        reserved_branch, # mode 2+: RESERVED
    ]

    safe_mode = jnp.clip(config.physics_mode, 0, 2)
    return jax.lax.switch(safe_mode, branches, state)

def step_simulation(state: UniverseState, config: UniverseConfig) -> UniverseState:
    """Execute one simulation timestep."""
    # 1. Update global time
    state = state.replace(time=state.time + config.dt)

    # 2. Apply physics update (Integration + Wrapping)
    state = dispatch_physics(state, config)

    # 3. Compute Diagnostics (Post-step)
    if config.enable_diagnostics:
        state = compute_diagnostics(state, config)

    # 4. Return updated state
    return state
