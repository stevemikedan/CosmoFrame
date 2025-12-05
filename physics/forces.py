"""
Force calculation module.

Computes forces acting on entities using topology-aware mathematics.
Integrates gravity and other interactions.
"""

import jax
import jax.numpy as jnp
from state import UniverseConfig, UniverseState
from environment.topology_math import compute_displacement, compute_distance

def compute_forces(state: UniverseState, config: UniverseConfig) -> jnp.ndarray:
    """
    Compute total acceleration on all entities.
    
    Includes:
    - Newtonian Gravity (topology-aware)
    - Softening (PS2.5 placeholder)
    
    Args:
        state: Current universe state
        config: Universe configuration
        
    Returns:
        Acceleration array (max_entities, dim)
    """
    pos = state.entity_pos
    mass = state.entity_mass
    active = state.entity_active
    
    # 1. Gravity
    # F_ij = G * m_i * m_j * disp_ij / (dist_ij^3 + eps)
    # acc_i = sum_j (G * m_j * disp_ij / ...)
    
    # We need pairwise displacements and distances
    # Shape: (N, N, dim) and (N, N)
    
    # Expand dims for broadcasting
    # p1: (N, 1, dim) - receiver
    # p2: (1, N, dim) - source
    p1 = pos[:, None, :]
    p2 = pos[None, :, :]
    
    # Compute topology-aware displacement (p2 - p1)
    # disp[i, j] is vector from i to j
    disp = compute_displacement(p1, p2, config.topology_type, config)
    
    # Compute distance
    # dist[i, j] is scalar distance between i and j
    dist = compute_distance(p1, p2, config.topology_type, config)
    
    # Softening (PS2.5 placeholder - currently 1e-12 or config value)
    eps = getattr(config, 'gravity_softening', 1e-12)
    
    # Compute 1/r^3 term (with softening)
    # Avoid division by zero for self-interaction (dist=0) by adding eps
    inv_dist3 = 1.0 / (dist**3 + eps**3)
    
    # Mask self-interactions and inactive entities
    # active_mask[j] is True if source j is active
    # We also mask i==j, though disp is 0 so force is 0 anyway.
    # But inv_dist3 might be large if eps is small.
    
    # Source mass (1, N)
    m_j = mass[None, :]
    
    # Active mask (1, N)
    active_j = active[None, :]
    
    # Force magnitude term: G * m_j * inv_dist3
    # Shape: (N, N)
    force_mag = config.G * m_j * inv_dist3
    
    # Apply mask: set force to 0 if source is inactive
    force_mag = jnp.where(active_j, force_mag, 0.0)
    
    # Calculate acceleration contributions
    # acc_i = sum_j (force_mag_ij * disp_ij)
    # disp shape: (N, N, dim)
    # force_mag shape: (N, N, 1)
    acc_contributions = force_mag[..., None] * disp
    
    # Sum over sources (axis 1)
    total_acc = jnp.sum(acc_contributions, axis=1)
    
    # Mask out inactive receivers (optional, but good for cleanliness)
    total_acc = jnp.where(active[:, None], total_acc, 0.0)
    
    return total_acc
