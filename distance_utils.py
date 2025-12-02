"""
Topology-Aware Distance Utilities for CosmoSim

This module provides unified distance and offset calculations that correctly
handle different topologies:
- FLAT: Standard Euclidean geometry
- TORUS: Periodic boundaries with minimum image convention
- SPHERE: Great-circle geodesic distances
- BUBBLE: Curved radial metric

IMPORTANT: These functions NEVER modify input positions. They only compute
separation vectors and distances respecting the topology.
"""

import jax.numpy as jnp
import numpy as np


def compute_offset(pos_i, pos_j, config):
    """
    Compute separation vector from pos_i to pos_j respecting topology.
    
    CRITICAL: This function MUST NOT modify or normalize input positions.
    It only computes the offset vector for force direction calculations.
    
    Args:
        pos_i: Position of first particle (shape: [dim])
        pos_j: Position of second particle (shape: [dim])
        config: UniverseConfig containing topology information
        
    Returns:
        Offset vector Δx from pos_i to pos_j (shape: [dim])
    """
    topology_type = getattr(config, 'topology_type', 0)
    
    # FLAT topology: Standard Euclidean offset
    if topology_type == 0:
        return pos_j - pos_i
    
    # TORUS topology: Minimum image convention
    elif topology_type == 1:
        # Periodic box size
        L = getattr(config, 'torus_size', None)
        if L is None:
            L = config.radius * 2.0
        
        # Compute raw offset
        dx = pos_j - pos_i
        
        # Apply minimum image: wrap to [-L/2, L/2]
        dx = dx - jnp.round(dx / L) * L
        
        return dx
    
    # SPHERE topology: Tangent-plane offset along great circle
    elif topology_type == 2:
        R = config.radius
        
        # Normalize to unit sphere
        u = pos_i / R
        v = pos_j / R
        
        # Compute angle between points
        cosθ = jnp.clip(jnp.dot(u, v), -1.0, 1.0)
        θ = jnp.arccos(cosθ)
        
        # Small angle: use direct difference as fallback
        small_eps = 1e-8
        if θ < small_eps:
            return pos_j - pos_i
        
        # Compute great-circle axis and tangent direction
        axis = jnp.cross(u, v)
        axis_norm = jnp.linalg.norm(axis)
        
        # Degenerate case (antipodal or coincident): return any perpendicular
        if axis_norm < small_eps:
            # Find arbitrary perpendicular vector
            if jnp.abs(u[0]) < 0.9:
                perp = jnp.array([1.0, 0.0, 0.0])
            else:
                perp = jnp.array([0.0, 1.0, 0.0])
            tangent = perp - jnp.dot(perp, u) * u
            tangent = tangent / jnp.linalg.norm(tangent)
        else:
            # Tangent direction at pos_i pointing toward pos_j
            tangent = jnp.cross(axis / axis_norm, u)
            tangent = tangent / jnp.linalg.norm(tangent)
        
        # Offset is tangent direction scaled by arc distance
        offset = tangent * (R * θ)
        
        return offset
    
    # BUBBLE topology: Euclidean offset (curvature handled in distance)
    elif topology_type == 3:
        return pos_j - pos_i
    
    # Fallback: Euclidean
    else:
        return pos_j - pos_i


def compute_distance(pos_i, pos_j, config):
    """
    Compute scalar distance from pos_i to pos_j respecting topology.
    
    NOTE: For sphere and bubble, this does NOT simply use norm(offset).
    Each topology has its own distance metric.
    
    Args:
        pos_i: Position of first particle (shape: [dim])
        pos_j: Position of second particle (shape: [dim])
        config: UniverseConfig containing topology information
        
    Returns:
        Scalar distance (float)
    """
    topology_type = getattr(config, 'topology_type', 0)
    
    # FLAT topology: Euclidean distance
    if topology_type == 0:
        offset = pos_j - pos_i
        return jnp.linalg.norm(offset)
    
    # TORUS topology: Euclidean distance after minimum image
    elif topology_type == 1:
        offset = compute_offset(pos_i, pos_j, config)
        return jnp.linalg.norm(offset)
    
    # SPHERE topology: Geodesic distance (great-circle)
    elif topology_type == 2:
        R = config.radius
        
        # Normalize to unit sphere
        u = pos_i / R
        v = pos_j / R
        
        # Compute angle
        cosθ = jnp.clip(jnp.dot(u, v), -1.0, 1.0)
        θ = jnp.arccos(cosθ)
        
        # Geodesic distance
        dist = R * θ
        
        return dist
    
    # BUBBLE topology: Curved radial metric
    elif topology_type == 3:
        # Euclidean offset
        offset = pos_j - pos_i
        r = jnp.linalg.norm(offset)
        
        # Apply curvature correction
        k = getattr(config, 'bubble_curvature', 0.0)
        
        if k == 0.0:
            # Flat metric
            dist = r
        else:
            # First-order Taylor expansion: dist = r * (1 + k*r²/6)
            # This is stable and avoids numerical integration
            dist = r * (1.0 + (k * r * r) / 6.0)
        
        return dist
    
    # Fallback: Euclidean
    else:
        offset = pos_j - pos_i
        return jnp.linalg.norm(offset)
