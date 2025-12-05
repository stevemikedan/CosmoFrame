"""
Topology mathematics module.

Provides unified displacement and distance calculations for all supported topologies:
- Flat (Euclidean)
- Torus (Nearest Image)
- Sphere (Geodesic Tangent-Plane)
- Bubble (Curved Tangent-Space Approximation)

All functions are JAX-compatible and support batching.
"""

import jax
import jax.numpy as jnp
from state import UniverseConfig

# Topology Constants (must match factory/state definitions)
TOPOLOGY_FLAT = 0
TOPOLOGY_TORUS = 1
TOPOLOGY_SPHERE = 2
TOPOLOGY_BUBBLE = 3
TOPOLOGY_MOBIUS = 5  # Added for completeness based on recent work


def compute_displacement(p1: jnp.ndarray, p2: jnp.ndarray, topology_type: int, config: UniverseConfig) -> jnp.ndarray:
    """
    Compute displacement vector from p1 to p2 (p2 - p1) respecting topology.
    
    Guarantees anti-symmetry: disp(p1, p2) == -disp(p2, p1).
    
    Args:
        p1: Source position(s) (..., dim)
        p2: Target position(s) (..., dim)
        topology_type: Integer topology ID
        config: Universe configuration
        
    Returns:
        Displacement vector(s) (..., dim)
    """
    # Helper functions for each topology (capture config from closure)
    def _flat(p1, p2):
        return p2 - p1
        
    def _torus(p1, p2):
        raw = p2 - p1
        # Use config.radius * 2 as domain size L if torus_size not set
        # (Assuming centered at 0, from -R to R)
        # Use getattr with explicit fallback calculation
        torus_size = getattr(config, 'torus_size', None)
        # If torus_size is None or not set, compute from radius
        # Use jnp.where to handle this in a JAX-friendly way
        L = torus_size if torus_size is not None else (config.radius * 2.0)
            
        wrapped = raw - L * jnp.round(raw / L)
        return wrapped
        
    def _sphere(p1, p2):
        R = config.radius
        # Normalize to unit sphere
        u = p1 / (R + 1e-12)
        v = p2 / (R + 1e-12)
        
        dot_uv = jnp.sum(u * v, axis=-1, keepdims=True)
        dot_uv = jnp.clip(dot_uv, -1.0, 1.0)
        
        theta = jnp.arccos(dot_uv)
        
        # Tangent-plane projection
        # Project v into p1's tangent space
        proj = v - u * dot_uv
        proj_norm = jnp.linalg.norm(proj, axis=-1, keepdims=True)
        
        # Direction
        direction = proj / (proj_norm + 1e-12)
        
        # Geodesic displacement
        disp = direction * (R * theta)
        
        # Small-angle fallback (if theta is very small, use linear chord)
        # Also handles p1 == p2 case where proj_norm is 0
        is_small = theta < 1e-6
        return jnp.where(is_small, p2 - p1, disp)
        
    def _bubble(p1, p2):
        delta = p2 - p1
        r = jnp.linalg.norm(p1, axis=-1, keepdims=True)
        
        # Tangent space projection (radial correction)
        # tangent = delta - (dot(delta, p1) / r^2) * p1
        dot_dp = jnp.sum(delta * p1, axis=-1, keepdims=True)
        tangent = delta - (dot_dp / (r*r + 1e-12)) * p1
        
        k = getattr(config, 'bubble_curvature', 0.2)
        disp = tangent * (1.0 + k * r * r)
        return disp

    def _mobius(p1, p2):
        # Mobius: fallback to flat (Mobius logic handled in legacy MobiusTopology class)
        return p2 - p1

    # Dispatch using jax.lax.switch (only pass JAX arrays, not config)
    pred = jnp.clip(topology_type, 0, 5)
    
    return jax.lax.switch(
        pred,
        [
            _flat,    # 0
            _torus,   # 1
            _sphere,  # 2
            _bubble,  # 3
            _flat,    # 4 (Hyperbolic - fallback to flat)
            _mobius,  # 5
        ],
        p1, p2
    )


def compute_distance(p1: jnp.ndarray, p2: jnp.ndarray, topology_type: int, config: UniverseConfig) -> jnp.ndarray:
    """
    Compute scalar distance between p1 and p2.
    
    Args:
        p1: Source position(s)
        p2: Target position(s)
        topology_type: Integer topology ID
        config: Universe configuration
        
    Returns:
        Distance scalar(s)
    """
    disp = compute_displacement(p1, p2, topology_type, config)
    return jnp.linalg.norm(disp, axis=-1)
