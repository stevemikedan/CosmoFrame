"""
Tests for Topology-Aware Distance Utilities
"""

import pytest
import jax.numpy as jnp
import numpy as np
from distance_utils import compute_offset, compute_distance
from state import UniverseConfig

class TestDistanceUtils:
    
    @pytest.fixture
    def flat_config(self):
        return UniverseConfig(
            physics_mode=0, radius=10.0, max_entities=100, max_nodes=10, dt=0.1, c=1.0, G=1.0,
            topology_type=0
        )
        
    @pytest.fixture
    def torus_config(self):
        return UniverseConfig(
            physics_mode=0, radius=5.0, max_entities=100, max_nodes=10, dt=0.1, c=1.0, G=1.0,
            topology_type=1, torus_size=10.0
        )
        
    @pytest.fixture
    def sphere_config(self):
        return UniverseConfig(
            physics_mode=0, radius=10.0, max_entities=100, max_nodes=10, dt=0.1, c=1.0, G=1.0,
            topology_type=2
        )
        
    @pytest.fixture
    def bubble_config(self):
        return UniverseConfig(
            physics_mode=0, radius=10.0, max_entities=100, max_nodes=10, dt=0.1, c=1.0, G=1.0,
            topology_type=3, bubble_curvature=0.1
        )

    # --- FLAT TOPOLOGY TESTS ---
    
    def test_flat_offset(self, flat_config):
        p1 = jnp.array([1.0, 2.0])
        p2 = jnp.array([4.0, 6.0])
        
        offset = compute_offset(p1, p2, flat_config)
        expected = p2 - p1
        
        assert jnp.allclose(offset, expected)
        
    def test_flat_distance(self, flat_config):
        p1 = jnp.array([0.0, 0.0])
        p2 = jnp.array([3.0, 4.0])
        
        dist = compute_distance(p1, p2, flat_config)
        assert jnp.isclose(dist, 5.0)

    # --- TORUS TOPOLOGY TESTS ---
    
    def test_torus_no_wrap(self, torus_config):
        # L=10, points separated by 2 (small separation)
        p1 = jnp.array([1.0, 1.0])
        p2 = jnp.array([3.0, 1.0])
        
        offset = compute_offset(p1, p2, torus_config)
        assert jnp.allclose(offset, jnp.array([2.0, 0.0]))
        
        dist = compute_distance(p1, p2, torus_config)
        assert jnp.isclose(dist, 2.0)
        
    def test_torus_wrap(self, torus_config):
        # L=10, points at 1 and 9. Separation is 8, but wrapped is -2
        p1 = jnp.array([1.0, 0.0])
        p2 = jnp.array([9.0, 0.0])
        
        offset = compute_offset(p1, p2, torus_config)
        # Should wrap: 9-1 = 8. 8 - round(8/10)*10 = 8 - 10 = -2
        assert jnp.allclose(offset, jnp.array([-2.0, 0.0]))
        
        dist = compute_distance(p1, p2, torus_config)
        assert jnp.isclose(dist, 2.0)
        
    def test_torus_half_box(self, torus_config):
        # L=10, separation exactly 5. Boundary case.
        p1 = jnp.array([0.0, 0.0])
        p2 = jnp.array([5.0, 0.0])
        
        offset = compute_offset(p1, p2, torus_config)
        # 5 - round(0.5)*10 = 5 - 0 = 5 (numpy round to nearest even is tricky, but here likely 0 or 1)
        # jnp.round(0.5) is usually 0.0 (round half to even)
        # Let's check behavior. If it doesn't wrap, it's 5.
        # If it wraps, it could be -5. Both have dist 5.
        
        dist = compute_distance(p1, p2, torus_config)
        assert jnp.isclose(dist, 5.0)

    # --- SPHERE TOPOLOGY TESTS ---
    
    def test_sphere_small_angle(self, sphere_config):
        # R=10. Points close on equator.
        # p1 at (10, 0, 0)
        # p2 slightly rotated around Z
        angle = 0.1
        p1 = jnp.array([10.0, 0.0, 0.0])
        p2 = jnp.array([10.0 * np.cos(angle), 10.0 * np.sin(angle), 0.0])
        
        dist = compute_distance(p1, p2, sphere_config)
        expected_dist = 10.0 * angle
        assert jnp.isclose(dist, expected_dist, atol=1e-5)
        
        # Offset should be tangent
        offset = compute_offset(p1, p2, sphere_config)
        # Tangent at p1 (1,0,0) pointing to p2 is along Y axis
        expected_dir = jnp.array([0.0, 1.0, 0.0])
        normalized_offset = offset / jnp.linalg.norm(offset)
        
        assert jnp.allclose(normalized_offset, expected_dir, atol=1e-2)
        assert jnp.isclose(jnp.linalg.norm(offset), expected_dist)

    def test_sphere_antipodal(self, sphere_config):
        # R=10. Antipodal points.
        p1 = jnp.array([10.0, 0.0, 0.0])
        p2 = jnp.array([-10.0, 0.0, 0.0])
        
        dist = compute_distance(p1, p2, sphere_config)
        expected = np.pi * 10.0
        assert jnp.isclose(dist, expected)
        
    def test_sphere_offset_magnitude(self, sphere_config):
        # Offset magnitude should equal geodesic distance
        p1 = jnp.array([10.0, 0.0, 0.0])
        p2 = jnp.array([0.0, 10.0, 0.0]) # 90 degrees away
        
        offset = compute_offset(p1, p2, sphere_config)
        dist = compute_distance(p1, p2, sphere_config)
        
        assert jnp.isclose(jnp.linalg.norm(offset), dist)
        assert jnp.isclose(dist, 10.0 * (np.pi / 2))

    # --- BUBBLE TOPOLOGY TESTS ---
    
    def test_bubble_flat_limit(self):
        # k=0 should be Euclidean
        config = UniverseConfig(
            physics_mode=0, radius=10.0, max_entities=100, max_nodes=10, dt=0.1, c=1.0, G=1.0,
            topology_type=3, bubble_curvature=0.0
        )
        p1 = jnp.array([0.0, 0.0])
        p2 = jnp.array([3.0, 4.0])
        
        dist = compute_distance(p1, p2, config)
        assert jnp.isclose(dist, 5.0)
        
    def test_bubble_curved(self, bubble_config):
        # k=0.1
        p1 = jnp.array([0.0, 0.0])
        p2 = jnp.array([3.0, 4.0]) # r=5
        
        # dist = r * (1 + k*r^2/6)
        # dist = 5 * (1 + 0.1*25/6) = 5 * (1 + 2.5/6) = 5 * (1 + 0.4166) = 7.0833
        
        dist = compute_distance(p1, p2, bubble_config)
        expected = 5.0 * (1.0 + (0.1 * 25.0) / 6.0)
        
        assert jnp.isclose(dist, expected)
        assert dist > 5.0 # Distance should be larger than Euclidean
