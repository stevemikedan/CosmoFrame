"""
Tests for topology-aware distance utilities (Phase PS2.1).

Verifies that distance and offset calculations correctly handle
different topologies: FLAT, TORUS, SPHERE, and BUBBLE.
"""

import pytest
import jax.numpy as jnp
import numpy as np
from state import UniverseConfig
from distance_utils import compute_offset, compute_distance


class TestDistanceUtils:
    """Test suite for topology-aware distance calculations."""
    
    @pytest.fixture
    def flat_config(self):
        """Configuration for flat (Euclidean) topology."""
        return UniverseConfig(
            physics_mode=0, radius=10.0, max_entities=100, max_nodes=10, 
            dt=0.1, c=1.0, G=1.0, topology_type=0
        )
        
    @pytest.fixture
    def torus_config(self):
        """Configuration for torus (periodic) topology."""
        return UniverseConfig(
            physics_mode=0, radius=5.0, max_entities=100, max_nodes=10, 
            dt=0.1, c=1.0, G=1.0, topology_type=1, torus_size=10.0
        )
        
    @pytest.fixture
    def sphere_config(self):
        """Configuration for spherical topology."""
        return UniverseConfig(
            physics_mode=0, radius=10.0, max_entities=100, max_nodes=10, 
            dt=0.1, c=1.0, G=1.0, topology_type=2
        )
        
    @pytest.fixture
    def bubble_config(self):
        """Configuration for bubble (curved) topology."""
        return UniverseConfig(
            physics_mode=0, radius=10.0, max_entities=100, max_nodes=10, 
            dt=0.1, c=1.0, G=1.0, topology_type=3, bubble_curvature=0.1
        )

    # ===================================================================
    # FLAT TOPOLOGY TESTS
    # ===================================================================
    
    def test_flat_offset(self, flat_config):
        """Verify flat topology offset matches Euclidean subtraction."""
        p1 = jnp.array([1.0, 2.0, 3.0])
        p2 = jnp.array([4.0, 6.0, 8.0])
        
        offset = compute_offset(p1, p2, flat_config)
        expected = p2 - p1
        
        assert jnp.allclose(offset, expected), \
            f"Flat offset incorrect: {offset} != {expected}"
    
    def test_flat_distance(self, flat_config):
        """Verify flat topology distance matches Euclidean norm."""
        p1 = jnp.array([0.0, 0.0, 0.0])
        p2 = jnp.array([3.0, 4.0, 0.0])
        
        dist = compute_distance(p1, p2, flat_config)
        expected = 5.0  # 3-4-5 triangle
        
        assert jnp.allclose(dist, expected), \
            f"Flat distance incorrect: {dist} != {expected}"

    # ===================================================================
    # TORUS TOPOLOGY TESTS
    # ===================================================================
    
    def test_torus_no_wrap(self, torus_config):
        """Verify torus does NOT wrap for separations < L/2."""
        # Box size L = 10, so L/2 = 5
        p1 = jnp.array([2.0, 2.0, 2.0])
        p2 = jnp.array([5.0, 5.0, 5.0])
        
        offset = compute_offset(p1, p2, torus_config)
        expected = jnp.array([3.0, 3.0, 3.0])  # No wrapping needed
        
        assert jnp.allclose(offset, expected), \
            f"Torus should not wrap: {offset} != {expected}"
    
    def test_torus_wrap(self, torus_config):
        """Verify torus DOES wrap for separations > L/2."""
        # Box size L = 10
        p1 = jnp.array([1.0, 1.0, 1.0])
        p2 = jnp.array([9.0, 9.0, 9.0])
        
        offset = compute_offset(p1, p2, torus_config)
        # Raw offset would be [8, 8, 8]
        # After wrapping: [8-10, 8-10, 8-10] = [-2, -2, -2]
        expected = jnp.array([-2.0, -2.0, -2.0])
        
        assert jnp.allclose(offset, expected), \
            f"Torus wrap incorrect: {offset} != {expected}"
    
    def test_torus_half_box(self, torus_config):
        """Verify torus boundary case at exactly L/2."""
        # Box size L = 10, test at L/2 = 5
        p1 = jnp.array([0.0, 0.0, 0.0])
        p2 = jnp.array([5.0, 0.0, 0.0])
        
        dist = compute_distance(p1, p2, torus_config)
        # At L/2, wrapping gives same result (5.0)
        assert jnp.allclose(dist, 5.0), \
            f"Torus distance at L/2 incorrect: {dist}"

    # ===================================================================
    # SPHERE TOPOLOGY TESTS
    # ===================================================================
    
    def test_sphere_small_angle(self, sphere_config):
        """Verify sphere distance approximates Euclidean for small angles."""
        R = sphere_config.radius
        
        # Two nearby points on sphere
        theta = 0.01  # Small angle in radians
        p1 = jnp.array([R, 0.0, 0.0])
        p2 = jnp.array([R * np.cos(theta), R * np.sin(theta), 0.0])
        
        dist = compute_distance(p1, p2, sphere_config)
        expected = R * theta  # Geodesic distance
        
        # For small angles, geodesic ≈ chord length with ~0.5% numerical error
        # Using relative tolerance of 0.1% to account for floating point precision
        assert jnp.allclose(dist, expected, rtol=1e-3), \
            f"Sphere small-angle distance incorrect: {dist} != {expected}"
    
    def test_sphere_antipodal(self, sphere_config):
        """Verify sphere distance for antipodal points is πR."""
        R = sphere_config.radius
        
        # Opposite points on sphere
        p1 = jnp.array([R, 0.0, 0.0])
        p2 = jnp.array([-R, 0.0, 0.0])
        
        dist = compute_distance(p1, p2, sphere_config)
        expected = np.pi * R
        
        assert jnp.allclose(dist, expected, rtol=1e-5), \
            f"Sphere antipodal distance incorrect: {dist} != {expected}"
    
    def test_sphere_offset_magnitude(self, sphere_config):
        """Verify sphere offset has correct magnitude (geodesic arc length)."""
        R = sphere_config.radius
        
        # Quarter-circle separation
        p1 = jnp.array([R, 0.0, 0.0])
        p2 = jnp.array([0.0, R, 0.0])
        
        offset = compute_offset(p1, p2, sphere_config)
        offset_mag = jnp.linalg.norm(offset)
        
        # Geodesic distance for 90° = πR/2
        expected_mag = (np.pi / 2) * R
        
        assert jnp.allclose(offset_mag, expected_mag, rtol=1e-5), \
            f"Sphere offset magnitude incorrect: {offset_mag} != {expected_mag}"

    # ===================================================================
    # BUBBLE TOPOLOGY TESTS
    # ===================================================================
    
    def test_bubble_flat_limit(self):
        """Verify bubble with k=0 matches Euclidean."""
        config = UniverseConfig(
            physics_mode=0, radius=10.0, max_entities=100, max_nodes=10, 
            dt=0.1, c=1.0, G=1.0, topology_type=3, bubble_curvature=0.0
        )
        
        p1 = jnp.array([0.0, 0.0, 0.0])
        p2 = jnp.array([3.0, 4.0, 0.0])
        
        dist = compute_distance(p1, p2, config)
        expected = 5.0  # Euclidean distance
        
        assert jnp.allclose(dist, expected), \
            f"Bubble k=0 should match flat: {dist} != {expected}"
    
    def test_bubble_curved(self, bubble_config):
        """Verify bubble with k>0 gives distance > Euclidean."""
        p1 = jnp.array([0.0, 0.0, 0.0])
        p2 = jnp.array([3.0, 4.0, 0.0])
        
        r_euclidean = 5.0
        dist = compute_distance(p1, p2, bubble_config)
        
        # With positive curvature, curved distance should be larger
        assert dist > r_euclidean, \
            f"Bubble curved distance should be > Euclidean: {dist} <= {r_euclidean}"
        
        # Verify formula: dist = r * (1 + k*r²/6)
        k = bubble_config.bubble_curvature
        expected = r_euclidean * (1.0 + (k * r_euclidean**2) / 6.0)
        
        assert jnp.allclose(dist, expected), \
            f"Bubble distance formula incorrect: {dist} != {expected}"
