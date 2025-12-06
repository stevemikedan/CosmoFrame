from state import UniverseConfig
from kernel import validate_topology_parameters
import pytest

def test_validation():
    print("Testing Topology Validation...")
    # Torus
    try:
        cfg = UniverseConfig(
            physics_mode=0, radius=10, max_entities=100, max_nodes=100, dt=0.1, c=1, G=1, 
            topology_type=1, torus_size=0
        )
        validate_topology_parameters(cfg)
        print("FAIL: Torus size 0 should raise ValueError")
    except ValueError as e:
        print(f"PASS: Torus caught {e}")
    
    # Sphere
    try:
        cfg = UniverseConfig(
            physics_mode=0, radius=0, max_entities=100, max_nodes=100, dt=0.1, c=1, G=1, 
            topology_type=2
        )
        validate_topology_parameters(cfg)
        print("FAIL: Sphere radius 0 should raise ValueError")
    except ValueError as e:
        print(f"PASS: Sphere caught {e}")
        
    # Bubble
    try:
        cfg = UniverseConfig(
            physics_mode=0, radius=10, max_entities=100, max_nodes=100, dt=0.1, c=1, G=1, 
            topology_type=3, bubble_radius=0
        )
        validate_topology_parameters(cfg)
        print("FAIL: Bubble radius 0 should raise ValueError")
    except ValueError as e:
        print(f"PASS: Bubble caught {e}")

if __name__ == "__main__":
    test_validation()
