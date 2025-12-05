"""
Diagnostics Validation Scenario

Demonstrates PS2.4 diagnostics system (energy, momentum, drift).
Developer scenario for PS2.4 validation.
"""
import jax
import jax.numpy as jnp
from state import UniverseConfig, UniverseState, initialize_state
from entities import spawn_entity
from kernel import step_simulation

DEVELOPER_SCENARIO = True

SCENARIO_PARAMS = {
    "N": {"type": "int", "default": 5, "min": 2, "max": 20},
    "G": {"type": "float", "default": 1.0, "min": 0.1, "max": 10.0},
    "dt": {"type": "float", "default": 0.01, "min": 0.001, "max": 0.1},
}

def build_config(params: dict | None = None) -> UniverseConfig:
    p = params or {}
    dt = p.get('dt', 0.01)
    G = p.get('G', 1.0)
    N = p.get('N', 5)
    
    return UniverseConfig(
        topology_type=0,  # Flat
        physics_mode=0,
        radius=50.0,
        max_entities=N,
        max_nodes=1,
        dt=dt,
        c=1.0,
        G=G,
        dim=2,
        enable_diagnostics=True
    )

def build_initial_state(config: UniverseConfig, params: dict | None = None) -> UniverseState:
    state = initialize_state(config)
    p = params or {}
    
    N = config.max_entities
    
    # Random cluster
    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)
    
    # Positions in a small cluster
    pos = jax.random.normal(k1, (N, 2)) * 5.0
    
    # Small random velocities
    vel = jax.random.normal(k2, (N, 2)) * 0.1
    
    # Uniform masses
    mass = jnp.ones(N)
    
    for i in range(N):
        state = spawn_entity(state, pos[i], vel[i], mass[i], 1)
    
    state.scenario_name = "diagnostics_demo"
    return state

def run(config, state, steps=300):
    """Run with detailed diagnostics output."""
    
    # Store initial energy for drift calculation
    initial_E = float(state.total_energy)
    
    print("="*60)
    print("PS2.4 DIAGNOSTICS VALIDATION")
    print("="*60)
    print(f"Particles: {config.max_entities}")
    print(f"Timestep: {config.dt}")
    print(f"Initial Energy: {initial_E:.6f}")
    print(f"Initial Momentum: [{float(state.momentum[0]):.4f}, {float(state.momentum[1]):.4f}]")
    print(f"Initial CoM: [{float(state.center_of_mass[0]):.4f}, {float(state.center_of_mass[1]):.4f}]")
    print("="*60)
    
    for i in range(steps):
        state = step_simulation(state, config)
        
        # Print detailed diagnostics every 50 steps
        if (i + 1) % 50 == 0:
            E_drift = (float(state.total_energy) - initial_E) / (abs(initial_E) + 1e-12)
            
            print(f"\nStep {i+1}:")
            print(f"  Kinetic:    {float(state.kinetic_energy):10.6f}")
            print(f"  Potential:  {float(state.potential_energy):10.6f}")
            print(f"  Total E:    {float(state.total_energy):10.6f}")
            print(f"  E Drift:    {E_drift:10.6%}")
            print(f"  Momentum:   [{float(state.momentum[0]):7.4f}, {float(state.momentum[1]):7.4f}]")
            print(f"  CoM:        [{float(state.center_of_mass[0]):7.4f}, {float(state.center_of_mass[1]):7.4f}]")
    
    final_E_drift = (float(state.total_energy) - initial_E) / (abs(initial_E) + 1e-12)
    
    print("\n" + "="*60)
    print("FINAL DIAGNOSTICS SUMMARY")
    print("="*60)
    print(f"Initial Energy:  {initial_E:.6f}")
    print(f"Final Energy:    {float(state.total_energy):.6f}")
    print(f"Energy Drift:    {final_E_drift:.6%}")
    print(f"Final Momentum:  [{float(state.momentum[0]):.4f}, {float(state.momentum[1]):.4f}]")
    print(f"Final CoM:       [{float(state.center_of_mass[0]):.4f}, {float(state.center_of_mass[1]):.4f}]")
    print("="*60)
    
    return state
