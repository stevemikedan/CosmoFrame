"""
CLI Scenario Runner for CosmoSim.

Allows testing substrates, expansion, and topology from the command line.
"""

import argparse
import json
import time
import numpy as np
import jax.numpy as jnp
import jax

from state import UniverseConfig, UniverseState, initialize_state
from environment.engine import EnvironmentEngine
from entities import allocate_entities


def compute_forces(pos, vel, mass, active, config):
    """
    Compute gravitational forces.
    Mirrors logic from kernel.update_vector_physics.
    """
    # 1. Compute pairwise displacement: r_j - r_i
    # Shape: (N, N, dim)
    disp = pos[None, :, :] - pos[:, None, :]
    
    # 2. Compute distances
    dist_sq = jnp.sum(disp**2, axis=-1) + 1e-6
    dist = jnp.sqrt(dist_sq)
    
    # Mask mass of inactive entities
    active_mass = jnp.where(active, mass, 0.0)
    
    # 3. Compute gravitational force magnitudes
    # F = G * m1 * m2 / r^2
    force_mag = config.G * mass[:, None] * active_mass[None, :] / dist_sq
    
    # 4. Compute force vectors
    force_vec = disp * (force_mag / dist)[:, :, None]
    
    # Sum forces
    total_force = jnp.sum(force_vec, axis=1)
    
    return total_force


def update_step(pos, vel, force, mass, active, dt):
    """
    Integrate positions and velocities.
    Mirrors logic from kernel.update_vector_physics.
    """
    # Acceleration = Force / Mass
    acc = force / (mass[:, None] + 1e-6)
    
    # Semi-implicit Euler
    new_vel = vel + acc * dt
    new_pos = pos + new_vel * dt
    
    # Apply active mask
    active_mask = active[:, None]
    final_vel = jnp.where(active_mask, new_vel, vel)
    final_pos = jnp.where(active_mask, new_pos, pos)
    
    return final_pos, final_vel


def run_sim(config, steps=200, gravity=True, seed=42, save_every=1):
    """Run the simulation loop."""
    print(f"Initializing simulation with seed {seed}...")
    np.random.seed(seed)
    
    # Initialize environment
    env = EnvironmentEngine(config)
    
    # Initialize state (for compatibility)
    state = initialize_state(config)
    
    # Allocate and initialize entities
    # We use numpy for initialization to respect the seed easily
    N = config.max_entities
    dim = config.dim
    
    # Random positions within radius/2
    pos_np = np.random.uniform(-config.radius/2, config.radius/2, size=(N, dim))
    # Random velocities
    vel_np = np.random.uniform(-0.1, 0.1, size=(N, dim))
    # Random masses
    mass_np = np.random.uniform(0.1, 1.0, size=(N,))
    # All active
    active_np = np.ones((N,), dtype=bool)
    
    # Convert to JAX arrays
    pos = jnp.array(pos_np)
    vel = jnp.array(vel_np)
    mass = jnp.array(mass_np)
    active = jnp.array(active_np)
    
    frames = []
    
    print(f"Running {steps} steps...")
    
    for step in range(steps):
        # 1. Compute Forces
        if gravity:
            force = compute_forces(pos, vel, mass, active, config)
        else:
            force = jnp.zeros_like(pos)
            
        # 2. Apply Environment (Substrate, Expansion, Topology)
        # We pass 'state' but update it partially to be safe
        # EnvironmentEngine expects a UniverseState for substrate.update
        # We'll update the state object with current pos/vel
        current_state = state.replace(
            entity_pos=pos,
            entity_vel=vel,
            entity_mass=mass,
            entity_active=active,
            time=step * config.dt
        )
        
        pos, vel, force = env.apply_environment(pos, vel, force, current_state)
        
        # 3. Update Physics (Integration)
        pos, vel = update_step(pos, vel, force, mass, active, config.dt)
        
        # 4. Save Frame
        if step % save_every == 0:
            # Convert to list for JSON serialization
            frames.append({
                "step": step,
                "pos": np.array(pos).tolist(),
                "vel": np.array(vel).tolist()
            })
            
    return frames


def main():
    parser = argparse.ArgumentParser(description="CosmoSim Scenario Runner")
    
    # Simulation Config
    parser.add_argument("--steps", type=int, default=200, help="Number of steps")
    
    timestamp = int(time.time())
    default_name = f"sim_output/sim_{timestamp}.json"
    parser.add_argument("--outfile", type=str, default=default_name, help="Output JSON file")
    
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save-every", type=int, default=1, help="Save interval")
    parser.add_argument("--gravity", type=str, default="on", choices=["on", "off"], help="Enable gravity")
    
    # Environment Config
    parser.add_argument("--substrate", type=str, default="none", help="Substrate type")
    parser.add_argument("--topology", type=str, default="flat", help="Topology type")
    parser.add_argument("--expansion", type=str, default="none", help="Expansion type")
    
    # Advanced Config (optional overrides)
    parser.add_argument("--radius", type=float, default=10.0, help="Universe radius")
    parser.add_argument("--dt", type=float, default=0.1, help="Time step")
    parser.add_argument("--entities", type=int, default=50, help="Number of entities")
    
    args = parser.parse_args()
    
    # Map topology string to int type
    topology_map = {
        "flat": 0,
        "torus": 1,
        "sphere": 2,
        "bubble": 3
    }
    topo_type = topology_map.get(args.topology, 0)
    
    # Construct Config
    config = UniverseConfig(
        physics_mode=0,
        radius=args.radius,
        max_entities=args.entities,
        max_nodes=1,
        dt=args.dt,
        c=1.0,
        G=1.0,
        dim=3,
        topology_type=topo_type,
        
        # Expansion
        expansion_type=args.expansion,
        expansion_rate=0.05, # Default rate
        expansion_mode="inflation" if args.expansion == "scale_factor" else "linear",
        H=0.1, # Default H
        bubble_expand=True, # Default for bubble test
        bubble_radius=args.radius,
        
        # Substrate
        substrate=args.substrate,
        substrate_params={
            "grid_size": (10, 10, 10),
            "amplitude": 0.5,
            "noise": False
        }
    )
    
    # Run Simulation
    frames = run_sim(
        config, 
        steps=args.steps, 
        gravity=(args.gravity == "on"),
        seed=args.seed,
        save_every=args.save_every
    )
    
    # Save Output
    with open(args.outfile, "w") as f:
        json.dump({"frames": frames}, f)
        
    print(f"Saved {len(frames)} frames to {args.outfile}")


if __name__ == "__main__":
    main()
