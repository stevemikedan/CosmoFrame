# CosmoSim
**A Polymorphic, Differentiable, JAX-Accelerated Universe Simulation Engine**

## Overview

CosmoSim is an extensible cosmological simulation engine built using a JAX-powered ECS architecture. It enables research, experimentation, and comparison of cosmological models across:

*   **Continuous vector physics** (N-body gravity)
*   **Multiple topologies** (Flat, 3-Torus, Mobius Strip, Sphere)
*   **Differentiable physics** and metrics for AI-driven optimization

CosmoSim is designed for developers, researchers, and agentic AI workflows.

---

## Quick Demo (3 commands)

```bash
# 1. Install
pip install -r requirements.txt

# 2. Generate simulation data
python cosmosim.py --scenario bulk_ring --steps 300 --export-json

# 3. Open the web viewer
python -m http.server 8000
# Then open http://localhost:8000/viewer/index.html and load the JSON from outputs/
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/stevemikedan/CosmoFrame.git
cd CosmoFrame

# Create virtual environment
python -m venv .venv

# Activate (pick your platform)
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows (cmd)
.venv\Scripts\Activate.ps1       # Windows (PowerShell)

# Install dependencies
pip install -r requirements.txt
```

---

## Running Simulations

The unified entry point for all simulations is `cosmosim.py`.

**Command syntax:**
```bash
python cosmosim.py --scenario <SCENARIO_NAME> [OPTIONS]
```

**Examples:**

```bash
# Interactive debug view (Matplotlib)
python cosmosim.py --scenario bulk_ring --view debug

# Headless run (fastest)
python cosmosim.py --scenario validation_2body_orbit --steps 200

# Custom parameters via PSS (Parameterized Scenario System)
python cosmosim.py --scenario random_nbody --params "N=500,radius=50.0" --view debug

# Use a preset configuration
python cosmosim.py --scenario random_nbody --preset large_cluster --steps 500

# Export to JSON for the web viewer
python cosmosim.py --scenario binary_star --steps 400 --export-json
```

**Key Flags:**
*   `--scenario`: Name of the scenario module (e.g., `random_nbody`, `bulk_ring`).
*   `--view`:
    *   `debug`: Opens the Matplotlib-based interactive viewer.
    *   `web`: Runs headless and prepares JSON output for the Web Viewer.
    *   `none`: Headless mode (fastest).
*   `--steps`: Number of simulation steps to run.
*   `--export-json`: Exports the simulation data to a JSON file.
*   `--params`: Comma-separated key=value parameter overrides.
*   `--preset`: Named preset from the scenario's SCENARIO_PRESETS.

### Available Scenarios

| Scenario | Description |
|---|---|
| `random_nbody` | N-body cluster with random particles |
| `random_nbody_3d` | 3D N-body variant |
| `bulk_ring` | Ring of orbiting bodies |
| `binary_star` | Two-body stellar system |
| `stellar_trio` | Three-body problem |
| `mini_solar` | Miniature solar system |
| `mobius_walk` | Mobius strip topology demo |
| `vortex_sheet` | Kelvin-Helmholtz instability |
| `bubble_collapse` | Bubble dynamics |
| `validation_2body_orbit` | Physics validation: 2-body orbit |
| `validation_sphere_geodesic` | Physics validation: sphere geodesic |
| `validation_torus_wrap` | Physics validation: torus wrapping |
| `validation_diagnostics` | Energy/momentum diagnostics |

---

## Visualization

CosmoSim offers two ways to visualize simulations:

### 1. Interactive Debugger (Matplotlib)
Best for quick validation and real-time interaction during development.

```bash
python cosmosim.py --scenario random_nbody --view debug
```
*   **Controls**: Mouse to pan/zoom. Keyboard shortcuts for overlays.

### 2. Web Viewer (Three.js)
High-fidelity, cinematic 3D visualization with topology overlays and 60fps playback.

**Workflow:**
1.  **Generate Data**: Run a simulation and export to JSON.
    ```bash
    python cosmosim.py --scenario bulk_ring --steps 500 --export-json
    ```
    *Output saved to `outputs/bulk_ring_500_steps_<timestamp>.json`*

2.  **Start Local Server**:
    ```bash
    python -m http.server 8000
    ```

3.  **Open Viewer**:
    *   Navigate to: [http://localhost:8000/viewer/index.html](http://localhost:8000/viewer/index.html)
    *   Click **"Load .json Simulation"**
    *   Select the generated JSON file from the `outputs/` directory.

---

## Core Features

### Differentiable Universe State (PyTree ECS)
All state is contained within a JAX PyTree, enabling JIT-accelerated physics, differentiable updates, and vectorized operations on static memory layouts.

### Polymorphic Topologies
The engine separates Metric Space from Physics Rules.
*   **Supported**: Flat (Euclidean), Torus (periodic), Sphere (Riemannian), Mobius.
*   **Planned**: Hyperbolic spaces, Organic manifolds.

### Physics Router
A strategy layer dynamically dispatches physics kernels based on the configuration.

### Parameterized Scenario System (PSS)
Every scenario declares a parameter schema and optional presets, enabling CLI-driven parameter sweeps without editing code.

---

## Project Structure

```text
CosmoSim/
├── cosmosim.py             # Main CLI entry point
├── kernel.py               # Core physics step & integrator
├── state.py                # UniverseState/Config data structures
├── topology.py             # Metric & topology definitions
├── entities.py             # Entity lifecycle (spawn/despawn)
│
├── physics/                # Physics computation modules
│   ├── integrator.py       #   Velocity Verlet integration
│   └── forces.py           #   Gravitational force calculation
│
├── environment/            # Environmental effects engine
│   ├── engine.py           #   Substrate + expansion + topology coordinator
│   ├── topology_math.py    #   Unified topology-aware distance math
│   └── expansion.py        #   Hubble expansion models
│
├── topologies/             # Topology implementations (flat, torus, sphere, ...)
├── scenarios/              # Simulation scenarios (bulk_ring, random_nbody, etc.)
├── viewer/                 # Web Viewer (Three.js) & Interactive Debugger
├── exporters/              # JSON frame export for web viewer
│
├── outputs/                # Simulation artifacts (JSON exports, plots)
├── tests/                  # Test suite (33 test files)
├── docs/                   # Documentation & architecture docs
│
└── requirements.txt        # Python dependencies
```

## Running Tests

```bash
# Run entire test suite
pytest

# Run fast tests only
pytest -q

# Run specific module tests
pytest tests/physics/
pytest tests/topology/
```
