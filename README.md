# CosmoSim
**A Polymorphic, Differentiable, JAX-Accelerated Universe Simulation Engine**

🌌 **Overview**

CosmoSim is an extensible cosmological simulation engine built using a JAX-powered ECS architecture. It enables research, experimentation, and comparison of cosmological models across:

*   **Continuous vector physics** (N-body gravity)
*   **Multiple topologies** (Flat, 3-Torus, Mobius Strip, Sphere)
*   **Differentiable physics** and metrics for AI-driven optimization

CosmoSim is designed for developers, researchers, and agentic AI workflows.

---

## 🚀 Running the App

The unified entry point for all simulations is `cosmosim.py`.

### 1. Installation

```powershell
# Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install jax matplotlib pytest
```

### 2. Run a Simulation

**Command syntax:**
```powershell
python cosmosim.py --scenario <SCENARIO_NAME> [OPTIONS]
```

**Common Examples:**

```powershell
# Run a specific scenario with interactive debug view
python cosmosim.py --scenario bulk_ring --view debug

# Run a validation scenario headless (fast)
python cosmosim.py --scenario validation_2body_orbit --steps 200

# Run with custom parameters (Scenario PSS)
python cosmosim.py --scenario random_nbody --params "N=500,radius=50.0" --view debug
```

**Key Flags:**
*   `--scenario`: Name of the scenario module (e.g., `random_nbody`, `bulk_ring`).
*   `--view`:
    *   `debug`: Opens the Matplotlib-based interactive viewer.
    *   `web`: Runs headless and prepares JSON output for the Web Viewer.
    *   `none`: Headless mode (fastest).
*   `--steps`: Number of simulation steps to run.
*   `--export-json`: Exports the simulation data to a JSON file.

---

## 🎨 Visualization

CosmoSim offers two primary ways to visualize simulations:

### 1. Interactive Debugger (Matplotlib)
Best for quick validation and real-time interaction during development.

```powershell
python cosmosim.py --scenario random_nbody --view debug
```
*   **Controls**: Mouse to pan/zoom. Window closes automatically at end of steps unless interactive mode is held.

### 2. Web Viewer (Three.js)
High-fidelity, cinematic visualization with topology overlays and smooth 60fps playback.

**Workflow:**
1.  **Generate Data**: Run a simulation and export to JSON.
    ```powershell
    python cosmosim.py --scenario bulk_ring --steps 500 --export-json
    ```
    *Output will be saved to `outputs/bulk_ring_500_steps_<timestamp>.json`*

2.  **Start Local Server**:
    ```powershell
    python -m http.server 8000
    ```

3.  **Open Viewer**:
    *   Navigate to: [http://localhost:8000/viewer/test.html](http://localhost:8000/viewer/test.html)
    *   Click **"Load .json Simulation"**
    *   Select the generated JSON file from the `outputs/` directory.

---

## 🧠 Core Features

### 1. Differentiable Universe State (PyTree ECS)
All state is contained within a JAX PyTree, enabling JIT-accelerated physics, differentiable updates, and vectorized operations on static memory layouts.

### 2. Polymorphic Topologies
The engine separates Metric Space from Physics Rules.
*   **Supported**: Flat (Euclidean), Torus (periodic), Sphere (Riemannian), Mobius.
*   **Planned**: Hyperbolic spaces, Organic manifolds.

### 3. Physics Router
A strategy layer dynamically dispatches physics kernels based on the configuration.

---

## 📂 Project Structure

```text
CosmoSim/
├── cosmosim.py             # Main CLI entry point
├── kernel.py               # Core physics step & integrator
├── state.py                # UniverseState data structure
├── topology.py             # Metric & topology definitions
│
├── scenarios/              # Simulation scenarios (bulk_ring, random_nbody, etc.)
├── topologies/             # Topology implementations
├── viewer/                 # Web Viewer (Three.js) & Interactive Debugger source
│
├── outputs/                # Simulation artifacts (JSON exports, plots)
│
├── tests/                  # Test suite
└── docs/                   # Documentation
```

## 🧪 Running Tests

```powershell
# Run entire test suite
pytest

# Run fast tests only
pytest -q
```