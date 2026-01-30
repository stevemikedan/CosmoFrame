# CosmoFrame Architecture Review

**Date:** 2026-01-30
**Scope:** Full codebase review — architecture, dependencies, code quality, demo readiness

---

## Table of Contents

1. [Codebase Overview](#1-codebase-overview)
2. [Module Dependency Map](#2-module-dependency-map)
3. [Critical Architecture Findings](#3-critical-architecture-findings)
4. [Duplicate & Dead Code](#4-duplicate--dead-code)
5. [Demo Readiness Assessment](#5-demo-readiness-assessment)
6. [Prioritized Improvement Roadmap](#6-prioritized-improvement-roadmap)

---

## 1. Codebase Overview

### Technology Stack
- **Python** (simulation engine): JAX, Chex, Matplotlib, NumPy
- **JavaScript** (web viewer): Three.js, ES modules
- **No build tools**: No bundler, no package manager for JS, no `requirements.txt`

### File Statistics
| Category | Files | Lines (approx) |
|---|---|---|
| Core Python (root) | 8 | ~2,400 |
| Physics modules | 2 | 200 |
| Environment | 6 | 400 |
| Topologies | 8 | 500 |
| Scenarios | 17 | 1,200 |
| Viewer (Python) | 8 | 800 |
| Viewer (JS — modular) | 3 | 600 |
| Viewer (JS — legacy monolithic) | 5 | 34,000+ |
| Tests | 33 | ~3,000 |
| Docs | 45 | ~4,000 |

### Architecture Style
ECS (Entity-Component-System) using JAX PyTrees for state. Physics is
functional — `step_simulation(state, config) -> state`. Scenarios implement
a standard interface: `build_config()`, `build_initial_state()`, `run()`.

---

## 2. Module Dependency Map

### Core Simulation Pipeline

```
                        ┌──────────────┐
                        │  cosmosim.py │  CLI entrypoint
                        │  (789 lines) │
                        └──────┬───────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                 │
              v                v                 v
     ┌────────────┐   ┌──────────────┐   ┌──────────────┐
     │ scenarios/  │   │ viewer/      │   │ exporters/   │
     │ *.py        │   │ viewer.py    │   │ json_export  │
     └─────┬──────┘   └──────────────┘   └──────┬───────┘
           │                                      │
           v                                      v
     ┌──────────┐                          ┌──────────┐
     │ state.py │◄─────────────────────────│kernel.py │
     │ Config + │                          │ step_sim │
     │ State    │                          └────┬─────┘
     └──────────┘                               │
                                   ┌────────────┼────────────┐
                                   │            │            │
                                   v            v            v
                           ┌────────────┐ ┌──────────┐ ┌──────────────┐
                           │ physics/   │ │topology  │ │ entities.py  │
                           │forces.py   │ │.py       │ │ spawn/desp.  │
                           │integrator  │ └────┬─────┘ └──────────────┘
                           └─────┬──────┘      │
                                 │             uses
                                 v              │
                        ┌─────────────────┐     │
                        │ environment/    │◄────┘
                        │ topology_math   │
                        │ engine.py       │
                        │ expansion.py    │
                        └────────┬────────┘
                                 │
                                 v
                        ┌─────────────────┐
                        │ topologies/     │
                        │ flat, torus,    │
                        │ sphere, bubble, │
                        │ mobius, factory │
                        └─────────────────┘
```

### Web Viewer Architecture

```
     viewer/index.html
           │
           ├── js/viewer.js ──────────► Three.js scene, camera, renderer
           │       │
           │       ├── js/trails.js ──► Particle trail rendering (ShaderMaterial)
           │       │
           │       └── autoCamera.js ─► Camera framing logic
           │
           └── js/ui.js ──────────────► All UI bindings, keyboard, HUD
                   │
                   ├── frameLoader.js ─► File loading (fetch, directory picker)
                   ├── player.js ──────► Frame playback + instanced mesh
                   ├── visualModes.js ─► Color mapping modes
                   ├── scenarioConfig ─► Scenario setup panel
                   └── topologyOverlay ► Topology wireframe rendering
```

### Distance/Topology Calculation Dependencies (THE PROBLEM)

```
  ┌─────────────────────────────────────────────────────────────────┐
  │                THREE SEPARATE IMPLEMENTATIONS                   │
  │                                                                 │
  │  1. topology.py              2. environment/          3. distance_utils.py  │
  │     compute_distance()          topology_math.py         compute_distance() │
  │     enforce_boundaries()        compute_displacement()   compute_offset()   │
  │     apply_topology()            compute_distance()                          │
  │                                                                 │
  │  Used by:                    Used by:                 Used by:             │
  │   kernel.py                   physics/forces.py        physics_utils.py    │
  │   (apply_topology)            kernel.py                (legacy code)       │
  │                               (compute_diagnostics)                        │
  └─────────────────────────────────────────────────────────────────┘

  AND ALSO:

  4. topologies/*.py (OOP implementations via factory)
     Used by: environment/engine.py
     NOT used by: kernel.py, physics/forces.py
```

### Scenario Interface Contract

```
  Every scenario module MUST export:

  SCENARIO_PARAMS: dict          # PSS parameter schema
  SCENARIO_PRESETS: dict         # Named parameter presets
  build_config(params) -> UniverseConfig
  build_initial_state(config, params) -> UniverseState
  run(config, state, steps) -> UniverseState
```

---

## 3. Critical Architecture Findings

### CRITICAL-1: Three Duplicate Distance/Topology Systems

There are **three independent implementations** of distance and displacement
calculations that coexist in the codebase:

| File | Functions | Used By |
|---|---|---|
| `topology.py` | `compute_distance()`, `enforce_boundaries()`, `apply_topology()` | `kernel.py` (boundary enforcement) |
| `environment/topology_math.py` | `compute_displacement()`, `compute_distance()` | `physics/forces.py`, `kernel.py` (diagnostics) |
| `distance_utils.py` | `compute_offset()`, `compute_distance()` | `physics_utils.py` (legacy) |

Each has **different signatures**, **different topology support**, and
**different numerical approaches**:

- `topology.py:compute_distance(p1, p2, topology_type, radius)` — takes bare `radius` float
- `topology_math.py:compute_distance(p1, p2, topology_type, config)` — takes full `config` object
- `distance_utils.py:compute_distance(pos_i, pos_j, config)` — no topology_type arg, reads from config

**Risk:** Inconsistent physics behavior between force calculation
(topology_math), boundary enforcement (topology.py), and any code using
the legacy path (distance_utils). A bug fix in one is not reflected in the others.

**Recommendation:** Consolidate into a single `topology_math.py` as the
canonical source. Delete `distance_utils.py`. Refactor `topology.py` to
delegate to `topology_math.py`.

---

### CRITICAL-2: EnvironmentEngine Is Disconnected

`environment/engine.py` defines `EnvironmentEngine` which orchestrates
substrates, expansion, and topology — but the main simulation loop in
`kernel.py:step_simulation()` **never instantiates or calls it**.

```python
# kernel.py step_simulation() does:
state = velocity_verlet(state, config, compute_forces)     # forces
final_pos, final_vel = apply_topology(new_pos, new_vel, config)  # topology.py
state = compute_diagnostics(state, config)                  # diagnostics

# What it SHOULD also do (but doesn't):
# env_engine.apply_environment(pos, vel, force, state)     # substrates + expansion
```

**Impact:** Substrate fields (vector fields, fluids) and expansion models
(Hubble flow) configured in `UniverseConfig` are **silently ignored** during
simulation. The entire `environment/`, `substrate/`, and expansion system is
dead code in practice.

**Recommendation:** Either integrate `EnvironmentEngine` into `kernel.py`'s
step loop, or document that these features are not yet connected.

---

### CRITICAL-3: Inconsistent Scenario Interface

`build_config()` has inconsistent signatures across scenarios:

```python
# random_nbody.py — accepts params
def build_config(params: dict | None = None) -> UniverseConfig:

# bulk_ring.py — NO params argument
def build_config() -> UniverseConfig:
```

The CLI (`cosmosim.py:run_scenario`) always calls `module.build_config(merged_params)`,
which means **`bulk_ring` will break** when called through the CLI with any
parameters because `build_config()` doesn't accept arguments.

The same inconsistency exists for `build_initial_state()` — the CLI uses
`inspect.signature()` to detect if `params` is supported, but only for
`build_initial_state`, not for `build_config`.

**Recommendation:** Standardize all scenarios to accept `params` dict in both
`build_config(params)` and `build_initial_state(config, params)`.

---

### HIGH-1: Legacy Monolithic JS Files Alongside Modular Refactor

The `viewer/` directory contains two parallel implementations:

**Modular (clean):**
```
viewer/js/viewer.js    (154 lines)
viewer/js/ui.js        (437 lines)
viewer/js/trails.js    (167 lines)
viewer/index.html      (150 lines)
```

**Legacy monolithic:**
```
viewer/player.js       (7,243 lines)
viewer/scenarioConfig.js (11,254 lines)
viewer/autoCamera.js   (5,516 lines)
viewer/visualModes.js  (5,683 lines)
viewer/topologyOverlay.js (7,025 lines)
viewer/test.html       (22,498 lines)
viewer/test.js         (3,095 lines)
viewer/frameLoader.js  (3,429 lines)
```

The modular `js/ui.js` **imports from the legacy files** (`player.js`,
`frameLoader.js`, `visualModes.js`, `scenarioConfig.js`), creating a hybrid
that depends on both systems.

**Impact:** ~65,000 lines of JS total. The legacy files are difficult to
maintain, debug, or extend. New contributors face confusion over which files
are canonical.

**Recommendation:** Complete the modular refactor. Extract the core logic
from each monolithic file into focused modules under `viewer/js/`. Remove
or archive the legacy files.

---

### HIGH-2: Redundant Force Computation in Kernel

`kernel.py:step_simulation()` computes forces **four times** per step:

```python
# 1. Initial force (conditional on step 0)
initial_acc = compute_forces(state, config)
state = jax.lax.cond(pred, _init_forces, _no_op, state)

# 2-3. Inside velocity_verlet() — computes forces TWICE:
acc = force_fn(state, config)           # current forces
acc_new = force_fn(state_new_pos, config)  # forces at new position

# 4. Post-integration force update
new_acc = compute_forces(state, config)
```

For N-body gravity, `compute_forces` is O(N^2). This means each step does
**4x** the necessary work. A standard Velocity Verlet implementation needs
at most 1 force evaluation per step (by caching the previous step's forces).

**Recommendation:** Store `entity_acc` in state (already exists) and reuse
the previous step's acceleration instead of recomputing. Remove the redundant
post-integration force call.

---

### HIGH-3: State Mutation of Frozen Dataclass

Multiple scenarios do:
```python
state.scenario_name = "random_nbody"  # random_nbody.py:97
state.scenario_name = "bulk_ring"     # bulk_ring.py:143
```

`UniverseState` is a `chex.dataclass` which is **frozen by default**. The
field `scenario_name` does not exist on `UniverseState`. This will raise
`FrozenInstanceError` if chex enforces immutability.

**Recommendation:** Remove these mutations. Pass scenario name through the
config or as a separate parameter rather than monkey-patching the state.

---

### MEDIUM-1: No Dependency Management

There is no `requirements.txt`, `pyproject.toml`, or `setup.py`. The README
lists `pip install jax matplotlib pytest` manually.

**Impact:** No version pinning. JAX has breaking API changes between versions.
No way to reproduce the exact environment.

**Recommendation:** Add a `requirements.txt` at minimum:
```
jax>=0.4.0
jaxlib>=0.4.0
chex>=0.1.0
matplotlib>=3.5.0
numpy>=1.22.0
pytest>=7.0.0
```

---

### MEDIUM-2: Topology Constants Defined in Multiple Places

```python
# topology.py
TOPOLOGY_FLAT = 0
TOPOLOGY_TORUS = 1
TOPOLOGY_SPHERE = 2
TOPOLOGY_BUBBLE = 3

# environment/topology_math.py
TOPOLOGY_FLAT = 0
TOPOLOGY_TORUS = 1
TOPOLOGY_SPHERE = 2
TOPOLOGY_BUBBLE = 3
TOPOLOGY_MOBIUS = 5

# topologies/mobius_topology.py
MOBIUS_TOPOLOGY = 5  # class constant

# cosmosim.py — inline in CORE_PHYSICS_PARAMS
"allowed": [0, 1, 2, MobiusTopology.MOBIUS_TOPOLOGY]
```

**Recommendation:** Define constants once in `state.py` or a dedicated
`constants.py` and import everywhere.

---

### MEDIUM-3: `physics_utils.py` Is Legacy Dead Code

`physics_utils.py` (471 lines) contains:
- `compute_gravity_forces()` — superseded by `physics/forces.py`
- `integrate_euler()` — superseded by `physics/integrator.py`
- `integrate_leapfrog()` — superseded by `physics/integrator.py`
- `kinetic_energy()`, `potential_energy()` — superseded by `kernel.py:compute_diagnostics()`
- `adjust_timestep()` — not called from the main simulation path

None of these are called by `kernel.py` or any scenario's `run()` function.
They exist only for backward compatibility with `run_sim.py` (also legacy).

**Recommendation:** Archive or delete `physics_utils.py` and `run_sim.py`.
If any utility functions are still needed, move them to appropriate modules.

---

### MEDIUM-4: `topology.py` vs `topologies/` Package Ambiguity

There are two overlapping topology systems:

1. **`topology.py`** (root) — Functional JAX-compatible functions using
   `jax.lax.switch` for JIT. Used by `kernel.py`.

2. **`topologies/`** (package) — OOP implementations with factory pattern,
   `Topology` base class, per-topology classes. Used by `environment/engine.py`.

These two systems don't share code. They implement the same mathematical
operations independently.

**Recommendation:** Choose one approach. Since `kernel.py` needs JIT-compatible
functions, keep the functional approach in `topology_math.py` and make the
OOP topologies delegate to those functions.

---

## 4. Duplicate & Dead Code

### Files That Can Be Removed or Archived

| File | Lines | Status | Reason |
|---|---|---|---|
| `physics_utils.py` | 471 | Dead | Superseded by `physics/`, `kernel.py` |
| `run_sim.py` | ~200 | Dead | Legacy runner, superseded by `cosmosim.py` |
| `distance_utils.py` | 171 | Dead | Superseded by `environment/topology_math.py` |
| `fix_test_html.py` | ~50 | Dead | One-off utility script |
| `viewer/test.html` | 22,498 | Legacy | Superseded by `viewer/index.html` |
| `viewer/test.js` | 3,095 | Legacy | Test harness for legacy viewer |

### Code Duplication Map

```
physics_utils.py::compute_gravity_forces  ←→  physics/forces.py::compute_forces
physics_utils.py::integrate_euler         ←→  physics/integrator.py::velocity_verlet
physics_utils.py::kinetic_energy          ←→  kernel.py::compute_diagnostics
physics_utils.py::potential_energy        ←→  kernel.py::compute_diagnostics
physics_utils.py::momentum                ←→  kernel.py::compute_diagnostics
physics_utils.py::center_of_mass          ←→  kernel.py::compute_diagnostics
distance_utils.py::compute_offset         ←→  environment/topology_math.py::compute_displacement
distance_utils.py::compute_distance       ←→  environment/topology_math.py::compute_distance
topology.py::compute_distance             ←→  environment/topology_math.py::compute_distance
topology.py::enforce_boundaries           ←→  topologies/*.py::wrap_position
```

### Estimated Removable Code
- **~1,000 lines Python** (physics_utils, distance_utils, run_sim, fix_test_html)
- **~25,000 lines JS** (legacy monolithic viewer files if modular refactor is completed)

---

## 5. Demo Readiness Assessment

### What Works Well
- Core physics simulation runs correctly (N-body gravity, Velocity Verlet)
- Multiple topology support (flat, torus, sphere)
- Parameterized Scenario System (PSS) with presets is well-designed
- JSON export pipeline produces valid Three.js-compatible data
- Web viewer (`index.html`) has clean architecture and good UX:
  - File picker for JSON loading
  - Playback controls (play/pause, frame stepping, FPS control)
  - Visual modes (mass color, velocity color, type color)
  - Trail rendering with configurable length/fade
  - Topology overlay visualization
  - Auto-camera with orbit controls
  - Keyboard shortcuts
- 33 test files provide reasonable coverage
- CLI (`cosmosim.py`) is well-structured with good help text

### Demo Blockers (Must Fix)

| # | Issue | Severity | Effort |
|---|---|---|---|
| D1 | No `requirements.txt` — reviewer can't install deps | **Blocker** | 10 min |
| D2 | `bulk_ring` crashes via CLI (`build_config` missing `params`) | **Blocker** | 30 min |
| D3 | README install instructions are PowerShell-only (Windows) | **High** | 15 min |
| D4 | State mutation (`scenario_name`) will error on frozen chex dataclass | **High** | 15 min |
| D5 | Two `index.html` / `test.html` — unclear which to demo | **Medium** | 10 min |
| D6 | No single-command "run demo" workflow documented | **Medium** | 20 min |

### Demo Strengths (Ready to Show)
- `python cosmosim.py --scenario random_nbody --export-json --steps 300` pipeline works
- Web viewer provides polished 3D visualization
- Multiple scenarios demonstrate different physics
- Preset system makes it easy to show different configurations
- Interactive debug viewer works for development demos

---

## 6. Prioritized Improvement Roadmap

### Phase 0 — Demo Readiness (Do First)

**Goal:** Get the codebase into a state where someone can clone, install, run,
and see a working demo without errors.

1. **Add `requirements.txt`** with pinned versions
2. **Fix scenario interface** — make all `build_config()` accept `params`
3. **Remove `state.scenario_name` mutations** from scenarios
4. **Update README** with cross-platform install instructions + demo workflow
5. **Designate `index.html`** as the canonical viewer entry point

### Phase 1 — Consolidate (Reduce Confusion)

**Goal:** Eliminate duplicate implementations so there's one way to do each thing.

1. **Consolidate distance/topology** into `environment/topology_math.py`
   - Delete `distance_utils.py`
   - Refactor `topology.py` to use `topology_math` for distance calculations
   - Keep `topology.py:apply_topology()` for boundary enforcement
2. **Archive legacy code** — move `physics_utils.py`, `run_sim.py` to `_archive/`
3. **Unify topology constants** — define once in `state.py`, import everywhere
4. **Connect EnvironmentEngine** to kernel or document it as future work

### Phase 2 — Performance (Fix Correctness & Speed)

**Goal:** Simulation produces correct and efficient results.

1. **Fix kernel force computation** — reduce from 4x to 1x per step
2. **Cache accelerations** between steps (Velocity Verlet optimization)
3. **JIT-compile the simulation loop** with `jax.lax.fori_loop`
4. **Profile N-body scaling** — verify O(N^2) behavior and spatial partition speedup

### Phase 3 — Web Viewer Cleanup

**Goal:** Maintainable viewer codebase.

1. **Complete modular JS refactor** — extract logic from monolithic files into `js/`
2. **Remove legacy `test.html`** and associated files
3. **Add JS module bundler** (optional, for production deployment)
4. **Add viewer error states** for common failure modes (invalid JSON, no WebGL)

### Phase 4 — Developer Experience

**Goal:** Easy for new contributors to understand and extend.

1. **Add `pyproject.toml`** with proper package configuration
2. **Add CI pipeline** (GitHub Actions) running tests on PR
3. **Add architecture diagram** to README (simplified version of this doc)
4. **Add scenario development guide** — how to create a new scenario
5. **Clean up `docs/`** — archive completed walkthroughs, organize active docs

---

## Appendix A: Full Import Graph

```
cosmosim.py
├── importlib, argparse, os, sys, pathlib, datetime, inspect
├── exporters.json_export.export_simulation
├── topologies.mobius_topology.MobiusTopology
├── (lazy) kernel
├── (lazy) viewer.viewer.Viewer
└── (dynamic) scenarios.*

kernel.py
├── jax, jax.numpy
├── state.UniverseConfig, state.UniverseState
├── entities.spawn_entity, entities.despawn_entity
├── topology.enforce_boundaries, topology.apply_topology
├── physics.integrator.velocity_verlet
├── physics.forces.compute_forces
└── environment.topology_math.compute_distance

physics/forces.py
├── jax, jax.numpy
├── state.UniverseConfig, state.UniverseState
└── environment.topology_math.compute_displacement, compute_distance

physics/integrator.py
├── jax, jax.numpy
└── state.UniverseConfig, state.UniverseState

topology.py
├── jax, jax.numpy
└── (standalone — no internal deps)

environment/topology_math.py
├── jax, jax.numpy
└── state.UniverseConfig

environment/engine.py
├── jax.numpy
├── state.UniverseConfig, state.UniverseState
├── topologies.get_topology_handler
├── topologies.base_topology.Topology
├── environment.expansion.get_expansion_handler
└── substrate.factory.get_substrate_handler

state.py
├── jax, jax.numpy
└── chex.dataclass

entities.py
├── jax, jax.numpy
└── state.UniverseState, state.UniverseConfig

exporters/json_export.py
├── json, math, pathlib
├── jax.numpy
└── kernel.step_simulation

distance_utils.py
├── jax.numpy, numpy
└── (standalone — reads config.topology_type)

physics_utils.py
├── jax.numpy
├── state.UniverseConfig
└── distance_utils.compute_offset, compute_distance
```

## Appendix B: Viewer JavaScript Import Graph

```
index.html
├── js/viewer.js
│   ├── three (Three.js)
│   ├── three/addons/OrbitControls
│   ├── ../autoCamera.js          ◄── LEGACY (5,516 lines)
│   └── ./trails.js
│
└── js/ui.js
    ├── ../frameLoader.js          ◄── LEGACY (3,429 lines)
    ├── ../player.js               ◄── LEGACY (7,243 lines)
    ├── ../visualModes.js          ◄── LEGACY (5,683 lines)
    └── ../scenarioConfig.js       ◄── LEGACY (11,254 lines)
```

## Appendix C: Topology System Comparison

| Feature | topology.py | topology_math.py | distance_utils.py | topologies/*.py |
|---|---|---|---|---|
| Flat distance | Yes | Yes | Yes | Yes |
| Torus wrapping | Yes | Yes | Yes | Yes |
| Sphere geodesic | Partial | Yes | Yes | Yes |
| Bubble metric | No | Yes | Yes | Yes |
| Mobius | No | Fallback | No | Yes |
| JAX JIT-safe | Yes | Yes | No (Python if) | No (Python) |
| Batching support | Limited | Full | No | No |
| Used by kernel | Yes | Yes | No | No |
| Used by env engine | No | No | No | Yes |

---

*End of Architecture Review*
