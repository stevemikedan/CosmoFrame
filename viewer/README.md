# CosmoSim Viewers

CosmoSim provides two distinct viewing tools, each serving a specific purpose in the development workflow.

## 1. Python Debug Viewer (`viewer.py`)
**Role:** Real-time Development & Debugging

This viewer runs the simulation live within Python. It is designed for immediate feedback, inspecting internal state, and verifying logic without waiting for exports.

**Features:**
- Real-time simulation loop
- Live state inspection (click entities)
- Debug HUD (Time, Energy, dt)
- Vector fields & Trajectories
- Matplotlib-based (works in standard Python environments)

**Usage:**
```bash
# Run with default test scenario
python cosmosim.py --scenario bulk_ring --view debug

# Import in your script
from viewer.viewer import Viewer
...
Viewer(config, state).run()
```

## 2. Web Playback Viewer (`test.html`)
**Role:** High-Fidelity Presentation & Analysis

This viewer runs in the browser using Three.js. It plays back pre-computed simulation data exported to JSON. It offers smoother animations, advanced camera controls, and is easier to share.

**Features:**
- Smooth 60FPS playback of complex scenes
- Auto-camera & Cinematic controls
- Topology visualization (3D overlays)
- No Python dependency (runs in browser)

**Usage:**
1. **Export Data**:
   ```bash
   python cosmosim.py --scenario bulk_ring --view web --export-json --steps 500
   ```
# CosmoSim Viewers

CosmoSim provides two distinct viewing tools, each serving a specific purpose in the development workflow.

## 1. Python Debug Viewer (`viewer.py`)
**Role:** Real-time Development & Debugging

This viewer runs the simulation live within Python. It is designed for immediate feedback, inspecting internal state, and verifying logic without waiting for exports.

**Features:**
- Real-time simulation loop
- Live state inspection (click entities)
- Debug HUD (Time, Energy, dt)
- Vector fields & Trajectories
- Matplotlib-based (works in standard Python environments)

**Usage:**
```bash
# Run with default test scenario
python cosmosim.py --scenario bulk_ring --view debug

# Import in your script
from viewer.viewer import Viewer
...
Viewer(config, state).run()
```

## 2. Web Playback Viewer (`test.html`)
**Role:** High-Fidelity Presentation & Analysis

This viewer runs in the browser using Three.js. It plays back pre-computed simulation data exported to JSON. It offers smoother animations, advanced camera controls, and is easier to share.

**Features:**
- Smooth 60FPS playback of complex scenes
- Auto-camera & Cinematic controls
- Topology visualization (3D overlays)
- No Python dependency (runs in browser)

**Usage:**
1. **Export Data**:
   ```bash
   python cosmosim.py --scenario bulk_ring --view web --export-json --steps 500
   ```
   Data will be saved to `outputs/frames/`.

2. **Open Viewer**:
   Open `viewer/test.html` in your browser.

3. **Load Data**:
   Click "Choose Directory" and select the exported folder inside `outputs/frames/`.

## 3. Diagnostics & Metrics (E1.3)

The Python viewer includes optional, non-blocking diagnostic metrics for real-time analysis.

**Keyboard Controls**:
- `Shift+D` – Toggle diagnostics panel
- `e` – Energy plot (KE, PE, Total)
- `m` – Momentum tracking
- `Shift+C` – Center of mass
- `h` – Velocity histogram
- `u` – Substrate diagnostic (if available)

**Features**:
- Separate diagnostics window (doesn't interfere with main view)
- Updates at ~4-5 FPS (non-blocking)
- Safe with missing data (graceful fallbacks)
- History buffers capped at 2000 frames

**Usage**:
```bash
python cosmosim.py --scenario bulk_ring --view debug --steps 500
# Press Shift+D in viewer to open diagnostics
# Press e, m, Shift+C, h to toggle individual metrics
```

See [`docs/viewer_diagnostics.md`](../docs/viewer_diagnostics.md) for complete documentation.

## Workflow Recommendation

1. **Develop**: Use `viewer.py` to write code, fix bugs, and verify physics stability.
2. **Export**: Once stable, run `cosmosim.py --export-json` to generate a high-quality dataset.
3. **Present**: Use `test.html` to view the results, record videos, or share with others.
