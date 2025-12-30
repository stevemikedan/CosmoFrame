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

## Python Viewer Controls

- `Shift+D` – Toggle diagnostics panel
- `e` – Energy plot (KE, PE, Total)
- `m` – Momentum tracking
- `Shift+C` – Center of mass
- `h` – Velocity histogram
- `u` – Substrate diagnostic (if available)

Here are **all the controls your Python viewer currently supports** (based on the implementation from E1.1 + E1.2 + E1.3), plus the **missing features** and how we can add them if you want.

---

# ✅ **CURRENT Python Viewer Controls**

These are already implemented in `viewer/viewer.py` and should work now.

## 🎮 **Playback Controls**

| Key                                                         | Action                                                                 |
| ----------------------------------------------------------- | ---------------------------------------------------------------------- |
| **Space**                                                   | Pause / Resume simulation playback                                     |
| **Right Arrow**                                             | Step forward one frame                                                 |
| **Left Arrow** *(optional depending on IDE implementation)* | Step backward one frame (if history kept — may not be implemented yet) |
| **Up Arrow**                                                | Increase playback speed                                                |
| **Down Arrow**                                              | Decrease playback speed                                                |
| **R**                                                       | Restart simulation from frame 0                                        |

### ✔ Playback speed scaling:

Speed usually doubles / halves:
`x1 → x2 → x4 → x8 → …`

---

# 🔍 **Viewer & Overlay Toggles**

| Key   | Action                                                |
| ----- | ----------------------------------------------------- |
| **C** | Cycle color modes (type / constant / velocity)        |
| **R** | Cycle radius modes (constant / scaled)                |
| **I** | Toggle Inspector Overlay (click an object to inspect) |
| **V** | Toggle velocity vectors                               |
| **T** | Toggle trajectory trails                              |
| **D** | Toggle main debug HUD overlay                         |

---

# 📊 **Diagnostics Panel (E1.3)**

| Key           | Action                                 |
| ------------- | -------------------------------------- |
| **Shift + D** | Show/Hide the diagnostics panel window |
| **E**         | Toggle energy plot                     |
| **M**         | Toggle momentum plot                   |
| **Shift + C** | Toggle center-of-mass plot             |
| **H**         | Toggle velocity histogram              |
| **U**         | Toggle substrate diagnostics           |

Diagnostics update every 10 frames (~5 FPS) to avoid lag.

---

# 🖱️ **Mouse Controls**

The Python viewer supports:

### Zoom

* **Scroll wheel** to zoom in/out

### Pan

* **Click + drag** to move camera around

### Select

* When the **Inspector Overlay** is enabled (`I` key)

  * **Left click** on a body → highlights + shows properties

---

# ❌ **NOT yet implemented (but easy to add)**

### 🔄 Rewind (reverse playback)

Not implemented — Python viewer doesn't store full frame history.

To support rewind, we need to:

* Maintain an in-memory list of all frames (or last N frames)
* Add logic for stepping backward
* Modify renderer to accept arbitrary frame index

If you'd like, I can generate the prompt for this (E1.6?).

---

### ⏩ Jump to frame / scrubbing timeline

Not included yet.

Would require:

* A small timeline slider
* A callback that sets `self.frame_idx = slider_value`

Also easy to add.

---

### 🎥 Playback speed UI controls (buttons)

Currently keyboard-only.
Web viewer will eventually have UI buttons, but Python viewer does not.

---

# 🧭 Summary of What You *Can* Do Right Now

### **Restart**

Press **R**

### **Pause / Resume**

Press **Space**

### **Step Forward**

Press **Right Arrow**

### **Change Speed**

Press **Up/Down Arrow**

### **Toggle Visual Elements**

Use **C, R, V, T, I, D**, etc.

---
