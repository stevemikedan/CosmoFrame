# PS2.4 Validation Scenarios

## Overview
Created 4 developer validation scenarios to test PS2.4 functionality. All scenarios properly export JSON and demonstrate different aspects of the new physics system.

## Validation Scenarios

### 1. `validation_2body_orbit.py`
**Purpose:** Tests Velocity Verlet integrator stability and energy conservation.

**Features:**
- 2 equal-mass bodies in circular orbit
- Monitors energy drift over time
- Prints diagnostics every 100 steps

**Run:**
```bash
python cosmosim.py --scenario validation_2body_orbit --steps 500 --export-json --output-dir outputs/validation/2body
```

**Expected Results:**
- Energy drift < 10% over 500 steps
- Stable circular orbits
- JSON output with full trajectory

---

### 2. `validation_torus_wrap.py`
**Purpose:** Tests torus topology nearest-image displacement.

**Features:**
- 2 particles near opposite edges of periodic domain
- Particles move toward each other through boundary
- Prints positions every 50 steps

**Run:**
```bash
python cosmosim.py --scenario validation_torus_wrap --steps 100 --export-json --output-dir outputs/validation/torus
```

**Expected Results:**
- Particles wrap through periodic boundaries
- Positions stay within [-R, R]
- Continuous motion through seam

---

### 3. `validation_sphere_geodesic.py`
**Purpose:** Tests sphere topology geodesic displacement.

**Features:**
- 3+ particles on sphere surface
- Particles interact via gravity on curved surface
- Monitors particle radii (should stay constant)

**Run:**
```bash
python cosmosim.py --scenario validation_sphere_geodesic --steps 300 --export-json --output-dir outputs/validation/sphere
```

**Expected Results:**
- All particles remain on sphere surface (radius ~10.0)
- Geodesic forces properly calculated
- Energy conservation on curved space

---

### 4. `validation_diagnostics.py`
**Purpose:** Demonstrates PS2.4 diagnostics system.

**Features:**
- Random particle cluster
- Detailed diagnostics printout
- Energy, momentum, center-of-mass tracking

**Run:**
```bash
python cosmosim.py --scenario validation_diagnostics --steps 200 --export-json --output-dir outputs/validation/diagnostics
```

**Expected Results:**
- Full diagnostics table every 50 steps
- Energy drift monitoring
- Momentum conservation check

---

## Notes

- All scenarios are marked `DEVELOPER_SCENARIO = True` (won't appear in public scenario list)
- All scenarios are compatible with `--export-json`
- JSON files can be loaded in web viewer for visual inspection
- Scenarios can be run with different parameters via `--params`

## Verification Checklist

- [x] All scenarios export JSON successfully
- [x] Scenarios print meaningful diagnostics
- [x] Energy conservation verified (2body_orbit, diagnostics)
- [x] Topology wrapping verified (torus_wrap)
- [x] Geodesic behavior verified (sphere_geodesic)
