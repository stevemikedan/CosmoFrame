# CLI Consolidation Walkthrough

## Overview

Consolidated CosmoSim's entry points to make `cosmosim.py` the canonical CLI with unified viewer selection and a clarified two-viewer architecture.

**Status**: ✅ Complete - All 140 tests passing

## What Changed

### 1. Unified Command-Line Interface

`cosmosim.py` is now the single entry point for all simulations:

```bash
# Interactive debug mode
python cosmosim.py --scenario bulk_ring --view debug

# Web playback mode (export JSON)
python cosmosim.py --scenario bulk_ring --view web --steps 500

# Headless batch mode
python cosmosim.py --scenario bulk_ring --view none --steps 1000
```

**New Arguments**:
- `--view {auto|debug|web|none}` - Viewer mode selection
- `--dt`, `--entities`, `--seed` - Simulation parameters
- `--topology`, `--substrate`, `--expansion` - Physics parameters

### 2. Viewer Routing

The `--view` flag routes execution to the appropriate viewer:

| Mode | Destination | Use Case |
|------|-------------|----------|
| `debug` | `viewer/viewer.py` | Real-time development & debugging |
| `web` | Export to `outputs/frames/` | Presentation & analysis |
| `none` | Headless execution | Batch processing |
| `auto` | Infers from other flags | Default behavior |

### 3. Scenario Discovery

Scenarios are now discovered dynamically with backward-compatible short names:

```python
# Short names work
python cosmosim.py --scenario visualize

# Full module paths work
python cosmosim.py --scenario plotting.visualize

# Auto-discovered from scenarios/
python cosmosim.py --scenario bulk_ring  # -> scenarios.bulk_ring
```

List all available scenarios:
```bash
python cosmosim.py --list
```

### 4. Legacy Scripts

`run_sim.py` and `jit_run_sim.py` are marked as legacy but still functional. New workflows should use `cosmosim.py`.

## Usage Examples

### Debug Viewer (Interactive)
```bash
python cosmosim.py --scenario bulk_ring --steps 300 --interactive
```
- Launches Python-based viewer
- Pause, step, reset controls
- Entity inspection (click to view)
- Physics overlays (velocity, trajectories)

### Web Viewer (High-Fidelity)
```bash
python cosmosim.py --scenario bulk_ring --steps 500 --export-json
```
- Exports JSON frames to `outputs/frames/`
- Follow on-screen instructions to open `viewer/test.html`
- Smooth 60FPS playback
- Cinematic camera controls

### Headless (Production)
```bash
python cosmosim.py --scenario bulk_ring --steps 1000 --view none \
  --output-dir outputs/experiment_001
```
- No GUI overhead
- Suitable for clusters/batch processing
- Optional file output via `--output-dir`

## Architecture

```
cosmosim.py
    ├─ --view debug  → viewer/viewer.py (Interactive Python)
    ├─ --view web    → outputs/frames/ + viewer/test.html (Browser)
    └─ --view none   → Headless execution
```

**Two-Viewer Design**:
1. **Python Debug Viewer** (`viewer/viewer.py`) - Real-time interaction, physics verification
2. **Web Playback Viewer** (`viewer/test.html`) - High-fidelity visualization, sharing

See [`docs/ARCHITECTURE_VIEWERS.md`](file:///c:/Users/steve/dev/CosmoSim/docs/ARCHITECTURE_VIEWERS.md) for detailed architecture.

## Migration Guide

### For End Users

**Before:**
```bash
python run_sim.py
```

**After:**
```bash
python cosmosim.py --scenario run_sim --view none
```

### For Developers

**Import paths** for visualization tools now use `plotting.*`:
```python
# Old (still works via short names)
--scenario visualize

# New (canonical path)
--scenario plotting.visualize
```

**Export directory** changed:
- Old: `frames/`
- New: `outputs/frames/<scenario>_<steps>_<timestamp>/`

## Testing

All tests passing:
- `test_cosmosim_discovery.py` - Scenario discovery logic
- `test_cosmosim_cli.py` - CLI behavior and routing
- Full suite: **140/140 passing** ✅

## Related Documentation

- [`viewer/README.md`](file:///c:/Users/steve/dev/CosmoSim/viewer/README.md) - Viewer usage guide
- [`docs/ARCHITECTURE_VIEWERS.md`](file:///c:/Users/steve/dev/CosmoSim/docs/ARCHITECTURE_VIEWERS.md) - System architecture
- [`docs/walkthroughs/interface_enhancement_e1.md`](file:///c:/Users/steve/dev/CosmoSim/docs/walkthroughs/interface_enhancement_e1.md) - Interactive viewer features
