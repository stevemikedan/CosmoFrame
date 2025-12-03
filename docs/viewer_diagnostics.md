# CosmoSim Viewer Diagnostics

## Overview

The CosmoSim viewer provides optional, non-blocking diagnostic metrics for analyzing simulation behavior in real-time.

**Features**:
- Energy monitoring (KE, PE, Total)
- Momentum tracking
- Center of mass trajectories
- Velocity distribution analysis
- Safe with missing data (graceful fallbacks)

## Python Viewer (viewer.py)

### Keyboard Controls

| Key | Action |
|-----|--------|
| `Shift+D` | Toggle diagnostics panel on/off |
| `e` | Toggle energy plot |
| `m` | Toggle momentum plot |
| `Shift+C` | Toggle center of mass plot |
| `h` | Toggle velocity histogram |
| `u` | Toggle substrate diagnostic (if available) |

### Usage

```bash
python cosmosim.py --scenario bulk_ring --steps 500 --view debug
```

Once the viewer is running:
1. Press `Shift+D` to open diagnostics panel
2. Press individual keys (`e`, `m`, `Shift+C`, `h`) to toggle specific metrics
3. Diagnostics update at ~4-5 FPS (non-blocking)
4. Panel persists when paused

### Diagnostics Panel Layout

The panel opens in a separate window with a 2x2 grid:

```
+------------------+------------------+
|  Energy (e)      |  Momentum (m)    |
|  KE, PE, Total   |  px, py, pz      |
+------------------+------------------+
|  Center of Mass  |  Vel Histogram   |
|  (Shift+C)       |  (h)             |
+------------------+------------------+
```

Each plot shows data vs. frame index for temporal analysis.

## Web Viewer (test.html)

> **Note**: Full Chart.js integration is planned for future release. Current web viewer focuses on playback visualization.

**Planned Features**:
- Checkbox toggles for each metric
- Chart.js time-series plots
- Synchronized with frame playback
- Diagnostic data from JSON export

## Metrics Details

### Energy

**Sources** (priority order):
1. Pre-computed diagnostics from JSON (`diagnostics["KE"]`, `diagnostics["PE"]`, `diagnostics["E"]`)
2. Kinetic energy from state: `KE = 0.5 * Σ(m * v²)`
3. Potential energy: Not computed (must come from diagnostics)

**Plot**: Shows KE (blue), PE (red if available), Total (green)

### Momentum

**Computation**: `p = Σ(m * v)` for all active entities

**Plot**: Shows px (red), py (green), pz (blue if 3D)

### Center of Mass

**Computation**: `COM = Σ(m * pos) / Σ(m)`

**Plot**: Shows cx (red), cy (green), cz (blue if 3D)

### Velocity Histogram

**Computation**: Distribution of `|v|` across 20 bins

**Plot**: Bar chart showing velocity magnitude distribution

### Substrate Diagnostic

**Availability**: Only if `diagnostics` contains substrate fields

**Fields**: `substrate_min`, `substrate_max`, `substrate_mean`, grid density stats

**Fallback**: Silently skipped if not available

## Technical Details

### Performance

- Metrics update at 4-5 FPS (every 10 frames)
- Non-blocking rendering (separate figure)
- History buffers capped at 2000 entries
- No impact on main viewer performance

### Data Safety

- All computations handle missing data gracefully
- No assumptions about state structure
- Works with 2D and 3D simulations
- PE never computed from scratch (avoids physics dependency)

### Frame Synchronization

- All plots use canonical `frame_idx` for x-axis
- Ensures alignment with simulation timeline
- Critical for temporal analysis and correlation

## Troubleshooting

**Diagnostics panel doesn't appear**:
- Ensure `Shift+D` is pressed (not just `d`)
- Check that matplotlib backend supports multiple figures

**Energy plot shows only KE**:
- PE requires pre-computed diagnostics
- Export simulation with `--export-json` to include diagnostics

**Histogram is empty**:
- Ensure entities are active and moving
- Check that `h` key was pressed to enable

**Performance issues**:
- Diagnostics update every 10 frames to minimize overhead
- Disable unused metrics to reduce computation
- Close diagnostics panel when not needed (`Shift+D`)

## Development

### Adding New Metrics

1. Add metric key to `MetricsEngine.enabled`
2. Implement `compute_<metric>()` method
3. Add history buffer
4. Update `update()` to call computation
5. Add rendering in `render_diagnostics_panel()`
6. Add keyboard toggle in `on_key_press()`
7. Add tests in `test_viewer_metrics.py`

### Testing

```bash
pytest tests/test_viewer_metrics.py -v
```

Tests cover:
- Metric computation correctness
- Missing data handling
- History buffer management
- Toggle functionality
- 2D/3D compatibility

## See Also

- [Viewer README](../viewer/README.md) - Main viewer documentation
- [Architecture: Viewers](ARCHITECTURE_VIEWERS.md) - Two-viewer system design
- [CLI Consolidation](walkthroughs/cli_consolidation.md) - Entry point documentation
