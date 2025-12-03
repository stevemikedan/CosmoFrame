# E1.3 Implementation Walkthrough: Unified Metrics & Diagnostics Panel

## Overview

Successfully implemented optional, non-blocking diagnostic metrics for the CosmoSim Python viewer, enabling real-time analysis of simulation behavior.

**Status**: ✅ Complete - All 155 tests passing (140 baseline + 15 new metrics tests)

## What Was Implemented

### 1. MetricsEngine Module ([`viewer/metrics.py`](file:///c:/Users/steve/dev/CosmoSim/viewer/metrics.py))

**Purpose**: Centralized metric computation and history management

**Features**:
- Safe computation with missing data handling (graceful fallbacks everywhere)
- History buffers capped at MAX_HISTORY=2000 to prevent memory bloat
- Toggle system for selective metric rendering
- No physics dependencies (viewer-only)

**Metrics Implemented**:

| Metric | Computation | Fallback Strategy |
|--------|-------------|-------------------|
| Energy | KE, PE (from diagnostics), TE | Compute KE from state, never compute PE from scratch |
| Momentum | `p = Σ(m * v)` | Returns None if data missing |
| Center of Mass | `COM = Σ(m * pos) / Σ(m)` | Returns None if data missing |
| Velocity Histogram | Distribution of `｜v｜` (20 bins) | Returns None if data missing |
| Substrate Diagnostic | Extract from diagnostics if available | Silently skipped if not present |

**Critical Implementation Details**:
-  **History Pruning**: `pop(0)` when buffer exceeds MAX_HISTORY
- **JSON Compatibility**: Prefers pre-computed diagnostics, computes from state as fallback
- **2D/3D Tolerance**: Uses `np.linalg.norm(vel, axis=-1)` for velocity magnitude
- **Frame Sync**: All updates use canonical `frame_idx` from playback loop

### 2. Python Viewer Integration ([`viewer/viewer.py`](file:///c:/Users/steve/dev/CosmoSim/viewer/viewer.py))

**Keyboard Controls**:

| Key | Action |
|-----|--------|
| `Shift+D` | Toggle diagnostics panel on/off |
| `e` | Toggle energy plot (KE, PE, Total) |
| `m` | Toggle momentum plot (px, py, pz) |
| `Shift+C` | Toggle center of mass plot |
| `h` | Toggle velocity histogram |
| `u` | Toggle substrate diagnostic |

**Implementation Approach**:
- Separate matplotlib Figure (2x2 subplot grid)
- Non-blocking refresh at ~4-5 FPS (updates every 10 frames)
- Dark theme consistent with main viewer
- Panel persists across pause/resume
- Does NOT interfere with camera controls or entity selection

**Integration Points**:
```python
# In __init__
self.metrics = MetricsEngine()
self.diagnostics_fig = None
self.frame_idx = 0

# In step_once()
self.metrics.update(self.frame_idx, state_dict)

# In render()
if self.show_diagnostics_panel and self.frame_idx % 10 == 0:
    self.render_diagnostics_panel()
```

### 3. Testing ([`tests/test_viewer_metrics.py`](file:///c:/Users/steve/dev/CosmoSim/tests/test_viewer_metrics.py))

**15 Comprehensive Tests**:
- ✅ MetricsEngine initialization
- ✅ Toggle functionality  
- ✅ Energy computation (with/without diagnostics)
- ✅ Momentum & COM calculations
- ✅ Velocity histogram (2D and 3D)
- ✅ Missing data handling
- ✅ History buffer pruning
- ✅ Update integration
- ✅ Selective metric computation

**Test Results**:
```bash
pytest tests/test_viewer_metrics.py -v
# 15 passed

pytest -q
# 155 passed (140 baseline + 15 new)
```

### 4. Documentation

**Created**:
- [`docs/viewer_diagnostics.md`](file:///c:/Users/steve/dev/CosmoSim/docs/viewer_diagnostics.md) - Complete diagnostics guide
- Updated [`viewer/README.md`](file:///c:/Users/steve/dev/CosmoSim/viewer/README.md) - Added E1.3 section

**Documentation Coverage**:
- Keyboard controls reference
- Usage examples
- Performance characteristics
- Data safety guarantees
- Troubleshooting guide
- Developer extension guide

## Usage Examples

### Basic Usage

```bash
python cosmosim.py --scenario bulk_ring --steps 500 --view debug
```

Once the viewer is running:
1. Press `Shift+D` to open diagnostics panel
2. Press `e` to show energy plot
3. Press `m` to show momentum
4. Press `Shift+C` to show center of mass
5. Press `h` to show velocity histogram

### Diagnostics Panel Layout

```
+------------------+------------------+
|  Energy (e)      |  Momentum (m)    |
|  KE (blue)       |  px (red)        |
|  PE (red)        |  py (green)      |
|  Total (green)   |  pz (blue)       |
+------------------+------------------+
|  Center of Mass  |  Vel Histogram   |
|  cx (red)        |  |v| distribution|
|  cy (green)      |  (20 bins)       |
|  cz (blue)       |                  |
+------------------+------------------+
```

## Technical Highlights

### Performance

- **Non-blocking**: Diagnostics render at 4-5 FPS (every 10 frames)
- **Separate figure**: No impact on main render loop
- **Memory safe**: History capped at 2000 entries
- **Selective computation**: Only enabled metrics are computed

### Data Safety

- **No assumptions**: All methods check for field existence
- **Graceful degradation**: Missing data returns None or empty
- **No physics coupling**: Never recomputes PE from scratch
- **Dimension agnostic**: Works with 2D and 3D simulations

### Frame Synchronization

- All plots use `frame_idx` for x-axis
- Ensures temporal alignment with simulation
- Critical for analyzing emergent behavior

## Constraints Preserved

✅ No physics modifications  
✅ No backend calculation changes  
✅ Viewer-only implementation  
✅ Optional and non-blocking  
✅ Safe with missing data  
✅ E1.1/E1.2 features intact  
✅ Camera controls unaffected  
✅ Diagnostics persist across pause  

## Known Limitations

### Web Viewer
- Full Chart.js integration not implemented in this phase
- Planned for future enhancement
- Python viewer fully functional and production-ready

### Energy Plot
- PE (Potential Energy) requires pre-computed diagnostics
- Not computed from scratch to maintain physics decoupling
- Falls back to KE-only plot if PE unavailable

## Files Modified

**New Files**:
- `viewer/metrics.py` (240 lines) - MetricsEngine implementation
- `tests/test_viewer_metrics.py` (220 lines) - Comprehensive test suite
- `docs/viewer_diagnostics.md` (200 lines) - User/dev documentation

**Modified Files**:
- `viewer/viewer.py` - Added diagnostics integration (+120 lines)
- `viewer/README.md` - Added E1.3 section (+25 lines)

**Total**: ~800 lines of new code + documentation

## Validation

### Automated Tests
```bash
pytest tests/test_viewer_metrics.py -v
# 15/15 passing

pytest -q  
# 155/155 passing (no regressions)
```

### Manual Verification
- ✅ Diagnostics panel opens/closes with `Shift+D`
- ✅ Individual metrics toggle correctly
- ✅ Plots update smoothly at 4-5 FPS
- ✅ No main viewer performance impact
- ✅ Works with pause/resume
- ✅ Safe with missing diagnostic data

## Future Enhancements

1. **Web Viewer Diagnostics** - Chart.js integration for test.html
2. **Export Diagnostics** - Save metrics history to JSON
3. **Custom Metrics** - User-defined diagnostic functions
4. **Correlation Analysis** - Cross-metric analysis tools

## See Also

- [Viewer README](file:///c:/Users/steve/dev/CosmoSim/viewer/README.md) - Main viewer documentation
- [Viewer Diagnostics](file:///c:/Users/steve/dev/CosmoSim/docs/viewer_diagnostics.md) - Complete diagnostics guide
- [CLI Consolidation](file:///c:/Users/steve/dev/CosmoSim/docs/walkthroughs/cli_consolidation.md) - Entry point documentation
- [Architecture: Viewers](file:///c:/Users/steve/dev/CosmoSim/docs/ARCHITECTURE_VIEWERS.md) - Two-viewer system design
