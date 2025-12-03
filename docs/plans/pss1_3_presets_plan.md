# PSS1.3 Implementation Plan: Demo Scenario Preset System

## Overview
Implement a preset system that provides named baseline configurations for scenarios. This allows users to run curated examples without manually configuring parameters.

## Architecture

### 1. Preset Format
Optional `SCENARIO_PRESETS` dictionary in scenario modules:
```python
SCENARIO_PRESETS = {
    "preset_name": { "param": value }
}
```

### 2. CLI Integration (`cosmosim.py`)
- Add `--preset <name>` argument
- `load_scenario_presets(module)` helper function

### 3. Merging Logic (`run_scenario`)
Pipeline:
1. Load Schema (PSS1.2)
2. Load Presets (PSS1.3)
3. **Merge**: Defaults -> Preset (if selected) -> CLI Overrides
4. **Validate**: Apply PSS1.2 validation to final merged dict

### 4. Logging
Strict PSS logging:
- `[PSS] Using preset 'name'`
- `[PSS WARNING] Unknown preset...`

### 5. Standard Presets
Add presets to:
- `scenarios/bulk_ring.py`
- `scenarios/random_nbody.py`

## Implementation Phases

### Phase 1: CLI & Loading Logic
- Update `cosmosim.py` to handle `--preset`
- Implement `load_scenario_presets`
- Update `run_scenario` merging logic

### Phase 2: Scenario Updates
- Add `SCENARIO_PRESETS` to `bulk_ring.py` and `random_nbody.py`

### Phase 3: Testing
- Create `tests/test_scenario_presets.py`
- Verify merging order and validation

## Constraints
- No physics changes
- No viewer changes
- Full backward compatibility
- Warnings only (non-fatal)
