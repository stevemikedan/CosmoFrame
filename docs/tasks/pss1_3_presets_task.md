# PSS1.3: Demo Scenario Preset System

## Phase 1: Preset Format
- [ ] Update `scenarios/bulk_ring.py` with `SCENARIO_PRESETS`
- [ ] Update `scenarios/random_nbody.py` with `SCENARIO_PRESETS`
- [ ] Verify non-breaking change for existing scenarios

## Phase 2: Preset Selection (CLI)
- [ ] Add `--preset` argument to `cosmosim.py`
- [ ] Implement preset loading logic `load_scenario_presets`
- [ ] Implement preset validation (reuse PSS1.2 logic)

## Phase 3: Merging Logic
- [ ] Update `run_scenario` merging pipeline
- [ ] Implement: Defaults -> Preset -> CLI Overrides
- [ ] Ensure strict logging format `[PSS]` and `[PSS WARNING]`

## Phase 4: Standard Presets
- [ ] Add "stable_ring" and "wide_ring" to `bulk_ring.py`
- [ ] Add "small_cluster", "medium_cluster", "large_cluster" to `random_nbody.py`

## Phase 5: Testing
- [ ] Create `tests/test_scenario_presets.py`
- [ ] test_load_presets_basic()
- [ ] test_unknown_preset_warning()
- [ ] test_preset_applies_defaults()
- [ ] test_preset_then_params_merging_order()
- [ ] test_preset_respects_schema_types()
- [ ] test_preset_respects_bounds_and_clamping()
- [ ] test_no_schema_scenario_with_presets()
- [ ] test_scenario_without_presets_ignores_preset()

## Validation
- [ ] All new tests pass
- [ ] All existing tests pass
- [ ] Manual verification of preset loading
