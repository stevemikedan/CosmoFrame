# Scenario Setup Module UI Overhaul

**Status:** 🔄 Planned

## Goal
Dynamic, schema-driven Scenario Setup UI using **JavaScript-only** metadata registry.

## Approach
- Expand `ScenarioSchemas` in `scenarioConfig.js` to include all public scenarios
- JS metadata registry mirrors Python SCENARIO_PARAMS and SCENARIO_PRESETS
- No Python modules or JSON export pipelines

## Changes

### viewer/scenarioConfig.js
- Expand ScenarioSchemas with all 8 public scenarios
- Add preset selector with autofill
- Add steps input (default 300)
- Add output-dir input
- Update CLI command generator

### viewer/test.html
- Add preset dropdown element
- Add steps input element
- Add output-dir input element

## CLI Output Format
```bash
python cosmosim.py --scenario <name> \
    --preset <preset> \
    --params "key=val,..." \
    --steps 300 \
    --view web \
    --output-dir outputs/demo
```

## Constraints
- All changes in viewer layer only
- No Python modules added
- No JSON generation pipeline
