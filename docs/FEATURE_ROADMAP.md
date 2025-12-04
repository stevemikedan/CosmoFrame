# CosmoSim Feature Roadmap

## Completed Features

### ✅ Generalized Parameterized Scenario System (PSS)
- Core physics parameters (dt, G, c, topology_type, physics_mode, dim)
- Scenario-specific parameters via SCENARIO_PARAMS
- Preset configurations via SCENARIO_PRESETS
- CLI override support (--params, --preset)
- [Documentation](scenario_system_expansion.md)

### ✅ Scenario System Expansion & Cleanup
- New flagship scenarios (random_nbody_3d, binary_star, mini_solar)
- 3D support for bubble_collapse, stellar_trio
- Developer scenario filtering (DEVELOPER_SCENARIO flag)
- [Documentation](scenario_system_expansion.md)

## In Progress

### 🔄 Scenario Setup Module UI Overhaul
- Dynamic schema-driven UI in test.html
- Preset selector with autofill
- CLI command generator
- [Implementation Plan](scenario_setup_ui_overhaul.md)

## Planned

### 📋 Parameter Documentation
- Update scenario writing guide
- Document all available core parameters
- Add inline help to scenarios

### 📋 Automated Testing
- PSS integration tests
- Scenario validation tests
- 3D mode verification
