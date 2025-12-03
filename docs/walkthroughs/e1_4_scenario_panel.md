# E1.4: Web Viewer Scenario Configuration Panel

## Overview
Implemented a dynamic scenario configuration panel in the web viewer that allows users to configure simulation parameters via a GUI and generate CLI commands. This feature leverages the PSS (Parameterized Scenario System) schemas.

**Status**: ✅ Complete

## Features
- **Dynamic UI Generation**: Automatically creates form fields based on scenario schemas
- **Type-Aware Inputs**:
  - `int`/`float`: Number inputs with min/max validation
  - `bool`: Checkboxes
  - `str`: Text inputs
- **Command Generation**: Real-time preview of `python cosmosim.py ...` commands
- **Clipboard Integration**: One-click copy to clipboard
- **Non-Blocking**: Floating panel that doesn't interfere with playback

## Usage
1. Open the web viewer (`viewer/test.html`)
2. Locate the **Scenario Setup** panel (top-right)
3. Select a scenario (e.g., `bulk_ring`)
4. Adjust parameters (N, radius, speed, etc.)
5. Click **Generate Simulation Command**
6. Click **Copy to Clipboard**
7. Paste into your terminal to run the simulation

## Technical Implementation

### 1. Schema Registry (`viewer/scenarioConfig.js`)
JS-side registry mirroring the Python `SCENARIO_PARAMS`:
```javascript
export const ScenarioSchemas = {
    "bulk_ring": {
        "N": { type: "int", default: 64, min: 1, max: 128 },
        ...
    }
};
```

### 2. UI Class (`ScenarioConfigUI`)
Handles DOM manipulation and event listeners:
- `populateScenarios()`: Fills dropdown
- `renderParams()`: Generates inputs based on selected schema
- `generateCommand()`: Builds CLI string

### 3. Integration (`viewer/test.html`)
- Added HTML container `#scenarioPanel`
- Imported `ScenarioConfigUI` module
- Initialized on page load:
```javascript
new ScenarioConfigUI(document.getElementById('scenarioPanel'), ScenarioSchemas);
```

## Constraints Preserved
✅ No Python backend changes
✅ No JSON format changes
✅ Backward compatible (viewer works without panel)
✅ No physics engine modifications

## Verification
- Panel appears in top-right
- Changing scenario updates parameters
- Generated command matches PSS format (`--params key=value`)
- Copy button works
