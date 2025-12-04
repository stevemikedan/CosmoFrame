#!/usr/bin/env python3
"""
Build-time Scenario Metadata Generator

Generates viewer/generated_scenario_metadata.js from Python scenario modules.
Run manually when scenarios change:
    python viewer/generate_scenario_metadata.py
"""

import importlib
import json
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Scenarios directory
SCENARIOS_DIR = PROJECT_ROOT / "scenarios"
OUTPUT_FILE = Path(__file__).parent / "generated_scenario_metadata.js"


def discover_scenarios():
    """Discover all scenario modules, excluding developer-only ones."""
    scenarios = {}
    
    if not SCENARIOS_DIR.exists():
        print(f"[ERROR] Scenarios directory not found: {SCENARIOS_DIR}")
        return scenarios
    
    for py_file in SCENARIOS_DIR.glob("*.py"):
        if py_file.name == "__init__.py":
            continue
        
        module_name = f"scenarios.{py_file.stem}"
        
        try:
            module = importlib.import_module(module_name)
            
            # Skip developer-only scenarios
            if getattr(module, "DEVELOPER_SCENARIO", False):
                print(f"[SKIP] {py_file.stem} (DEVELOPER_SCENARIO)")
                continue
            
            # Extract metadata
            params = getattr(module, "SCENARIO_PARAMS", None)
            presets = getattr(module, "SCENARIO_PRESETS", None)
            
            # Only include if has params or presets
            if params or presets:
                scenarios[py_file.stem] = {
                    "params": params or {},
                    "presets": presets or {}
                }
                print(f"[OK] {py_file.stem}")
            else:
                print(f"[SKIP] {py_file.stem} (no SCENARIO_PARAMS/PRESETS)")
                
        except ImportError as e:
            print(f"[ERROR] Failed to import {module_name}: {e}")
        except Exception as e:
            print(f"[ERROR] Error processing {py_file.stem}: {e}")
    
    return scenarios


def generate_js(scenarios):
    """Generate JavaScript module from scenario metadata."""
    
    # Convert to JSON-compatible format
    js_content = """/**
 * Auto-generated Scenario Metadata
 * DO NOT EDIT MANUALLY
 * 
 * Regenerate with: python viewer/generate_scenario_metadata.py
 */

export const ScenarioSchemas = """
    
    # Pretty-print JSON
    json_str = json.dumps(scenarios, indent=4)
    
    js_content += json_str + ";\n"
    
    return js_content


def main():
    print("=" * 50)
    print("Scenario Metadata Generator")
    print("=" * 50)
    
    scenarios = discover_scenarios()
    
    if not scenarios:
        print("\n[WARNING] No scenarios found with metadata.")
        return 1
    
    print(f"\n[INFO] Found {len(scenarios)} scenarios with metadata")
    
    js_content = generate_js(scenarios)
    
    with open(OUTPUT_FILE, "w") as f:
        f.write(js_content)
    
    print(f"[SUCCESS] Generated: {OUTPUT_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
