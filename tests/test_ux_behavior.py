import pytest
import importlib
import sys
import os
from unittest.mock import patch
import matplotlib.pyplot as plt

# Ensure root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from state import UniverseConfig, UniverseState

def test_printing_behavior_manual_run(capsys):
    """Confirm scenarios.manual_run prints 'Step' lines."""
    from scenarios import manual_run
    cfg = manual_run.build_config()
    state = manual_run.build_initial_state(cfg)
    
    # Mock step_simulation to avoid JAX string type errors
    def mock_step(s, c):
        return s
    
    with patch("jax.jit", side_effect=lambda f: f):
        with patch.object(manual_run, "step_simulation", side_effect=mock_step):
            manual_run.run(cfg, state)
            
    captured = capsys.readouterr()
    assert "Step" in captured.out
    assert "Running manual physics test" in captured.out

FILE_OUTPUT_MODULES = [
    "plotting.visualize",
    "plotting.snapshot_plot",
    "plotting.trajectory_plot",
    "plotting.energy_plot",
]

@pytest.mark.parametrize("module_name", FILE_OUTPUT_MODULES)
def test_file_output_behavior(module_name, tmp_path):
    """Confirm that file-generating modules create a file in the expected directory.
    We monkeypatch os.path.join to redirect 'outputs' to tmp_path.
    """
    module = importlib.import_module(module_name)
    cfg = module.build_config()
    state = module.build_initial_state(cfg)
    
    # Real os.path.join to use in our fake
    real_join = os.path.join
    
    def fake_join(*args):
        # If the path starts with "outputs", redirect to tmp_path
        if args and args[0] == "outputs":
            return real_join(str(tmp_path), *args[1:])
        return real_join(*args)
    
    # Mock step_simulation to avoid JAX string type errors
    def mock_step(s, c):
        return s
    
    with patch("os.path.join", side_effect=fake_join):
        with patch("jax.jit", side_effect=lambda f, *args, **kwargs: f):
            # Patch step_simulation in the module's namespace
            if hasattr(module, "step_simulation"):
                with patch.object(module, "step_simulation", side_effect=mock_step):
                    module.run(cfg, state)
            else:
                module.run(cfg, state)
                 
    # Check if any file was created in tmp_path (recursively or flat)
    # The modules create subdirs like tmp_path/animations, tmp_path/snapshots, etc.
    files_found = []
    for root, dirs, files in os.walk(tmp_path):
        for file in files:
            files_found.append(os.path.join(root, file))
            
    assert len(files_found) > 0, f"{module_name} did not create any output file in {tmp_path}"

def test_silent_execution_scenario_runner(capsys):
    """Confirm scenario_runner.py executes silently."""
    from scenarios import scenario_runner
    cfg = scenario_runner.build_config()
    state = scenario_runner.build_initial_state(cfg)
    
    scenario_runner.run(cfg, state)
    
    captured = capsys.readouterr()
    assert captured.out == "", "scenario_runner.py should be silent"
