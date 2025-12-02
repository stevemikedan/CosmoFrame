import pytest
import importlib
import sys
import os
from unittest.mock import MagicMock, patch

# Ensure root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from state import UniverseConfig, UniverseState

TARGET_MODULES = [
    "plotting.visualize",
    "plotting.snapshot_plot",
    "plotting.trajectory_plot",
    "plotting.energy_plot",
    "scenarios.manual_run",
    "scenarios.random_nbody",
    "scenarios.scenario_runner",
]

@pytest.mark.parametrize("module_name", TARGET_MODULES)
def test_interface_existence(module_name):
    """Validate that the module exposes the required interface functions."""
    module = importlib.import_module(module_name)
    
    assert hasattr(module, "build_config"), f"{module_name} missing build_config"
    assert hasattr(module, "build_initial_state"), f"{module_name} missing build_initial_state"
    assert hasattr(module, "run"), f"{module_name} missing run"

@pytest.mark.parametrize("module_name", TARGET_MODULES)
def test_build_config_returns_config(module_name):
    """Validate that build_config returns a UniverseConfig."""
    module = importlib.import_module(module_name)
    cfg = module.build_config()
    assert isinstance(cfg, UniverseConfig), f"{module_name}.build_config did not return UniverseConfig"

@pytest.mark.parametrize("module_name", TARGET_MODULES)
def test_build_initial_state_returns_state(module_name):
    """Validate that build_initial_state returns a UniverseState."""
    module = importlib.import_module(module_name)
    cfg = module.build_config()
    state = module.build_initial_state(cfg)
    assert isinstance(state, UniverseState), f"{module_name}.build_initial_state did not return UniverseState"

@pytest.mark.parametrize("module_name", TARGET_MODULES)
def test_run_executes_and_returns_state(module_name):
    """Validate that run executes without error and returns UniverseState.
    
    We patch jax.jit to avoid compilation overhead and patch step_simulation
    to be a no-op to avoid computational overhead.
    """
    module = importlib.import_module(module_name)
    cfg = module.build_config()
    state = module.build_initial_state(cfg)

    # Mocking to speed up tests and avoid long loops/files
    # 1. Patch jax.jit to return the function as-is (identity)
    # 2. Patch step_simulation in the MODULE'S namespace (if it exists)
    # 3. Patch matplotlib.pyplot.savefig to no-op (avoid file IO)
    # 4. Patch matplotlib.pyplot.show to no-op
    
    # Create a mock that returns the state unchanged
    def mock_step(s, c):
        return s
    
    with patch("jax.jit", side_effect=lambda f, *args, **kwargs: f):
        with patch("matplotlib.pyplot.savefig"):
            with patch("matplotlib.pyplot.show"):
                # Only patch step_simulation if the module has it
                if hasattr(module, "step_simulation"):
                    with patch.object(module, "step_simulation", side_effect=mock_step):
                        final_state = module.run(cfg, state)
                else:
                    # Module doesn't use step_simulation (e.g., random_nbody)
                    final_state = module.run(cfg, state)
                
                assert isinstance(final_state, UniverseState), f"{module_name}.run did not return UniverseState"
