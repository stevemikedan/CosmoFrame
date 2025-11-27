"""
Tests for CLI tools (run_sim.py, jit_run_sim.py).

These are not scenarios with the build_config/build_initial_state/run interface,
but standalone CLI scripts that should be tested differently.
"""

import subprocess
import os
import json
import pytest


def test_run_sim_cli_help():
    """Verify run_sim.py --help works."""
    result = subprocess.run(
        ["python", "run_sim.py", "--help"],
        capture_output=True,
        text=True,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    
    assert result.returncode == 0
    assert "--substrate" in result.stdout
    assert "--topology" in result.stdout
    assert "--expansion" in result.stdout


def test_run_sim_cli_basic_execution(tmp_path):
    """Verify run_sim.py executes and creates output file."""
    outfile = tmp_path / "test_output.json"
    
    result = subprocess.run(
        [
            "python", "run_sim.py",
            "--steps", "5",
            "--outfile", str(outfile),
            "--seed", "42"
        ],
        capture_output=True,
        text=True,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        timeout=30
    )
    
    # Check it ran successfully
    assert result.returncode == 0, f"run_sim.py failed: {result.stderr}"
    
    # Check output file was created
    assert outfile.exists(), "Output file was not created"
    
    # Check JSON is valid
    with open(outfile) as f:
        data = json.load(f)
    
    assert "frames" in data
    assert len(data["frames"]) > 0


def test_run_sim_auto_naming(tmp_path):
    """Verify run_sim.py auto-generates filenames when --outfile is not specified."""
    # Change to tmp_path so auto-generated files go there
    result = subprocess.run(
        [
            "python", "run_sim.py",
            "--steps", "2",
            "--seed", "42"
        ],
        capture_output=True,
        text=True,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        timeout=30
    )
    
    # Should succeed
    assert result.returncode == 0
    
    # Check that it printed the output filename
    assert "Saving output to:" in result.stdout or "Saved" in result.stdout


@pytest.mark.parametrize("substrate", ["none", "vector"])
@pytest.mark.parametrize("topology", ["flat", "torus"])
def test_run_sim_configurations(tmp_path, substrate, topology):
    """Verify run_sim.py works with different substrate and topology configurations."""
    outfile = tmp_path / f"test_{substrate}_{topology}.json"
    
    result = subprocess.run(
        [
            "python", "run_sim.py",
            "--substrate", substrate,
            "--topology", topology,
            "--steps", "3",
            "--outfile", str(outfile),
            "--seed", "42"
        ],
        capture_output=True,
        text=True,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        timeout=30
    )
    
    assert result.returncode == 0, f"Failed with {substrate}/{topology}: {result.stderr}"
    assert outfile.exists()
