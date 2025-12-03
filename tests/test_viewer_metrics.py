"""
Tests for viewer/metrics.py MetricsEngine.

Validates safe metric computation, history management, and toggle behavior.
"""

import pytest
import numpy as np
from viewer.metrics import MetricsEngine


def test_metrics_engine_initialization():
    """Test MetricsEngine initializes with correct defaults."""
    engine = MetricsEngine()
    
    assert engine.enabled == {
        "energy": False,
        "momentum": False,
        "com": False,
        "velocity_hist": False,
        "substrate_diag": False,
    }
    assert engine.energy_history == []
    assert engine.com_history == []
    assert engine.momentum_history == []


def test_toggle_functionality():
    """Test toggle enables and disables metrics."""
    engine = MetricsEngine()
    
    assert engine.enabled["energy"] == False
    engine.toggle("energy")
    assert engine.enabled["energy"] == True
    engine.toggle("energy")
    assert engine.enabled["energy"] == False


def test_compute_energy_with_full_diagnostics():
    """Test energy computation when diagnostics are provided."""
    engine = MetricsEngine()
    
    # Mock data
    pos = np.array([[1.0, 0.0], [0.0, 1.0]])
    vel = np.array([[1.0, 0.0], [0.0, 1.0]])
    mass = np.array([1.0, 2.0])
    active = np.array([True, True])
    diagnostics = {'KE': 1.5, 'PE': -2.0, 'E': -0.5}
    
    result = engine.compute_energy(pos, vel, mass, active, diagnostics)
    
    assert result['KE'] == 1.5
    assert result['PE'] == -2.0
    assert result['TE'] == -0.5


def test_compute_energy_from_state():
    """Test energy computation when no diagnostics provided."""
    engine = MetricsEngine()
    
    # Mock data
    pos = np.array([[1.0, 0.0], [0.0, 1.0]])
    vel = np.array([[1.0, 0.0], [0.0, 1.0]])
    mass = np.array([1.0, 2.0])
    active = np.array([True, True])
    diagnostics = {}
    
    result = engine.compute_energy(pos, vel, mass, active, diagnostics)
    
    # KE = 0.5 * (1.0 * 1.0^2 + 2.0 * 1.0^2) = 0.5 * 3.0 = 1.5
    assert result['KE'] == pytest.approx(1.5)
    assert 'PE' not in result  # PE not computed from scratch
    assert result['TE'] == pytest.approx(1.5)  # TE = KE when PE missing


def test_compute_energy_missing_data():
    """Test energy computation with missing data."""
    engine = MetricsEngine()
    
    result = engine.compute_energy(None, None, None, None, {})
    assert result == {} or result is None


def test_compute_momentum():
    """Test momentum computation."""
    engine = MetricsEngine()
    
    vel = np.array([[1.0, 0.0], [0.0, 1.0]])
    mass = np.array([1.0, 2.0])
    active = np.array([True, True])
    
    p = engine.compute_momentum(vel, mass, active)
    
    # p = m1*v1 + m2*v2 = [1.0, 0.0] + [0.0, 2.0] = [1.0, 2.0]
    assert np.allclose(p, [1.0, 2.0])


def test_compute_momentum_missing_data():
    """Test momentum computation with missing data."""
    engine = MetricsEngine()
    
    result = engine.compute_momentum(None, None, None)
    assert result is None


def test_compute_com():
    """Test center of mass computation."""
    engine = MetricsEngine()
    
    pos = np.array([[1.0, 0.0], [0.0, 1.0]])
    mass = np.array([1.0, 2.0])
    active = np.array([True, True])
    
    com = engine.compute_com(pos, mass, active)
    
    # com = (1.0*[1,0] + 2.0*[0,1]) / 3.0 = [1/3, 2/3]
    assert np.allclose(com, [1.0/3.0, 2.0/3.0])


def test_compute_com_missing_data():
    """Test COM computation with missing data."""
    engine = MetricsEngine()
    
    result = engine.compute_com(None, None, None)
    assert result is None


def test_velocity_histogram():
    """Test velocity histogram computation."""
    engine = MetricsEngine()
    
    vel = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    active = np.array([True, True, True])
    
    counts, bin_edges = engine.compute_velocity_histogram(vel, active, bins=10)
    
    assert len(counts) == 10
    assert len(bin_edges) == 11
    assert counts.sum() == 3  # 3 active entities


def test_velocity_histogram_2d_and_3d():
    """Test histogram works for both 2D and 3D velocities."""
    engine = MetricsEngine()
    
    # 2D
    vel_2d = np.array([[1.0, 0.0], [0.0, 1.0]])
    active = np.array([True, True])
    counts_2d, _ = engine.compute_velocity_histogram(vel_2d, active)
    assert counts_2d.sum() == 2
    
    # 3D
    vel_3d = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    counts_3d, _ = engine.compute_velocity_histogram(vel_3d, active)
    assert counts_3d.sum() == 2


def test_history_buffer_pruning():
    """Test history buffers are pruned at MAX_HISTORY."""
    engine = MetricsEngine()
    engine.MAX_HISTORY = 10  # Set low for testing
    
    # Add more than MAX_HISTORY entries
    for i in range(15):
        engine.energy_history.append((i, i, i, i))
        engine._prune_history(engine.energy_history)
    
    assert len(engine.energy_history) == 10
    assert engine.energy_history[0][0] == 5  # Oldest should be frame 5


def test_update_with_complete_state():
    """Test update() integrates all metrics."""
    engine = MetricsEngine()
    engine.toggle("energy")
    engine.toggle("momentum")
    engine.toggle("com")
    
    state = {
        'pos': np.array([[1.0, 0.0], [0.0, 1.0]]),
        'vel': np.array([[1.0, 0.0], [0.0, 1.0]]),
        'mass': np.array([1.0, 2.0]),
        'active': np.array([True, True]),
        'diagnostics': {}
    }
    
    engine.update(frame_idx=1, state=state)
    
    assert len(engine.energy_history) == 1
    assert len(engine.momentum_history) == 1
    assert len(engine.com_history) == 1


def test_update_with_partial_state():
    """Test update() handles missing fields gracefully."""
    engine = MetricsEngine()
    engine.toggle("energy")
    
    state = {
        'vel': np.array([[1.0, 0.0]]),
        # Missing pos, mass, active, diagnostics
    }
    
    # Should not crash
    engine.update(frame_idx=1, state=state)


def test_update_only_enabled_metrics():
    """Test update() only computes enabled metrics."""
    engine = MetricsEngine()
    engine.toggle("energy")  # Only enable energy
    
    state = {
        'pos': np.array([[1.0, 0.0]]),
        'vel': np.array([[1.0, 0.0]]),
        'mass': np.array([1.0]),
        'active': np.array([True]),
        'diagnostics': {}
    }
    
    engine.update(frame_idx=1, state=state)
    
    assert len(engine.energy_history) == 1
    assert len(engine.momentum_history) == 0  # Not enabled
    assert len(engine.com_history) == 0  # Not enabled
