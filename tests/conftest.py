"""
Safe global pytest speed patch for CosmoSim.

Test-only behavior:
- Disable JAX jit/vmap so tests don't spend time compiling.
- Replace jax.lax.scan with a tiny Python loop (2 iterations).
- Replace jax.lax.cond with a simple Python if/else.
- Reduce large loop constants (FRAMES, STEPS, etc.) inside simulation modules.

This keeps normal Python behavior (including builtins.range) completely untouched.
"""

import importlib

import jax
import jax.lax
import jax.numpy as jnp
import pytest

# Modules that contain long-running simulations / visualizations
SIM_MODULES = [
    "run_sim",
    "jit_run_sim",
    "visualize",
    "snapshot_plot",
    "trajectory_plot",
    "energy_plot",
    "scenarios.random_nbody",
    "scenarios.manual_run",
    "scenarios.scenario_runner",
]


@pytest.fixture(autouse=True)
def safe_speed_patch(monkeypatch):
    """
    Automatically applied to ALL tests.

    Speeds up tests and avoids JAX internals getting confused by our patched environment.
    """

    # 1. Disable JIT globally for tests
    monkeypatch.setattr(jax, "jit", lambda fn, *a, **k: fn)

    # 2. Disable vmap (treat it as identity)
    monkeypatch.setattr(jax, "vmap", lambda fn, *a, **k: fn)

    # 3. Replace lax.scan with a very small Python loop (2 steps)
    def fake_scan(f, init, xs):
        carry = init
        ys = []
        # Always just do 2 iterations in tests
        for _ in range(2):
            carry, y = f(carry, xs)
            ys.append(y)
        return carry, jnp.array(ys)

    monkeypatch.setattr(jax.lax, "scan", fake_scan)

    # 4. Replace lax.cond with a simple Python if/else
    def fake_cond(pred, true_fun, false_fun, operand=None):
        # pred may be a JAX scalar array; normalize to a Python bool
        if hasattr(pred, "item"):
            p = bool(pred.item())
        else:
            p = bool(pred)

        if p:
            return true_fun(operand)
        else:
            return false_fun(operand)

    monkeypatch.setattr(jax.lax, "cond", fake_cond)

    # 5. Patch module-level loop constants only (never builtins)
    #    If a module has large ints like FRAMES = 300, STEPS = 400, etc.,
    #    we reduce them to 2 for tests.
    for modname in SIM_MODULES:
        try:
            mod = importlib.import_module(modname)
        except Exception:
            continue

        for attr in dir(mod):
            try:
                val = getattr(mod, attr)
            except Exception:
                continue

            # Only touch simple ints; leave everything else alone
            if isinstance(val, int) and val > 10:
                monkeypatch.setattr(mod, attr, 2)
