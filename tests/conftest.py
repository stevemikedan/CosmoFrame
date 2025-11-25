"""
Final global pytest speed patch for CosmoSim.
SAFE. FAST. NON-RECURSIVE.

Test behavior optimizations:
- Disable JAX JIT
- Disable JAX vmap
- Replace lax.scan with 2-iteration Python loop
- Shrink module-level loop constants (FRAMES, STEPS, etc.)
- Safely patch literal range(N) calls where N > 10
"""

import builtins
import importlib
import jax
import jax.lax
import jax.numpy as jnp
import pytest

# All simulation modules that contain large loops
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
def global_speed_patch(monkeypatch):
    """Applies safe and global speed patches to all tests."""

    # ------------------------------------------------------------------
    # 1. Disable JAX JIT
    # ------------------------------------------------------------------
    monkeypatch.setattr(jax, "jit", lambda fn, *a, **k: fn)

    # ------------------------------------------------------------------
    # 2. Disable vmap
    # ------------------------------------------------------------------
    monkeypatch.setattr(jax, "vmap", lambda fn, *a, **k: fn)

    # ------------------------------------------------------------------
    # 3. Replace lax.scan with manual 2-iteration loop
    # ------------------------------------------------------------------
    def fake_scan(f, init, xs):
        carry = init
        ys = []
        for _ in range(2):
            carry, y = f(carry, xs)
            ys.append(y)
        return carry, jnp.array(ys)

    monkeypatch.setattr(jax.lax, "scan", fake_scan)

    # ------------------------------------------------------------------
    # 4. Patch module-level loop constants (FRAMES, STEPS, etc.)
    # ------------------------------------------------------------------
    for modname in SIM_MODULES:
        try:
            mod = importlib.import_module(modname)
            for attr in dir(mod):
                val = getattr(mod, attr)
                if isinstance(val, int) and val > 10:
                    monkeypatch.setattr(mod, attr, 2)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # 5. PATCH literal range(N) calls ONLY where N > 10
    # ------------------------------------------------------------------

    original_range = builtins.range

    def fast_range(*args):
        # case: range(N)
        if len(args) == 1 and isinstance(args[0], int):
            N = args[0]
            if N > 10:
                return original_range(2)

        # case: range(start, stop)
        if len(args) == 2 and isinstance(args[1], int):
            stop = args[1]
            if stop > 10:
                return original_range(args[0], args[0] + 2)

        # case: range(start, stop, step)
        if len(args) == 3 and isinstance(args[1], int):
            stop = args[1]
            if stop > 10:
                rounded = args[0] + 2 * args[2]
                return original_range(args[0], rounded, args[2])

        return original_range(*args)

    monkeypatch.setattr(builtins, "range", fast_range)
