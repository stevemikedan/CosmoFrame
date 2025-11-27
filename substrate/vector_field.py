"""
Vector field substrate implementation.
"""

from __future__ import annotations
import jax.numpy as jnp
import numpy as np
from substrate.base_substrate import Substrate
from state import UniverseConfig, UniverseState


class VectorFieldSubstrate(Substrate):
    """
    A simple 3D vector field substrate.

    The field is defined over a regular grid with shape (Nx, Ny, Nz, 3).
    Entities at arbitrary positions receive forces interpolated
    from the surrounding grid cells (trilinear interpolation).
    """

    def __init__(self, config: UniverseConfig):
        super().__init__(config)

        params = getattr(config, "substrate_params", {}) or {}

        # Grid configuration
        self.grid_size = tuple(params.get("grid_size", (10, 10, 10)))
        self.amplitude = float(params.get("amplitude", 0.1))
        self.noise = bool(params.get("noise", False))

        # Pre-generate a static vector field
        gx, gy, gz = self.grid_size
        # Random field with amplitude scaling
        # Use numpy for initialization
        field_np = np.random.uniform(-1, 1, size=(gx, gy, gz, 3)) * self.amplitude
        self.field = jnp.array(field_np)

        # For coordinate normalization
        self.bounds = float(getattr(config, "radius", 10.0))

    def update(self, state: UniverseState, dt: float):
        # Optional noisy evolution of the field
        if self.noise:
            # slow drift to avoid chaotic blowup
            noise = np.random.normal(
                scale=0.01 * self.amplitude, size=self.field.shape
            )
            self.field = self.field + jnp.array(noise)

    def _interpolate(self, positions: jnp.ndarray) -> jnp.ndarray:
        """
        Trilinear interpolation from the vector field grid.
        positions: (N, 3)
        returns interpolated vectors: (N, 3)
        """
        gx, gy, gz = self.grid_size
        bounds = self.bounds

        # Normalize positions into [0, grid)
        # Map [-bounds, bounds] to [0, 1] then to [0, grid]
        p = (positions + bounds) / (2 * bounds)
        grid_dims = jnp.array([gx, gy, gz])
        p = p * grid_dims

        # Indices
        # Clip to ensure we stay within valid range for interpolation
        # We need i0 up to size-2 so i1 (i0+1) is at most size-1
        i0 = jnp.clip(jnp.floor(p[:, 0]).astype(int), 0, gx - 2)
        j0 = jnp.clip(jnp.floor(p[:, 1]).astype(int), 0, gy - 2)
        k0 = jnp.clip(jnp.floor(p[:, 2]).astype(int), 0, gz - 2)

        i1 = i0 + 1
        j1 = j0 + 1
        k1 = k0 + 1

        # Fractions
        fx = p[:, 0] - i0
        fy = p[:, 1] - j0
        fz = p[:, 2] - k0

        # Helper to get field values
        # self.field is (gx, gy, gz, 3)
        # We want to extract (N, 3) for each corner
        def get_val(ii, jj, kk):
            return self.field[ii, jj, kk]

        # Compute trilinear interpolation
        v000 = get_val(i0, j0, k0)
        v100 = get_val(i1, j0, k0)
        v010 = get_val(i0, j1, k0)
        v110 = get_val(i1, j1, k0)

        v001 = get_val(i0, j0, k1)
        v101 = get_val(i1, j0, k1)
        v011 = get_val(i0, j1, k1)
        v111 = get_val(i1, j1, k1)

        # Reshape fractions for broadcasting: (N,) -> (N, 1)
        fx_ = fx[:, None]
        fy_ = fy[:, None]
        fz_ = fz[:, None]
        
        # Interpolated vector
        result = (
            v000 * (1 - fx_) * (1 - fy_) * (1 - fz_) +
            v100 * fx_       * (1 - fy_) * (1 - fz_) +
            v010 * (1 - fx_) * fy_       * (1 - fz_) +
            v110 * fx_       * fy_       * (1 - fz_) +
            v001 * (1 - fx_) * (1 - fy_) * fz_ +
            v101 * fx_       * (1 - fy_) * fz_ +
            v011 * (1 - fx_) * fy_       * fz_ +
            v111 * fx_       * fy_       * fz_
        )

        return result

    def force_at(self, pos: jnp.ndarray, vel: jnp.ndarray) -> jnp.ndarray:
        # Forces = vector field sample at positions
        return self._interpolate(pos)
