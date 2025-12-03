"""
Metrics Engine for CosmoSim Viewer Diagnostics.

Provides optional, non-blocking metric computation for both Python and Web viewers.
All methods are safe with missing data and gracefully fallback.
"""

import numpy as np
from typing import Dict, Tuple, Optional, List


class MetricsEngine:
    """Compute and track simulation diagnostics over time."""
    
    MAX_HISTORY = 2000  # Prevent memory bloat in long simulations
    
    def __init__(self):
        """Initialize metrics engine with empty state."""
        self.enabled = {
            "energy": False,
            "momentum": False,
            "com": False,
            "velocity_hist": False,
            "substrate_diag": False,
        }
        
        # History buffers: [(frame_idx, value), ...]
        self.energy_history = []      # (step, KE, PE, TE)
        self.com_history = []          # (step, x, y, z)
        self.momentum_history = []     # (step, px, py, pz)
        self.vel_hist_data = []        # Per-frame histogram
        
    def toggle(self, key: str):
        """Toggle a specific diagnostic on/off."""
        if key in self.enabled:
            self.enabled[key] = not self.enabled[key]
            
    def update(self, frame_idx: int, state: Dict):
        """
        Update all enabled metrics from current frame state.
        
        Args:
            frame_idx: Current simulation frame index (for x-axis sync)
            state: Dictionary with keys:
                - 'pos': np.ndarray [N, dim] (required)
                - 'vel': np.ndarray [N, dim] (required)
                - 'mass': np.ndarray [N] (required)
                - 'active': np.ndarray [N] bool (required)
                - 'diagnostics': dict (optional, may contain E, KE, PE)
                
        All fields are optional; methods handle missing data gracefully.
        """
        # Extract fields safely
        pos = state.get('pos', None)
        vel = state.get('vel', None)
        mass = state.get('mass', None)
        active = state.get('active', None)
        diagnostics = state.get('diagnostics', {})
        
        # Update energy if enabled
        if self.enabled.get("energy", False):
            energy_data = self.compute_energy(pos, vel, mass, active, diagnostics)
            if energy_data:
                entry = (frame_idx, energy_data['KE'], energy_data.get('PE'), energy_data.get('TE'))
                self.energy_history.append(entry)
                self._prune_history(self.energy_history)
        
        # Update momentum if enabled
        if self.enabled.get("momentum", False):
            p = self.compute_momentum(vel, mass, active)
            if p is not None:
                entry = (frame_idx, p[0], p[1] if len(p) > 1 else 0, p[2] if len(p) > 2 else 0)
                self.momentum_history.append(entry)
                self._prune_history(self.momentum_history)
        
        # Update COM if enabled
        if self.enabled.get("com", False):
            com = self.compute_com(pos, mass, active)
            if com is not None:
                entry = (frame_idx, com[0], com[1] if len(com) > 1 else 0, com[2] if len(com) > 2 else 0)
                self.com_history.append(entry)
                self._prune_history(self.com_history)
        
        # Update velocity histogram if enabled
        if self.enabled.get("velocity_hist", False):
            hist_data = self.compute_velocity_histogram(vel, active)
            self.vel_hist_data = hist_data  # Store most recent
            
    def _prune_history(self, history: List):
        """Remove oldest entries if history exceeds MAX_HISTORY."""
        while len(history) > self.MAX_HISTORY:
            history.pop(0)
            
    def compute_energy(
        self, 
        pos: Optional[np.ndarray], 
        vel: Optional[np.ndarray], 
        mass: Optional[np.ndarray], 
        active: Optional[np.ndarray],
        diagnostics: Dict
    ) -> Optional[Dict[str, float]]:
        """
        Compute energy metrics (KE, PE, TE).
        
        Priority:
        1. Use diagnostics["E"], diagnostics["KE"], diagnostics["PE"] if available
        2. Compute KE from state (0.5 * m * v^2)
        3. PE = None if not in diagnostics (NEVER compute PE from scratch)
        4. TE = KE + PE if both available, else TE = KE
        
        Returns:
            dict with keys 'KE', 'PE', 'TE' or None if insufficient data
        """
        result = {}
        
        # Try diagnostics first (pre-computed)
        if diagnostics:
            if 'KE' in diagnostics:
                result['KE'] = diagnostics['KE']
            if 'PE' in diagnostics:
                result['PE'] = diagnostics['PE']
            if 'E' in diagnostics:
                result['TE'] = diagnostics['E']
                
        # Compute KE from state if not in diagnostics
        if 'KE' not in result and vel is not None and mass is not None and active is not None:
            if len(vel) > 0 and len(mass) > 0:
                # Compute speed: |v| regardless of dimensionality
                speeds = np.linalg.norm(vel[active], axis=-1)
                ke = 0.5 * np.sum(mass[active] * speeds**2)
                result['KE'] = float(ke)
        
        # Compute TE if both KE and PE available
        if 'TE' not in result:
            if 'KE' in result and 'PE' in result:
                result['TE'] = result['KE'] + result['PE']
            elif 'KE' in result:
                result['TE'] = result['KE']  # Fallback: TE = KE when PE unknown
                
        return result if result else None
        
    def compute_momentum(
        self, 
        vel: Optional[np.ndarray], 
        mass: Optional[np.ndarray], 
        active: Optional[np.ndarray]
    ) -> Optional[np.ndarray]:
        """
        Compute total momentum vector: p = sum(m * v).
        
        Returns:
            np.ndarray [dim] or None if insufficient data
        """
        if vel is None or mass is None or active is None:
            return None
            
        if len(vel) == 0 or len(mass) == 0:
            return None
            
        # p = sum(m * v) for active entities
        active_vel = vel[active]
        active_mass = mass[active]
        
        if len(active_vel) == 0:
            return np.zeros(vel.shape[-1] if len(vel.shape) > 1 else 1)
            
        # Broadcast mass to match velocity dimensions
        momentum = np.sum(active_mass[:, np.newaxis] * active_vel, axis=0)
        return momentum
        
    def compute_com(
        self, 
        pos: Optional[np.ndarray], 
        mass: Optional[np.ndarray], 
        active: Optional[np.ndarray]
    ) -> Optional[np.ndarray]:
        """
        Compute center of mass: com = sum(m * pos) / sum(m).
        
        Returns:
            np.ndarray [dim] or None if insufficient data
        """
        if pos is None or mass is None or active is None:
            return None
            
        if len(pos) == 0 or len(mass) == 0:
            return None
            
        # com = sum(m * pos) / sum(m) for active entities
        active_pos = pos[active]
        active_mass = mass[active]
        
        if len(active_pos) == 0 or np.sum(active_mass) == 0:
            return np.zeros(pos.shape[-1] if len(pos.shape) > 1 else 1)
            
        total_mass = np.sum(active_mass)
        com = np.sum(active_mass[:, np.newaxis] * active_pos, axis=0) / total_mass
        return com
        
    def compute_velocity_histogram(
        self, 
        vel: Optional[np.ndarray], 
        active: Optional[np.ndarray],
        bins: int = 20
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Compute velocity magnitude histogram.
        
        Args:
            vel: Velocity array [N, dim]
            active: Active entities mask [N]
            bins: Number of histogram bins
            
        Returns:
            (counts, bin_edges) or None if insufficient data
        """
        if vel is None or active is None:
            return None
            
        if len(vel) == 0:
            return None
            
        # Compute speed: |v| (works for 2D and 3D)
        active_vel = vel[active]
        if len(active_vel) == 0:
            return None
            
        speeds = np.linalg.norm(active_vel, axis=-1)
        counts, bin_edges = np.histogram(speeds, bins=bins)
        return (counts, bin_edges)
        
    def get_substrate_diagnostic(self, diagnostics: Dict) -> Optional[Dict[str, float]]:
        """
        Extract substrate diagnostic summary from diagnostics dict.
        
        Looks for substrate-related fields like:
        - substrate_min, substrate_max, substrate_mean
        - grid_density_min, grid_density_max, etc.
        
        Returns:
            dict with substrate metrics or None if not available
        """
        result = {}
        
        # Check for substrate fields
        substrate_keys = ['substrate_min', 'substrate_max', 'substrate_mean',
                          'grid_density_min', 'grid_density_max', 'grid_density_mean']
        
        for key in substrate_keys:
            if key in diagnostics:
                result[key] = diagnostics[key]
                
        return result if result else None
