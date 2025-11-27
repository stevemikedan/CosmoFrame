"""
Substrate factory.
"""

from typing import Optional
from state import UniverseConfig
from substrate.base_substrate import Substrate, NullSubstrate
from substrate.vector_field import VectorFieldSubstrate


def get_substrate_handler(config: UniverseConfig) -> Substrate:
    """
    Factory function for substrate models.
    
    Args:
        config: UniverseConfig containing substrate type
        
    Returns:
        Substrate instance
    """
    kind = getattr(config, "substrate", "none")
    
    if kind == "none" or kind is None:
        return NullSubstrate(config)
    
    kind_lower = kind.lower()
    
    if kind_lower == "vector":
        return VectorFieldSubstrate(config)
    
    raise ValueError(f"Unknown substrate type: {kind}")
