"""
Hanoi Power Grid Simulation Module

Module mô phỏng lưới điện Hà Nội tương thích với pandapower.
Cung cấp dữ liệu lưới điện làm input cho RL reward function.
"""

from .citywide_generator import generate_hanoi_citywide_grid
from .grid_loader import GridLoader
from .hanoi_substations import (
    SUBSTATIONS_500KV,
    SUBSTATIONS_220KV,
    SUBSTATIONS_110KV,
    get_substations_by_district,
    get_all_substations,
)

# CSPRL Adapter - tích hợp với hệ thống CSPRL
from .csprl_adapter import CSPRLGridAdapter, create_adapter_for_location

# Optional: POI-based load generator (requires geopandas)
try:
    from .poi_load_generator import POILoadGenerator, generate_loads_from_pois
    _POI_AVAILABLE = True
except ImportError:
    _POI_AVAILABLE = False

__all__ = [
    'generate_hanoi_citywide_grid',
    'GridLoader',
    'CSPRLGridAdapter',
    'create_adapter_for_location',
    'SUBSTATIONS_500KV',
    'SUBSTATIONS_220KV',
    'SUBSTATIONS_110KV',
    'get_substations_by_district',
    'get_all_substations',
]

if _POI_AVAILABLE:
    __all__.extend(['POILoadGenerator', 'generate_loads_from_pois'])

