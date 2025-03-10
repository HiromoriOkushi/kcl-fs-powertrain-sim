"""
Engine module for Formula Student powertrain simulation.

This module provides classes and functions for modeling motorcycle engines
in Formula Student applications, with a focus on the Honda CBR600F4i engine
commonly used in competition. It includes models for performance characteristics,
thermal behavior, fuel systems, and torque curve analysis.
"""

# Import main engine model
from .motorcycle_engine import MotorcycleEngine

# Import torque curve analysis tools
from .torque_curve import TorqueCurve

# Import fuel system components
from .fuel_systems import (
    FuelType, FuelProperties, FuelInjector,
    FuelPump, FuelConsumption, FuelSystem
)

# Import thermal model components
from .engine_thermal import (
    ThermalConfig, CoolingSystem, EngineHeatModel,
    ThermalSimulation, CoolingPerformance
)

__all__ = [
    # Engine model
    'MotorcycleEngine',
    
    # Torque curve
    'TorqueCurve',
    
    # Fuel systems
    'FuelType', 'FuelProperties', 'FuelInjector',
    'FuelPump', 'FuelConsumption', 'FuelSystem',
    
    # Engine thermal
    'ThermalConfig', 'CoolingSystem', 'EngineHeatModel',
    'ThermalSimulation', 'CoolingPerformance'
]