"""
Thermal module for Formula Student powertrain simulation.

This module provides comprehensive thermal modeling capabilities for Formula Student
vehicles, including cooling systems, radiators, side pods, and electric cooling assistance.
It enables simulation of heat generation, transfer, and dissipation throughout the powertrain,
which is critical for optimizing vehicle performance in racing conditions.

The module includes:
- Cooling system components (radiators, water pumps, fans, thermostats)
- Rear-mounted radiator systems with specialized airflow modeling
- Side pod thermal and aerodynamic models with radiator integration
- Electric compressor systems for supplementary cooling at low speeds

Components can be used individually or integrated into a complete vehicle
thermal management system for comprehensive simulation.
"""

# Import cooling system components
from .cooling_system import (
    RadiatorType, PumpType, FanType,
    Radiator, WaterPump, CoolingFan, Thermostat, CoolingSystem,
    create_cbr600f4i_cooling_system, create_formula_student_cooling_system
)

# Import rear radiator components
from .rear_radiator import (
    MountingPosition, DuctType, RearRadiator, RearRadiatorDuct, RearRadiatorSystem,
    create_default_rear_radiator_system, create_optimized_rear_radiator_system,
    create_minimal_weight_rear_radiator_system
)

# Import side pod components
from .side_pod import (
    SidePodType, RadiatorOrientation, SidePod, SidePodRadiator,
    SidePodSystem, DualSidePodSystem,
    create_standard_side_pod_system, create_aero_optimized_side_pod_system,
    create_cooling_optimized_side_pod_system, create_minimum_weight_side_pod_system
)

# Import electric compressor components
from .electric_compressor import (
    CompressorType, CompressorControl, ElectricCompressor,
    CompressorControlModule, CoolingAssistSystem,
    create_default_cooling_assist_system, create_high_performance_cooling_assist_system,
    create_lightweight_cooling_assist_system, create_integrated_cooling_system
)

# Define public API
__all__ = [
    # Cooling system types and classes
    'RadiatorType', 'PumpType', 'FanType',
    'Radiator', 'WaterPump', 'CoolingFan', 'Thermostat', 'CoolingSystem',
    'create_cbr600f4i_cooling_system', 'create_formula_student_cooling_system',
    
    # Rear radiator types and classes
    'MountingPosition', 'DuctType', 'RearRadiator', 'RearRadiatorDuct', 'RearRadiatorSystem',
    'create_default_rear_radiator_system', 'create_optimized_rear_radiator_system',
    'create_minimal_weight_rear_radiator_system',
    
    # Side pod types and classes
    'SidePodType', 'RadiatorOrientation', 'SidePod', 'SidePodRadiator',
    'SidePodSystem', 'DualSidePodSystem',
    'create_standard_side_pod_system', 'create_aero_optimized_side_pod_system',
    'create_cooling_optimized_side_pod_system', 'create_minimum_weight_side_pod_system',
    
    # Electric compressor types and classes
    'CompressorType', 'CompressorControl', 'ElectricCompressor',
    'CompressorControlModule', 'CoolingAssistSystem',
    'create_default_cooling_assist_system', 'create_high_performance_cooling_assist_system',
    'create_lightweight_cooling_assist_system', 'create_integrated_cooling_system'
]