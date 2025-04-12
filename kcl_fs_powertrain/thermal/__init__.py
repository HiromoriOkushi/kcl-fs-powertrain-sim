"""
Thermal management module for Formula Student powertrain simulation.

Provides detailed models for cooling system components like radiators, pumps, fans,
and specialized configurations like side pods and rear radiators, along with
cooling assist systems (electric compressors).
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