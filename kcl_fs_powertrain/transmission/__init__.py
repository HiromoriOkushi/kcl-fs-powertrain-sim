"""
Transmission module for Formula Student powertrain simulation.

Models the gearbox, final drive, differential, clutch-less shifting system (CAS),
and various shift strategies.
"""

# Import main gearing components
from .gearing import (
    Transmission,
    FinalDrive,
    Differential,
    DrivetrainSystem
)

# Import CAS system components
from .cas_system import (
    CASSystem,
    ShiftState,
    ShiftDirection
)

# Import shift strategy components
from .shift_strategy import (
    ShiftStrategy,
    MaxAccelerationStrategy,
    MaxEfficiencyStrategy,
    EnduranceStrategy,
    AccelerationEventStrategy, # Keep even if specific event has own file
    SkidpadStrategy, # Added for clarity
    AutocrossStrategy, # Added for clarity
    StrategyManager,
    create_formula_student_strategies,
    ShiftPoint,
    ShiftCondition,
    StrategyType
)

# Define public API
__all__ = [
    # Gearing components
    'Transmission',
    'FinalDrive',
    'Differential',
    'DrivetrainSystem',

    # CAS system components
    'CASSystem',
    'ShiftState',
    'ShiftDirection',

    # Shift strategy components
    'ShiftStrategy',
    'MaxAccelerationStrategy',
    'MaxEfficiencyStrategy',
    'EnduranceStrategy',
    'AccelerationEventStrategy',
    'SkidpadStrategy',
    'AutocrossStrategy',
    'StrategyManager',
    'create_formula_student_strategies',
    'ShiftPoint',
    'ShiftCondition',
    'StrategyType'
]