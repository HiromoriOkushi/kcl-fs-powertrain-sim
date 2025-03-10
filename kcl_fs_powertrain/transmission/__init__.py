"""
Transmission module for Formula Student powertrain simulation.

This module provides classes and functions for modeling the transmission system
of a Formula Student car, including the gearing system, Clutch-less Automatic
Shifter (CAS) system, and shift strategies optimized for different racing events.

The transmission system consists of:
1. Gearing components (transmission, final drive, differential)
2. CAS system for rapid clutch-less gear shifts
3. Shift strategy management for optimized gear selection

Together, these components provide a complete transmission model that can be
integrated with the engine and other powertrain components for a Formula Student
race car simulation.
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
    AccelerationEventStrategy,
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
    'StrategyManager',
    'create_formula_student_strategies',
    'ShiftPoint',
    'ShiftCondition',
    'StrategyType'
]