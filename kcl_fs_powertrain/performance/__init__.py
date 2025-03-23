"""
Performance analysis module for Formula Student powertrain simulation.

This module provides tools for simulating, analyzing, and optimizing the 
performance of a Formula Student vehicle across different dynamic events.
It includes specialized analysis tools for acceleration, lap time, endurance,
and weight sensitivity calculations.
"""

# Import from acceleration module
from .acceleration import (
    AccelerationSimulator,
    create_acceleration_simulator,
    run_fs_acceleration_simulation
)

# Import from lap time module
from .lap_time import (
    LapTimeSimulator,
    CorneringPerformance,
    create_lap_time_simulator,
    run_fs_lap_simulation,
    create_example_track
)

# Import from optimal lap time module
from .optimal_lap_time import (
    OptimalLapTimeOptimizer,
    run_advanced_lap_optimization
)

# Import from lap time optimization module
from .lap_time_optimization import (
    run_lap_optimization,
    compare_optimization_methods
)

# Import from weight sensitivity module
from .weight_sensitivity import (
    WeightSensitivityAnalyzer
)

# Import from endurance module
from .endurance import (
    EnduranceSimulator,
    EnduranceAnalysis,
    ReliabilityEvent,
    create_endurance_simulator,
    run_endurance_simulation,
    optimize_endurance_setup,
    compare_endurance_configurations
)

# Define package exports
__all__ = [
    # Acceleration
    'AccelerationSimulator',
    'create_acceleration_simulator',
    'run_fs_acceleration_simulation',
    
    # Lap time
    'LapTimeSimulator',
    'CorneringPerformance',
    'create_lap_time_simulator',
    'run_fs_lap_simulation',
    'create_example_track',
    
    # Optimal lap time
    'OptimalLapTimeOptimizer',
    'run_advanced_lap_optimization',
    
    # Lap time optimization
    'run_lap_optimization',
    'compare_optimization_methods',
    
    # Weight sensitivity
    'WeightSensitivityAnalyzer',
    
    # Endurance
    'EnduranceSimulator',
    'EnduranceAnalysis',
    'ReliabilityEvent',
    'create_endurance_simulator',
    'run_endurance_simulation',
    'optimize_endurance_setup',
    'compare_endurance_configurations'
]