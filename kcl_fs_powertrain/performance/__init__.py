"""
Performance analysis module for Formula Student powertrain simulation.

Provides tools for simulating and analyzing vehicle performance in standard
Formula Student dynamic events like Acceleration, Lap Time (Autocross/Endurance),
and Skidpad. Includes optimization capabilities and sensitivity analysis.
"""

# Import core simulation classes needed by performance modules
# Use relative imports within the package
try:
    from ..core.vehicle import Vehicle
    from ..core.track import Track
    from ..core.track_integration import TrackProfile
    from ..core.simulator import Simulator # Base simulator if needed
except ImportError:
    # Define placeholders if core modules aren't found (e.g., during isolated testing)
    class Vehicle: pass
    class Track: pass
    class TrackProfile: pass
    class Simulator: pass

# Import specific performance simulators and analyzers
from .acceleration import AccelerationSimulator, run_fs_acceleration_simulation
from .lap_time import LapTimeSimulator, CorneringPerformance, run_fs_lap_simulation
from .optimal_lap_time import OptimalLapTimeOptimizer, run_advanced_lap_optimization
from .lap_time_optimization import run_lap_optimization, compare_optimization_methods
from .endurance import EnduranceSimulator, EnduranceAnalysis, ReliabilityEvent, run_endurance_simulation
from .weight_sensitivity import WeightSensitivityAnalyzer

# Define package exports
__all__ = [
    # Acceleration
    'AccelerationSimulator',
    'run_fs_acceleration_simulation',

    # Lap time (Basic/Track Following)
    'LapTimeSimulator',
    'CorneringPerformance',
    'run_fs_lap_simulation',

    # Optimal Lap Time (Advanced Dynamics/Optimization)
    'OptimalLapTimeOptimizer',
    'run_advanced_lap_optimization',

    # Lap Time Optimization Interface
    'run_lap_optimization',
    'compare_optimization_methods',

    # Endurance
    'EnduranceSimulator',
    'EnduranceAnalysis',
    'ReliabilityEvent',
    'run_endurance_simulation',

    # Weight Sensitivity
    'WeightSensitivityAnalyzer',
]