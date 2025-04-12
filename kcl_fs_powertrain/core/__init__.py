"""Core simulation components."""

from .vehicle import Vehicle, create_formula_student_vehicle
from .track import Track, TrackSegment, TrackSegmentType
from .track_integration import TrackProfile, calculate_optimal_racing_line
from .simulator import Simulator, SimulationEvent, EventType, IntegrationMethod, EnvironmentConditions, ControlInputs, DataLogger

__all__ = [
    'Vehicle', 'create_formula_student_vehicle',
    'Track', 'TrackSegment', 'TrackSegmentType',
    'TrackProfile', 'calculate_optimal_racing_line',
    'Simulator', 'SimulationEvent', 'EventType', 'IntegrationMethod',
    'EnvironmentConditions', 'ControlInputs', 'DataLogger'
]