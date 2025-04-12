# KCL Formula Student Powertrain Simulation Package

# Import key classes and functions for easier access at the package level

# Core components
from .core.vehicle import Vehicle, create_formula_student_vehicle
from .core.track import Track, TrackSegment, TrackSegmentType
from .core.simulator import Simulator, SimulationEvent, EventType, IntegrationMethod
from .core.track_integration import TrackProfile

# Engine components
from .engine.motorcycle_engine import MotorcycleEngine
from .engine.torque_curve import TorqueCurve
from .engine.fuel_systems import FuelType, FuelProperties, FuelInjector, FuelPump, FuelSystem
from .engine.engine_thermal import EngineHeatModel, ThermalSimulation

# Transmission components
from .transmission.gearing import Transmission, FinalDrive, Differential, DrivetrainSystem
from .transmission.cas_system import CASSystem, ShiftState, ShiftDirection
from .transmission.shift_strategy import StrategyManager, create_formula_student_strategies

# Thermal components
from .thermal.cooling_system import CoolingSystem as ThermalCoolingSystem, Radiator, WaterPump, CoolingFan, Thermostat
from .thermal.side_pod import SidePod, SidePodRadiator, SidePodSystem, DualSidePodSystem
from .thermal.rear_radiator import RearRadiator, RearRadiatorSystem
from .thermal.electric_compressor import ElectricCompressor, CoolingAssistSystem

# Performance analysis
from .performance.acceleration import AccelerationSimulator, run_fs_acceleration_simulation
from .performance.lap_time import LapTimeSimulator, run_fs_lap_simulation
from .performance.optimal_lap_time import OptimalLapTimeOptimizer, run_advanced_lap_optimization
from .performance.endurance import EnduranceSimulator, run_endurance_simulation
from .performance.weight_sensitivity import WeightSensitivityAnalyzer

# Track generation
from .track_generator.generator import FSTrackGenerator
from .track_generator.enums import TrackMode, SimType
from .track_generator.utils import generate_multiple_tracks

# Utilities
from .utils.constants import * # Import all constants
from .utils.plotting import * # Import all plotting functions
from .utils.validation import * # Import all validation functions
from .utils.track_utils import * # Import track utilities

# Define package-level attributes if needed
__version__ = '0.1.0'

# Define what gets imported with 'from kcl_fs_powertrain import *'
# It's generally better to import specific modules, but this can be defined.
__all__ = [
    # Core
    'Vehicle', 'create_formula_student_vehicle', 'Track', 'TrackSegment', 'TrackSegmentType',
    'Simulator', 'SimulationEvent', 'EventType', 'IntegrationMethod', 'TrackProfile',
    # Engine
    'MotorcycleEngine', 'TorqueCurve', 'FuelType', 'FuelProperties', 'FuelInjector',
    'FuelPump', 'FuelSystem', 'EngineHeatModel', 'ThermalSimulation',
    # Transmission
    'Transmission', 'FinalDrive', 'Differential', 'DrivetrainSystem', 'CASSystem',
    'ShiftState', 'ShiftDirection', 'StrategyManager', 'create_formula_student_strategies',
    # Thermal
    'ThermalCoolingSystem', 'Radiator', 'WaterPump', 'CoolingFan', 'Thermostat',
    'SidePod', 'SidePodRadiator', 'SidePodSystem', 'DualSidePodSystem',
    'RearRadiator', 'RearRadiatorSystem', 'ElectricCompressor', 'CoolingAssistSystem',
    # Performance
    'AccelerationSimulator', 'run_fs_acceleration_simulation',
    'LapTimeSimulator', 'run_fs_lap_simulation',
    'OptimalLapTimeOptimizer', 'run_advanced_lap_optimization',
    'EnduranceSimulator', 'run_endurance_simulation',
    'WeightSensitivityAnalyzer',
    # Track Generation
    'FSTrackGenerator', 'TrackMode', 'SimType', 'generate_multiple_tracks',
    # Utilities (selectively importing common ones)
    'GRAVITY', 'KW_TO_HP', 'MS_TO_KMH', 'set_plot_style', 'save_plot', 'validate_in_range'
]

print("KCL Formula Student Powertrain Simulation Package Initialized")
