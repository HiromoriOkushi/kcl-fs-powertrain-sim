"""
Simulator module for Formula Student powertrain simulation.

This module provides the core simulation engine that integrates all powertrain components
and enables comprehensive performance analysis across different racing scenarios.
It features a modular, event-driven architecture with various simulation modes and
advanced data collection capabilities.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
import logging
import time
import heapq  # For priority queue in event system
from enum import Enum, auto
import yaml
import os
import pandas as pd
from scipy.interpolate import interp1d
from collections import defaultdict

# Local imports
from ..core.vehicle import Vehicle
from ..core.track import Track, TrackSegment, TrackSegmentType
from ..engine.motorcycle_engine import MotorcycleEngine
from ..transmission.gearing import DrivetrainSystem
from ..transmission.cas_system import CASSystem, ShiftDirection, ShiftState
from ..transmission.shift_strategy import StrategyManager
from ..thermal.cooling_system import CoolingSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("Simulator")


class IntegrationMethod(Enum):
    """Numerical integration methods for simulation."""
    EULER = auto()       # First-order Euler method
    MODIFIED_EULER = auto()  # Modified Euler method (Heun's method)
    RK4 = auto()         # Fourth-order Runge-Kutta method
    ADAPTIVE_RK4 = auto() # Adaptive step-size RK4


class EventType(Enum):
    """Types of events that can occur during simulation."""
    GEAR_SHIFT = auto()        # Gear shift event
    LAP_COMPLETED = auto()     # Lap completion event
    SEGMENT_TRANSITION = auto() # Track segment transition
    THERMAL_WARNING = auto()   # Thermal system warning
    THERMAL_CRITICAL = auto()  # Thermal system critical alert
    TARGET_SPEED_REACHED = auto()  # Target speed reached
    TARGET_DISTANCE_REACHED = auto()  # Target distance reached
    TARGET_TIME_REACHED = auto()  # Target time reached
    CUSTOM = auto()            # Custom event type


class EnvironmentConditions:
    """Represents environmental conditions for the simulation."""
    
    def __init__(self, ambient_temperature: float = 25.0, 
                 air_pressure: float = 101325.0, 
                 humidity: float = 0.5,
                 wind_speed: float = 0.0,
                 wind_direction: float = 0.0):
        """
        Initialize environment conditions.
        
        Args:
            ambient_temperature: Ambient temperature in °C
            air_pressure: Air pressure in Pa
            humidity: Relative humidity (0-1)
            wind_speed: Wind speed in m/s
            wind_direction: Wind direction in radians (0 = from east)
        """
        self.ambient_temperature = ambient_temperature
        self.air_pressure = air_pressure
        self.humidity = humidity
        self.wind_speed = wind_speed
        self.wind_direction = wind_direction
        
        # Derived properties
        self.air_density = self.calculate_air_density()
    
    def calculate_air_density(self) -> float:
        """
        Calculate air density based on temperature, pressure, and humidity.
        
        Returns:
            Air density in kg/m³
        """
        # Simplified air density calculation
        # For more accuracy, a full psychrometric calculation could be implemented
        T = self.ambient_temperature + 273.15  # Convert to Kelvin
        R = 287.05  # Specific gas constant for dry air, J/(kg·K)
        
        # Basic density calculation (P = ρRT)
        density = self.air_pressure / (R * T)
        
        # Apply humidity correction (simplified)
        humidity_factor = 1.0 - (0.02 * self.humidity)
        density *= humidity_factor
        
        return density


class ControlInputs:
    """Represents driver control inputs for the simulation."""
    
    def __init__(self, throttle: float = 0.0, brake: float = 0.0, 
                 steering: float = 0.0, clutch: float = 0.0):
        """
        Initialize control inputs.
        
        Args:
            throttle: Throttle position (0-1)
            brake: Brake position (0-1)
            steering: Steering angle (radians)
            clutch: Clutch position (0-1, 0 = fully engaged)
        """
        self.throttle = max(0.0, min(1.0, throttle))
        self.brake = max(0.0, min(1.0, brake))
        self.steering = steering
        self.clutch = max(0.0, min(1.0, clutch))


class SimulationEvent:
    """Represents a discrete event in the simulation."""
    
    def __init__(self, event_type: EventType, time: float, 
                 data: Dict = None, priority: int = 0):
        """
        Initialize simulation event.
        
        Args:
            event_type: Type of event
            time: Simulation time when event occurs
            data: Additional event data
            priority: Event priority (higher values = higher priority)
        """
        self.event_type = event_type
        self.time = time
        self.data = data if data else {}
        self.priority = priority
    
    def __lt__(self, other):
        """Comparison for priority queue ordering."""
        if self.time == other.time:
            return self.priority > other.priority
        return self.time < other.time


class DataLogger:
    """Data logging system for the simulation."""
    
    def __init__(self, variables: List[str], 
                 sampling_rate: Optional[float] = None,
                 sampling_distance: Optional[float] = None):
        """
        Initialize data logger.
        
        Args:
            variables: List of variable names to log
            sampling_rate: Time-based sampling rate (Hz)
            sampling_distance: Distance-based sampling interval (m)
        """
        self.variables = variables
        self.sampling_rate = sampling_rate
        self.sampling_interval = 1.0 / sampling_rate if sampling_rate else None
        self.sampling_distance = sampling_distance
        
        # Data storage
        self.data = defaultdict(list)
        self.last_sample_time = 0.0
        self.last_sample_distance = 0.0
        
        logger.info(f"Data logger initialized with {len(variables)} variables")
    
    def should_sample(self, time: float, distance: float) -> bool:
        """
        Determine if data should be sampled at current time/distance.
        
        Args:
            time: Current simulation time
            distance: Current distance traveled
            
        Returns:
            True if sampling should occur
        """
        if self.sampling_rate and time >= self.last_sample_time + self.sampling_interval:
            return True
        
        if self.sampling_distance and distance >= self.last_sample_distance + self.sampling_distance:
            return True
        
        return False
    
    def log_data(self, time: float, distance: float, state: Dict):
        """
        Log current simulation state.
        
        Args:
            time: Current simulation time
            distance: Current distance traveled
            state: Current simulation state
        """
        if not self.should_sample(time, distance):
            return
        
        # Record common parameters
        self.data['time'].append(time)
        self.data['distance'].append(distance)
        
        # Record requested variables
        for var in self.variables:
            # Handle nested attributes with dot notation
            if '.' in var:
                value = self._get_nested_attribute(state, var)
            else:
                value = state.get(var, None)
            
            self.data[var].append(value)
        
        # Update last sample time/distance
        self.last_sample_time = time
        self.last_sample_distance = distance
    
    def _get_nested_attribute(self, state: Dict, attribute_path: str):
        """
        Get nested attribute value using dot notation.
        
        Args:
            state: State dictionary
            attribute_path: Attribute path with dot notation
            
        Returns:
            Attribute value or None if not found
        """
        current = state
        for attr in attribute_path.split('.'):
            if isinstance(current, dict) and attr in current:
                current = current[attr]
            else:
                return None
        return current
    
    def get_data_frame(self) -> pd.DataFrame:
        """
        Convert logged data to DataFrame.
        
        Returns:
            Pandas DataFrame with logged data
        """
        return pd.DataFrame(self.data)
    
    def clear(self):
        """Clear all logged data."""
        self.data = defaultdict(list)
        self.last_sample_time = 0.0
        self.last_sample_distance = 0.0


class ThermalLimits:
    """Thermal limit thresholds for the simulation."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize thermal limits from configuration.
        
        Args:
            config_path: Path to configuration file
        """
        # Default values
        self.engine_warning = 110.0  # °C
        self.engine_critical = 120.0  # °C
        self.coolant_warning = 95.0  # °C
        self.coolant_critical = 105.0  # °C
        self.oil_warning = 110.0  # °C
        self.oil_critical = 130.0  # °C
        
        # Load from config if provided
        if config_path:
            self._load_config(config_path)
    
    def _load_config(self, config_path: str):
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file
        """
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            if 'thermal_limits' in config:
                limits = config['thermal_limits']
                
                # Update limits if specified in config
                self.engine_warning = limits.get('engine_warning', self.engine_warning)
                self.engine_critical = limits.get('engine_critical', self.engine_critical)
                self.coolant_warning = limits.get('coolant_warning', self.coolant_warning)
                self.coolant_critical = limits.get('coolant_critical', self.coolant_critical)
                self.oil_warning = limits.get('oil_warning', self.oil_warning)
                self.oil_critical = limits.get('oil_critical', self.oil_critical)
                
                logger.info(f"Thermal limits loaded from {config_path}")
        except Exception as e:
            logger.error(f"Error loading thermal limits from {config_path}: {str(e)}")
    
    def get_limits(self) -> Dict:
        """
        Get all thermal limits.
        
        Returns:
            Dictionary with thermal limits
        """
        return {
            'engine_warning': self.engine_warning,
            'engine_critical': self.engine_critical,
            'coolant_warning': self.coolant_warning,
            'coolant_critical': self.coolant_critical,
            'oil_warning': self.oil_warning,
            'oil_critical': self.oil_critical
        }


class Simulator:
    """
    Core simulation engine for Formula Student powertrain simulation.
    
    This class integrates all powertrain components and provides a comprehensive
    simulation environment for analyzing vehicle performance across different
    racing scenarios.
    """
    
    def __init__(self, vehicle: Optional[Vehicle] = None, 
                 track: Optional[Track] = None, 
                 config: Optional[Union[Dict, str]] = None):
        """
        Initialize simulator with optional vehicle model, track, and configuration.
        
        Args:
            vehicle: Vehicle model instance
            track: Track model instance
            config: Configuration dictionary or path to config file
        """
        # Primary components
        self.vehicle = vehicle
        self.track = track
        
        # Simulation state
        self.current_time = 0.0
        self.current_inputs = ControlInputs()
        self.track_position = 0.0
        self.current_track_properties = {}
        self.last_lap_time = 0.0
        self.lap_count = 0
        self.last_segment = None
        self.shift_in_progress = False
        self.shift_end_time = 0.0
        
        # Simulation parameters
        self.dt = 0.01  # Base time step in seconds
        self.time_factor = 1.0  # Simulation speed factor (>1 = faster than real-time)
        self.adaptive_stepping = False
        self.min_dt = 0.001  # Minimum time step for adaptive stepping
        self.max_dt = 0.05  # Maximum time step for adaptive stepping
        self.integration_method = IntegrationMethod.EULER
        self.integration_params = {}  # Method-specific parameters
        
        # Environment
        self.environment = EnvironmentConditions()
        
        # Thermal limits
        self.thermal_limits = ThermalLimits()
        
        # Event system
        self.event_queue = []  # Priority queue of events
        self.event_handlers = {event_type: [] for event_type in EventType}
        
        # Data logging
        self.loggers = []
        self.main_logger = None
        
        # Results storage
        self.results = {}
        
        # Optional subsystem references for convenience
        self.engine = None
        self.drivetrain = None
        self.shift_manager = None
        self.cas_system = None
        
        # Load configuration if provided
        if config:
            if isinstance(config, dict):
                self.configure(config_dict=config)
            else:
                self.configure(config_file=config)
        
        # Initialize subsystem references if vehicle is provided
        if vehicle:
            self._initialize_subsystem_references()
        
        logger.info("Simulator initialized")
    
    def configure(self, config_dict: Optional[Dict] = None, 
                  config_file: Optional[str] = None) -> bool:
        """
        Configure simulator parameters from dictionary or file.
        
        Args:
            config_dict: Configuration dictionary
            config_file: Path to configuration file
            
        Returns:
            Boolean indicating success
        """
        try:
            if config_file:
                # Load configuration from file
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
            elif config_dict:
                config = config_dict
            else:
                logger.error("No configuration provided")
                return False
            
            # Configure simulation parameters
            if 'simulation' in config:
                sim_config = config['simulation']
                
                # Time stepping
                self.dt = sim_config.get('time_step', self.dt)
                self.adaptive_stepping = sim_config.get('adaptive_stepping', self.adaptive_stepping)
                self.min_dt = sim_config.get('min_time_step', self.min_dt)
                self.max_dt = sim_config.get('max_time_step', self.max_dt)
                self.time_factor = sim_config.get('time_factor', self.time_factor)
                
                # Integration method
                method_name = sim_config.get('integration_method', 'EULER')
                try:
                    self.integration_method = IntegrationMethod[method_name]
                except KeyError:
                    logger.warning(f"Unknown integration method: {method_name}. Using EULER.")
                    self.integration_method = IntegrationMethod.EULER
                
                # Integration parameters
                self.integration_params = sim_config.get('integration_params', {})
            
            # Configure environment
            if 'environment' in config:
                env_config = config['environment']
                
                self.environment = EnvironmentConditions(
                    ambient_temperature=env_config.get('ambient_temperature', 25.0),
                    air_pressure=env_config.get('air_pressure', 101325.0),
                    humidity=env_config.get('humidity', 0.5),
                    wind_speed=env_config.get('wind_speed', 0.0),
                    wind_direction=env_config.get('wind_direction', 0.0)
                )
            
            # Configure thermal limits
            if 'thermal_limits' in config:
                limits = config['thermal_limits']
                self.thermal_limits = ThermalLimits()
                
                if isinstance(limits, str):
                    # Load from file
                    self.thermal_limits._load_config(limits)
                else:
                    # Direct configuration
                    self.thermal_limits.engine_warning = limits.get('engine_warning', self.thermal_limits.engine_warning)
                    self.thermal_limits.engine_critical = limits.get('engine_critical', self.thermal_limits.engine_critical)
                    self.thermal_limits.coolant_warning = limits.get('coolant_warning', self.thermal_limits.coolant_warning)
                    self.thermal_limits.coolant_critical = limits.get('coolant_critical', self.thermal_limits.coolant_critical)
                    self.thermal_limits.oil_warning = limits.get('oil_warning', self.thermal_limits.oil_warning)
                    self.thermal_limits.oil_critical = limits.get('oil_critical', self.thermal_limits.oil_critical)
            
            # Configure data logging
            if 'logging' in config:
                log_config = config['logging']
                
                # Default logger
                if 'default_logger' in log_config:
                    logger_config = log_config['default_logger']
                    variables = logger_config.get('variables', [])
                    sampling_rate = logger_config.get('sampling_rate', 10.0)
                    sampling_distance = logger_config.get('sampling_distance', None)
                    
                    # Create main logger
                    self.main_logger = DataLogger(variables, sampling_rate, sampling_distance)
                    self.loggers.append(self.main_logger)
            
            logger.info("Simulator configured successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error configuring simulator: {str(e)}")
            return False
    
    def add_vehicle(self, vehicle: Vehicle) -> int:
        """
        Add a vehicle to the simulation.
        
        Args:
            vehicle: Vehicle model instance
            
        Returns:
            Vehicle ID in the simulation (currently always 0)
        """
        self.vehicle = vehicle
        self._initialize_subsystem_references()
        logger.info(f"Vehicle added: {type(vehicle).__name__}")
        return 0  # Currently only supports one vehicle
    
    def add_track(self, track: Track) -> int:
        """
        Add a track to the simulation.
        
        Args:
            track: Track model instance
            
        Returns:
            Track ID in the simulation (currently always 0)
        """
        self.track = track
        logger.info(f"Track added: {track.name}, length: {track.total_length:.1f}m")
        return 0  # Currently only supports one track
    
    def set_integration_method(self, method: Union[str, IntegrationMethod], 
                              params: Optional[Dict] = None):
        """
        Set numerical integration method for simulation.
        
        Args:
            method: Integration method ('euler', 'rk4', etc.) or IntegrationMethod enum
            params: Method-specific parameters
        """
        if isinstance(method, str):
            try:
                self.integration_method = IntegrationMethod[method.upper()]
            except KeyError:
                logger.warning(f"Unknown integration method: {method}. Using EULER.")
                self.integration_method = IntegrationMethod.EULER
        else:
            self.integration_method = method
        
        if params:
            self.integration_params = params
        
        logger.info(f"Integration method set to {self.integration_method.name}")
    
    def set_time_step(self, dt: float, adaptive: bool = False, 
                     min_dt: Optional[float] = None, 
                     max_dt: Optional[float] = None):
        """
        Set simulation time step with optional adaptive stepping.
        
        Args:
            dt: Base time step in seconds
            adaptive: Whether to use adaptive time stepping
            min_dt: Minimum time step for adaptive stepping
            max_dt: Maximum time step for adaptive stepping
        """
        self.dt = dt
        self.adaptive_stepping = adaptive
        
        if min_dt is not None:
            self.min_dt = min_dt
        
        if max_dt is not None:
            self.max_dt = max_dt
        
        logger.info(f"Time step set to {dt}s (adaptive: {adaptive})")
    
    def register_event_handler(self, event_type: EventType, 
                             handler_function: Callable[[Dict], None]):
        """
        Register a function to handle specific event types.
        
        Args:
            event_type: Type of event to handle
            handler_function: Function to call when event occurs
        """
        if event_type in self.event_handlers:
            self.event_handlers[event_type].append(handler_function)
        else:
            self.event_handlers[event_type] = [handler_function]
        
        logger.info(f"Event handler registered for {event_type.name}")
    
    def add_data_logger(self, variables: List[str], 
                       sampling_rate: Optional[float] = None,
                       sampling_distance: Optional[float] = None) -> int:
        """
        Add data logger for specific vehicle/simulation variables.
        
        Args:
            variables: List of variable names to log
            sampling_rate: Time-based sampling rate (Hz)
            sampling_distance: Distance-based sampling interval (m)
            
        Returns:
            Logger ID
        """
        logger_id = len(self.loggers)
        new_logger = DataLogger(variables, sampling_rate, sampling_distance)
        self.loggers.append(new_logger)
        
        # Set as main logger if it's the first one
        if logger_id == 0:
            self.main_logger = new_logger
        
        logger.info(f"Data logger {logger_id} added with {len(variables)} variables")
        return logger_id
    
    def reset(self) -> bool:
        """
        Reset simulation to initial state.
        
        Returns:
            Boolean indicating success
        """
        try:
            # Reset simulation state
            self.current_time = 0.0
            self.current_inputs = ControlInputs()
            self.track_position = 0.0
            self.current_track_properties = {}
            self.last_lap_time = 0.0
            self.lap_count = 0
            self.last_segment = None
            self.shift_in_progress = False
            self.shift_end_time = 0.0
            
            # Clear event queue
            self.event_queue = []
            
            # Reset data loggers
            for data_logger in self.loggers:
                data_logger.clear()
            
            # Clear results
            self.results = {}
            
            logger.info("Simulation reset to initial state")
            return True
            
        except Exception as e:
            logger.error(f"Error resetting simulation: {str(e)}")
            return False
    
    def step(self) -> float:
        """
        Execute a single simulation step.
        
        Returns:
            Current simulation time
        """
        # Process any events due at or before current time
        self._process_events()
        
        # Determine time step (adaptive or fixed)
        dt = self._calculate_time_step()
        
        # Handle vehicle and track updates
        if self.vehicle:
            # Update vehicle state
            self._update_vehicle_state(dt)
            
            # Update track position if track is available
            if self.track:
                self._update_track_position(dt)
            
            # Update thermal systems
            self._update_thermal_systems(dt)
        
        # Log data for this step
        self._log_simulation_data()
        
        # Update simulation time
        self.current_time += dt
        
        return self.current_time
    
    def run(self, duration: Optional[float] = None, 
           distance: Optional[float] = None, 
           condition: Optional[Callable[[], bool]] = None) -> Dict:
        """
        Run simulation until specified duration, distance, or condition.
        
        Args:
            duration: Maximum simulation time in seconds
            distance: Maximum distance to simulate in meters
            condition: Function that returns True when simulation should stop
            
        Returns:
            Simulation results
        """
        logger.info("Starting simulation run")
        
        # Reset simulation if needed
        if self.current_time > 0:
            logger.info("Resetting simulation before run")
            self.reset()
        
        # Initialize stop conditions
        stop_time = float('inf') if duration is None else duration
        stop_distance = float('inf') if distance is None else distance
        
        # Add event for time-based stopping
        if duration is not None:
            self._queue_event(EventType.TARGET_TIME_REACHED, {
                'time': duration,
                'target_type': 'duration'
            })
        
        # Add event for distance-based stopping
        if distance is not None:
            self._queue_event(EventType.TARGET_DISTANCE_REACHED, {
                'distance': distance,
                'target_type': 'distance'
            })
        
        # Run simulation loop
        start_real_time = time.time()
        try:
            while (self.current_time < stop_time and 
                  (self.track_position < stop_distance or self.track is None) and
                  (condition is None or not condition())):
                
                # Execute simulation step
                self.step()
                
                # Throttle simulation speed if running slower than real-time
                if self.time_factor < 1.0:
                    elapsed_sim_time = self.current_time
                    elapsed_real_time = time.time() - start_real_time
                    target_real_time = elapsed_sim_time / self.time_factor
                    
                    if elapsed_real_time < target_real_time:
                        time.sleep(target_real_time - elapsed_real_time)
            
            # Compile results
            self._compile_results()
            
            logger.info(f"Simulation completed at t={self.current_time:.2f}s, distance={self.track_position:.2f}m")
            return self.results
            
        except Exception as e:
            logger.error(f"Error during simulation run: {str(e)}")
            return {'error': str(e)}
    
    def run_acceleration_event(self, distance: float = 75.0, 
                              max_time: float = 10.0) -> Dict:
        """
        Run a Formula Student acceleration event simulation.
        
        Args:
            distance: Acceleration distance in meters (default 75m for FS)
            max_time: Maximum simulation time in seconds
            
        Returns:
            Acceleration event results
        """
        logger.info(f"Starting acceleration event simulation ({distance}m)")
        
        if not self.vehicle:
            logger.error("No vehicle model available for acceleration event")
            return {'error': 'No vehicle model available'}
        
        # Reset simulation
        self.reset()
        
        # Set initial conditions
        self.current_inputs = ControlInputs(throttle=1.0, brake=0.0)  # Full throttle
        
        # Set initial gear
        if self.vehicle.drivetrain:
            self.vehicle.drivetrain.change_gear(1)  # Start in first gear
        
        # Register handler for distance reached event
        def on_distance_reached(event_data):
            self.results['finish_time'] = self.current_time
            self.results['finish_speed'] = self.vehicle.current_speed
        
        self.register_event_handler(EventType.TARGET_DISTANCE_REACHED, on_distance_reached)
        
        # Run simulation
        results = self.run(duration=max_time, distance=distance)
        
        # Calculate additional metrics
        if 'finish_time' in results:
            results['average_speed'] = distance / results['finish_time']
        
        # Calculate time to speed metrics
        if self.main_logger:
            df = self.main_logger.get_data_frame()
            if 'time' in df and 'vehicle.current_speed' in df:
                speeds_to_check = [10.0, 20.0, 30.0, 60*0.44704]  # 10, 20, 30 m/s and 60 mph
                
                for speed in speeds_to_check:
                    speed_key = f"time_to_{int(speed)}_mps"
                    if speed == 60*0.44704:
                        speed_key = "time_to_60_mph"
                    
                    # Find first time at or above this speed
                    above_speed = df[df['vehicle.current_speed'] >= speed]
                    if not above_speed.empty:
                        results[speed_key] = above_speed['time'].iloc[0]
        
        logger.info(f"Acceleration event completed in {results.get('finish_time', max_time):.2f}s")
        return results
    
    def run_skidpad_event(self) -> Dict:
        """
        Run a Formula Student skidpad event simulation.
        
        Returns:
            Skidpad event results
        """
        logger.info("Starting skidpad event simulation")
        
        if not self.vehicle:
            logger.error("No vehicle model available for skidpad event")
            return {'error': 'No vehicle model available'}
        
        # A skidpad event requires a custom track or control logic
        # Here we'll do a simplified version using the vehicle's skidpad method if available
        
        if hasattr(self.vehicle, 'simulate_skidpad'):
            # Use vehicle's built-in skidpad simulation
            skidpad_results = self.vehicle.simulate_skidpad(circle_radius=8.5)
            
            # Add to simulation results
            self.results.update(skidpad_results)
            
            logger.info(f"Skidpad event completed with avg time: {skidpad_results.get('average_lap_time', 0.0):.2f}s")
            return self.results
        else:
            logger.error("Vehicle model does not support skidpad simulation")
            return {'error': 'Vehicle model does not support skidpad simulation'}
    
    def run_autocross_lap(self) -> Dict:
        """
        Run a Formula Student autocross lap simulation.
        
        Returns:
            Autocross lap results
        """
        logger.info("Starting autocross lap simulation")
        
        if not self.vehicle:
            logger.error("No vehicle model available for autocross event")
            return {'error': 'No vehicle model available'}
        
        if not self.track:
            logger.error("No track available for autocross event")
            return {'error': 'No track available'}
        
        # Reset simulation
        self.reset()
        
        # Register handler for lap completion
        def on_lap_completed(event_data):
            self.results['lap_time'] = event_data.get('lap_time', 0.0)
        
        self.register_event_handler(EventType.LAP_COMPLETED, on_lap_completed)
        
        # If vehicle has a built-in lap simulation, use it
        if hasattr(self.vehicle, 'simulate_lap'):
            lap_results = self.vehicle.simulate_lap(self.track)
            self.results.update(lap_results)
            
            logger.info(f"Autocross lap completed in {lap_results.get('lap_time', 0.0):.2f}s")
            return self.results
        
        # Otherwise, run a standard simulation
        # Use racing line if available
        racing_line = None
        if hasattr(self.track, 'racing_line') and self.track.racing_line:
            racing_line = self.track.racing_line
        elif hasattr(self.track, 'calculate_racing_line'):
            racing_line = self.track.calculate_racing_line(vehicle=self.vehicle)
        
        # Calculate speed profile if racing line available
        if racing_line and hasattr(racing_line, 'calculate_speed_profile'):
            racing_line.calculate_speed_profile(self.vehicle)
        
        # Run simulation for one lap
        lap_distance = self.track.total_length
        results = self.run(distance=lap_distance)
        
        # If no lap completion event triggered, estimate lap time
        if 'lap_time' not in results and self.main_logger:
            df = self.main_logger.get_data_frame()
            if not df.empty and 'time' in df:
                results['lap_time'] = df['time'].max()
        
        logger.info(f"Autocross lap completed in {results.get('lap_time', 0.0):.2f}s")
        return results
    
    def run_endurance_event(self, laps: int = 8) -> Dict:
        """
        Run a Formula Student endurance event simulation.
        
        Args:
            laps: Number of laps to simulate
            
        Returns:
            Endurance event results
        """
        logger.info(f"Starting endurance event simulation ({laps} laps)")
        
        if not self.vehicle:
            logger.error("No vehicle model available for endurance event")
            return {'error': 'No vehicle model available'}
        
        if not self.track:
            logger.error("No track available for endurance event")
            return {'error': 'No track available'}
        
        # If vehicle has built-in endurance simulation, use it
        if hasattr(self.vehicle, 'simulate_endurance'):
            endurance_results = self.vehicle.simulate_endurance(self.track, laps=laps)
            self.results.update(endurance_results)
            
            logger.info(f"Endurance event completed in {endurance_results.get('total_time', 0.0):.2f}s")
            return self.results
        
        # Otherwise, manually simulate multiple laps
        # Reset simulation
        self.reset()
        
        # Initialize lap tracking
        lap_times = []
        completed_laps = 0
        running_total_time = 0.0
        
        # Register handler for lap completion
        def on_lap_completed(event_data):
            nonlocal lap_times, completed_laps, running_total_time
            lap_time = event_data.get('lap_time', 0.0)
            lap_times.append(lap_time)
            completed_laps += 1
            running_total_time += lap_time
            
            # Add to results as we go
            self.results['lap_times'] = lap_times
            self.results['completed_laps'] = completed_laps
            self.results['total_time'] = running_total_time
        
        self.register_event_handler(EventType.LAP_COMPLETED, on_lap_completed)
        
        # Calculate total distance to simulate
        total_distance = self.track.total_length * laps
        
        # Run simulation
        results = self.run(distance=total_distance)
        
        # Calculate additional metrics
        if lap_times:
            results['average_lap_time'] = sum(lap_times) / len(lap_times)
            results['fastest_lap'] = min(lap_times)
            results['lap_time_consistency'] = 1.0 - (np.std(lap_times) / results['average_lap_time'])
        
        logger.info(f"Endurance event completed: {completed_laps} laps in {running_total_time:.2f}s")
        return results
    
    def optimize_setup(self, parameters: List[str], 
                     objective: Callable[[Dict], float], 
                     bounds: Optional[Dict[str, Tuple[float, float]]] = None, 
                     method: str = 'grid') -> Dict:
        """
        Optimize vehicle setup for a specific objective.
        
        Args:
            parameters: List of parameters to optimize
            objective: Objective function to minimize/maximize
            bounds: Parameter bounds
            method: Optimization method
            
        Returns:
            Optimization results
        """
        logger.info(f"Starting setup optimization for {len(parameters)} parameters using {method} method")
        
        if not self.vehicle:
            logger.error("No vehicle model available for optimization")
            return {'error': 'No vehicle model available'}
        
        # Default bounds if not provided
        if bounds is None:
            bounds = {}
            for param in parameters:
                # Try to get current value
                current_value = self._get_parameter_value(param)
                if current_value is not None:
                    # Set bounds to ±20% of current value
                    bounds[param] = (current_value * 0.8, current_value * 1.2)
                else:
                    logger.warning(f"Could not determine bounds for parameter {param}")
                    bounds[param] = (0.5, 1.5)  # Default bounds
        
        # Store original parameter values
        original_values = {}
        for param in parameters:
            original_values[param] = self._get_parameter_value(param)
        
        # Optimization results
        best_params = {}
        best_objective = float('inf')
        all_evaluations = []
        
        if method == 'grid':
            # Simple grid search
            # Define grid points for each parameter
            n_points = 5  # Number of points per dimension
            grid_points = {}
            
            for param in parameters:
                param_bounds = bounds.get(param, (0.5, 1.5))
                grid_points[param] = np.linspace(param_bounds[0], param_bounds[1], n_points)
            
            # Generate all combinations of parameters
            import itertools
            param_combinations = list(itertools.product(*[grid_points[param] for param in parameters]))
            
            # Evaluate each combination
            for i, param_values in enumerate(param_combinations):
                param_dict = {param: value for param, value in zip(parameters, param_values)}
                
                # Set parameters
                for param, value in param_dict.items():
                    self._set_parameter_value(param, value)
                
                # Reset simulation
                self.reset()
                
                # Evaluate objective
                obj_value = objective(param_dict)
                
                # Track evaluation
                evaluation = {
                    'params': param_dict,
                    'objective': obj_value
                }
                all_evaluations.append(evaluation)
                
                # Update best if improved
                if obj_value < best_objective:
                    best_objective = obj_value
                    best_params = param_dict.copy()
                
                logger.info(f"Optimization progress: {i+1}/{len(param_combinations)}, "
                           f"current best: {best_objective:.4f}")
            
        elif method == 'random':
            # Random search
            n_evaluations = 50
            
            for i in range(n_evaluations):
                # Generate random parameter values within bounds
                param_dict = {}
                for param in parameters:
                    param_bounds = bounds.get(param, (0.5, 1.5))
                    param_dict[param] = np.random.uniform(param_bounds[0], param_bounds[1])
                
                # Set parameters
                for param, value in param_dict.items():
                    self._set_parameter_value(param, value)
                
                # Reset simulation
                self.reset()
                
                # Evaluate objective
                obj_value = objective(param_dict)
                
                # Track evaluation
                evaluation = {
                    'params': param_dict,
                    'objective': obj_value
                }
                all_evaluations.append(evaluation)
                
                # Update best if improved
                if obj_value < best_objective:
                    best_objective = obj_value
                    best_params = param_dict.copy()
                
                logger.info(f"Optimization progress: {i+1}/{n_evaluations}, "
                           f"current best: {best_objective:.4f}")
        
        else:
            logger.error(f"Unsupported optimization method: {method}")
            return {'error': f'Unsupported optimization method: {method}'}
        
        # Restore original parameter values
        for param, value in original_values.items():
            self._set_parameter_value(param, value)
        
        # Compile results
        optimization_results = {
            'best_params': best_params,
            'best_objective': best_objective,
            'all_evaluations': all_evaluations,
            'original_params': original_values
        }
        
        logger.info(f"Optimization completed with best objective: {best_objective:.4f}")
        return optimization_results
    
    def get_state(self) -> Dict:
        """
        Get current simulation state.
        
        Returns:
            Dictionary with current state
        """
        state = {
            'time': self.current_time,
            'inputs': {
                'throttle': self.current_inputs.throttle,
                'brake': self.current_inputs.brake,
                'steering': self.current_inputs.steering,
                'clutch': self.current_inputs.clutch
            },
            'environment': {
                'ambient_temperature': self.environment.ambient_temperature,
                'air_pressure': self.environment.air_pressure,
                'humidity': self.environment.humidity,
                'air_density': self.environment.air_density
            }
        }
        
        # Add vehicle state if available
        if self.vehicle:
            vehicle_state = {
                'speed': self.vehicle.current_speed,
                'acceleration': self.vehicle.current_acceleration,
                'position': self.vehicle.current_position,
                'gear': self.vehicle.current_gear,
                'engine_rpm': self.vehicle.current_engine_rpm,
                'engine_torque': self.vehicle.current_engine_torque,
                'wheel_torque': self.vehicle.current_wheel_torque
            }
            
            # Add thermal state if available
            if hasattr(self.vehicle.engine, 'get_temperature_state'):
                vehicle_state['temperature'] = self.vehicle.engine.get_temperature_state()
            elif hasattr(self.vehicle.engine, 'engine_temperature'):
                vehicle_state['temperature'] = {
                    'engine': self.vehicle.engine.engine_temperature,
                    'coolant': self.vehicle.engine.coolant_temperature,
                    'oil': self.vehicle.engine.oil_temperature
                }
            
            state['vehicle'] = vehicle_state
        
        # Add track state if available
        if self.track:
            track_state = {
                'position': self.track_position,
                'lap_count': self.lap_count,
                'lap_time': self.current_time - self.last_lap_time,
                'last_lap_time': self.last_lap_time
            }
            
            if self.current_track_properties:
                track_state['properties'] = self.current_track_properties
            
            state['track'] = track_state
        
        return state
    
    def get_results(self, result_type: Optional[str] = None) -> Dict:
        """
        Get simulation results.
        
        Args:
            result_type: Type of results to return
            
        Returns:
            Results dictionary
        """
        if not self.results:
            # Compile results if not already done
            self._compile_results()
        
        if result_type:
            # Return specific result type if available
            if result_type in self.results:
                return {result_type: self.results[result_type]}
            else:
                logger.warning(f"Result type {result_type} not found")
                return {}
        
        return self.results
    
    def visualize_results(self, result_type: Optional[str] = None, 
                        plot_type: Optional[str] = None):
        """
        Visualize simulation results.
        
        Args:
            result_type: Type of results to visualize
            plot_type: Type of plot to generate
            
        Returns:
            Plot handle
        """
        if not self.results:
            logger.warning("No results available to visualize")
            return None
        
        # Determine plot type if not specified
        if plot_type is None:
            if result_type == 'acceleration':
                plot_type = 'speed_vs_time'
            elif result_type == 'lap':
                plot_type = 'speed_vs_distance'
            else:
                plot_type = 'speed_vs_time'
        
        # Get data from main logger
        if not self.main_logger:
            logger.warning("No data logger available for visualization")
            return None
        
        df = self.main_logger.get_data_frame()
        if df.empty:
            logger.warning("No data available for visualization")
            return None
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        if plot_type == 'speed_vs_time':
            if 'time' in df and 'vehicle.current_speed' in df:
                plt.plot(df['time'], df['vehicle.current_speed'] * 3.6, 'b-', linewidth=2)
                plt.xlabel('Time (s)')
                plt.ylabel('Speed (km/h)')
                plt.title('Vehicle Speed vs Time')
                plt.grid(True)
            else:
                logger.warning("Required data not available for speed vs time plot")
                return None
        
        elif plot_type == 'speed_vs_distance':
            if 'distance' in df and 'vehicle.current_speed' in df:
                plt.plot(df['distance'], df['vehicle.current_speed'] * 3.6, 'b-', linewidth=2)
                plt.xlabel('Distance (m)')
                plt.ylabel('Speed (km/h)')
                plt.title('Vehicle Speed vs Distance')
                plt.grid(True)
            else:
                logger.warning("Required data not available for speed vs distance plot")
                return None
        
        elif plot_type == 'engine_rpm_vs_time':
            if 'time' in df and 'vehicle.current_engine_rpm' in df:
                plt.plot(df['time'], df['vehicle.current_engine_rpm'], 'r-', linewidth=2)
                plt.xlabel('Time (s)')
                plt.ylabel('Engine Speed (RPM)')
                plt.title('Engine RPM vs Time')
                plt.grid(True)
            else:
                logger.warning("Required data not available for RPM vs time plot")
                return None
        
        elif plot_type == 'engine_torque_vs_rpm':
            if 'vehicle.current_engine_rpm' in df and 'vehicle.current_engine_torque' in df:
                plt.scatter(df['vehicle.current_engine_rpm'], df['vehicle.current_engine_torque'], 
                          c=df['time'], cmap='viridis', s=10, alpha=0.7)
                plt.colorbar(label='Time (s)')
                plt.xlabel('Engine Speed (RPM)')
                plt.ylabel('Engine Torque (Nm)')
                plt.title('Engine Torque vs RPM')
                plt.grid(True)
            else:
                logger.warning("Required data not available for torque vs RPM plot")
                return None
        
        elif plot_type == 'thermal':
            # Check available temperature data
            temp_cols = [col for col in df.columns if 'temp' in col.lower()]
            if temp_cols:
                for col in temp_cols:
                    plt.plot(df['time'], df[col], linewidth=2, label=col)
                
                plt.xlabel('Time (s)')
                plt.ylabel('Temperature (°C)')
                plt.title('Thermal Performance')
                plt.grid(True)
                plt.legend()
            else:
                logger.warning("No temperature data available for thermal plot")
                return None
        
        elif plot_type == 'custom':
            # Return figure for custom plotting
            return plt.gcf()
        
        else:
            logger.warning(f"Unknown plot type: {plot_type}")
            plt.close()
            return None
        
        plt.tight_layout()
        return plt.gcf()
    
    def export_results(self, filepath: str, format: str = 'csv', 
                      result_type: Optional[str] = None) -> bool:
        """
        Export simulation results to file.
        
        Args:
            filepath: Output file path
            format: Output format ('csv', 'json', 'excel', etc.)
            result_type: Type of results to export
            
        Returns:
            Boolean indicating success
        """
        try:
            # Get data
            if not self.main_logger:
                logger.warning("No data logger available for export")
                return False
            
            df = self.main_logger.get_data_frame()
            if df.empty:
                logger.warning("No data available for export")
                return False
            
            # Filter by result type if specified
            if result_type:
                # Filter columns based on result type prefix
                result_cols = [col for col in df.columns if col.startswith(result_type)]
                if result_cols:
                    df = df[['time', 'distance'] + result_cols]
            
            # Export based on format
            if format.lower() == 'csv':
                df.to_csv(filepath, index=False)
            elif format.lower() == 'json':
                df.to_json(filepath, orient='records')
            elif format.lower() == 'excel':
                df.to_excel(filepath, index=False)
            else:
                logger.warning(f"Unsupported export format: {format}")
                return False
            
            logger.info(f"Results exported to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting results: {str(e)}")
            return False
    
    def _initialize_subsystem_references(self):
        """Initialize references to vehicle subsystems for convenience."""
        if not self.vehicle:
            return
        
        # Engine reference
        if hasattr(self.vehicle, 'engine'):
            self.engine = self.vehicle.engine
        
        # Drivetrain reference
        if hasattr(self.vehicle, 'drivetrain'):
            self.drivetrain = self.vehicle.drivetrain
        
        # Shift manager reference
        if hasattr(self.vehicle, 'shift_manager'):
            self.shift_manager = self.vehicle.shift_manager
        
        # CAS system reference
        if hasattr(self.vehicle, 'cas_system'):
            self.cas_system = self.vehicle.cas_system
    
    def _calculate_time_step(self) -> float:
        """
        Calculate time step based on current conditions.
        
        Returns:
            Time step in seconds
        """
        if not self.adaptive_stepping:
            return self.dt
        
        # Default to base time step
        dt = self.dt
        
        # Adjust time step based on conditions
        # 1. Smaller steps during gear shifts
        if self.shift_in_progress:
            dt = self.min_dt
        
        # 2. Smaller steps in corners (if tracking track properties)
        elif self.current_track_properties and 'curvature' in self.current_track_properties:
            curvature = abs(self.current_track_properties['curvature'])
            if curvature > 0.05:  # Significant corner
                dt = max(self.min_dt, self.dt * 0.5)
        
        # 3. Smaller steps during rapid acceleration/deceleration
        elif self.vehicle and abs(self.vehicle.current_acceleration) > 10.0:
            dt = max(self.min_dt, self.dt * 0.7)
        
        # 4. Larger steps during steady cruise
        elif (self.vehicle and abs(self.vehicle.current_acceleration) < 1.0 and 
              self.vehicle.current_speed > 5.0):
            dt = min(self.max_dt, self.dt * 1.5)
        
        return dt
    
    def _process_events(self):
        """Process all events due at or before current time."""
        while self.event_queue and self.event_queue[0].time <= self.current_time:
            event = heapq.heappop(self.event_queue)
            
            # Get event type
            event_type = event.event_type
            
            # Call appropriate event handlers
            if event_type in self.event_handlers:
                for handler in self.event_handlers[event_type]:
                    handler(event.data)
            
            # Handle built-in events
            if event_type == EventType.GEAR_SHIFT:
                self._handle_gear_shift_event(event.data)
            elif event_type == EventType.LAP_COMPLETED:
                self._handle_lap_completed_event(event.data)
            elif event_type == EventType.THERMAL_WARNING:
                self._handle_thermal_warning_event(event.data)
            elif event_type == EventType.THERMAL_CRITICAL:
                self._handle_thermal_critical_event(event.data)
    
    def _queue_event(self, event_type: EventType, data: Dict):
        """
        Add an event to the queue.
        
        Args:
            event_type: Type of event
            data: Event data dictionary
        """
        # Create event
        event = SimulationEvent(event_type, data.get('time', self.current_time), data)
        
        # Add to queue
        heapq.heappush(self.event_queue, event)
    
    def _update_vehicle_state(self, dt: float):
        """
        Update vehicle state based on current inputs and conditions.
        
        Args:
            dt: Time step in seconds
        """
        # Skip if no vehicle
        if not self.vehicle:
            return
        
        # Update engine state
        self.vehicle.update_engine_state(
            throttle=self.current_inputs.throttle,
            engine_rpm=self.vehicle.current_engine_rpm,
            ambient_temp=self.environment.ambient_temperature
        )
        
        # Update drivetrain state
        self.vehicle.update_drivetrain_state(self.vehicle.current_gear)
        
        # Check for shift conditions
        if self.shift_manager and not self.shift_in_progress:
            shift_needed = self.shift_manager.evaluate_shift(
                current_gear=self.vehicle.current_gear,
                engine_rpm=self.vehicle.current_engine_rpm,
                vehicle_speed=self.vehicle.current_speed,
                engine_load=self.vehicle.current_engine_torque / self.vehicle.engine.max_torque,
                throttle_position=self.current_inputs.throttle
            )
            
            if shift_needed is not None:
                self._queue_event(EventType.GEAR_SHIFT, {
                    "from_gear": self.vehicle.current_gear,
                    "to_gear": shift_needed,
                    "time": self.current_time
                })
        
        # Reset shift in progress flag if shift completed
        if self.shift_in_progress and self.current_time >= self.shift_end_time:
            self.shift_in_progress = False
        
        # Calculate acceleration
        self.vehicle.calculate_acceleration(
            throttle=self.current_inputs.throttle,
            brake=self.current_inputs.brake,
            gear=self.vehicle.current_gear,
            current_speed=self.vehicle.current_speed
        )
        
        # Update vehicle kinematics
        self.vehicle.update_vehicle_state(dt)
    
    def _update_track_position(self, dt: float):
        """
        Update vehicle position on track.
        
        Args:
            dt: Time step in seconds
        """
        # Skip if no track
        if not self.track:
            return
        
        # Calculate distance traveled in this step
        distance_traveled = self.vehicle.current_speed * dt
        
        # Update track position
        self.track_position += distance_traveled
        
        # Handle track wrapping for closed circuits
        if self.track.is_closed_circuit and self.track_position > self.track.total_length:
            self.track_position %= self.track.total_length
            
            # Trigger lap completion event
            lap_time = self.current_time - self.last_lap_time
            self._queue_event(EventType.LAP_COMPLETED, {
                "time": self.current_time,
                "lap_time": lap_time,
                "lap_number": self.lap_count + 1
            })
            
            self.last_lap_time = self.current_time
            self.lap_count += 1
        
        # Interpolate track properties at current position
        track_index = self._find_track_index(self.track_position)
        self.current_track_properties = {
            "curvature": self.track.curvature[track_index],
            "width": self.track.width[track_index]
        }
        
        # Add additional properties if available
        if hasattr(self.track, 'banking') and len(self.track.banking) > track_index:
            self.current_track_properties["banking"] = self.track.banking[track_index]
        
        if hasattr(self.track, 'elevation') and len(self.track.elevation) > track_index:
            self.current_track_properties["elevation"] = self.track.elevation[track_index]
        
        if hasattr(self.track, 'surface_friction') and len(self.track.surface_friction) > track_index:
            self.current_track_properties["surface_friction"] = self.track.surface_friction[track_index]
        
        # Check for segment transitions
        if self.track.segments:
            current_segment = self._get_current_segment(track_index)
            if current_segment != self.last_segment:
                self._queue_event(EventType.SEGMENT_TRANSITION, {
                    "from_segment": self.last_segment,
                    "to_segment": current_segment,
                    "time": self.current_time
                })
                self.last_segment = current_segment
    
    def _find_track_index(self, position: float) -> int:
        """
        Find track index closest to current position.
        
        Args:
            position: Current position along track in meters
            
        Returns:
            Index into track points array
        """
        if not self.track or not hasattr(self.track, 'distances') or len(self.track.distances) == 0:
            return 0
        
        # Find index where track distance is closest to current position
        diffs = np.abs(self.track.distances - position)
        return np.argmin(diffs)
    
    def _get_current_segment(self, track_index: int) -> Optional[TrackSegment]:
        """
        Get current track segment based on track index.
        
        Args:
            track_index: Index into track points array
            
        Returns:
            Current track segment or None
        """
        if not self.track or not self.track.segments:
            return None
        
        # Find segment containing track index
        for segment in self.track.segments:
            if segment.start_idx <= track_index <= segment.end_idx:
                return segment
        
        return None
    
    def _update_thermal_systems(self, dt: float):
        """
        Update thermal systems based on current conditions.
        
        Args:
            dt: Time step in seconds
        """
        if not self.vehicle or not hasattr(self.vehicle, 'cooling_system'):
            return
        
        # Get current thermal state
        if hasattr(self.vehicle.engine, 'get_temperature_state'):
            engine_temps = self.vehicle.engine.get_temperature_state()
            engine_temp = engine_temps.get('engine', 90.0)
            coolant_temp = engine_temps.get('coolant', 85.0)
            oil_temp = engine_temps.get('oil', 95.0)
        else:
            engine_temp = self.vehicle.engine.engine_temperature
            coolant_temp = self.vehicle.engine.coolant_temperature
            oil_temp = self.vehicle.engine.oil_temperature
        
        # Calculate heat input to cooling system
        engine_power = self.vehicle.engine.get_power(
            self.vehicle.current_engine_rpm,
            self.vehicle.engine.throttle_position
        ) * 1000  # Convert to watts
        
        # Estimate heat rejection (typically 60-70% of engine power)
        heat_input = engine_power * 0.65
        
        # Update cooling system
        self.vehicle.cooling_system.update_engine_state(
            engine_temp=engine_temp,
            engine_rpm=self.vehicle.current_engine_rpm,
            engine_load=self.vehicle.engine.throttle_position,
            engine_heat_input=heat_input
        )
        
        # Update cooling system with current vehicle speed
        self.vehicle.cooling_system.update_ambient_conditions(
            ambient_temp=self.environment.ambient_temperature,
            vehicle_speed=self.vehicle.current_speed
        )
        
        # Create automatic temperature control
        self.vehicle.cooling_system.create_automatic_control()
        
        # Update cooling system state
        self.vehicle.cooling_system.update_system_state(dt)
        
        # Check for thermal limits and generate events if needed
        if coolant_temp > self.thermal_limits.coolant_critical:
            self._queue_event(EventType.THERMAL_CRITICAL, {
                "system": "coolant",
                "temperature": coolant_temp,
                "time": self.current_time
            })
        elif coolant_temp > self.thermal_limits.coolant_warning:
            self._queue_event(EventType.THERMAL_WARNING, {
                "system": "coolant",
                "temperature": coolant_temp,
                "time": self.current_time
            })
        
        if engine_temp > self.thermal_limits.engine_critical:
            self._queue_event(EventType.THERMAL_CRITICAL, {
                "system": "engine",
                "temperature": engine_temp,
                "time": self.current_time
            })
        elif engine_temp > self.thermal_limits.engine_warning:
            self._queue_event(EventType.THERMAL_WARNING, {
                "system": "engine",
                "temperature": engine_temp,
                "time": self.current_time
            })
        
        if oil_temp > self.thermal_limits.oil_critical:
            self._queue_event(EventType.THERMAL_CRITICAL, {
                "system": "oil",
                "temperature": oil_temp,
                "time": self.current_time
            })
        elif oil_temp > self.thermal_limits.oil_warning:
            self._queue_event(EventType.THERMAL_WARNING, {
                "system": "oil",
                "temperature": oil_temp,
                "time": self.current_time
            })
    
    def _log_simulation_data(self):
        """Log current simulation state to data loggers."""
        # Get current state
        state = self.get_state()
        
        # Log data in each logger
        for logger in self.loggers:
            track_position = state.get('track', {}).get('position', 0.0)
            logger.log_data(self.current_time, track_position, state)
    
    def _compile_results(self):
        """Compile simulation results from logged data."""
        if not self.main_logger:
            return
        
        # Get data from main logger
        df = self.main_logger.get_data_frame()
        if df.empty:
            return
        
        # Add basic metrics to results
        self.results['simulation_time'] = self.current_time
        self.results['track_position'] = self.track_position
        self.results['lap_count'] = self.lap_count
        
        # Add lap times if available
        if 'lap_times' not in self.results and self.lap_count > 0:
            last_lap_time = self.current_time - self.last_lap_time
            self.results['lap_times'] = [last_lap_time]
            self.results['last_lap_time'] = last_lap_time
        
        # Extract maximum speed
        if 'vehicle.current_speed' in df:
            self.results['max_speed'] = df['vehicle.current_speed'].max()
        
        # Extract maximum acceleration
        if 'vehicle.current_acceleration' in df:
            self.results['max_acceleration'] = df['vehicle.current_acceleration'].max()
        
        # Add thermal metrics if available
        temp_cols = [col for col in df.columns if 'temp' in col.lower()]
        if temp_cols:
            thermal_metrics = {}
            for col in temp_cols:
                thermal_metrics[f'max_{col}'] = df[col].max()
                thermal_metrics[f'avg_{col}'] = df[col].mean()
            
            self.results['thermal_metrics'] = thermal_metrics
        
        # Add data frame to results
        self.results['data_frame'] = df
    
    def _handle_gear_shift_event(self, event_data: Dict):
        """
        Handle gear shift event.
        
        Args:
            event_data: Gear shift event details
        """
        if not self.vehicle or not hasattr(self.vehicle, 'drivetrain'):
            return
        
        from_gear = event_data.get('from_gear', self.vehicle.current_gear)
        to_gear = event_data.get('to_gear', from_gear)
        
        # If using CAS system
        if hasattr(self.vehicle, 'cas_system') and self.vehicle.cas_system:
            # Determine shift direction
            if to_gear > from_gear:
                direction_name = "UP"
            elif to_gear < from_gear:
                direction_name = "DOWN"
            else:
                return  # No actual gear change
            
            # Request shift from CAS system
            from ..transmission.cas_system import ShiftDirection
            shift_dir = getattr(ShiftDirection, direction_name)
            
            success = self.vehicle.cas_system.request_shift(shift_dir, to_gear)
            
            if success:
                # Apply CAS shift time penalty
                self.shift_in_progress = True
                self.shift_end_time = self.current_time + self.vehicle.cas_system.ignition_cut_time / 1000.0
                
                # Record shift
                if self.vehicle.shift_manager:
                    self.vehicle.shift_manager.record_shift(
                        from_gear, to_gear, 
                        self.vehicle.current_engine_rpm,
                        self.vehicle.current_speed
                    )
            
            return success
        else:
            # Basic gear change without CAS
            return self.vehicle.drivetrain.change_gear(to_gear)
    
    def _handle_lap_completed_event(self, event_data: Dict):
        """
        Handle lap completed event.
        
        Args:
            event_data: Lap completed event details
        """
        lap_time = event_data.get('lap_time', 0.0)
        lap_number = event_data.get('lap_number', self.lap_count)
        
        # Store lap time in results
        if 'lap_times' not in self.results:
            self.results['lap_times'] = []
        
        self.results['lap_times'].append(lap_time)
        self.results['last_lap_time'] = lap_time
        
        logger.info(f"Lap {lap_number} completed in {lap_time:.3f}s")
    
    def _handle_thermal_warning_event(self, event_data: Dict):
        """
        Handle thermal warning event.
        
        Args:
            event_data: Thermal warning event details
        """
        system = event_data.get('system', 'unknown')
        temperature = event_data.get('temperature', 0.0)
        
        # Add to results
        if 'thermal_warnings' not in self.results:
            self.results['thermal_warnings'] = []
        
        self.results['thermal_warnings'].append({
            'time': self.current_time,
            'system': system,
            'temperature': temperature
        })
        
        logger.warning(f"Thermal warning: {system} temperature {temperature:.1f}°C")
    
    def _handle_thermal_critical_event(self, event_data: Dict):
        """
        Handle thermal critical event.
        
        Args:
            event_data: Thermal critical event details
        """
        system = event_data.get('system', 'unknown')
        temperature = event_data.get('temperature', 0.0)
        
        # Add to results
        if 'thermal_critical' not in self.results:
            self.results['thermal_critical'] = []
        
        self.results['thermal_critical'].append({
            'time': self.current_time,
            'system': system,
            'temperature': temperature
        })
        
        logger.error(f"Thermal critical: {system} temperature {temperature:.1f}°C")
        
        # In critical situations, reduce engine output
        if self.vehicle and hasattr(self.vehicle.engine, 'throttle_position'):
            # Limit throttle to 50%
            self.current_inputs.throttle = min(self.current_inputs.throttle, 0.5)
            self.vehicle.engine.throttle_position = min(self.vehicle.engine.throttle_position, 0.5)
    
    def _get_parameter_value(self, param: str) -> Optional[float]:
        """
        Get the value of a parameter.
        
        Args:
            param: Parameter name (can use dot notation for nested parameters)
            
        Returns:
            Parameter value or None if not found
        """
        # Handle nested parameters with dot notation
        parts = param.split('.')
        obj = self
        
        for part in parts:
            if hasattr(obj, part):
                obj = getattr(obj, part)
            elif isinstance(obj, dict) and part in obj:
                obj = obj[part]
            else:
                return None
        
        if isinstance(obj, (int, float)):
            return float(obj)
        
        return None
    
    def _set_parameter_value(self, param: str, value: float) -> bool:
        """
        Set the value of a parameter.
        
        Args:
            param: Parameter name (can use dot notation for nested parameters)
            value: Parameter value
            
        Returns:
            True if parameter was set, False otherwise
        """
        # Handle nested parameters with dot notation
        parts = param.split('.')
        obj = self
        
        # Navigate to parent object
        for part in parts[:-1]:
            if hasattr(obj, part):
                obj = getattr(obj, part)
            elif isinstance(obj, dict) and part in obj:
                obj = obj[part]
            else:
                return False
        
        # Set attribute on parent object
        last_part = parts[-1]
        try:
            if hasattr(obj, last_part):
                setattr(obj, last_part, value)
                return True
            elif isinstance(obj, dict) and last_part in obj:
                obj[last_part] = value
                return True
        except Exception:
            return False
        
        return False