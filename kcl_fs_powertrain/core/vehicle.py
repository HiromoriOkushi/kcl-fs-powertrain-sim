"""
Vehicle module for Formula Student powertrain simulation.

This module defines the Vehicle class, which integrates all powertrain components
(engine, transmission, thermal systems) and provides methods for simulating
vehicle dynamics, calculating performance metrics, and visualizing results.

The Vehicle class serves as the central point of the powertrain simulation,
connecting all subsystems and enabling comprehensive vehicle performance analysis.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Callable
import yaml
import logging

# Import powertrain components
from ..engine import MotorcycleEngine, TorqueCurve
from ..transmission import DrivetrainSystem, CASSystem, StrategyManager
from ..thermal import CoolingSystem, DualSidePodSystem, RearRadiatorSystem, CoolingAssistSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("Vehicle")


class Vehicle:
    """
    Complete vehicle model for Formula Student powertrain simulation.
    
    This class integrates all powertrain components (engine, transmission, cooling)
    into a complete vehicle model and provides methods for simulating vehicle
    dynamics, calculating performance metrics, and analyzing results.
    """
    
    def __init__(self, 
                 config_path: Optional[str] = None,
                 engine: Optional[MotorcycleEngine] = None,
                 drivetrain: Optional[DrivetrainSystem] = None,
                 cooling_system: Optional[CoolingSystem] = None,
                 team_name: str = "KCL Formula Student"):
        """
        Initialize the vehicle model.
        
        Args:
            config_path: Path to vehicle configuration file
            engine: Optional pre-configured engine
            drivetrain: Optional pre-configured drivetrain
            cooling_system: Optional pre-configured cooling system
            team_name: Team name for identification
        """
        self.team_name = team_name
        
        # Default vehicle parameters
        self.mass = 230.0  # kg, typical for FS car with driver
        self.frontal_area = 1.0  # m²
        self.drag_coefficient = 0.8  # Typical for open-wheel race car
        self.lift_coefficient = -1.5  # Negative for downforce
        self.rolling_resistance = 0.015  # Typical for racing tires
        
        # Tire parameters
        self.tire_radius = 0.2286  # m (9-inch for 13-inch wheels)
        self.tire_width = 0.18  # m
        self.tire_rolling_circumference = 2 * np.pi * self.tire_radius
        
        # Performance parameters
        self.weight_distribution_front = 0.45  # 45% front, 55% rear
        self.wheelbase = 1.6  # m
        self.track_width_front = 1.2  # m
        self.track_width_rear = 1.15  # m
        self.cg_height = 0.3  # m
        
        # Initialize subsystems
        self.engine = engine
        self.drivetrain = drivetrain
        self.cooling_system = cooling_system
        self.shift_manager = None
        self.cas_system = None
        self.side_pods = None
        self.rear_radiator = None
        self.cooling_assist = None
        
        # Current state
        self.current_gear = 0  # Neutral
        self.current_speed = 0.0  # m/s
        self.current_acceleration = 0.0  # m/s²
        self.current_position = 0.0  # m
        self.current_engine_rpm = 0.0  # RPM
        self.current_engine_torque = 0.0  # Nm
        self.current_wheel_torque = 0.0  # Nm
        self.current_throttle = 0.0  # 0-1
        self.current_brake = 0.0  # 0-1
        
        # Load configuration if provided
        if config_path:
            self.load_config(config_path)
        
        # Create default subsystems if not provided
        self._initialize_subsystems()
        
        logger.info(f"{team_name} Formula Student vehicle initialized")
    
    def load_config(self, config_path: str):
        """
        Load vehicle configuration from YAML file.
        
        Args:
            config_path: Path to vehicle configuration file
        """
        if not os.path.exists(config_path):
            logger.warning(f"Configuration file not found: {config_path}")
            return
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Load vehicle parameters
            if 'vehicle' in config:
                vehicle_config = config['vehicle']
                self.mass = vehicle_config.get('mass', self.mass)
                self.frontal_area = vehicle_config.get('frontal_area', self.frontal_area)
                self.drag_coefficient = vehicle_config.get('drag_coefficient', self.drag_coefficient)
                self.lift_coefficient = vehicle_config.get('lift_coefficient', self.lift_coefficient)
                self.rolling_resistance = vehicle_config.get('rolling_resistance', self.rolling_resistance)
                self.weight_distribution_front = vehicle_config.get('weight_distribution_front', self.weight_distribution_front)
                self.wheelbase = vehicle_config.get('wheelbase', self.wheelbase)
                self.track_width_front = vehicle_config.get('track_width_front', self.track_width_front)
                self.track_width_rear = vehicle_config.get('track_width_rear', self.track_width_rear)
                self.cg_height = vehicle_config.get('cg_height', self.cg_height)
            
            # Load tire parameters
            if 'tires' in config:
                tire_config = config['tires']
                self.tire_radius = tire_config.get('radius', self.tire_radius)
                self.tire_width = tire_config.get('width', self.tire_width)
                self.tire_rolling_circumference = 2 * np.pi * self.tire_radius
            
            # Store full configuration for subsystem initialization
            self.config = config
            
            logger.info(f"Vehicle configuration loaded from {config_path}")
        
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
    
    def _initialize_subsystems(self):
        """Initialize vehicle subsystems if not already provided."""
        # Initialize engine if not provided
        if self.engine is None:
            try:
                # Try to create engine from config
                if hasattr(self, 'config') and 'engine' in self.config:
                    engine_config = self.config['engine']
                    if 'config_path' in engine_config:
                        self.engine = MotorcycleEngine(config_path=engine_config['config_path'])
                    else:
                        self.engine = MotorcycleEngine(engine_params=engine_config)
                else:
                    # Create default engine
                    from ..engine import MotorcycleEngine
                    config_path = os.path.join("configs", "engine", "cbr600f4i.yaml")
                    if os.path.exists(config_path):
                        self.engine = MotorcycleEngine(config_path=config_path)
                    else:
                        self.engine = MotorcycleEngine()  # Default parameters
                
                logger.info("Engine initialized")
            
            except Exception as e:
                logger.error(f"Error initializing engine: {str(e)}")
                self.engine = MotorcycleEngine()  # Create with defaults
        
        # Initialize drivetrain if not provided
        if self.drivetrain is None:
            try:
                # Try to create drivetrain from config
                from ..transmission import Transmission, FinalDrive, Differential, DrivetrainSystem
                
                # Default gear ratios (Honda CBR600F4i)
                gear_ratios = [2.750, 2.000, 1.667, 1.444, 1.304, 1.208]
                
                if hasattr(self, 'config') and 'transmission' in self.config:
                    trans_config = self.config['transmission']
                    
                    # Get gear ratios
                    if 'gear_ratios' in trans_config:
                        gear_ratios = trans_config['gear_ratios']
                    
                    # Create transmission
                    transmission = Transmission(gear_ratios)
                    
                    # Create final drive
                    fd_config = trans_config.get('final_drive', {})
                    drive_teeth = fd_config.get('drive_sprocket_teeth', 14)
                    driven_teeth = fd_config.get('driven_sprocket_teeth', 53)
                    final_drive = FinalDrive(drive_teeth, driven_teeth)
                    
                    # Create differential
                    diff_config = trans_config.get('differential', {})
                    locked = diff_config.get('locked', True)
                    differential = Differential(locked=locked)
                    
                    # Create drivetrain
                    self.drivetrain = DrivetrainSystem(
                        transmission, final_drive, differential, 
                        wheel_radius=self.tire_radius
                    )
                else:
                    # Create default drivetrain
                    transmission = Transmission(gear_ratios)
                    final_drive = FinalDrive(14, 53)  # Common for FS
                    differential = Differential(locked=True)  # Solid axle for FS
                    
                    self.drivetrain = DrivetrainSystem(
                        transmission, final_drive, differential, 
                        wheel_radius=self.tire_radius
                    )
                
                logger.info("Drivetrain initialized")
                
                # Initialize shifting systems
                self._initialize_shifting_systems()
                
            except Exception as e:
                logger.error(f"Error initializing drivetrain: {str(e)}")
                # Create minimal drivetrain with defaults
                from ..transmission import Transmission, FinalDrive, Differential, DrivetrainSystem
                transmission = Transmission([2.750, 2.000, 1.667, 1.444, 1.304, 1.208])
                final_drive = FinalDrive(14, 53)
                differential = Differential(locked=True)
                self.drivetrain = DrivetrainSystem(
                    transmission, final_drive, differential, 
                    wheel_radius=self.tire_radius
                )
        
        # Initialize cooling system if not provided
        if self.cooling_system is None:
            try:
                # Try to create cooling system from config
                from ..thermal import create_formula_student_cooling_system
                self.cooling_system = create_formula_student_cooling_system()
                
                # Try to initialize side pods
                self._initialize_thermal_systems()
                
                logger.info("Cooling system initialized")
            
            except Exception as e:
                logger.error(f"Error initializing cooling system: {str(e)}")
                # Create default cooling system
                from ..thermal import create_formula_student_cooling_system
                self.cooling_system = create_formula_student_cooling_system()
    
    def _initialize_shifting_systems(self):
        """Initialize shifting strategy and CAS system."""
        try:
            from ..transmission import (
                create_formula_student_strategies, CASSystem,
                MaxAccelerationStrategy
            )
            
            # Initialize shift strategy manager
            if self.engine and self.drivetrain:
                # Create strategies
                self.shift_manager = create_formula_student_strategies(
                    self.engine.redline,
                    self.engine.max_power_rpm,
                    self.engine.max_torque_rpm,
                    self.drivetrain.transmission.gear_ratios,
                    self.drivetrain.wheel_radius,
                    self.mass
                )
                
                # Create CAS system
                self.cas_system = CASSystem(
                    self.drivetrain.transmission.gear_ratios,
                    self.engine
                )
                
                logger.info("Shifting systems initialized")
        except Exception as e:
            logger.error(f"Error initializing shifting systems: {str(e)}")
    
    def _initialize_thermal_systems(self):
        """Initialize additional thermal systems (side pods, rear radiator, etc.)."""
        try:
            from ..thermal import (
                create_standard_side_pod_system,
                create_default_rear_radiator_system,
                create_default_cooling_assist_system
            )
            
            # Create side pods
            self.side_pods = create_standard_side_pod_system()
            
            # Create rear radiator
            self.rear_radiator = create_default_rear_radiator_system()
            
            # Create cooling assist
            self.cooling_assist = create_default_cooling_assist_system()
            
            logger.info("Additional thermal systems initialized")
        except Exception as e:
            logger.error(f"Error initializing additional thermal systems: {str(e)}")
    
    def update_engine_state(self, throttle: float, engine_rpm: float, ambient_temp: float = 25.0):
        """
        Update engine state based on inputs.
        
        Args:
            throttle: Throttle position (0-1)
            engine_rpm: Engine speed in RPM
            ambient_temp: Ambient temperature in °C
        """
        if self.engine is None:
            return
        
        # Update engine state
        self.engine.throttle_position = max(0.0, min(1.0, throttle))
        self.engine.current_rpm = max(self.engine.idle_rpm, min(engine_rpm, self.engine.redline))
        
        # Calculate torque
        self.current_engine_torque = self.engine.get_torque(
            self.engine.current_rpm, 
            self.engine.throttle_position,
            self.engine.engine_temperature
        )
        
        # Update engine thermal state if cooling system exists
        if self.cooling_system:
            # Get cooling effectiveness (simplified)
            cooling_effectiveness = 0.8
            
            # Update engine temperatures
            self.engine.update_thermal_state(
                self.engine.current_rpm,
                self.engine.throttle_position,
                ambient_temp,
                cooling_effectiveness,
                0.1  # Small time step
            )
    
    def update_drivetrain_state(self, gear: int):
        """
        Update drivetrain state based on inputs.
        
        Args:
            gear: Target gear number
        """
        if self.drivetrain is None:
            return
        
        # Update gear if changed
        if gear != self.current_gear:
            # Change gear in drivetrain
            self.drivetrain.change_gear(gear)
            self.current_gear = gear
        
        # Calculate wheel torque from engine torque
        self.current_wheel_torque = self.drivetrain.calculate_wheel_torque(
            self.current_engine_torque, self.current_gear
        )
    
    def update_cooling_system(self, 
                            coolant_temp: float = 90.0, 
                            ambient_temp: float = 25.0, 
                            vehicle_speed: float = 0.0):
        """
        Update cooling system state.
        
        Args:
            coolant_temp: Coolant temperature in °C
            ambient_temp: Ambient temperature in °C
            vehicle_speed: Vehicle speed in m/s
        """
        if self.cooling_system is None:
            return
        
        # Update cooling system
        self.cooling_system.update_ambient_conditions(ambient_temp, vehicle_speed)
        
        # Update engine state in cooling system
        self.cooling_system.update_engine_state(
            self.engine.engine_temperature,
            self.engine.current_rpm,
            self.engine.throttle_position,
            30000.0  # Estimated heat input (W) - would be calculated from engine
        )
        
        # Create automatic control
        self.cooling_system.create_automatic_control(
            target_temp=90.0,
            hysteresis=5.0
        )
        
        # Update system state
        self.cooling_system.update_system_state(0.1)  # Small time step
    
    def calculate_acceleration(self, 
                             throttle: float,
                             brake: float = 0.0,
                             gear: Optional[int] = None,
                             current_speed: Optional[float] = None) -> float:
        """
        Calculate vehicle acceleration based on current state.
        
        Args:
            throttle: Throttle position (0-1)
            brake: Brake position (0-1)
            gear: Gear number (if None, uses current gear)
            current_speed: Current speed in m/s (if None, uses current speed)
            
        Returns:
            Acceleration in m/s²
        """
        # Use provided values or current state
        if gear is None:
            gear = self.current_gear
        
        if current_speed is None:
            current_speed = self.current_speed
        
        # Store inputs
        self.current_throttle = max(0.0, min(1.0, throttle))
        self.current_brake = max(0.0, min(1.0, brake))
        
        # Calculate engine RPM from vehicle speed if in gear
        if gear > 0:
            self.current_engine_rpm = self.drivetrain.calculate_engine_speed(current_speed, gear)
        else:
            # In neutral, engine RPM is decoupled from wheels
            self.current_engine_rpm = self.engine.idle_rpm
        
        # Update engine state
        self.update_engine_state(throttle, self.current_engine_rpm)
        
        # Update drivetrain state
        self.update_drivetrain_state(gear)
        
        # Calculate tractive force
        if gear > 0:
            tractive_force = self.current_wheel_torque / self.tire_radius
        else:
            tractive_force = 0.0  # No force in neutral
        
        # Calculate resistance forces
        # Rolling resistance
        rolling_resistance_force = self.mass * 9.81 * self.rolling_resistance
        
        # Aerodynamic drag
        air_density = 1.225  # kg/m³ at sea level
        aero_drag = 0.5 * air_density * self.drag_coefficient * self.frontal_area * current_speed**2
        
        # Aerodynamic downforce (affects rolling resistance)
        downforce = 0.5 * air_density * -self.lift_coefficient * self.frontal_area * current_speed**2
        
        # Adjust rolling resistance with downforce
        adjusted_rolling_resistance = rolling_resistance_force + (downforce * self.rolling_resistance)
        
        # Braking force
        max_braking_force = (self.mass * 9.81 + downforce) * 1.5  # 1.5g deceleration
        braking_force = max_braking_force * self.current_brake
        
        # Net force
        net_force = tractive_force - adjusted_rolling_resistance - aero_drag - braking_force
        
        # Calculate acceleration (F = ma)
        acceleration = net_force / self.mass
        
        # Store current acceleration
        self.current_acceleration = acceleration
        
        return acceleration
    
    def update_vehicle_state(self, dt: float = 0.01):
        """
        Update vehicle state over a time step.
        
        Args:
            dt: Time step in seconds
        """
        # Update speed based on acceleration
        self.current_speed += self.current_acceleration * dt
        self.current_speed = max(0.0, self.current_speed)  # Prevent negative speed
        
        # Update position
        self.current_position += self.current_speed * dt
        
        # Update engine RPM based on speed
        if self.current_gear > 0:
            self.current_engine_rpm = self.drivetrain.calculate_engine_speed(
                self.current_speed, self.current_gear
            )
        
        # Update cooling system
        self.update_cooling_system(
            coolant_temp=self.engine.coolant_temperature if self.engine else 90.0,
            ambient_temp=25.0,  # Ambient temp
            vehicle_speed=self.current_speed
        )
    
    def simulate_acceleration_run(self, 
                               distance: float = 75.0,
                               max_time: float = 10.0,
                               dt: float = 0.01) -> Dict:
        """
        Simulate an acceleration run over a specified distance.
        
        Args:
            distance: Distance in meters
            max_time: Maximum simulation time in seconds
            dt: Time step in seconds
            
        Returns:
            Dictionary with simulation results
        """
        # Reset vehicle state
        self.current_speed = 0.0
        self.current_position = 0.0
        self.current_acceleration = 0.0
        self.current_gear = 1  # Start in first gear
        self.current_engine_rpm = self.engine.idle_rpm
        
        # Initialize results storage
        time_points = [0.0]
        speed_points = [0.0]
        position_points = [0.0]
        acceleration_points = [0.0]
        rpm_points = [self.current_engine_rpm]
        gear_points = [self.current_gear]
        
        # Shifting parameters
        shift_rpm = self.engine.max_power_rpm * 0.98  # Shift at 98% of max power RPM
        
        # Initialize engine
        self.update_engine_state(1.0, self.current_engine_rpm)  # Full throttle
        
        # Simulation loop
        current_time = 0.0
        
        while current_time < max_time and self.current_position < distance:
            # Apply full throttle
            throttle = 1.0
            
            # Check for gear shift
            if self.current_engine_rpm > shift_rpm and self.current_gear < self.drivetrain.transmission.num_gears:
                self.current_gear += 1
            
            # Calculate acceleration
            self.calculate_acceleration(throttle, 0.0, self.current_gear, self.current_speed)
            
            # Update vehicle state
            self.update_vehicle_state(dt)
            
            # Update time
            current_time += dt
            
            # Store results
            time_points.append(current_time)
            speed_points.append(self.current_speed)
            position_points.append(self.current_position)
            acceleration_points.append(self.current_acceleration)
            rpm_points.append(self.current_engine_rpm)
            gear_points.append(self.current_gear)
        
        # Calculate results
        finish_time = None
        finish_speed = None
        
        # Interpolate to find exact finish time
        if self.current_position >= distance:
            idx = np.searchsorted(position_points, distance)
            if idx > 0 and idx < len(time_points):
                # Linear interpolation
                t0, t1 = time_points[idx-1], time_points[idx]
                p0, p1 = position_points[idx-1], position_points[idx]
                finish_time = t0 + (t1 - t0) * (distance - p0) / (p1 - p0)
                
                # Interpolate finish speed
                s0, s1 = speed_points[idx-1], speed_points[idx]
                finish_speed = s0 + (s1 - s0) * (finish_time - t0) / (t1 - t0)
        
        # Calculate 0-60 mph time
        time_to_60mph = None
        mps_to_mph = 2.23694  # m/s to mph conversion
        target_speed = 60.0 / mps_to_mph  # 60 mph in m/s
        
        if max(speed_points) >= target_speed:
            idx = np.searchsorted(speed_points, target_speed)
            if idx > 0 and idx < len(time_points):
                # Linear interpolation
                t0, t1 = time_points[idx-1], time_points[idx]
                s0, s1 = speed_points[idx-1], speed_points[idx]
                time_to_60mph = t0 + (t1 - t0) * (target_speed - s0) / (s1 - s0)
        
        # Compile results
        results = {
            'time': np.array(time_points),
            'speed': np.array(speed_points),
            'position': np.array(position_points),
            'acceleration': np.array(acceleration_points),
            'engine_rpm': np.array(rpm_points),
            'gear': np.array(gear_points),
            'finish_time': finish_time,
            'finish_speed': finish_speed,
            'time_to_60mph': time_to_60mph
        }
        
        return results
    
    def simulate_lap(self, 
                   track_data: Dict,
                   max_time: float = 120.0,
                   dt: float = 0.01) -> Dict:
        """
        Simulate a lap around a track.
        
        Args:
            track_data: Track data with distance, curvature, etc.
            max_time: Maximum simulation time in seconds
            dt: Time step in seconds
            
        Returns:
            Dictionary with simulation results
        """
        # This is a simplified implementation - a full lap simulation would be more complex
        # and would include lateral dynamics, line optimization, etc.
        
        # Extract track data
        distance = track_data.get('distance', np.array([0.0]))
        curvature = track_data.get('curvature', np.array([0.0]))
        
        # Reset vehicle state
        self.current_speed = 0.0
        self.current_position = 0.0
        self.current_acceleration = 0.0
        self.current_gear = 1  # Start in first gear
        self.current_engine_rpm = self.engine.idle_rpm
        
        # Initialize results storage
        time_points = [0.0]
        speed_points = [0.0]
        position_points = [0.0]
        acceleration_points = [0.0]
        rpm_points = [self.current_engine_rpm]
        gear_points = [self.current_gear]
        
        # Initialize engine
        self.update_engine_state(1.0, self.current_engine_rpm)  # Full throttle
        
        # Simulation loop
        current_time = 0.0
        track_length = distance[-1]
        lap_completed = False
        
        while current_time < max_time and not lap_completed:
            # Get current track position (wrap around track length)
            track_position = self.current_position % track_length
            
            # Interpolate curvature at current position
            current_curvature = np.interp(track_position, distance, curvature)
            
            # Calculate target speed based on curvature (simplified)
            # v² = a_lat / r where r = 1/curvature
            max_lateral_accel = 1.5 * 9.81  # 1.5g lateral acceleration
            
            if abs(current_curvature) > 0.001:  # Non-straight section
                target_speed = np.sqrt(max_lateral_accel / abs(current_curvature))
            else:
                target_speed = float('inf')  # No speed limit on straight
            
            # Determine throttle and brake inputs
            if self.current_speed > target_speed * 1.05:
                # Need to brake
                throttle = 0.0
                brake = min(1.0, (self.current_speed - target_speed) / 10.0)
            else:
                # Can accelerate or maintain speed
                throttle = 1.0
                brake = 0.0
            
            # Determine optimal gear
            if self.shift_manager and self.drivetrain:
                # Ask shift manager for recommendation
                vehicle_state = {
                    'gear_ratios': self.drivetrain.transmission.gear_ratios,
                    'current_gear': self.current_gear,
                    'wheel_radius': self.drivetrain.wheel_radius
                }
                self.shift_manager.update_vehicle_state(vehicle_state)
                
                if self.current_gear > 0:
                    shift_direction = self.shift_manager.active_strategy.should_shift(
                        self.current_engine_rpm, throttle
                    )
                    
                    if shift_direction:
                        if shift_direction.name == "UP" and self.current_gear < self.drivetrain.transmission.num_gears:
                            self.current_gear += 1
                        elif shift_direction.name == "DOWN" and self.current_gear > 1:
                            self.current_gear -= 1
            else:
                # Simple shift strategy
                if self.current_engine_rpm > self.engine.max_power_rpm * 0.98 and self.current_gear < self.drivetrain.transmission.num_gears:
                    self.current_gear += 1
                elif self.current_engine_rpm < self.engine.max_torque_rpm * 0.8 and self.current_gear > 1:
                    self.current_gear -= 1
            
            # Calculate acceleration
            self.calculate_acceleration(throttle, brake, self.current_gear, self.current_speed)
            
            # Update vehicle state
            self.update_vehicle_state(dt)
            
            # Check if lap completed
            if self.current_position >= track_length:
                lap_completed = True
            
            # Update time
            current_time += dt
            
            # Store results
            time_points.append(current_time)
            speed_points.append(self.current_speed)
            position_points.append(self.current_position)
            acceleration_points.append(self.current_acceleration)
            rpm_points.append(self.current_engine_rpm)
            gear_points.append(self.current_gear)
        
        # Compile results
        results = {
            'time': np.array(time_points),
            'speed': np.array(speed_points),
            'position': np.array(position_points),
            'acceleration': np.array(acceleration_points),
            'engine_rpm': np.array(rpm_points),
            'gear': np.array(gear_points),
            'lap_time': current_time if lap_completed else None,
            'lap_completed': lap_completed
        }
        
        return results
    
    def calculate_weight_transfer(self, acceleration: float) -> Tuple[float, float]:
        """
        Calculate weight transfer during acceleration/braking.
        
        Args:
            acceleration: Longitudinal acceleration in m/s²
            
        Returns:
            Tuple of (front_weight_fraction, rear_weight_fraction)
        """
        # Static weight distribution
        static_front_weight = self.mass * self.weight_distribution_front
        static_rear_weight = self.mass * (1 - self.weight_distribution_front)
        
        # Weight transfer during acceleration/braking
        weight_transfer = self.mass * acceleration * self.cg_height / self.wheelbase
        
        # New weight distribution
        front_weight = static_front_weight - weight_transfer
        rear_weight = static_rear_weight + weight_transfer
        
        # Convert to fractions
        total_weight = front_weight + rear_weight
        front_fraction = front_weight / total_weight
        rear_fraction = rear_weight / total_weight
        
        return front_fraction, rear_fraction
    
    def plot_acceleration_results(self, results: Dict, save_path: Optional[str] = None):
        """
        Plot acceleration simulation results.
        
        Args:
            results: Results from simulate_acceleration_run
            save_path: Optional path to save the plot
        """
        # Extract data
        time = results['time']
        speed = results['speed']
        acceleration = results['acceleration']
        rpm = results['engine_rpm']
        gear = results['gear']
        
        # Convert speed to mph for display
        speed_mph = speed * 2.23694
        
        # Create figure with multiple subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        
        # Plot speed
        ax1.plot(time, speed_mph, 'b-', linewidth=2)
        ax1.set_ylabel('Speed (mph)')
        ax1.set_title('Acceleration Run Results')
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Add 0-60 mph time if available
        if results['time_to_60mph'] is not None:
            ax1.axhline(y=60, color='r', linestyle='--', alpha=0.5)
            ax1.axvline(x=results['time_to_60mph'], color='r', linestyle='--', alpha=0.5)
            ax1.text(
                results['time_to_60mph'] + 0.1, 
                62, 
                f"0-60 mph: {results['time_to_60mph']:.2f}s",
                color='r',
                fontweight='bold'
            )
        
        # Plot acceleration
        ax2.plot(time, acceleration, 'g-', linewidth=2)
        ax2.set_ylabel('Acceleration (m/s²)')
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        # Plot RPM and gear
        color1 = 'tab:blue'
        ax3.set_ylabel('Engine RPM', color=color1)
        ax3.plot(time, rpm, color=color1, linewidth=2)
        ax3.tick_params(axis='y', labelcolor=color1)
        ax3.grid(True, linestyle='--', alpha=0.7)
        
        color2 = 'tab:red'
        ax3_twin = ax3.twinx()
        ax3_twin.set_ylabel('Gear', color=color2)
        ax3_twin.step(time, gear, color=color2, linewidth=2, where='post')
        ax3_twin.tick_params(axis='y', labelcolor=color2)
        ax3_twin.set_yticks(range(0, self.drivetrain.transmission.num_gears + 1))
        
        ax3.set_xlabel('Time (s)')
        
        # Add summary text
        if results['finish_time'] is not None:
            plt.figtext(
                0.5, 0.01,
                f"Distance: 75m, Time: {results['finish_time']:.2f}s, Final Speed: {results['finish_speed'] * 2.23694:.1f} mph",
                ha='center',
                fontsize=12,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray')
            )
        
        plt.tight_layout(rect=[0, 0.03, 1, 1])
        
        # Save plot if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        plt.show()
    
    def plot_lap_results(self, results: Dict, track_data: Dict, save_path: Optional[str] = None):
        """
        Plot lap simulation results.
        
        Args:
            results: Results from simulate_lap
            track_data: Track data with distance, curvature, etc.
            save_path: Optional path to save the plot
        """
        # Extract data
        time = results['time']
        speed = results['speed']
        position = results['position']
        rpm = results['engine_rpm']
        gear = results['gear']
        
        # Extract track data
        distance = track_data.get('distance', np.array([0.0]))
        curvature = track_data.get('curvature', np.array([0.0]))
        
        # Convert speed to mph for display
        speed_mph = speed * 2.23694
        
        # Create figure with multiple subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 14), sharex=True)
        
        # Plot speed vs. distance
        ax1.plot(position, speed_mph, 'b-', linewidth=2)
        ax1.set_ylabel('Speed (mph)')
        ax1.set_title('Lap Simulation Results')
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Plot curvature
        # Repeat track data if lap is longer than track length
        track_length = distance[-1]
        repeated_distance = np.array([])
        repeated_curvature = np.array([])
        
        max_pos = max(position)
        repetitions = int(np.ceil(max_pos / track_length))
        
        for i in range(repetitions):
            repeated_distance = np.append(repeated_distance, distance + i * track_length)
            repeated_curvature = np.append(repeated_curvature, curvature)
        
        # Plot track curvature
        ax2.plot(repeated_distance, repeated_curvature, 'r-', linewidth=2)
        ax2.set_ylabel('Track Curvature (1/m)')
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        # Plot RPM and gear
        color1 = 'tab:blue'
        ax3.set_ylabel('Engine RPM', color=color1)
        ax3.plot(position, rpm, color=color1, linewidth=2)
        ax3.tick_params(axis='y', labelcolor=color1)
        ax3.grid(True, linestyle='--', alpha=0.7)
        
        color2 = 'tab:red'
        ax3_twin = ax3.twinx()
        ax3_twin.set_ylabel('Gear', color=color2)
        ax3_twin.step(position, gear, color=color2, linewidth=2, where='post')
        ax3_twin.tick_params(axis='y', labelcolor=color2)
        ax3_twin.set_yticks(range(0, self.drivetrain.transmission.num_gears + 1))
        
        ax3.set_xlabel('Distance (m)')
        
        # Add lap time if available
        if results['lap_time'] is not None:
            plt.figtext(
                0.5, 0.01,
                f"Lap Time: {results['lap_time']:.2f}s, Average Speed: {np.mean(speed_mph):.1f} mph",
                ha='center',
                fontsize=12,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray')
            )
        
        plt.tight_layout(rect=[0, 0.03, 1, 1])
        
        # Save plot if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        plt.show()
    
    def get_vehicle_specs(self) -> Dict:
        """
        Get comprehensive vehicle specifications.
        
        Returns:
            Dictionary with vehicle specifications
        """
        specs = {
            'team_name': self.team_name,
            'vehicle': {
                'mass': self.mass,
                'frontal_area': self.frontal_area,
                'drag_coefficient': self.drag_coefficient,
                'lift_coefficient': self.lift_coefficient,
                'rolling_resistance': self.rolling_resistance,
                'weight_distribution_front': self.weight_distribution_front,
                'wheelbase': self.wheelbase,
                'track_width_front': self.track_width_front,
                'track_width_rear': self.track_width_rear,
                'cg_height': self.cg_height
            },
            'tires': {
                'radius': self.tire_radius,
                'width': self.tire_width,
                'rolling_circumference': self.tire_rolling_circumference
            }
        }
        
        # Add engine specs if available
        if self.engine:
            specs['engine'] = self.engine.get_engine_specs()
        
        # Add drivetrain specs if available
        if self.drivetrain:
            specs['drivetrain'] = self.drivetrain.get_drivetrain_specs()
        
        # Add cooling system specs if available
        if self.cooling_system:
            specs['cooling'] = self.cooling_system.get_system_specs()
        
        return specs


def create_formula_student_vehicle() -> Vehicle:
    """
    Create a default Formula Student vehicle configuration.
    
    Returns:
        Configured Vehicle
    """
    # Create engine
    from ..engine import MotorcycleEngine
    engine_config_path = os.path.join("configs", "engine", "cbr600f4i.yaml")
    engine = MotorcycleEngine(config_path=engine_config_path)
    
    # Create drivetrain
    from ..transmission import (
        Transmission, FinalDrive, Differential, DrivetrainSystem
    )
    
    gear_ratios = [2.750, 2.000, 1.667, 1.444, 1.304, 1.208]
    transmission = Transmission(gear_ratios)
    final_drive = FinalDrive(14, 53)  # 14:53 sprocket ratio
    differential = Differential(locked=True)  # Solid axle
    
    # Standard tire radius for Formula Student (13-inch wheels)
    tire_radius = 0.2286  # m (9-inch for 13-inch wheels)
    
    drivetrain = DrivetrainSystem(
        transmission, final_drive, differential, wheel_radius=tire_radius
    )
    
    # Create cooling system
    from ..thermal import create_formula_student_cooling_system
    cooling_system = create_formula_student_cooling_system()
    
    # Create vehicle
    vehicle = Vehicle(
        engine=engine,
        drivetrain=drivetrain,
        cooling_system=cooling_system,
        team_name="KCL Formula Student"
    )
    
    return vehicle
def simulate_skidpad(self, circle_radius: float = 8.5, max_time: float = 15.0, dt: float = 0.01) -> Dict:
    """
    Simulate a skidpad run (figure-8 pattern).
    
    Args:
        circle_radius: Radius of skidpad circle in meters
        max_time: Maximum simulation time in seconds
        dt: Time step in seconds
        
    Returns:
        Dictionary with simulation results
    """
    # Reset vehicle state
    self.current_speed = 0.0
    self.current_position = 0.0
    self.current_acceleration = 0.0
    self.current_gear = 1  # Start in first gear
    self.current_engine_rpm = self.engine.idle_rpm
    
    # Initialize results storage
    time_points = [0.0]
    speed_points = [0.0]
    lateral_accel_points = [0.0]
    position_points = [0.0]
    rpm_points = [self.current_engine_rpm]
    gear_points = [self.current_gear]
    
    # First corner entry phase - acceleration
    corner_entry_time = 3.0
    steady_state_laps = 2
    
    # Circle parameters
    circle_circumference = 2 * np.pi * circle_radius
    
    # Simulation loop
    current_time = 0.0
    phase = "acceleration"  # Start with acceleration phase
    lap_count = 0
    lap_times = []
    lap_start_time = 0.0
    
    while current_time < max_time:
        # Calculate max cornering speed based on lateral acceleration limit
        # v² = a_lat * r
        max_lateral_accel = 1.5 * 9.81  # 1.5g lateral acceleration limit
        max_corner_speed = np.sqrt(max_lateral_accel * circle_radius)
        
        if phase == "acceleration" and current_time < corner_entry_time:
            # Acceleration phase - build up to cornering speed
            throttle = 1.0
            target_speed = min(max_corner_speed, self.current_speed + 1.0 * dt)
        elif phase == "acceleration" and current_time >= corner_entry_time:
            # Transition to steady-state cornering
            phase = "steady_state"
            lap_start_time = current_time
        elif phase == "steady_state":
            # Maintain constant speed around circle
            if self.current_speed > max_corner_speed * 1.02:
                # Too fast - reduce speed
                throttle = 0.3
            elif self.current_speed < max_corner_speed * 0.98:
                # Too slow - increase speed
                throttle = 0.8
            else:
                # Maintain speed
                throttle = 0.5
                
            # Calculate position around circle
            distance_traveled = self.current_speed * dt
            self.current_position += distance_traveled
            
            # Check if lap completed
            if self.current_position >= circle_circumference and (current_time - lap_start_time) > 1.0:
                lap_count += 1
                lap_times.append(current_time - lap_start_time)
                lap_start_time = current_time
                self.current_position = 0.0
                
                # If completed required laps, end simulation
                if lap_count >= steady_state_laps:
                    break
        
        # Determine optimal gear (simplified)
        if self.current_gear > 0:
            # Simple shift strategy based on RPM
            if self.current_engine_rpm > self.engine.max_power_rpm * 0.95 and self.current_gear < self.drivetrain.transmission.num_gears:
                self.current_gear += 1
            elif self.current_engine_rpm < self.engine.max_torque_rpm * 0.8 and self.current_gear > 1:
                self.current_gear -= 1
        
        # Calculate lateral acceleration
        lateral_accel = self.current_speed**2 / circle_radius if self.current_speed > 0.1 else 0.0
        
        # Calculate longitudinal acceleration
        self.calculate_acceleration(throttle, 0.0, self.current_gear, self.current_speed)
        
        # Update vehicle state
        self.update_vehicle_state(dt)
        
        # Update time
        current_time += dt
        
        # Store results
        time_points.append(current_time)
        speed_points.append(self.current_speed)
        lateral_accel_points.append(lateral_accel)
        position_points.append(self.current_position)
        rpm_points.append(self.current_engine_rpm)
        gear_points.append(self.current_gear)
    
    # Calculate average lap time for steady state
    avg_lap_time = sum(lap_times) / len(lap_times) if lap_times else None
    
    # Compile results
    results = {
        'time': np.array(time_points),
        'speed': np.array(speed_points),
        'lateral_acceleration': np.array(lateral_accel_points),
        'position': np.array(position_points),
        'engine_rpm': np.array(rpm_points),
        'gear': np.array(gear_points),
        'lap_times': lap_times,
        'average_lap_time': avg_lap_time,
        'max_lateral_acceleration': np.max(lateral_accel_points),
        'max_speed': np.max(speed_points)
    }
    
    return results

def calculate_performance_metrics(self) -> Dict:
    """
    Calculate various vehicle performance metrics.
    
    Returns:
        Dictionary with performance metrics
    """
    # Calculate power-to-weight ratio
    if self.engine:
        # Convert hp to kW for SI units
        power_kw = self.engine.max_power * 0.7457
        power_to_weight = power_kw / (self.mass / 1000)  # kW/kg
    else:
        power_to_weight = None
    
    # Calculate theoretical top speed (simplified)
    if self.engine and self.drivetrain:
        # Get maximum torque and corresponding RPM
        max_torque = self.engine.max_torque
        
        # Calculate wheel torque in highest gear
        highest_gear = self.drivetrain.transmission.num_gears
        wheel_torque = self.drivetrain.calculate_wheel_torque(max_torque, highest_gear)
        
        # Calculate tractive force
        tractive_force = wheel_torque / self.tire_radius
        
        # Iteratively solve for top speed (where tractive force equals drag)
        speed = 10.0  # m/s initial guess
        max_iterations = 50
        tolerance = 0.01
        
        for _ in range(max_iterations):
            # Calculate drag force at current speed
            air_density = 1.225  # kg/m³
            drag_force = 0.5 * air_density * self.drag_coefficient * self.frontal_area * speed**2
            rolling_resistance = self.rolling_resistance * self.mass * 9.81
            
            # Calculate net force
            net_force = tractive_force - drag_force - rolling_resistance
            
            # Check if converged
            if abs(net_force) < tolerance:
                break
            
            # Update speed estimate
            speed_update = net_force / (air_density * self.drag_coefficient * self.frontal_area * speed)
            speed += speed_update
            
            # Limit to reasonable range
            speed = max(0.1, min(100.0, speed))
        
        top_speed = speed
    else:
        top_speed = None
    
    # Calculate theoretical max lateral acceleration
    # Simplified model based on downforce and tire grip
    base_tire_grip = 1.5  # Lateral g in static condition
    downforce_coefficient = -self.lift_coefficient  # Convert lift to downforce
    
    # Reference speed for downforce calculation
    reference_speed = 20.0  # m/s (~45 mph)
    air_density = 1.225  # kg/m³
    
    # Downforce at reference speed
    downforce = 0.5 * air_density * downforce_coefficient * self.frontal_area * reference_speed**2
    
    # Additional grip from downforce
    downforce_grip_contribution = downforce / (self.mass * 9.81) * 0.8  # 80% effectiveness
    
    # Total lateral grip
    max_lateral_accel = (base_tire_grip + downforce_grip_contribution) * 9.81
    
    # Compile metrics
    metrics = {
        'power_to_weight': power_to_weight,  # kW/kg
        'top_speed': top_speed,  # m/s
        'top_speed_mph': top_speed * 2.23694 if top_speed else None,  # mph
        'max_lateral_acceleration': max_lateral_accel,  # m/s²
        'max_lateral_acceleration_g': max_lateral_accel / 9.81  # g
    }
    
    return metrics

def analyze_thermal_performance(self, 
                             ambient_temp: float = 25.0, 
                             vehicle_speed_range: List[float] = None) -> Dict:
    """
    Analyze thermal system performance across a range of vehicle speeds.
    
    Args:
        ambient_temp: Ambient temperature in °C
        vehicle_speed_range: List of vehicle speeds to analyze in m/s
        
    Returns:
        Dictionary with thermal performance analysis
    """
    if vehicle_speed_range is None:
        vehicle_speed_range = np.linspace(0, 30, 7)  # 0-30 m/s
    
    # Initialize result arrays
    n_speeds = len(vehicle_speed_range)
    coolant_temps = np.zeros(n_speeds)
    oil_temps = np.zeros(n_speeds)
    heat_rejections = np.zeros(n_speeds)
    
    # Engine conditions for analysis
    engine_rpm = 8000  # Representative RPM
    engine_load = 0.7  # 70% load
    
    # Run analysis for each speed
    for i, speed in enumerate(vehicle_speed_range):
        # Update engine state
        if self.engine:
            self.engine.throttle_position = engine_load
            self.engine.current_rpm = engine_rpm
        
        # Update cooling system
        if self.cooling_system:
            self.update_cooling_system(
                coolant_temp=90.0,  # Start from typical operating temperature
                ambient_temp=ambient_temp,
                vehicle_speed=speed
            )
            
            # Simulate for a short time to reach steady state
            for _ in range(10):
                self.cooling_system.update_system_state(1.0)  # 1 second steps
            
            # Get thermal state
            state = self.cooling_system.get_system_state()
            coolant_temps[i] = state['coolant_temp']
            heat_rejections[i] = state['radiator_heat_rejection']
        
        # Get engine temperatures
        if self.engine:
            oil_temps[i] = self.engine.oil_temperature
    
    # Calculate thermal margins
    coolant_margins = 105.0 - coolant_temps  # Margin to boiling
    oil_margins = 130.0 - oil_temps  # Margin to oil degradation
    
    # Analyze side pods if available
    side_pod_performance = None
    if self.side_pods:
        side_pod_performance = self.side_pods.analyze_system_performance(
            vehicle_speed_range=vehicle_speed_range,
            coolant_temp=90.0,
            ambient_temp=ambient_temp,
            coolant_flow_rate=50.0
        )
    
    # Analyze rear radiator if available
    rear_rad_performance = None
    if self.rear_radiator:
        rear_rad_performance = self.rear_radiator.analyze_performance(
            vehicle_speed_range=vehicle_speed_range,
            coolant_temp=90.0,
            ambient_temp=ambient_temp,
            coolant_flow_rate=50.0
        )
    
    # Compile results
    results = {
        'vehicle_speeds': vehicle_speed_range,
        'coolant_temps': coolant_temps,
        'oil_temps': oil_temps,
        'heat_rejections': heat_rejections,
        'coolant_margins': coolant_margins,
        'oil_margins': oil_margins,
        'ambient_temp': ambient_temp,
        'side_pod_performance': side_pod_performance,
        'rear_radiator_performance': rear_rad_performance
    }
    
    return results

def plot_thermal_analysis(self, analysis_results: Dict, save_path: Optional[str] = None):
    """
    Plot thermal system analysis results.
    
    Args:
        analysis_results: Results from analyze_thermal_performance
        save_path: Optional path to save the plot
    """
    # Extract data
    speeds = analysis_results['vehicle_speeds']
    coolant_temps = analysis_results['coolant_temps']
    oil_temps = analysis_results['oil_temps']
    heat_rejections = analysis_results['heat_rejections']
    
    # Create figure with multiple subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    
    # Plot temperatures
    color1 = 'tab:red'
    ax1.set_ylabel('Temperature (°C)', color=color1)
    ax1.plot(speeds, coolant_temps, color=color1, linewidth=2, marker='o', label='Coolant')
    ax1.plot(speeds, oil_temps, color='tab:orange', linewidth=2, marker='s', label='Oil')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(loc='upper right')
    
    # Add critical temperature lines
    ax1.axhline(y=105, color='tab:red', linestyle='--', alpha=0.5, label='Critical Coolant')
    ax1.axhline(y=130, color='tab:orange', linestyle='--', alpha=0.5, label='Critical Oil')
    
    # Plot heat rejection
    color2 = 'tab:blue'
    ax2.set_ylabel('Heat Rejection (kW)', color=color2)
    ax2.plot(speeds, heat_rejections / 1000, color=color2, linewidth=2, marker='o')
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    ax2.set_xlabel('Vehicle Speed (m/s)')
    
    # Add title
    ambient_temp = analysis_results['ambient_temp']
    plt.suptitle(f'Thermal System Performance Analysis (Ambient: {ambient_temp}°C)')
    
    plt.tight_layout()
    
    # Save plot if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    plt.show()

def plot_skidpad_results(self, results: Dict, save_path: Optional[str] = None):
    """
    Plot skidpad simulation results.
    
    Args:
        results: Results from simulate_skidpad
        save_path: Optional path to save the plot
    """
    # Extract data
    time = results['time']
    speed = results['speed']
    lateral_accel = results['lateral_acceleration']
    rpm = results['engine_rpm']
    gear = results['gear']
    
    # Convert to more readable units
    speed_mph = speed * 2.23694
    lateral_accel_g = lateral_accel / 9.81
    
    # Create figure with multiple subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    # Plot speed
    ax1.plot(time, speed_mph, 'b-', linewidth=2)
    ax1.set_ylabel('Speed (mph)')
    ax1.set_title('Skidpad Simulation Results')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Plot lateral acceleration
    ax2.plot(time, lateral_accel_g, 'g-', linewidth=2)
    ax2.set_ylabel('Lateral Acceleration (g)')
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Plot RPM and gear
    color1 = 'tab:blue'
    ax3.set_ylabel('Engine RPM', color=color1)
    ax3.plot(time, rpm, color=color1, linewidth=2)
    ax3.tick_params(axis='y', labelcolor=color1)
    ax3.grid(True, linestyle='--', alpha=0.7)
    
    color2 = 'tab:red'
    ax3_twin = ax3.twinx()
    ax3_twin.set_ylabel('Gear', color=color2)
    ax3_twin.step(time, gear, color=color2, linewidth=2, where='post')
    ax3_twin.tick_params(axis='y', labelcolor=color2)
    ax3_twin.set_yticks(range(0, self.drivetrain.transmission.num_gears + 1))
    
    ax3.set_xlabel('Time (s)')
    
    # Add lap time if available
    if results['average_lap_time'] is not None:
        lap_time = results['average_lap_time']
        max_lat_g = results['max_lateral_acceleration'] / 9.81
        plt.figtext(
            0.5, 0.01,
            f"Average Lap Time: {lap_time:.2f}s, Max Lateral Acceleration: {max_lat_g:.2f}g",
            ha='center',
            fontsize=12,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray')
        )
    
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    
    # Save plot if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    plt.show()

def optimize_shift_points(self, max_rpm: Optional[float] = None) -> Dict:
    """
    Optimize gear shift points for maximum acceleration.
    
    Args:
        max_rpm: Maximum RPM limit (defaults to engine redline)
        
    Returns:
        Dictionary with optimized shift points
    """
    if not self.engine or not self.drivetrain:
        return {}
    
    # Use engine redline if max_rpm not specified
    if max_rpm is None:
        max_rpm = self.engine.redline
    
    # Initialize shift points array
    shift_points = []
    
    # Get gear ratios
    gear_ratios = self.drivetrain.transmission.gear_ratios
    
    # For each gear (except highest)
    for i in range(1, len(gear_ratios)):
        current_gear = i
        next_gear = i + 1
        
        # Get gear ratios
        current_ratio = gear_ratios[current_gear - 1]
        next_ratio = gear_ratios[next_gear - 1]
        
        # Calculate engine RPM after shift for a range of RPMs
        rpm_range = np.linspace(self.engine.max_torque_rpm, max_rpm, 50)
        best_shift_rpm = self.engine.max_power_rpm  # Default to max power
        best_acceleration = 0.0
        
        for rpm in rpm_range:
            # Calculate engine torque at current RPM
            torque_current = self.engine.get_torque(rpm)
            
            # Calculate engine RPM after shift
            rpm_after_shift = rpm * (current_ratio / next_ratio)
            
            # Calculate engine torque after shift
            torque_after_shift = self.engine.get_torque(rpm_after_shift)
            
            # Calculate wheel torque and tractive force for both scenarios
            wheel_torque_current = self.drivetrain.calculate_wheel_torque(torque_current, current_gear)
            wheel_torque_after = self.drivetrain.calculate_wheel_torque(torque_after_shift, next_gear)
            
            tractive_force_current = wheel_torque_current / self.tire_radius
            tractive_force_after = wheel_torque_after / self.tire_radius
            
            # Calculate acceleration for both scenarios
            accel_current = tractive_force_current / self.mass
            accel_after = tractive_force_after / self.mass
            
            # If acceleration after shift is better, this is a good shift point
            if accel_after > accel_current and accel_after > best_acceleration:
                best_acceleration = accel_after
                best_shift_rpm = rpm
        
        shift_points.append(best_shift_rpm)
    
    # Create results dictionary
    results = {
        'upshift_points_rpm': shift_points,
        'upshift_points_by_gear': {i+1: rpm for i, rpm in enumerate(shift_points)}
    }
    
    return results

if __name__ == "__main__":
    # Create a Formula Student vehicle
    vehicle = create_formula_student_vehicle()
    
    # Print vehicle specifications
    specs = vehicle.get_vehicle_specs()
    print(f"Vehicle: {specs['team_name']}")
    print(f"Mass: {specs['vehicle']['mass']} kg")
    print(f"Engine: {specs['engine']['make']} {specs['engine']['model']}")
    print(f"Max Power: {specs['engine']['max_power_hp']} hp @ {specs['engine']['max_power_rpm']} RPM")
    print(f"Gear Ratios: {specs['drivetrain']['transmission_ratios']}")
    
    # Calculate performance metrics
    print("\nCalculating performance metrics...")
    metrics = vehicle.calculate_performance_metrics()
    print(f"Power-to-Weight Ratio: {metrics['power_to_weight']:.2f} kW/kg")
    print(f"Theoretical Top Speed: {metrics['top_speed_mph']:.1f} mph")
    print(f"Maximum Lateral Acceleration: {metrics['max_lateral_acceleration_g']:.2f}g")
    
    # Optimize shift points
    print("\nOptimizing shift points...")
    shift_points = vehicle.optimize_shift_points()
    print("Optimized Upshift Points (RPM):")
    for gear, rpm in shift_points['upshift_points_by_gear'].items():
        print(f"  Gear {gear}: {rpm:.0f} RPM")
    
    # Run an acceleration simulation
    print("\nRunning acceleration simulation...")
    accel_results = vehicle.simulate_acceleration_run()
    
    # Display acceleration results
    if accel_results['finish_time'] is not None:
        print(f"75m Acceleration Time: {accel_results['finish_time']:.2f} seconds")
        print(f"Top Speed: {accel_results['finish_speed'] * 2.23694:.1f} mph")
    
    if accel_results['time_to_60mph'] is not None:
        print(f"0-60 mph Time: {accel_results['time_to_60mph']:.2f} seconds")
    
    # Run a skidpad simulation
    print("\nRunning skidpad simulation...")
    skidpad_results = vehicle.simulate_skidpad()
    
    # Display skidpad results
    if skidpad_results['average_lap_time'] is not None:
        print(f"Skidpad Lap Time: {skidpad_results['average_lap_time']:.2f} seconds")
        print(f"Maximum Lateral Acceleration: {skidpad_results['max_lateral_acceleration']/9.81:.2f}g")
    
    # Perform thermal analysis
    print("\nPerforming thermal analysis...")
    thermal_results = vehicle.analyze_thermal_performance(ambient_temp=30.0)
    
    # Display thermal results
    max_coolant_temp = max(thermal_results['coolant_temps'])
    print(f"Maximum Coolant Temperature: {max_coolant_temp:.1f}°C")
    print(f"Maximum Heat Rejection: {max(thermal_results['heat_rejections'])/1000:.1f} kW")
    
    # Plot results (comment out if not needed)
    print("\nPlotting results...")
    vehicle.plot_acceleration_results(accel_results)
    vehicle.plot_skidpad_results(skidpad_results)
    vehicle.plot_thermal_analysis(thermal_results)
    
    print("\nSimulation complete!")
