#!/usr/bin/env python3
"""
Formula Student Vehicle Performance Simulator

A comprehensive simulation package for Formula Student vehicle performance analysis,
optimization, and engineering decision support. This program integrates multiple
simulation modules to provide a complete performance evaluation across various
Formula Student competition events.

Features:
- Multi-event simulation (acceleration, skidpad, autocross, endurance)
- Thermal analysis and cooling system optimization
- Weight sensitivity analysis and optimization
- Lap time optimization with advanced racing line calculation
- Transmission and shifting strategy evaluation
- Custom track generation and management
- Parametric studies and batch processing
- Comprehensive reporting and visualization
"""

import os
import sys
import time
import argparse
import yaml
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path

# Add project root to Python path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# =========================================================================
# Import required modules
# =========================================================================

# Core modules
from kcl_fs_powertrain.core.vehicle import create_formula_student_vehicle
from kcl_fs_powertrain.core.simulator import Simulator
from kcl_fs_powertrain.core.track_integration import TrackProfile, calculate_optimal_racing_line
from kcl_fs_powertrain.core.track import Track, RacingLine, TrackSegment, TrackSegmentType

# Performance modules
from kcl_fs_powertrain.performance.acceleration import AccelerationSimulator, run_fs_acceleration_simulation
from kcl_fs_powertrain.performance.lap_time import LapTimeSimulator, create_example_track, run_fs_lap_simulation
from kcl_fs_powertrain.performance.endurance import EnduranceSimulator, run_endurance_simulation
from kcl_fs_powertrain.performance.weight_sensitivity import WeightSensitivityAnalyzer
from kcl_fs_powertrain.performance.lap_time_optimization import run_lap_optimization, compare_optimization_methods
from kcl_fs_powertrain.performance.optimal_lap_time import OptimalLapTimeOptimizer, run_advanced_lap_optimization

# Engine and thermal modules
from kcl_fs_powertrain.engine.motorcycle_engine import MotorcycleEngine
from kcl_fs_powertrain.engine.torque_curve import TorqueCurve
from kcl_fs_powertrain.engine.fuel_systems import FuelSystem, FuelConsumption, FuelType, FuelProperties
from kcl_fs_powertrain.engine.engine_thermal import ThermalConfig, CoolingSystem, EngineHeatModel, ThermalSimulation, CoolingPerformance

# Thermal system modules
from kcl_fs_powertrain.thermal.cooling_system import Radiator, RadiatorType, WaterPump, CoolingFan, FanType, Thermostat, create_formula_student_cooling_system
from kcl_fs_powertrain.thermal.side_pod import SidePod, SidePodRadiator, SidePodSystem, DualSidePodSystem, create_standard_side_pod_system, create_cooling_optimized_side_pod_system
from kcl_fs_powertrain.thermal.rear_radiator import RearRadiator, RearRadiatorSystem, create_optimized_rear_radiator_system
from kcl_fs_powertrain.thermal.electric_compressor import ElectricCompressor, CoolingAssistSystem, create_high_performance_cooling_assist_system

# Transmission modules
from kcl_fs_powertrain.transmission.gearing import Transmission, FinalDrive, Differential, DrivetrainSystem
from kcl_fs_powertrain.transmission.shift_strategy import ShiftStrategy, StrategyManager, MaxAccelerationStrategy, EnduranceStrategy
from kcl_fs_powertrain.transmission.cas_system import CASSystem

# Track generator
from kcl_fs_powertrain.track_generator.generator import FSTrackGenerator, TrackMode, SimType
from kcl_fs_powertrain.track_generator.utils import generate_multiple_tracks

# Utilities
from kcl_fs_powertrain.utils.plotting import set_plot_style, save_plot, plot_engine_performance, plot_track_layout
from kcl_fs_powertrain.utils.constants import GRAVITY, AIR_DENSITY_SEA_LEVEL
from kcl_fs_powertrain.utils.validation import validate_vehicle_specs, validate_acceleration_performance


# =========================================================================
# Configuration Management
# =========================================================================

class ConfigurationManager:
    """Manages loading, validation, and access to all configuration settings."""
    
    def __init__(self, args: Optional[argparse.Namespace] = None):
        """
        Initialize the configuration manager.
        
        Args:
            args: Command line arguments (optional)
        """
        self.args = args
        self.config = {}
        self.config_dir = "configs"
        self.output_dir = "data/output/simulation"
        
        # Set output directory from args if provided
        if args and hasattr(args, 'output_dir') and args.output_dir:
            self.output_dir = args.output_dir
            
        # Set config directory from args if provided
        if args and hasattr(args, 'config_dir') and args.config_dir:
            self.config_dir = args.config_dir
            
        # Default configuration paths
        self.default_paths = {
            'engine': os.path.join(self.config_dir, 'engine', 'cbr600f4i.yaml'),
            'thermal': os.path.join(self.config_dir, 'thermal', 'cooling_system.yaml'),
            'side_pod': os.path.join(self.config_dir, 'thermal', 'side_pod.yaml'),
            'electric_compressor': os.path.join(self.config_dir, 'thermal', 'electric_compressor.yaml'),
            'thermal_limits': os.path.join(self.config_dir, 'targets', 'thermal_limits.yaml'),
            'transmission': os.path.join(self.config_dir, 'transmission', 'gearing.yaml'),
            'shift_strategy': os.path.join(self.config_dir, 'transmission', 'shift_strategy.yaml'),
            'lap_time_optimization': os.path.join(self.config_dir, 'lap_time', 'optimal_lap_time.yaml'),
            'acceleration': os.path.join(self.config_dir, 'targets', 'acceleration.yaml'),
            'track_generator': os.path.join(self.config_dir, 'track_generator', 'generator_settings.yaml'),
        }
        
        # Initialize simulation settings
        self.simulation_settings = {
            'time_step': 0.01,  # s
            'max_time': 30.0,   # s
            'acceleration_distance': 75.0,  # m
            'endurance_laps': 3,  # Number of laps for endurance simulation
            
            # Event flags (which events to run)
            'run_acceleration': True,
            'run_lap_time': True,
            'run_endurance': True,
            'run_skidpad': False,  # Off by default
            
            # Analysis flags
            'run_weight_sensitivity': True,
            'run_lap_optimization': True,
            'run_cooling_comparison': True,
            'run_transmission_optimization': False,  # Off by default
            
            # Track generation settings
            'generate_tracks': False,  # Off by default
            'num_tracks': 3,
            'track_mode': 'EXTEND',
        }
        
        # Optimization settings
        self.optimization_settings = {
            'weight_range': [180, 250],  # kg
            'optimization_iterations': 10,
            'cooling_configurations': ['standard', 'optimized', 'minimal'],
            'param_ranges': {
                'mass': (180, 250),  # Vehicle mass range (kg)
                'drag_coefficient': (0.7, 1.2),  # Aero drag coefficient range
                'weight_distribution': (0.4, 0.6)  # Front weight distribution
            }
        }
        
        # Batch processing settings
        self.batch_settings = {
            'enable_batch': False,
            'parameter': 'mass',
            'values': [180, 190, 200, 210, 220, 230],
            'parallel_processing': False,
        }
    
    def load_configurations(self):
        """
        Load all configuration files.
        
        Returns:
            Dict: Complete configuration dictionary
        """
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize config dictionary
        self.config = {
            'paths': self.default_paths,
            'simulation_settings': self.simulation_settings,
            'optimization_settings': self.optimization_settings,
            'batch_settings': self.batch_settings,
            'output_dir': self.output_dir,
        }
        
        # Load each configuration file if it exists
        for key, path in self.default_paths.items():
            if os.path.exists(path):
                try:
                    with open(path, 'r') as f:
                        self.config[key] = yaml.safe_load(f)
                    print(f"Loaded {key} configuration from: {path}")
                except Exception as e:
                    print(f"Warning: Could not load {key} configuration from {path}: {str(e)}")
        
        # Override settings from command line arguments if provided
        if self.args:
            self._apply_command_line_overrides()
        
        # Create event-specific output directories
        self._create_output_directories()
        
        return self.config
    
    def _apply_command_line_overrides(self):
        """Apply command line argument overrides to configuration."""
        if not self.args:
            return
            
        # Override simulation settings from args
        if hasattr(self.args, 'time_step') and self.args.time_step:
            self.simulation_settings['time_step'] = self.args.time_step
            
        if hasattr(self.args, 'endurance_laps') and self.args.endurance_laps:
            self.simulation_settings['endurance_laps'] = self.args.endurance_laps
            
        # Override event flags
        for event in ['acceleration', 'lap_time', 'endurance', 'skidpad']:
            arg_name = f'run_{event}'
            if hasattr(self.args, arg_name) and getattr(self.args, arg_name) is not None:
                self.simulation_settings[arg_name] = getattr(self.args, arg_name)
                
        # Override analysis flags
        for analysis in ['weight_sensitivity', 'lap_optimization', 'cooling_comparison']:
            arg_name = f'run_{analysis}'
            if hasattr(self.args, arg_name) and getattr(self.args, arg_name) is not None:
                self.simulation_settings[arg_name] = getattr(self.args, arg_name)
        
        # Track generation settings
        if hasattr(self.args, 'generate_tracks') and self.args.generate_tracks:
            self.simulation_settings['generate_tracks'] = True
            
            if hasattr(self.args, 'num_tracks') and self.args.num_tracks:
                self.simulation_settings['num_tracks'] = self.args.num_tracks
    
    def _create_output_directories(self):
        """Create all necessary output directories."""
        # Main output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create subdirectories for different analyses
        subdirs = [
            'acceleration',
            'lap_time',
            'endurance',
            'skidpad',
            'weight_sensitivity',
            'lap_optimization',
            'cooling_comparison',
            'thermal_analysis',
            'track_generator',
            'transmission_analysis',
            'reports'
        ]
        
        for subdir in subdirs:
            os.makedirs(os.path.join(self.output_dir, subdir), exist_ok=True)
        
        # Store paths in config
        self.config['output_paths'] = {
            subdir: os.path.join(self.output_dir, subdir) for subdir in subdirs
        }
    
    def get_output_path(self, analysis_type: str) -> str:
        """
        Get the output directory path for a specific analysis type.
        
        Args:
            analysis_type: Type of analysis ('acceleration', 'lap_time', etc.)
            
        Returns:
            str: Path to output directory
        """
        if analysis_type in self.config['output_paths']:
            return self.config['output_paths'][analysis_type]
        else:
            # Default to main output directory
            return self.output_dir
    
    def save_configuration(self):
        """Save the complete configuration to a file for reference."""
        # Create a simplified version of the config for saving
        # (avoid circular references and non-serializable objects)
        save_config = {
            'simulation_settings': self.simulation_settings,
            'optimization_settings': self.optimization_settings,
            'paths': self.default_paths,
            'output_dir': self.output_dir,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Save the configuration
        config_path = os.path.join(self.output_dir, 'simulation_config.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(save_config, f, default_flow_style=False)
            
        print(f"Configuration saved to: {config_path}")


# =========================================================================
# Vehicle Factory
# =========================================================================

class VehicleFactory:
    """Factory class for creating and configuring vehicle models."""
    
    @staticmethod
    def create_vehicle(config: Dict, cooling_config: str = 'standard') -> Any:
        """
        Create and configure a Formula Student vehicle with specified configuration.
        
        Args:
            config: Configuration dictionary
            cooling_config: Type of cooling configuration ('standard', 'optimized', 'minimal', or 'custom')
            
        Returns:
            Vehicle: Configured Formula Student vehicle
        """
        # Create a default Formula Student vehicle
        vehicle = create_formula_student_vehicle()
        
        # Load engine configuration if specified
        if 'engine' in config and config['engine'] and 'engine' in config['paths']:
            try:
                # Check if engine_config file exists
                engine_config_path = config['paths']['engine']
                if os.path.exists(engine_config_path):
                    # Update engine parameters from config file
                    vehicle.engine.load_config(engine_config_path)
                    print(f"Engine configured from: {engine_config_path}")
            except Exception as e:
                print(f"Warning: Could not load engine configuration: {str(e)}")
        
        # Load transmission configuration if specified
        if 'transmission' in config and config['transmission'] and 'transmission' in config['paths']:
            try:
                # Check if transmission_config file exists
                transmission_config_path = config['paths']['transmission']
                if os.path.exists(transmission_config_path):
                    # Update transmission parameters from config file
                    # This is a placeholder - in practice, you would need to create
                    # an appropriate method on the vehicle or drivetrain object
                    if hasattr(vehicle, 'drivetrain') and hasattr(vehicle.drivetrain, 'load_config'):
                        vehicle.drivetrain.load_config(transmission_config_path)
                    print(f"Transmission configured from: {transmission_config_path}")
            except Exception as e:
                print(f"Warning: Could not load transmission configuration: {str(e)}")
        
        # Load shift strategy configuration if specified
        if 'shift_strategy' in config and config['shift_strategy'] and 'shift_strategy' in config['paths']:
            try:
                # Check if shift_strategy file exists
                shift_config_path = config['paths']['shift_strategy']
                if os.path.exists(shift_config_path):
                    # Update shift strategy parameters from config file
                    if hasattr(vehicle, 'drivetrain') and hasattr(vehicle.drivetrain, 'strategy_manager'):
                        # Assume the strategy manager has a load_config method
                        vehicle.drivetrain.strategy_manager.load_config(shift_config_path)
                    print(f"Shift strategy configured from: {shift_config_path}")
            except Exception as e:
                print(f"Warning: Could not load shift strategy configuration: {str(e)}")
        
        # Configure cooling system based on the specified configuration
        if cooling_config == 'standard':
            print("Using standard cooling configuration")
            # Use the standard cooling system
            if hasattr(vehicle, '_initialize_thermal_systems'):
                # The vehicle object already has thermal systems initialized
                pass
            else:
                # Set up thermal systems manually
                cooling_system = create_formula_student_cooling_system()
                vehicle.cooling_system = cooling_system
                
                # Add side pod cooling if possible
                try:
                    side_pod_system = create_standard_side_pod_system()
                    vehicle.side_pod_system = side_pod_system
                except Exception as e:
                    print(f"Note: Could not initialize side pod system: {str(e)}")
        
        elif cooling_config == 'optimized':
            print("Using optimized cooling configuration")
            # Create an optimized cooling setup with both side pods and rear radiator
            cooling_system = create_formula_student_cooling_system()
            side_pod_system = create_cooling_optimized_side_pod_system()
            rear_radiator = create_optimized_rear_radiator_system()
            cooling_assist = create_high_performance_cooling_assist_system()
            
            # Assign to vehicle
            vehicle.cooling_system = cooling_system
            vehicle.side_pod_system = side_pod_system
            vehicle.rear_radiator = rear_radiator
            vehicle.cooling_assist = cooling_assist
            
            # Update engine thermal model with optimized cooling
            if hasattr(vehicle.engine, 'heat_model'):
                vehicle.engine.heat_model.combustion_efficiency = 0.32  # Slight improvement
        
        elif cooling_config == 'minimal':
            print("Using minimal weight cooling configuration")
            # Create minimal cooling setup (prioritizing weight)
            cooling_system = create_formula_student_cooling_system()
            
            # Use a single radiator with lightweight configuration
            radiator = Radiator(
                radiator_type=RadiatorType.SINGLE_CORE_ALUMINUM,
                core_area=0.14,  # Smaller radiator
                core_thickness=0.03,
                fin_density=16,
                tube_rows=1  # Single row for weight reduction
            )
            cooling_system.radiator = radiator
            
            # Assign to vehicle
            vehicle.cooling_system = cooling_system
        
        elif cooling_config == 'custom':
            print("Using custom cooling configuration from config files")
            # Try to load cooling configuration from config files
            try:
                # Create thermal configuration
                thermal_config = ThermalConfig()
                if 'thermal' in config['paths'] and os.path.exists(config['paths']['thermal']):
                    thermal_config.load_from_file(config['paths']['thermal'])
                
                # Create engine heat model with this configuration
                heat_model = EngineHeatModel(thermal_config, vehicle.engine)
                
                # Create cooling system
                cooling_system = CoolingSystem()
                
                # Create side pod system if config available
                if 'side_pod' in config['paths'] and os.path.exists(config['paths']['side_pod']):
                    with open(config['paths']['side_pod'], 'r') as f:
                        side_pod_data = yaml.safe_load(f)
                        # Simplified creation - in a real implementation this would use the config file
                        side_pod_system = create_standard_side_pod_system()
                else:
                    side_pod_system = create_standard_side_pod_system()
                
                # Assign to vehicle
                vehicle.engine.heat_model = heat_model
                vehicle.cooling_system = cooling_system
                vehicle.side_pod_system = side_pod_system
                
            except Exception as e:
                print(f"Warning: Could not load custom cooling configuration: {str(e)}")
                print("Falling back to standard cooling configuration")
                
                # Fall back to standard configuration
                cooling_system = create_formula_student_cooling_system()
                vehicle.cooling_system = cooling_system
        
        # Set thermal limits if available in config
        if 'thermal_limits' in config and config['thermal_limits']:
            try:
                limits = config['thermal_limits']
                
                if 'engine' in limits:
                    vehicle.engine_warning_temp = limits['engine'].get('warning_temp', 110.0)
                    vehicle.engine_critical_temp = limits['engine'].get('critical_temp', 120.0)
                    vehicle.engine_shutdown_temp = limits['engine'].get('shutdown_temp', 125.0)
                
                if 'coolant' in limits:
                    vehicle.coolant_warning_temp = limits['coolant'].get('warning_temp', 95.0)
                    vehicle.coolant_critical_temp = limits['coolant'].get('critical_temp', 105.0)
                
                print(f"Thermal limits set - Engine warning: {vehicle.engine_warning_temp}°C, critical: {vehicle.engine_critical_temp}°C")
                
            except Exception as e:
                print(f"Warning: Could not set thermal limits: {str(e)}")
        
        # Print vehicle specifications
        specs = vehicle.get_vehicle_specs()
        print("\nVehicle Specifications:")
        print(f"  Engine: {specs['engine']['make']} {specs['engine']['model']}")
        print(f"  Max Power: {specs['engine']['max_power_hp']} hp @ {specs['engine']['max_power_rpm']} RPM")
        print(f"  Max Torque: {specs['engine']['max_torque_nm']} Nm @ {specs['engine']['max_torque_rpm']} RPM")
        print(f"  Mass: {specs['vehicle']['mass']} kg")
        print(f"  Drag Coefficient: {specs['vehicle']['drag_coefficient']}")
        print(f"  Gear Ratios: {specs['drivetrain']['transmission_ratios']}")
        
        # Print cooling system info if available
        if hasattr(vehicle, 'cooling_system'):
            print("\nCooling System Information:")
            if hasattr(vehicle.cooling_system, 'radiator'):
                radiator = vehicle.cooling_system.radiator
                print(f"  Radiator: {radiator.__class__.__name__}, type: {getattr(radiator, 'radiator_type', 'Unknown').name if hasattr(radiator, 'radiator_type') else 'Unknown'}")
                print(f"  Radiator Area: {getattr(radiator, 'core_area', 0.0):.3f} m²")
            
            if hasattr(vehicle, 'side_pod_system'):
                print("  Side Pod Cooling: Enabled")
            
            if hasattr(vehicle, 'rear_radiator'):
                print("  Rear Radiator: Enabled")
            
            if hasattr(vehicle, 'cooling_assist'):
                print("  Electric Cooling Assist: Enabled")
        
        return vehicle


# =========================================================================
# Track Management
# =========================================================================

class TrackManager:
    """Manages track creation, loading, and generation."""
    
    def __init__(self, config: Dict):
        """
        Initialize the track manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.output_dir = config['output_paths']['track_generator']
        self.track_files = []
        
    def create_example_track(self, difficulty: str = 'medium') -> str:
        """
        Create a sample track for testing.
        
        Args:
            difficulty: Track difficulty ('easy', 'medium', or 'hard')
            
        Returns:
            str: Path to the created track file
        """
        # Create track file path
        track_file = os.path.join(self.output_dir, f"test_track_{difficulty}.yaml")
        
        # Create a sample track with specified difficulty
        create_example_track(track_file, difficulty=difficulty)
        
        print(f"Test track created and saved to {track_file}")
        self.track_files.append(track_file)
        
        return track_file
    
    def generate_tracks(self, num_tracks: int = 3, mode_str: str = 'EXTEND', visualize: bool = False) -> List[str]:
        """
        Generate multiple Formula Student tracks.
        
        Args:
            num_tracks: Number of tracks to generate
            mode_str: Track generation mode ('EXPAND', 'EXTEND', or 'RANDOM')
            visualize: Whether to visualize the generated tracks
            
        Returns:
            List[str]: Paths to the generated track files
        """
        print(f"\nGenerating {num_tracks} tracks with mode: {mode_str}")
        
        # Convert mode string to TrackMode enum
        try:
            mode = TrackMode[mode_str]
        except KeyError:
            print(f"Invalid track mode: {mode_str}, defaulting to EXTEND")
            mode = TrackMode.EXTEND
        
        # Generate tracks
        try:
            tracks = generate_multiple_tracks(
                num_tracks=num_tracks,
                base_dir=self.output_dir,
                mode=mode,
                visualize=visualize,
                export_formats=[SimType.FSDS]
            )
            
            # Extract file paths and add to track_files list
            for track in tracks:
                if 'filepath' in track:
                    self.track_files.append(track['filepath'])
            
            print(f"Successfully generated {len(tracks)} tracks")
            
            return self.track_files
        
        except Exception as e:
            print(f"Error generating tracks: {str(e)}")
            return []
    
    def load_track(self, track_file: str) -> Optional[Any]:
        """
        Load a track from file.
        
        Args:
            track_file: Path to track file
            
        Returns:
            Track: Loaded track object or None if loading failed
        """
        try:
            track = Track()
            track.load_from_file(track_file)
            return track
        except Exception as e:
            print(f"Error loading track from {track_file}: {str(e)}")
            return None
    
    def get_available_tracks(self) -> List[str]:
        """
        Get a list of available track files.
        
        Returns:
            List[str]: Paths to available track files
        """
        return self.track_files


# =========================================================================
# Event Simulation
# =========================================================================

class EventSimulator:
    """Base class for Formula Student event simulators."""
    
    def __init__(self, vehicle: Any, config: Dict):
        """
        Initialize the event simulator.
        
        Args:
            vehicle: Vehicle model to simulate
            config: Configuration dictionary
        """
        self.vehicle = vehicle
        self.config = config
        self.results = {}
    
    def run(self) -> Dict:
        """
        Run the event simulation.
        
        Returns:
            Dict: Simulation results
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def visualize_results(self) -> None:
        """Visualize the simulation results."""
        raise NotImplementedError("Subclasses must implement this method")
    
    def export_results(self) -> None:
        """Export the simulation results to files."""
        raise NotImplementedError("Subclasses must implement this method")


class AccelerationEventSimulator(EventSimulator):
    """Simulator for acceleration event."""
    
    def __init__(self, vehicle: Any, config: Dict):
        """
        Initialize the acceleration event simulator.
        
        Args:
            vehicle: Vehicle model to simulate
            config: Configuration dictionary
        """
        super().__init__(vehicle, config)
        self.output_dir = config['output_paths']['acceleration']
        
        # Create simulator
        self.simulator = AccelerationSimulator(vehicle)
        
        # Configure simulator from settings
        distance = config['simulation_settings'].get('acceleration_distance', 75.0)
        time_step = config['simulation_settings'].get('time_step', 0.01)
        max_time = config['simulation_settings'].get('max_time', 10.0)
        
        self.simulator.configure(
            distance=distance, 
            time_step=time_step, 
            max_time=max_time
        )
        
        # Configure launch control with reasonable defaults
        self.simulator.configure_launch_control(
            launch_rpm=vehicle.engine.max_torque_rpm * 1.1,
            launch_slip_target=0.2,
            launch_duration=0.5
        )
    
    def run(self) -> Dict:
        """
        Run the acceleration simulation.
        
        Returns:
            Dict: Acceleration simulation results
        """
        print("\n--- Running Acceleration Event ---")
        
        # Run simulation
        start_time = time.time()
        results = self.simulator.simulate_acceleration(use_launch_control=True, optimized_shifts=True)
        end_time = time.time()
        
        # Analyze results
        metrics = self.simulator.analyze_performance_metrics(results)
        
        # Store results
        self.results = {
            'results': results,
            'metrics': metrics,
            'simulator': self.simulator,
            'simulation_time': end_time - start_time
        }
        
        # Print key results
        print(f"Simulation completed in {end_time - start_time:.2f}s")
        if results['finish_time'] is not None:
            print(f"75m Acceleration Time: {results['finish_time']:.3f}s")
            print(f"Final Speed: {results['finish_speed'] * 3.6:.1f} km/h")
        
        if results['time_to_60mph'] is not None:
            print(f"0-60 mph Time: {results['time_to_60mph']:.3f}s")
        
        if results['time_to_100kph'] is not None:
            print(f"0-100 km/h Time: {results['time_to_100kph']:.3f}s")
        
        return self.results
    
    def visualize_results(self) -> None:
        """Visualize the acceleration simulation results."""
        if not self.results or 'results' not in self.results:
            print("No acceleration results to visualize")
            return
        
        # Plot results
        self.simulator.plot_acceleration_results(
            self.results['results'],
            plot_wheel_slip=True,
            save_path=os.path.join(self.output_dir, "acceleration_results.png")
        )
        
        print(f"Acceleration visualization saved to: {self.output_dir}")
    
    def export_results(self) -> None:
        """Export the acceleration simulation results to files."""
        if not self.results or 'results' not in self.results:
            print("No acceleration results to export")
            return
        
        # Export results as CSV
        results = self.results['results']
        
        if 'time' in results and 'speed' in results:
            df = pd.DataFrame({
                'time': results['time'],
                'speed': results['speed'],
                'acceleration': results.get('acceleration', [0] * len(results['time'])),
                'distance': results.get('distance', [0] * len(results['time'])),
                'engine_rpm': results.get('engine_rpm', [0] * len(results['time'])),
                'gear': results.get('gear', [0] * len(results['time'])),
                'wheel_slip': results.get('wheel_slip', [0] * len(results['time']))
            })
            
            csv_path = os.path.join(self.output_dir, "acceleration_data.csv")
            df.to_csv(csv_path, index=False)
            
            # Export metrics
            metrics = self.results['metrics']
            metrics_df = pd.DataFrame([metrics])
            metrics_path = os.path.join(self.output_dir, "acceleration_metrics.csv")
            metrics_df.to_csv(metrics_path, index=False)
            
            print(f"Acceleration data exported to: {csv_path}")
            print(f"Acceleration metrics exported to: {metrics_path}")


class LapTimeEventSimulator(EventSimulator):
    """Simulator for lap time event."""
    
    def __init__(self, vehicle: Any, config: Dict, track_file: str):
        """
        Initialize the lap time event simulator.
        
        Args:
            vehicle: Vehicle model to simulate
            config: Configuration dictionary
            track_file: Path to track file
        """
        super().__init__(vehicle, config)
        self.track_file = track_file
        self.output_dir = config['output_paths']['lap_time']
        
        # Create simulator
        self.simulator = LapTimeSimulator(vehicle)
        
        # Load track
        self.simulator.load_track(track_file)
    
    def run(self) -> Dict:
        """
        Run the lap time simulation.
        
        Returns:
            Dict: Lap time simulation results
        """
        print("\n--- Running Lap Time Simulation ---")
        
        # Run the lap simulation
        start_time = time.time()
        results = self.simulator.simulate_lap(include_thermal=True)
        end_time = time.time()
        
        # Store results
        self.results = results
        self.results['simulation_time'] = end_time - start_time
        
        # Print key results
        print(f"Simulation completed in {end_time - start_time:.2f}s")
        print(f"Lap Time: {results['lap_time']:.3f}s")
        print(f"Average Speed: {results['metrics']['avg_speed_kph']:.1f} km/h")
        print(f"Maximum Speed: {results['metrics']['max_speed_kph']:.1f} km/h")
        
        if results['metrics']['thermal_limited']:
            print("Note: Vehicle was thermally limited during the lap")
        
        return self.results
    
    def visualize_results(self) -> None:
        """Visualize the lap time simulation results."""
        if not self.results:
            print("No lap time results to visualize")
            return
        
        # Plot speed profile
        if 'results' in self.results and 'distance' in self.results['results'] and 'speed' in self.results['results']:
            plt.figure(figsize=(10, 6))
            
            distance = self.results['results']['distance']
            speed = self.results['results']['speed'] * 3.6  # Convert to km/h
            
            plt.plot(distance, speed, 'b-', linewidth=2)
            plt.xlabel('Distance (m)')
            plt.ylabel('Speed (km/h)')
            plt.title('Speed Profile')
            plt.grid(True, alpha=0.3)
            
            plt.savefig(os.path.join(self.output_dir, "lap_speed_profile.png"), dpi=300, bbox_inches='tight')
            plt.close()
        
        # Plot track with racing line if available
        try:
            plot_track_layout(
                self.simulator.track,
                racing_line=self.simulator.racing_line,
                color_by_speed=True,
                show_segments=True,
                save_path=os.path.join(self.output_dir, "track_layout.png")
            )
        except Exception as e:
            print(f"Could not plot track layout: {str(e)}")
        
        print(f"Lap time visualization saved to: {self.output_dir}")
    
    def export_results(self) -> None:
        """Export the lap time simulation results to files."""
        if not self.results:
            print("No lap time results to export")
            return
        
        # Export results as CSV
        if 'results' in self.results:
            results = self.results['results']
            
            # Create DataFrame from results
            data = {}
            
            # Add all available data series
            for key in ['distance', 'time', 'speed', 'lateral_acceleration', 
                       'engine_rpm', 'gear', 'throttle', 'engine_temp', 'coolant_temp']:
                if key in results:
                    data[key] = results[key]
            
            if data:
                df = pd.DataFrame(data)
                csv_path = os.path.join(self.output_dir, "lap_time_data.csv")
                df.to_csv(csv_path, index=False)
                
                # Export metrics
                metrics = self.results['metrics']
                metrics_df = pd.DataFrame([metrics])
                metrics_path = os.path.join(self.output_dir, "lap_time_metrics.csv")
                metrics_df.to_csv(metrics_path, index=False)
                
                print(f"Lap time data exported to: {csv_path}")
                print(f"Lap time metrics exported to: {metrics_path}")


class EnduranceEventSimulator(EventSimulator):
    """Simulator for endurance event."""
    
    def __init__(self, vehicle: Any, config: Dict, track_file: str):
        """
        Initialize the endurance event simulator.
        
        Args:
            vehicle: Vehicle model to simulate
            config: Configuration dictionary
            track_file: Path to track file
        """
        super().__init__(vehicle, config)
        self.track_file = track_file
        self.output_dir = config['output_paths']['endurance']
        
        # Get number of laps from configuration
        self.num_laps = config['simulation_settings'].get('endurance_laps', 3)
        
        # Create simulator
        self.simulator = EnduranceSimulator(vehicle)
        
        # Load track
        self.simulator.lap_simulator.load_track(track_file)
        
        # Configure for specified number of laps
        self.simulator.configure_event(num_laps=self.num_laps, driver_change_lap=0)
    
    def run(self) -> Dict:
        """
        Run the endurance simulation.
        
        Returns:
            Dict: Endurance simulation results
        """
        print(f"\n--- Running Endurance Simulation ({self.num_laps} laps) ---")
        
        # Run simulation
        start_time = time.time()
        results = self.simulator.simulate_endurance(include_thermal=True)
        end_time = time.time()
        
        # Calculate score
        score = self.simulator.calculate_score(results)
        
        # Store results
        self.results = {
            'results': results,
            'score': score,
            'simulator': self.simulator,
            'simulation_time': end_time - start_time
        }
        
        # Print key results
        print(f"Simulation completed in {end_time - start_time:.2f}s")
        
        if results['completed']:
            print(f"Endurance completed successfully")
            print(f"Total Time: {results['total_time']:.2f}s")
            print(f"Average Lap Time: {results['average_lap']:.2f}s")
            print(f"Fuel Used: {results['total_fuel']:.2f}L")
        else:
            print(f"Endurance not completed: {results['dnf_reason']}")
            print(f"Completed {results['lap_count']} of {self.num_laps} laps")
        
        return self.results
    
    def visualize_results(self) -> None:
        """Visualize the endurance simulation results."""
        if not self.results or 'results' not in self.results:
            print("No endurance results to visualize")
            return
        
        # Generate report with visualizations
        self.simulator.generate_endurance_report(
            self.results['results'], 
            self.results['score'], 
            self.output_dir
        )
        
        # Plot lap times
        results = self.results['results']
        if 'lap_times' in results and results['lap_times']:
            plt.figure(figsize=(10, 6))
            
            lap_numbers = range(1, len(results['lap_times']) + 1)
            lap_times = results['lap_times']
            
            plt.bar(lap_numbers, lap_times, color='blue', alpha=0.7)
            plt.axhline(y=results['average_lap'], color='r', linestyle='--', alpha=0.7, 
                      label=f"Avg: {results['average_lap']:.2f}s")
            
            plt.xlabel('Lap Number')
            plt.ylabel('Lap Time (s)')
            plt.title('Endurance Lap Times')
            plt.grid(True, axis='y', alpha=0.3)
            plt.legend()
            
            plt.savefig(os.path.join(self.output_dir, "endurance_lap_times.png"), dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"Endurance visualization saved to: {self.output_dir}")
    
    def export_results(self) -> None:
        """Export the endurance simulation results to files."""
        if not self.results or 'results' not in self.results:
            print("No endurance results to export")
            return
        
        # Export results as CSV
        results = self.results['results']
        
        # Create DataFrame for lap times
        if 'lap_times' in results and results['lap_times']:
            lap_df = pd.DataFrame({
                'lap_number': range(1, len(results['lap_times']) + 1),
                'lap_time': results['lap_times'],
                'fuel_consumption': results.get('lap_fuel_consumption', [0] * len(results['lap_times']))
            })
            
            csv_path = os.path.join(self.output_dir, "endurance_lap_data.csv")
            lap_df.to_csv(csv_path, index=False)
            
            # Export summary
            summary = {
                'total_time': results.get('total_time', 0),
                'average_lap': results.get('average_lap', 0),
                'completed': results.get('completed', False),
                'lap_count': results.get('lap_count', 0),
                'total_fuel': results.get('total_fuel', 0),
                'fuel_efficiency': results.get('fuel_efficiency', 0),
                'dnf_reason': results.get('dnf_reason', '')
            }
            
            # Add score if available
            if 'score' in self.results:
                score = self.results['score']
                summary.update({
                    'endurance_score': score.get('endurance_score', 0),
                    'efficiency_score': score.get('efficiency_score', 0),
                    'total_score': score.get('total_score', 0)
                })
            
            summary_df = pd.DataFrame([summary])
            summary_path = os.path.join(self.output_dir, "endurance_summary.csv")
            summary_df.to_csv(summary_path, index=False)
            
            print(f"Endurance lap data exported to: {csv_path}")
            print(f"Endurance summary exported to: {summary_path}")


# =========================================================================
# Analysis Manager
# =========================================================================

class AnalysisManager:
    """Manages various performance analyses."""
    
    def __init__(self, vehicle: Any, config: Dict, track_file: str):
        """
        Initialize the analysis manager.
        
        Args:
            vehicle: Vehicle model to analyze
            config: Configuration dictionary
            track_file: Path to track file
        """
        self.vehicle = vehicle
        self.config = config
        self.track_file = track_file
        self.analyses = {}
    
    def run_weight_sensitivity_analysis(self) -> Dict:
        """
        Run weight sensitivity analysis.
        
        Returns:
            Dict: Weight sensitivity analysis results
        """
        print("\n--- Weight Sensitivity Analysis ---")
        
        # Get output directory
        output_dir = self.config['output_paths']['weight_sensitivity']
        
        # Get weight range from configuration
        weight_range = self.config['optimization_settings'].get('weight_range', [180, 250])
        
        # Create analyzer
        analyzer = WeightSensitivityAnalyzer(self.vehicle)
        
        # Analyze acceleration sensitivity
        accel_sensitivity = analyzer.analyze_acceleration_sensitivity(
            weight_range=tuple(weight_range),
            num_points=5,
            use_launch_control=True,
            use_optimized_shifts=True
        )
        
        # Analyze lap time sensitivity
        lap_sensitivity = analyzer.analyze_lap_time_sensitivity(
            track_file=self.track_file,
            weight_range=tuple(weight_range),
            num_points=5,
            include_thermal=True
        )
        
        # Calculate optimal weight for lap time
        current_lap_time = lap_sensitivity['lap_times'][0]
        lap_target = current_lap_time * 0.93  # 7% improvement
        
        lap_reduction = analyzer.calculate_weight_reduction_targets(
            lap_target,
            sensitivity=lap_sensitivity['sensitivity_lap_time'],
            performance_type='lap_time'
        )
        
        # Plot sensitivity curves
        analyzer.plot_weight_sensitivity_curves(save_path=os.path.join(output_dir, "weight_sensitivity_curves.png"))
        
        # Print key findings
        print("\nWeight Sensitivity Key Findings:")
        print(f"  Acceleration Sensitivity: {accel_sensitivity['seconds_per_10kg_75m']:.3f} seconds per 10kg")
        print(f"  Lap Time Sensitivity: {lap_sensitivity['seconds_per_10kg_lap']:.3f} seconds per 10kg")
        
        if lap_reduction:
            print(f"\nLap Time Optimization Target:")
            print(f"  To achieve {lap_target:.2f}s lap time:")
            print(f"    Required weight reduction: {lap_reduction['required_weight_reduction']:.1f} kg")
            print(f"    Target weight: {lap_reduction['target_weight']:.1f} kg")
            print(f"    Achievable: {'Yes' if lap_reduction['is_achievable'] else 'No'}")
        
        # Store and return results
        results = {
            'accel_sensitivity': accel_sensitivity,
            'lap_sensitivity': lap_sensitivity,
            'lap_reduction_target': lap_reduction
        }
        
        self.analyses['weight_sensitivity'] = results
        return results
    
    def run_cooling_comparison(self) -> Dict:
        """
        Compare different cooling system configurations.
        
        Returns:
            Dict: Cooling comparison results
        """
        print("\n--- Cooling Configuration Comparison ---")
        
        # Get output directory
        output_dir = self.config['output_paths']['cooling_comparison']
        
        # Define configurations to test
        cooling_configs = self.config['optimization_settings'].get('cooling_configurations', 
                                                                 ['standard', 'optimized', 'minimal'])
        
        config_descriptions = {
            'standard': 'Standard cooling system',
            'optimized': 'Optimized cooling system with side pods and rear radiator',
            'minimal': 'Minimum weight cooling system'
        }
        
        # List to store results
        results = []
        
        # Test each configuration with a lap simulation
        for config_name in cooling_configs:
            desc = config_descriptions.get(config_name, config_name)
            print(f"\nTesting {desc}...")
            
            # Create vehicle with this cooling configuration
            vehicle = VehicleFactory.create_vehicle(self.config, cooling_config=config_name)
            
            # Create lap simulator
            lap_simulator = LapTimeSimulator(vehicle)
            lap_simulator.load_track(self.track_file)
            
            # Simulate lap
            start_time = time.time()
            lap_results = lap_simulator.simulate_lap(include_thermal=True)
            sim_time = time.time() - start_time
            
            print(f"  Lap simulation completed in {sim_time:.2f}s")
            print(f"  Lap Time: {lap_results['lap_time']:.3f}s")
            
            # Extract thermal data
            thermal_data = None
            if 'results' in lap_results and 'engine_temp' in lap_results['results']:
                thermal_data = {
                    'distance': lap_results['results'].get('distance', []),
                    'engine_temp': lap_results['results'].get('engine_temp', []),
                    'coolant_temp': lap_results['results'].get('coolant_temp', []) if 'coolant_temp' in lap_results['results'] else None
                }
            
            # Calculate performance metrics
            metrics = {
                'configuration': config_name,
                'description': desc,
                'lap_time': lap_results['lap_time'],
                'max_speed': lap_results['metrics'].get('max_speed_kph', 0),
                'avg_speed': lap_results['metrics'].get('avg_speed_kph', 0),
                'thermal_limited': lap_results['metrics'].get('thermal_limited', False)
            }
            
            # Add thermal metrics if available
            if thermal_data and len(thermal_data['engine_temp']) > 0:
                metrics.update({
                    'max_engine_temp': max(thermal_data['engine_temp']),
                    'avg_engine_temp': sum(thermal_data['engine_temp']) / len(thermal_data['engine_temp']),
                    'temp_rise': max(thermal_data['engine_temp']) - thermal_data['engine_temp'][0]
                })
                
                if thermal_data['coolant_temp'] is not None:
                    metrics.update({
                        'max_coolant_temp': max(thermal_data['coolant_temp']),
                        'avg_coolant_temp': sum(thermal_data['coolant_temp']) / len(thermal_data['coolant_temp'])
                    })
            
            # Add to results list
            results.append(metrics)
            
            # Save the thermal data for plotting
            if thermal_data and len(thermal_data['distance']) > 0:
                np.savez(
                    os.path.join(output_dir, f"{config_name}_thermal_data.npz"),
                    distance=thermal_data['distance'],
                    engine_temp=thermal_data['engine_temp'],
                    coolant_temp=thermal_data['coolant_temp'] if thermal_data['coolant_temp'] is not None else []
                )
        
        # Create comparison plot
        self._create_cooling_comparison_plot(results, cooling_configs, output_dir)
        
        # Create a summary table
        print("\nCooling Configuration Comparison Summary:")
        print("-" * 80)
        print(f"{'Configuration':<15} | {'Lap Time':<10} | {'Max Temp':<10} | {'Thermal Limited':<15} | {'Temp Rise':<10}")
        print("-" * 80)
        
        for r in results:
            print(f"{r['configuration']:<15} | {r['lap_time']:<10.3f} | {r.get('max_engine_temp', 'N/A'):<10} | {str(r['thermal_limited']):<15} | {r.get('temp_rise', 'N/A'):<10.1f}")
        
        print("-" * 80)
        
        # Calculate composite scores
        lap_times = [r['lap_time'] for r in results]
        
        if all('max_engine_temp' in r for r in results):
            for r in results:
                lap_time_score = min(lap_times) / r['lap_time']
                thermal_score = 1.0 - (r['max_engine_temp'] / max(r['max_engine_temp'] for r in results))
                r['composite_score'] = 0.3 * lap_time_score + 0.7 * thermal_score
            
            # Find the best configuration
            best_config = max(results, key=lambda r: r['composite_score'])
            print(f"\nBest overall configuration: {best_config['configuration']} (score: {best_config['composite_score']:.3f})")
        
        # Save comparison results
        pd.DataFrame(results).to_csv(os.path.join(output_dir, 'cooling_comparison.csv'), index=False)
        
        # Store and return results
        comparison_results = {
            'configurations': cooling_configs,
            'results': results,
            'output_dir': output_dir
        }
        
        self.analyses['cooling_comparison'] = comparison_results
        return comparison_results
    
    def _create_cooling_comparison_plot(self, results: List[Dict], cooling_configs: List[str], output_dir: str) -> None:
        """
        Create cooling comparison plot.
        
        Args:
            results: List of result dictionaries
            cooling_configs: List of cooling configuration names
            output_dir: Output directory path
        """
        plt.figure(figsize=(12, 10))
        plt.suptitle('Cooling Configuration Comparison', fontsize=16)
        
        # Create subplots for different metrics
        plt.subplot(2, 2, 1)
        names = [r['configuration'] for r in results]
        lap_times = [r['lap_time'] for r in results]
        plt.bar(names, lap_times, color=['blue', 'green', 'red'])
        plt.ylabel('Lap Time (s)')
        plt.title('Lap Time Comparison')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Maximum engine temperature
        if all('max_engine_temp' in r for r in results):
            plt.subplot(2, 2, 2)
            max_temps = [r['max_engine_temp'] for r in results]
            plt.bar(names, max_temps, color=['blue', 'green', 'red'])
            plt.ylabel('Maximum Engine Temperature (°C)')
            plt.title('Maximum Temperature Comparison')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Add warning temperature line if available
            warning_temp = self.config.get('thermal_limits', {}).get('engine', {}).get('warning_temp', 110.0)
            plt.axhline(y=warning_temp, color='orange', linestyle='--', alpha=0.7, label=f'Warning ({warning_temp}°C)')
            plt.legend()
        
        # Temperature rise
        if all('temp_rise' in r for r in results):
            plt.subplot(2, 2, 3)
            temp_rises = [r['temp_rise'] for r in results]
            plt.bar(names, temp_rises, color=['blue', 'green', 'red'])
            plt.ylabel('Temperature Rise (°C)')
            plt.title('Temperature Rise Comparison')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Thermal profiles
        plt.subplot(2, 2, 4)
        for config_name in cooling_configs:
            npz_path = os.path.join(output_dir, f"{config_name}_thermal_data.npz")
            if os.path.exists(npz_path):
                data = np.load(npz_path)
                plt.plot(data['distance'], data['engine_temp'], label=config_name)
        
        plt.xlabel('Distance (m)')
        plt.ylabel('Engine Temperature (°C)')
        plt.title('Thermal Profiles')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Save the plot
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(output_dir, 'cooling_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def run_lap_optimization(self, method: str = 'advanced') -> Dict:
        """
        Run lap time optimization.
        
        Args:
            method: Optimization method ('basic', 'advanced', or 'compare')
            
        Returns:
            Dict: Lap optimization results
        """
        print(f"\n--- Lap Time Optimization ({method}) ---")
        
        # Get output directory
        output_dir = self.config['output_paths']['lap_optimization']
        
        # Get config file path
        config_file = self.config['paths'].get('lap_time_optimization')
        
        # Run lap time optimization
        try:
            if method == 'compare':
                # Compare basic and advanced optimization methods
                results = compare_optimization_methods(
                    self.vehicle,
                    self.track_file,
                    config_file=config_file,
                    include_thermal=True,
                    save_dir=output_dir
                )
                
                # Print comparison results
                if 'basic' in results and 'advanced' in results:
                    print("\nOptimization Methods Comparison:")
                    print(f"  Basic Method Lap Time: {results['basic']['lap_time']:.3f}s")
                    print(f"  Advanced Method Lap Time: {results['advanced']['lap_time']:.3f}s")
                    print(f"  Difference: {results['difference']['lap_time_diff']:.3f}s "
                        f"({results['difference']['lap_time_percent']:.2f}%)")
                
            else:
                # Run either basic or advanced optimization
                results = run_lap_optimization(
                    self.vehicle,
                    self.track_file,
                    method=method,
                    config_file=config_file,
                    include_thermal=True,
                    save_dir=output_dir
                )
                
                # Print optimization results
                print(f"\nLap Optimization Results ({method}):")
                if 'lap_time' in results:
                    print(f"  Optimized Lap Time: {results['lap_time']:.3f}s")
                
                if method == 'advanced' and 'optimization_success' in results:
                    print(f"  Optimization Success: {results['optimization_success']}")
                    print(f"  Optimization Time: {results.get('optimization_time', 0):.1f}s")
        
        except Exception as e:
            print(f"Error in lap time optimization: {str(e)}")
            results = {'error': str(e)}
        
        # Store and return results
        optimization_results = {
            'results': results,
            'method': method,
            'output_dir': output_dir
        }
        
        self.analyses['lap_optimization'] = optimization_results
        return optimization_results
    
    def calculate_optimal_racing_lines(self) -> Dict:
        """
        Calculate and visualize optimal racing lines for a track.
        
        Returns:
            Dict: Racing line calculation results
        """
        print("\n--- Optimal Racing Line Calculation ---")
        
        # Get output directory
        output_dir = self.config['output_paths']['lap_optimization']
        
        try:
            # Load track profile
            track_profile = TrackProfile(self.track_file)
            
            # Calculate basic racing line
            basic_racing_line = calculate_optimal_racing_line(track_profile)
            
            # Create optimal lap time optimizer for advanced racing line
            optimizer = OptimalLapTimeOptimizer(self.vehicle, track_profile)
            
            # Reduce complexity for faster demonstration
            optimizer.num_control_points = 30  # Fewer control points for faster optimization
            optimizer.max_iterations = 10     # Fewer iterations for demonstration
            
            # Run optimization
            print("Running advanced racing line optimization...")
            advanced_results = optimizer.optimize_lap_time()
            
            # Get the advanced racing line
            advanced_racing_line = advanced_results.get('racing_line')
            
            # Visualize both racing lines
            if basic_racing_line is not None and advanced_racing_line is not None:
                # Plot racing lines
                self._plot_racing_lines(track_profile, basic_racing_line, advanced_racing_line, 
                                      advanced_results, output_dir)
            
            # Calculate lap time using the optimized racing line
            lap_time = advanced_results.get('lap_time')
            
            print(f"Racing line calculation completed")
            if lap_time:
                print(f"  Optimized Lap Time: {lap_time:.3f}s")
            
            # Store and return results
            racing_line_results = {
                'basic_racing_line': basic_racing_line,
                'advanced_racing_line': advanced_racing_line,
                'advanced_results': advanced_results,
                'lap_time': lap_time,
                'output_dir': output_dir
            }
            
            self.analyses['racing_lines'] = racing_line_results
            return racing_line_results
            
        except Exception as e:
            print(f"Error in racing line calculation: {str(e)}")
            racing_line_results = {'error': str(e)}
            self.analyses['racing_lines'] = racing_line_results
            return racing_line_results
    
    def _plot_racing_lines(self, track_profile: Any, basic_racing_line: np.ndarray, 
                         advanced_racing_line: np.ndarray, advanced_results: Dict, output_dir: str) -> None:
        """
        Plot racing line comparison.
        
        Args:
            track_profile: TrackProfile object
            basic_racing_line: Basic racing line array
            advanced_racing_line: Advanced racing line array
            advanced_results: Advanced optimization results
            output_dir: Output directory path
        """
        # Get track data for visualization
        track_data = track_profile.get_track_data()
        track_points = track_data['points']
        
        plt.figure(figsize=(12, 10))
        plt.subplot(2, 1, 1)
        
        # Plot track outline
        plt.plot(track_points[:, 0], track_points[:, 1], 'k--', alpha=0.5, label='Track Centerline')
        
        # Plot basic racing line
        plt.plot(basic_racing_line[:, 0], basic_racing_line[:, 1], 'b-', linewidth=2, label='Basic Racing Line')
        
        # Plot advanced racing line
        plt.plot(advanced_racing_line[:, 0], advanced_racing_line[:, 1], 'r-', linewidth=2, label='Advanced Racing Line')
        
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.title('Racing Line Comparison')
        plt.legend()
        plt.axis('equal')
        plt.grid(True, alpha=0.3)
        
        # Plot speed profiles if available
        if 'vehicle_states' in advanced_results:
            plt.subplot(2, 1, 2)
            
            # Extract speeds from vehicle states
            states = advanced_results['vehicle_states']
            distances = [state.distance for state in states if hasattr(state, 'distance')]
            speeds = [state.speed * 3.6 for state in states if hasattr(state, 'speed')]  # m/s to km/h
            
            if distances and speeds:
                plt.plot(distances, speeds, 'r-', linewidth=2, label='Speed Profile')
                plt.xlabel('Distance (m)')
                plt.ylabel('Speed (km/h)')
                plt.title('Speed Profile Along Racing Line')
                plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'racing_line_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def analyze_performance_tradeoffs(self) -> Dict:
        """
        Analyze tradeoffs between weight, lap time, and thermal performance.
        
        Returns:
            Dict: Performance tradeoff analysis
        """
        print("\n--- Performance Tradeoff Analysis ---")
        
        # Extract required analyses
        weight_results = self.analyses.get('weight_sensitivity')
        lap_results = self.analyses.get('lap_optimization')
        cooling_results = self.analyses.get('cooling_comparison')
        
        tradeoffs = {
            'weight_vs_lap_time': {},
            'weight_vs_thermal': {},
            'cooling_vs_performance': {},
            'recommendations': []
        }
        
        # Extract data for analysis
        # Weight vs Lap Time tradeoff
        if (weight_results and 'lap_sensitivity' in weight_results and 
                'lap_times' in weight_results['lap_sensitivity']):
            
            weights = weight_results['lap_sensitivity']['weights']
            lap_times = weight_results['lap_sensitivity']['lap_times']
            
            # Calculate time improvement per kg
            if len(weights) > 1 and len(lap_times) > 1:
                time_per_kg = (lap_times[-1] - lap_times[0]) / (weights[-1] - weights[0])
                tradeoffs['weight_vs_lap_time'] = {
                    'time_per_kg': time_per_kg,
                    'percent_improvement_per_kg': (time_per_kg / lap_times[0]) * 100
                }
                
                print(f"Weight vs Lap Time Tradeoff:")
                print(f"  {tradeoffs['weight_vs_lap_time']['time_per_kg']:.4f} seconds per kg")
                print(f"  {tradeoffs['weight_vs_lap_time']['percent_improvement_per_kg']:.4f}% improvement per kg")
        
        # Weight vs Thermal tradeoff
        # We need to estimate this based on available data
        if (weight_results and cooling_results and 'results' in cooling_results):
            cooling_configs = cooling_results['results']
            
            # First, find a reasonable estimate of thermal impact per kg
            # This is simplified and would be more accurate with experimental data
            try:
                # If we can find cooling configurations with different weights
                if len(cooling_configs) >= 2 and all('configuration' in cfg for cfg in cooling_configs):
                    std_config = next((cfg for cfg in cooling_configs if cfg['configuration'] == 'standard'), None)
                    min_config = next((cfg for cfg in cooling_configs if cfg['configuration'] == 'minimal'), None)
                    
                    if std_config and min_config and 'max_engine_temp' in std_config and 'max_engine_temp' in min_config:
                        # Estimate weight difference between configurations (this is approximate)
                        weight_diff = 5.0  # Assume ~5kg difference between standard and minimal
                        temp_diff = min_config['max_engine_temp'] - std_config['max_engine_temp']
                        
                        temp_per_kg = temp_diff / weight_diff
                        
                        tradeoffs['weight_vs_thermal'] = {
                            'temp_per_kg': temp_per_kg,
                            'weight_estimate_difference': weight_diff
                        }
                        
                        print(f"\nWeight vs Thermal Tradeoff:")
                        print(f"  {tradeoffs['weight_vs_thermal']['temp_per_kg']:.2f}°C increase per kg reduction")
            except Exception as e:
                print(f"Could not calculate weight vs thermal tradeoff: {str(e)}")
        
        # Cooling vs Performance tradeoff
        if cooling_results and 'results' in cooling_results and cooling_results['results']:
            cooling_configs = cooling_results['results']
            
            if len(cooling_configs) >= 2:
                # Find the configs with the best lap time and the best thermal performance
                best_lap_config = min(cooling_configs, key=lambda x: x.get('lap_time', float('inf')))
                best_thermal_config = min(cooling_configs, key=lambda x: x.get('max_engine_temp', float('inf')))
                
                if (best_lap_config != best_thermal_config and 
                    'lap_time' in best_lap_config and 'lap_time' in best_thermal_config and
                    'max_engine_temp' in best_lap_config and 'max_engine_temp' in best_thermal_config):
                    
                    lap_time_diff = best_thermal_config['lap_time'] - best_lap_config['lap_time']
                    temp_diff = best_lap_config['max_engine_temp'] - best_thermal_config['max_engine_temp']
                    
                    tradeoffs['cooling_vs_performance'] = {
                        'lap_time_diff': lap_time_diff,
                        'temp_diff': temp_diff,
                        'time_per_degree': lap_time_diff / temp_diff if temp_diff != 0 else 0
                    }
                    
                    print(f"\nCooling vs Performance Tradeoff:")
                    print(f"  {tradeoffs['cooling_vs_performance']['lap_time_diff']:.3f}s lap time penalty for "
                          f"{tradeoffs['cooling_vs_performance']['temp_diff']:.1f}°C temperature reduction")
                    print(f"  {tradeoffs['cooling_vs_performance']['time_per_degree']:.4f}s per °C")
        
        # Generate recommendations based on tradeoff analysis
        recommendations = []
        
        # Weight recommendation
        if 'weight_vs_lap_time' in tradeoffs and tradeoffs['weight_vs_lap_time']:
            if abs(tradeoffs['weight_vs_lap_time']['time_per_kg']) > 0.01:
                recommendations.append("Weight reduction should be a high priority given the significant lap time improvement per kg.")
            else:
                recommendations.append("Weight reduction has moderate impact on lap time. Balance with other priorities.")
        
        # Cooling recommendation
        if 'cooling_vs_performance' in tradeoffs and tradeoffs['cooling_vs_performance']:
            time_penalty = tradeoffs['cooling_vs_performance'].get('lap_time_diff', 0)
            if time_penalty < 0.1:
                recommendations.append("Prioritize thermal optimization as it has minimal impact on lap time but provides reliability benefits.")
            elif time_penalty < 0.3:
                recommendations.append("Consider the optimized cooling configuration for better reliability with moderate lap time impact.")
            else:
                recommendations.append("The standard cooling configuration offers the best balance between performance and reliability.")
        
        # Overall vehicle configuration recommendation
        if weight_results and 'lap_reduction_target' in weight_results:
            lap_reduction = weight_results['lap_reduction_target']
            if lap_reduction and lap_reduction.get('is_achievable', False):
                target_weight = lap_reduction.get('target_weight', 0)
                recommendations.append(f"Target vehicle weight of {target_weight:.1f}kg is recommended to achieve optimal lap time.")
        
        tradeoffs['recommendations'] = recommendations
        
        print("\nRecommendations based on Performance Tradeoffs:")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")
        
        # Store and return results
        self.analyses['tradeoffs'] = tradeoffs
        return tradeoffs


# =========================================================================
# Report Generator
# =========================================================================

class ReportGenerator:
    """Generates comprehensive reports from simulation results."""
    
    def __init__(self, config: Dict):
        """
        Initialize the report generator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.output_dir = config['output_paths']['reports']
        
    def generate_summary_report(self, simulation_results: Dict) -> str:
        """
        Generate a summary report of all simulation results.
        
        Args:
            simulation_results: Dictionary containing all simulation results
            
        Returns:
            str: Path to the generated report
        """
        print("\n--- Generating Summary Report ---")
        
        # Create summary data structure
        summary = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'vehicle': {},
            'acceleration': {},
            'lap_time': {},
            'endurance': {},
            'thermal': {},
            'weight_sensitivity': {},
            'optimization': {},
            'recommendations': []
        }
        
        # Add vehicle specifications if available
        if 'vehicle' in simulation_results and simulation_results['vehicle']:
            vehicle = simulation_results['vehicle']
            if hasattr(vehicle, 'get_vehicle_specs'):
                specs = vehicle.get_vehicle_specs()
                summary['vehicle'] = {
                    'mass': specs['vehicle'].get('mass', 0),
                    'engine': f"{specs['engine'].get('make', '')} {specs['engine'].get('model', '')}",
                    'max_power': specs['engine'].get('max_power_hp', 0),
                    'max_torque': specs['engine'].get('max_torque_nm', 0),
                    'cooling_system': getattr(vehicle, 'cooling_system', None) is not None,
                    'side_pods': getattr(vehicle, 'side_pod_system', None) is not None
                }
        
        # Add acceleration results if available
        if 'acceleration' in simulation_results and simulation_results['acceleration']:
            accel_results = simulation_results['acceleration']
            if 'results' in accel_results and 'metrics' in accel_results:
                results = accel_results['results']
                metrics = accel_results['metrics']
                
                summary['acceleration'] = {
                    '75m_time': results.get('finish_time', 0),
                    'time_to_60mph': results.get('time_to_60mph', 0),
                    'time_to_100kph': results.get('time_to_100kph', 0),
                    'finish_speed': results.get('finish_speed', 0) * 3.6,  # m/s to km/h
                    'performance_grade': metrics.get('performance_grade', '')
                }
        
        # Add lap time results if available
        if 'lap_time' in simulation_results and simulation_results['lap_time']:
            lap_results = simulation_results['lap_time']
            
            summary['lap_time'] = {
                'lap_time': lap_results.get('lap_time', 0),
                'avg_speed': lap_results.get('metrics', {}).get('avg_speed_kph', 0),
                'max_speed': lap_results.get('metrics', {}).get('max_speed_kph', 0),
                'thermal_limited': lap_results.get('metrics', {}).get('thermal_limited', False)
            }
        
        # Add endurance results if available
        if 'endurance' in simulation_results and simulation_results['endurance']:
            endurance_results = simulation_results['endurance']
            if 'results' in endurance_results and 'score' in endurance_results:
                results = endurance_results['results']
                score = endurance_results['score']
                
                summary['endurance'] = {
                    'completed': results.get('completed', False),
                    'total_time': results.get('total_time', 0),
                    'average_lap': results.get('average_lap', 0),
                    'total_fuel': results.get('total_fuel', 0),
                    'endurance_score': score.get('endurance_score', 0),
                    'efficiency_score': score.get('efficiency_score', 0)
                }
        
        # Add thermal analysis if available
        if 'thermal_analysis' in simulation_results and simulation_results['thermal_analysis']:
            thermal_analysis = simulation_results['thermal_analysis']
            
            # Extract relevant metrics
            if 'endurance' in thermal_analysis:
                summary['thermal'] = {
                    'max_engine_temp': thermal_analysis['endurance'].get('max_engine_temp', 0),
                    'max_coolant_temp': thermal_analysis['endurance'].get('max_coolant_temp', 0),
                    'thermal_limited': thermal_analysis['endurance'].get('thermal_limited', False),
                    'cooling_status': thermal_analysis.get('cooling_effectiveness', {}).get('status', '')
                }
        
        # Add weight sensitivity results if available
        if 'weight_sensitivity' in simulation_results and simulation_results['weight_sensitivity']:
            weight_results = simulation_results['weight_sensitivity']
            
            if 'accel_sensitivity' in weight_results and 'lap_sensitivity' in weight_results:
                summary['weight_sensitivity'] = {
                    'accel_sensitivity': weight_results['accel_sensitivity'].get('seconds_per_10kg_75m', 0),
                    'lap_sensitivity': weight_results['lap_sensitivity'].get('seconds_per_10kg_lap', 0),
                    'target_weight': weight_results.get('lap_reduction_target', {}).get('target_weight', 0)
                }
        
        # Add lap optimization results if available
        if 'lap_optimization' in simulation_results and simulation_results['lap_optimization']:
            opt_results = simulation_results['lap_optimization']
            
            if 'results' in opt_results and isinstance(opt_results['results'], dict):
                results = opt_results['results']
                summary['optimization'] = {
                    'optimized_lap_time': results.get('lap_time', 0),
                    'optimization_method': opt_results.get('method', 'unknown'),
                    'improvement': results.get('improvement', 0)
                }
        
        # Add recommendations if available
        if 'tradeoffs' in simulation_results and simulation_results['tradeoffs']:
            tradeoffs = simulation_results['tradeoffs']
            
            if 'recommendations' in tradeoffs:
                summary['recommendations'] = tradeoffs['recommendations']
        
        # Generate CSV report
        csv_path = os.path.join(self.output_dir, "simulation_summary.csv")
        
        # Flatten and export to CSV
        flat_summary = self._flatten_dict(summary)
        pd.DataFrame([flat_summary]).to_csv(csv_path, index=False)
        
        # Generate JSON report
        json_path = os.path.join(self.output_dir, "simulation_summary.json")
        
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=4)
        
        # Generate text report
        txt_path = os.path.join(self.output_dir, "simulation_summary.txt")
        
        with open(txt_path, 'w') as f:
            f.write(f"Formula Student Simulation Summary\n")
            f.write(f"Generated: {summary['timestamp']}\n")
            f.write(f"=" * 80 + "\n\n")
            
            # Write vehicle info
            if summary['vehicle']:
                f.write(f"Vehicle Information:\n")
                f.write(f"  Engine: {summary['vehicle'].get('engine', 'Unknown')}\n")
                f.write(f"  Mass: {summary['vehicle'].get('mass', 0):.1f} kg\n")
                f.write(f"  Maximum Power: {summary['vehicle'].get('max_power', 0):.1f} hp\n")
                f.write(f"  Maximum Torque: {summary['vehicle'].get('max_torque', 0):.1f} Nm\n")
                f.write(f"\n")
            
            # Write acceleration results
            if summary['acceleration']:
                f.write(f"Acceleration Performance:\n")
                f.write(f"  75m Time: {summary['acceleration'].get('75m_time', 0):.3f} s\n")
                f.write(f"  0-60 mph: {summary['acceleration'].get('time_to_60mph', 0):.3f} s\n")
                f.write(f"  0-100 km/h: {summary['acceleration'].get('time_to_100kph', 0):.3f} s\n")
                f.write(f"  Final Speed: {summary['acceleration'].get('finish_speed', 0):.1f} km/h\n")
                f.write(f"  Performance Grade: {summary['acceleration'].get('performance_grade', '')}\n")
                f.write(f"\n")
            
            # Write lap time results
            if summary['lap_time']:
                f.write(f"Lap Time Performance:\n")
                f.write(f"  Lap Time: {summary['lap_time'].get('lap_time', 0):.3f} s\n")
                f.write(f"  Average Speed: {summary['lap_time'].get('avg_speed', 0):.1f} km/h\n")
                f.write(f"  Maximum Speed: {summary['lap_time'].get('max_speed', 0):.1f} km/h\n")
                f.write(f"  Thermally Limited: {'Yes' if summary['lap_time'].get('thermal_limited', False) else 'No'}\n")
                f.write(f"\n")
            
            # Write endurance results
            if summary['endurance']:
                f.write(f"Endurance Performance:\n")
                f.write(f"  Completed: {'Yes' if summary['endurance'].get('completed', False) else 'No'}\n")
                f.write(f"  Total Time: {summary['endurance'].get('total_time', 0):.2f} s\n")
                f.write(f"  Average Lap: {summary['endurance'].get('average_lap', 0):.2f} s\n")
                f.write(f"  Total Fuel: {summary['endurance'].get('total_fuel', 0):.2f} L\n")
                f.write(f"  Endurance Score: {summary['endurance'].get('endurance_score', 0):.1f}\n")
                f.write(f"  Efficiency Score: {summary['endurance'].get('efficiency_score', 0):.1f}\n")
                f.write(f"\n")
            
            # Write thermal analysis
            if summary['thermal']:
                f.write(f"Thermal Performance:\n")
                f.write(f"  Maximum Engine Temperature: {summary['thermal'].get('max_engine_temp', 0):.1f}°C\n")
                f.write(f"  Maximum Coolant Temperature: {summary['thermal'].get('max_coolant_temp', 0):.1f}°C\n")
                f.write(f"  Thermally Limited: {'Yes' if summary['thermal'].get('thermal_limited', False) else 'No'}\n")
                f.write(f"  Cooling Status: {summary['thermal'].get('cooling_status', '')}\n")
                f.write(f"\n")
            
            # Write weight sensitivity
            if summary['weight_sensitivity']:
                f.write(f"Weight Sensitivity Analysis:\n")
                f.write(f"  Acceleration: {summary['weight_sensitivity'].get('accel_sensitivity', 0):.3f} s per 10kg\n")
                f.write(f"  Lap Time: {summary['weight_sensitivity'].get('lap_sensitivity', 0):.3f} s per 10kg\n")
                
                if 'target_weight' in summary['weight_sensitivity']:
                    f.write(f"  Target Weight: {summary['weight_sensitivity'].get('target_weight', 0):.1f} kg\n")
                
                f.write(f"\n")
            
            # Write optimization results
            if summary['optimization']:
                f.write(f"Lap Optimization Results:\n")
                f.write(f"  Method: {summary['optimization'].get('optimization_method', '')}\n")
                f.write(f"  Optimized Lap Time: {summary['optimization'].get('optimized_lap_time', 0):.3f} s\n")
                f.write(f"  Improvement: {summary['optimization'].get('improvement', 0):.2f}%\n")
                f.write(f"\n")
            
            # Write recommendations
            if summary['recommendations']:
                f.write(f"Key Recommendations:\n")
                for i, rec in enumerate(summary['recommendations'], 1):
                    f.write(f"  {i}. {rec}\n")
                f.write(f"\n")
        
        print(f"Summary report generated at: {txt_path}")
        return txt_path
    
    def _flatten_dict(self, nested_dict: Dict, prefix: str = '') -> Dict:
        """
        Flatten a nested dictionary for CSV export.
        
        Args:
            nested_dict: Nested dictionary to flatten
            prefix: Prefix for keys in the flattened dictionary
            
        Returns:
            Dict: Flattened dictionary
        """
        flattened = {}
        
        for key, value in nested_dict.items():
            if isinstance(value, dict):
                # Recursively flatten nested dictionaries
                flattened.update(self._flatten_dict(value, f"{prefix}{key}_"))
            elif isinstance(value, list):
                # Convert lists to string representation
                flattened[f"{prefix}{key}"] = str(value)
            else:
                # Add scalar values
                flattened[f"{prefix}{key}"] = value
        
        return flattened


# =========================================================================
# Simulation Manager
# =========================================================================

class SimulationManager:
    """
    Main simulation manager that controls the overall simulation process.
    """
    
    def __init__(self, args: Optional[argparse.Namespace] = None):
        """
        Initialize the simulation manager.
        
        Args:
            args: Command line arguments (optional)
        """
        self.args = args
        self.config_manager = ConfigurationManager(args)
        self.config = None
        self.vehicle = None
        self.track_manager = None
        self.track_file = None
        self.simulation_results = {}
    
    def setup(self):
        """Set up the simulation environment."""
        print("Setting up simulation environment...")
        
        # Load configuration
        self.config = self.config_manager.load_configurations()
        
        # Create track manager
        self.track_manager = TrackManager(self.config)
        
        # Save configuration for reference
        self.config_manager.save_configuration()
    
    def create_vehicle(self):
        """Create the vehicle model."""
        print("\nCreating vehicle model...")
        
        # Create vehicle with standard cooling configuration
        self.vehicle = VehicleFactory.create_vehicle(self.config, cooling_config='standard')
        
        # Store vehicle in results
        self.simulation_results['vehicle'] = self.vehicle
    
    def create_track(self):
        """Create or generate the track."""
        print("\nCreating track...")
        
        # Check if we should generate tracks
        if self.config['simulation_settings'].get('generate_tracks', False):
            num_tracks = self.config['simulation_settings'].get('num_tracks', 3)
            track_mode = self.config['simulation_settings'].get('track_mode', 'EXTEND')
            
            # Generate tracks
            track_files = self.track_manager.generate_tracks(
                num_tracks=num_tracks,
                mode_str=track_mode,
                visualize=True
            )
            
            # Use the first track for simulations
            if track_files:
                self.track_file = track_files[0]
            else:
                # Fall back to example track if generation fails
                self.track_file = self.track_manager.create_example_track()
        else:
            # Create an example track
            self.track_file = self.track_manager.create_example_track()
    
    def run_event_simulations(self):
        """Run all selected event simulations."""
        sim_settings = self.config['simulation_settings']
        
        # Run acceleration simulation if enabled
        if sim_settings.get('run_acceleration', True):
            accel_simulator = AccelerationEventSimulator(self.vehicle, self.config)
            self.simulation_results['acceleration'] = accel_simulator.run()
            accel_simulator.visualize_results()
            accel_simulator.export_results()
        
        # Run lap time simulation if enabled
        if sim_settings.get('run_lap_time', True):
            lap_simulator = LapTimeEventSimulator(self.vehicle, self.config, self.track_file)
            self.simulation_results['lap_time'] = lap_simulator.run()
            lap_simulator.visualize_results()
            lap_simulator.export_results()
        
        # Run endurance simulation if enabled
        if sim_settings.get('run_endurance', True):
            endurance_simulator = EnduranceEventSimulator(self.vehicle, self.config, self.track_file)
            self.simulation_results['endurance'] = endurance_simulator.run()
            endurance_simulator.visualize_results()
            endurance_simulator.export_results()
    
    def run_performance_analyses(self):
        """Run all selected performance analyses."""
        sim_settings = self.config['simulation_settings']
        
        # Create analysis manager
        analysis_manager = AnalysisManager(self.vehicle, self.config, self.track_file)
        
        # Run thermal analysis
        if 'acceleration' in self.simulation_results or 'lap_time' in self.simulation_results or 'endurance' in self.simulation_results:
            from analyze_thermal_performance import analyze_thermal_performance
            self.simulation_results['thermal_analysis'] = analyze_thermal_performance(
                self.vehicle,
                self.simulation_results.get('acceleration', {}),
                self.simulation_results.get('lap_time', {}),
                self.simulation_results.get('endurance', {}),
                self.config['output_paths']['thermal_analysis']
            )
        
        # Run cooling comparison if enabled
        if sim_settings.get('run_cooling_comparison', True):
            self.simulation_results['cooling_comparison'] = analysis_manager.run_cooling_comparison()
        
        # Run weight sensitivity analysis if enabled
        if sim_settings.get('run_weight_sensitivity', True):
            self.simulation_results['weight_sensitivity'] = analysis_manager.run_weight_sensitivity_analysis()
        
        # Run lap time optimization if enabled
        if sim_settings.get('run_lap_optimization', True):
            self.simulation_results['lap_optimization'] = analysis_manager.run_lap_optimization(method='advanced')
            
            # Calculate optimal racing lines
            self.simulation_results['racing_lines'] = analysis_manager.calculate_optimal_racing_lines()
        
        # Analyze performance tradeoffs
        self.simulation_results['tradeoffs'] = analysis_manager.analyze_performance_tradeoffs()
    
    def generate_report(self):
        """Generate a comprehensive report of all simulation results."""
        # Create report generator
        report_generator = ReportGenerator(self.config)
        
        # Generate summary report
        report_path = report_generator.generate_summary_report(self.simulation_results)
        
        return report_path
    
    def print_conclusion(self):
        """Print a conclusion of the simulation results."""
        print("\n=== Overall Conclusion ===")
        
        # Add thermal performance conclusion
        if 'thermal_analysis' in self.simulation_results:
            thermal_analysis = self.simulation_results['thermal_analysis']
            if 'cooling_effectiveness' in thermal_analysis and 'status' in thermal_analysis['cooling_effectiveness']:
                print(f"Thermal Performance: {thermal_analysis['cooling_effectiveness']['status']}")
        
        # Add cooling comparison conclusion if available
        if 'cooling_comparison' in self.simulation_results:
            cooling_comparison = self.simulation_results['cooling_comparison']
            if 'results' in cooling_comparison:
                results = cooling_comparison['results']
                if results and all('composite_score' in r for r in results):
                    best_config = max(results, key=lambda r: r['composite_score'])
                    print(f"Recommended Cooling Configuration: {best_config['description']}")
        
        # Add weight sensitivity conclusion if available
        if 'weight_sensitivity' in self.simulation_results:
            weight_sensitivity = self.simulation_results['weight_sensitivity']
            if 'lap_reduction_target' in weight_sensitivity:
                target = weight_sensitivity['lap_reduction_target']
                if target:
                    print(f"Optimal Weight Target: {target.get('target_weight', 0):.1f} kg "
                         f"({target.get('required_weight_reduction', 0):.1f} kg reduction)")
        
        # Add lap optimization conclusion if available
        if 'lap_optimization' in self.simulation_results:
            lap_optimization = self.simulation_results['lap_optimization']
            if 'results' in lap_optimization:
                results = lap_optimization['results']
                if isinstance(results, dict) and 'lap_time' in results:
                    print(f"Optimized Lap Time: {results['lap_time']:.3f}s")
        
        # Add tradeoff recommendations
        if 'tradeoffs' in self.simulation_results:
            tradeoffs = self.simulation_results['tradeoffs']
            if 'recommendations' in tradeoffs:
                print("\nKey Recommendations:")
                for i, rec in enumerate(tradeoffs['recommendations'][:2], 1):  # Show top 2 recommendations
                    print(f"  {i}. {rec}")
    
    def run(self):
        """Run the complete simulation process."""
        try:
            # Setup
            self.setup()
            
            # Create vehicle
            self.create_vehicle()
            
            # Create track
            self.create_track()
            
            # Run event simulations
            self.run_event_simulations()
            
            # Run performance analyses
            self.run_performance_analyses()
            
            # Generate report
            self.generate_report()
            
            # Print conclusion
            self.print_conclusion()
            
            print("\nSimulation completed successfully!")
            print(f"Results saved to: {self.config['output_dir']}")
            return True
        
        except Exception as e:
            print(f"Error during simulation: {str(e)}")
            import traceback
            traceback.print_exc()
            return False


# =========================================================================
# Thermal Analysis Function
# =========================================================================

def analyze_thermal_performance(vehicle, accel_results, lap_results, endurance_results, output_dir):
    """
    Analyze thermal performance of the vehicle during different events.
    
    Args:
        vehicle: Vehicle model used in simulations
        accel_results: Results from acceleration simulation
        lap_results: Results from lap time simulation
        endurance_results: Results from endurance simulation
        output_dir: Directory to save analysis results
        
    Returns:
        dict: Thermal performance analysis metrics
    """
    print("\n--- Thermal Performance Analysis ---")
    
    # Create output directory for thermal analysis
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize thermal analysis dictionary
    thermal_analysis = {
        'acceleration': {},
        'lap_time': {},
        'endurance': {},
        'comparison': {},
        'cooling_effectiveness': {}
    }
    
    # Extract thermal data from different simulations
    
    # Extract from acceleration results
    accel_thermal_data = None
    if accel_results and 'results' in accel_results:
        results = accel_results['results']
        if 'engine_temp' in results:
            accel_thermal_data = {
                'time': results.get('time', []),
                'engine_temp': results.get('engine_temp', []),
                'coolant_temp': results.get('coolant_temp', []) if 'coolant_temp' in results else None,
                'oil_temp': results.get('oil_temp', []) if 'oil_temp' in results else None
            }
            
            # Calculate metrics
            if len(accel_thermal_data['engine_temp']) > 0:
                thermal_analysis['acceleration'] = {
                    'max_engine_temp': max(accel_thermal_data['engine_temp']),
                    'temp_rise': max(accel_thermal_data['engine_temp']) - accel_thermal_data['engine_temp'][0],
                    'max_temp_rate': max(np.diff(accel_thermal_data['engine_temp']) / np.diff(accel_thermal_data['time'])) if len(accel_thermal_data['time']) > 1 else 0
                }
                
                print(f"Acceleration Event Thermal Analysis:")
                print(f"  Maximum Engine Temperature: {thermal_analysis['acceleration']['max_engine_temp']:.1f}°C")
                print(f"  Temperature Rise: {thermal_analysis['acceleration']['temp_rise']:.1f}°C")
                
                # Plot acceleration thermal data
                if len(accel_thermal_data['time']) > 1:
                    plt.figure(figsize=(10, 6))
                    plt.plot(accel_thermal_data['time'], accel_thermal_data['engine_temp'], 'r-', linewidth=2, label='Engine')
                    
                    if accel_thermal_data['coolant_temp'] is not None:
                        plt.plot(accel_thermal_data['time'], accel_thermal_data['coolant_temp'], 'b-', linewidth=2, label='Coolant')
                        
                    if accel_thermal_data['oil_temp'] is not None:
                        plt.plot(accel_thermal_data['time'], accel_thermal_data['oil_temp'], 'g-', linewidth=2, label='Oil')
                    
                    plt.xlabel('Time (s)')
                    plt.ylabel('Temperature (°C)')
                    plt.title('Thermal Profile during Acceleration')
                    plt.grid(True, linestyle='--', alpha=0.7)
                    plt.legend()
                    
                    plt.savefig(os.path.join(output_dir, 'acceleration_thermal.png'), dpi=300, bbox_inches='tight')
                    plt.close()
    
    # Extract from lap time results
    lap_thermal_data = None
    if lap_results and 'results' in lap_results:
        results = lap_results['results']
        if 'engine_temp' in results:
            lap_thermal_data = {
                'distance': results.get('distance', []),
                'time': results.get('time', []),
                'engine_temp': results.get('engine_temp', []),
                'coolant_temp': results.get('coolant_temp', []) if 'coolant_temp' in results else None,
                'oil_temp': results.get('oil_temp', []) if 'oil_temp' in results else None
            }
            
            # Calculate metrics
            if len(lap_thermal_data['engine_temp']) > 0:
                thermal_analysis['lap_time'] = {
                    'max_engine_temp': max(lap_thermal_data['engine_temp']),
                    'avg_engine_temp': sum(lap_thermal_data['engine_temp']) / len(lap_thermal_data['engine_temp']),
                    'temp_rise': max(lap_thermal_data['engine_temp']) - lap_thermal_data['engine_temp'][0],
                    'thermal_limited': lap_results.get('metrics', {}).get('thermal_limited', False)
                }
                
                if lap_thermal_data['coolant_temp'] is not None:
                    thermal_analysis['lap_time']['max_coolant_temp'] = max(lap_thermal_data['coolant_temp'])
                
                print(f"Lap Time Event Thermal Analysis:")
                print(f"  Maximum Engine Temperature: {thermal_analysis['lap_time']['max_engine_temp']:.1f}°C")
                print(f"  Average Engine Temperature: {thermal_analysis['lap_time']['avg_engine_temp']:.1f}°C")
                print(f"  Thermally Limited: {'Yes' if thermal_analysis['lap_time'].get('thermal_limited', False) else 'No'}")
                
                # Plot lap thermal data
                if len(lap_thermal_data['distance']) > 1:
                    plt.figure(figsize=(10, 6))
                    plt.plot(lap_thermal_data['distance'], lap_thermal_data['engine_temp'], 'r-', linewidth=2, label='Engine')
                    
                    if lap_thermal_data['coolant_temp'] is not None:
                        plt.plot(lap_thermal_data['distance'], lap_thermal_data['coolant_temp'], 'b-', linewidth=2, label='Coolant')
                        
                    if lap_thermal_data['oil_temp'] is not None:
                        plt.plot(lap_thermal_data['distance'], lap_thermal_data['oil_temp'], 'g-', linewidth=2, label='Oil')
                    
                    plt.xlabel('Distance (m)')
                    plt.ylabel('Temperature (°C)')
                    plt.title('Thermal Profile during Lap')
                    plt.grid(True, linestyle='--', alpha=0.7)
                    plt.legend()
                    
                    plt.savefig(os.path.join(output_dir, 'lap_thermal.png'), dpi=300, bbox_inches='tight')
                    plt.close()
    
    # Extract from endurance results
    endurance_thermal_data = None
    if endurance_results and 'results' in endurance_results:
        results = endurance_results['results']
        if 'detailed_results' in results and 'thermal_states' in results['detailed_results']:
            thermal_states = results['detailed_results']['thermal_states']
            
            if thermal_states and len(thermal_states) > 0:
                # Extract thermal data
                time_points = range(len(thermal_states))
                engine_temps = [state.get('engine_temp', 0) for state in thermal_states]
                coolant_temps = [state.get('coolant_temp', 0) for state in thermal_states]
                oil_temps = [state.get('oil_temp', 0) for state in thermal_states]
                
                endurance_thermal_data = {
                    'time_points': time_points,
                    'engine_temp': engine_temps,
                    'coolant_temp': coolant_temps,
                    'oil_temp': oil_temps
                }
                
                # Calculate metrics
                thermal_analysis['endurance'] = {
                    'max_engine_temp': max(engine_temps),
                    'avg_engine_temp': sum(engine_temps) / len(engine_temps),
                    'max_coolant_temp': max(coolant_temps),
                    'avg_coolant_temp': sum(coolant_temps) / len(coolant_temps),
                    'max_oil_temp': max(oil_temps),
                    'avg_oil_temp': sum(oil_temps) / len(oil_temps),
                    'thermal_limited': any(temp > getattr(vehicle, 'engine_warning_temp', 110.0) for temp in engine_temps)
                }
                
                print(f"Endurance Event Thermal Analysis:")
                print(f"  Maximum Engine Temperature: {thermal_analysis['endurance']['max_engine_temp']:.1f}°C")
                print(f"  Maximum Coolant Temperature: {thermal_analysis['endurance']['max_coolant_temp']:.1f}°C")
                print(f"  Maximum Oil Temperature: {thermal_analysis['endurance']['max_oil_temp']:.1f}°C")
                print(f"  Thermally Limited: {'Yes' if thermal_analysis['endurance']['thermal_limited'] else 'No'}")
                
                # Plot endurance thermal data
                plt.figure(figsize=(10, 6))
                plt.plot(time_points, engine_temps, 'r-', linewidth=2, label='Engine')
                plt.plot(time_points, coolant_temps, 'b-', linewidth=2, label='Coolant')
                plt.plot(time_points, oil_temps, 'g-', linewidth=2, label='Oil')
                
                # Add warning and critical temperature lines
                warning_temp = getattr(vehicle, 'engine_warning_temp', 110.0)
                critical_temp = getattr(vehicle, 'engine_critical_temp', 120.0)
                
                plt.axhline(y=warning_temp, color='orange', linestyle='--', alpha=0.7, label=f'Warning ({warning_temp}°C)')
                plt.axhline(y=critical_temp, color='red', linestyle='--', alpha=0.7, label=f'Critical ({critical_temp}°C)')
                
                plt.xlabel('Time Point')
                plt.ylabel('Temperature (°C)')
                plt.title('Thermal Profile during Endurance')
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.legend()
                
                plt.savefig(os.path.join(output_dir, 'endurance_thermal.png'), dpi=300, bbox_inches='tight')
                plt.close()
    
    # Calculate comparative metrics
    if 'lap_time' in thermal_analysis and 'endurance' in thermal_analysis:
        thermal_analysis['comparison'] = {
            'endurance_vs_lap_temp_increase': thermal_analysis['endurance']['max_engine_temp'] - thermal_analysis['lap_time']['max_engine_temp'],
            'thermal_margin_to_warning': getattr(vehicle, 'engine_warning_temp', 110.0) - thermal_analysis['endurance']['max_engine_temp'],
            'thermal_margin_to_critical': getattr(vehicle, 'engine_critical_temp', 120.0) - thermal_analysis['endurance']['max_engine_temp']
        }
        
        print("\nThermal Comparison:")
        print(f"  Temperature Increase in Endurance vs Single Lap: {thermal_analysis['comparison']['endurance_vs_lap_temp_increase']:.1f}°C")
        print(f"  Thermal Margin to Warning: {thermal_analysis['comparison']['thermal_margin_to_warning']:.1f}°C")
        print(f"  Thermal Margin to Critical: {thermal_analysis['comparison']['thermal_margin_to_critical']:.1f}°C")
    
    # Evaluate cooling system effectiveness
    if hasattr(vehicle, 'cooling_system') or hasattr(vehicle, 'side_pod_system') or hasattr(vehicle, 'rear_radiator'):
        if 'endurance' in thermal_analysis:
            max_temp = thermal_analysis['endurance']['max_engine_temp']
            warning_temp = getattr(vehicle, 'engine_warning_temp', 110.0)
            critical_temp = getattr(vehicle, 'engine_critical_temp', 120.0)
            
            if max_temp < warning_temp - 10:
                cooling_status = "Excellent - Significant margin to warning temperature"
                cooling_score = 5
            elif max_temp < warning_temp:
                cooling_status = "Good - Within safe operating range"
                cooling_score = 4
            elif max_temp < critical_temp:
                cooling_status = "Marginal - Operating in warning zone"
                cooling_score = 3
            else:
                cooling_status = "Inadequate - Exceeding critical temperature"
                cooling_score = 1
            
            thermal_analysis['cooling_effectiveness'] = {
                'status': cooling_status,
                'score': cooling_score,
                'margin_to_warning': warning_temp - max_temp,
                'margin_to_critical': critical_temp - max_temp
            }
            
            print(f"\nCooling System Effectiveness: {cooling_status}")
    
    # Create a thermal comparison plot
    if accel_thermal_data or lap_thermal_data or endurance_thermal_data:
        plt.figure(figsize=(12, 8))
        plt.title('Thermal Performance Comparison Across Events', fontsize=14)
        
        # Plot data for each event
        if accel_thermal_data and len(accel_thermal_data['time']) > 0:
            # Normalize time to percentage of event
            norm_time = np.array(accel_thermal_data['time']) / max(accel_thermal_data['time']) * 100
            plt.plot(norm_time, accel_thermal_data['engine_temp'], 'r-', linewidth=2, label='Acceleration - Engine')
        
        if lap_thermal_data and len(lap_thermal_data['time']) > 0:
            # Normalize time to percentage of event
            norm_time = np.array(lap_thermal_data['time']) / max(lap_thermal_data['time']) * 100
            plt.plot(norm_time, lap_thermal_data['engine_temp'], 'g-', linewidth=2, label='Lap - Engine')
        
        if endurance_thermal_data and len(endurance_thermal_data['time_points']) > 0:
            # Normalize time to percentage of event
            norm_time = np.array(endurance_thermal_data['time_points']) / max(endurance_thermal_data['time_points']) * 100
            plt.plot(norm_time, endurance_thermal_data['engine_temp'], 'b-', linewidth=2, label='Endurance - Engine')
        
        # Add warning and critical temperature lines
        warning_temp = getattr(vehicle, 'engine_warning_temp', 110.0)
        critical_temp = getattr(vehicle, 'engine_critical_temp', 120.0)
        
        plt.axhline(y=warning_temp, color='orange', linestyle='--', alpha=0.7, label=f'Warning ({warning_temp}°C)')
        plt.axhline(y=critical_temp, color='red', linestyle='--', alpha=0.7, label=f'Critical ({critical_temp}°C)')
        
        plt.xlabel('Event Progress (%)')
        plt.ylabel('Engine Temperature (°C)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        plt.savefig(os.path.join(output_dir, 'thermal_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    return thermal_analysis


# =========================================================================
# Main Function
# =========================================================================

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Formula Student Performance Simulator")
    
    # General settings
    parser.add_argument('--output-dir', type=str, help="Output directory for simulation results")
    parser.add_argument('--config-dir', type=str, help="Configuration directory")
    
    # Simulation settings
    parser.add_argument('--time-step', type=float, help="Simulation time step (seconds)")
    parser.add_argument('--endurance-laps', type=int, help="Number of laps for endurance simulation")
    
    # Event flags
    parser.add_argument('--run-acceleration', action='store_true', help="Run acceleration simulation")
    parser.add_argument('--skip-acceleration', dest='run_acceleration', action='store_false', help="Skip acceleration simulation")
    parser.add_argument('--run-lap-time', action='store_true', help="Run lap time simulation")
    parser.add_argument('--skip-lap-time', dest='run_lap_time', action='store_false', help="Skip lap time simulation")
    parser.add_argument('--run-endurance', action='store_true', help="Run endurance simulation")
    parser.add_argument('--skip-endurance', dest='run_endurance', action='store_false', help="Skip endurance simulation")
    parser.add_argument('--run-skidpad', action='store_true', help="Run skidpad simulation")
    
    # Analysis flags
    parser.add_argument('--run-weight-sensitivity', action='store_true', help="Run weight sensitivity analysis")
    parser.add_argument('--skip-weight-sensitivity', dest='run_weight_sensitivity', action='store_false', help="Skip weight sensitivity analysis")
    parser.add_argument('--run-lap-optimization', action='store_true', help="Run lap time optimization")
    parser.add_argument('--skip-lap-optimization', dest='run_lap_optimization', action='store_false', help="Skip lap time optimization")
    parser.add_argument('--run-cooling-comparison', action='store_true', help="Run cooling system comparison")
    parser.add_argument('--skip-cooling-comparison', dest='run_cooling_comparison', action='store_false', help="Skip cooling system comparison")
    
    # Track generation
    parser.add_argument('--generate-tracks', action='store_true', help="Generate test tracks")
    parser.add_argument('--num-tracks', type=int, help="Number of tracks to generate")
    
    # Set defaults for boolean flags
    parser.set_defaults(
        run_acceleration=True,
        run_lap_time=True,
        run_endurance=True,
        run_skidpad=False,
        run_weight_sensitivity=True,
        run_lap_optimization=True,
        run_cooling_comparison=True,
        generate_tracks=False
    )
    
    return parser.parse_args()


def main():
    """Main entry point for the simulation program."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Create and run simulation manager
    manager = SimulationManager(args)
    success = manager.run()
    
    # Return success status
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
        
        # Generate summary report
        report_path =