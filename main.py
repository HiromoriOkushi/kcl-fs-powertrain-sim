#!/usr/bin/env python3
"""
Full Race Simulation Example

This script demonstrates a comprehensive Formula Student vehicle simulation
through various racing events (acceleration, lap time, endurance), with detailed
thermal simulation, cooling system optimization, and advanced performance analysis.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import yaml
from datetime import datetime
from typing import Dict, Any

# Add project root to Python path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import required modules
from kcl_fs_powertrain.core.vehicle import create_formula_student_vehicle
from kcl_fs_powertrain.core.simulator import Simulator
from kcl_fs_powertrain.performance.acceleration import AccelerationSimulator, run_fs_acceleration_simulation
from kcl_fs_powertrain.performance.lap_time import LapTimeSimulator, create_example_track, run_fs_lap_simulation
from kcl_fs_powertrain.performance.endurance import EnduranceSimulator, run_endurance_simulation

# Import thermal modules
from kcl_fs_powertrain.engine.engine_thermal import ThermalConfig, CoolingSystem, EngineHeatModel, ThermalSimulation, CoolingPerformance
from kcl_fs_powertrain.thermal.cooling_system import Radiator, RadiatorType, WaterPump, CoolingFan, FanType, Thermostat, create_formula_student_cooling_system
from kcl_fs_powertrain.thermal.side_pod import SidePod, SidePodRadiator, SidePodSystem, DualSidePodSystem, create_standard_side_pod_system, create_cooling_optimized_side_pod_system
from kcl_fs_powertrain.thermal.rear_radiator import RearRadiator, RearRadiatorSystem, create_optimized_rear_radiator_system
from kcl_fs_powertrain.thermal.electric_compressor import ElectricCompressor, CoolingAssistSystem, create_high_performance_cooling_assist_system

# Import advanced performance analysis modules
from kcl_fs_powertrain.performance.weight_sensitivity import WeightSensitivityAnalyzer
from kcl_fs_powertrain.performance.lap_time_optimization import run_lap_optimization, compare_optimization_methods
from kcl_fs_powertrain.performance.optimal_lap_time import OptimalLapTimeOptimizer, run_advanced_lap_optimization
from kcl_fs_powertrain.core.track_integration import TrackProfile, calculate_optimal_racing_line

def load_configurations():
    """
    Load engine, thermal, and simulation configurations.
    
    Returns:
        dict: Dictionary containing configuration settings
    """
    config = {
        'engine_config': os.path.join('configs', 'engine', 'cbr600f4i.yaml'),
        'thermal_config': os.path.join('configs', 'thermal', 'cooling_system.yaml'),
        'side_pod_config': os.path.join('configs', 'thermal', 'side_pod.yaml'),
        'electric_compressor_config': os.path.join('configs', 'thermal', 'electric_compressor.yaml'),
        'thermal_limits_config': os.path.join('configs', 'targets', 'thermal_limits.yaml'),
        'lap_time_optimization_config': os.path.join('configs', 'lap_time', 'optimal_lap_time.yaml'),
        'acceleration_config': os.path.join('configs', 'targets', 'acceleration.yaml'),
        'transmission_config': os.path.join('configs', 'transmission', 'gearing.yaml'),
        'shift_strategy_config': os.path.join('configs', 'transmission', 'shift_strategy.yaml'),
        'output_dir': os.path.join('data', 'output', 'full_race_simulation'),
        'simulation_settings': {
            'time_step': 0.01,  # s
            'max_time': 30.0,   # s
            'acceleration_distance': 75.0,  # m
            'endurance_laps': 3,  # Number of laps for endurance simulation
        },
        'optimization_settings': {
            'enable_weight_sensitivity': True,
            'enable_lap_optimization': True,
            'optimization_iterations': 10,
            'weight_range': [180, 250],  # kg
            'param_ranges': {
                'mass': (180, 250),  # Vehicle mass range (kg)
                'drag_coefficient': (0.7, 1.2),  # Aero drag coefficient range
                'weight_distribution': (0.4, 0.6)  # Front weight distribution
            }
        }
    }
    
    # Create output directory if it doesn't exist
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Load thermal limits configuration
    if os.path.exists(config['thermal_limits_config']):
        with open(config['thermal_limits_config'], 'r') as f:
            thermal_limits = yaml.safe_load(f)
            config['thermal_limits'] = thermal_limits
    
    # Load lap time optimization configuration if available
    if os.path.exists(config['lap_time_optimization_config']):
        with open(config['lap_time_optimization_config'], 'r') as f:
            lap_time_opt = yaml.safe_load(f)
            config['lap_time_optimization'] = lap_time_opt
    
    # Load acceleration configuration if available
    if os.path.exists(config['acceleration_config']):
        with open(config['acceleration_config'], 'r') as f:
            accel_config = yaml.safe_load(f)
            config['acceleration'] = accel_config
    
    print(f"Configuration loaded. Output directory: {config['output_dir']}")
    return config

def create_vehicle(config, cooling_config='standard'):
    """
    Create and configure a Formula Student vehicle with detailed cooling systems.
    
    Args:
        config: Configuration dictionary
        cooling_config: Type of cooling configuration ('standard', 'optimized', 'minimal', or 'custom')
        
    Returns:
        Vehicle: Configured Formula Student vehicle
    """
    # Create a default Formula Student vehicle
    vehicle = create_formula_student_vehicle()
    
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
        from kcl_fs_powertrain.thermal.cooling_system import Radiator, RadiatorType
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
            if os.path.exists(config['thermal_config']):
                thermal_config.load_from_file(config['thermal_config'])
            
            # Create engine heat model with this configuration
            heat_model = EngineHeatModel(thermal_config, vehicle.engine)
            
            # Create cooling system
            cooling_system = CoolingSystem()
            
            # Create side pod system if config available
            if os.path.exists(config['side_pod_config']):
                with open(config['side_pod_config'], 'r') as f:
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
    if 'thermal_limits' in config:
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

# Configuration Manager class
class ConfigurationManager:
    """Manager class for handling simulation configurations and output directories."""
    
    def __init__(self, config_path=None, command_line_args=None):
        """
        Initializes configuration manager with default settings and paths.
        
        Args:
            config_path: Path to configuration file (optional)
            command_line_args: Command line arguments (optional)
        """
        self.default_paths = {
            'engine': os.path.join('configs', 'engine', 'cbr600f4i.yaml'),
            'thermal': os.path.join('configs', 'thermal', 'cooling_system.yaml'),
            'transmission': os.path.join('configs', 'transmission', 'gearing.yaml'),
            'shift_strategy': os.path.join('configs', 'transmission', 'shift_strategy.yaml'),
            'output': os.path.join('data', 'output'),
        }
        
        self.simulation_settings = {
            'time_step': 0.01,  # s
            'max_time': 30.0,   # s
            'acceleration_distance': 75.0,  # m
            'endurance_laps': 3,  # Number of laps for endurance simulation
        }
        
        self.optimization_settings = {
            'enable_weight_sensitivity': True,
            'enable_lap_optimization': True,
            'optimization_iterations': 10
        }
        
        # Load configuration from file if provided
        self.config = {}
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        
        # Create timestamp for unique output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(self.default_paths['output'], f'simulation_{timestamp}')
        
        # Apply command line overrides if provided
        if command_line_args:
            self._apply_command_line_overrides(command_line_args)
        
        # Create output directories
        self._ensure_output_directory()
    
    def load_configurations(self):
        """
        Loads all configuration files and creates a complete configuration dictionary.
        
        Returns:
            dict: Complete configuration dictionary
        """
        # Initialize configuration dict with default paths
        config = {
            'paths': self.default_paths,
            'simulation_settings': self.simulation_settings,
            'optimization_settings': self.optimization_settings,
            'output_dir': self.output_dir,
        }
        
        # Add output paths for different analysis types
        config['output_paths'] = {
            'acceleration': os.path.join(self.output_dir, 'acceleration'),
            'lap_time': os.path.join(self.output_dir, 'lap_time'),
            'endurance': os.path.join(self.output_dir, 'endurance'),
            'thermal': os.path.join(self.output_dir, 'thermal'),
            'weight_sensitivity': os.path.join(self.output_dir, 'weight_sensitivity'),
            'lap_optimization': os.path.join(self.output_dir, 'lap_optimization')
        }
        
        # Create output directories
        self._create_output_directories(config['output_paths'])
        
        # Load individual configuration files if they exist
        try:
            # Engine configuration
            engine_config_path = self.default_paths['engine']
            if os.path.exists(engine_config_path):
                with open(engine_config_path, 'r') as f:
                    config['engine'] = yaml.safe_load(f)
                print(f"Engine configuration loaded from: {engine_config_path}")
            
            # Thermal configuration
            thermal_config_path = self.default_paths['thermal']
            if os.path.exists(thermal_config_path):
                with open(thermal_config_path, 'r') as f:
                    config['thermal'] = yaml.safe_load(f)
                print(f"Thermal configuration loaded from: {thermal_config_path}")
            
            # Transmission configuration
            transmission_config_path = self.default_paths['transmission']
            if os.path.exists(transmission_config_path):
                with open(transmission_config_path, 'r') as f:
                    config['transmission'] = yaml.safe_load(f)
                print(f"Transmission configuration loaded from: {transmission_config_path}")
            
            # Shift strategy configuration
            shift_strategy_config_path = self.default_paths['shift_strategy']
            if os.path.exists(shift_strategy_config_path):
                with open(shift_strategy_config_path, 'r') as f:
                    config['shift_strategy'] = yaml.safe_load(f)
                print(f"Shift strategy configuration loaded from: {shift_strategy_config_path}")
            
        except Exception as e:
            print(f"Warning: Error loading configuration files: {str(e)}")
        
        # Save the loaded configuration
        self.config = config
        
        # Save configuration to output directory for reference
        self.save_configuration()
        
        return config
    
    def _ensure_output_directory(self):
        """Ensures main output directory exists."""
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Output directory created: {self.output_dir}")
    
    def _apply_command_line_overrides(self, args):
        """
        Applies command-line argument overrides to configuration.
        
        Args:
            args: Command line arguments
        """
        if hasattr(args, 'output_dir') and args.output_dir:
            self.output_dir = args.output_dir
        
        if hasattr(args, 'time_step') and args.time_step:
            self.simulation_settings['time_step'] = args.time_step
        
        if hasattr(args, 'max_time') and args.max_time:
            self.simulation_settings['max_time'] = args.max_time
        
        if hasattr(args, 'endurance_laps') and args.endurance_laps:
            self.simulation_settings['endurance_laps'] = args.endurance_laps
    
    def _create_output_directories(self, output_paths):
        """
        Creates all necessary output directories for analyses.
        
        Args:
            output_paths: Dictionary of output paths
        """
        for path in output_paths.values():
            os.makedirs(path, exist_ok=True)
    
    def get_output_path(self, analysis_type=None):
        """
        Gets output directory path for a specific analysis type.
        
        Args:
            analysis_type: Type of analysis ('acceleration', 'lap_time', etc.)
            
        Returns:
            str: Output directory path
        """
        if analysis_type and 'output_paths' in self.config and analysis_type in self.config['output_paths']:
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
            'output_paths': self.config['output_paths'],
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Save the configuration
        config_path = os.path.join(self.output_dir, 'simulation_config.yaml')
        try:
            with open(config_path, 'w') as f:
                yaml.dump(save_config, f, default_flow_style=False)
                
            print(f"Configuration saved to: {config_path}")
        except Exception as e:
            print(f"Warning: Could not save configuration to {config_path}: {str(e)}")
    
    def _ensure_directory_exists(self, file_path: str):
        """
        Ensure that the directory for a file exists before writing to it.
        
        Args:
            file_path: Path to the file that will be written
        """
        directory = os.path.dirname(file_path)
        if directory:
            os.makedirs(directory, exist_ok=True)

    def export_results(self) -> None:
        """Export the simulation results to files."""
        if not hasattr(self, 'results') or not self.results or 'results' not in self.results:
            print("No results to export")
            return

        # Export results as CSV
        results = self.results['results']

        if 'time' in results and 'speed' in results:
            self._create_dataframe_and_export(results)

    def _create_dataframe_and_export(self, results):
        """
        Create a DataFrame from results and export it to CSV.
        
        Args:
            results: Dictionary containing simulation results
        """
        df = pd.DataFrame({
            'time': results['time'],
            'speed': results['speed'],
            'acceleration': results.get('acceleration', [0] * len(results['time'])),
            'distance': results.get('distance', [0] * len(results['time'])),
            'engine_rpm': results.get('engine_rpm', [0] * len(results['time'])),
            'gear': results.get('gear', [0] * len(results['time'])),
            'wheel_slip': results.get('wheel_slip', [0] * len(results['time']))
        })

        csv_path = self._export_dataframe_to_csv("acceleration_data.csv", df)
        
        # Export metrics
        if hasattr(self, 'results') and 'metrics' in self.results:
            metrics = self.results['metrics']
            metrics_df = pd.DataFrame([metrics])
            metrics_path = self._export_dataframe_to_csv("acceleration_metrics.csv", metrics_df)
            print(f"Acceleration metrics exported to: {metrics_path}")
        
        print(f"Acceleration data exported to: {csv_path}")

    def _export_dataframe_to_csv(self, filename: str, df: pd.DataFrame) -> str:
        """
        Export a DataFrame to a CSV file.
        
        Args:
            filename: Name of the CSV file
            df: DataFrame to export
            
        Returns:
            str: Path to the exported CSV file
        """
        csv_path = os.path.join(self.output_dir, filename)

        # Ensure the directory exists
        self._ensure_directory_exists(csv_path)

        df.to_csv(csv_path, index=False)

        return csv_path

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
        if 'engine' in config and config['engine'] and 'paths' in config and 'engine' in config['paths']:
            try:
                # Check if engine_config file exists
                engine_config_path = config['paths']['engine']
                if os.path.exists(engine_config_path):
                    # Update engine parameters from config file
                    vehicle.engine.load_config(engine_config_path)
                    print(f"Engine configured from: {engine_config_path}")
            except Exception as e:
                print(f"Warning: Could not load engine configuration: {str(e)}")
                
        if hasattr(vehicle, 'engine'):
            vehicle.engine.thermal_factor = 1.0  # Initialize thermal performance factor
            vehicle.engine.engine_temperature = 60.0  # Starting temperature in °C

        # Load transmission configuration if specified
        if 'transmission' in config and config['transmission'] and 'paths' in config and 'transmission' in config['paths']:
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
        if 'shift_strategy' in config and config['shift_strategy'] and 'paths' in config and 'shift_strategy' in config['paths']:
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
        if cooling_config == 'custom':
            print("Using custom cooling configuration from config files")
            # Try to load cooling configuration from config files
            try:
                # Create thermal configuration
                thermal_config = ThermalConfig()
                if 'paths' in config and 'thermal' in config['paths'] and os.path.exists(config['paths']['thermal']):
                    thermal_config.load_from_file(config['paths']['thermal'])

                # Create engine heat model with this configuration
                heat_model = EngineHeatModel(thermal_config, vehicle.engine)

                # Create cooling system
                cooling_system = CoolingSystem()

                # Create side pod system if config available
                if 'paths' in config and 'side_pod' in config['paths'] and os.path.exists(config['paths']['side_pod']):
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

        elif cooling_config == 'minimal':
            print("Using minimal weight cooling configuration")
            # Create minimal cooling setup (prioritizing weight)
            cooling_system = create_formula_student_cooling_system()
            
            # Use a single radiator with lightweight configuration
            from kcl_fs_powertrain.thermal.cooling_system import Radiator, RadiatorType
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
        
        else:  # standard configuration
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
        
        # Set thermal limits if available in config
        if 'thermal_limits' in config:
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

def create_test_track(output_dir):
    """
    Create a simple test track for lap simulation.
    
    Args:
        output_dir: Directory to save track file
        
    Returns:
        str: Path to the created track file
    """
    # Create track file path
    track_file = os.path.join(output_dir, "test_track.yaml")
    
    # Create a medium difficulty track
    create_example_track(track_file, difficulty='medium')
    
    print(f"Test track created and saved to {track_file}")
    return track_file

def run_acceleration_test(vehicle, output_dir):
    """
    Run an acceleration test simulation and return results.
    
    Args:
        vehicle: Vehicle model to simulate
        output_dir: Directory to save results
        
    Returns:
        dict: Acceleration simulation results
    """
    print("\n--- Running Acceleration Test ---")
    
    # Create simulator
    simulator = AccelerationSimulator(vehicle)
    
    # Configure for standard FS acceleration event (75m)
    simulator.configure(distance=75.0, time_step=0.01, max_time=10.0)
    
    # Configure launch control with reasonable defaults
    simulator.configure_launch_control(
        launch_rpm=vehicle.engine.max_torque_rpm * 1.1,
        launch_slip_target=0.2,
        launch_duration=0.5
    )
    
    # Create output directory for acceleration results
    accel_output_dir = os.path.join(output_dir, "acceleration")
    os.makedirs(accel_output_dir, exist_ok=True)
    
    # Run simulation
    start_time = time.time()
    results = simulator.simulate_acceleration(use_launch_control=True, optimized_shifts=True)
    end_time = time.time()
    
    # Analyze results
    metrics = simulator.analyze_performance_metrics(results)
    
    # Plot results
    simulator.plot_acceleration_results(
        results,
        plot_wheel_slip=True,
        save_path=os.path.join(accel_output_dir, "acceleration_results.png")
    )
    
    # Print key results
    print(f"Simulation completed in {end_time - start_time:.2f}s")
    if results['finish_time'] is not None:
        print(f"75m Acceleration Time: {results['finish_time']:.3f}s")
        print(f"Final Speed: {results['finish_speed'] * 3.6:.1f} km/h")
    
    if results['time_to_60mph'] is not None:
        print(f"0-60 mph Time: {results['time_to_60mph']:.3f}s")
    
    if results['time_to_100kph'] is not None:
        print(f"0-100 km/h Time: {results['time_to_100kph']:.3f}s")
    
    return {
        'results': results,
        'metrics': metrics,
        'simulator': simulator
    }

def run_lap_simulation(vehicle, track_file, output_dir):
    """
    Run a single lap simulation and return results.
    
    Args:
        vehicle: Vehicle model to simulate
        track_file: Path to track file
        output_dir: Directory to save results
        
    Returns:
        dict: Lap simulation results
    """
    print("\n--- Running Lap Time Simulation ---")
    
    # Create output directory for lap results
    lap_output_dir = os.path.join(output_dir, "lap_time")
    os.makedirs(lap_output_dir, exist_ok=True)
    
    # Run the lap simulation
    start_time = time.time()
    results = run_fs_lap_simulation(
        vehicle,
        track_file,
        include_thermal=True,
        save_dir=lap_output_dir
    )
    end_time = time.time()
    
    # Print key results
    print(f"Simulation completed in {end_time - start_time:.2f}s")
    print(f"Lap Time: {results['lap_time']:.3f}s")
    print(f"Average Speed: {results['metrics']['avg_speed_kph']:.1f} km/h")
    print(f"Maximum Speed: {results['metrics']['max_speed_kph']:.1f} km/h")
    
    if results['metrics']['thermal_limited']:
        print("Note: Vehicle was thermally limited during the lap")
    
    return results

def run_endurance_simulation(vehicle, track_file, output_dir, laps=3):
    """
    Run an endurance simulation and return results.
    
    Args:
        vehicle: Vehicle model to simulate
        track_file: Path to track file
        output_dir: Directory to save results
        laps: Number of laps to simulate
        
    Returns:
        dict: Endurance simulation results
    """
    print(f"\n--- Running Endurance Simulation ({laps} laps) ---")
    
    # Create output directory for endurance results
    endurance_output_dir = os.path.join(output_dir, "endurance")
    os.makedirs(endurance_output_dir, exist_ok=True)
    
    # Create endurance simulator
    simulator = EnduranceSimulator(vehicle)
    
    # Load track
    simulator.lap_simulator.load_track(track_file)
    
    # Configure for specified number of laps
    simulator.configure_event(num_laps=laps, driver_change_lap=0)
    
    # Run simulation
    start_time = time.time()
    results = simulator.simulate_endurance(include_thermal=True)
    end_time = time.time()
    
    # Calculate score
    score = simulator.calculate_score(results)
    
    # Generate report
    simulator.generate_endurance_report(
        results, 
        score, 
        endurance_output_dir
    )
    
    # Print key results
    print(f"Simulation completed in {end_time - start_time:.2f}s")
    
    if results['completed']:
        print(f"Endurance completed successfully")
        print(f"Total Time: {results['total_time']:.2f}s")
        print(f"Average Lap Time: {results['average_lap']:.2f}s")
        print(f"Fuel Used: {results['total_fuel']:.2f}L")
    else:
        print(f"Endurance not completed: {results['dnf_reason']}")
        print(f"Completed {results['lap_count']} of {laps} laps")
    
    return {
        'results': results,
        'score': score,
        'simulator': simulator
    }

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
    thermal_dir = os.path.join(output_dir, "thermal_analysis")
    os.makedirs(thermal_dir, exist_ok=True)
    
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
                    
                    plt.savefig(os.path.join(thermal_dir, 'acceleration_thermal.png'), dpi=300, bbox_inches='tight')
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
                max_temp = max(lap_thermal_data['engine_temp'])
                avg_temp = sum(lap_thermal_data['engine_temp']) / len(lap_thermal_data['engine_temp'])
                
                # Check for unreasonable values (greater than 200°C or less than 0°C)
                if max_temp > 200 or max_temp < 0 or avg_temp > 200 or avg_temp < 0:
                    print("Warning: Unreasonable temperature values detected in lap simulation")
                    max_temp = 95.0  # Use reasonable default
                    avg_temp = 85.0  # Use reasonable default
                
                thermal_analysis['lap_time'] = {
                    'max_engine_temp': max_temp,
                    'avg_engine_temp': avg_temp,
                    'temp_rise': max_temp - lap_thermal_data['engine_temp'][0],
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
                    
                    plt.savefig(os.path.join(thermal_dir, 'lap_thermal.png'), dpi=300, bbox_inches='tight')
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
                
                plt.savefig(os.path.join(thermal_dir, 'endurance_thermal.png'), dpi=300, bbox_inches='tight')
                plt.close()
    
    if 'lap_time' in thermal_analysis and 'endurance' in thermal_analysis:
    # Check if the required keys exist in endurance data
        if all(key in thermal_analysis['endurance'] for key in ['max_engine_temp']):
            thermal_analysis['comparison'] = {
                'endurance_vs_lap_temp_increase': thermal_analysis['endurance']['max_engine_temp'] - thermal_analysis['lap_time']['max_engine_temp'],
                'thermal_margin_to_warning': getattr(vehicle, 'engine_warning_temp', 110.0) - thermal_analysis['endurance']['max_engine_temp'],
                'thermal_margin_to_critical': getattr(vehicle, 'engine_critical_temp', 120.0) - thermal_analysis['endurance']['max_engine_temp']
            }
        else:
            # Create a limited comparison without the missing data
            thermal_analysis['comparison'] = {
                'note': 'Limited comparison available due to incomplete endurance data'
            }
    # Calculate comparative metrics
    if 'lap_time' in thermal_analysis:
        thermal_analysis['comparison'] = {}
        
        # If we have endurance data and it has max_engine_temp
        if ('endurance' in thermal_analysis and 
            'max_engine_temp' in thermal_analysis['endurance'] and 
            'max_engine_temp' in thermal_analysis['lap_time']):
            
            thermal_analysis['comparison']['endurance_vs_lap_temp_increase'] = (
                thermal_analysis['endurance']['max_engine_temp'] - 
                thermal_analysis['lap_time']['max_engine_temp']
            )
            thermal_analysis['comparison']['thermal_margin_to_warning'] = (
                getattr(vehicle, 'engine_warning_temp', 110.0) - 
                thermal_analysis['endurance']['max_engine_temp']
            )
            thermal_analysis['comparison']['thermal_margin_to_critical'] = (
                getattr(vehicle, 'engine_critical_temp', 120.0) - 
                thermal_analysis['endurance']['max_engine_temp']
            )
        else:
            # Limited comparison based only on lap time data
            thermal_analysis['comparison']['note'] = "Limited thermal data available (endurance data missing)"
            thermal_analysis['comparison']['lap_thermal_margin_to_warning'] = (
                getattr(vehicle, 'engine_warning_temp', 110.0) - 
                thermal_analysis['lap_time'].get('max_engine_temp', 90.0)
            )
            thermal_analysis['comparison']['lap_thermal_margin_to_critical'] = (
                getattr(vehicle, 'engine_critical_temp', 120.0) - 
                thermal_analysis['lap_time'].get('max_engine_temp', 90.0)
            )
        
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
        
        plt.savefig(os.path.join(thermal_dir, 'thermal_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    return thermal_analysis

def analyze_results(accel_results, lap_results, endurance_results):
    """
    Analyze and compare results from different simulations.
    
    Args:
        accel_results: Results from acceleration simulation
        lap_results: Results from lap time simulation
        endurance_results: Results from endurance simulation
        
    Returns:
        dict: Analyzed results and comparative metrics
    """
    print("\n--- Performance Analysis ---")
    
    # Initialize analysis dictionary
    analysis = {
        'acceleration': {},
        'lap_time': {},
        'endurance': {},
        'overall': {}
    }
    
    # Analyze acceleration performance
    if accel_results and 'metrics' in accel_results:
        metrics = accel_results['metrics']
        analysis['acceleration'] = {
            '75m_time': metrics.get('finish_time', 0),
            'time_to_60mph': metrics.get('time_to_60mph', 0),
            'time_to_100kph': metrics.get('time_to_100kph', 0),
            'peak_acceleration_g': metrics.get('peak_acceleration_g', 0),
            'average_acceleration': metrics.get('average_acceleration', 0),
            'performance_grade': metrics.get('performance_grade', 'N/A')
        }
        
        # Compare to typical Formula Student benchmarks
        print("\nAcceleration Performance:")
        print(f"  75m Time: {metrics.get('finish_time', 'N/A')} (Benchmark: 3.9-4.5s)")
        
        # Handle None value for time_to_60mph
        time_to_60mph = metrics.get('time_to_60mph')
        if time_to_60mph is not None:
            print(f"  0-60 mph: {time_to_60mph:.3f}s (Benchmark: 3.6-4.2s)")
        else:
            print(f"  0-60 mph: N/A (Benchmark: 3.6-4.2s)")
            
        print(f"  Grade: {metrics.get('performance_grade', 'N/A')}")
    
    # Analyze lap time performance
    if lap_results and 'metrics' in lap_results:
        metrics = lap_results['metrics']
        analysis['lap_time'] = {
            'lap_time': lap_results.get('lap_time', 0),
            'avg_speed': metrics.get('avg_speed_kph', 0),
            'max_speed': metrics.get('max_speed_kph', 0),
            'time_in_corners': metrics.get('time_in_corners', 0),
            'time_in_straights': metrics.get('time_in_straights', 0),
            'thermal_limited': metrics.get('thermal_limited', False)
        }
        
        print("\nLap Time Performance:")
        lap_time = lap_results.get('lap_time')
        if lap_time is not None:
            print(f"  Lap Time: {lap_time:.3f}s")
        else:
            print(f"  Lap Time: N/A")
            
        avg_speed = metrics.get('avg_speed_kph')
        if avg_speed is not None:
            print(f"  Average Speed: {avg_speed:.1f} km/h")
        else:
            print(f"  Average Speed: N/A")
            
        time_in_corners = metrics.get('time_in_corners')
        if time_in_corners is not None:
            print(f"  Time in Corners: {time_in_corners:.2f}s")
        else:
            print(f"  Time in Corners: N/A")
            
        print(f"  Thermal Limited: {'Yes' if metrics.get('thermal_limited', False) else 'No'}")
    
    # Analyze endurance performance
    if endurance_results and 'results' in endurance_results:
        results = endurance_results['results']
        score = endurance_results.get('score', {})
        analysis['endurance'] = {
            'completed': results.get('completed', False),
            'total_time': results.get('total_time', 0),
            'average_lap': results.get('average_lap', 0),
            'total_fuel': results.get('total_fuel', 0),
            'fuel_efficiency': results.get('fuel_efficiency', 0),
            'endurance_score': score.get('endurance_score', 0),
            'efficiency_score': score.get('efficiency_score', 0),
            'total_score': score.get('total_score', 0)
        }
        
        print("\nEndurance Performance:")
        if results.get('completed', False):
            print(f"  Status: Completed")
            
            total_time = results.get('total_time')
            if total_time is not None:
                print(f"  Total Time: {total_time:.2f}s")
            else:
                print(f"  Total Time: N/A")
                
            avg_lap = results.get('average_lap')
            if avg_lap is not None:
                print(f"  Average Lap: {avg_lap:.2f}s")
            else:
                print(f"  Average Lap: N/A")
                
            fuel_eff = results.get('fuel_efficiency')
            if fuel_eff is not None:
                print(f"  Fuel Efficiency: {fuel_eff:.3f}L/lap")
            else:
                print(f"  Fuel Efficiency: N/A")
                
            endurance_score = score.get('endurance_score')
            max_endurance_score = score.get('max_endurance_score')
            if endurance_score is not None and max_endurance_score is not None:
                print(f"  Endurance Score: {endurance_score:.1f}/{max_endurance_score}")
            else:
                print(f"  Endurance Score: N/A")
                
            efficiency_score = score.get('efficiency_score')
            max_efficiency_score = score.get('max_efficiency_score')
            if efficiency_score is not None and max_efficiency_score is not None:
                print(f"  Efficiency Score: {efficiency_score:.1f}/{max_efficiency_score}")
            else:
                print(f"  Efficiency Score: N/A")
        else:
            print(f"  Status: DNF - {results.get('dnf_reason', 'Unknown reason')}")
            print(f"  Completed Laps: {results.get('lap_count', 'N/A')}")
    
    # Calculate overall performance metrics
    analysis['overall'] = {
        'total_score': (
            analysis['endurance'].get('total_score', 0)
        ),
        'performance_index': 0  # Placeholder for a combined performance metric
    }
    
    # Calculate a simple performance index based on normalized metrics
    # This is just an example approach for a combined score
    try:
        # Get values safely
        time_75m = analysis['acceleration'].get('75m_time', 0)
        lap_time = analysis['lap_time'].get('lap_time', 0)
        completed = analysis['endurance'].get('completed', False)
        
        # Only proceed if we have valid values
        if time_75m and lap_time:
            accel_norm = 4.0 / time_75m  # Normalized to 4.0s benchmark
            lap_norm = 60.0 / lap_time  # Normalized to 60s benchmark
            endurance_norm = 1.0 if completed else 0.5
            
            # Weight factors for each area
            analysis['overall']['performance_index'] = (
                0.3 * accel_norm + 0.3 * lap_norm + 0.4 * endurance_norm
            )
            
            print("\nOverall Performance Index:", f"{analysis['overall']['performance_index']:.3f}")
            
            # Qualitative assessment
            if analysis['overall']['performance_index'] > 0.9:
                print("Assessment: Excellent - Competitive at international level")
            elif analysis['overall']['performance_index'] > 0.8:
                print("Assessment: Good - Competitive at national level")
            elif analysis['overall']['performance_index'] > 0.7:
                print("Assessment: Average - Typical Formula Student performance")
            else:
                print("Assessment: Needs improvement for competition")
        else:
            print("\nOverall Performance: Unable to calculate index due to missing data")
                
    except ZeroDivisionError:
        analysis['overall']['performance_index'] = 0
        print("\nOverall Performance: Unable to calculate index due to missing data")
    
    return analysis

def compare_cooling_configurations(config, vehicle_base, track_file, output_dir):
    """
    Compare different cooling system configurations for thermal performance.
    
    Args:
        config: Configuration dictionary
        vehicle_base: Base vehicle model (will be copied and modified)
        track_file: Path to track file
        output_dir: Directory to save results
        
    Returns:
        dict: Comparison results
    """
    print("\n--- Cooling Configuration Comparison ---")
    
    # Create output directory for comparison
    comparison_dir = os.path.join(output_dir, "cooling_comparison")
    os.makedirs(comparison_dir, exist_ok=True)
    
    # Define configurations to test
    cooling_configs = [
        ('standard', 'Standard cooling system'),
        ('optimized', 'Optimized cooling system with side pods and rear radiator'),
        ('minimal', 'Minimum weight cooling system')
    ]
    
    # List to store results
    results = []
    
    # Test each configuration with a lap simulation
    for config_name, config_desc in cooling_configs:
        print(f"\nTesting {config_desc}...")
        
        # Create vehicle with this cooling configuration
        vehicle = create_vehicle(config, cooling_config=config_name)
        
        # Create lap simulator
        lap_simulator = LapTimeSimulator(vehicle)
        lap_simulator.load_track(track_file)
        
        # Simulate lap
        start_time = time.time()
        lap_results = lap_simulator.simulate_lap(include_thermal=True)
        
        if 'engine_temp' in lap_results.get('results', {}):
            max_temp = max(lap_results['results']['engine_temp'])
            avg_temp = np.mean(lap_results['results']['engine_temp'])
            thermal_limited = lap_results.get('thermal_limited', False)
        else:
            max_temp = "N/A"
            avg_temp = "N/A"
            thermal_limited = False

        # Store in results dict
        record['max_engine_temp'] = max_temp
        record['avg_engine_temp'] = avg_temp
        record['thermal_limited'] = thermal_limited
        
        # Calculate performance metrics
        metrics = lap_simulator.analyze_lap_performance(lap_results)
        lap_results['metrics'] = metrics  # Add metrics to the results dictionary
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
            'description': config_desc,
            'lap_time': lap_results['lap_time'],
            'max_speed': max(lap_results.get('speed', [0])) * 3.6 if 'speed' in lap_results else 0,  # Convert m/s to km/h
            'avg_speed': (lap_results.get('distance', [0])[-1] / lap_results.get('time', [1])[-1]) * 3.6 if 'distance' in lap_results and 'time' in lap_results else 0,
            'thermal_limited': lap_results.get('thermal_limited', False)
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
                os.path.join(comparison_dir, f"{config_name}_thermal_data.npz"),
                distance=thermal_data['distance'],
                engine_temp=thermal_data['engine_temp'],
                coolant_temp=thermal_data['coolant_temp'] if thermal_data['coolant_temp'] is not None else []
            )
    
    # Create comparison plot
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
        warning_temp = config.get('thermal_limits', {}).get('engine', {}).get('warning_temp', 110.0)
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
    for config_name, config_desc in cooling_configs:
        npz_path = os.path.join(comparison_dir, f"{config_name}_thermal_data.npz")
        if os.path.exists(npz_path):
            data = np.load(npz_path)
            plt.plot(data['distance'], data['engine_temp'], label=config_name)
    
    plt.xlabel('Distance (m)')
    plt.ylabel('Engine Temperature (°C)')
    plt.title('Thermal Profiles')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Save the comparison plot
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(comparison_dir, 'cooling_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a summary table
    print("\nCooling Configuration Comparison Summary:")
    print("-" * 80)
    print(f"{'Configuration':<15} | {'Lap Time':<10} | {'Max Temp':<10} | {'Thermal Limited':<15} | {'Temp Rise':<10}")
    print("-" * 80)
    
    for r in results:
        print(f"{r['configuration']:<15} | {r['lap_time']:<10.3f} | {r.get('max_engine_temp', 'N/A'):<10} | {str(r['thermal_limited']):<15} | {r.get('temp_rise', 'N/A'):<10.1}")
    
    print("-" * 80)
    
    # Determine best configuration based on a weighted score
    # (30% lap time, 70% thermal performance)
    if all('max_engine_temp' in r for r in results):
        for r in results:
            lap_time_score = min(lap_times) / r['lap_time']
            thermal_score = 1.0 - (r['max_engine_temp'] / max(r['max_engine_temp'] for r in results))
            r['composite_score'] = 0.3 * lap_time_score + 0.7 * thermal_score
        
        # Find the best configuration
        best_config = max(results, key=lambda r: r['composite_score'])
        print(f"\nBest overall configuration: {best_config['configuration']} (score: {best_config['composite_score']:.3f})")
    
    # Save comparison results
    pd.DataFrame(results).to_csv(os.path.join(comparison_dir, 'cooling_comparison.csv'), index=False)
    
    return {
        'configurations': cooling_configs,
        'results': results,
        'output_dir': comparison_dir
    }

def plot_results(accel_results, lap_results, endurance_results, output_dir):
    """
    Create visualizations of simulation results.
    
    Args:
        accel_results: Results from acceleration simulation
        lap_results: Results from lap time simulation
        endurance_results: Results from endurance simulation
        output_dir: Directory to save plots
        
    Returns:
        bool: True if plotting was successful
    """
    print("\n--- Generating Result Visualizations ---")
    
    # Create a summary plot of key performance metrics
    try:
        plt.figure(figsize=(12, 8))
        plt.suptitle('Formula Student Vehicle Performance Summary', fontsize=16)
        
        # Create subplots
        subplot_count = 0
        
        # Acceleration subplot (if data available)
        if accel_results and 'results' in accel_results:
            subplot_count += 1
            accel_data = accel_results['results']
            
            if 'time' in accel_data and 'speed' in accel_data:
                plt.subplot(2, 2, 1)
                plt.plot(accel_data['time'], accel_data['speed'] * 3.6, 'b-', linewidth=2)
                plt.xlabel('Time (s)')
                plt.ylabel('Speed (km/h)')
                plt.title('Acceleration Profile')
                plt.grid(True, linestyle='--', alpha=0.7)
                
                # Add markers for key milestones
                if 'time_to_60mph' in accel_data and accel_data['time_to_60mph'] is not None:
                    plt.axvline(x=accel_data['time_to_60mph'], color='r', linestyle='--', alpha=0.7)
                    plt.text(accel_data['time_to_60mph'] + 0.1, 60, '60 mph', color='r')
                
                if 'time_to_100kph' in accel_data and accel_data['time_to_100kph'] is not None:
                    plt.axvline(x=accel_data['time_to_100kph'], color='g', linestyle='--', alpha=0.7)
                    plt.text(accel_data['time_to_100kph'] + 0.1, 80, '100 km/h', color='g')
        
        # Lap time subplot (if data available)
        if lap_results and 'results' in lap_results:
            subplot_count += 1
            lap_data = lap_results['results']
            
            if 'distance' in lap_data and 'speed' in lap_data:
                plt.subplot(2, 2, 2)
                plt.plot(lap_data['distance'], lap_data['speed'] * 3.6, 'g-', linewidth=2)
                plt.xlabel('Distance (m)')
                plt.ylabel('Speed (km/h)')
                plt.title('Lap Speed Profile')
                plt.grid(True, linestyle='--', alpha=0.7)
        
        # Endurance subplot (if data available)
        if endurance_results and 'results' in endurance_results:
            subplot_count += 1
            endurance_data = endurance_results['results']
            
            if 'lap_times' in endurance_data and endurance_data['lap_times']:
                plt.subplot(2, 2, 3)
                plt.bar(range(1, len(endurance_data['lap_times']) + 1), endurance_data['lap_times'], color='blue', alpha=0.7)
                plt.axhline(y=endurance_data['average_lap'], color='r', linestyle='--', alpha=0.7, label=f"Avg: {endurance_data['average_lap']:.2f}s")
                plt.xlabel('Lap Number')
                plt.ylabel('Lap Time (s)')
                plt.title('Endurance Lap Times')
                plt.grid(True, axis='y', linestyle='--', alpha=0.7)
                plt.legend()
        
        # Thermal subplot (use data from endurance or lap simulation)
        thermal_data = None
        if lap_results and 'results' in lap_results and 'engine_temp' in lap_results['results']:
            thermal_data = lap_results['results']
            title = 'Thermal Profile during Lap'
        elif endurance_results and 'detailed_results' in endurance_results['results']:
            detailed = endurance_results['results']['detailed_results']
            if 'thermal_states' in detailed and detailed['thermal_states']:
                # Convert endurance thermal states to a plot-friendly format
                thermal_data = {
                    'distance': np.arange(len(detailed['thermal_states'])) * 100,  # Approximate distance
                    'engine_temp': np.array([state['engine_temp'] for state in detailed['thermal_states']]),
                    'coolant_temp': np.array([state['coolant_temp'] for state in detailed['thermal_states']]),
                    'oil_temp': np.array([state['oil_temp'] for state in detailed['thermal_states']])
                }
                title = 'Thermal Profile during Endurance'
        
        if thermal_data is not None:
            subplot_count += 1
            plt.subplot(2, 2, 4)
            
            if 'engine_temp' in thermal_data:
                plt.plot(thermal_data['distance'], thermal_data['engine_temp'], 'r-', linewidth=2, label='Engine')
            
            if 'coolant_temp' in thermal_data:
                plt.plot(thermal_data['distance'], thermal_data['coolant_temp'], 'b-', linewidth=2, label='Coolant')
            
            if 'oil_temp' in thermal_data:
                plt.plot(thermal_data['distance'], thermal_data['oil_temp'], 'g-', linewidth=2, label='Oil')
            
            plt.xlabel('Distance (m)')
            plt.ylabel('Temperature (°C)')
            plt.title(title)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
        
        # Adjust layout and save
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle
        summary_plot_path = os.path.join(output_dir, "performance_summary.png")
        plt.savefig(summary_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Performance summary plot saved to: {summary_plot_path}")
        return True
        
    except Exception as e:
        print(f"Error generating plots: {str(e)}")
        return False

def export_results(accel_results, lap_results, endurance_results, output_dir):
    """
    Export simulation results to files.
    
    Args:
        accel_results: Results from acceleration simulation
        lap_results: Results from lap time simulation
        endurance_results: Results from endurance simulation
        output_dir: Directory to save results
        
    Returns:
        bool: True if export was successful
    """
    print("\n--- Exporting Results ---")
    
    # Define export paths
    summary_csv_path = os.path.join(output_dir, "summary_results.csv")
    summary_json_path = os.path.join(output_dir, "summary_results.json")
    
    try:
        # Prepare summary data
        summary = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'acceleration': {},
            'lap_time': {},
            'endurance': {}
        }
        
        # Add acceleration results
        if accel_results and 'metrics' in accel_results:
            metrics = accel_results['metrics']
            summary['acceleration'] = {
                '75m_time': metrics.get('finish_time', None),
                'time_to_60mph': metrics.get('time_to_60mph', None),
                'time_to_100kph': metrics.get('time_to_100kph', None),
                'peak_acceleration_g': metrics.get('peak_acceleration_g', None),
                'finish_speed_kph': metrics.get('finish_speed', 0) * 3.6 if metrics.get('finish_speed') else None
            }
        
        # Add lap time results
        if lap_results:
            summary['lap_time'] = {
                'lap_time': lap_results.get('lap_time', None),
                'avg_speed_kph': lap_results.get('metrics', {}).get('avg_speed_kph', None),
                'max_speed_kph': lap_results.get('metrics', {}).get('max_speed_kph', None),
                'thermal_limited': lap_results.get('metrics', {}).get('thermal_limited', False)
            }
        
        # Add endurance results
        if endurance_results and 'results' in endurance_results:
            results = endurance_results['results']
            score = endurance_results['score']
            
            summary['endurance'] = {
                'completed': results.get('completed', False),
                'total_time': results.get('total_time', None),
                'average_lap': results.get('average_lap', None),
                'laps_completed': results.get('lap_count', 0),
                'total_fuel': results.get('total_fuel', None),
                'endurance_score': score.get('endurance_score', None),
                'efficiency_score': score.get('efficiency_score', None),
                'total_score': score.get('total_score', None)
            }
        
        # Export as CSV
        df = pd.DataFrame({k: [v] for k, v in summary.items()})
        df.to_csv(summary_csv_path, index=False)
        
        # Export as JSON
        import json
        with open(summary_json_path, 'w') as f:
            json.dump(summary, f, indent=4)
        
        print(f"Summary results exported to:")
        print(f"  CSV: {summary_csv_path}")
        print(f"  JSON: {summary_json_path}")
        
        return True
        
    except Exception as e:
        print(f"Error exporting results: {str(e)}")
        return False

def analyze_performance_tradeoffs(weight_results, lap_results, cooling_results):
    """
    Analyze tradeoffs between weight, lap time, and thermal performance.
    
    Args:
        weight_results: Results from weight sensitivity analysis
        lap_results: Results from lap time optimization
        cooling_results: Results from cooling configuration comparison
        
    Returns:
        dict: Performance tradeoff analysis
    """
    print("\n--- Performance Tradeoff Analysis ---")
    
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
    
    return tradeoffs

def optimize_lap_time(vehicle, track_file, output_dir, method='advanced', config_file=None):
    """
    Optimize lap time using advanced numerical methods.
    
    Args:
        vehicle: Vehicle model to use for optimization
        track_file: Path to track file
        output_dir: Directory to save results
        method: Optimization method ('basic' or 'advanced')
        config_file: Optional path to configuration file
        
    Returns:
        dict: Lap time optimization results
    """
    print(f"\n--- Lap Time Optimization ({method}) ---")
    
    # Create output directory for lap time optimization
    lap_opt_dir = os.path.join(output_dir, "lap_optimization")
    os.makedirs(lap_opt_dir, exist_ok=True)
    
    # Run lap time optimization
    try:
        if method == 'compare':
            # Compare basic and advanced optimization methods
            results = compare_optimization_methods(
                vehicle,
                track_file,
                config_file=config_file,
                include_thermal=True,
                save_dir=lap_opt_dir
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
                vehicle,
                track_file,
                method=method,
                config_file=config_file,
                include_thermal=True,
                save_dir=lap_opt_dir
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
    
    return {
        'results': results,
        'method': method,
        'output_dir': lap_opt_dir
    }

def calculate_optimal_racing_lines(vehicle, track_file, output_dir):
    """
    Calculate and visualize optimal racing lines for a track.
    
    Args:
        vehicle: Vehicle model to use for calculation
        track_file: Path to track file
        output_dir: Directory to save results
        
    Returns:
        dict: Racing line calculation results
    """
    print("\n--- Optimal Racing Line Calculation ---")
    
    # Create output directory for racing line results
    racing_line_dir = os.path.join(output_dir, "racing_lines")
    os.makedirs(racing_line_dir, exist_ok=True)
    
    try:
        # Load track profile
        track_profile = TrackProfile(track_file)
        
        # Calculate basic racing line using track_integration module
        basic_racing_line = calculate_optimal_racing_line(track_profile)
        
        # Create optimal lap time optimizer for advanced racing line
        optimizer = OptimalLapTimeOptimizer(vehicle, track_profile)
        
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
            plt.savefig(os.path.join(racing_line_dir, 'racing_line_comparison.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        # Calculate lap time using the optimized racing line
        lap_time = advanced_results.get('lap_time')
        
        print(f"Racing line calculation completed")
        if lap_time:
            print(f"  Optimized Lap Time: {lap_time:.3f}s")
        
        # Return the results
        return {
            'basic_racing_line': basic_racing_line,
            'advanced_racing_line': advanced_racing_line,
            'advanced_results': advanced_results,
            'lap_time': lap_time,
            'output_dir': racing_line_dir
        }
        
    except Exception as e:
        print(f"Error in racing line calculation: {str(e)}")
        return {'error': str(e)}

def analyze_weight_sensitivity(vehicle, track_file, output_dir, weight_range=None, num_points=5):
    """
    Analyze the sensitivity of vehicle performance to weight changes.
    
    Args:
        vehicle: Vehicle model to use for analysis
        track_file: Path to track file
        output_dir: Directory to save results
        weight_range: Tuple of (min_weight, max_weight) in kg, or None to use defaults
        num_points: Number of weight points to analyze
        
    Returns:
        dict: Weight sensitivity analysis results
    """
    print("\n--- Weight Sensitivity Analysis ---")
    
    # Create output directory for weight sensitivity analysis
    weight_dir = os.path.join(output_dir, "weight_sensitivity")
    os.makedirs(weight_dir, exist_ok=True)
    
    # Create a weight sensitivity analyzer
    analyzer = WeightSensitivityAnalyzer(vehicle)
    
    # Use default weight range if not specified
    if weight_range is None:
        # Default range from -15% to +15% of current weight
        base_weight = vehicle.mass
        min_weight = int(base_weight * 0.85)
        max_weight = int(base_weight * 1.15)
        weight_range = (min_weight, max_weight)
    
    print(f"Analyzing weight sensitivity from {weight_range[0]} kg to {weight_range[1]} kg")
    
    # Analyze acceleration sensitivity
    accel_sensitivity = analyzer.analyze_acceleration_sensitivity(
        weight_range=weight_range,
        num_points=num_points,
        use_launch_control=True,
        use_optimized_shifts=True
    )
    
    # Analyze lap time sensitivity
    lap_sensitivity = analyzer.analyze_lap_time_sensitivity(
        track_file=track_file,
        weight_range=weight_range,
        num_points=num_points,
        include_thermal=True
    )
    
    # Skip weight distribution sensitivity as the method doesn't exist
    # This method doesn't exist in the WeightSensitivityAnalyzer class
    distribution_sensitivity = {}
    print("Weight distribution sensitivity analysis not available in this version")
    
    # Plot sensitivity curves
    analyzer.plot_weight_sensitivity_curves(save_path=os.path.join(weight_dir, "weight_sensitivity_curves.png"))
    
    # Generate report
    report = analyzer.generate_weight_sensitivity_report(save_dir=weight_dir)
    
    # Calculate weight reduction targets for specific performance goals
    # Example: Target a 4.0s acceleration time (or closest to current minus 10%)
    current_accel_time = accel_sensitivity['time_75m'][0]
    accel_target = min(4.0, current_accel_time * 0.9)  # 10% improvement or 4.0s, whichever is better
    
    # Example: Target a lap time improvement of 7%
    current_lap_time = lap_sensitivity['lap_times'][0]
    lap_target = current_lap_time * 0.93  # 7% improvement
    
    # Calculate required reductions
    accel_reduction = analyzer.calculate_weight_reduction_targets(
        accel_target, 
        sensitivity=accel_sensitivity['sensitivity_75m'],
        performance_type='acceleration'
    )
    
    lap_reduction = analyzer.calculate_weight_reduction_targets(
        lap_target,
        sensitivity=lap_sensitivity['sensitivity_lap_time'],
        performance_type='lap_time'
    )
    
    # Print key findings
    print("\nWeight Sensitivity Key Findings:")
    print(f"  Acceleration Sensitivity: {accel_sensitivity['seconds_per_10kg_75m']:.3f} seconds per 10kg")
    print(f"  Lap Time Sensitivity: {lap_sensitivity['seconds_per_10kg_lap']:.3f} seconds per 10kg")
    
    print("\nWeight Reduction Targets:")
    if accel_reduction:
        print(f"  To achieve {accel_target:.2f}s acceleration:")
        print(f"    Required weight reduction: {accel_reduction['required_weight_reduction']:.1f} kg")
        print(f"    Target weight: {accel_reduction['target_weight']:.1f} kg")
        print(f"    Achievable: {'Yes' if accel_reduction['is_achievable'] else 'No'}")
    
    if lap_reduction:
        print(f"  To achieve {lap_target:.2f}s lap time:")
        print(f"    Required weight reduction: {lap_reduction['required_weight_reduction']:.1f} kg")
        print(f"    Target weight: {lap_reduction['target_weight']:.1f} kg")
        print(f"    Achievable: {'Yes' if lap_reduction['is_achievable'] else 'No'}")
    
    # Return combined results
    results = {
        'accel_sensitivity': accel_sensitivity,
        'lap_sensitivity': lap_sensitivity,
        'distribution_sensitivity': distribution_sensitivity,
        'accel_reduction_target': accel_reduction,
        'lap_reduction_target': lap_reduction,
        'report': report,
        'output_dir': weight_dir
    }
    
    return results

def main():
    """Main function to run the complete simulation example."""
    print("KCL Formula Student - Full Race Simulation Example")
    print("=================================================")
    
    # Load configurations
    config = load_configurations()
    output_dir = config['output_dir']
    
    # Create vehicle with standard cooling configuration
    print("\n=== Creating Vehicle with Standard Cooling Configuration ===")
    vehicle = create_vehicle(config, cooling_config='standard')
    
    # Create test track
    track_file = create_test_track(output_dir)
    
    # Run acceleration test
    accel_results = run_acceleration_test(vehicle, output_dir)
    
    # Run lap simulation
    lap_results = run_lap_simulation(vehicle, track_file, output_dir)
    
    # Run endurance simulation
    endurance_results = run_endurance_simulation(
        vehicle, 
        track_file, 
        output_dir,
        laps=config['simulation_settings']['endurance_laps']
    )
    
    # Perform thermal analysis
    thermal_analysis = analyze_thermal_performance(
        vehicle,
        accel_results,
        lap_results,
        endurance_results,
        output_dir
    )
    
    # Compare different cooling configurations
    print("\n=== Comparing Different Cooling Configurations ===")
    cooling_comparison = compare_cooling_configurations(
        config,
        vehicle,
        track_file,
        output_dir
    )
    
    # Run weight sensitivity analysis (if enabled)
    weight_sensitivity_results = None
    if config.get('optimization_settings', {}).get('enable_weight_sensitivity', False):
        print("\n=== Performing Weight Sensitivity Analysis ===")
        weight_range = config.get('optimization_settings', {}).get('weight_range', [180, 250])
        weight_sensitivity_results = analyze_weight_sensitivity(
            vehicle,
            track_file,
            output_dir,
            weight_range=tuple(weight_range),
            num_points=5
        )
    
    # Perform lap time optimization (if enabled)
    lap_optimization_results = None
    if config.get('optimization_settings', {}).get('enable_lap_optimization', False):
        print("\n=== Performing Lap Time Optimization ===")
        lap_optimization_results = optimize_lap_time(
            vehicle,
            track_file,
            output_dir,
            method='advanced',
            config_file=config.get('lap_time_optimization_config')
        )
    
    # Calculate optimal racing lines
    racing_line_results = calculate_optimal_racing_lines(
        vehicle,
        track_file,
        output_dir
    )
    
    # Analyze performance tradeoffs
    tradeoff_analysis = analyze_performance_tradeoffs(
        weight_sensitivity_results,
        lap_optimization_results,
        cooling_comparison
    )
    
    # Analyze results
    analysis = analyze_results(accel_results, lap_results, endurance_results)
    
    # Plot summary results
    plot_results(accel_results, lap_results, endurance_results, output_dir)
    
    # Export results
    export_results(accel_results, lap_results, endurance_results, output_dir)
    
    # Print overall conclusion
    print("\n=== Overall Conclusion ===")
    
    # Add thermal performance conclusion
    if 'cooling_effectiveness' in thermal_analysis and 'status' in thermal_analysis['cooling_effectiveness']:
        print(f"Thermal Performance: {thermal_analysis['cooling_effectiveness']['status']}")
    
    # Add cooling comparison conclusion if available
    if cooling_comparison and 'results' in cooling_comparison:
        results = cooling_comparison['results']
        if results and all('composite_score' in r for r in results):
            best_config = max(results, key=lambda r: r['composite_score'])
            print(f"Recommended Cooling Configuration: {best_config['description']}")
    
    # Add weight sensitivity conclusion if available
    if weight_sensitivity_results and 'lap_reduction_target' in weight_sensitivity_results:
        target = weight_sensitivity_results['lap_reduction_target']
        if target:
            print(f"Optimal Weight Target: {target.get('target_weight', 0):.1f} kg "
                 f"({target.get('required_weight_reduction', 0):.1f} kg reduction)")
    
    # Add lap optimization conclusion if available
    if lap_optimization_results and 'results' in lap_optimization_results:
        results = lap_optimization_results['results']
        if isinstance(results, dict) and 'lap_time' in results:
            print(f"Optimized Lap Time: {results['lap_time']:.3f}s")
    
    # Add performance tradeoff recommendations
    if tradeoff_analysis and 'recommendations' in tradeoff_analysis:
        print("\nKey Recommendations:")
        for i, rec in enumerate(tradeoff_analysis['recommendations'][:2], 1):  # Show top 2 recommendations
            print(f"  {i}. {rec}")
    
    print("\nSimulation completed successfully!")
    print(f"Results saved to: {output_dir}")

if __name__ == "__main__":
    main()