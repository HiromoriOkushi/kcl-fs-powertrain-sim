#!/usr/bin/env python3
"""
Main entry point for the KCL Formula Student Powertrain Simulation.

Orchestrates the setup, execution, and reporting of various vehicle
performance simulations and analyses based on configuration files and
command-line arguments.
"""

import os
import sys
import numpy as np
import pandas as pd
import time
import yaml
import logging
import argparse
import copy
import json
from datetime import datetime
from typing import Dict, Any, Optional, List
from enum import Enum 

# --- Add project root to Python path ---
# This allows running main.py from the project root directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
#print(f"Project Root added to path: {project_root}")
#print(f"Python Path: {sys.path}")


# --- Import Simulation Components ---
# Use absolute imports from the package root
try:
    from kcl_fs_powertrain.core.vehicle import Vehicle, create_formula_student_vehicle
    from kcl_fs_powertrain.core.track import Track
    from kcl_fs_powertrain.core.track_integration import TrackProfile
    from kcl_fs_powertrain.core.simulator import Simulator, EnvironmentConditions

    # Event Simulators
    from kcl_fs_powertrain.performance.acceleration import AccelerationSimulator, run_fs_acceleration_simulation
    # Try importing CorneringPerformance carefully
    try:
        from kcl_fs_powertrain.performance.lap_time import LapTimeSimulator, run_fs_lap_simulation, CorneringPerformance
        CorneringPerformance_available = True
    except ImportError:
        from kcl_fs_powertrain.performance.lap_time import LapTimeSimulator, run_fs_lap_simulation
        CorneringPerformance = None # Define as None if unavailable
        CorneringPerformance_available = False
        logging.warning("CorneringPerformance could not be imported from lap_time module.")

    from kcl_fs_powertrain.performance.endurance import EnduranceSimulator, run_endurance_simulation

    # Analysis Tools
    from kcl_fs_powertrain.performance.weight_sensitivity import WeightSensitivityAnalyzer, analyze_weight_sensitivity
    from kcl_fs_powertrain.performance.lap_time_optimization import run_lap_optimization, compare_optimization_methods

    # Track Generation
    from kcl_fs_powertrain.track_generator.generator import FSTrackGenerator
    from kcl_fs_powertrain.track_generator.utils import generate_multiple_tracks
    from kcl_fs_powertrain.track_generator.enums import TrackMode, SimType

    # Plotting & Utils
    # V-- Import the missing function here
    from kcl_fs_powertrain.utils.plotting import set_plot_style, save_plot
    from kcl_fs_powertrain.utils.validation import validate_full_vehicle_performance
except ImportError as e:
    print(f"ERROR: Failed to import necessary simulation modules: {e}")
    print("Please ensure the package is installed correctly (`pip install -e .`) or run from the project root directory.")
    sys.exit(1)


# --- Configure Logging ---
# Moved configuration to main block for flexibility
# logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
log = logging.getLogger("MainSimulation")


# =========================================================================
# Configuration Management
# =========================================================================
class ConfigurationManager:
    """Handles loading, merging, and saving of simulation configurations."""
    def __init__(self, default_config_dir: str = "configs"):
        self.default_config_dir = default_config_dir
        self.config: Dict[str, Any] = {}
        self.output_dir: Optional[str] = None
        self.timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")

    def load_configuration(self, main_config_path: Optional[str] = None, cli_args: Optional[argparse.Namespace] = None) -> Dict:
        """Loads configurations from defaults, main file, and CLI overrides."""
        log.info("Loading configuration...")
        # 1. Start with some basic defaults
        self.config = {
            'simulation_settings': {'time_step': 0.01, 'include_thermal': True},
            'analysis_settings': {'enable_weight_sensitivity': False, 'enable_lap_optimization': False},
            'event_settings': {'acceleration': True, 'lap_time': True, 'endurance': True},
            'output_settings': {'base_dir': 'data/output', 'save_plots': True, 'save_results': True}
        }

        # 2. Load main configuration file if provided
        if main_config_path and os.path.exists(main_config_path):
            try:
                with open(main_config_path, 'r') as f:
                    main_cfg = yaml.safe_load(f)
                    if main_cfg: # Ensure file is not empty
                        # Deep merge main config over defaults
                        self._deep_merge(self.config, main_cfg)
                log.info(f"Loaded main configuration from: {main_config_path}")
            except Exception as e:
                log.error(f"Error loading main config '{main_config_path}': {e}")
        elif main_config_path:
            log.warning(f"Main config file specified but not found: {main_config_path}")

        # 3. Load component configurations (paths defined in main config or defaults)
        # Example: Load engine config path from main config or default path
        self._load_component_config('engine', 'engine/cbr600f4i.yaml')
        self._load_component_config('transmission', 'transmission/gearing.yaml')
        self._load_component_config('shift_strategy', 'transmission/shift_strategy.yaml')
        self._load_component_config('cooling_system', 'thermal/cooling_system.yaml')
        self._load_component_config('thermal_limits', 'targets/thermal_limits.yaml')
        # Add more component configs as needed...

        # 4. Apply CLI argument overrides
        if cli_args:
            self._apply_cli_overrides(cli_args)

        # 5. Setup Output Directory
        self._setup_output_directory(cli_args)

        # 6. Save Final Configuration
        self.save_configuration("initial_config.yaml") # Save the config used for this run

        log.info(f"Configuration loaded. Output Directory: {self.output_dir}")
        return self.config

    def _deep_merge(self, target: Dict, source: Dict):
        """Recursively merge source dict into target dict."""
        for key, value in source.items():
            if isinstance(value, dict) and key in target and isinstance(target[key], dict):
                self._deep_merge(target[key], value)
            else:
                target[key] = value

    def _load_component_config(self, component_key: str, default_rel_path: str):
        """Loads a specific component config file."""
        # Check if path is defined in the already loaded main config
        # Allow inline config as well
        config_value = self.config.get(f'{component_key}_config_path', self.config.get(component_key))
        config_path = None

        if isinstance(config_value, str): # It's a path
            config_path = os.path.join(self.default_config_dir, config_value) if not os.path.isabs(config_value) else config_value
            if not os.path.exists(config_path):
                # Try default path if specific path doesn't exist
                default_path_full = os.path.join(self.default_config_dir, default_rel_path)
                if os.path.exists(default_path_full):
                    config_path = default_path_full
                    log.debug(f"Specified path '{config_value}' not found, using default: {default_path_full}")
                else:
                     log.warning(f"{component_key} config file not found: {config_path} or default {default_path_full}")
                     config_path = None # Path invalid
        elif isinstance(config_value, dict): # Inline config already exists
            log.debug(f"Using inline config for {component_key}")
            self.config[component_key] = config_value # Ensure it's stored under the key
            return # Already loaded
        else: # Not specified, use default path
             config_path = os.path.join(self.default_config_dir, default_rel_path)
             if not os.path.exists(config_path):
                 log.warning(f"{component_key} default config file not found: {config_path}")
                 config_path = None # Default path also invalid

        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    component_cfg = yaml.safe_load(f)
                    if component_cfg: # Check if file has content
                        # Store the loaded component config under its key
                        # Handle potential nested structure (e.g., 'engine' key within engine file)
                        if component_key in component_cfg:
                             self.config[component_key] = component_cfg[component_key]
                        else:
                             self.config[component_key] = component_cfg # Assume file contains the dict directly
                        log.debug(f"Loaded {component_key} config from: {config_path}")
                    else:
                         log.warning(f"Config file {config_path} is empty.")
            except Exception as e:
                log.error(f"Error loading {component_key} config '{config_path}': {e}")
        elif config_path: # Log only if a path was determined but not found
             pass # Already logged warning above

    def _apply_cli_overrides(self, args: argparse.Namespace):
        """Override configuration values with CLI arguments."""
        log.debug("Applying CLI overrides...")
        if args.time_step: self.config.setdefault('simulation_settings', {})['time_step'] = args.time_step
        if args.no_thermal: self.config.setdefault('simulation_settings', {})['include_thermal'] = False
        if args.run_accel is not None: self.config.setdefault('event_settings', {})['acceleration'] = args.run_accel
        if args.run_lap is not None: self.config.setdefault('event_settings', {})['lap_time'] = args.run_lap
        if args.run_endurance is not None: self.config.setdefault('event_settings', {})['endurance'] = args.run_endurance
        if args.run_sensitivity: self.config.setdefault('analysis_settings', {})['enable_weight_sensitivity'] = True
        if args.run_optimization: self.config.setdefault('analysis_settings', {})['enable_lap_optimization'] = True
        if args.no_plots: self.config.setdefault('output_settings', {})['save_plots'] = False
        # Add more overrides as needed

    def _setup_output_directory(self, args: Optional[argparse.Namespace] = None):
        """Set up the main output directory for this simulation run."""
        base_dir = self.config.get('output_settings', {}).get('base_dir', 'data/output')
        # Allow CLI to override base directory
        if args and args.output_dir:
             self.output_dir = args.output_dir # User specified exact directory
             # May or may not include timestamp based on user intent
             log.info(f"Using specified output directory: {self.output_dir}")
        else:
             # Create a timestamped subdirectory within the base directory
             run_name = f"sim_{self.timestamp}"
             self.output_dir = os.path.join(base_dir, run_name)
             log.info(f"Creating timestamped output directory: {self.output_dir}")

        # Ensure the directory exists
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            self.config['output_dir'] = self.output_dir # Store final output dir
            # Create subdirectories for different outputs
            for sub in ['acceleration', 'lap_time', 'endurance', 'thermal', 'analysis', 'plots', 'configs']:
                 os.makedirs(os.path.join(self.output_dir, sub), exist_ok=True)
        except OSError as e:
            log.error(f"Failed to create output directory {self.output_dir}: {e}")
            # Fallback to a default directory? For now, just log error.
            self.output_dir = "." # Fallback to current directory


    def get_output_path(self, *args) -> str:
        """Construct a path within the main output directory."""
        if not self.output_dir: self._setup_output_directory() # Ensure output dir exists
        return os.path.join(self.output_dir, *args)

    def save_configuration(self, filename: str):
        """Save the currently active configuration to the output directory."""
        if not self.output_dir: return
        save_path = self.get_output_path('configs', filename)
        try:
            # Clean config for saving (remove complex objects if any were added)
            config_to_save = copy.deepcopy(self.config)
            # Remove CLI args as they are not part of the config file structure typically
            config_to_save.pop('cli_args', None)
            # Add logic here to remove non-serializable objects if needed

            with open(save_path, 'w') as f:
                yaml.dump(config_to_save, f, default_flow_style=False, sort_keys=False)
            log.info(f"Current configuration saved to: {save_path}")
        except Exception as e:
            log.error(f"Failed to save configuration to {save_path}: {e}")


# =========================================================================
# Vehicle Factory
# =========================================================================
class VehicleFactory:
    """Creates Vehicle instances based on configuration."""
    @staticmethod
    def create_vehicle(config: Dict) -> Vehicle:
        """Instantiate the Vehicle with components defined in config."""
        log.info("Creating vehicle instance...")
        # Use the factory function from core.vehicle
        # Pass the relevant sections of the config to the Vehicle constructor or its initializer methods
        # The Vehicle constructor now handles parsing its own config dictionary
        try:
            vehicle_instance = Vehicle(config=config)
            log.info("Vehicle instance created successfully.")
            return vehicle_instance
        except Exception as e:
            log.error(f"Failed to create vehicle from configuration: {e}", exc_info=True)
            log.warning("Attempting to create default FS vehicle as fallback.")
            # Try creating default vehicle using its factory (might need default configs)
            try:
                return create_formula_student_vehicle() # Fallback
            except Exception as fallback_e:
                log.critical(f"Failed to create fallback vehicle: {fallback_e}", exc_info=True)
                raise RuntimeError("Vehicle creation failed completely.") from fallback_e


# =========================================================================
# Track Management
# =========================================================================
class TrackManager:
    """Handles loading or generating tracks."""
    def __init__(self, config: Dict):
        self.config = config
        self.track_config_dir = config.get('track_config_dir', 'configs/tracks')
        self.track_output_dir = config.get('output_settings', {}).get('generated_track_dir',
                                          config.get('output_dir', 'data/output') + '/tracks') # Append /tracks


    def get_track(self, track_arg: Optional[str] = None, generate_if_missing: bool = True) -> Optional[Track]:
        """Loads a specific track file or generates one if requested/missing."""
        track_file = track_arg or self.config.get('track_file')
        source_dir = self.config.get('input_settings', {}).get('track_dir', 'data/input/tracks')

        if track_file:
             # Try finding the track file relative to source_dir or as absolute path
             potential_path1 = os.path.join(source_dir, track_file)
             potential_path2 = os.path.abspath(track_file)

             if os.path.exists(potential_path1):
                 track_file_path = potential_path1
             elif os.path.exists(potential_path2):
                  track_file_path = potential_path2
             else:
                 track_file_path = None
                 log.warning(f"Specified track file not found: {track_file} (looked in {source_dir} and as absolute path)")

             if track_file_path:
                 log.info(f"Loading track from specified file: {track_file_path}")
                 track = Track(name=os.path.splitext(os.path.basename(track_file_path))[0])
                 if track.load_from_file(track_file_path):
                     return track
                 else:
                     log.error(f"Failed to load specified track file: {track_file_path}")
                     # Proceed to generate if allowed, otherwise return None
                     if not generate_if_missing: return None
             elif not generate_if_missing:
                 return None

        # If no file specified or file not found, and generation is allowed
        if generate_if_missing:
            log.info("Generating a new track...")
            try:
                gen_config = self.config.get('track_generator', {})
                # Allow overriding generator settings via main config
                gen_settings_file = gen_config.get('settings_file')
                gen_params = {}
                if gen_settings_file:
                    gen_settings_path_abs = os.path.join('configs/track_generator', gen_settings_file)
                    if os.path.exists(gen_settings_path_abs):
                        with open(gen_settings_path_abs, 'r') as f:
                            gen_params = yaml.safe_load(f).get('generation',{})
                    else:
                        log.warning(f"Track generator settings file not found: {gen_settings_path_abs}")
                # Merge with inline params from main config
                gen_params.update(gen_config)

                # Ensure output directories exist
                os.makedirs(os.path.dirname(self.track_output_dir), exist_ok=True)
                os.makedirs(self.track_output_dir, exist_ok=True)

                generator = FSTrackGenerator(
                     base_dir=os.path.dirname(self.track_output_dir), # Dir for metadata
                     output_dir_override=self.track_output_dir, # Dir for track files
                     track_width=float(gen_params.get('track_width', 3.0)),
                     min_length=float(gen_params.get('min_length', 250)),
                     max_length=float(gen_params.get('max_length', 450)),
                     n_points=int(gen_params.get('n_points_voronoi', 60)),
                     n_regions=int(gen_params.get('n_regions_select', 20)),
                     bounds=(float(gen_params.get('min_bounds', 0.0)), float(gen_params.get('max_bounds', 150.0))),
                     cone_spacing=float(gen_params.get('cone_spacing', 3.5)),
                     curvature_threshold=float(gen_params.get('curvature_threshold', 0.3)),
                     straight_threshold=float(gen_params.get('straight_threshold', 0.05)),
                     start_straight_length=float(gen_params.get('start_finish', {}).get('min_straight_length', 15.0))
                )
                mode_str = gen_params.get('preferred_mode', 'EXTEND').upper()
                mode = TrackMode[mode_str] if mode_str in TrackMode.__members__ else TrackMode.EXTEND
                metadata = generator.generate_track(mode=mode)

                if metadata and 'filepath' in metadata:
                    track = Track(name=os.path.splitext(metadata['filename'])[0])
                    # Use the new Track class loading mechanism
                    if track.load_from_file(metadata['filepath']):
                         return track
                    else:
                         log.error(f"Failed to load the newly generated track: {metadata['filepath']}")
                else:
                    log.error("Track generation failed.")
            except Exception as e:
                log.error(f"Error during track generation: {e}", exc_info=True)

        log.error("Could not load or generate a track.")
        return None


# =========================================================================
# Simulation Manager (Orchestrator)
# =========================================================================
class SimulationManager:
    """Orchestrates the simulation setup, execution, and reporting."""
    def __init__(self, config: Dict):
        self.config = config
        self.output_dir = config['output_dir']
        self.vehicle: Optional[Vehicle] = None
        self.track: Optional[Track] = None
        self.results: Dict[str, Any] = {} # Store results from different simulations

    def setup(self):
        """Set up the simulation environment (vehicle, track)."""
        log.info("--- Setting up Simulation Environment ---")
        self.vehicle = VehicleFactory.create_vehicle(self.config)

        # Ensure cornering calculator is available and initialized
        if CorneringPerformance_available and self.vehicle:
            if not hasattr(self.vehicle, 'cornering') or self.vehicle.cornering is None:
                 log.debug("Initializing CorneringPerformance for vehicle.")
                 self.vehicle.cornering = CorneringPerformance(self.vehicle)
        elif not CorneringPerformance_available:
             log.warning("CorneringPerformance module not available, some analyses might fail.")


        track_manager = TrackManager(self.config)
        track_arg = self.config.get('cli_args', {}).get('track_file') # Get from parsed args if stored
        generate_track = self.config.get('cli_args', {}).get('generate_track', False)
        # Only get track if needed by enabled events
        run_lap = self.config.get('event_settings',{}).get('lap_time')
        run_endurance = self.config.get('event_settings',{}).get('endurance')
        run_sensitivity = self.config.get('analysis_settings',{}).get('enable_weight_sensitivity')
        run_optimization = self.config.get('analysis_settings',{}).get('enable_lap_optimization')

        if run_lap or run_endurance or run_sensitivity or run_optimization:
             self.track = track_manager.get_track(track_arg, generate_if_missing=generate_track)
             if not self.track:
                  log.error("Track is required but could not be loaded/generated. Disabling relevant events/analyses.")
                  self.config['event_settings']['lap_time'] = False
                  self.config['event_settings']['endurance'] = False
                  self.config['analysis_settings']['enable_weight_sensitivity'] = False
                  self.config['analysis_settings']['enable_lap_optimization'] = False


    def run_simulations(self):
        """Run enabled simulations."""
        log.info("--- Running Enabled Simulations ---")
        event_settings = self.config.get('event_settings', {})

        if event_settings.get('acceleration') and self.vehicle:
            log.info("--- Running Acceleration Event ---")
            accel_save_dir = os.path.join(self.output_dir, 'acceleration')
            try:
                # Pass the configured vehicle and output dir
                accel_result = run_fs_acceleration_simulation(self.vehicle, accel_save_dir)
                # Store best metrics (or whole report if needed)
                best_time = float('inf')
                best_metrics_run = {}
                if accel_result and 'report' in accel_result and 'simulations' in accel_result['report']:
                     for run_name, run_metrics in accel_result['report']['metrics'].items():
                          if run_metrics.get('finish_time') is not None and run_metrics['finish_time'] < best_time:
                               best_time = run_metrics['finish_time']
                               best_metrics_run = run_metrics
                     self.results['acceleration'] = best_metrics_run
                else:
                     self.results['acceleration'] = {'error': 'No valid simulation metrics found.'}

            except Exception as e:
                log.error(f"Acceleration simulation failed: {e}", exc_info=True)
                self.results['acceleration'] = {'error': str(e)}

        # Add Skidpad simulation here if implemented...

        if event_settings.get('lap_time') and self.track and self.vehicle:
            log.info("--- Running Lap Time (Autocross) Event ---")
            lap_save_dir = os.path.join(self.output_dir, 'lap_time')
            try:
                 # Ensure track source file exists
                 if self.track.source_file is None:
                      raise ValueError("Track source file path is missing.")

                 lap_result = run_fs_lap_simulation(
                      self.vehicle,
                      self.track.source_file, # Need the file path
                      include_thermal=self.config.get('simulation_settings',{}).get('include_thermal', True),
                      save_dir=lap_save_dir
                 )
                 self.results['lap_time'] = lap_result.get('metrics', {}) # Store metrics
            except Exception as e:
                 log.error(f"Lap Time simulation failed: {e}", exc_info=True)
                 self.results['lap_time'] = {'error': str(e)}

        if event_settings.get('endurance') and self.track and self.vehicle:
            log.info("--- Running Endurance Event ---")
            endurance_save_dir = os.path.join(self.output_dir, 'endurance')
            try:
                 # Ensure track source file exists
                 if self.track.source_file is None:
                      raise ValueError("Track source file path is missing.")

                 endurance_cfg = self.config.get('endurance_settings', {})
                 endurance_output = run_endurance_simulation(
                      self.vehicle,
                      self.track.source_file,
                      output_dir=endurance_save_dir,
                      include_thermal=self.config.get('simulation_settings',{}).get('include_thermal', True),
                      num_laps=int(endurance_cfg.get('laps', 22)) # Ensure laps is int
                 )
                 self.results['endurance'] = endurance_output.get('results', {}) # Store detailed results
                 self.results['endurance_score'] = endurance_output.get('score', {}) # Store score
            except Exception as e:
                 log.error(f"Endurance simulation failed: {e}", exc_info=True)
                 self.results['endurance'] = {'error': str(e)}

    def run_analyses(self):
        """Run enabled analyses."""
        log.info("--- Running Enabled Analyses ---")
        analysis_settings = self.config.get('analysis_settings', {})

        if analysis_settings.get('enable_weight_sensitivity') and self.track and self.vehicle:
            log.info("--- Running Weight Sensitivity Analysis ---")
            analysis_save_dir = os.path.join(self.output_dir, 'analysis', 'weight_sensitivity')
            try:
                 # Ensure track source file exists
                 if self.track.source_file is None:
                      raise ValueError("Track source file path is missing for sensitivity analysis.")

                 ws_config = self.config.get('weight_sensitivity_settings',{})
                 # Use base vehicle weight if range not specified
                 base_weight = self.vehicle.mass
                 weight_range_rel = ws_config.get('weight_range_relative_kg', (-30, 30))
                 weight_range = (base_weight + weight_range_rel[0], base_weight + weight_range_rel[1])

                 base_dist = self.vehicle.weight_distribution_front
                 dist_range_rel = ws_config.get('distribution_range_relative', (-0.05, 0.05)) # +/- 5%
                 dist_range = (base_dist + dist_range_rel[0], base_dist + dist_range_rel[1])

                 num_points = int(ws_config.get('num_points', 5)) # Ensure int

                 ws_report = analyze_weight_sensitivity(
                      vehicle=self.vehicle, # Pass the main vehicle
                      track_file=self.track.source_file,
                      weight_range_kg=weight_range,
                      distribution_range=dist_range,
                      num_points=num_points,
                      save_dir=analysis_save_dir
                 )
                 self.results['weight_sensitivity'] = ws_report # Store the report
            except Exception as e:
                 log.error(f"Weight Sensitivity analysis failed: {e}", exc_info=True)
                 self.results['weight_sensitivity'] = {'error': str(e)}

        if analysis_settings.get('enable_lap_optimization') and self.track and self.vehicle:
             log.info("--- Running Lap Time Optimization Analysis ---")
             analysis_save_dir = os.path.join(self.output_dir, 'analysis', 'lap_optimization')
             try:
                  # Ensure track source file exists
                  if self.track.source_file is None:
                      raise ValueError("Track source file path is missing for optimization analysis.")

                  optim_config_path = self.config.get('lap_optimization_config_path', 'configs/lap_time/optimal_lap_time.yaml')
                  # Ensure path exists before passing
                  if not os.path.exists(optim_config_path):
                       log.warning(f"Optimization config file not found: {optim_config_path}. Advanced optimization might use defaults.")
                       optim_config_path = None # Pass None if not found

                  # Compare basic vs advanced
                  optim_comparison = compare_optimization_methods(
                       vehicle=self.vehicle,
                       track_file=self.track.source_file,
                       config_file=optim_config_path,
                       include_thermal=self.config.get('simulation_settings',{}).get('include_thermal', True),
                       save_dir=analysis_save_dir
                  )
                  self.results['lap_optimization_comparison'] = optim_comparison
             except Exception as e:
                  log.error(f"Lap Time Optimization analysis failed: {e}", exc_info=True)
                  self.results['lap_optimization_comparison'] = {'error': str(e)}


    def generate_report(self):
        """Generate final summary report and save results."""
        log.info("--- Generating Final Report ---")
        # 1. Save combined results dictionary
        results_path = os.path.join(self.output_dir, 'final_results.json')
        try:
             # Custom encoder to handle numpy types and potentially other objects
             def default_serializer(obj):
                 if isinstance(obj, np.ndarray): return obj.tolist()
                 if isinstance(obj, (np.int_, np.integer)): return int(obj)
                 if isinstance(obj, (np.float_, np.floating)): return float(obj)
                 if isinstance(obj, (np.bool_)): return bool(obj)
                 # Handle potential enums
                 if isinstance(obj, Enum): return obj.name
                 # Try converting other known types or fallback to string
                 try:
                     if pd.isna(obj): return None # Handle pandas NaT/NaN specifically
                     # Add checks for other specific non-serializable types if needed
                     if isinstance(obj, complex): return {'real': obj.real, 'imag': obj.imag}
                     return str(obj) # Fallback to string representation
                 except Exception:
                     log.debug(f"Could not serialize object of type {type(obj)}, returning None.")
                     return None # Return None if conversion fails

             # Create a deep copy to avoid modifying original results during serialization attempt
             results_to_save = copy.deepcopy(self.results)

             with open(results_path, 'w') as f:
                  json.dump(results_to_save, f, indent=2, default=default_serializer)
             log.info(f"Combined results saved to: {results_path}")
        except TypeError as te:
             log.error(f"Failed to serialize results to JSON: {te}. Check for non-serializable types.")
             # Optionally save as YAML as a fallback?
             try:
                 yaml_path = os.path.join(self.output_dir, 'final_results.yaml')
                 # Use a modified serializer for YAML if necessary, or hope it handles more types
                 with open(yaml_path, 'w') as yf:
                     yaml.dump(results_to_save, yf, default_flow_style=False, sort_keys=False, allow_unicode=True, default_style=None)
                 log.info(f"Saved results as YAML fallback: {yaml_path}")
             except Exception as ye:
                 log.error(f"Failed to save results as YAML fallback either: {ye}")
        except Exception as e:
             log.error(f"Failed to save combined results: {e}")


        # 2. Create a simple text summary
        summary_path = os.path.join(self.output_dir, 'summary_report.txt')
        try:
             with open(summary_path, 'w') as f:
                 f.write("KCL Formula Student Powertrain Simulation Report\n")
                 f.write("="*50 + "\n")
                 f.write(f"Timestamp: {self.config.get('timestamp', 'N/A')}\n") # Get timestamp from config
                 f.write(f"Output Directory: {self.output_dir}\n")
                 if self.vehicle:
                     f.write(f"Vehicle Base Mass: {getattr(self.vehicle, 'mass', 'N/A'):.1f} kg\n")
                 else:
                     f.write("Vehicle: N/A\n")
                 if self.track:
                     track_len = getattr(self.track, 'total_length', 0.0) or getattr(self.track,'track_length', 0.0)
                     f.write(f"Track: {getattr(self.track, 'name', 'N/A')} ({track_len:.1f} m)\n")
                 else:
                     f.write("Track: N/A\n")

                 f.write("\n--- Event Results Summary ---\n")
                 if 'acceleration' in self.results and isinstance(self.results['acceleration'], dict) and 'error' not in self.results['acceleration']:
                      accel = self.results['acceleration']
                      finish_time = accel.get('finish_time', 'N/A')
                      t60 = accel.get('time_to_60mph', 'N/A')
                      f.write(f" Acceleration (75m): {finish_time if finish_time != 'N/A' else 'N/A':.3f} s\n")
                      f.write(f"  0-60 mph: {t60 if t60 != 'N/A' else 'N/A':.3f} s\n")
                 elif 'acceleration' in self.results and isinstance(self.results['acceleration'], dict):
                      f.write(f" Acceleration: ERROR ({self.results['acceleration'].get('error', 'Unknown')})\n")
                 else: f.write(" Acceleration: Not run or no results.\n")

                 if 'lap_time' in self.results and isinstance(self.results['lap_time'], dict) and 'error' not in self.results['lap_time']:
                      lap = self.results['lap_time']
                      lap_t = lap.get('lap_time', 'N/A')
                      avg_s = lap.get('avg_speed_kph', 'N/A')
                      f.write(f" Lap Time (Autocross): {lap_t if lap_t != 'N/A' else 'N/A':.3f} s\n")
                      f.write(f"  Avg Speed: {avg_s if avg_s != 'N/A' else 'N/A':.1f} km/h\n")
                 elif 'lap_time' in self.results and isinstance(self.results['lap_time'], dict):
                      f.write(f" Lap Time (Autocross): ERROR ({self.results['lap_time'].get('error', 'Unknown')})\n")
                 else: f.write(" Lap Time (Autocross): Not run or no results.\n")

                 if 'endurance' in self.results and isinstance(self.results['endurance'], dict) and 'error' not in self.results['endurance']:
                      endurance = self.results['endurance']
                      score = self.results.get('endurance_score', {})
                      f.write(f" Endurance ({endurance.get('completed_laps',0)} laps): {endurance.get('status','DNF')}\n")
                      if endurance.get('completed'):
                           tot_t = endurance.get('total_time_s','N/A')
                           tot_f = endurance.get('total_fuel_L','N/A')
                           tot_s = score.get('total_score','N/A')
                           f.write(f"  Total Time: {tot_t if tot_t != 'N/A' else 'N/A':.2f} s\n")
                           f.write(f"  Total Fuel: {tot_f if tot_f != 'N/A' else 'N/A':.2f} L\n")
                           f.write(f"  Total Score: {tot_s if tot_s != 'N/A' else 'N/A':.1f}\n")
                 elif 'endurance' in self.results and isinstance(self.results['endurance'], dict):
                     f.write(f" Endurance: ERROR ({self.results['endurance'].get('error', 'Unknown')})\n")
                 else: f.write(" Endurance: Not run or no results.\n")


                 # Add summaries from analyses if run
                 f.write("\n--- Analysis Summaries ---\n")
                 if 'weight_sensitivity' in self.results and isinstance(self.results['weight_sensitivity'], dict) and 'error' not in self.results['weight_sensitivity']:
                     f.write(" Weight Sensitivity: Analysis performed (see JSON/plots).\n")
                     # Add key sensitivity value if available
                     accel_sens = self.results['weight_sensitivity'].get('acceleration_sensitivity', {}).get('finish_time', {}).get('slope_per_10kg')
                     lap_sens = self.results['weight_sensitivity'].get('lap_time_sensitivity', {}).get('lap_time', {}).get('slope_per_10kg')
                     if accel_sens: f.write(f"  Accel Sensitivity: {accel_sens:+.3f} s/10kg\n")
                     if lap_sens: f.write(f"  Lap Time Sensitivity: {lap_sens:+.3f} s/10kg\n")
                 elif 'weight_sensitivity' in self.results and isinstance(self.results['weight_sensitivity'], dict):
                      f.write(f" Weight Sensitivity: ERROR ({self.results['weight_sensitivity'].get('error', 'Unknown')})\n")
                 else: f.write(" Weight Sensitivity: Not run.\n")

                 if 'lap_optimization_comparison' in self.results and isinstance(self.results['lap_optimization_comparison'], dict) and 'error' not in self.results['lap_optimization_comparison']:
                      comp = self.results['lap_optimization_comparison']
                      basic_t = comp.get('basic',{}).get('lap_time','N/A')
                      adv_t = comp.get('advanced',{}).get('lap_time','N/A')
                      f.write(f" Lap Optimization Comparison:\n")
                      f.write(f"  Basic: {basic_t if basic_t != 'N/A' else 'N/A':.3f}s | Advanced: {adv_t if adv_t != 'N/A' else 'N/A':.3f}s\n")
                 elif 'lap_optimization_comparison' in self.results and isinstance(self.results['lap_optimization_comparison'], dict):
                     f.write(f" Lap Optimization Comparison: ERROR ({self.results['lap_optimization_comparison'].get('error', 'Unknown')})\n")
                 else: f.write(" Lap Optimization Comparison: Not run.\n")

             log.info(f"Summary report saved to: {summary_path}")
        except Exception as e:
            log.error(f"Failed to save summary report: {e}")

# =========================================================================
# Main Execution Block
# =========================================================================
def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="KCL Formula Student Powertrain Simulation")
    parser.add_argument("-c", "--config", type=str, default=None, help="Path to main simulation config file (optional).")
    parser.add_argument("--track-file", type=str, help="Path to a specific track file (overrides config).")
    parser.add_argument("--generate-track", action="store_true", help="Generate a new track instead of loading.")
    parser.add_argument("--output-dir", type=str, help="Specify exact output directory (overrides default timestamped dir).")
    parser.add_argument("--log-level", type=str, default="INFO", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], help="Set logging level.")
    parser.add_argument("--time-step", type=float, help="Override simulation time step.")
    parser.add_argument("--no-thermal", action="store_true", help="Disable thermal simulations.")
    parser.add_argument("--run-accel", type=bool, default=None, help="Explicitly enable/disable Acceleration event.")
    parser.add_argument("--run-lap", type=bool, default=None, help="Explicitly enable/disable Lap Time event.")
    parser.add_argument("--run-endurance", type=bool, default=None, help="Explicitly enable/disable Endurance event.")
    parser.add_argument("--run-sensitivity", action="store_true", help="Enable weight sensitivity analysis.")
    parser.add_argument("--run-optimization", action="store_true", help="Enable lap time optimization comparison.")
    parser.add_argument("--no-plots", action="store_true", help="Disable saving of plot files.")

    return parser.parse_args()

def main():
    """Main function to orchestrate the simulation."""
    # 1. Parse Arguments
    args = parse_arguments()

    # 2. Configure Logging (configure only once)
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    log_format = '%(asctime)s [%(levelname)-7s] %(name)-25s: %(message)s'
    log_datefmt = '%H:%M:%S'
    # Remove existing handlers before configuring to avoid duplicate logs
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    logging.basicConfig(level=log_level, format=log_format, datefmt=log_datefmt)

    log.info("Starting KCL FS Powertrain Simulation...")
    start_time = time.time()

    # 3. Load Configuration
    # Pass CLI args to config manager for overrides and output dir setup
    config_manager = ConfigurationManager()
    # Pass CLI args explicitly here
    # If no --config is provided, args.config will be None, loading defaults
    config = config_manager.load_configuration(main_config_path=args.config, cli_args=args)
    # Store parsed args in config for later access if needed (e.g., track file override)
    config['cli_args'] = vars(args) # Store args as dict
    # Add timestamp to config for reporting
    config['timestamp'] = config_manager.timestamp

    # Apply plot setting override (double check it's applied)
    if args.no_plots:
         config.setdefault('output_settings', {})['save_plots'] = False
    # Set plot style (imported from utils)
    try:
        set_plot_style('clean') # Set default plotting style
    except Exception as e:
        log.warning(f"Failed to set plot style: {e}")


    # 4. Initialize Simulation Manager (pass the final config dict)
    simulation_manager = SimulationManager(config)

    # 5. Setup Environment (Vehicle, Track)
    try:
        simulation_manager.setup()
        # Additional check: Ensure vehicle has cornering calculator after setup
        if simulation_manager.vehicle and CorneringPerformance_available and \
           (not hasattr(simulation_manager.vehicle, 'cornering') or simulation_manager.vehicle.cornering is None):
             log.debug("Re-initializing CorneringPerformance within main setup.")
             simulation_manager.vehicle.cornering = CorneringPerformance(simulation_manager.vehicle)
    except Exception as e:
         log.critical(f"Failed to set up simulation environment: {e}", exc_info=True)
         sys.exit(1)

    # 6. Run Simulations
    try:
        simulation_manager.run_simulations()
    except Exception as e:
         log.critical(f"Critical error during event simulations: {e}", exc_info=True)
         # Continue to analysis if possible? Or exit? For now, continue.

    # 7. Run Analyses
    try:
        # Update analysis settings from CLI args if provided
        # Ensure analysis_settings dictionary exists
        if 'analysis_settings' not in simulation_manager.config: simulation_manager.config['analysis_settings'] = {}
        if args.run_sensitivity: simulation_manager.config['analysis_settings']['enable_weight_sensitivity'] = True
        if args.run_optimization: simulation_manager.config['analysis_settings']['enable_lap_optimization'] = True
        simulation_manager.run_analyses()
    except Exception as e:
         log.error(f"Error during analysis: {e}", exc_info=True)

    # 8. Generate Report
    try:
        simulation_manager.generate_report()
    except Exception as e:
         log.error(f"Error generating final report: {e}", exc_info=True)

    # 9. Final Summary
    end_time = time.time()
    total_duration = end_time - start_time
    log.info("--- Simulation Finished ---")
    log.info(f"Total execution time: {total_duration:.2f} seconds")
    if simulation_manager.output_dir:
        log.info(f"Results saved in: {simulation_manager.output_dir}")
    else:
        log.warning("Output directory was not set, results may not be saved correctly.")

if __name__ == "__main__":
    main()