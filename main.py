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
import matplotlib.pyplot as plt
import pandas as pd
import time
import yaml
import logging
import argparse
import copy
from datetime import datetime
from typing import Dict, Any, Optional, List

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
    from kcl_fs_powertrain.performance.lap_time import LapTimeSimulator, run_fs_lap_simulation
    from kcl_fs_powertrain.performance.endurance import EnduranceSimulator, run_endurance_simulation

    # Analysis Tools
    from kcl_fs_powertrain.performance.weight_sensitivity import WeightSensitivityAnalyzer, analyze_weight_sensitivity
    from kcl_fs_powertrain.performance.lap_time_optimization import run_lap_optimization, compare_optimization_methods

    # Track Generation
    from kcl_fs_powertrain.track_generator.generator import FSTrackGenerator
    from kcl_fs_powertrain.track_generator.utils import generate_multiple_tracks
    from kcl_fs_powertrain.track_generator.enums import TrackMode, SimType

    # Plotting & Utils
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
        config_path = self.config.get(f'{component_key}_config_path',
                                      os.path.join(self.default_config_dir, default_rel_path))

        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    component_cfg = yaml.safe_load(f)
                    # Store the loaded component config under its key
                    # Handle potential nested structure (e.g., 'engine' key within engine file)
                    if component_key in component_cfg:
                         self.config[component_key] = component_cfg[component_key]
                    else:
                         self.config[component_key] = component_cfg # Assume file contains the dict directly
                log.debug(f"Loaded {component_key} config from: {config_path}")
            except Exception as e:
                log.error(f"Error loading {component_key} config '{config_path}': {e}")
        else:
            log.warning(f"{component_key} config file not found: {config_path}")


    def _apply_cli_overrides(self, args: argparse.Namespace):
        """Override configuration values with CLI arguments."""
        log.debug("Applying CLI overrides...")
        if args.time_step: self.config['simulation_settings']['time_step'] = args.time_step
        if args.no_thermal: self.config['simulation_settings']['include_thermal'] = False
        if args.run_accel is not None: self.config['event_settings']['acceleration'] = args.run_accel
        if args.run_lap is not None: self.config['event_settings']['lap_time'] = args.run_lap
        if args.run_endurance is not None: self.config['event_settings']['endurance'] = args.run_endurance
        # Add more overrides as needed

    def _setup_output_directory(self, args: Optional[argparse.Namespace] = None):
        """Set up the main output directory for this simulation run."""
        base_dir = self.config.get('output_settings', {}).get('base_dir', 'data/output')
        # Allow CLI to override base directory
        if args and args.output_dir:
             self.output_dir = args.output_dir # User specified exact directory
             # May or may not include timestamp based on user intent
        else:
             # Create a timestamped subdirectory within the base directory
             run_name = f"sim_{self.timestamp}"
             self.output_dir = os.path.join(base_dir, run_name)

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
        # It should handle loading component configs based on paths in the main config
        # For now, assume the main config dict *contains* the component parameters needed by constructors
        try:
            # Pass the relevant sections of the config to the Vehicle constructor or its initializer methods
            # This requires Vehicle.__init__ or helper methods to parse these sections
            vehicle_instance = Vehicle(config=config) # Vehicle init needs to handle nested config
            log.info("Vehicle instance created successfully.")
            return vehicle_instance
        except Exception as e:
            log.error(f"Failed to create vehicle from configuration: {e}", exc_info=True)
            log.warning("Attempting to create default FS vehicle as fallback.")
            return create_formula_student_vehicle() # Fallback


# =========================================================================
# Track Management
# =========================================================================
class TrackManager:
    """Handles loading or generating tracks."""
    def __init__(self, config: Dict):
        self.config = config
        self.track_config_dir = config.get('track_config_dir', 'configs/tracks')
        self.track_output_dir = config.get('output_settings', {}).get('generated_track_dir',
                                          config.get('output_dir', 'data/output/tracks')) # Save generated tracks in run output

    def get_track(self, track_arg: Optional[str] = None, generate_if_missing: bool = True) -> Optional[Track]:
        """Loads a specific track file or generates one if requested/missing."""
        track_file = track_arg or self.config.get('track_file')

        if track_file and os.path.exists(track_file):
            log.info(f"Loading track from specified file: {track_file}")
            track = Track(name=os.path.splitext(os.path.basename(track_file))[0])
            if track.load_from_file(track_file):
                return track
            else:
                log.error(f"Failed to load specified track file: {track_file}")
                return None # Explicitly return None on failure
        elif track_file:
            log.warning(f"Specified track file not found: {track_file}")
            if not generate_if_missing:
                return None

        # If no file specified or file not found, and generation is allowed
        if generate_if_missing:
            log.info("Generating a new track...")
            try:
                gen_config = self.config.get('track_generator', {})
                gen_settings_path = gen_config.get('settings_file', 'configs/track_generator/generator_settings.yaml')
                gen_params = {}
                if os.path.exists(gen_settings_path):
                     with open(gen_settings_path, 'r') as f:
                          gen_params = yaml.safe_load(f).get('generation',{})

                # Pass relevant params from main config or generator settings
                generator = FSTrackGenerator(
                     base_dir=os.path.dirname(self.track_output_dir), # Dir for metadata
                     output_dir_override=self.track_output_dir, # Dir for track files
                     track_width=gen_params.get('track_width', 3.0),
                     min_length=gen_params.get('min_length', 250),
                     max_length=gen_params.get('max_length', 450),
                     # Add other params from gen_params...
                )
                mode_str = gen_params.get('preferred_mode', 'EXTEND').upper()
                mode = TrackMode[mode_str] if mode_str in TrackMode.__members__ else TrackMode.EXTEND
                metadata = generator.generate_track(mode=mode)

                if metadata:
                    track = Track(name=os.path.splitext(metadata['filename'])[0])
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
        # Ensure cornering calculator is available
        if not hasattr(self.vehicle, 'cornering') or self.vehicle.cornering is None:
             self.vehicle.cornering = CorneringPerformance(self.vehicle)

        track_manager = TrackManager(self.config)
        track_arg = self.config.get('cli_args', {}).get('track_file') # Get from parsed args if stored
        generate_track = self.config.get('cli_args', {}).get('generate_track', False)
        # Only get track if needed by enabled events
        if self.config.get('event_settings',{}).get('lap_time') or \
           self.config.get('event_settings',{}).get('endurance'):
             self.track = track_manager.get_track(track_arg, generate_if_missing=generate_track)
             if not self.track:
                  log.error("Track is required for lap time/endurance but could not be loaded/generated. Disabling relevant events.")
                  self.config['event_settings']['lap_time'] = False
                  self.config['event_settings']['endurance'] = False


    def run_simulations(self):
        """Run enabled simulations."""
        log.info("--- Running Enabled Simulations ---")
        event_settings = self.config.get('event_settings', {})

        if event_settings.get('acceleration'):
            log.info("--- Running Acceleration Event ---")
            accel_save_dir = os.path.join(self.output_dir, 'acceleration')
            try:
                # Pass the configured vehicle and output dir
                accel_result = run_fs_acceleration_simulation(self.vehicle, accel_save_dir)
                self.results['acceleration'] = accel_result.get('report', {}).get('metrics',{}).get('full_optimized',{}) # Store best metrics
            except Exception as e:
                log.error(f"Acceleration simulation failed: {e}", exc_info=True)
                self.results['acceleration'] = {'error': str(e)}

        # Add Skidpad simulation here if implemented...

        if event_settings.get('lap_time') and self.track:
            log.info("--- Running Lap Time (Autocross) Event ---")
            lap_save_dir = os.path.join(self.output_dir, 'lap_time')
            try:
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

        if event_settings.get('endurance') and self.track:
            log.info("--- Running Endurance Event ---")
            endurance_save_dir = os.path.join(self.output_dir, 'endurance')
            try:
                 endurance_cfg = self.config.get('endurance_settings', {})
                 endurance_result = run_endurance_simulation(
                      self.vehicle,
                      self.track.source_file,
                      output_dir=endurance_save_dir,
                      include_thermal=self.config.get('simulation_settings',{}).get('include_thermal', True),
                      num_laps=endurance_cfg.get('laps', 22)
                 )
                 self.results['endurance'] = endurance_result.get('results', {}) # Store detailed results
                 self.results['endurance_score'] = endurance_result.get('score', {}) # Store score
            except Exception as e:
                 log.error(f"Endurance simulation failed: {e}", exc_info=True)
                 self.results['endurance'] = {'error': str(e)}

    def run_analyses(self):
        """Run enabled analyses."""
        log.info("--- Running Enabled Analyses ---")
        analysis_settings = self.config.get('analysis_settings', {})

        if analysis_settings.get('enable_weight_sensitivity') and self.track:
            log.info("--- Running Weight Sensitivity Analysis ---")
            analysis_save_dir = os.path.join(self.output_dir, 'analysis', 'weight_sensitivity')
            try:
                 ws_config = self.config.get('weight_sensitivity_settings',{})
                 weight_range = tuple(ws_config.get('weight_range_kg', (self.vehicle.mass-30, self.vehicle.mass+30)))
                 dist_range = tuple(ws_config.get('distribution_range', (0.43, 0.53)))
                 num_points = ws_config.get('num_points', 5)

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

        if analysis_settings.get('enable_lap_optimization') and self.track:
             log.info("--- Running Lap Time Optimization Analysis ---")
             analysis_save_dir = os.path.join(self.output_dir, 'analysis', 'lap_optimization')
             try:
                  optim_config_path = self.config.get('lap_optimization_config_path', 'configs/lap_time/optimal_lap_time.yaml')
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
        results_path = self.get_output_path('final_results.json')
        try:
             with open(results_path, 'w') as f:
                  # Custom encoder needed for potential numpy arrays or objects
                  def default_serializer(obj):
                      if isinstance(obj, np.ndarray): return obj.tolist()
                      if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)): return int(obj)
                      if isinstance(obj, (np.float_, np.float16, np.float32, np.float64)): return float(obj)
                      if isinstance(obj, (np.bool_)): return bool(obj)
                      if hasattr(obj, 'to_dict') and callable(obj.to_dict): return obj.to_dict()
                      try: return str(obj) # Fallback to string
                      except: return None
                  json.dump(self.results, f, indent=2, default=default_serializer)
             log.info(f"Combined results saved to: {results_path}")
        except Exception as e:
             log.error(f"Failed to save combined results JSON: {e}")

        # 2. Create a simple text summary
        summary_path = self.get_output_path('summary_report.txt')
        try:
             with open(summary_path, 'w') as f:
                 f.write("KCL Formula Student Powertrain Simulation Report\n")
                 f.write("="*50 + "\n")
                 f.write(f"Timestamp: {self.timestamp}\n")
                 f.write(f"Output Directory: {self.output_dir}\n")
                 f.write(f"Vehicle Base Mass: {self.vehicle.mass:.1f} kg\n")
                 if self.track: f.write(f"Track: {self.track.name} ({self.track.total_length:.1f} m)\n")
                 f.write("\n--- Event Results Summary ---\n")
                 if 'acceleration' in self.results and 'error' not in self.results['acceleration']:
                      accel = self.results['acceleration']
                      f.write(f" Acceleration (75m): {accel.get('finish_time', 'N/A'):.3f} s\n")
                      f.write(f"  0-60 mph: {accel.get('time_to_60mph', 'N/A'):.3f} s\n")
                 if 'lap_time' in self.results and 'error' not in self.results['lap_time']:
                      lap = self.results['lap_time']
                      f.write(f" Lap Time (Autocross): {lap.get('lap_time', 'N/A'):.3f} s\n")
                      f.write(f"  Avg Speed: {lap.get('avg_speed_kph', 'N/A'):.1f} km/h\n")
                 if 'endurance' in self.results and 'error' not in self.results['endurance']:
                      endurance = self.results['endurance']
                      score = self.results['endurance_score']
                      f.write(f" Endurance ({endurance.get('completed_laps',0)} laps): {endurance.get('status','DNF')}\n")
                      if endurance.get('completed'):
                           f.write(f"  Total Time: {endurance.get('total_time_s','N/A'):.2f} s\n")
                           f.write(f"  Total Fuel: {endurance.get('total_fuel_L','N/A'):.2f} L\n")
                           f.write(f"  Total Score: {score.get('total_score','N/A'):.1f}\n")

                 # Add summaries from analyses if run
                 f.write("\n--- Analysis Summaries ---\n")
                 if 'weight_sensitivity' in self.results and 'error' not in self.results['weight_sensitivity']:
                     f.write(" Weight Sensitivity: Analysis performed (see JSON/plots).\n")
                 if 'lap_optimization_comparison' in self.results and 'error' not in self.results['lap_optimization_comparison']:
                      comp = self.results['lap_optimization_comparison']
                      f.write(f" Lap Optimization Comparison:\n")
                      f.write(f"  Basic: {comp.get('basic',{}).get('lap_time','N/A'):.3f}s | Advanced: {comp.get('advanced',{}).get('lap_time','N/A'):.3f}s\n")

             log.info(f"Summary report saved to: {summary_path}")
        except Exception as e:
            log.error(f"Failed to save summary report: {e}")

    def get_output_path(self, *args) -> str:
         """Construct a path within the simulation's output directory."""
         return os.path.join(self.output_dir, *args)


# =========================================================================
# Main Execution Block
# =========================================================================
def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="KCL Formula Student Powertrain Simulation")
    parser.add_argument("-c", "--config", type=str, default="configs/main_config.yaml", help="Path to main simulation config file.")
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

    # 2. Configure Logging
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(level=log_level, format='%(asctime)s [%(levelname)-7s] %(name)-25s: %(message)s', datefmt='%H:%M:%S')
    # Optionally configure file logging
    # file_handler = logging.FileHandler('simulation.log')
    # file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)-7s] %(name)s: %(message)s'))
    # logging.getLogger().addHandler(file_handler)

    log.info("Starting KCL FS Powertrain Simulation...")
    start_time = time.time()

    # 3. Load Configuration
    # Pass CLI args to config manager for overrides and output dir setup
    config_manager = ConfigurationManager()
    config = config_manager.load_configuration(main_config_path=args.config, cli_args=args)
    # Store parsed args in config for later access if needed
    config['cli_args'] = args

    # Apply plot setting override
    if args.no_plots:
         config['output_settings']['save_plots'] = False
    set_plot_style('clean') # Set default plotting style

    # 4. Initialize Simulation Manager
    simulation_manager = SimulationManager(config)

    # 5. Setup Environment (Vehicle, Track)
    try:
        simulation_manager.setup()
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
        if args.run_sensitivity: config['analysis_settings']['enable_weight_sensitivity'] = True
        if args.run_optimization: config['analysis_settings']['enable_lap_optimization'] = True
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
    log.info(f"Results saved in: {simulation_manager.output_dir}")


if __name__ == "__main__":
    main()