"""
Vehicle model integrating powertrain, thermal, and basic dynamics.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Callable
import yaml
import logging
import copy
import matplotlib.pyplot as plt

# --- Import constants FIRST ---
# Import the necessary constants directly at the module level
try:
    from ..utils.constants import (
        GRAVITY, AIR_DENSITY_SEA_LEVEL, KW_TO_HP, HP_TO_KW, MS_TO_KMH, MS_TO_MPH,
        FS_ACCELERATION_LENGTH, # <--- Import Here
        FS_SKIDPAD_RADIUS     # <--- Import Here too
    )
except ImportError:
    # Define fallbacks if constants module cannot be imported
    GRAVITY = 9.81; AIR_DENSITY_SEA_LEVEL = 1.225; KW_TO_HP = 1.341; HP_TO_KW = 1/KW_TO_HP; MS_TO_KMH = 3.6; MS_TO_MPH = 2.237
    FS_ACCELERATION_LENGTH = 75.0 # Define fallback
    FS_SKIDPAD_RADIUS = 15.25 / 2.0 # Define fallback

# Import powertrain components (handle potential errors)
try:
    from ..engine.motorcycle_engine import MotorcycleEngine
    from ..engine.engine_thermal import EngineHeatModel, ThermalConfig
    from ..transmission.gearing import DrivetrainSystem, Transmission, FinalDrive, Differential
    from ..transmission.cas_system import CASSystem, ShiftDirection, ShiftState # Added ShiftState
    from ..transmission.shift_strategy import StrategyManager, create_formula_student_strategies, StrategyType
    from ..thermal.cooling_system import CoolingSystem as ExternalCoolingSystem
    from ..thermal.cooling_system import create_formula_student_cooling_system
    from ..thermal.side_pod import DualSidePodSystem, create_standard_side_pod_system
    from ..thermal.rear_radiator import RearRadiatorSystem, create_default_rear_radiator_system
    from ..thermal.electric_compressor import CoolingAssistSystem, create_default_cooling_assist_system
    from ..utils.plotting import plot_vehicle_performance_summary, plot_acceleration_results as plot_accel_results_util, save_plot
    from ..performance.lap_time import CorneringPerformance
    from ..performance.acceleration import AccelerationSimulator 
    from ..performance.lap_time import LapTimeSimulator       
except ImportError as e:
    # Define placeholders if imports fail
    logger_fallback = logging.getLogger("Vehicle_Fallback") 
    logger_fallback.error(f"Error importing vehicle components: {e}. Using placeholders.")
    class MotorcycleEngine: pass
    class EngineHeatModel: pass
    class ThermalConfig: pass
    class DrivetrainSystem: pass
    class Transmission: pass
    class FinalDrive: pass
    class Differential: pass
    class CASSystem: pass
    class ShiftState: IDLE = 0; SHIFT_IN_PROGRESS = 1 
    class ShiftDirection: UP = 1; DOWN = -1
    class StrategyManager: pass
    class ExternalCoolingSystem: pass
    class DualSidePodSystem: pass
    class RearRadiatorSystem: pass
    class CoolingAssistSystem: pass
    class CorneringPerformance: pass
    class AccelerationSimulator: pass
    class LapTimeSimulator: pass
    def create_formula_student_strategies(*args, **kwargs): return None
    def create_standard_side_pod_system(*args, **kwargs): return None
    def create_default_rear_radiator_system(*args, **kwargs): return None
    def create_default_cooling_assist_system(*args, **kwargs): return None
    def create_formula_student_cooling_system(*args, **kwargs): return None
    def plot_vehicle_performance_summary(*args, **kwargs): plt.figure(); plt.plot([0,1]); plt.title("Fallback Plot"); plt.show(); plt.close(); return plt.gcf()
    def plot_accel_results_util(*args, **kwargs): plt.figure(); plt.plot([0,1]); plt.title("Fallback Plot"); plt.show(); plt.close(); return plt.gcf()
    def save_plot(fig, path, **kwargs): pass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Vehicle")


class Vehicle:
    """
    Integrates powertrain components into a simulated Formula Student vehicle.
    Handles basic longitudinal dynamics, component interactions, and state updates.
    """

    def __init__(self,
                 config: Optional[Dict] = None, # Accept pre-loaded config dict
                 config_path: Optional[str] = None,
                 engine: Optional[MotorcycleEngine] = None,
                 drivetrain: Optional[DrivetrainSystem] = None,
                 cooling_system: Optional[ExternalCoolingSystem] = None, # Main external cooling
                 shift_manager: Optional[StrategyManager] = None,
                 cas_system: Optional[CASSystem] = None,
                 side_pods: Optional[DualSidePodSystem] = None,
                 rear_radiator: Optional[RearRadiatorSystem] = None,
                 cooling_assist: Optional[CoolingAssistSystem] = None,
                 team_name: str = "KCL Formula Student"):
        """
        Initialize the vehicle model.

        Args:
            config: Optional pre-loaded configuration dictionary.
            config_path: Path to the main vehicle YAML configuration file (used if config dict not provided).
            engine, drivetrain, etc.: Optional pre-configured component instances.
            team_name: Identifying name for the vehicle/team.
        """
        self.team_name = team_name
        self.config_path = config_path
        self.config: Dict = {} # Stores loaded configuration

        # --- Default Vehicle Parameters ---
        self.mass: float = 230.0
        self.frontal_area_m2: float = 1.1
        self.drag_coefficient: float = 1.0
        self.lift_coefficient: float = -2.0
        self.rolling_resistance_coeff: float = 0.015
        self.weight_distribution_front: float = 0.48
        self.wheelbase_m: float = 1.58
        self.track_width_front_m: float = 1.20
        self.track_width_rear_m: float = 1.15
        self.cg_height_m: float = 0.28
        self.tire_radius_m: float = 0.2286
        # Braking parameter (moved here for consistency)
        self.max_braking_g: float = 1.8 # Max braking deceleration relative to g

        # --- Load configuration ---
        # Priority: 1. Passed config dict, 2. config_path
        if config is not None:
            self.config = copy.deepcopy(config) # Use passed dict
            logger.info("Vehicle initialized using provided configuration dictionary.")
        elif config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    self.config = yaml.safe_load(f) or {}
                self.config_path = config_path # Store path if loaded successfully
                logger.info(f"Vehicle base configuration loaded from {config_path}")
            except Exception as e:
                logger.error(f"Error loading vehicle config from {config_path}: {e}")
                self.config = {} # Ensure config is dict even on error
        else:
            logger.warning(f"Vehicle config not provided or path invalid: {config_path}. Using defaults.")
            self.config = {} # Ensure config is dict

        # Apply base vehicle parameters from the config (or defaults if config empty)
        self._apply_base_config()

        # --- Component Initialization ---
        # These methods now rely on self.config being populated
        self._initialize_engine(engine)
        self._initialize_drivetrain(drivetrain)
        # Pass self to cooling system init for potential back-references if needed by factories
        self._initialize_cooling_system(cooling_system, self)
        self._initialize_shifting_systems(shift_manager, cas_system)
        self._initialize_aero_cooling(side_pods, rear_radiator, cooling_assist)

        # Initialize Cornering Performance calculator after core components are set up
        self.cornering = CorneringPerformance(self)

        # --- Current State Variables ---
        self.current_speed_mps: float = 0.0
        self.current_acceleration_mpss: float = 0.0
        self.current_position_m: float = 0.0 # Longitudinal position
        self.current_gear: int = 0 # Neutral initially
        self.current_engine_rpm: float = self.engine.idle_rpm if self.engine else 0.0
        # Control inputs (driver commands)
        self.throttle_input: float = 0.0
        self.brake_input: float = 0.0
        self.steering_angle_rad: float = 0.0

        # Internal simulation state
        self.last_update_time: float = 0.0
        self.include_thermal: bool = self.config.get('simulation_settings',{}).get('include_thermal', True)

        # Initialize thermal state from components if possible
        # Prioritize external cooling system's temp as it represents the bulk fluid temp
        self.coolant_temperature = getattr(self.cooling_system, 'coolant_temp_C', 25.0)
        # Engine block and oil temp might still be tracked separately by engine model
        self.engine_temperature = getattr(self.engine, 'engine_temperature', self.coolant_temperature + 5.0) # Start slightly warmer
        self.oil_temperature = getattr(self.engine, 'oil_temperature', self.coolant_temperature)
        # Vehicle's own thermal factor reflects engine's current derating
        self.thermal_factor = getattr(self.engine, 'thermal_factor', 1.0)

        logger.info(f"{self.team_name} Vehicle initialized. Mass: {self.mass:.1f} kg")
    
    def load_config(self, config_path: str):
        """Load vehicle base parameters from YAML file."""
        if not os.path.exists(config_path):
            logger.error(f"Vehicle configuration file not found: {config_path}")
            return
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f) or {} # Ensure config is a dict

            vehicle_cfg = self.config.get('vehicle', {})
            self.mass = float(vehicle_cfg.get('mass', self.mass))
            self.frontal_area_m2 = float(vehicle_cfg.get('frontal_area_m2', self.frontal_area_m2))
            self.drag_coefficient = float(vehicle_cfg.get('drag_coefficient', self.drag_coefficient))
            self.lift_coefficient = float(vehicle_cfg.get('lift_coefficient', self.lift_coefficient))
            self.rolling_resistance_coeff = float(vehicle_cfg.get('rolling_resistance_coeff', self.rolling_resistance_coeff))
            self.weight_distribution_front = float(vehicle_cfg.get('weight_distribution_front', self.weight_distribution_front))
            self.wheelbase_m = float(vehicle_cfg.get('wheelbase_m', self.wheelbase_m))
            self.track_width_front_m = float(vehicle_cfg.get('track_width_front_m', self.track_width_front_m))
            self.track_width_rear_m = float(vehicle_cfg.get('track_width_rear_m', self.track_width_rear_m))
            self.cg_height_m = float(vehicle_cfg.get('cg_height_m', self.cg_height_m))

            tire_cfg = self.config.get('tires', {})
            self.tire_radius_m = float(tire_cfg.get('radius_m', self.tire_radius_m))

            logger.info(f"Vehicle base parameters loaded from {config_path}")

        except Exception as e:
            logger.error(f"Error loading vehicle config from {config_path}: {e}")
    
    def _apply_base_config(self):
         """Apply base vehicle parameters from the self.config dictionary."""
         vehicle_cfg = self.config.get('vehicle', {})
         self.mass = float(vehicle_cfg.get('mass', self.mass))
         self.frontal_area_m2 = float(vehicle_cfg.get('frontal_area_m2', self.frontal_area_m2))
         self.drag_coefficient = float(vehicle_cfg.get('drag_coefficient', self.drag_coefficient))
         self.lift_coefficient = float(vehicle_cfg.get('lift_coefficient', self.lift_coefficient))
         self.rolling_resistance_coeff = float(vehicle_cfg.get('rolling_resistance_coeff', self.rolling_resistance_coeff))
         self.weight_distribution_front = float(vehicle_cfg.get('weight_distribution_front', self.weight_distribution_front))
         self.wheelbase_m = float(vehicle_cfg.get('wheelbase_m', self.wheelbase_m))
         self.track_width_front_m = float(vehicle_cfg.get('track_width_front_m', self.track_width_front_m))
         self.track_width_rear_m = float(vehicle_cfg.get('track_width_rear_m', self.track_width_rear_m))
         self.cg_height_m = float(vehicle_cfg.get('cg_height_m', self.cg_height_m))
         self.max_braking_g = float(vehicle_cfg.get('max_braking_g', self.max_braking_g))

         tire_cfg = self.config.get('tires', {})
         self.tire_radius_m = float(tire_cfg.get('radius_m', self.tire_radius_m))

         # Apply simulation settings
         sim_cfg = self.config.get('simulation_settings', self.config.get('simulation', {})) # Check both keys
         self.include_thermal = bool(sim_cfg.get('include_thermal', self.include_thermal))
         logger.debug(f"Base parameters applied. Mass={self.mass:.1f}, IncludeThermal={self.include_thermal}")

    def _initialize_engine(self, engine_instance: Optional[MotorcycleEngine]):
        """Initialize the engine component using self.config."""
        if isinstance(engine_instance, MotorcycleEngine):
            self.engine = engine_instance
            logger.info("Using pre-configured Engine instance.")
        else:
            engine_config_ref = self.config.get('engine_config_path', self.config.get('engine')) # Allow path or inline dict
            if isinstance(engine_config_ref, str) and os.path.exists(engine_config_ref):
                logger.info(f"Initializing Engine from config file: {engine_config_ref}")
                self.engine = MotorcycleEngine(config_path=engine_config_ref)
            elif isinstance(engine_config_ref, dict):
                logger.info("Initializing Engine from inline config in vehicle config.")
                self.engine = MotorcycleEngine(engine_params=engine_config_ref)
            else:
                logger.warning("No valid engine config found. Creating default MotorcycleEngine.")
                # Attempt to find default config path relative to vehicle config path if available
                default_engine_path = None
                if self.config_path:
                     # Construct path relative to the *vehicle* config file's location
                     base_cfg_dir = os.path.dirname(self.config_path)
                     # Assume standard project structure: configs/engine/cbr600f4i.yaml relative to project root
                     # This requires knowing the depth of the vehicle config relative to root
                     # Safer approach: Assume a fixed relative path from the vehicle config's dir
                     default_engine_path = os.path.normpath(os.path.join(base_cfg_dir, '..', 'configs', 'engine', 'cbr600f4i.yaml')) # Adjust relative path if needed
                if default_engine_path and os.path.exists(default_engine_path):
                     logger.info(f"Attempting to load default engine config: {default_engine_path}")
                     self.engine = MotorcycleEngine(config_path=default_engine_path)
                else:
                     self.engine = MotorcycleEngine() # Absolute default

        # Ensure essential attributes
        for attr, default in [('idle_rpm', 1300.0), ('redline_rpm', 14000.0),
                              ('max_torque_nm', 65.0), ('engine_temperature', 25.0),
                              ('coolant_temperature', 25.0), ('oil_temperature', 25.0),
                              ('thermal_factor', 1.0), ('max_power_rpm', 12500.0), # Needed by strategy factory
                              ('max_torque_rpm', 10500.0)]: # Needed by strategy factory
            if not hasattr(self.engine, attr): setattr(self.engine, attr, default)

        # Ensure engine has a heat model instance
        if not hasattr(self.engine, 'heat_model') or self.engine.heat_model is None:
             logger.debug("Creating default heat model for engine.")
             # Use thermal config potentially loaded by engine, or default
             thermal_cfg = getattr(self.engine, 'thermal_config', ThermalConfig())
             # Ensure the heat model can be instantiated (check import)
             if EngineHeatModel:
                self.engine.heat_model = EngineHeatModel(thermal_cfg, self.engine)
             else:
                logger.error("EngineHeatModel class not available to create instance.")
                self.engine.heat_model = None

    def _initialize_drivetrain(self, drivetrain_instance: Optional[DrivetrainSystem]):
        """Initialize the drivetrain component using self.config."""
        if isinstance(drivetrain_instance, DrivetrainSystem):
            self.drivetrain = drivetrain_instance
            logger.info("Using pre-configured Drivetrain instance.")
        else:
             # Try loading from config paths or inline dicts specified in main config
             dt_config_path_ref = self.config.get('drivetrain_config_path')
             trans_config_ref = self.config.get('transmission_config_path', self.config.get('transmission'))
             fd_config_ref = self.config.get('final_drive_config_path', self.config.get('final_drive'))
             diff_config_ref = self.config.get('differential_config_path', self.config.get('differential'))

             transmission = None
             final_drive = None
             differential = None

             # Find base directory for relative paths (if main config path exists)
             base_cfg_dir = os.path.dirname(self.config_path) if self.config_path else '.'

             # Helper to resolve path
             def resolve_path(ref):
                 if isinstance(ref, str):
                     path = os.path.join(base_cfg_dir, ref) if not os.path.isabs(ref) else ref
                     return path if os.path.exists(path) else None
                 return None

             # Load Transmission
             trans_path = resolve_path(trans_config_ref)
             if trans_path:
                  with open(trans_path, 'r') as f: trans_params = yaml.safe_load(f).get('transmission', {})
                  transmission = Transmission(**trans_params)
             elif isinstance(trans_config_ref, dict): # Inline dict
                  transmission = Transmission(**trans_config_ref)
             if transmission is None: # Default if loading failed
                 logger.debug("Using default transmission parameters.")
                 transmission = Transmission([2.750, 2.000, 1.667, 1.444, 1.304, 1.208])

             # Load Final Drive (similar logic)
             fd_path = resolve_path(fd_config_ref)
             if fd_path:
                  with open(fd_path, 'r') as f: fd_params = yaml.safe_load(f).get('final_drive', {})
                  final_drive = FinalDrive(**fd_params)
             elif isinstance(fd_config_ref, dict):
                  final_drive = FinalDrive(**fd_config_ref)
             if final_drive is None:
                 logger.debug("Using default final drive parameters.")
                 final_drive = FinalDrive(14, 53)

             # Load Differential (similar logic)
             diff_path = resolve_path(diff_config_ref)
             if diff_path:
                 with open(diff_path, 'r') as f: diff_params = yaml.safe_load(f).get('differential', {})
                 differential = Differential(**diff_params)
             elif isinstance(diff_config_ref, dict):
                 differential = Differential(**diff_config_ref)
             if differential is None:
                 logger.debug("Using default differential parameters (LOCKED).")
                 differential = Differential(diff_type="LOCKED")

             # Use drivetrain config path if available for inertia etc.
             dt_config_path = resolve_path(dt_config_path_ref)

             self.drivetrain = DrivetrainSystem(
                 transmission, final_drive, differential,
                 wheel_radius_m=self.tire_radius_m, # Use vehicle's radius
                 config_path=dt_config_path # Pass specific path if exists
             )
             logger.info("Drivetrain initialized from config/defaults.")

        if not hasattr(self.drivetrain, 'num_gears'): # Ensure attribute exists
             self.drivetrain.num_gears = len(getattr(self.drivetrain.transmission, 'gear_ratios', []))

    def _initialize_cooling_system(self, cooling_instance: Optional[ExternalCoolingSystem], vehicle_ref):
        """Initialize the main external cooling system using self.config."""
        if isinstance(cooling_instance, ExternalCoolingSystem):
            self.cooling_system = cooling_instance
            logger.info("Using pre-configured external CoolingSystem instance.")
        else:
            cooling_config_path_ref = self.config.get('cooling_system_config_path')
            cooling_config_inline = self.config.get('cooling_system')
            # Determine config directory (relative to main vehicle config if possible)
            base_cfg_dir = os.path.dirname(self.config_path) if self.config_path else '.'
            config_path = None
            if cooling_config_path_ref:
                resolved_path = os.path.join(base_cfg_dir, cooling_config_path_ref) if not os.path.isabs(cooling_config_path_ref) else cooling_config_path_ref
                if os.path.exists(resolved_path):
                    config_path = resolved_path
                else:
                    logger.warning(f"Cooling system config path not found: {resolved_path}")

            config_dir = os.path.dirname(config_path) if config_path else os.path.normpath(os.path.join(base_cfg_dir, '..', 'configs', 'thermal')) # Default relative location

            if config_path:
                logger.info(f"Initializing external CoolingSystem from config file: {config_path}")
                # Use factory with the specific config directory containing the file
                self.cooling_system = create_formula_student_cooling_system(config_dir=config_dir)
            elif isinstance(cooling_config_inline, dict):
                 logger.info("Initializing external CoolingSystem from inline config.")
                 # Manually create components from dict - requires component classes
                 try:
                      from ..thermal.cooling_system import Radiator, WaterPump, CoolingFan, Thermostat # Local import
                      rad_cfg = cooling_config_inline.get('radiator', {})
                      pump_cfg = cooling_config_inline.get('water_pump', {})
                      fan_cfg = cooling_config_inline.get('cooling_fan', {})
                      thermo_cfg = cooling_config_inline.get('thermostat', {})
                      system_cfg = cooling_config_inline.get('system', {})

                      radiator = Radiator(**rad_cfg) if rad_cfg else Radiator()
                      pump = WaterPump(**pump_cfg) if pump_cfg else WaterPump()
                      fan = CoolingFan(**fan_cfg) if fan_cfg else None # Fan optional
                      thermostat = Thermostat(**thermo_cfg) if thermo_cfg else Thermostat()

                      self.cooling_system = ExternalCoolingSystem(radiator, pump, fan, thermostat, **system_cfg)
                 except Exception as e:
                      logger.error(f"Failed to create cooling system from inline config: {e}. Creating default.")
                      self.cooling_system = create_formula_student_cooling_system(config_dir=config_dir) # Pass default dir
            else:
                logger.info("No cooling system config found. Creating default FS cooling system.")
                self.cooling_system = create_formula_student_cooling_system(config_dir=config_dir) # Pass default dir

        # Ensure essential cooling system attributes exist
        if not hasattr(self.cooling_system, 'coolant_temp_C'): self.cooling_system.coolant_temp_C = 25.0
        if not hasattr(self.cooling_system, 'total_thermal_capacity_J_K'): # Ensure capacity is set
            self.cooling_system.total_coolant_mass_kg = self.cooling_system.coolant_volume_L * self.cooling_system.coolant_density_kg_L
            self.cooling_system.total_thermal_capacity_J_K = self.cooling_system.total_coolant_mass_kg * self.cooling_system.coolant_specific_heat_J_kgK
            if self.cooling_system.total_thermal_capacity_J_K <= 0: self.cooling_system.total_thermal_capacity_J_K = 1e-3

    def _initialize_shifting_systems(self, manager_instance: Optional[StrategyManager], cas_instance: Optional[CASSystem]):
        """Initialize shift manager and CAS system using self.config."""
        if isinstance(manager_instance, StrategyManager):
            self.shift_manager = manager_instance
            logger.info("Using pre-configured StrategyManager instance.")
        else:
            if self.engine and self.drivetrain:
                 # Factory needs engine params and drivetrain info
                 self.shift_manager = create_formula_student_strategies(
                     engine_max_rpm=self.engine.redline_rpm,
                     engine_peak_power_rpm=self.engine.max_power_rpm,
                     engine_peak_torque_rpm=self.engine.max_torque_rpm,
                     gear_ratios=self.drivetrain.transmission.gear_ratios,
                     num_gears=self.drivetrain.num_gears,
                     idle_rpm=self.engine.idle_rpm
                 )
                 # Load strategy config from path specified in main vehicle config
                 strat_cfg_path_ref = self.config.get('shift_strategy_config_path')
                 base_cfg_dir = os.path.dirname(self.config_path) if self.config_path else '.'
                 if strat_cfg_path_ref:
                      strat_cfg_path = os.path.join(base_cfg_dir, strat_cfg_path_ref) if not os.path.isabs(strat_cfg_path_ref) else strat_cfg_path_ref
                      if os.path.exists(strat_cfg_path):
                          self.shift_manager.load_strategies_from_config(strat_cfg_path)
                      else:
                           logger.warning(f"Shift strategy config path not found: {strat_cfg_path}")
                 logger.info("Initialized StrategyManager with FS strategies (potentially customized).")
            else:
                 logger.warning("Cannot initialize StrategyManager: Engine or Drivetrain missing.")
                 self.shift_manager = None

        if isinstance(cas_instance, CASSystem):
            self.cas_system = cas_instance
            logger.info("Using pre-configured CASSystem instance.")
        elif self.drivetrain and self.engine:
            # Load CAS config if path specified in main vehicle config
            cas_cfg_path_ref = self.config.get('cas_system_config_path')
            cas_cfg_path = None
            base_cfg_dir = os.path.dirname(self.config_path) if self.config_path else '.'
            if cas_cfg_path_ref:
                 resolved_path = os.path.join(base_cfg_dir, cas_cfg_path_ref) if not os.path.isabs(cas_cfg_path_ref) else cas_cfg_path_ref
                 if os.path.exists(resolved_path):
                      cas_cfg_path = resolved_path
                 else:
                      logger.warning(f"CAS config path specified but not found: {resolved_path}")

            self.cas_system = CASSystem(
                gear_ratios=self.drivetrain.transmission.gear_ratios,
                engine=self.engine,
                config_path=cas_cfg_path # Pass path, constructor handles loading
            )
            logger.info("Initialized CASSystem.")
        else:
             logger.warning("Cannot initialize CASSystem: Engine or Drivetrain missing.")
             self.cas_system = None
             
    def _initialize_aero_cooling(self, side_pods_instance, rear_rad_instance, assist_instance):
        """Initialize optional side pods, rear radiator, cooling assist using self.config."""
        base_cfg_dir = os.path.dirname(self.config_path) if self.config_path else '.'

        def resolve_path(ref_key):
            ref = self.config.get(ref_key)
            if isinstance(ref, str):
                path = os.path.join(base_cfg_dir, ref) if not os.path.isabs(ref) else ref
                return path if os.path.exists(path) else None
            return None

        # Side Pods
        if isinstance(side_pods_instance, DualSidePodSystem):
            self.side_pods = side_pods_instance
        elif self.config.get('side_pods') or self.config.get('side_pod_config_path'):
             config_path = resolve_path('side_pod_config_path')
             config_dir = os.path.dirname(config_path) if config_path else os.path.normpath(os.path.join(base_cfg_dir, '..', 'configs', 'thermal'))
             self.side_pods = create_standard_side_pod_system(config_dir=config_dir) # Use factory
             logger.info("Initialized DualSidePodSystem.")
        else: self.side_pods = None

        # Rear Radiator
        if isinstance(rear_rad_instance, RearRadiatorSystem):
            self.rear_radiator = rear_rad_instance
        elif self.config.get('rear_radiator') or self.config.get('rear_radiator_config_path'):
             config_path = resolve_path('rear_radiator_config_path')
             config_dir = os.path.dirname(config_path) if config_path else os.path.normpath(os.path.join(base_cfg_dir, '..', 'configs', 'thermal'))
             self.rear_radiator = create_default_rear_radiator_system(config_dir=config_dir)
             logger.info("Initialized RearRadiatorSystem.")
        else: self.rear_radiator = None

        # Cooling Assist
        if isinstance(assist_instance, CoolingAssistSystem):
             self.cooling_assist = assist_instance
        elif self.config.get('cooling_assist') or self.config.get('cooling_assist_config_path'):
             config_path = resolve_path('cooling_assist_config_path')
             config_dir = os.path.dirname(config_path) if config_path else os.path.normpath(os.path.join(base_cfg_dir, '..', 'configs', 'thermal'))
             self.cooling_assist = create_default_cooling_assist_system(config_dir=config_dir)
             logger.info("Initialized CoolingAssistSystem.")
        else: self.cooling_assist = None

    def update_engine_state(self):
        """Update engine RPM and calculate torque based on current vehicle state."""
        if not self.engine or not self.drivetrain: return

        # Calculate engine RPM from vehicle speed and current gear
        if self.current_gear > 0:
             self.current_engine_rpm = self.drivetrain.calculate_engine_speed_rpm(
                 self.current_speed_mps, self.current_gear
             )
             # Clamp RPM
             self.current_engine_rpm = np.clip(self.current_engine_rpm, self.engine.idle_rpm, self.engine.redline_rpm)
        else: # Neutral
            # Allow RPM to decay towards idle (simplified)
            idle = self.engine.idle_rpm
            decay_rate = 2000.0 # RPM per second decay rate
            self.current_engine_rpm = max(idle, self.current_engine_rpm - decay_rate * (time.time() - self.last_update_time if self.last_update_time else 0.01))


        # Update engine's internal state (needed for temp factor in get_torque)
        self.engine.current_rpm = self.current_engine_rpm
        self.engine.throttle_position = self.throttle_input

        # Calculate engine torque based on current RPM, throttle, and *engine's* temperature
        engine_torque_nm = self.engine.get_torque(
            rpm=self.current_engine_rpm,
            throttle=self.throttle_input,
            engine_temp=self.engine.engine_temperature # Use engine's internal temp
        )

    def update_drivetrain_state(self):
        """Update drivetrain based on requested gear and engine torque."""
        if not self.drivetrain or not self.engine: return

        # Gear changing is handled externally by simulator/shift manager potentially calling self.change_gear()
        # Here, we just calculate wheel torque based on current gear and engine torque
        # Engine torque should be calculated first in the update cycle
        engine_torque_nm = self.engine.get_torque(self.current_engine_rpm, self.throttle_input, self.engine.engine_temperature)

        # Calculate total wheel torque
        # In a more complex model, this might involve wheel slip and differential logic
        total_wheel_torque_nm = self.drivetrain.calculate_total_wheel_torque(
            engine_torque_nm=engine_torque_nm,
            gear=self.current_gear
        )
        # Store for force calculation
        self._current_total_wheel_torque = total_wheel_torque_nm

    def update_thermal_state(self, dt: float, ambient_temp_C: Optional[float] = None):
        """
        Update the thermal state of the engine and cooling system.
        Relies on EngineHeatModel for heat generation and ExternalCoolingSystem for rejection.
        """
        if not self.include_thermal or not self.engine or not self.cooling_system or not hasattr(self.engine, 'heat_model') or self.engine.heat_model is None:
            self.thermal_factor = 1.0 # Ensure no thermal penalty if not simulating
            return

        ambient_temp = ambient_temp_C if ambient_temp_C is not None else 25.0

        # 1. Calculate Engine Heat Generation (using engine's heat model)
        heat_gen = self.engine.heat_model.calculate_heat_generation(self.current_engine_rpm, self.throttle_input)
        heat_to_coolant_W = heat_gen.get('to_coolant', 0.0)
        heat_to_oil_W = heat_gen.get('to_oil', 0.0)
        heat_block_ambient_gen = heat_gen.get('to_ambient', 0.0) # Heat generated that goes directly to ambient

        # 2. Update External Cooling System State (simulates radiator, fan, pump)
        # Provide current engine block temperature for heat transfer calculation within cooling system if needed
        self.cooling_system.simulate_step(
            ambient_temp_C=ambient_temp,
            vehicle_speed_mps=self.current_speed_mps,
            engine_temp=self.engine_temperature, # Pass current engine block temp
            engine_rpm=self.current_engine_rpm,
            engine_load=self.throttle_input, # Use throttle as proxy
            engine_heat_input_W=heat_to_coolant_W, # Heat transferred FROM engine TO coolant
            dt=dt
        )

        # Get heat rejected by the radiator from the cooling system's state
        heat_rejected_W = self.cooling_system.radiator_heat_rejection_W
        # Get the updated coolant temperature from the cooling system
        self.coolant_temperature = self.cooling_system.coolant_temp_C

        # 3. Update Engine/Oil Temperatures using heat flows and capacities
        thermal_cfg = self.engine.heat_model.config
        capacities = thermal_cfg.get_thermal_capacities()
        # Ensure capacities dict is valid or use safe defaults
        safe_caps = {'engine_block': 50000, 'engine_oil': 10000, 'coolant_engine': 8000}
        safe_caps.update(capacities) # Update with actual values if they exist

        # Calculate heat transfer rates between components
        # Use the current temperatures *before* updating them
        temps_current = {'engine': self.engine_temperature, 'oil': self.oil_temperature, 'coolant': self.coolant_temperature}
        internal_transfer = self.engine.heat_model.calculate_internal_heat_transfer(temps_current)
        ambient_loss = self.engine.heat_model.calculate_ambient_heat_loss(temps_current, ambient_temp, self.current_speed_mps)

        q_block_to_coolant = internal_transfer['coolant_to_block'] # Heat from block TO coolant
        q_block_to_oil = internal_transfer['oil_to_block']         # Heat from block TO oil
        q_block_to_ambient_loss = ambient_loss['block_to_ambient'] # Heat from block TO ambient air
        q_oil_to_ambient_loss = ambient_loss['oil_to_ambient']     # Heat from oil TO ambient air

        # Net heat flow for engine block: heat generated directly to ambient - transfer to coolant - transfer to oil - direct loss to air
        q_net_engine = heat_block_ambient_gen - q_block_to_coolant - q_block_to_oil - q_block_to_ambient_loss

        # Net heat flow for oil: heat generated to oil + heat from block - direct loss to air
        q_net_oil = heat_to_oil_W + q_block_to_oil - q_oil_to_ambient_loss

        # Coolant temp is managed by the external system, but we track engine block and oil temps here
        self.engine_temperature += (q_net_engine * dt) / max(1e-3, safe_caps['engine_block'])
        self.oil_temperature += (q_net_oil * dt) / max(1e-3, safe_caps['engine_oil'])

        # Clamp temperatures
        self.engine_temperature = max(ambient_temp - 5, self.engine_temperature)
        self.oil_temperature = max(ambient_temp - 5, self.oil_temperature)

        # Update engine's internal temperature estimates and thermal factor
        if hasattr(self.engine, 'engine_temperature'): self.engine.engine_temperature = self.engine_temperature
        if hasattr(self.engine, 'coolant_temperature'): self.engine.coolant_temperature = self.coolant_temperature # Keep engine's view synced
        if hasattr(self.engine, 'oil_temperature'): self.engine.oil_temperature = self.oil_temperature
        if hasattr(self.engine, '_get_thermal_performance_factor'):
             current_thermal_factor = self.engine._get_thermal_performance_factor(self.engine_temperature)
             self.thermal_factor = current_thermal_factor
             if hasattr(self.engine, 'thermal_factor'): self.engine.thermal_factor = current_thermal_factor # Keep engine's factor synced

    def change_gear(self, target_gear: int) -> Tuple[bool, float]:
        """
        Request a gear change. Returns success and estimated shift time.

        Args:
            target_gear: The desired gear number (0 for Neutral).

        Returns:
            Tuple (success: bool, shift_duration_s: float).
            Shift duration is 0 if no shift occurs or direct change.
        """
        shift_duration_s = 0.0
        success = False

        if self.drivetrain is None:
            logger.error("Cannot change gear: No Drivetrain system.")
            return False, 0.0

        if target_gear == self.current_gear:
             return True, 0.0 # No change needed

        # Determine shift direction
        direction = ShiftDirection.NEUTRAL
        if target_gear > self.current_gear: direction = ShiftDirection.UP
        elif target_gear < self.current_gear: direction = ShiftDirection.DOWN

        if self.cas_system:
            # CAS handles readiness checks and overrev protection internally now
            # Use monotonic time for shift initiation check
            current_time_s = time.monotonic()
            # Pass current time to CAS for readiness check
            if self.cas_system._check_shift_readiness(current_time_s * 1000.0): # CAS uses ms
                success = self.cas_system.request_shift(direction, target_gear_override=target_gear)
                if success:
                    # Shift *initiated*. Gear change happens later via event or state check.
                    # Get estimated time for the *calling simulator* to handle.
                    shift_duration_s = self.cas_system.get_total_shift_time_ms(direction) / 1000.0
                    # DO NOT update self.current_gear here. It's updated when the shift completes.
                    logger.debug(f"CAS shift {self.current_gear}->{target_gear} initiated. Estimated duration: {shift_duration_s*1000:.1f} ms.")
                else:
                    logger.debug(f"CAS shift request {self.current_gear}->{target_gear} rejected by internal CAS logic (e.g., overrev).")
            else:
                 logger.debug(f"CAS shift request {self.current_gear}->{target_gear} rejected by readiness check (busy/cooldown).")
                 success = False # Explicitly false if readiness check fails

        else:
            # Direct transmission change if no CAS
            if 0 <= target_gear <= self.drivetrain.num_gears:
                 success = self.drivetrain.change_gear(target_gear)
                 if success:
                     self.current_gear = target_gear # Direct change, update immediately
                     shift_duration_s = 0.050 # Default direct shift time penalty if no CAS
                     logger.debug(f"Direct shift to gear {target_gear} successful.")
                 else:
                     logger.warning(f"Direct gear change to {target_gear} failed.")
            else:
                 logger.error(f"Invalid target gear {target_gear} for direct change.")
                 success = False

        return success, shift_duration_s
    def calculate_forces(self) -> Dict[str, float]:
        """Calculate major longitudinal forces acting on the vehicle."""
        # 1. Tractive Force (from wheel torque calculated in drivetrain update)
        F_tractive = getattr(self, '_current_total_wheel_torque', 0.0) / self.tire_radius_m if self.tire_radius_m > 0 else 0.0

        # 2. Aerodynamic Drag
        F_drag = 0.5 * AIR_DENSITY_SEA_LEVEL * self.drag_coefficient * self.frontal_area_m2 * self.current_speed_mps**2

        # 3. Aerodynamic Downforce (negative lift)
        F_downforce = -0.5 * AIR_DENSITY_SEA_LEVEL * self.lift_coefficient * self.frontal_area_m2 * self.current_speed_mps**2

        # 4. Rolling Resistance (increases with downforce)
        normal_load = self.mass * GRAVITY + F_downforce
        F_rolling = self.rolling_resistance_coeff * max(0, normal_load) # Ensure normal load isn't negative

        # 5. Braking Force
        # Max braking force is limited by friction (mu * Normal Load) and brake system capability
        # Simplified: Use a max G limit affected by downforce
        max_brake_force = (self.mass * GRAVITY + F_downforce) * self.max_braking_g # Use max_braking_g defined earlier
        F_brake = self.brake_input * max_brake_force

        return {
            'tractive': F_tractive,
            'drag': F_drag,
            'rolling': F_rolling,
            'brake': F_brake,
            'downforce': F_downforce
        }

    def calculate_acceleration(self, throttle: Optional[float] = None, brake: Optional[float] = None) -> float:
        """
        Calculate current longitudinal acceleration based on forces.
        Updates internal throttle/brake inputs if provided.

        Returns:
            Longitudinal acceleration (m/s²).
        """
        # Update inputs if provided
        if throttle is not None: self.throttle_input = np.clip(throttle, 0.0, 1.0)
        if brake is not None: self.brake_input = np.clip(brake, 0.0, 1.0)

        # Ensure engine and drivetrain states are updated based on current speed/gear/inputs
        self.update_engine_state() # Calculates engine torque based on current RPM/throttle/temp
        self.update_drivetrain_state() # Calculates wheel torque based on engine torque/gear

        # Calculate forces based on the *updated* state
        forces = self.calculate_forces()

        # Net Force = Tractive - Drag - Rolling - Brake
        net_force = forces['tractive'] - forces['drag'] - forces['rolling'] - forces['brake']

        # Acceleration = Net Force / Mass
        self.current_acceleration_mpss = net_force / self.mass
        return self.current_acceleration_mpss


    def update_vehicle_state(self, dt: float, ambient_temp_C: float = 25.0):
        """
        Update vehicle kinematics and thermal state over a time step.
        This is the core physics update step.

        Args:
            dt: Time step (seconds).
            ambient_temp_C: Ambient temperature (°C).
        """
        if dt <= 0: return # Nothing to update

        start_time = time.monotonic()
        self.last_update_time = start_time # Store time for internal calculations

        # 1. Calculate current acceleration based on existing state and inputs
        # Note: calculate_acceleration also calls engine/drivetrain updates internally
        current_accel = self.calculate_acceleration()

        # 2. Update Kinematics (Speed and Position) - Simple Euler integration
        self.current_speed_mps += current_accel * dt
        self.current_speed_mps = max(0.0, self.current_speed_mps) # Prevent negative speed
        self.current_position_m += self.current_speed_mps * dt

        # 3. Update Engine RPM (based on new speed) - Handled within update_engine_state called by calc_accel
        # Re-call update_engine_state to ensure RPM is consistent with the *new* speed for the *next* step's torque calc
        self.update_engine_state()

        # 4. Update Thermal State (Engine internal and External system)
        if self.include_thermal: # Assuming include_thermal is a class attribute
             self.update_thermal_state(dt, ambient_temp_C)

        # 5. Update Shift System (e.g., CAS cooldown timer) if applicable
        if self.cas_system:
             self.cas_system.update(dt)

        update_duration = time.monotonic() - start_time
        # logger.debug(f"Vehicle state updated in {update_duration*1000:.2f} ms. Speed: {self.current_speed_mps:.2f} m/s")


    # --- Simulation Wrappers ---
    # These methods might call the specialized simulators from performance package
    # or implement simplified versions directly using the vehicle's step updates.

    def simulate_acceleration_run(self, distance: float = FS_ACCELERATION_LENGTH, # Now defined
                                max_time: float = 10.0, dt: float = 0.01,
                                use_launch_control: bool = True,
                                use_optimized_shifts: bool = True) -> Dict:
        """Simulate a standard acceleration run using AccelerationSimulator."""
        logger.info("Running simulate_acceleration_run within Vehicle class...")
        # Check if the specialized simulator class is available
        if AccelerationSimulator is None:
            logger.error("AccelerationSimulator class not available. Cannot simulate.")
            return {'error': 'AccelerationSimulator not available'}

        # Use the dedicated AccelerationSimulator for consistency
        try:
            # Pass a deep copy of the *current* vehicle state to the simulator
            vehicle_copy = copy.deepcopy(self)
            accel_sim = AccelerationSimulator(vehicle_copy)
            accel_sim.configure(distance_m=distance, time_step_s=dt, max_time_s=max_time)
            accel_sim.configure_launch_control(use_traction_control=True) # Use default LC params initially
            accel_sim.configure_shifting(use_optimized=use_optimized_shifts)
            results = accel_sim.simulate_acceleration(use_launch_control=use_launch_control)
            # Analyze results using the simulator's method
            metrics = accel_sim.analyze_performance_metrics(results)
            results.update(metrics) # Add metrics to the results dict
            return results
        except Exception as e:
             logger.error(f"Error during AccelerationSimulation: {e}", exc_info=True)
             return {'error': str(e)}

    def simulate_skidpad(self, circle_radius_m: float = FS_SKIDPAD_RADIUS,
                       target_gear: int = 2, max_laps: int = 4, dt: float = 0.01) -> Dict:
        """Simulate a skidpad event (constant radius cornering)."""
        logger.info("Running simulate_skidpad within Vehicle class...")
        # --- Reset ---
        self.current_speed_mps = 0.0
        self.change_gear(target_gear)
        lap_angle = 0.0
        lap_times = []
        t = 0.0
        max_time = max_laps * 10.0 # Estimate max time
        history = {'time': [], 'speed': [], 'lat_g': []}

        # --- Simulation Loop ---
        while t < max_time and len(lap_times) < max_laps:
            # Calculate max speed for this radius
            max_corner_speed = self.cornering.calculate_max_cornering_speed(circle_radius_m)

            # Control logic: try to maintain max speed
            throttle = 0.0; brake = 0.0
            if self.current_speed_mps < max_corner_speed * 0.98:
                throttle = 0.8
            elif self.current_speed_mps > max_corner_speed * 1.02:
                brake = 0.2
            else: # Maintain speed (balance drag/rolling resistance)
                 throttle = 0.4 # Needs tuning

            # Calculate acceleration (longitudinal only for this simple model)
            self.calculate_acceleration(throttle=throttle, brake=brake)

            # Update speed
            self.current_speed_mps += self.current_acceleration_mpss * dt
            self.current_speed_mps = max(0.0, self.current_speed_mps)

            # Update angle turned
            angular_vel = self.current_speed_mps / circle_radius_m if circle_radius_m > 0 else 0
            lap_angle += angular_vel * dt
            t += dt

            # Store history
            history['time'].append(t)
            history['speed'].append(self.current_speed_mps)
            history['lat_g'].append((self.current_speed_mps**2 / circle_radius_m) / GRAVITY if circle_radius_m > 0 else 0)

            # Check for lap completion
            if lap_angle >= 2 * np.pi:
                 lap_time = t - sum(lap_times) # Time for this lap
                 lap_times.append(lap_time)
                 lap_angle -= 2 * np.pi # Reset angle for next lap
                 logger.debug(f"Skidpad lap {len(lap_times)} completed in {lap_time:.3f}s")

        # --- Results ---
        avg_lap_time = np.mean(lap_times) if lap_times else None
        max_lat_g = np.max(history['lat_g']) if history['lat_g'] else None

        results = {
            'average_lap_time': avg_lap_time,
            'lap_times': lap_times,
            'max_lateral_g': max_lat_g,
            'history': history
        }
        logger.info(f"Skidpad simulation finished. Avg Lap: {avg_lap_time:.3f}s, Max Lateral G: {max_lat_g:.3f}g")
        return results


    def simulate_lap(self, track_file: str, include_thermal: bool = True) -> Dict:
         """Simulate a single lap using the LapTimeSimulator."""
         logger.info("Running simulate_lap via LapTimeSimulator...")
         # Check if the specialized simulator class is available
         if LapTimeSimulator is None:
             logger.error("LapTimeSimulator class not available. Cannot simulate.")
             return {'error': 'LapTimeSimulator not available'}

         # Use the dedicated LapTimeSimulator for consistency
         try:
             # Pass a deep copy of the *current* vehicle state
             vehicle_copy = copy.deepcopy(self)
             lap_sim = LapTimeSimulator(vehicle_copy, track_file=track_file)
             results = lap_sim.simulate_lap(include_thermal=include_thermal)
             # Add metrics for convenience
             metrics = lap_sim.analyze_lap_performance(results)
             results['metrics'] = metrics
             return results
         except Exception as e:
              logger.error(f"Error during LapTimeSimulation: {e}", exc_info=True)
              return {'error': str(e)}
          
    # --- Analysis and Helper Methods ---

    def calculate_weight_transfer(self, longitudinal_accel_mpss: float = 0.0, lateral_accel_mpss: float = 0.0) -> Dict:
        """Calculate longitudinal and lateral weight transfer."""
        # Longitudinal Transfer (positive accel = transfer to rear)
        long_transfer = (self.mass * longitudinal_accel_mpss * self.cg_height_m) / self.wheelbase_m

        # Lateral Transfer (positive accel = transfer to outside, assume right turn)
        lat_transfer = (self.mass * abs(lateral_accel_mpss) * self.cg_height_m) / self.track_width_rear_m # Use rear track width

        # Static weights per axle/side
        static_front_N = self.mass * GRAVITY * self.weight_distribution_front
        static_rear_N = self.mass * GRAVITY * (1.0 - self.weight_distribution_front)
        static_left_N = self.mass * GRAVITY * 0.5
        static_right_N = self.mass * GRAVITY * 0.5

        # Dynamic weights per axle
        dynamic_front_N = static_front_N - long_transfer
        dynamic_rear_N = static_rear_N + long_transfer

        # Dynamic weights per wheel (simplified, assumes equal distribution per axle)
        # Positive lateral accel assumed right turn (more weight on left)
        dynamic_FL_N = dynamic_front_N / 2.0 + lat_transfer / 2.0 # Front axle portion of lat transfer
        dynamic_FR_N = dynamic_front_N / 2.0 - lat_transfer / 2.0
        dynamic_RL_N = dynamic_rear_N / 2.0 + lat_transfer / 2.0 # Rear axle portion
        dynamic_RR_N = dynamic_rear_N / 2.0 - lat_transfer / 2.0

        return {
            'longitudinal_transfer_N': long_transfer,
            'lateral_transfer_N': lat_transfer,
            'dynamic_front_axle_load_N': dynamic_front_N,
            'dynamic_rear_axle_load_N': dynamic_rear_N,
            'dynamic_FL_wheel_load_N': max(0, dynamic_FL_N), # Wheels cannot have negative load
            'dynamic_FR_wheel_load_N': max(0, dynamic_FR_N),
            'dynamic_RL_wheel_load_N': max(0, dynamic_RL_N),
            'dynamic_RR_wheel_load_N': max(0, dynamic_RR_N),
        }

    def get_vehicle_specs(self) -> Dict:
        """Return a comprehensive dictionary of vehicle specifications."""
        specs = {
            'team_name': self.team_name,
            'vehicle': {
                'mass': self.mass,
                'frontal_area_m2': self.frontal_area_m2,
                'drag_coefficient': self.drag_coefficient,
                'lift_coefficient': self.lift_coefficient,
                'rolling_resistance_coeff': self.rolling_resistance_coeff,
                'weight_distribution_front': self.weight_distribution_front,
                'wheelbase_m': self.wheelbase_m,
                'track_width_front_m': self.track_width_front_m,
                'track_width_rear_m': self.track_width_rear_m,
                'cg_height_m': self.cg_height_m
            },
            'tires': {
                'radius_m': self.tire_radius_m,
                # Add more tire specs if available in model
            }
        }
        if self.engine: specs['engine'] = self.engine.get_engine_specs()
        if self.drivetrain: specs['drivetrain'] = self.drivetrain.get_drivetrain_specs()
        if self.cooling_system: specs['cooling'] = self.cooling_system.get_system_specs()
        if self.side_pods: specs['side_pods'] = self.side_pods.get_system_specs()
        if self.rear_radiator: specs['rear_radiator'] = self.rear_radiator.get_system_specs()
        if self.cooling_assist: specs['cooling_assist'] = self.cooling_assist.get_system_specs()
        # Add CAS info
        if self.cas_system: specs['cas'] = self.cas_system.get_status()

        return specs

    def calculate_performance_metrics(self) -> Dict:
        """Calculate key theoretical performance metrics."""
        metrics = {}
        # Power-to-weight
        if self.engine and self.mass > 0:
             power_kw = self.engine.max_power_hp * HP_TO_KW
             metrics['power_to_weight_kw_kg'] = power_kw / self.mass
             metrics['power_to_weight_hp_kg'] = self.engine.max_power_hp / self.mass

        # Theoretical Max Speed (drag limited)
        try:
             max_speed = self.calculate_max_speed()
             metrics['max_speed_mps'] = max_speed
             metrics['max_speed_kph'] = max_speed * MS_TO_KMH
        except Exception as e:
             logger.warning(f"Could not calculate max speed: {e}")
             metrics['max_speed_mps'] = None

        # Theoretical Max Lateral G (using cornering calculator)
        try:
             # Calculate at a reference speed (e.g., 20 m/s)
             max_lat_g = self.cornering.calculate_max_lateral_acceleration(speed_mps=20.0) / GRAVITY
             metrics['max_lateral_g'] = max_lat_g
        except Exception as e:
             logger.warning(f"Could not calculate max lateral G: {e}")
             metrics['max_lateral_g'] = None

        return metrics

    # --- Plotting Wrappers ---
    def plot_acceleration_results(self, results: Dict, save_path: Optional[str] = None):
         """Plot acceleration results using the utility function."""
         fig = plot_accel_results_util(results, save_path=save_path, plot_wheel_slip=True)
         # if fig: plt.close(fig)

    # plot_lap_results, plot_skidpad_results, plot_thermal_analysis etc. would be similar wrappers
    # calling the respective functions from utils.plotting or performance modules.


# --- Factory Function ---
def create_formula_student_vehicle(config_path: Optional[str] = None) -> Vehicle:
    """
    Factory function to create a Vehicle instance with typical FS components.
    Loads configurations if available, otherwise uses defaults.

    Args:
        config_path: Optional path to a main vehicle config file.

    Returns:
        A configured Vehicle instance.
    """
    logger.info("Creating Formula Student Vehicle...")
    # Pass the config path to the Vehicle constructor, which handles loading
    # and initializing components based on the config or defaults.
    vehicle = Vehicle(config_path=config_path)
    logger.info("Formula Student Vehicle created successfully.")
    return vehicle


# Example Usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Vehicle Module Demo")
    print("-" * 20)

    # Create vehicle (will try to load default configs if available)
    # Ensure default config files exist in ../configs/ relative to this file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    # We don't pass a specific vehicle config, so it relies on component defaults/configs
    vehicle = create_formula_student_vehicle()

    print("\n--- Vehicle Specs ---")
    specs = vehicle.get_vehicle_specs()
    # Print selected specs
    print(f" Mass: {specs['vehicle']['mass']:.1f} kg")
    print(f" Engine: {specs['engine']['make']} {specs['engine']['model']}")
    print(f" Max Power: {specs['engine']['max_power_hp']:.1f} HP @ {specs['engine']['max_power_rpm']:.0f} RPM")
    print(f" Gears: {specs['drivetrain']['num_gears']}")
    print(f" Final Drive: {specs['drivetrain']['final_drive_ratio']:.3f}")

    print("\n--- Performance Metrics ---")
    metrics = vehicle.calculate_performance_metrics()
    print(f" Power/Weight: {metrics.get('power_to_weight_kw_kg', 0):.3f} kW/kg")
    print(f" Max Speed: {metrics.get('max_speed_kph', 0):.1f} km/h")
    print(f" Max Lateral G: {metrics.get('max_lateral_g', 0):.2f} g")

    # --- Example Simulation Step ---
    print("\n--- Simulating 1 Step ---")
    vehicle.throttle_input = 0.8
    vehicle.brake_input = 0.0
    vehicle.change_gear(1)
    vehicle.update_vehicle_state(dt=0.1, ambient_temp_C=28.0)
    print(f" Speed after 0.1s: {vehicle.current_speed_mps * MS_TO_KMH:.1f} km/h")
    print(f" Accel: {vehicle.current_acceleration_mpss:.2f} m/s^2")
    print(f" Engine Temp: {vehicle.engine_temperature:.1f} C")
    print(f" Coolant Temp: {vehicle.coolant_temperature:.1f} C")

    # --- Example Acceleration Run ---
    print("\n--- Simulating Acceleration Run ---")
    accel_results = vehicle.simulate_acceleration_run()
    if accel_results.get('finish_time'):
        print(f" 75m Time: {accel_results['finish_time']:.3f} s")
        print(f" 0-60 mph: {accel_results.get('time_to_60mph', -1):.3f} s")
        # Plotting (optional)
        # plot_dir = os.path.join(project_root, "plots", "vehicle_demo")
        # os.makedirs(plot_dir, exist_ok=True)
        # vehicle.plot_acceleration_results(accel_results, save_path=os.path.join(plot_dir, "demo_accel_run.png"))

    print("\nVehicle demo finished.")