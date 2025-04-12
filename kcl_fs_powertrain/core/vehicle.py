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
import time # <-- Import time for monotonic time

# --- Import constants FIRST ---
try:
    from ..utils.constants import (
        GRAVITY, AIR_DENSITY_SEA_LEVEL, KW_TO_HP, HP_TO_KW, MS_TO_KMH, MS_TO_MPH,
        FS_ACCELERATION_LENGTH, FS_SKIDPAD_RADIUS, LITERS_TO_M3, BAR_TO_PA # <-- Added BAR_TO_PA, LITERS_TO_M3
    )
except ImportError:
    GRAVITY = 9.81; AIR_DENSITY_SEA_LEVEL = 1.225; KW_TO_HP = 1.341; HP_TO_KW = 1/KW_TO_HP; MS_TO_KMH = 3.6; MS_TO_MPH = 2.237
    FS_ACCELERATION_LENGTH = 75.0; FS_SKIDPAD_RADIUS = 15.25 / 2.0; LITERS_TO_M3 = 0.001; BAR_TO_PA = 100000.0

# Import powertrain components (handle potential errors)
try:
    from ..engine.motorcycle_engine import MotorcycleEngine
    from ..engine.engine_thermal import EngineHeatModel, ThermalConfig
    from ..transmission.gearing import DrivetrainSystem, Transmission, FinalDrive, Differential
    from ..transmission.cas_system import CASSystem, ShiftDirection, ShiftState
    from ..transmission.shift_strategy import StrategyManager, create_formula_student_strategies, StrategyType
    from ..thermal.cooling_system import CoolingSystem as ExternalCoolingSystem
    from ..thermal.cooling_system import create_formula_student_cooling_system
    from ..thermal.side_pod import DualSidePodSystem, create_standard_side_pod_system
    from ..thermal.rear_radiator import RearRadiatorSystem, create_default_rear_radiator_system
    from ..thermal.electric_compressor import CoolingAssistSystem, create_default_cooling_assist_system
    from ..utils.plotting import plot_acceleration_results as plot_accel_results_util, save_plot
    # Removed plot_vehicle_performance_summary import - not used here

    # Conditional imports for performance classes
    try:
        from ..performance.lap_time import CorneringPerformance
        CorneringPerformance_available = True
    except ImportError:
        CorneringPerformance = None # Define as None if unavailable
        CorneringPerformance_available = False
        logging.warning("CorneringPerformance class not available.")

    try:
        from ..performance.acceleration import AccelerationSimulator
        AccelerationSimulator_available = True
    except ImportError:
        AccelerationSimulator = None # Define as None if unavailable
        AccelerationSimulator_available = False
        logging.warning("AccelerationSimulator class not available.")

    try:
        from ..performance.lap_time import LapTimeSimulator
        LapTimeSimulator_available = True
    except ImportError:
        LapTimeSimulator = None # Define as None if unavailable
        LapTimeSimulator_available = False
        logging.warning("LapTimeSimulator class not available.")

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
    class ShiftState: IDLE = 0; SHIFT_IN_PROGRESS = 1 # Placeholder Enum states
    class ShiftDirection: UP = 1; DOWN = -1; NEUTRAL = 0 # Placeholder Enum states
    class StrategyManager: pass
    class ExternalCoolingSystem: pass
    class DualSidePodSystem: pass
    class RearRadiatorSystem: pass
    class CoolingAssistSystem: pass
    CorneringPerformance = None; CorneringPerformance_available = False
    AccelerationSimulator = None; AccelerationSimulator_available = False
    LapTimeSimulator = None; LapTimeSimulator_available = False
    def create_formula_student_strategies(*args, **kwargs): return None
    def create_standard_side_pod_system(*args, **kwargs): return None
    def create_default_rear_radiator_system(*args, **kwargs): return None
    def create_default_cooling_assist_system(*args, **kwargs): return None
    def create_formula_student_cooling_system(*args, **kwargs): return None
    def plot_accel_results_util(*args, **kwargs): plt.figure(); plt.plot([0,1],[0,1]); plt.title("Fallback Plot"); plt.show(); plt.close(); return plt.gcf()
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
        self.max_braking_g: float = 1.8 # Max braking deceleration relative to g

        # --- Load configuration ---
        if config is not None:
            self.config = copy.deepcopy(config)
            logger.info("Vehicle initialized using provided configuration dictionary.")
        elif config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    loaded_cfg = yaml.safe_load(f)
                    self.config = loaded_cfg if loaded_cfg else {}
                self.config_path = config_path
                logger.info(f"Vehicle base configuration loaded from {config_path}")
            except Exception as e:
                logger.error(f"Error loading vehicle config from {config_path}: {e}")
                self.config = {}
        else:
            logger.warning(f"Vehicle config not provided or path invalid: {config_path}. Using defaults.")
            self.config = {}

        self._apply_base_config()

        # --- Component Initialization ---
        self._initialize_engine(engine)
        self._initialize_drivetrain(drivetrain)
        self._initialize_cooling_system(cooling_system, self)
        self._initialize_shifting_systems(shift_manager, cas_system)
        self._initialize_aero_cooling(side_pods, rear_radiator, cooling_assist)

        if CorneringPerformance_available:
            self.cornering = CorneringPerformance(self)
        else:
            self.cornering = None

        # --- Current State Variables ---
        self.current_speed_mps: float = 0.0
        self.current_acceleration_mpss: float = 0.0
        self.current_position_m: float = 0.0
        self.current_gear: int = 0 # Neutral
        self.current_engine_rpm: float = getattr(self.engine, 'idle_rpm', 1300.0)
        self.throttle_input: float = 0.0
        self.brake_input: float = 0.0
        self.steering_angle_rad: float = 0.0

        self.last_update_time: float = time.monotonic()
        self.include_thermal: bool = self.config.get('simulation_settings', {}).get('include_thermal', True)

        # Initialize thermal state
        self.coolant_temperature: float = 25.0
        self.engine_temperature: float = 25.0
        self.oil_temperature: float = 25.0
        self.thermal_factor: float = 1.0
        if self.include_thermal:
            if self.cooling_system and hasattr(self.cooling_system, 'coolant_temp_C'):
                self.coolant_temperature = self.cooling_system.coolant_temp_C
            self.engine_temperature = getattr(self.engine, 'engine_temperature', self.coolant_temperature + 5.0)
            self.oil_temperature = getattr(self.engine, 'oil_temperature', self.coolant_temperature)
            self.thermal_factor = getattr(self.engine, 'thermal_factor', 1.0)

        logger.info(f"{self.team_name} Vehicle initialized. Mass: {self.mass:.1f} kg")

    def load_config(self, config_path: str):
        """Load vehicle base parameters from YAML file."""
        if not os.path.exists(config_path):
            logger.error(f"Vehicle configuration file not found: {config_path}")
            return
        try:
            with open(config_path, 'r') as f:
                loaded_cfg = yaml.safe_load(f)
                self.config = loaded_cfg if loaded_cfg else {}

            self._apply_base_config() # Apply the loaded base parameters
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

        sim_cfg = self.config.get('simulation_settings', self.config.get('simulation', {}))
        self.include_thermal = bool(sim_cfg.get('include_thermal', self.include_thermal))
        logger.debug(f"Base parameters applied. Mass={self.mass:.1f}, IncludeThermal={self.include_thermal}")

    def _initialize_engine(self, engine_instance: Optional[MotorcycleEngine]):
        """Initialize the engine component using self.config."""
        if MotorcycleEngine and isinstance(engine_instance, MotorcycleEngine): # Check if MotorcycleEngine class exists
            self.engine = engine_instance
            logger.info("Using pre-configured Engine instance.")
        else:
            engine_config_ref = self.config.get('engine_config_path', self.config.get('engine')) # Allow path or inline dict
            if isinstance(engine_config_ref, str): # Path provided
                if os.path.exists(engine_config_ref):
                    logger.info(f"Initializing Engine from config file: {engine_config_ref}")
                    self.engine = MotorcycleEngine(config_path=engine_config_ref) if MotorcycleEngine else None
                else:
                    logger.warning(f"Engine config path not found: {engine_config_ref}. Trying default.")
                    self.engine = None # Set to None first
            elif isinstance(engine_config_ref, dict): # Inline config
                logger.info("Initializing Engine from inline config in vehicle config.")
                self.engine = MotorcycleEngine(engine_params=engine_config_ref) if MotorcycleEngine else None
            else: # No specific config provided
                 self.engine = None

            # If engine still None, try default path
            if self.engine is None:
                 logger.warning("No valid engine config found. Attempting to create default MotorcycleEngine.")
                 default_engine_path = os.path.join('configs', 'engine', 'cbr600f4i.yaml')
                 if os.path.exists(default_engine_path):
                     logger.info(f"Loading default engine config: {default_engine_path}")
                     self.engine = MotorcycleEngine(config_path=default_engine_path) if MotorcycleEngine else None
                 else:
                     logger.warning("Default engine config not found. Creating default instance.")
                     self.engine = MotorcycleEngine() if MotorcycleEngine else None

        if self.engine is None:
             logger.error("Failed to initialize Engine component.")
             self.engine = type('MockEngine', (object,), {'idle_rpm': 1300.0, 'redline_rpm': 14000.0, 'max_power_rpm': 12500.0, 'max_torque_rpm': 10500.0, 'heat_model': None, 'thermal_factor': 1.0, 'engine_temperature': 25.0, 'coolant_temperature': 25.0, 'oil_temperature': 25.0, 'get_torque': lambda s, r, th, t=None: 0.0})()

        # Ensure essential attributes (use getattr with default)
        self.engine.idle_rpm = getattr(self.engine, 'idle_rpm', 1300.0)
        self.engine.redline_rpm = getattr(self.engine, 'redline_rpm', 14000.0)
        self.engine.max_power_rpm = getattr(self.engine, 'max_power_rpm', 12500.0)
        self.engine.max_torque_rpm = getattr(self.engine, 'max_torque_rpm', 10500.0)
        self.engine.engine_temperature = getattr(self.engine, 'engine_temperature', 25.0)
        self.engine.coolant_temperature = getattr(self.engine, 'coolant_temperature', 25.0)
        self.engine.oil_temperature = getattr(self.engine, 'oil_temperature', 25.0)
        self.engine.thermal_factor = getattr(self.engine, 'thermal_factor', 1.0)

        # Ensure engine has a heat model instance if thermal sim is enabled
        if self.include_thermal and (not hasattr(self.engine, 'heat_model') or self.engine.heat_model is None):
             if EngineHeatModel and ThermalConfig:
                 logger.debug("Creating default heat model for engine.")
                 thermal_cfg = getattr(self.engine, 'thermal_config', ThermalConfig())
                 self.engine.heat_model = EngineHeatModel(thermal_cfg, self.engine)
             else:
                logger.warning("EngineHeatModel/ThermalConfig class not available, cannot create heat model instance.")
                self.engine.heat_model = None

    def _initialize_drivetrain(self, drivetrain_instance: Optional[DrivetrainSystem]):
        """Initialize the drivetrain component using self.config."""
        if DrivetrainSystem and isinstance(drivetrain_instance, DrivetrainSystem):
            self.drivetrain = drivetrain_instance
            logger.info("Using pre-configured Drivetrain instance.")
        else:
            dt_config_path_ref = self.config.get('drivetrain_config_path')
            drivetrain_config = self.config.get('drivetrain', {})
            trans_config_ref = drivetrain_config.get('transmission_config_path', drivetrain_config.get('transmission'))
            fd_config_ref = drivetrain_config.get('final_drive_config_path', drivetrain_config.get('final_drive'))
            diff_config_ref = drivetrain_config.get('differential_config_path', drivetrain_config.get('differential'))

            transmission = None
            final_drive = None
            differential = None

            base_cfg_dir = os.path.dirname(self.config_path) if self.config_path else '.'

            def load_component(comp_class, comp_ref, default_args, class_name_override=None):
                class_name = class_name_override or (comp_class.__name__.lower() if comp_class else "component")
                if not comp_class: return None # Class itself is missing

                if isinstance(comp_ref, str): # Path
                    path = os.path.join(base_cfg_dir, comp_ref) if not os.path.isabs(comp_ref) else comp_ref
                    if os.path.exists(path):
                        try:
                            with open(path, 'r') as f:
                                file_content = yaml.safe_load(f)
                                params = file_content.get(class_name, {}) if file_content else {}
                            return comp_class(**params)
                        except Exception as e: logger.error(f"Error loading {class_name} from {path}: {e}")
                    else: logger.warning(f"{class_name} config not found: {path}")
                elif isinstance(comp_ref, dict): # Inline dict
                    return comp_class(**comp_ref)
                # Fallback to default
                logger.debug(f"Using default {class_name} parameters.")
                return comp_class(**default_args)

            transmission = load_component(Transmission, trans_config_ref, {'gear_ratios': [2.75, 2.0, 1.67, 1.44, 1.3, 1.2]})
            final_drive = load_component(FinalDrive, fd_config_ref, {'drive_sprocket_teeth': 14, 'driven_sprocket_teeth': 53})
            differential = load_component(Differential, diff_config_ref, {'diff_type': "LOCKED"})

            dt_config = {}
            dt_config_path = None
            if dt_config_path_ref and isinstance(dt_config_path_ref, str):
                path = os.path.join(base_cfg_dir, dt_config_path_ref) if not os.path.isabs(dt_config_path_ref) else dt_config_path_ref
                if os.path.exists(path): dt_config_path = path

            if dt_config_path:
                try:
                    with open(dt_config_path, 'r') as f: dt_config = yaml.safe_load(f).get('drivetrain_system', {})
                except Exception as e: logger.error(f"Error loading drivetrain system config: {e}")

            # Check if DrivetrainSystem class is available
            if DrivetrainSystem and transmission and final_drive:
                 self.drivetrain = DrivetrainSystem(
                     transmission, final_drive, differential,
                     wheel_radius_m=self.tire_radius_m,
                     drivetrain_inertia_kgm2=dt_config.get('drivetrain_inertia_kgm2', 0.15)
                 )
                 logger.info("Drivetrain initialized from config/defaults.")
            else:
                 self.drivetrain = None
                 logger.warning("DrivetrainSystem class or components unavailable. Drivetrain set to None.")

        if self.drivetrain is None:
            logger.error("Failed to initialize Drivetrain component.")
            self.drivetrain = type('MockDrivetrain', (object,), {'num_gears': 6, 'transmission': transmission, 'get_overall_ratio': lambda s,g=1: 10.0, 'calculate_engine_speed_rpm': lambda s, spd, g: 3000.0, 'calculate_total_wheel_torque': lambda s, tq, g: tq * 10.0})()
            self.drivetrain.transmission = transmission # Attach transmission if available

        if self.drivetrain and not hasattr(self.drivetrain, 'num_gears') and self.drivetrain.transmission:
             self.drivetrain.num_gears = len(getattr(self.drivetrain.transmission, 'gear_ratios', [0]*6))

    def _initialize_cooling_system(self, cooling_instance: Optional[ExternalCoolingSystem], vehicle_ref):
        """Initialize the main external cooling system using self.config."""
        if ExternalCoolingSystem and isinstance(cooling_instance, ExternalCoolingSystem): # Check class existence
            self.cooling_system = cooling_instance
            logger.info("Using pre-configured external CoolingSystem instance.")
        else:
            cooling_config_path_ref = self.config.get('cooling_system_config_path')
            cooling_config_inline = self.config.get('cooling_system')
            base_cfg_dir = os.path.dirname(self.config_path) if self.config_path else '.'
            config_path = None
            if cooling_config_path_ref:
                resolved_path = os.path.join(base_cfg_dir, cooling_config_path_ref) if not os.path.isabs(cooling_config_path_ref) else cooling_config_path_ref
                if os.path.exists(resolved_path): config_path = resolved_path
                else: logger.warning(f"Cooling system config path not found: {resolved_path}")

            config_dir = os.path.dirname(config_path) if config_path else os.path.normpath(os.path.join(base_cfg_dir, '..', 'configs', 'thermal')) # Default relative location

            if config_path and create_formula_student_cooling_system: # Check factory exists
                logger.info(f"Initializing external CoolingSystem from config file: {config_path}")
                self.cooling_system = create_formula_student_cooling_system(config_dir=config_dir)
            elif isinstance(cooling_config_inline, dict):
                 logger.info("Initializing external CoolingSystem from inline config.")
                 try:
                      # Check needed components exist
                      from ..thermal.cooling_system import Radiator, WaterPump, CoolingFan, Thermostat # Local import
                      if Radiator and WaterPump and Thermostat and ExternalCoolingSystem: # Check essential classes
                          rad_cfg = cooling_config_inline.get('radiator', {})
                          pump_cfg = cooling_config_inline.get('water_pump', {})
                          fan_cfg = cooling_config_inline.get('cooling_fan', {})
                          thermo_cfg = cooling_config_inline.get('thermostat', {})
                          system_cfg = cooling_config_inline.get('system', {})

                          radiator = Radiator(**rad_cfg) if rad_cfg else Radiator()
                          pump = WaterPump(**pump_cfg) if pump_cfg else WaterPump()
                          fan = CoolingFan(**fan_cfg) if fan_cfg and CoolingFan else None # Fan optional
                          thermostat = Thermostat(**thermo_cfg) if thermo_cfg else Thermostat()

                          self.cooling_system = ExternalCoolingSystem(radiator, pump, fan, thermostat, **system_cfg)
                      else: raise ImportError("Missing essential cooling component classes")
                 except Exception as e:
                      logger.error(f"Failed to create cooling system from inline config: {e}. Creating default.")
                      self.cooling_system = create_formula_student_cooling_system(config_dir=config_dir) if create_formula_student_cooling_system else None
            else:
                logger.info("No cooling system config found. Creating default FS cooling system.")
                self.cooling_system = create_formula_student_cooling_system(config_dir=config_dir) if create_formula_student_cooling_system else None

        if self.cooling_system is None:
             logger.error("Failed to initialize CoolingSystem component.")
             self.cooling_system = type('MockCooling', (object,), {'coolant_temp_C': 25.0, 'total_thermal_capacity_J_K': 10000.0, 'radiator_heat_rejection_W': 0.0, 'simulate_step': lambda *args, **kwargs: None, 'get_system_specs': lambda: {}})()

        # Ensure essential cooling system attributes exist
        if not hasattr(self.cooling_system, 'coolant_temp_C'): self.cooling_system.coolant_temp_C = 25.0
        if not hasattr(self.cooling_system, 'total_thermal_capacity_J_K') or getattr(self.cooling_system, 'total_thermal_capacity_J_K', 0) <= 0:
             vol = getattr(self.cooling_system, 'coolant_volume_L', 2.5)
             dens = getattr(self.cooling_system, 'coolant_density_kg_L', 1.0)
             spec_heat = getattr(self.cooling_system, 'coolant_specific_heat_J_kgK', 4186)
             self.cooling_system.total_thermal_capacity_J_K = vol * dens * spec_heat
             if self.cooling_system.total_thermal_capacity_J_K <= 0: self.cooling_system.total_thermal_capacity_J_K = 1e-3

    def _initialize_shifting_systems(self, manager_instance: Optional[StrategyManager], cas_instance: Optional[CASSystem]):
        """Initialize shift manager and CAS system using self.config."""
        if StrategyManager and isinstance(manager_instance, StrategyManager):
            self.shift_manager = manager_instance
            logger.info("Using pre-configured StrategyManager instance.")
        else:
            if self.engine and self.drivetrain and create_formula_student_strategies:
                 self.shift_manager = create_formula_student_strategies(
                     engine_max_rpm=getattr(self.engine, 'redline_rpm', 14000.0),
                     engine_peak_power_rpm=getattr(self.engine, 'max_power_rpm', 12500.0),
                     engine_peak_torque_rpm=getattr(self.engine, 'max_torque_rpm', 10500.0),
                     gear_ratios=getattr(self.drivetrain.transmission, 'gear_ratios', []),
                     num_gears=getattr(self.drivetrain, 'num_gears', 6),
                     idle_rpm=getattr(self.engine, 'idle_rpm', 1300.0)
                 )
                 strat_cfg_path_ref = self.config.get('shift_strategy_config_path')
                 base_cfg_dir = os.path.dirname(self.config_path) if self.config_path else '.'
                 if strat_cfg_path_ref:
                      strat_cfg_path = os.path.join(base_cfg_dir, strat_cfg_path_ref) if not os.path.isabs(strat_cfg_path_ref) else strat_cfg_path_ref
                      if os.path.exists(strat_cfg_path) and hasattr(self.shift_manager, 'load_strategies_from_config'):
                          self.shift_manager.load_strategies_from_config(strat_cfg_path)
                      else:
                           logger.warning(f"Shift strategy config path not found or manager cannot load: {strat_cfg_path}")
                 logger.info("Initialized StrategyManager with FS strategies (potentially customized).")
            else:
                 logger.warning("Cannot initialize StrategyManager: Engine, Drivetrain, or factory missing.")
                 self.shift_manager = None

        if CASSystem and isinstance(cas_instance, CASSystem):
            self.cas_system = cas_instance
            logger.info("Using pre-configured CASSystem instance.")
        elif CASSystem and self.drivetrain and self.engine:
            cas_cfg_path_ref = self.config.get('cas_system_config_path')
            cas_cfg_path = None
            base_cfg_dir = os.path.dirname(self.config_path) if self.config_path else '.'
            if cas_cfg_path_ref:
                 resolved_path = os.path.join(base_cfg_dir, cas_cfg_path_ref) if not os.path.isabs(cas_cfg_path_ref) else cas_cfg_path_ref
                 if os.path.exists(resolved_path): cas_cfg_path = resolved_path
                 else: logger.warning(f"CAS config path specified but not found: {resolved_path}")

            self.cas_system = CASSystem(
                gear_ratios=getattr(self.drivetrain.transmission, 'gear_ratios', []),
                engine=self.engine,
                config_path=cas_cfg_path
            )
            logger.info("Initialized CASSystem.")
        else:
             logger.warning("Cannot initialize CASSystem: Class missing or Engine/Drivetrain missing.")
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
        if DualSidePodSystem and isinstance(side_pods_instance, DualSidePodSystem):
            self.side_pods = side_pods_instance
        elif create_standard_side_pod_system and (self.config.get('side_pods') or self.config.get('side_pod_config_path')):
             config_path = resolve_path('side_pod_config_path')
             config_dir = os.path.dirname(config_path) if config_path else os.path.normpath(os.path.join(base_cfg_dir, '..', 'configs', 'thermal'))
             self.side_pods = create_standard_side_pod_system(config_dir=config_dir) # Use factory
             logger.info("Initialized DualSidePodSystem.")
        else: self.side_pods = None

        # Rear Radiator
        if RearRadiatorSystem and isinstance(rear_rad_instance, RearRadiatorSystem):
            self.rear_radiator = rear_rad_instance
        elif create_default_rear_radiator_system and (self.config.get('rear_radiator') or self.config.get('rear_radiator_config_path')):
             config_path = resolve_path('rear_radiator_config_path')
             config_dir = os.path.dirname(config_path) if config_path else os.path.normpath(os.path.join(base_cfg_dir, '..', 'configs', 'thermal'))
             self.rear_radiator = create_default_rear_radiator_system(config_dir=config_dir)
             logger.info("Initialized RearRadiatorSystem.")
        else: self.rear_radiator = None

        # Cooling Assist
        if CoolingAssistSystem and isinstance(assist_instance, CoolingAssistSystem):
             self.cooling_assist = assist_instance
        elif create_default_cooling_assist_system and (self.config.get('cooling_assist') or self.config.get('cooling_assist_config_path')):
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
             # Ensure drivetrain method exists
             if hasattr(self.drivetrain, 'calculate_engine_speed_rpm'):
                 self.current_engine_rpm = self.drivetrain.calculate_engine_speed_rpm(
                     self.current_speed_mps, self.current_gear
                 )
                 # Clamp RPM
                 self.current_engine_rpm = np.clip(self.current_engine_rpm, self.engine.idle_rpm, self.engine.redline_rpm)
             else:
                  self.current_engine_rpm = self.engine.idle_rpm # Fallback
        else: # Neutral
            # Allow RPM to decay towards idle (simplified)
            idle = self.engine.idle_rpm
            decay_rate = 2000.0 # RPM per second decay rate
            time_now = time.monotonic()
            dt = time_now - self.last_update_time
            self.current_engine_rpm = max(idle, self.current_engine_rpm - decay_rate * dt)


        # Update engine's internal state (needed for temp factor in get_torque)
        if hasattr(self.engine, 'current_rpm'): self.engine.current_rpm = self.current_engine_rpm
        if hasattr(self.engine, 'throttle_position'): self.engine.throttle_position = self.throttle_input

        # Calculate engine torque based on current RPM, throttle, and *engine's* temperature
        # Engine torque calculation is needed before drivetrain update
        # Store it temporarily if needed later, or rely on drivetrain re-calculating it
        if hasattr(self.engine, 'get_torque'):
             engine_torque_nm = self.engine.get_torque(
                 rpm=self.current_engine_rpm,
                 throttle=self.throttle_input,
                 engine_temp=self.engine_temperature # Use vehicle's view of engine temp
             )
             self._current_engine_torque = engine_torque_nm # Store temporarily
        else:
             self._current_engine_torque = 0.0


    def update_drivetrain_state(self):
        """Update drivetrain based on requested gear and engine torque."""
        if not self.drivetrain or not self.engine:
             self._current_total_wheel_torque = 0.0
             return

        # Use the engine torque calculated in update_engine_state
        engine_torque_nm = getattr(self, '_current_engine_torque', 0.0)

        # Calculate total wheel torque using drivetrain method
        if hasattr(self.drivetrain, 'calculate_total_wheel_torque'):
            total_wheel_torque_nm = self.drivetrain.calculate_total_wheel_torque(
                engine_torque_nm=engine_torque_nm,
                gear=self.current_gear
            )
        else:
             total_wheel_torque_nm = 0.0 # Fallback if method missing

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
        try:
            heat_gen = self.engine.heat_model.calculate_heat_generation(self.current_engine_rpm, self.throttle_input)
            heat_to_coolant_W = heat_gen.get('to_coolant', 0.0)
            heat_to_oil_W = heat_gen.get('to_oil', 0.0)
            heat_block_ambient_gen = heat_gen.get('to_ambient', 0.0)
        except Exception as e:
            logger.warning(f"Could not calculate heat generation: {e}. Using zero input.")
            heat_to_coolant_W = 0.0
            heat_to_oil_W = 0.0
            heat_block_ambient_gen = 0.0

        # 2. Update External Cooling System State
        try:
             self.cooling_system.simulate_step(
                 ambient_temp_C=ambient_temp,
                 vehicle_speed_mps=self.current_speed_mps,
                 engine_temp=self.engine_temperature,
                 engine_rpm=self.current_engine_rpm,
                 engine_load=self.throttle_input,
                 engine_heat_input_W=heat_to_coolant_W,
                 dt=dt
             )
             self.radiator_heat_rejection_W = getattr(self.cooling_system, 'radiator_heat_rejection_W', 0.0)
             self.coolant_temperature = getattr(self.cooling_system, 'coolant_temp_C', ambient_temp)
        except AttributeError as e:
             logger.warning(f"Cooling system simulation step failed: {e}. Temps may not update correctly.")
             self.radiator_heat_rejection_W = 0.0

        # 3. Update Engine/Oil Temperatures using heat flows and capacities
        try:
            thermal_cfg = self.engine.heat_model.config
            capacities = thermal_cfg.get_thermal_capacities()
            safe_caps = {'engine_block': 50000, 'engine_oil': 10000, 'coolant_engine': 8000}
            safe_caps.update(capacities)

            temps_current = {'engine': self.engine_temperature, 'oil': self.oil_temperature, 'coolant': self.coolant_temperature}
            internal_transfer = self.engine.heat_model.calculate_internal_heat_transfer(temps_current)
            ambient_loss = self.engine.heat_model.calculate_ambient_heat_loss(temps_current, ambient_temp, self.current_speed_mps)

            q_block_to_coolant = internal_transfer.get('coolant_to_block', 0.0)
            q_block_to_oil = internal_transfer.get('oil_to_block', 0.0)
            q_block_to_ambient_loss = ambient_loss.get('block_to_ambient', 0.0)
            q_oil_to_ambient_loss = ambient_loss.get('oil_to_ambient', 0.0)

            q_net_engine = heat_block_ambient_gen - q_block_to_coolant - q_block_to_oil - q_block_to_ambient_loss
            q_net_oil = heat_to_oil_W + q_block_to_oil - q_oil_to_ambient_loss

            self.engine_temperature += (q_net_engine * dt) / max(1e-3, safe_caps['engine_block'])
            self.oil_temperature += (q_net_oil * dt) / max(1e-3, safe_caps['engine_oil'])

            # Clamp temperatures
            self.engine_temperature = max(ambient_temp - 5, self.engine_temperature)
            self.oil_temperature = max(ambient_temp - 5, self.oil_temperature)
        except Exception as e:
             logger.warning(f"Error updating engine/oil temperatures: {e}")

        # 4. Update engine's internal state and thermal factor
        if hasattr(self.engine, 'engine_temperature'): self.engine.engine_temperature = self.engine_temperature
        if hasattr(self.engine, 'coolant_temperature'): self.engine.coolant_temperature = self.coolant_temperature
        if hasattr(self.engine, 'oil_temperature'): self.engine.oil_temperature = self.oil_temperature
        if hasattr(self.engine, '_get_thermal_performance_factor'):
             try:
                 current_thermal_factor = self.engine._get_thermal_performance_factor(self.engine_temperature)
                 self.thermal_factor = current_thermal_factor
                 if hasattr(self.engine, 'thermal_factor'): self.engine.thermal_factor = current_thermal_factor
             except Exception as e:
                  logger.warning(f"Error updating thermal factor: {e}")
                  self.thermal_factor = 1.0
                  if hasattr(self.engine, 'thermal_factor'): self.engine.thermal_factor = 1.0

    # change_gear remains the same

    def calculate_forces(self) -> Dict[str, float]:
        """Calculate major longitudinal forces acting on the vehicle."""
        F_tractive = getattr(self, '_current_total_wheel_torque', 0.0) / self.tire_radius_m if self.tire_radius_m > 0 else 0.0
        F_drag = 0.5 * AIR_DENSITY_SEA_LEVEL * self.drag_coefficient * self.frontal_area_m2 * self.current_speed_mps**2
        # Calculate downforce using cornering calculator if available, else use vehicle Lc
        if self.cornering:
             F_downforce = self.cornering.calculate_downforce_N(self.current_speed_mps)
        else:
             F_downforce = -0.5 * AIR_DENSITY_SEA_LEVEL * self.lift_coefficient * self.frontal_area_m2 * self.current_speed_mps**2

        normal_load = self.mass * GRAVITY + F_downforce
        F_rolling = self.rolling_resistance_coeff * max(0, normal_load)

        # Braking force: Use max_braking_g limit applied to total normal load
        max_brake_force = max(0, normal_load) * self.max_braking_g * GRAVITY # Ensure normal load isn't negative before multiply
        F_brake = self.brake_input * max_brake_force

        return {
            'tractive': F_tractive, 'drag': F_drag, 'rolling': F_rolling,
            'brake': F_brake, 'downforce': F_downforce
        }

    def calculate_acceleration(self, throttle: Optional[float] = None, brake: Optional[float] = None) -> float:
        """Calculate current longitudinal acceleration."""
        if throttle is not None: self.throttle_input = np.clip(throttle, 0.0, 1.0)
        if brake is not None: self.brake_input = np.clip(brake, 0.0, 1.0)

        # Update engine/drivetrain first to get current forces
        self.update_engine_state()
        self.update_drivetrain_state()

        forces = self.calculate_forces()
        net_force = forces['tractive'] - forces['drag'] - forces['rolling'] - forces['brake']
        self.current_acceleration_mpss = net_force / self.mass if self.mass > 0 else 0.0
        return self.current_acceleration_mpss

    # update_vehicle_state remains the same

    # simulate_acceleration_run remains the same

    # simulate_skidpad remains the same

    # simulate_lap remains the same

    # calculate_weight_transfer remains the same

    # get_vehicle_specs remains the same

    # calculate_performance_metrics remains the same

    def calculate_max_speed(self) -> float:
        """Estimate theoretical maximum speed where tractive force equals drag+rolling."""
        # This requires solving F_tractive(v, gear_max) = F_drag(v) + F_rolling(v)
        # It's an iterative process or requires solving a polynomial if forces are simplified.

        # Simplified iterative approach:
        guess_speed_mps = 50.0 # Start guess ~180 kph
        top_gear = self.drivetrain.num_gears if self.drivetrain else 1
        rpm_limit = self.engine.redline_rpm if self.engine else 14000

        for _ in range(10): # Iterate to converge
            # 1. Calculate forces at current guess speed
            rpm_at_guess = self.drivetrain.calculate_engine_speed_rpm(guess_speed_mps, top_gear) if self.drivetrain else 0
            if rpm_at_guess > rpm_limit: # RPM limited
                 guess_speed_mps = self.drivetrain.calculate_vehicle_speed_mps(rpm_limit, top_gear) if self.drivetrain else 0
                 continue # Re-evaluate forces at the RPM limited speed

            engine_torque = self.engine.get_torque(rpm_at_guess, throttle=1.0) if self.engine else 0
            f_tractive = self.calculate_tractive_force_N(engine_torque, top_gear)

            # Calculate drag/rolling at this speed
            forces = self.calculate_forces() # Use internal speed state updated implicitly
            f_drag = forces['drag']
            f_rolling = forces['rolling']
            f_resist = f_drag + f_rolling

            # 2. Check balance
            force_diff = f_tractive - f_resist
            if abs(force_diff) < 1.0: # Converged (within 1 Newton)
                break

            # 3. Adjust guess (simple proportional step)
            # If F_tractive > F_resist, speed can increase. If F_tractive < F_resist, speed must decrease.
            # Need a sensitivity term d(F_resist - F_tractive)/dv - very complex
            # Simplification: adjust speed by a fraction of the force difference
            speed_adjustment = force_diff * 0.05 # Adjust speed by 0.05 m/s per Newton diff (tuning factor)
            guess_speed_mps += speed_adjustment
            guess_speed_mps = max(0.1, guess_speed_mps) # Ensure positive speed

        else:
            logger.warning(f"Max speed calculation did not fully converge. Final guess: {guess_speed_mps:.1f} m/s")

        return guess_speed_mps

    # plot_acceleration_results remains the same

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
    try:
        vehicle = Vehicle(config_path=config_path)
        logger.info("Formula Student Vehicle created successfully.")
        return vehicle
    except Exception as e:
         logger.critical(f"Failed to create Formula Student vehicle: {e}", exc_info=True)
         # Attempt to create with absolutely no config (pure defaults)
         try:
             logger.warning("Attempting to create vehicle with pure defaults...")
             vehicle = Vehicle()
             logger.info("Formula Student Vehicle created with defaults.")
             return vehicle
         except Exception as fallback_e:
              logger.critical(f"Failed to create vehicle even with pure defaults: {fallback_e}", exc_info=True)
              raise RuntimeError("Vehicle creation failed completely.") from fallback_e


# Example Usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Vehicle Module Demo")
    print("-" * 20)

    try:
        # Create vehicle using factory function
        vehicle = create_formula_student_vehicle() # Tries to load defaults

        print("\n--- Vehicle Specs ---")
        specs = vehicle.get_vehicle_specs()
        print(f" Mass: {specs.get('vehicle',{}).get('mass', 'N/A'):.1f} kg")
        print(f" Engine: {specs.get('engine',{}).get('make', 'N/A')} {specs.get('engine',{}).get('model', 'N/A')}")
        print(f" Max Power: {specs.get('engine',{}).get('max_power_hp', 'N/A'):.1f} HP @ {specs.get('engine',{}).get('max_power_rpm', 'N/A'):.0f} RPM")
        print(f" Gears: {specs.get('drivetrain',{}).get('num_gears', 'N/A')}")
        print(f" Final Drive: {specs.get('drivetrain',{}).get('final_drive_ratio', 'N/A'):.3f}")

        print("\n--- Performance Metrics ---")
        metrics = vehicle.calculate_performance_metrics()
        print(f" Power/Weight: {metrics.get('power_to_weight_kw_kg', 0):.3f} kW/kg")
        print(f" Max Speed (Est): {metrics.get('max_speed_kph', 0):.1f} km/h")
        print(f" Max Lateral G (Est): {metrics.get('max_lateral_g', 0):.2f} g")

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
        if AccelerationSimulator_available:
            accel_results = vehicle.simulate_acceleration_run()
            if accel_results and 'error' not in accel_results:
                print(f" 75m Time: {accel_results.get('finish_time', -1):.3f} s")
                print(f" 0-60 mph: {accel_results.get('time_to_60mph', -1):.3f} s")
            else:
                 print(f" Acceleration simulation failed: {accel_results.get('error', 'Unknown error')}")
        else:
            print(" Acceleration simulation skipped: Simulator not available.")

        print("\nVehicle demo finished.")

    except ImportError as e:
        print(f"Import Error during demo: {e}")
    except FileNotFoundError as e:
         print(f"Config file not found: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()