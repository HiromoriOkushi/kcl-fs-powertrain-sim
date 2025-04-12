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
import time

logger = logging.getLogger("Vehicle")

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
    # --- Import the plotting module itself ---
    from ..utils import plotting as plotting_utils
    # -----------------------------------------

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
    # Mock plotting utils if primary import fails
    class MockPlotting:
        def plot_acceleration_results(self, *args, **kwargs): pass
        def save_plot(self, *args, **kwargs): pass
    plotting_utils = MockPlotting()
    CorneringPerformance = None; CorneringPerformance_available = False
    AccelerationSimulator = None; AccelerationSimulator_available = False
    LapTimeSimulator = None; LapTimeSimulator_available = False
    def create_formula_student_strategies(*args, **kwargs): return None
    def create_standard_side_pod_system(*args, **kwargs): return None
    def create_default_rear_radiator_system(*args, **kwargs): return None
    def create_default_cooling_assist_system(*args, **kwargs): return None
    def create_formula_student_cooling_system(*args, **kwargs): return None


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
        self.include_thermal: bool = True
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
        if MotorcycleEngine and isinstance(engine_instance, MotorcycleEngine):
            self.engine = engine_instance
            logger.info("Using pre-configured Engine instance.")
        else:
            engine_config_ref = self.config.get('engine_config_path', self.config.get('engine'))
            self.engine = None # Initialize as None
            if isinstance(engine_config_ref, str):
                if os.path.exists(engine_config_ref):
                    logger.info(f"Initializing Engine from config file: {engine_config_ref}")
                    self.engine = MotorcycleEngine(config_path=engine_config_ref) if MotorcycleEngine else None
                else:
                    logger.warning(f"Engine config path not found: {engine_config_ref}.")
            elif isinstance(engine_config_ref, dict):
                logger.info("Initializing Engine from inline config in vehicle config.")
                self.engine = MotorcycleEngine(engine_params=engine_config_ref) if MotorcycleEngine else None

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

        self.engine.idle_rpm = getattr(self.engine, 'idle_rpm', 1300.0)
        self.engine.redline_rpm = getattr(self.engine, 'redline_rpm', 14000.0)
        self.engine.max_power_rpm = getattr(self.engine, 'max_power_rpm', 12500.0)
        self.engine.max_torque_rpm = getattr(self.engine, 'max_torque_rpm', 10500.0)
        self.engine.engine_temperature = getattr(self.engine, 'engine_temperature', 25.0)
        self.engine.coolant_temperature = getattr(self.engine, 'coolant_temperature', 25.0)
        self.engine.oil_temperature = getattr(self.engine, 'oil_temperature', 25.0)
        self.engine.thermal_factor = getattr(self.engine, 'thermal_factor', 1.0)

        if self.include_thermal and (not hasattr(self.engine, 'heat_model') or self.engine.heat_model is None):
             if EngineHeatModel and ThermalConfig:
                 logger.debug("Creating default heat model for engine.")
                 thermal_cfg = getattr(self.engine, 'thermal_config', ThermalConfig())
                 if thermal_cfg: self.engine.heat_model = EngineHeatModel(thermal_cfg, self.engine)
                 else: self.engine.heat_model = None
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
                if not comp_class: return None # Return None if class missing

                instance = None
                if isinstance(comp_ref, str): # Path
                    path = os.path.join(base_cfg_dir, comp_ref) if not os.path.isabs(comp_ref) else comp_ref
                    if os.path.exists(path):
                        try:
                            with open(path, 'r') as f:
                                file_content = yaml.safe_load(f)
                                params = file_content.get(class_name, {}) if file_content else {}
                            instance = comp_class(**params)
                        except Exception as e: logger.error(f"Error loading {class_name} from {path}: {e}")
                    else: logger.warning(f"{class_name} config not found: {path}")
                elif isinstance(comp_ref, dict): # Inline dict
                    instance = comp_class(**comp_ref)

                if instance is None: # Fallback to default if loading failed or no config provided
                    logger.debug(f"Using default {class_name} parameters.")
                    instance = comp_class(**default_args)
                return instance

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

            if DrivetrainSystem and transmission and final_drive:
                 self.drivetrain = DrivetrainSystem(
                     transmission, final_drive, differential,
                     wheel_radius_m=self.tire_radius_m,
                 )
                 logger.info("Drivetrain initialized from config/defaults.")
            else:
                 self.drivetrain = None
                 logger.warning("DrivetrainSystem class or components unavailable. Drivetrain set to None.")

        if self.drivetrain is None:
            logger.error("Failed to initialize Drivetrain component.")
            self.drivetrain = type('MockDrivetrain', (object,), {'num_gears': 6, 'transmission': transmission, 'get_overall_ratio': lambda s,g=1: 10.0, 'calculate_engine_speed_rpm': lambda s, spd, g: 3000.0, 'calculate_total_wheel_torque': lambda s, tq, g: tq * 10.0})()
            self.drivetrain.transmission = transmission

        # Ensure num_gears is set
        if self.drivetrain and not hasattr(self.drivetrain, 'num_gears') and self.drivetrain.transmission:
             gear_ratios = getattr(self.drivetrain.transmission, 'gear_ratios', [])
             self.drivetrain.num_gears = len(gear_ratios) if isinstance(gear_ratios, (list, np.ndarray)) else 0

    def _initialize_cooling_system(self, cooling_instance: Optional[ExternalCoolingSystem], vehicle_ref):
        """Initialize the main external cooling system using self.config."""
        if ExternalCoolingSystem and isinstance(cooling_instance, ExternalCoolingSystem):
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

            config_dir = os.path.dirname(config_path) if config_path else os.path.normpath(os.path.join(base_cfg_dir, '..', 'configs', 'thermal'))

            self.cooling_system = None # Initialize as None
            if config_path and create_formula_student_cooling_system:
                logger.info(f"Initializing external CoolingSystem from config file: {config_path}")
                self.cooling_system = create_formula_student_cooling_system(config_dir=config_dir)
            elif isinstance(cooling_config_inline, dict):
                 logger.info("Initializing external CoolingSystem from inline config.")
                 try:
                      from ..thermal.cooling_system import Radiator, WaterPump, CoolingFan, Thermostat
                      if Radiator and WaterPump and Thermostat and ExternalCoolingSystem:
                          rad_cfg = cooling_config_inline.get('radiator', {})
                          pump_cfg = cooling_config_inline.get('water_pump', {})
                          fan_cfg = cooling_config_inline.get('cooling_fan', {})
                          thermo_cfg = cooling_config_inline.get('thermostat', {})
                          system_cfg = cooling_config_inline.get('system', {})
                          rad_cfg.pop('type', None)
                          pump_cfg.pop('type', None)
                          fan_cfg.pop('type', None)
                          
                          if 'voltage' in fan_cfg and 'voltage_V' not in fan_cfg:
                            fan_cfg['voltage_V'] = fan_cfg.pop('voltage')
                          if 'max_flow_rate' in pump_cfg and 'max_flow_rate_lpm' not in pump_cfg:
                            pump_cfg['max_flow_rate_lpm'] = pump_cfg.pop('max_flow_rate')
                          if 'max_pressure' in pump_cfg and 'max_pressure_bar' not in pump_cfg:
                            pump_cfg['max_pressure_bar'] = pump_cfg.pop('max_pressure')
                          if 'nominal_speed' in pump_cfg and 'nominal_speed_rpm' not in pump_cfg:
                            pump_cfg['nominal_speed_rpm'] = pump_cfg.pop('nominal_speed')
                          
                          radiator = Radiator(**rad_cfg) if rad_cfg else Radiator()   
                          pump = WaterPump(**pump_cfg) if pump_cfg else WaterPump()
                          fan = CoolingFan(**fan_cfg) if fan_cfg and CoolingFan else None
                          thermostat = Thermostat(**thermo_cfg) if thermo_cfg else Thermostat()
                          self.cooling_system = ExternalCoolingSystem(radiator, pump, fan, thermostat, **system_cfg)
                      else: 
                          raise ImportError("Missing essential cooling component classes")
                 except Exception as e:
                      logger.error(f"Failed to create cooling system from inline config: {e}.")
                      self.cooling_system = None # Failed

            # Fallback to default if still None
            if self.cooling_system is None and create_formula_student_cooling_system:
                logger.info("No cooling system config found or creation failed. Creating default FS cooling system.")
                self.cooling_system = create_formula_student_cooling_system(config_dir=config_dir)

        if self.cooling_system is None:
             logger.error("Failed to initialize CoolingSystem component.")
             self.cooling_system = type('MockCooling', (object,), {'coolant_temp_C': 25.0, 'total_thermal_capacity_J_K': 10000.0, 'radiator_heat_rejection_W': 0.0, 'simulate_step': lambda *args, **kwargs: None, 'get_system_specs': lambda: {}})()

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
                 # Gather parameters safely using getattr
                 gear_ratios = getattr(getattr(self.drivetrain, 'transmission', None), 'gear_ratios', [])
                 num_gears = getattr(self.drivetrain, 'num_gears', 0)
                 if (gear_ratios is None or len(gear_ratios) == 0) and num_gears > 0:
                     gear_ratios = [0.0] * num_gears # Handle missing ratios
                     logger.warning(f"Transmission gear ratios were missing, created placeholder for {num_gears} gears.")

                 self.shift_manager = create_formula_student_strategies(
                     engine_max_rpm=getattr(self.engine, 'redline_rpm', 14000.0),
                     engine_peak_power_rpm=getattr(self.engine, 'max_power_rpm', 12500.0),
                     engine_peak_torque_rpm=getattr(self.engine, 'max_torque_rpm', 10500.0),
                     gear_ratios=gear_ratios,
                     num_gears=num_gears,
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
             try: self.side_pods = create_standard_side_pod_system(config_dir=config_dir)
             except Exception as e: logger.error(f"Failed to create side pods: {e}"); self.side_pods = None
             if self.side_pods: logger.info("Initialized DualSidePodSystem.")
        else: self.side_pods = None

        # Rear Radiator
        if RearRadiatorSystem and isinstance(rear_rad_instance, RearRadiatorSystem):
            self.rear_radiator = rear_rad_instance
        elif create_default_rear_radiator_system and (self.config.get('rear_radiator') or self.config.get('rear_radiator_config_path')):
             config_path = resolve_path('rear_radiator_config_path')
             config_dir = os.path.dirname(config_path) if config_path else os.path.normpath(os.path.join(base_cfg_dir, '..', 'configs', 'thermal'))
             try: self.rear_radiator = create_default_rear_radiator_system(config_dir=config_dir)
             except Exception as e: logger.error(f"Failed to create rear radiator: {e}"); self.rear_radiator = None
             if self.rear_radiator: logger.info("Initialized RearRadiatorSystem.")
        else: self.rear_radiator = None

        # Cooling Assist
        if CoolingAssistSystem and isinstance(assist_instance, CoolingAssistSystem):
             self.cooling_assist = assist_instance
        elif create_default_cooling_assist_system and (self.config.get('cooling_assist') or self.config.get('cooling_assist_config_path')):
             config_path = resolve_path('cooling_assist_config_path')
             config_dir = os.path.dirname(config_path) if config_path else os.path.normpath(os.path.join(base_cfg_dir, '..', 'configs', 'thermal'))
             try: self.cooling_assist = create_default_cooling_assist_system(config_dir=config_dir)
             except Exception as e: logger.error(f"Failed to create cooling assist: {e}"); self.cooling_assist = None
             if self.cooling_assist: logger.info("Initialized CoolingAssistSystem.")
        else: self.cooling_assist = None

    def update_engine_state(self):
        """Update engine RPM and store calculated torque."""
        if not self.engine or not self.drivetrain:
            self._current_engine_torque = 0.0
            return

        if self.current_gear > 0 and hasattr(self.drivetrain, 'calculate_engine_speed_rpm'):
            self.current_engine_rpm = self.drivetrain.calculate_engine_speed_rpm(self.current_speed_mps, self.current_gear)
            self.current_engine_rpm = np.clip(self.current_engine_rpm, self.engine.idle_rpm, self.engine.redline_rpm)
        else:
            idle = self.engine.idle_rpm
            decay_rate = 2000.0
            time_now = time.monotonic()
            dt = time_now - self.last_update_time
            self.current_engine_rpm = max(idle, self.current_engine_rpm - decay_rate * dt)

        if hasattr(self.engine, 'current_rpm'): self.engine.current_rpm = self.current_engine_rpm
        if hasattr(self.engine, 'throttle_position'): self.engine.throttle_position = self.throttle_input

        if hasattr(self.engine, 'get_torque'):
            engine_torque_nm = self.engine.get_torque(rpm=self.current_engine_rpm, throttle=self.throttle_input, engine_temp=self.engine_temperature)
            self._current_engine_torque = engine_torque_nm
        else:
            self._current_engine_torque = 0.0

    def update_drivetrain_state(self):
        """Update wheel torque based on current engine torque and gear."""
        if not self.drivetrain:
            self._current_total_wheel_torque = 0.0
            return

        engine_torque_nm = getattr(self, '_current_engine_torque', 0.0)
        if hasattr(self.drivetrain, 'calculate_total_wheel_torque'):
            total_wheel_torque_nm = self.drivetrain.calculate_total_wheel_torque(engine_torque_nm=engine_torque_nm, gear=self.current_gear)
        else:
            total_wheel_torque_nm = 0.0
        self._current_total_wheel_torque = total_wheel_torque_nm

    def update_thermal_state(self, dt: float, ambient_temp_C: Optional[float] = None):
        """Update thermal state (engine, coolant, oil)."""
        if not self.include_thermal or not self.engine or not self.cooling_system:
            # Set thermal factor to 1 if thermal simulation is disabled or components missing
            self.thermal_factor = 1.0
            if hasattr(self.engine, 'thermal_factor'):
                 self.engine.thermal_factor = 1.0
            return 

        ambient_temp = ambient_temp_C if ambient_temp_C is not None else 25.0
        heat_to_coolant_W = 0.0
        heat_to_oil_W = 0.0
        heat_block_ambient_gen = 0.0

        # 1. Calculate Heat Generation
        if hasattr(self.engine, 'heat_model') and self.engine.heat_model:
            try:
                rpm_for_heat = getattr(self, 'current_engine_rpm', getattr(self.engine, 'idle_rpm', 1300))
                throttle_for_heat = getattr(self, 'throttle_input', 0.0)

                heat_gen = self.engine.heat_model.calculate_heat_generation(rpm_for_heat, throttle_for_heat)
                heat_to_coolant_W = heat_gen.get('to_coolant', 0.0)
                heat_to_oil_W = heat_gen.get('to_oil', 0.0)
                heat_block_ambient_gen = heat_gen.get('to_ambient', 0.0)
            except Exception as e:
                logger.warning(f"Could not calculate heat generation: {e}")
        else:
            logger.debug("No engine heat model available for heat generation calculation.")
            rpm_for_heat = getattr(self, 'current_engine_rpm', getattr(self.engine, 'idle_rpm', 1300))
            throttle_for_heat = getattr(self, 'throttle_input', 0.0)
            estimated_power_kw = self.engine.get_power(rpm_for_heat, throttle_for_heat) if hasattr(self.engine, 'get_power') else 0
            heat_to_coolant_W = (estimated_power_kw * 1500) + (3000 * throttle_for_heat) # Rough estimate

        # 2. Update External Cooling System
        try:
             self.cooling_system.simulate_step(
                 ambient_temp_C=ambient_temp,
                 vehicle_speed_mps=self.current_speed_mps, 
                 engine_rpm=getattr(self, 'current_engine_rpm', 0),
                 engine_load=getattr(self, 'throttle_input', 0.0),
                 engine_heat_input_W=heat_to_coolant_W,
                 dt=dt
             )
             self.radiator_heat_rejection_W = getattr(self.cooling_system, 'radiator_heat_rejection_W', 0.0)
             self.coolant_temperature = getattr(self.cooling_system, 'coolant_temp_C', self.coolant_temperature)
        except AttributeError as e:
             logger.warning(f"Cooling system simulation step or attribute access failed: {e}.")
             self.radiator_heat_rejection_W = 0.0
        except TypeError as e:
             logger.error(f"TypeError calling cooling_system.simulate_step: {e}. Check arguments.", exc_info=True)
             self.radiator_heat_rejection_W = 0.0
        except Exception as e:
             logger.error(f"Unexpected error during cooling system update: {e}", exc_info=True)
             self.radiator_heat_rejection_W = 0.0

        # 3. Update Engine/Oil Temperatures (if detailed engine heat model exists)
        if hasattr(self.engine, 'heat_model') and self.engine.heat_model:
            try:
                thermal_cfg = self.engine.heat_model.config
                capacities = thermal_cfg.get_thermal_capacities()
                cap_eng_block = capacities.get('engine_block', 50000.0)
                cap_eng_oil = capacities.get('engine_oil', 10000.0)
                if cap_eng_block <= 0: cap_eng_block = 1e-3
                if cap_eng_oil <= 0: cap_eng_oil = 1e-3

                temps_current = {'engine': self.engine_temperature, 'oil': self.oil_temperature, 'coolant': self.coolant_temperature}
                internal_transfer = self.engine.heat_model.calculate_internal_heat_transfer(temps_current)
                ambient_loss = self.engine.heat_model.calculate_ambient_heat_loss(temps_current, ambient_temp, self.current_speed_mps)

                q_block_to_coolant = -internal_transfer.get('coolant_to_block', 0.0)
                q_block_to_oil = -internal_transfer.get('oil_to_block', 0.0)
                q_block_to_ambient_loss = ambient_loss.get('block_to_ambient', 0.0)
                q_oil_to_ambient_loss = ambient_loss.get('oil_to_ambient', 0.0)

                q_net_engine = internal_transfer.get('oil_to_block', 0.0) + \
                               internal_transfer.get('coolant_to_block', 0.0) + \
                               heat_block_ambient_gen - \
                               q_block_to_ambient_loss

                q_net_oil = heat_to_oil_W - internal_transfer.get('oil_to_block', 0.0) - q_oil_to_ambient_loss

                self.engine_temperature += (q_net_engine * dt) / cap_eng_block
                self.oil_temperature += (q_net_oil * dt) / cap_eng_oil

                self.engine_temperature = max(ambient_temp - 5, self.engine_temperature)
                self.oil_temperature = max(ambient_temp - 5, self.oil_temperature)

            except AttributeError as e: # Catch the specific error if calculate_ambient_heat_loss fails
                 logger.warning(f"AttributeError updating engine/oil temperatures: {e}. Using fallback.")
                 self.engine_temperature += (self.coolant_temperature + 5 - self.engine_temperature) * 0.05 * dt
                 self.oil_temperature += (self.coolant_temperature + 10 - self.oil_temperature) * 0.03 * dt
            except Exception as e:
                 logger.warning(f"Error updating engine/oil temperatures via heat model: {e}", exc_info=True)
                 self.engine_temperature += (self.coolant_temperature + 5 - self.engine_temperature) * 0.05 * dt
                 self.oil_temperature += (self.coolant_temperature + 10 - self.oil_temperature) * 0.03 * dt
        else:
            self.engine_temperature += (self.coolant_temperature + 5 - self.engine_temperature) * 0.05 * dt
            self.oil_temperature += (self.coolant_temperature + 10 - self.oil_temperature) * 0.03 * dt

        # 4. Update engine's internal state and thermal factor
        if hasattr(self.engine, 'engine_temperature'): self.engine.engine_temperature = self.engine_temperature
        if hasattr(self.engine, 'coolant_temperature'): self.engine.coolant_temperature = self.coolant_temperature
        if hasattr(self.engine, 'oil_temperature'): self.engine.oil_temperature = self.oil_temperature

        if hasattr(self.engine, '_get_thermal_performance_factor'):
             try:
                 current_thermal_factor = self.engine._get_thermal_performance_factor(self.engine_temperature)
                 self.thermal_factor = current_thermal_factor
                 if hasattr(self.engine, 'thermal_factor'):
                      self.engine.thermal_factor = current_thermal_factor
             except Exception as e:
                  logger.warning(f"Error updating thermal factor: {e}")
                  self.thermal_factor = 1.0
                  if hasattr(self.engine, 'thermal_factor'): self.engine.thermal_factor = 1.0
        else:
             self.thermal_factor = 1.0

    def change_gear(self, target_gear: int) -> Tuple[bool, float]:
        """Request a gear change via CAS if available, otherwise direct change."""
        shift_duration_s = 0.0
        success = False
        if self.drivetrain is None: return False, 0.0
        gear_before = self.current_gear
        if target_gear == gear_before: return True, 0.0

        if self.cas_system:
            current_time_s = time.monotonic()
            direction = ShiftDirection.NEUTRAL
            if target_gear > gear_before: direction = ShiftDirection.UP
            elif target_gear < gear_before: direction = ShiftDirection.DOWN
            current_rpm = getattr(self.engine, 'current_rpm', 0)

            # Pass current_time in ms to CAS readiness check
            if self.cas_system._check_shift_readiness(current_time_s * 1000.0):
                 # CAS request needs direction, current RPM (for overrev), and optional target override
                 initiated = self.cas_system.request_shift(direction, current_rpm, target_gear_override=target_gear)
                 if initiated:
                     # Mark vehicle state as potentially shifting - simulator needs to check CAS state
                     # Get duration for simulator event scheduling
                     shift_duration_s = self.cas_system.get_total_shift_time_ms(direction) / 1000.0
                     logger.debug(f"CAS shift {gear_before}->{target_gear} initiated. Duration: {shift_duration_s*1000:.1f} ms.")
                     success = True
                 else:
                     logger.debug(f"CAS shift request {gear_before}->{target_gear} rejected by CAS logic.")
            else:
                 logger.debug(f"CAS shift request {gear_before}->{target_gear} rejected by readiness check.")
        else:
            if 0 <= target_gear <= self.drivetrain.num_gears:
                 if hasattr(self.drivetrain, 'change_gear'):
                     success = self.drivetrain.change_gear(target_gear)
                     if success:
                         self.current_gear = target_gear
                         shift_duration_s = 0.050
                         logger.debug(f"Direct shift to gear {target_gear} successful.")
                     else: logger.warning(f"Direct gear change to {target_gear} failed.")
                 else: logger.error("Drivetrain object missing change_gear method.")
            else: logger.error(f"Invalid target gear {target_gear} for direct change.")

        return success, shift_duration_s

    def calculate_forces(self) -> Dict[str, float]:
        """Calculate major longitudinal forces acting on the vehicle."""
        F_tractive = getattr(self, '_current_total_wheel_torque', 0.0) / self.tire_radius_m if self.tire_radius_m > 0 else 0.0
        F_drag = 0.5 * AIR_DENSITY_SEA_LEVEL * self.drag_coefficient * self.frontal_area_m2 * self.current_speed_mps**2

        if self.cornering:
             F_downforce = self.cornering.calculate_downforce_N(self.current_speed_mps)
        else:
             F_downforce = -0.5 * AIR_DENSITY_SEA_LEVEL * self.lift_coefficient * self.frontal_area_m2 * self.current_speed_mps**2

        normal_load = self.mass * GRAVITY + F_downforce
        F_rolling = self.rolling_resistance_coeff * max(0, normal_load)

        max_brake_force = max(0, normal_load) * self.max_braking_g # Use max_braking_g, which is positive
        F_brake = self.brake_input * max_brake_force * GRAVITY # Multiply by G to get force

        return {
            'tractive': F_tractive, 'drag': F_drag, 'rolling': F_rolling,
            'brake': F_brake, 'downforce': F_downforce
        }

    def calculate_acceleration(self, throttle: Optional[float] = None, brake: Optional[float] = None) -> float:
        """Calculate current longitudinal acceleration."""
        if throttle is not None: self.throttle_input = np.clip(throttle, 0.0, 1.0)
        if brake is not None: self.brake_input = np.clip(brake, 0.0, 1.0)

        # Ensure engine/drivetrain states are updated
        self.update_engine_state()
        self.update_drivetrain_state()

        forces = self.calculate_forces()
        net_force = forces['tractive'] - forces['drag'] - forces['rolling'] - forces['brake']
        self.current_acceleration_mpss = net_force / self.mass if self.mass > 0 else 0.0
        return self.current_acceleration_mpss

    def update_vehicle_state(self, dt: float, ambient_temp_C: float = 25.0):
        """Update vehicle kinematics and thermal state."""
        if dt <= 0: return
        start_time_mono = time.monotonic()
        # Removed: self.last_update_time = start_time_mono - Not used locally

        # --- Handle CAS Shift Completion ---
        is_shifting = False
        cas_is_ready = not (self.cas_system and self.cas_system.system_state != ShiftState.IDLE) # Assume ready if no CAS
        if self.cas_system and self.cas_system.system_state == ShiftState.SHIFT_IN_PROGRESS:
            is_shifting = True # Mark as currently shifting for potential overrides
            # Check completion based on stored start time and duration
            shift_start_s = getattr(self.cas_system, 'shift_start_time_s', 0.0)
            if shift_start_s > 0:
                 shift_elapsed_s = start_time_mono - shift_start_s
                 last_direction = getattr(self.cas_system, '_last_direction', ShiftDirection.UP) # Assume UP if missing
                 required_duration_s = self.cas_system.get_total_shift_time_ms(last_direction) / 1000.0
                 if required_duration_s > 0 and shift_elapsed_s >= required_duration_s:
                      self.cas_system.complete_shift(start_time_mono) # Call completion
                      self.current_gear = self.cas_system.current_gear # Sync gear
                      is_shifting = False # Shift just completed
                      cas_is_ready = True # System is now ready
                      logger.debug(f"CAS shift completed this step. New gear: {self.current_gear}")

        # --- Evaluate Shift Strategy ---
        # Only evaluate if CAS is ready (i.e., not mid-shift)
        if self.shift_manager and self.current_gear != -1 and cas_is_ready:
            state_for_shift = {
                'engine_rpm': self.current_engine_rpm,
                'vehicle_speed': self.current_speed_mps,
                'throttle_position': self.throttle_input,
                'engine_load': self.throttle_input, # Approximate load with throttle
                'num_gears': self.drivetrain.num_gears if self.drivetrain else 0,
                'engine_redline_rpm': self.engine.redline_rpm if self.engine else 14000
            }
            target_gear = self.shift_manager.evaluate_shift(self.current_gear, state_for_shift)
            if target_gear is not None and target_gear != self.current_gear:
                logger.debug(f"Shift requested by strategy: {self.current_gear}->{target_gear}")
                # change_gear handles CAS initiation and returns success/duration
                # Success just means it was initiated, completion is checked next step
                self.change_gear(target_gear)
                # Note: self.current_gear is NOT updated here

        # --- Apply overrides if CAS is actively mid-shift ---
        throttle_override = self.throttle_input
        engine_factor_override = 1.0
        if is_shifting and self.cas_system:
             if self.cas_system.system_state == ShiftState.IGNITION_CUT: # Check specific sub-states if defined
                  engine_factor_override = 0.0
                  throttle_override = 0.0 # Cut throttle during ignition cut too? Optional.
             elif self.cas_system.system_state == ShiftState.PREPARE_UPSHIFT: # Example state
                  throttle_override *= (1.0 - self.cas_system.throttle_cut_percent / 100.0)
             # Add other states like THROTTLE_BLIP if modeled

        # 1. Calculate acceleration (using potentially overridden throttle/engine factor)
        # Temporarily modify engine torque calculation if needed
        original_engine_get_torque = getattr(self.engine, 'get_torque', None)
        if engine_factor_override < 1.0 and original_engine_get_torque and callable(original_engine_get_torque):
             def modified_get_torque(*args, **kwargs):
                  return original_engine_get_torque(*args, **kwargs) * engine_factor_override
             self.engine.get_torque = modified_get_torque

        # Use overridden throttle for calculation
        current_accel = self.calculate_acceleration(throttle=throttle_override)

        # Restore original engine method if modified
        if engine_factor_override < 1.0 and original_engine_get_torque and callable(original_engine_get_torque):
             self.engine.get_torque = original_engine_get_torque

        # 2. Update Kinematics
        self.current_speed_mps += current_accel * dt
        # Prevent backward movement unless intended (e.g., reverse gear)
        if self.current_speed_mps < 0 and self.current_gear >= 0: # Check if not in reverse
            self.current_speed_mps = 0.0
            # Only reset accel if speed is zero AND net force is negative
            if self.current_speed_mps == 0.0 and current_accel < 0.0:
                self.current_acceleration_mpss = 0.0
        else:
             self.current_acceleration_mpss = current_accel # Store the calculated accel

        self.current_position_m += self.current_speed_mps * dt

        # 3. Update Engine RPM (based on new speed and CURRENT gear)
        # Gear change only takes effect *after* completion.
        if self.current_gear > 0 and self.drivetrain:
            self.current_engine_rpm = self.drivetrain.calculate_engine_speed_rpm(self.current_speed_mps, self.current_gear)
            if self.engine: self.current_engine_rpm = np.clip(self.current_engine_rpm, self.engine.idle_rpm, self.engine.redline_rpm)
        elif self.engine:
             idle = self.engine.idle_rpm
             decay_rate = 5000.0 # RPM/s decay rate in neutral
             self.current_engine_rpm = max(idle, self.current_engine_rpm - decay_rate * dt)

        # Update engine's internal RPM state if exists
        if hasattr(self.engine, 'current_rpm'):
            self.engine.current_rpm = self.current_engine_rpm

        # 4. Update Thermal State
        if self.include_thermal:
             self.update_thermal_state(dt, ambient_temp_C)

    def simulate_acceleration_run(self, distance: float = FS_ACCELERATION_LENGTH,
                                max_time: float = 10.0, dt: float = 0.01,
                                use_launch_control: bool = True,
                                use_optimized_shifts: bool = True) -> Dict:
        """Simulate a standard acceleration run using AccelerationSimulator."""
        logger.info("Running simulate_acceleration_run within Vehicle class...")
        if not AccelerationSimulator_available:
            logger.error("AccelerationSimulator class not available. Cannot simulate.")
            return {'error': 'AccelerationSimulator not available'}
        try:
            vehicle_copy = copy.deepcopy(self)
            if CorneringPerformance_available and vehicle_copy.cornering is None:
                 vehicle_copy.cornering = CorneringPerformance(vehicle_copy)
            accel_sim = AccelerationSimulator(vehicle_copy)
            accel_sim.configure(distance_m=distance, time_step_s=dt, max_time_s=max_time)
            accel_sim.configure_launch_control(use_traction_control=True)
            accel_sim.configure_shifting(use_optimized=use_optimized_shifts)
            results = accel_sim.simulate_acceleration(use_launch_control=use_launch_control)
            metrics = accel_sim.analyze_performance_metrics(results)
            results.update(metrics)
            return results
        except Exception as e:
             logger.error(f"Error during AccelerationSimulation: {e}", exc_info=True)
             return {'error': str(e)}

    def simulate_skidpad(self, circle_radius_m: float = FS_SKIDPAD_RADIUS,
                       target_gear: int = 2, max_laps: int = 4, dt: float = 0.01) -> Dict:
        """Simulate a skidpad event (constant radius cornering)."""
        logger.info("Running simulate_skidpad within Vehicle class...")
        if not self.cornering:
             logger.error("Cannot simulate skidpad: CorneringPerformance calculator not available.")
             return {'error': 'CorneringPerformance unavailable'}

        # --- Reset ---
        initial_state = copy.deepcopy(self)
        self.current_speed_mps = 0.0
        self.current_position_m = 0.0
        self.change_gear(target_gear)
        self.current_engine_rpm = self.engine.idle_rpm if self.engine else 1300.0

        lap_angle = 0.0
        lap_times = []
        t = 0.0
        max_time = max_laps * 15.0
        history = {'time': [0.0], 'speed': [0.0], 'lat_g': [0.0], 'lon_accel': [0.0], 'rpm': [self.current_engine_rpm]}

        # --- Simulation Loop ---
        while t < max_time and len(lap_times) < max_laps:
            max_corner_speed = self.cornering.calculate_max_cornering_speed(circle_radius_m)
            speed_error = max_corner_speed - self.current_speed_mps
            throttle = np.clip(0.4 + speed_error * 0.5, 0.1, 0.9)
            brake = 0.0
            if self.current_speed_mps > max_corner_speed * 1.01:
                 throttle = 0.0
                 brake = 0.3

            # Update vehicle state using its own method
            self.throttle_input = throttle
            self.brake_input = brake
            self.update_vehicle_state(dt) # This updates speed, accel, rpm etc.

            lon_accel = self.current_acceleration_mpss
            angular_vel = self.current_speed_mps / circle_radius_m if circle_radius_m > 0 else 0
            lap_angle += angular_vel * dt
            t += dt

            history['time'].append(t)
            history['speed'].append(self.current_speed_mps)
            history['lat_g'].append((self.current_speed_mps**2 / circle_radius_m) / GRAVITY if circle_radius_m > 0 else 0)
            history['lon_accel'].append(lon_accel)
            history['rpm'].append(self.current_engine_rpm)

            if lap_angle >= 2 * np.pi:
                 lap_time = t - sum(lap_times)
                 lap_times.append(lap_time)
                 lap_angle -= 2 * np.pi
                 logger.debug(f"Skidpad lap {len(lap_times)} completed in {lap_time:.3f}s")

        # --- Results ---
        avg_lap_time = np.mean(lap_times) if lap_times else None
        max_lat_g = np.max(history['lat_g']) if history['lat_g'] else None
        results = {'average_lap_time': avg_lap_time, 'lap_times': lap_times, 'max_lateral_g': max_lat_g, 'history': {k: np.array(v) for k,v in history.items()}}
        logger.info(f"Skidpad simulation finished. Avg Lap: {avg_lap_time:.3f}s, Max Lateral G: {max_lat_g:.3f}g")
        self.__dict__.update(initial_state.__dict__) # Restore state
        return results

    def simulate_lap(self, track_file: str, include_thermal: bool = True) -> Dict:
         """Simulate a single lap using the LapTimeSimulator."""
         logger.info("Running simulate_lap via LapTimeSimulator...")
         if not LapTimeSimulator_available:
             logger.error("LapTimeSimulator class not available. Cannot simulate.")
             return {'error': 'LapTimeSimulator not available'}
         try:
             vehicle_copy = copy.deepcopy(self)
             if CorneringPerformance_available and vehicle_copy.cornering is None:
                  vehicle_copy.cornering = CorneringPerformance(vehicle_copy)
             lap_sim = LapTimeSimulator(vehicle_copy, track_file=track_file)
             if lap_sim.track_data is None: return {'error': f"Failed to load track file {track_file}"}
             results = lap_sim.simulate_lap(include_thermal=include_thermal)
             metrics = lap_sim.analyze_lap_performance(results)
             results['metrics'] = metrics
             return results
         except Exception as e:
              logger.error(f"Error during LapTimeSimulation: {e}", exc_info=True)
              return {'error': str(e)}

    def calculate_weight_transfer(self, longitudinal_accel_mpss: float = 0.0, lateral_accel_mpss: float = 0.0) -> Dict:
        """Calculate longitudinal and lateral weight transfer."""
        long_transfer = (self.mass * longitudinal_accel_mpss * self.cg_height_m) / self.wheelbase_m if self.wheelbase_m > 0 else 0
        lat_transfer = (self.mass * abs(lateral_accel_mpss) * self.cg_height_m) / self.track_width_rear_m if self.track_width_rear_m > 0 else 0
        static_front_N = self.mass * GRAVITY * self.weight_distribution_front
        static_rear_N = self.mass * GRAVITY * (1.0 - self.weight_distribution_front)
        dynamic_front_N = static_front_N - long_transfer
        dynamic_rear_N = static_rear_N + long_transfer
        # Simple split of lateral transfer based on weight distribution (can be refined)
        lat_transfer_front = lat_transfer * self.weight_distribution_front
        lat_transfer_rear = lat_transfer * (1.0 - self.weight_distribution_front)
        dynamic_FL_N = dynamic_front_N / 2.0 + lat_transfer_front / 2.0
        dynamic_FR_N = dynamic_front_N / 2.0 - lat_transfer_front / 2.0
        dynamic_RL_N = dynamic_rear_N / 2.0 + lat_transfer_rear / 2.0
        dynamic_RR_N = dynamic_rear_N / 2.0 - lat_transfer_rear / 2.0
        return {
            'longitudinal_transfer_N': long_transfer, 'lateral_transfer_N': lat_transfer,
            'dynamic_front_axle_load_N': dynamic_front_N, 'dynamic_rear_axle_load_N': dynamic_rear_N,
            'dynamic_FL_wheel_load_N': max(0, dynamic_FL_N), 'dynamic_FR_wheel_load_N': max(0, dynamic_FR_N),
            'dynamic_RL_wheel_load_N': max(0, dynamic_RL_N), 'dynamic_RR_wheel_load_N': max(0, dynamic_RR_N),
        }

    def get_vehicle_specs(self) -> Dict:
        """Return a comprehensive dictionary of vehicle specifications."""
        specs = {'team_name': self.team_name, 'vehicle': {}, 'tires': {}}
        vehicle_attrs = ['mass', 'frontal_area_m2', 'drag_coefficient', 'lift_coefficient', 'rolling_resistance_coeff',
                         'weight_distribution_front', 'wheelbase_m', 'track_width_front_m', 'track_width_rear_m', 'cg_height_m']
        for attr in vehicle_attrs: specs['vehicle'][attr] = getattr(self, attr, None)
        specs['tires']['radius_m'] = getattr(self, 'tire_radius_m', None)

        if self.engine and hasattr(self.engine, 'get_engine_specs'): specs['engine'] = self.engine.get_engine_specs()
        if self.drivetrain and hasattr(self.drivetrain, 'get_drivetrain_specs'): specs['drivetrain'] = self.drivetrain.get_drivetrain_specs()
        if self.cooling_system and hasattr(self.cooling_system, 'get_system_specs'): specs['cooling'] = self.cooling_system.get_system_specs()
        if self.side_pods and hasattr(self.side_pods, 'get_system_specs'): specs['side_pods'] = self.side_pods.get_system_specs()
        if self.rear_radiator and hasattr(self.rear_radiator, 'get_system_specs'): specs['rear_radiator'] = self.rear_radiator.get_system_specs()
        if self.cooling_assist and hasattr(self.cooling_assist, 'get_system_specs'): specs['cooling_assist'] = self.cooling_assist.get_system_specs()
        if self.cas_system and hasattr(self.cas_system, 'get_status'): specs['cas'] = self.cas_system.get_status()
        return specs

    def calculate_performance_metrics(self) -> Dict:
        """Calculate key theoretical performance metrics."""
        metrics = {}
        if self.engine and self.mass > 0 and hasattr(self.engine, 'max_power_hp'):
             power_kw = self.engine.max_power_hp * HP_TO_KW
             metrics['power_to_weight_kw_kg'] = power_kw / self.mass
             metrics['power_to_weight_hp_kg'] = self.engine.max_power_hp / self.mass
        try:
             max_speed = self.calculate_max_speed()
             metrics['max_speed_mps'] = max_speed
             metrics['max_speed_kph'] = max_speed * MS_TO_KMH
        except Exception as e: metrics['max_speed_mps'] = None
        if self.cornering:
             try:
                 max_lat_g = self.cornering.calculate_max_lateral_acceleration(speed_mps=20.0) / GRAVITY
                 metrics['max_lateral_g'] = max_lat_g
             except Exception as e: metrics['max_lateral_g'] = None
        return metrics

    def calculate_max_speed(self) -> float:
        """Estimate theoretical maximum speed where tractive force equals drag+rolling."""
        if not self.engine or not self.drivetrain or not hasattr(self.engine, 'get_torque'):
            logger.warning("Cannot calculate max speed: Missing engine or drivetrain.")
            return 0.0

        top_gear = self.drivetrain.num_gears
        rpm_limit = self.engine.redline_rpm
        guess_speed_mps = 50.0

        for _ in range(10):
            rpm_at_guess = self.drivetrain.calculate_engine_speed_rpm(guess_speed_mps, top_gear)
            if rpm_at_guess > rpm_limit:
                 speed_at_redline = self.drivetrain.calculate_vehicle_speed_mps(rpm_limit, top_gear)
                 guess_speed_mps = speed_at_redline # RPM limited, use this speed
                 continue

            engine_torque = self.engine.get_torque(rpm_at_guess, throttle=1.0)
            f_tractive = self.calculate_tractive_force_N(engine_torque, top_gear)

            # Need forces at the current guess speed
            temp_speed = self.current_speed_mps # Store current speed
            self.current_speed_mps = guess_speed_mps # Temporarily set speed
            forces = self.calculate_forces()
            self.current_speed_mps = temp_speed # Restore speed
            f_resist = forces['drag'] + forces['rolling']

            force_diff = f_tractive - f_resist
            if abs(force_diff) < 1.0: break

            # Adjust guess based on which force is larger
            if f_tractive > f_resist: guess_speed_mps *= 1.05 # Increase guess
            else: guess_speed_mps *= 0.95 # Decrease guess
            guess_speed_mps = max(0.1, guess_speed_mps)
        else:
            logger.warning(f"Max speed calculation did not fully converge. Final guess: {guess_speed_mps:.1f} m/s")
        return guess_speed_mps

    def plot_acceleration_results(self, results: Dict, save_path: Optional[str] = None):
         """Plot acceleration results using the utility function."""
         if plotting_utils:
             fig = plotting_utils.plot_acceleration_results(results, save_path=save_path, plot_wheel_slip=True)
             # if fig: plt.close(fig) # Optional: Close plot after saving/showing
         else:
             logger.warning("Plotting utilities not available.")


# --- Factory Function ---
def create_formula_student_vehicle(config_path: Optional[str] = None, config_dict: Optional[Dict] = None) -> 'Vehicle':
    """
    Factory function to create a Vehicle instance representing a typical FS car.

    It initializes the Vehicle class, which in turn handles loading component
    configurations based on paths defined internally or within the provided
    config_path/config_dict.

    Args:
        config_path: Optional path to a main vehicle config file containing paths
                     to component configs or component parameters themselves.
                     (Often handled by ConfigurationManager before calling this).
        config_dict: Optional pre-loaded configuration dictionary. Takes precedence
                     over config_path if both are provided.

    Returns:
        A configured Vehicle instance.

    Raises:
        RuntimeError: If vehicle creation fails critically.
    """
    # Use a logger specific to the factory or the vehicle module
    factory_logger = logging.getLogger("VehicleFactory") # Or use logger=logging.getLogger("Vehicle")
    factory_logger.info("Creating default Formula Student Vehicle instance...")
    try:
        # Instantiate the Vehicle class.
        # Pass config_dict if provided, otherwise pass config_path.
        # The Vehicle constructor's logic handles loading and component init.
        vehicle = Vehicle(config=config_dict, config_path=config_path)

        # Optional: Perform post-instantiation checks or setup if needed
        # For example, ensure the cornering calculator is initialized if it wasn't in __init__
        if not hasattr(vehicle, 'cornering') or vehicle.cornering is None:
             # This might indicate an issue during Vehicle.__init__ if it was supposed
             # to be created there. Adding it here ensures it exists.
             # Need to import CorneringPerformance locally if not already imported at module level
             try:
                 from ..performance.lap_time import CorneringPerformance
                 factory_logger.debug("Initializing CorneringPerformance in factory function.")
                 vehicle.cornering = CorneringPerformance(vehicle)
             except ImportError:
                 factory_logger.error("Could not import CorneringPerformance to initialize in factory.")

        factory_logger.info("Default Formula Student Vehicle instance created successfully.")
        return vehicle
    except Exception as e:
        factory_logger.critical(f"Failed to create Formula Student vehicle: {e}", exc_info=True)
        # Raising might be better than returning None, as the simulation
        # likely cannot proceed without a vehicle.
        raise RuntimeError(f"Failed to create Formula Student vehicle: {e}") from e
# Example Usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Vehicle Module Demo")
    print("-" * 20)

    try:
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
                # Plotting is handled within simulate_acceleration_run if save_path is provided
                # Optionally call plot here if needed for direct display
                # vehicle.plot_acceleration_results(accel_results)
            else:
                 print(f" Acceleration simulation failed: {accel_results.get('error', 'Unknown error')}")
        else:
            print(" Acceleration simulation skipped: Simulator not available.")

        # --- Example Skidpad Run ---
        print("\n--- Simulating Skidpad Run ---")
        skidpad_results = vehicle.simulate_skidpad()
        if skidpad_results and 'error' not in skidpad_results:
             print(f" Avg Lap Time: {skidpad_results.get('average_lap_time', -1):.3f} s")
             print(f" Max Lateral G: {skidpad_results.get('max_lateral_g', -1):.3f} g")
        else:
            print(f" Skidpad simulation failed: {skidpad_results.get('error', 'Unknown error')}")


        print("\nVehicle demo finished.")

    except ImportError as e:
        print(f"Import Error during demo: {e}")
    except FileNotFoundError as e:
         print(f"Config file not found: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()