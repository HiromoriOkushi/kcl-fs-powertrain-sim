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

# Import powertrain components (handle potential errors)
try:
    from ..engine.motorcycle_engine import MotorcycleEngine
    from ..engine.engine_thermal import EngineHeatModel, ThermalConfig # Assuming thermal model is here
    from ..transmission.gearing import DrivetrainSystem, Transmission, FinalDrive, Differential
    from ..transmission.cas_system import CASSystem, ShiftDirection
    from ..transmission.shift_strategy import StrategyManager, create_formula_student_strategies, StrategyType
    # Import primary cooling system interface (adjust if needed)
    # Alias ExternalCoolingSystem to avoid name clash with EngineCoolingSystemComponent if both exist
    from ..thermal.cooling_system import CoolingSystem as ExternalCoolingSystem
    from ..thermal.cooling_system import create_formula_student_cooling_system # Import factory
    from ..thermal.side_pod import DualSidePodSystem, create_standard_side_pod_system
    from ..thermal.rear_radiator import RearRadiatorSystem, create_default_rear_radiator_system
    from ..thermal.electric_compressor import CoolingAssistSystem, create_default_cooling_assist_system
    from ..utils.constants import GRAVITY, AIR_DENSITY_SEA_LEVEL, KW_TO_HP, HP_TO_KW, MS_TO_KMH, MS_TO_MPH
    from ..utils.plotting import plot_vehicle_performance_summary, plot_acceleration_results as plot_accel_results_util, save_plot
    from ..performance.lap_time import CorneringPerformance # For cornering calcs
except ImportError as e:
    # Define placeholders if imports fail (e.g., for testing)
    logger = logging.getLogger("Vehicle_Fallback")
    logger.error(f"Error importing vehicle components: {e}. Using placeholders.")
    class MotorcycleEngine: pass
    class DrivetrainSystem: pass
    class Transmission: pass
    class FinalDrive: pass
    class Differential: pass
    class CASSystem: pass
    class StrategyManager: pass
    class ExternalCoolingSystem: pass
    class DualSidePodSystem: pass
    class RearRadiatorSystem: pass
    class CoolingAssistSystem: pass
    class EngineHeatModel: pass
    class ThermalConfig: pass
    class CorneringPerformance: pass
    def create_formula_student_strategies(*args, **kwargs): return None
    def create_standard_side_pod_system(*args, **kwargs): return None
    def create_default_rear_radiator_system(*args, **kwargs): return None
    def create_default_cooling_assist_system(*args, **kwargs): return None
    def create_formula_student_cooling_system(*args, **kwargs): return None
    def plot_vehicle_performance_summary(*args, **kwargs): plt.figure(); plt.plot([0,1]); plt.title("Fallback Plot"); plt.show(); plt.close(); return plt.gcf()
    def plot_accel_results_util(*args, **kwargs): plt.figure(); plt.plot([0,1]); plt.title("Fallback Plot"); plt.show(); plt.close(); return plt.gcf()
    def save_plot(fig, path, **kwargs): pass
    GRAVITY = 9.81; AIR_DENSITY_SEA_LEVEL = 1.225; KW_TO_HP = 1.341; HP_TO_KW = 1/KW_TO_HP; MS_TO_KMH = 3.6; MS_TO_MPH = 2.237

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
            config_path: Path to the main vehicle YAML configuration file.
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

        # Load config first
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
        elif config_path:
            logger.warning(f"Vehicle config file not found: {config_path}. Using defaults.")

        # --- Component Initialization ---
        self._initialize_engine(engine)
        self._initialize_drivetrain(drivetrain)
        # Pass self to cooling system init for potential back-references if needed by factories
        self._initialize_cooling_system(cooling_system, self)
        self._initialize_shifting_systems(shift_manager, cas_system)
        self._initialize_aero_cooling(side_pods, rear_radiator, cooling_assist)

        # Initialize Cornering Performance calculator after core components
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

        # Initialize thermal state from engine if possible
        self.engine_temperature = getattr(self.engine, 'engine_temperature', 25.0)
        self.coolant_temperature = getattr(self.engine, 'coolant_temperature', 25.0)
        self.oil_temperature = getattr(self.engine, 'oil_temperature', 25.0)

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


    def _initialize_engine(self, engine_instance: Optional[MotorcycleEngine]):
        """Initialize the engine component."""
        if isinstance(engine_instance, MotorcycleEngine):
            self.engine = engine_instance
            logger.info("Using pre-configured Engine instance.")
        else:
            engine_config_ref = self.config.get('engine')
            if isinstance(engine_config_ref, str) and os.path.exists(engine_config_ref):
                logger.info(f"Initializing Engine from config file: {engine_config_ref}")
                self.engine = MotorcycleEngine(config_path=engine_config_ref)
            elif isinstance(engine_config_ref, dict):
                logger.info("Initializing Engine from inline config.")
                self.engine = MotorcycleEngine(engine_params=engine_config_ref)
            else:
                logger.warning("No valid engine config found. Creating default MotorcycleEngine.")
                self.engine = MotorcycleEngine()

        # Ensure essential attributes
        for attr, default in [('idle_rpm', 1300.0), ('redline_rpm', 14000.0),
                              ('max_torque_nm', 65.0), ('engine_temperature', 25.0),
                              ('coolant_temperature', 25.0), ('oil_temperature', 25.0),
                              ('thermal_factor', 1.0)]:
            if not hasattr(self.engine, attr): setattr(self.engine, attr, default)


    def _initialize_drivetrain(self, drivetrain_instance: Optional[DrivetrainSystem]):
        """Initialize the drivetrain component."""
        if isinstance(drivetrain_instance, DrivetrainSystem):
            self.drivetrain = drivetrain_instance
            logger.info("Using pre-configured Drivetrain instance.")
        else:
             # Try loading from config
             dt_config_ref = self.config.get('drivetrain')
             trans_config_ref = self.config.get('transmission') # Allow separate file/dict
             fd_config_ref = self.config.get('final_drive')
             diff_config_ref = self.config.get('differential')

             transmission = None
             final_drive = None
             differential = None

             # Load Transmission
             if isinstance(trans_config_ref, str) and os.path.exists(trans_config_ref):
                  with open(trans_config_ref, 'r') as f: trans_params = yaml.safe_load(f).get('transmission', {})
                  transmission = Transmission(**trans_params)
             elif isinstance(trans_config_ref, dict):
                  transmission = Transmission(**trans_config_ref)
             else: # Default
                  transmission = Transmission([2.750, 2.000, 1.667, 1.444, 1.304, 1.208])

             # Load Final Drive
             if isinstance(fd_config_ref, str) and os.path.exists(fd_config_ref):
                  with open(fd_config_ref, 'r') as f: fd_params = yaml.safe_load(f).get('final_drive', {})
                  final_drive = FinalDrive(**fd_params)
             elif isinstance(fd_config_ref, dict):
                  final_drive = FinalDrive(**fd_config_ref)
             else: # Default
                  final_drive = FinalDrive(14, 53)

             # Load Differential
             if isinstance(diff_config_ref, str) and os.path.exists(diff_config_ref):
                 with open(diff_config_ref, 'r') as f: diff_params = yaml.safe_load(f).get('differential', {})
                 differential = Differential(**diff_params)
             elif isinstance(diff_config_ref, dict):
                 differential = Differential(**diff_config_ref)
             else: # Default
                 differential = Differential(locked=True)

             self.drivetrain = DrivetrainSystem(transmission, final_drive, differential, self.tire_radius_m)
             logger.info("Drivetrain initialized from config/defaults.")

        if not hasattr(self.drivetrain, 'num_gears'): # Ensure attribute exists
             self.drivetrain.num_gears = len(getattr(self.drivetrain.transmission, 'gear_ratios', []))


    def _initialize_cooling_system(self, cooling_instance: Optional[ExternalCoolingSystem], vehicle_ref):
        """Initialize the main external cooling system."""
        if isinstance(cooling_instance, ExternalCoolingSystem):
            self.cooling_system = cooling_instance
            logger.info("Using pre-configured external CoolingSystem instance.")
        else:
            cooling_config_ref = self.config.get('cooling_system')
            if isinstance(cooling_config_ref, str) and os.path.exists(cooling_config_ref):
                logger.info(f"Initializing external CoolingSystem from config file: {cooling_config_ref}")
                # Assume config file contains component details or use factory
                self.cooling_system = create_formula_student_cooling_system(config_dir=os.path.dirname(cooling_config_ref))
            elif isinstance(cooling_config_ref, dict):
                 logger.info("Initializing external CoolingSystem from inline config.")
                 # Manually create components from dict - requires component classes
                 try:
                      rad_cfg = cooling_config_ref.get('radiator', {})
                      pump_cfg = cooling_config_ref.get('water_pump', {})
                      fan_cfg = cooling_config_ref.get('cooling_fan', {})
                      thermo_cfg = cooling_config_ref.get('thermostat', {})
                      system_cfg = cooling_config_ref.get('system', {})

                      from ..thermal.cooling_system import Radiator, WaterPump, CoolingFan, Thermostat
                      radiator = Radiator(**rad_cfg) if rad_cfg else Radiator()
                      pump = WaterPump(**pump_cfg) if pump_cfg else WaterPump()
                      fan = CoolingFan(**fan_cfg) if fan_cfg else None # Fan optional
                      thermostat = Thermostat(**thermo_cfg) if thermo_cfg else Thermostat()

                      self.cooling_system = ExternalCoolingSystem(radiator, pump, fan, thermostat, **system_cfg)
                 except Exception as e:
                      logger.error(f"Failed to create cooling system from inline config: {e}. Creating default.")
                      self.cooling_system = create_formula_student_cooling_system()
            else:
                logger.info("No cooling system config found. Creating default FS cooling system.")
                self.cooling_system = create_formula_student_cooling_system()

        # Ensure essential cooling system attributes exist
        if not hasattr(self.cooling_system, 'coolant_temp_C'): self.cooling_system.coolant_temp_C = 25.0


    def _initialize_shifting_systems(self, manager_instance: Optional[StrategyManager], cas_instance: Optional[CASSystem]):
        """Initialize shift manager and CAS system."""
        if isinstance(manager_instance, StrategyManager):
            self.shift_manager = manager_instance
            logger.info("Using pre-configured StrategyManager instance.")
        else:
            # Create default FS strategies
            if self.engine and self.drivetrain:
                 self.shift_manager = create_formula_student_strategies(
                     engine_max_rpm=self.engine.redline_rpm,
                     engine_peak_power_rpm=self.engine.max_power_rpm,
                     engine_peak_torque_rpm=self.engine.max_torque_rpm,
                     gear_ratios=self.drivetrain.transmission.gear_ratios,
                     num_gears=self.drivetrain.num_gears, # Pass num_gears
                     idle_rpm=self.engine.idle_rpm
                 )
                 # Optionally load strategy config to customize points
                 strat_cfg_ref = self.config.get('shift_strategy')
                 if isinstance(strat_cfg_ref, str) and os.path.exists(strat_cfg_ref):
                      self.shift_manager.load_strategies_from_config(strat_cfg_ref)
                 logger.info("Initialized default StrategyManager with FS strategies.")
            else:
                 logger.warning("Cannot initialize StrategyManager: Engine or Drivetrain missing.")
                 self.shift_manager = None

        if isinstance(cas_instance, CASSystem):
            self.cas_system = cas_instance
            logger.info("Using pre-configured CASSystem instance.")
        elif self.drivetrain and self.engine:
            # Load CAS config if available
            cas_cfg_ref = self.config.get('cas_system', self.config.get('transmission')) # Check both potential locations
            cas_config_path = None
            if isinstance(cas_cfg_ref, str) and os.path.exists(cas_cfg_ref):
                 cas_config_path = cas_cfg_ref
            elif isinstance(cas_cfg_ref, dict) and 'cas' in cas_cfg_ref:
                 # If CAS params are nested within transmission config
                  cas_config_path = self.config_path # Pass main vehicle config to load nested dict

            self.cas_system = CASSystem(
                gear_ratios=self.drivetrain.transmission.gear_ratios,
                engine=self.engine,
                config_path=cas_config_path
            )
            logger.info("Initialized CASSystem.")
        else:
             logger.warning("Cannot initialize CASSystem: Engine or Drivetrain missing.")
             self.cas_system = None

    def _initialize_aero_cooling(self, side_pods_instance, rear_rad_instance, assist_instance):
        """Initialize optional side pods, rear radiator, cooling assist."""
        if isinstance(side_pods_instance, DualSidePodSystem):
            self.side_pods = side_pods_instance
        elif self.config.get('side_pods'):
             # Logic to load/create from config
             self.side_pods = create_standard_side_pod_system() # Use factory
             logger.info("Initialized default DualSidePodSystem.")
        else: self.side_pods = None

        if isinstance(rear_rad_instance, RearRadiatorSystem):
            self.rear_radiator = rear_rad_instance
        elif self.config.get('rear_radiator'):
             self.rear_radiator = create_default_rear_radiator_system()
             logger.info("Initialized default RearRadiatorSystem.")
        else: self.rear_radiator = None

        if isinstance(assist_instance, CoolingAssistSystem):
             self.cooling_assist = assist_instance
        elif self.config.get('cooling_assist'):
             self.cooling_assist = create_default_cooling_assist_system()
             logger.info("Initialized default CoolingAssistSystem.")
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

    def update_thermal_state(self, dt: float, ambient_temp_C: float):
         """Update the thermal state of the engine and cooling system."""
         if not self.engine or not self.cooling_system: return

         # 1. Update Engine Internal Thermal State (using its own method)
         # This calculates heat generation and internal transfers based on current op point
         if hasattr(self.engine, 'update_thermal_state') and callable(self.engine.update_thermal_state):
             # Pass necessary external info to engine's thermal update
             # We need cooling effectiveness from the external system
             # Placeholder: Estimate effectiveness based on speed/fan (improve this)
             cooling_effectiveness_est = 0.5 + 0.5 * np.clip(self.current_speed_mps / 20.0, 0, 1)
             if self.cooling_system.cooling_fan and self.cooling_system.cooling_fan.is_active:
                  cooling_effectiveness_est = max(cooling_effectiveness_est, 0.8) # Fan boost

             engine_temps = self.engine.update_thermal_state(
                 ambient_temp=ambient_temp_C,
                 cooling_effectiveness=cooling_effectiveness_est, # Pass estimated effectiveness
                 dt=dt
             )
             # Update vehicle's temperature mirrors
             self.engine_temperature = engine_temps.get('engine_temp', self.engine_temperature)
             self.coolant_temperature = engine_temps.get('coolant_temp', self.coolant_temperature)
             self.oil_temperature = engine_temps.get('oil_temp', self.oil_temperature)

         # 2. Update External Cooling System State
         # Provide engine heat input to the external system
         # This requires the engine model to estimate heat *to the coolant*
         heat_to_coolant_W = 0.0
         if hasattr(self.engine, 'heat_model') and hasattr(self.engine.heat_model, 'calculate_heat_generation'):
              heat_gen = self.engine.heat_model.calculate_heat_generation(self.current_engine_rpm, self.throttle_input)
              heat_to_coolant_W = heat_gen.get('to_coolant', 0.0)
         elif self.engine: # Estimate if no detailed model
             power_kw = self.engine.get_power(self.current_engine_rpm, self.throttle_input)
             heat_to_coolant_W = power_kw * 1000 * 1.5 # Rough estimate: 1.5x power is heat to coolant

         # Update the external cooling system
         self.cooling_system.simulate_step(
              ambient_temp_C=ambient_temp_C,
              vehicle_speed_mps=self.current_speed_mps,
              engine_temp=self.engine_temperature, # Pass engine block temp
              engine_rpm=self.current_engine_rpm,
              engine_load=self.throttle_input, # Use throttle as load proxy
              engine_heat_input_W=heat_to_coolant_W,
              dt=dt
         )
         # Update vehicle's coolant temp mirror from the system's result
         self.coolant_temperature = self.cooling_system.coolant_temp_C


    def change_gear(self, target_gear: int) -> bool:
         """Request a gear change via the CAS system if available, else directly."""
         if self.cas_system:
             direction = ShiftDirection.NEUTRAL
             if target_gear > self.current_gear: direction = ShiftDirection.UP
             elif target_gear < self.current_gear: direction = ShiftDirection.DOWN
             success = self.cas_system.request_shift(direction, target_gear)
             # Update vehicle's current gear if CAS succeeded (CAS updates its internal state)
             if success: self.current_gear = self.cas_system.current_gear
             return success
         elif self.drivetrain:
              success = self.drivetrain.change_gear(target_gear)
              if success: self.current_gear = target_gear
              return success
         else:
              logger.error("Cannot change gear: No CAS or Drivetrain system.")
              return False

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

    def simulate_acceleration_run(self, distance: float = FS_ACCELERATION_LENGTH,
                                max_time: float = 10.0, dt: float = 0.01,
                                use_launch_control: bool = True,
                                use_optimized_shifts: bool = True) -> Dict:
        """Simulate a standard acceleration run."""
        logger.info("Running simulate_acceleration_run within Vehicle class...")
        # Use the dedicated AccelerationSimulator for consistency
        accel_sim = AccelerationSimulator(copy.deepcopy(self)) # Simulate on a copy
        accel_sim.configure(distance_m=distance, time_step_s=dt, max_time_s=max_time)
        accel_sim.configure_launch_control(use_traction_control=True) # Use default LC params initially
        accel_sim.configure_shifting(use_optimized=use_optimized_shifts)
        results = accel_sim.simulate_acceleration(use_launch_control=use_launch_control)
        return results

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
         # Use the dedicated LapTimeSimulator for consistency
         lap_sim = LapTimeSimulator(copy.deepcopy(self), track_file=track_file)
         results = lap_sim.simulate_lap(include_thermal=include_thermal)
         # Add metrics for convenience
         results['metrics'] = lap_sim.analyze_lap_performance(results)
         return results

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