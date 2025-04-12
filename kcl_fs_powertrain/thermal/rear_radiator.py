"""
Rear-mounted radiator module for Formula Student powertrain simulation.

This module extends the cooling system functionality to specifically model
rear-mounted radiator configurations in Formula Student cars. It provides
specialized airflow modeling, ducting optimization, and integration with the
car's aerodynamic package, particularly diffusers and bodywork.

Rear-mounted radiators offer potential advantages in weight distribution,
packaging, and aerodynamic performance, but present unique challenges in
achieving adequate airflow and managing heat soak in low-speed conditions.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Callable
import logging
from enum import Enum, auto
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("RearRadiator")

# Import base components (handle potential import errors if run standalone)
try:
    from .cooling_system import Radiator, RadiatorType, CoolingFan, FanType
except ImportError:
    # Placeholders if run directly
    class Radiator: pass
    class CoolingFan: pass
    class RadiatorType(Enum): SINGLE_CORE_ALUMINUM=auto(); DOUBLE_CORE_ALUMINUM=auto()
    class FanType(Enum): VARIABLE_SPEED=auto(); DUAL_FAN=auto(); SINGLE_SPEED=auto()
    logger.warning("Could not import base cooling system components. Using placeholders.")

# Constants (import or define fallback)
try:
    from ..utils.constants import AIR_DENSITY_SEA_LEVEL, AIR_SPECIFIC_HEAT_CP, BAR_TO_PA, LITERS_TO_M3
    from ..utils.plotting import save_plot, _apply_common_ax_settings, COLOR_SCHEMES # Import plotting utils
except ImportError:
    AIR_DENSITY_SEA_LEVEL = 1.225
    AIR_SPECIFIC_HEAT_CP = 1005.0
    BAR_TO_PA = 100000.0
    LITERS_TO_M3 = 0.001
    # Fallback plotting utils
    def save_plot(fig, path, **kwargs): pass
    def _apply_common_ax_settings(ax, **kwargs): pass
    COLOR_SCHEMES = {'default': plt.cm.tab10.colors}
    logger.warning("Could not import utils.constants or utils.plotting. Using fallback values/functions.")



class MountingPosition(Enum):
    """Possible rear radiator mounting positions."""
    ABOVE_DIFFUSER = auto()
    BEHIND_DRIVER = auto()
    SIDE_POD_REAR_EXIT = auto() # Integrated into sidepod rear
    ANGLED_UPWARD = auto()      # For natural convection
    CUSTOM = auto()

class DuctType(Enum):
    """Types of ducting feeding a rear radiator."""
    NACA_INLET = auto()
    SIDE_SCOOP = auto()
    TOP_INLET = auto()
    DIFFUSER_INTEGRATED = auto() # Air drawn from under diffuser
    CUSTOM = auto()

class RearRadiator:
    """Extends Radiator model for rear mounting specifics."""
    def __init__(self,
                 radiator: Radiator, # Base radiator object
                 mounting_position: MountingPosition = MountingPosition.ABOVE_DIFFUSER,
                 angle_degrees: float = 20.0, # Angle relative to vertical
                 config_path: Optional[str] = None,
                 custom_params: Optional[Dict] = None):
        """
        Initialize a rear-mounted radiator.

        Args:
            radiator: The base Radiator object.
            mounting_position: Where the radiator is mounted.
            angle_degrees: Installation angle relative to vertical (0=vertical).
            config_path: Optional path to YAML config file for rear-specific params.
            custom_params: Optional dict for CUSTOM type or overrides.
        """
        if not isinstance(radiator, Radiator):
             raise TypeError("Radiator must be an instance of the Radiator class.")
        self.base_radiator = radiator
        self.mounting_position = mounting_position
        self.angle_degrees = angle_degrees
        self.angle_radians = np.radians(angle_degrees)

        # Rear-specific airflow factors (defaults, can be overridden)
        self.base_airflow_efficiency = 0.7 # Base efficiency of capturing free-stream air
        self.low_speed_airflow_factor = 0.2 # Factor for natural convection/induced flow at low speed
        self.diffuser_interaction_factor = 1.0 # Multiplier if interacting with diffuser (can be >1 or <1)
        self.wake_effect_factor = 1.0 # Reduction factor due to driver/bodywork wake
        self.drag_coefficient_increase = 0.03 # Base increase in Cd

        # Load config or apply custom params
        rear_config = {}
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                rear_config = config.get('rear_radiator', {})
                # Safely get enum from name
                mount_pos_name = rear_config.get('mounting_position', self.mounting_position.name).upper()
                self.mounting_position = MountingPosition[mount_pos_name] if mount_pos_name in MountingPosition.__members__ else self.mounting_position
                self.angle_degrees = float(rear_config.get('angle_degrees', self.angle_degrees))
                self.angle_radians = np.radians(self.angle_degrees)
                logger.info(f"Rear radiator config loaded from {config_path}")
            except Exception as e:
                logger.error(f"Error loading rear radiator config from {config_path}: {e}. Using defaults.")
        if custom_params: rear_config.update(custom_params) # Custom overrides config file

        self._apply_config_and_defaults(rear_config)

        # Derived: Effective area presented to airflow (approx)
        self.effective_frontal_area = self.base_radiator.core_area * np.cos(self.angle_radians)

        logger.info(f"Rear Radiator initialized: Position={self.mounting_position.name}, Angle={self.angle_degrees:.1f}deg")

    def _apply_config_and_defaults(self, params: Dict):
        """Apply configuration and set defaults based on mounting position."""
        self.base_airflow_efficiency = float(params.get('base_airflow_efficiency', 0.7))
        self.low_speed_airflow_factor = float(params.get('low_speed_airflow_factor', 0.2))
        self.diffuser_interaction_factor = float(params.get('diffuser_interaction_factor', 1.0))
        self.wake_effect_factor = float(params.get('wake_effect_factor', 1.0))
        self.drag_coefficient_increase = float(params.get('drag_coefficient_increase', 0.03))

        # Adjust factors based on mounting position (illustrative)
        if self.mounting_position == MountingPosition.ABOVE_DIFFUSER:
             self.diffuser_interaction_factor = params.get('diffuser_interaction_factor', 1.1)
             self.wake_effect_factor = params.get('wake_effect_factor', 0.9)
             self.drag_coefficient_increase = params.get('drag_coefficient_increase', 0.04)
        elif self.mounting_position == MountingPosition.BEHIND_DRIVER:
             self.wake_effect_factor = params.get('wake_effect_factor', 0.7)
             self.drag_coefficient_increase = params.get('drag_coefficient_increase', 0.02)
        elif self.mounting_position == MountingPosition.ANGLED_UPWARD:
             self.low_speed_airflow_factor = params.get('low_speed_airflow_factor', 0.3)
             self.drag_coefficient_increase = params.get('drag_coefficient_increase', 0.05)
        elif self.mounting_position == MountingPosition.SIDE_POD_REAR_EXIT:
             self.wake_effect_factor = params.get('wake_effect_factor', 0.85)
             self.drag_coefficient_increase = params.get('drag_coefficient_increase', 0.035)


    def calculate_effective_airflow_velocity(self, vehicle_speed_mps: float) -> float:
        """Estimate the effective air velocity reaching the radiator face."""
        # Base velocity relative to free stream, considering wake and base efficiency
        base_velocity = vehicle_speed_mps * self.base_airflow_efficiency * self.wake_effect_factor

        # Add low speed contribution (more effect at lower speeds)
        # Use exponential decay instead of inverse for smoother behavior near zero
        low_speed_contribution = self.low_speed_airflow_factor * np.exp(-vehicle_speed_mps / 3.0) * 5.0 # Tunable parameters

        # Diffuser interaction modifies the effective velocity
        effective_velocity = (base_velocity + low_speed_contribution) * self.diffuser_interaction_factor

        return max(0.1, effective_velocity) # Ensure a small minimum positive velocity

    def calculate_airflow_m3s(self, vehicle_speed_mps: float, fan_airflow_m3s: float = 0.0) -> float:
        """Calculate total airflow (m³/s) through the radiator including fan assist."""
        effective_velocity = self.calculate_effective_airflow_velocity(vehicle_speed_mps)
        ram_air_flow = effective_velocity * self.effective_frontal_area

        # Combine ram air and fan air
        # Simple addition is a rough approximation. More complex models exist.
        # Consider if fan helps pull air at speed or fights ram air. Assume additive for now.
        total_airflow = ram_air_flow + fan_airflow_m3s
        return max(0.0, total_airflow)

    def calculate_heat_rejection(self, coolant_temp_C: float, ambient_temp_C: float,
                               coolant_flow_lpm: float, vehicle_speed_mps: float,
                               fan_airflow_m3s: float = 0.0) -> float:
        """Calculate heat rejection (W) considering rear mounting specifics."""
        total_airflow_m3s = self.calculate_airflow_m3s(vehicle_speed_mps, fan_airflow_m3s)

        # Use the base radiator's calculation method
        heat_rejection_W = self.base_radiator.calculate_heat_rejection(
            coolant_temp_C, ambient_temp_C, coolant_flow_lpm, total_airflow_m3s
        )
        return heat_rejection_W

    def get_drag_increase(self, vehicle_speed_mps: float, vehicle_frontal_area_m2: float = 1.1) -> float:
        """Estimate the drag force (N) added by this radiator installation."""
        dynamic_pressure = 0.5 * AIR_DENSITY_SEA_LEVEL * vehicle_speed_mps**2
        # Drag increase is applied to the vehicle's overall frontal area
        drag_force = self.drag_coefficient_increase * vehicle_frontal_area_m2 * dynamic_pressure
        return drag_force

    def get_radiator_specs(self) -> Dict:
        """Get combined specifications including rear mounting details."""
        base_specs = self.base_radiator.get_radiator_specs()
        rear_specs = {
            'mounting_position': self.mounting_position.name,
            'angle_degrees': self.angle_degrees,
            'effective_frontal_area_m2': self.effective_frontal_area,
            'base_airflow_efficiency': self.base_airflow_efficiency,
            'low_speed_airflow_factor': self.low_speed_airflow_factor,
            'diffuser_interaction_factor': self.diffuser_interaction_factor,
            'wake_effect_factor': self.wake_effect_factor,
            'drag_coefficient_increase': self.drag_coefficient_increase
        }
        return {**base_specs, **rear_specs}


class RearRadiatorDuct:
    """Models the ducting associated with a rear radiator."""
    def __init__(self,
                 duct_type: DuctType = DuctType.NACA_INLET,
                 inlet_area_m2: float = 0.03,
                 outlet_area_m2: float = 0.04, # Typically matches radiator area
                 length_m: float = 0.4,
                 efficiency: float = 0.85, # Overall pressure recovery efficiency
                 config_path: Optional[str] = None,
                 custom_params: Optional[Dict] = None):
        """
        Initialize the duct model.

        Args:
            duct_type: Type of duct inlet/design.
            inlet_area_m2: Area of the duct inlet (m²).
            outlet_area_m2: Area of the duct outlet (m²).
            length_m: Length of the duct (m).
            efficiency: Duct pressure recovery efficiency (0-1).
            config_path: Optional path to YAML config file.
            custom_params: Optional dictionary for CUSTOM type or overrides.
        """
        self.duct_type = duct_type
        self.inlet_area_m2 = inlet_area_m2
        self.outlet_area_m2 = outlet_area_m2
        self.length_m = length_m
        self.efficiency = efficiency # Pressure recovery efficiency

        # Duct drag coefficient (Cd based on inlet area) - Default, override below
        self.drag_coefficient = 0.05
        # Flow resistance coefficient (k where DeltaP = k * 0.5 * rho * v^2) - Default
        self.flow_resistance_k = 1.5

        # Load config or apply custom params
        duct_config = {}
        if config_path and os.path.exists(config_path):
             try:
                 with open(config_path, 'r') as f:
                     config = yaml.safe_load(f)
                 # Assuming duct config might be under 'rear_radiator' or a specific 'duct' key
                 duct_config = config.get('rear_duct', config.get('inlet_duct', {})) # Check multiple keys
                 # Safely get enum from name
                 duct_type_name = duct_config.get('duct_type', self.duct_type.name).upper()
                 self.duct_type = DuctType[duct_type_name] if duct_type_name in DuctType.__members__ else self.duct_type
                 self.inlet_area_m2 = float(duct_config.get('inlet_area_m2', self.inlet_area_m2))
                 self.outlet_area_m2 = float(duct_config.get('outlet_area_m2', self.outlet_area_m2))
                 self.length_m = float(duct_config.get('length_m', self.length_m))
                 self.efficiency = float(duct_config.get('efficiency', self.efficiency))
                 logger.info(f"Rear duct config loaded from {config_path}")
             except Exception as e:
                 logger.error(f"Error loading rear duct config from {config_path}: {e}. Using defaults.")
        if custom_params: duct_config.update(custom_params)

        self._apply_type_defaults_and_custom(duct_config)
        self.flow_resistance_k = self._calculate_flow_resistance_k() # Calculate after defaults applied

        logger.info(f"Rear Duct initialized: Type={self.duct_type.name}, Inlet={self.inlet_area_m2:.3f}m², Outlet={self.outlet_area_m2:.3f}m², Efficiency={self.efficiency:.2f}")

    def _apply_type_defaults_and_custom(self, params: Dict):
        """Apply duct type specific defaults and custom parameters."""
        defaults = {}
        # Efficiency here refers to pressure recovery efficiency
        if self.duct_type == DuctType.NACA_INLET:
            defaults = {'efficiency': 0.88, 'drag_coefficient': 0.02, 'base_resistance_k': 0.5}
        elif self.duct_type == DuctType.SIDE_SCOOP:
            defaults = {'efficiency': 0.75, 'drag_coefficient': 0.06, 'base_resistance_k': 1.0}
        elif self.duct_type == DuctType.TOP_INLET:
            defaults = {'efficiency': 0.80, 'drag_coefficient': 0.04, 'base_resistance_k': 0.8}
        elif self.duct_type == DuctType.DIFFUSER_INTEGRATED:
            defaults = {'efficiency': 0.92, 'drag_coefficient': 0.015, 'base_resistance_k': 0.4}
        elif self.duct_type == DuctType.CUSTOM:
             defaults = {'efficiency': 0.80, 'drag_coefficient': 0.05, 'base_resistance_k': 0.8}

        self.efficiency = float(params.get('efficiency', defaults.get('efficiency', 0.85)))
        self.drag_coefficient = float(params.get('drag_coefficient', defaults.get('drag_coefficient', 0.05)))
        self.base_resistance_k = float(params.get('base_resistance_k', defaults.get('base_resistance_k', 0.8)))

    def _calculate_flow_resistance_k(self) -> float:
        """Estimate the flow resistance coefficient k (for DeltaP = k * 0.5 * rho * v^2)."""
        # Start with base resistance for the duct type
        k = self.base_resistance_k
        # Add effect of length (longer ducts = more friction loss)
        k += 0.05 * (self.length_m / 0.5) # Add 0.05 to k for every 0.5m length
        # Add effect of expansion/contraction
        area_ratio = self.outlet_area_m2 / self.inlet_area_m2
        if area_ratio > 1.1: # Expansion loss
             k += 0.2 * (area_ratio - 1.0)**1.5
        elif area_ratio < 0.9: # Contraction loss
             k += 0.1 * (1.0 - area_ratio)**1.5
        # Modify by overall efficiency (higher efficiency means lower k)
        k /= (self.efficiency**0.5) # Inverse relationship, square root scaling is arbitrary
        return max(0.1, k) # Ensure minimum resistance

    def calculate_pressure_recovery_pa(self, vehicle_speed_mps: float) -> float:
        """Calculate the static pressure potentially recovered at the outlet (Pa)."""
        dynamic_pressure = 0.5 * AIR_DENSITY_SEA_LEVEL * vehicle_speed_mps**2
        pressure_recovery = dynamic_pressure * self.efficiency
        return pressure_recovery

    def calculate_pressure_drop_pa(self, airflow_m3s: float) -> float:
        """Calculate the pressure drop through the duct for a given airflow (Pa)."""
        if self.inlet_area_m2 <= 0: return float('inf')
        avg_velocity = airflow_m3s / self.inlet_area_m2 # Use inlet area for velocity calc
        dynamic_pressure = 0.5 * AIR_DENSITY_SEA_LEVEL * avg_velocity**2
        pressure_drop = self.flow_resistance_k * dynamic_pressure
        return pressure_drop

    def calculate_duct_airflow_m3s(self, pressure_diff_pa: float) -> float:
         """Calculate airflow (m³/s) driven by a pressure difference across the duct."""
         # DeltaP = k * 0.5 * rho * v^2 => v = sqrt(2 * DeltaP / (k * rho))
         # Q = A * v = A * sqrt(2 * DeltaP / (k * rho))
         if pressure_diff_pa <= 0 or self.flow_resistance_k <= 0:
             return 0.0
         velocity = np.sqrt(2 * pressure_diff_pa / (self.flow_resistance_k * AIR_DENSITY_SEA_LEVEL))
         airflow = self.inlet_area_m2 * velocity # Based on inlet area velocity
         return airflow

    def calculate_drag_force_N(self, vehicle_speed_mps: float) -> float:
        """Calculate the aerodynamic drag force (N) of the duct based on inlet area."""
        dynamic_pressure = 0.5 * AIR_DENSITY_SEA_LEVEL * vehicle_speed_mps**2
        drag_force = self.drag_coefficient * self.inlet_area_m2 * dynamic_pressure
        return drag_force

    def get_duct_specs(self) -> Dict:
        """Get duct specifications."""
        return {
            'duct_type': self.duct_type.name,
            'inlet_area_m2': self.inlet_area_m2,
            'outlet_area_m2': self.outlet_area_m2,
            'length_m': self.length_m,
            'efficiency': self.efficiency,
            'drag_coefficient': self.drag_coefficient,
            'flow_resistance_k': self.flow_resistance_k
        }


class RearRadiatorSystem:
    """Integrates rear radiator, ducts, and optional fan."""
    def __init__(self,
                 rear_radiator: RearRadiator,
                 inlet_duct: RearRadiatorDuct,
                 outlet_duct: Optional[RearRadiatorDuct] = None, # Outlet optional
                 cooling_fan: Optional[CoolingFan] = None):
        """
        Initialize the complete rear radiator system.

        Args:
            rear_radiator: RearRadiator instance.
            inlet_duct: Inlet duct instance.
            outlet_duct: Optional outlet duct instance.
            cooling_fan: Optional CoolingFan instance positioned relative to radiator.
        """
        self.radiator = rear_radiator
        self.inlet_duct = inlet_duct
        self.outlet_duct = outlet_duct
        self.cooling_fan = cooling_fan

        # If no outlet duct specified, create a simple default one
        if self.outlet_duct is None:
            self.outlet_duct = RearRadiatorDuct(
                duct_type=DuctType.CUSTOM,
                inlet_area_m2=inlet_duct.outlet_area_m2, # Match inlet outlet
                outlet_area_m2=inlet_duct.outlet_area_m2 * 1.1, # Slight expansion
                length_m=0.1,
                efficiency=0.9
            )
            logger.info("Default outlet duct created for RearRadiatorSystem.")


        # State
        self.current_airflow_m3s: float = 0.0
        self.heat_rejection_W: float = 0.0
        self.total_drag_N: float = 0.0
        self.system_pressure_drop_pa: float = 0.0 # Total pressure drop across rad+ducts

        logger.info("Rear Radiator System initialized.")

    def update_fan_control(self, control_signal: float):
        """Update fan speed based on control signal (0-1)."""
        if self.cooling_fan:
            self.cooling_fan.update_control(control_signal)
        # else: logger.warning("Attempted to control fan, but no fan is present.") # Reduce verbosity

    def _calculate_system_pressure_drop(self, airflow_m3s: float) -> float:
         """Estimate the total pressure drop across the system for a given airflow."""
         # Pressure drop = Inlet Duct + Radiator + Outlet Duct
         inlet_drop = self.inlet_duct.calculate_pressure_drop_pa(airflow_m3s)
         radiator_drop = self.radiator.base_radiator.calculate_pressure_drop_air_pa(airflow_m3s)
         outlet_drop = self.outlet_duct.calculate_pressure_drop_pa(airflow_m3s) if self.outlet_duct else 0.0
         return inlet_drop + radiator_drop + outlet_drop

    def calculate_system_airflow_m3s(self, vehicle_speed_mps: float) -> float:
        """Calculate airflow (m³/s) through the system considering ram air, fan, and resistance."""
        # --- This requires solving the system operating point ---
        # Find airflow Q where FanPressure(Q) + RamPressure(Q) = SystemResistance(Q)

        # 1. Fan pressure curve: P_fan = Pmax_fan(speed) * (1 - (Q / Qmax_fan(speed))^2)
        # 2. Ram pressure effective at radiator inlet: P_ram = InletDuctRecovery(V_veh) - InletDuctLoss(Q)
        # 3. System resistance pressure drop: P_sys = RadDrop(Q) + OutletDuctLoss(Q)

        # --- Simplified Iterative Approach ---
        # Start with an initial guess for airflow (e.g., based on ram air only)
        initial_ram_pressure = self.inlet_duct.calculate_pressure_recovery_pa(vehicle_speed_mps)
        q_guess = self.inlet_duct.calculate_duct_airflow_m3s(initial_ram_pressure / 2.0) # Guess flow based on half the pressure

        max_iterations = 10
        tolerance = 0.001 # m³/s

        for _ in range(max_iterations):
             # Calculate system pressure drop at current flow guess
             system_drop = self._calculate_system_pressure_drop(q_guess)

             # Calculate fan pressure contribution at current flow guess
             fan_p_max = self.cooling_fan.max_static_pressure_pa * self.cooling_fan.current_duty_cycle**2 if self.cooling_fan else 0.0
             fan_q_max = self.cooling_fan.max_airflow_m3s * self.cooling_fan.current_duty_cycle**3 if self.cooling_fan else 0.0
             if fan_p_max > 0 and fan_q_max > 0 and q_guess < fan_q_max:
                 fan_pressure = fan_p_max * (1.0 - (q_guess / fan_q_max)**2)
             else:
                 fan_pressure = 0.0

             # Calculate effective ram pressure at radiator inlet
             inlet_pressure_recovery = self.inlet_duct.calculate_pressure_recovery_pa(vehicle_speed_mps)
             inlet_duct_drop = self.inlet_duct.calculate_pressure_drop_pa(q_guess)
             ram_pressure_at_rad = inlet_pressure_recovery - inlet_duct_drop

             # Total driving pressure = Ram Pressure + Fan Pressure
             driving_pressure = ram_pressure_at_rad + fan_pressure

             # Required pressure drop for the rest of the system (Radiator + Outlet)
             required_downstream_drop = driving_pressure
             downstream_resistance_k = self.radiator.base_radiator._calculate_flow_resistance_k() + \
                                      (self.outlet_duct.flow_resistance_k if self.outlet_duct else 0.0)

             # Estimate new flow based on downstream resistance and driving pressure
             # Q = A * sqrt(2*DeltaP / (k*rho)) -> Use radiator area as reference A
             if required_downstream_drop > 0 and downstream_resistance_k > 0:
                  new_q = self.radiator.base_radiator.core_area * \
                          np.sqrt(2 * required_downstream_drop / (downstream_resistance_k * AIR_DENSITY_SEA_LEVEL))
             else:
                  new_q = 0.0

             # Check for convergence
             if abs(new_q - q_guess) < tolerance:
                  q_guess = new_q
                  break

             q_guess = 0.8 * q_guess + 0.2 * new_q # Damped update

        else:
             logger.debug(f"Airflow calculation did not fully converge after {max_iterations} iterations.")

        self.current_airflow_m3s = max(0.0, q_guess)
        self.system_pressure_drop_pa = self._calculate_system_pressure_drop(self.current_airflow_m3s)

        return self.current_airflow_m3s

    def calculate_heat_rejection(self, coolant_temp_C: float, ambient_temp_C: float,
                               coolant_flow_lpm: float, vehicle_speed_mps: float) -> float:
        """Calculate heat rejection (W) for the entire system."""
        # Calculate system airflow first to know the actual airflow through the radiator
        system_airflow_m3s = self.calculate_system_airflow_m3s(vehicle_speed_mps)

        # Use the base radiator's calculation method with the calculated system airflow
        self.heat_rejection_W = self.radiator.base_radiator.calculate_heat_rejection(
             coolant_temp_C, ambient_temp_C, coolant_flow_lpm, system_airflow_m3s
        )
        return self.heat_rejection_W

    def calculate_total_drag_N(self, vehicle_speed_mps: float) -> float:
        """Calculate the total drag force (N) added by the system."""
        inlet_drag = self.inlet_duct.calculate_drag_force_N(vehicle_speed_mps)
        outlet_drag = self.outlet_duct.calculate_drag_force_N(vehicle_speed_mps) if self.outlet_duct else 0.0
        radiator_installation_drag = self.radiator.get_drag_increase(vehicle_speed_mps)

        # Internal drag (momentum loss of air passing through system)
        # Use airflow calculated at this speed
        system_airflow = self.calculate_system_airflow_m3s(vehicle_speed_mps)
        # Estimate exit velocity relative to free stream (highly approximate)
        exit_velocity_factor = 0.6 # Assume air exits slower than vehicle speed
        internal_drag = AIR_DENSITY_SEA_LEVEL * system_airflow * (vehicle_speed_mps * (1.0 - exit_velocity_factor))

        self.total_drag_N = inlet_drag + outlet_drag + radiator_installation_drag + internal_drag
        return self.total_drag_N

    def simulate_step(self, coolant_temp_C: float, ambient_temp_C: float,
                      coolant_flow_lpm: float, vehicle_speed_mps: float,
                      auto_fan_control: bool = True, target_temp: float = 90.0) -> Dict:
        """Simulate one step of the system."""
        if auto_fan_control and self.cooling_fan:
            temp_error = coolant_temp_C - target_temp
            control_signal = np.clip(temp_error / 10.0, 0.0, 1.0) # Ramp over 10C
            speed_factor = max(0.0, 1.0 - vehicle_speed_mps / 20.0) # Reduce fan at speed > 20 m/s
            self.update_fan_control(control_signal * speed_factor)
        elif not self.cooling_fan:
             self.update_fan_control(0.0)

        self.calculate_heat_rejection(coolant_temp_C, ambient_temp_C, coolant_flow_lpm, vehicle_speed_mps)
        self.calculate_total_drag_N(vehicle_speed_mps)
        # Note: Doesn't update coolant temp; that's done in the main thermal sim loop

        return self.get_system_state()

    def get_system_state(self) -> Dict:
        """Get current state of the system."""
        state = {
            'airflow_m3s': self.current_airflow_m3s,
            'heat_rejection_W': self.heat_rejection_W,
            'total_drag_N': self.total_drag_N,
            'system_pressure_drop_pa': self.system_pressure_drop_pa,
        }
        if self.cooling_fan:
             state['fan_state'] = self.cooling_fan.get_fan_state()
        return state

    def get_system_specs(self) -> Dict:
        """Get specifications of the system."""
        specs = {
            'radiator': self.radiator.get_radiator_specs(),
            'inlet_duct': self.inlet_duct.get_duct_specs(),
            'outlet_duct': self.outlet_duct.get_duct_specs() if self.outlet_duct else None,
        }
        if self.cooling_fan:
            specs['cooling_fan'] = self.cooling_fan.get_fan_specs()
        return specs

    def analyze_performance(self, vehicle_speed_range: np.ndarray,
                          coolant_temp: float = 90.0, ambient_temp: float = 25.0,
                          coolant_flow_rate: float = 50.0) -> Dict:
        """Analyze system performance over a range of speeds."""
        results = {'vehicle_speeds_mps': vehicle_speed_range, 'airflows': [], 'heat_rejections': [], 'drags': []}
        # Simulate with full fan for max cooling potential
        self.update_fan_control(1.0)

        for speed in vehicle_speed_range:
            state = self.simulate_step(coolant_temp, ambient_temp, coolant_flow_rate, speed, auto_fan_control=False)
            results['airflows'].append(state['airflow_m3s'])
            results['heat_rejections'].append(state['heat_rejection_W'])
            results['drags'].append(state['total_drag_N'])

        # Convert lists to numpy arrays
        for key in ['airflows', 'heat_rejections', 'drags']:
            results[key] = np.array(results[key])
        results['conditions'] = {'coolant_temp':coolant_temp, 'ambient_temp':ambient_temp, 'flow_rate':coolant_flow_rate}
        return results

    def plot_performance_curves(self, analysis_results: Dict, save_path: Optional[str] = None):
        """Plot performance curves using the centralized plotting utility."""
        from ..utils.plotting import save_plot, _apply_common_ax_settings, COLOR_SCHEMES # Local import

        speeds = analysis_results['vehicle_speeds_mps']
        airflows = analysis_results['airflows']
        heat_rejections = analysis_results['heat_rejections']
        drags = analysis_results['drags']
        conditions = analysis_results['conditions']

        fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

        axes[0].plot(speeds, airflows, color=COLOR_SCHEMES['default'][0])
        _apply_common_ax_settings(axes[0], ylabel='Airflow (m³/s)', title='Rear Radiator System Performance')

        axes[1].plot(speeds, heat_rejections / 1000.0, color=COLOR_SCHEMES['default'][1]) # kW
        _apply_common_ax_settings(axes[1], ylabel='Heat Rejection (kW)')

        axes[2].plot(speeds, drags, color=COLOR_SCHEMES['default'][2])
        _apply_common_ax_settings(axes[2], xlabel='Vehicle Speed (m/s)', ylabel='Total System Drag (N)')

        fig.suptitle(f"Conditions: {conditions['coolant_temp']}°C Coolant, {conditions['ambient_temp']}°C Ambient, {conditions['flow_rate']} LPM Flow")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        if save_path: save_plot(fig, save_path)
        plt.show()
        plt.close(fig)


# --- Factory Functions ---
# (Keep existing factory functions, potentially update defaults if needed)

def create_default_rear_radiator_system() -> RearRadiatorSystem:
    """Create a default rear radiator system configuration."""
    try:
        base_radiator = Radiator(radiator_type=RadiatorType.SINGLE_CORE_ALUMINUM, core_area=0.16)
        rear_radiator = RearRadiator(base_radiator, mounting_position=MountingPosition.ABOVE_DIFFUSER, angle_degrees=20.0)
        inlet_duct = RearRadiatorDuct(duct_type=DuctType.NACA_INLET, inlet_area_m2=0.03, outlet_area_m2=0.04)
        outlet_duct = RearRadiatorDuct(duct_type=DuctType.CUSTOM, inlet_area_m2=0.04, outlet_area_m2=0.05, length_m=0.15)
        cooling_fan = CoolingFan(fan_type=FanType.VARIABLE_SPEED, max_airflow_m3s=0.3, diameter_m=0.28)
        return RearRadiatorSystem(rear_radiator, inlet_duct, outlet_duct, cooling_fan)
    except Exception as e:
        logger.error(f"Error creating default rear radiator: {e}. Using basic fallback.")
        return RearRadiatorSystem(RearRadiator(Radiator()), RearRadiatorDuct(), cooling_fan=CoolingFan()) # Basic fallback

def create_optimized_rear_radiator_system() -> RearRadiatorSystem:
    """Create an optimized rear radiator system."""
    try:
        base_radiator = Radiator(radiator_type=RadiatorType.DOUBLE_CORE_ALUMINUM, core_area=0.18, fin_density=18)
        rear_radiator = RearRadiator(base_radiator, mounting_position=MountingPosition.DIFFUSER_INTEGRATED, angle_degrees=10)
        inlet_duct = RearRadiatorDuct(duct_type=DuctType.DIFFUSER_INTEGRATED, inlet_area_m2=0.035, outlet_area_m2=0.05, efficiency=0.92)
        outlet_duct = RearRadiatorDuct(duct_type=DuctType.CUSTOM, inlet_area_m2=0.05, outlet_area_m2=0.06, length_m=0.12, efficiency=0.95)
        cooling_fan = CoolingFan(fan_type=FanType.DUAL_FAN, max_airflow_m3s=0.4, diameter_m=0.22)
        return RearRadiatorSystem(rear_radiator, inlet_duct, outlet_duct, cooling_fan)
    except Exception as e:
        logger.error(f"Error creating optimized rear radiator: {e}. Using default.")
        return create_default_rear_radiator_system()


def create_minimal_weight_rear_radiator_system() -> RearRadiatorSystem:
    """Create a minimal weight rear radiator system."""
    try:
        base_radiator = Radiator(radiator_type=RadiatorType.SINGLE_CORE_ALUMINUM, core_area=0.14, core_thickness=0.035, tube_rows=1)
        rear_radiator = RearRadiator(base_radiator, mounting_position=MountingPosition.ANGLED_UPWARD, angle_degrees=30.0)
        inlet_duct = RearRadiatorDuct(duct_type=DuctType.TOP_INLET, inlet_area_m2=0.025, outlet_area_m2=0.03)
        outlet_duct = RearRadiatorDuct(duct_type=DuctType.CUSTOM, inlet_area_m2=0.03, outlet_area_m2=0.035, length_m=0.1)
        cooling_fan = CoolingFan(fan_type=FanType.SINGLE_SPEED, max_airflow_m3s=0.2, diameter_m=0.25, weight_kg=0.4) # Lighter fan
        return RearRadiatorSystem(rear_radiator, inlet_duct, outlet_duct, cooling_fan)
    except Exception as e:
        logger.error(f"Error creating minimal weight rear radiator: {e}. Using default.")
        return create_default_rear_radiator_system()


# Example Usage
if __name__ == "__main__":
    opt_system = create_optimized_rear_radiator_system()
    print("\n--- Optimized Rear Radiator System Specs ---")
    specs = opt_system.get_system_specs()
    # Use yaml dump for cleaner printing of nested dicts
    print(yaml.dump(specs, default_flow_style=False, sort_keys=False))

    print("\n--- Analyzing Optimized System Performance ---")
    speeds = np.linspace(0, 30, 11)
    perf_data = opt_system.analyze_performance(speeds)

    print("\nPerformance Summary:")
    print("Speed (m/s) | Airflow (m³/s) | Heat Rej (kW) | Drag (N)")
    print("-" * 55)
    for i, speed in enumerate(perf_data['vehicle_speeds_mps']):
        airflow = perf_data['airflows'][i]
        heat_rej_kw = perf_data['heat_rejections'][i] / 1000.0
        drag = perf_data['drags'][i]
        print(f"{speed:^11.1f} | {airflow:^14.3f} | {heat_rej_kw:^15.2f} | {drag:^8.1f}")

    # Plotting (requires utils.plotting)
    try:
         from ..utils.plotting import set_plot_style
         set_plot_style('clean')
         opt_system.plot_performance_curves(perf_data) # Show plot
         # opt_system.plot_performance_curves(perf_data, save_path="plots/rear_radiator_optimized_perf.png") # Save plot
    except ImportError:
         print("\nPlotting skipped: utils.plotting not found.")
    except Exception as e:
        print(f"\nPlotting error: {e}")