"""
Core cooling system components module for Formula Student simulation.

Models radiators, water pumps, cooling fans, and thermostats, integrated
into a complete cooling system model.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Callable
import logging
from enum import Enum, auto
import yaml
from scipy.interpolate import interp1d

# Constants (ideally import from utils, define fallback here)
try:
    from ..utils.constants import AIR_DENSITY_SEA_LEVEL, AIR_SPECIFIC_HEAT_CP, WATER_DENSITY, WATER_SPECIFIC_HEAT,LITERS_TO_M3, BAR_TO_PA
except ImportError:
    AIR_DENSITY_SEA_LEVEL = 1.225
    AIR_SPECIFIC_HEAT_CP = 1005.0
    WATER_DENSITY = 1000.0
    WATER_SPECIFIC_HEAT = 4186.0
    LITERS_TO_M3 = 0.001
    BAR_TO_PA = 100000.0
    logger = logging.getLogger("CoolingSystemComponents_Fallback")
    logger.warning("Could not import utils.constants. Using fallback values.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("CoolingSystemComponents")


class RadiatorType(Enum):
    """Radiator core construction types."""
    SINGLE_CORE_ALUMINUM = auto()
    DOUBLE_CORE_ALUMINUM = auto()
    SINGLE_CORE_COPPER = auto()
    CUSTOM = auto()

class PumpType(Enum):
    """Water pump drive types."""
    MECHANICAL = auto() # Engine driven
    ELECTRIC = auto()   # Electrically driven
    CUSTOM = auto()

class FanType(Enum):
    """Cooling fan types."""
    SINGLE_SPEED = auto()
    VARIABLE_SPEED = auto()
    DUAL_FAN = auto()
    CUSTOM = auto()

class Radiator:
    """Models a heat exchanger (radiator)."""
    def __init__(self,
                 radiator_type: RadiatorType = RadiatorType.SINGLE_CORE_ALUMINUM,
                 core_area: float = 0.15,        # m²
                 core_thickness: float = 0.045,  # m
                 fin_density: float = 16,        # fins/inch
                 tube_rows: int = 2,
                 max_pressure: float = 1.5,      # bar (gauge)
                 coolant_volume: float = 0.5,    # Liters (volume within radiator)
                 config_path: Optional[str] = None,
                 custom_params: Optional[Dict] = None):
        """
        Initialize the radiator model.

        Args:
            radiator_type: Type of radiator construction.
            core_area: Radiator core frontal area (m²).
            core_thickness: Radiator core depth (m).
            fin_density: Cooling fins per inch.
            tube_rows: Number of coolant tube rows.
            max_pressure: Maximum operating pressure (bar gauge).
            coolant_volume: Internal coolant volume (L).
            config_path: Optional path to YAML config file.
            custom_params: Optional dictionary for CUSTOM type or overrides.
        """
        self.radiator_type = radiator_type
        self.core_area = core_area
        self.core_thickness = core_thickness
        self.fin_density = fin_density
        self.tube_rows = tube_rows
        self.max_pressure_bar = max_pressure
        self.coolant_volume_L = coolant_volume

        # Load from config if path provided
        if config_path and os.path.exists(config_path):
             self._load_config(config_path)

        # Apply custom params or set defaults based on type
        params = custom_params or {}
        self._apply_type_defaults_and_custom(params)

        # Derived properties
        self.total_core_volume_m3 = self.core_area * self.core_thickness
        self.air_side_surface_area_m2 = self._calculate_surface_area()

        # State variables
        self.current_effectiveness: float = self.base_effectiveness # Initial assumption

        logger.info(f"Radiator initialized: Type={self.radiator_type.name}, Area={self.core_area:.3f}m², Thick={self.core_thickness*1000:.0f}mm, Weight={self.weight_kg:.2f}kg")

    def _load_config(self, config_path: str):
         """Load parameters from YAML config file."""
         try:
             with open(config_path, 'r') as f:
                 config = yaml.safe_load(f)
             rad_config = config.get('radiator', {})
             # Update attributes if present in config
             self.radiator_type = RadiatorType[rad_config.get('type', self.radiator_type.name).upper()]
             self.core_area = float(rad_config.get('core_area', self.core_area))
             self.core_thickness = float(rad_config.get('core_thickness', self.core_thickness))
             self.fin_density = float(rad_config.get('fin_density', self.fin_density))
             self.tube_rows = int(rad_config.get('tube_rows', self.tube_rows))
             self.max_pressure_bar = float(rad_config.get('max_pressure', self.max_pressure_bar))
             self.coolant_volume_L = float(rad_config.get('coolant_volume', self.coolant_volume_L))
             # Allow custom params in config to override defaults
             self._apply_type_defaults_and_custom(rad_config)
             logger.info(f"Radiator config loaded from {config_path}")
         except Exception as e:
             logger.error(f"Error loading radiator config from {config_path}: {e}. Using existing values.")


    def _apply_type_defaults_and_custom(self, params: Dict):
        """Set default properties based on type, overridden by params."""
        defaults = {}
        if self.radiator_type == RadiatorType.SINGLE_CORE_ALUMINUM:
            defaults = {'base_effectiveness': 0.68, 'thermal_conductivity': 205, 'weight_factor': 30.0} # kg/m^2 estimate
        elif self.radiator_type == RadiatorType.DOUBLE_CORE_ALUMINUM:
            defaults = {'base_effectiveness': 0.75, 'thermal_conductivity': 205, 'weight_factor': 45.0}
        elif self.radiator_type == RadiatorType.SINGLE_CORE_COPPER:
            defaults = {'base_effectiveness': 0.72, 'thermal_conductivity': 385, 'weight_factor': 55.0}
        elif self.radiator_type == RadiatorType.CUSTOM:
            defaults = {'base_effectiveness': 0.70, 'thermal_conductivity': 205, 'weight_factor': 35.0} # Generic custom

        self.base_effectiveness = float(params.get('base_effectiveness', defaults.get('base_effectiveness', 0.7)))
        self.thermal_conductivity_W_mK = float(params.get('thermal_conductivity', defaults.get('thermal_conductivity', 205)))
        weight_factor = float(params.get('weight_factor', defaults.get('weight_factor', 35.0)))
        self.weight_kg = float(params.get('weight_kg', self.core_area * weight_factor)) # Estimate weight if not given

    def _calculate_surface_area(self) -> float:
        """Estimate the air-side heat transfer surface area."""
        # Simplified model - more accurate would need detailed fin/tube geometry
        fins_per_meter = self.fin_density * 39.37
        # Approx fin height assuming some space for tubes
        fin_height = self.core_thickness * 0.8 / self.tube_rows if self.tube_rows > 0 else self.core_thickness * 0.8
        # Approx number of channels based on area width (assume square root for width)
        approx_width = np.sqrt(self.core_area)
        num_channels = approx_width * fins_per_meter # Channels between fins

        # Area = 2 * height * length * num_channels (for both sides of fins)
        # Assume fin length is related to core area / width
        fin_area = 2 * fin_height * approx_width * num_channels

        # Add tube surface area (rough estimate)
        tube_area = self.core_area * self.tube_rows * 1.5 # Factor for tube surface exposure

        return fin_area + tube_area

    def _calculate_effectiveness(self, coolant_flow_kg_s: float, air_flow_kg_s: float) -> float:
        """Calculate dynamic effectiveness based on flow rates using NTU method."""
        # Heat capacities
        Cp_coolant = WATER_SPECIFIC_HEAT # Assume water-like coolant
        Cp_air = AIR_SPECIFIC_HEAT_CP

        C_coolant = coolant_flow_kg_s * Cp_coolant
        C_air = air_flow_kg_s * Cp_air

        if min(C_coolant, C_air) < 1e-3: return 0.0 # Avoid division by zero if no flow

        C_min = min(C_coolant, C_air)
        C_max = max(C_coolant, C_air)
        C_ratio = C_min / C_max

        # Overall heat transfer coefficient * Area (UA) - This is the tricky part to estimate
        # Let's link UA to the base_effectiveness and a reference condition
        # Assume reference condition C_min_ref leads to base_effectiveness
        # Ref flows: coolant 50 L/min (~0.8 kg/s), air 15 m/s through 0.15 m^2 (~2.7 kg/s) -> C_air is C_min
        C_min_ref = 2.7 * Cp_air # Reference C_min
        NTU_ref = -np.log(1 - self.base_effectiveness) # Assuming C_ratio ~ 0 (simplification) for ref NTU
        UA_ref = NTU_ref * C_min_ref

        # Scale UA based on current flow conditions (e.g., using correlations like Dittus-Boelter)
        # Simplified scaling: UA scales roughly with flow^0.8
        # Assume UA scales primarily with the limiting fluid (C_min)
        flow_ratio = C_min / C_min_ref if C_min_ref > 0 else 1.0
        UA = UA_ref * (flow_ratio ** 0.6) # Simplified scaling exponent

        # Calculate current NTU
        NTU = UA / C_min

        # Effectiveness for cross-flow (unmixed-unmixed is common for radiators)
        # More complex formula, simplified as epsilon = 1 - exp(-NTU) for C_ratio ~ 0
        # Or use a generic correlation:
        epsilon = 1 - np.exp(-NTU * (1 + C_ratio**0.22) / (C_ratio**0.22)) # Approximate correlation

        # Use simpler formula if C_ratio is very small
        if C_ratio < 0.1:
             epsilon = 1 - np.exp(-NTU)

        effectiveness = np.clip(epsilon, 0.0, 0.95) # Cap effectiveness
        self.current_effectiveness = effectiveness # Store current value
        return effectiveness

    def calculate_heat_rejection(self, coolant_temp_C: float, ambient_temp_C: float,
                           coolant_flow_lpm: float, air_flow_m3_s: float) -> float:
        """Calculate heat rejection rate (W) using NTU method."""
        # Convert flows to kg/s
        coolant_density_kg_L = WATER_DENSITY / 1000.0 # Approx.
        coolant_flow_kg_s = coolant_flow_lpm * coolant_density_kg_L / 60.0
        air_density_kg_m3 = AIR_DENSITY_SEA_LEVEL # Use standard density
        air_flow_kg_s = air_flow_m3_s * air_density_kg_m3

        # Calculate dynamic effectiveness
        effectiveness = self._calculate_effectiveness(coolant_flow_kg_s, air_flow_kg_s)

        # Calculate Cmin
        C_coolant = coolant_flow_kg_s * WATER_SPECIFIC_HEAT
        C_air = air_flow_kg_s * AIR_SPECIFIC_HEAT_CP
        C_min = min(C_coolant, C_air) if min(C_coolant, C_air) > 1e-3 else 0.0

        # Temperature difference
        delta_T = coolant_temp_C - ambient_temp_C

        # Heat rejection Q = epsilon * Cmin * deltaT
        heat_rejection_W = effectiveness * C_min * delta_T
        return max(0.0, heat_rejection_W) # Heat rejection cannot be negative

    def calculate_pressure_drop_coolant_bar(self, coolant_flow_lpm: float) -> float:
        """Estimate coolant pressure drop (bar) based on flow rate."""
        # Simplified quadratic model: DeltaP = k * flow^2
        # Need a reference point (e.g., 0.1 bar drop at 50 L/min)
        ref_flow = 50.0 # L/min
        ref_drop = 0.10 # bar
        k = ref_drop / (ref_flow**2) if ref_flow > 0 else 0
        pressure_drop = k * coolant_flow_lpm**2
        return pressure_drop

    def calculate_pressure_drop_air_pa(self, air_flow_m3_s: float) -> float:
        """Estimate air-side pressure drop (Pa) based on flow rate."""
        # Simplified quadratic model: DeltaP = k * flow^2
        # Estimate k based on typical FS radiator data (e.g., 100 Pa drop at 0.5 m^3/s)
        ref_flow = 0.5 # m^3/s
        ref_drop = 100.0 # Pa
        k = ref_drop / (ref_flow**2) if ref_flow > 0 else 0
        pressure_drop = k * air_flow_m3_s**2
        return pressure_drop

    def calculate_coolant_exit_temp(self, inlet_temp_C: float, heat_rejection_W: float,
                                  coolant_flow_lpm: float) -> float:
        """Calculate coolant exit temperature (°C)."""
        coolant_density_kg_L = WATER_DENSITY / 1000.0
        mass_flow_kg_s = coolant_flow_lpm * coolant_density_kg_L / 60.0

        if mass_flow_kg_s <= 1e-6: # No flow, no temperature change
             return inlet_temp_C

        # Q = m_dot * Cp * delta_T => delta_T = Q / (m_dot * Cp)
        delta_T = heat_rejection_W / (mass_flow_kg_s * WATER_SPECIFIC_HEAT)
        exit_temp_C = inlet_temp_C - delta_T
        return exit_temp_C # Can potentially be lower than ambient if heat rejection is very high

    def get_radiator_specs(self) -> Dict:
        """Return radiator specifications as a dictionary."""
        return {
            'type': self.radiator_type.name,
            'core_area_m2': self.core_area,
            'core_thickness_m': self.core_thickness,
            'fin_density_fpi': self.fin_density,
            'tube_rows': self.tube_rows,
            'max_pressure_bar': self.max_pressure_bar,
            'coolant_volume_L': self.coolant_volume_L,
            'base_effectiveness': self.base_effectiveness,
            'thermal_conductivity_W_mK': self.thermal_conductivity_W_mK,
            'weight_kg': self.weight_kg,
            'air_side_surface_area_m2': self.air_side_surface_area_m2
        }


class WaterPump:
    """Models a coolant water pump."""
    def __init__(self,
                 pump_type: PumpType = PumpType.ELECTRIC,
                 max_flow_rate_lpm: float = 80.0,
                 max_pressure_bar: float = 0.5, # Typical head for automotive pumps
                 nominal_speed_rpm: float = 6000.0, # For electric pumps
                 mechanical_efficiency: float = 0.60,
                 config_path: Optional[str] = None,
                 custom_params: Optional[Dict] = None):
        """
        Initialize water pump model.

        Args:
            pump_type: Type of water pump.
            max_flow_rate_lpm: Max flow rate (L/min) at zero pressure head.
            max_pressure_bar: Max pressure head (bar gauge) at zero flow.
            nominal_speed_rpm: Speed for nominal performance (electric).
            mechanical_efficiency: Pump shaft-to-hydraulic efficiency (0-1).
            config_path: Optional path to YAML config file.
            custom_params: Optional dictionary for CUSTOM type or overrides.
        """
        self.pump_type = pump_type
        self.max_flow_rate_lpm = max_flow_rate_lpm
        self.max_pressure_bar = max_pressure_bar
        self.nominal_speed_rpm = nominal_speed_rpm
        self.mechanical_efficiency = mechanical_efficiency

        # Load from config if path provided
        if config_path and os.path.exists(config_path):
             self._load_config(config_path)

        # Apply custom params or set defaults based on type
        params = custom_params or {}
        self._apply_type_defaults_and_custom(params)

        # State variables
        self.current_speed_rpm: float = 0.0
        self.current_flow_lpm: float = 0.0
        self.current_pressure_bar: float = 0.0 # Pressure generated
        self.current_power_W: float = 0.0    # Electrical/Mechanical power consumed

        # Pump curve (flow vs pressure head at nominal speed)
        self._create_pump_curve()

        logger.info(f"Water Pump initialized: Type={self.pump_type.name}, MaxFlow={self.max_flow_rate_lpm:.1f}LPM, MaxHead={self.max_pressure_bar:.2f}bar")

    def _load_config(self, config_path: str):
        """Load parameters from YAML config file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            pump_config = config.get('water_pump', {})
            self.pump_type = PumpType[pump_config.get('type', self.pump_type.name).upper()]
            self.max_flow_rate_lpm = float(pump_config.get('max_flow_rate', self.max_flow_rate_lpm))
            self.max_pressure_bar = float(pump_config.get('max_pressure', self.max_pressure_bar))
            self.nominal_speed_rpm = float(pump_config.get('nominal_speed', self.nominal_speed_rpm))
            self.mechanical_efficiency = float(pump_config.get('mechanical_efficiency', self.mechanical_efficiency))
            # Allow custom params in config to override defaults
            self._apply_type_defaults_and_custom(pump_config)
            logger.info(f"Pump config loaded from {config_path}")
        except Exception as e:
            logger.error(f"Error loading pump config from {config_path}: {e}. Using existing values.")

    def _apply_type_defaults_and_custom(self, params: Dict):
        """Set default properties based on type, overridden by params."""
        defaults = {}
        if self.pump_type == PumpType.MECHANICAL:
             defaults = {'speed_ratio': 1.0, 'power_consumption_factor': 0.08, 'weight_kg': 0.8} # W/RPM approx, weight
        elif self.pump_type == PumpType.ELECTRIC:
             defaults = {'voltage': 12.0, 'current_draw_max_A': 12.0, 'weight_kg': 0.5}
        elif self.pump_type == PumpType.CUSTOM:
             defaults = {'voltage': 12.0, 'current_draw_max_A': 10.0, 'weight_kg': 0.6}

        # Set attributes, using params value if present, else defaults value
        self.speed_ratio = float(params.get('speed_ratio', defaults.get('speed_ratio', 0.0))) # Only relevant for MECHANICAL
        self.power_consumption_factor = float(params.get('power_consumption_factor', defaults.get('power_consumption_factor', 0.0))) # W/RPM for mech, or baseline for elec
        self.weight_kg = float(params.get('weight_kg', defaults.get('weight_kg', 0.6)))
        self.voltage_V = float(params.get('voltage', defaults.get('voltage', 12.0))) # Only relevant for ELECTRIC
        self.current_draw_max_A = float(params.get('current_draw_max_A', defaults.get('current_draw_max_A', 0.0))) # Only relevant for ELECTRIC

    def _create_pump_curve(self):
        """Create the pump's flow vs pressure head curve at nominal speed."""
        # Simplified quadratic curve: Flow = MaxFlow * (1 - (Pressure / MaxPressure)^0.5)
        # Or use linear for simplicity if preferred: Flow = MaxFlow * (1 - Pressure / MaxPressure)
        pressures = np.linspace(0, self.max_pressure_bar, 20)
        # Linear model:
        flows = self.max_flow_rate_lpm * (1 - pressures / self.max_pressure_bar)
        # Quadratic model (more realistic for centrifugal):
        # flows = self.max_flow_rate_lpm * (1 - np.sqrt(pressures / self.max_pressure_bar))
        flows = np.maximum(0, flows) # Ensure non-negative flow

        # Store points for affinity law scaling
        self._nominal_curve_pressures = pressures
        self._nominal_curve_flows = flows

        # Create interpolation function for nominal speed
        self._nominal_flow_func = interp1d(pressures, flows, kind='linear',
                                          bounds_error=False, fill_value=(self.max_flow_rate_lpm, 0.0))

    def update_pump_speed(self, engine_rpm: Optional[float] = None, control_signal: Optional[float] = None):
        """Update pump speed based on engine RPM (mechanical) or control signal (electric)."""
        if self.pump_type == PumpType.MECHANICAL:
            if engine_rpm is None: logger.warning("Engine RPM needed for mechanical pump speed.")
            self.current_speed_rpm = (engine_rpm or 0) * self.speed_ratio
        elif self.pump_type == PumpType.ELECTRIC:
            if control_signal is None: logger.warning("Control signal needed for electric pump speed.")
            self.current_speed_rpm = self.nominal_speed_rpm * np.clip(control_signal or 0, 0.0, 1.0)
        else: # Custom
             # Assuming custom pump speed is set externally or via control_signal
             self.current_speed_rpm = self.nominal_speed_rpm * np.clip(control_signal or 0, 0.0, 1.0)

        self.current_speed_rpm = max(0, self.current_speed_rpm) # Ensure non-negative speed

    def calculate_flow_rate_lpm(self, system_pressure_drop_bar: float) -> float:
        """Calculate flow rate (LPM) based on current speed and system pressure drop."""
        if self.nominal_speed_rpm <= 0: return 0.0 # Cannot scale if nominal speed is zero
        speed_ratio = self.current_speed_rpm / self.nominal_speed_rpm

        # Affinity Laws: Flow ~ Speed, Pressure ~ Speed^2
        scaled_max_flow = self.max_flow_rate_lpm * speed_ratio
        scaled_max_pressure = self.max_pressure_bar * speed_ratio**2

        # Use the scaled curve to find flow at the system pressure drop
        if scaled_max_pressure <= system_pressure_drop_bar:
            # Pump cannot overcome system resistance at this speed
            flow_rate = 0.0
            self.current_pressure_bar = scaled_max_pressure # Max pressure it can generate
        else:
            # Interpolate on the scaled curve (recreate function with scaled values)
            scaled_pressures = self._nominal_curve_pressures * speed_ratio**2
            scaled_flows = self._nominal_curve_flows * speed_ratio
            scaled_func = interp1d(scaled_pressures, scaled_flows, kind='linear',
                                   bounds_error=False, fill_value=(scaled_max_flow, 0.0))
            flow_rate = float(scaled_func(system_pressure_drop_bar))
            self.current_pressure_bar = system_pressure_drop_bar # Pressure generated matches system drop

        self.current_flow_lpm = max(0.0, flow_rate)
        return self.current_flow_lpm

    def calculate_power_consumption_W(self) -> float:
        """Estimate pump power consumption (W)."""
        if self.current_speed_rpm <= 0 or self.mechanical_efficiency <= 0:
            self.current_power_W = 0.0
            return 0.0

        # Hydraulic Power (W) = Flow (m^3/s) * Pressure (Pa)
        flow_m3_s = self.current_flow_lpm * LITERS_TO_M3 / 60.0
        pressure_Pa = self.current_pressure_bar * BAR_TO_PA
        hydraulic_power = flow_m3_s * pressure_Pa

        # Shaft/Electrical Power = Hydraulic Power / Efficiency
        shaft_power = hydraulic_power / self.mechanical_efficiency

        # Add baseline losses (mechanical friction or electrical standby)
        # Simplified: Assume some baseline power proportional to max power and speed ratio cubed
        baseline_power = 0.0
        if self.pump_type == PumpType.MECHANICAL:
            # Estimate max shaft power needed at max flow/pressure/speed
            max_hyd_power = (self.max_flow_rate_lpm * LITERS_TO_M3 / 60.0) * (self.max_pressure_bar * BAR_TO_PA)
            max_shaft_power = max_hyd_power / self.mechanical_efficiency
            baseline_power = max_shaft_power * 0.05 * (self.current_speed_rpm / self.nominal_speed_rpm)**3 # 5% baseline loss factor
        elif self.pump_type == PumpType.ELECTRIC:
            # Max electrical power ~ V * Imax
            max_elec_power = self.voltage_V * self.current_draw_max_A
            baseline_power = max_elec_power * 0.05 * (self.current_speed_rpm / self.nominal_speed_rpm) # Small baseline electrical loss

        self.current_power_W = shaft_power + baseline_power
        # Ensure non-negative power
        self.current_power_W = max(0.0, self.current_power_W)

        # Cap at max electrical power for electric pumps
        if self.pump_type == PumpType.ELECTRIC and self.current_draw_max_A > 0:
             self.current_power_W = min(self.current_power_W, self.voltage_V * self.current_draw_max_A)

        return self.current_power_W

    def get_pump_state(self) -> Dict:
        """Get current operating state of the pump."""
        return {
            'speed_rpm': self.current_speed_rpm,
            'flow_lpm': self.current_flow_lpm,
            'pressure_bar': self.current_pressure_bar, # Pressure generated
            'power_W': self.current_power_W
        }

    def get_pump_specs(self) -> Dict:
        """Get pump specifications."""
        specs = {
            'type': self.pump_type.name,
            'max_flow_rate_lpm': self.max_flow_rate_lpm,
            'max_pressure_bar': self.max_pressure_bar,
            'nominal_speed_rpm': self.nominal_speed_rpm,
            'mechanical_efficiency': self.mechanical_efficiency,
            'weight_kg': self.weight_kg
        }
        if self.pump_type == PumpType.MECHANICAL:
            specs['speed_ratio'] = self.speed_ratio
        elif self.pump_type == PumpType.ELECTRIC:
            specs['voltage_V'] = self.voltage_V
            specs['current_draw_max_A'] = self.current_draw_max_A
        return specs


class CoolingFan:
    """Models an electric cooling fan."""
    def __init__(self,
                 fan_type: FanType = FanType.VARIABLE_SPEED,
                 max_airflow_m3s: float = 0.3,
                 diameter_m: float = 0.25,
                 max_power_W: float = 90.0,
                 voltage_V: float = 12.0,
                 config_path: Optional[str] = None,
                 custom_params: Optional[Dict] = None):
        """
        Initialize cooling fan model.

        Args:
            fan_type: Type of cooling fan.
            max_airflow_m3s: Max airflow (m³/s) at zero static pressure.
            diameter_m: Fan blade diameter (m).
            max_power_W: Max electrical power consumption (W).
            voltage_V: Operating voltage (V).
            config_path: Optional path to YAML config file.
            custom_params: Optional dictionary for CUSTOM type or overrides.
        """
        self.fan_type = fan_type
        self.max_airflow_m3s = max_airflow_m3s
        self.diameter_m = diameter_m
        self.max_power_W = max_power_W
        self.voltage_V = voltage_V
        self.max_static_pressure_pa: float = 150.0

        # Load from config if path provided
        if config_path and os.path.exists(config_path):
             self._load_config(config_path)

        # Apply custom params or set defaults based on type
        params = custom_params or {}
        self._apply_type_defaults_and_custom(params)

        # Derived properties
        self.area_m2 = np.pi * (self.diameter_m / 2.0)**2

        # State variables
        self.current_duty_cycle: float = 0.0 # 0-1 (or 0/1 for on/off)
        self.current_airflow_m3s: float = 0.0
        self.current_power_W: float = 0.0
        self.is_active: bool = False

        # Fan curve (Pressure vs Flow) - Simplified placeholder
        # P = Pmax * (1 - (Q/Qmax)^2)
        self.max_static_pressure_pa = 150.0 # Estimated max pressure at zero flow

        logger.info(f"Cooling Fan initialized: Type={self.fan_type.name}, MaxAirflow={self.max_airflow_m3s:.2f}m³/s, Dia={self.diameter_m*1000:.0f}mm")

    def _load_config(self, config_path: str):
        """Load parameters from YAML config file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            fan_config = config.get('cooling_fan', {})
            self.fan_type = FanType[fan_config.get('type', self.fan_type.name).upper()]
            self.max_airflow_m3s = float(fan_config.get('max_airflow', self.max_airflow_m3s))
            self.diameter_m = float(fan_config.get('diameter', self.diameter_m))
            self.max_power_W = float(fan_config.get('max_power', self.max_power_W))
            self.voltage_V = float(fan_config.get('voltage', self.voltage_V))
            self._apply_type_defaults_and_custom(fan_config) # Apply other params
            logger.info(f"Fan config loaded from {config_path}")
        except Exception as e:
            logger.error(f"Error loading fan config from {config_path}: {e}. Using existing values.")

    def _apply_type_defaults_and_custom(self, params: Dict):
        """Set default properties based on type, overridden by params."""
        defaults = {}
        if self.fan_type == FanType.SINGLE_SPEED:
            defaults = {'control_type': 'on_off', 'num_fans': 1, 'weight_kg': 0.5}
        elif self.fan_type == FanType.VARIABLE_SPEED:
            defaults = {'control_type': 'pwm', 'num_fans': 1, 'weight_kg': 0.55}
        elif self.fan_type == FanType.DUAL_FAN:
             # For DUAL_FAN, max_airflow/power should represent the COMBINED effect
            defaults = {'control_type': 'on_off', 'num_fans': 2, 'weight_kg': 1.0}
        elif self.fan_type == FanType.CUSTOM:
            defaults = {'control_type': 'pwm', 'num_fans': 1, 'weight_kg': 0.6}

        self.control_type = params.get('control_type', defaults.get('control_type', 'pwm'))
        self.num_fans = int(params.get('num_fans', defaults.get('num_fans', 1)))
        self.weight_kg = float(params.get('weight_kg', defaults.get('weight_kg', 0.6)))
        self.max_static_pressure_pa = float(params.get('max_static_pressure_pa', self.max_static_pressure_pa))


    def update_control(self, control_signal: float):
        """Update fan state based on control signal (0-1)."""
        control_signal = np.clip(control_signal, 0.0, 1.0)

        if self.control_type == "on_off":
             # Simple threshold logic
             self.current_duty_cycle = 1.0 if control_signal > 0.5 else 0.0
        else: # pwm or variable speed
             self.current_duty_cycle = control_signal

        self.is_active = self.current_duty_cycle > 0.05 # Active if duty > 5%
        self._update_outputs()

    def _update_outputs(self):
        """Update airflow and power based on duty cycle."""
        if not self.is_active:
            self.current_airflow_m3s = 0.0
            self.current_power_W = 0.0
            return

        # Fan laws: Airflow ~ Speed, Pressure ~ Speed^2, Power ~ Speed^3
        # Assume speed is proportional to duty cycle for PWM fans
        speed_ratio = self.current_duty_cycle

        # Airflow and power scale with speed^3 (roughly)
        self.current_airflow_m3s = self.max_airflow_m3s * speed_ratio**3
        self.current_power_W = self.max_power_W * speed_ratio**3

        # For on/off, it's just max values when on
        if self.control_type == "on_off":
             self.current_airflow_m3s = self.max_airflow_m3s if self.current_duty_cycle > 0 else 0.0
             self.current_power_W = self.max_power_W if self.current_duty_cycle > 0 else 0.0


    def get_airflow_m3s(self, system_pressure_drop_pa: float = 0.0) -> float:
        """
        Calculate actual airflow (m³/s) considering system pressure drop.

        Args:
            system_pressure_drop_pa: Pressure drop the fan works against (Pa).

        Returns:
            Actual airflow in m³/s.
        """
        if not self.is_active:
            return 0.0

        # Use the fan curve P = Pmax * (1 - (Q/Qmax)^2) to find Q for given P
        # Q = Qmax * sqrt(1 - P/Pmax)
        speed_ratio = self.current_duty_cycle
        effective_q_max = self.max_airflow_m3s * speed_ratio**3 # Airflow scales with speed^3
        effective_p_max = self.max_static_pressure_pa * speed_ratio**2 # Pressure scales with speed^2

        if effective_p_max <= 0 or system_pressure_drop_pa >= effective_p_max:
            return 0.0 # Cannot overcome pressure drop

        flow_rate = effective_q_max * np.sqrt(1.0 - system_pressure_drop_pa / effective_p_max)

        # Store current airflow state
        self.current_airflow_m3s = max(0.0, flow_rate)
        return self.current_airflow_m3s

    def get_fan_state(self) -> Dict:
        """Get current fan state."""
        return {
            'is_active': self.is_active,
            'duty_cycle': self.current_duty_cycle,
            'airflow_m3s': self.current_airflow_m3s,
            'power_W': self.current_power_W
        }

    def get_fan_specs(self) -> Dict:
        """Get fan specifications."""
        return {
            'type': self.fan_type.name,
            'max_airflow_m3s': self.max_airflow_m3s,
            'diameter_m': self.diameter_m,
            'max_power_W': self.max_power_W,
            'voltage_V': self.voltage_V,
            'area_m2': self.area_m2,
            'control_type': self.control_type,
            'num_fans': self.num_fans,
            'weight_kg': self.weight_kg,
            'max_static_pressure_pa': self.max_static_pressure_pa
        }


class Thermostat:
    """Models a coolant thermostat."""
    def __init__(self,
                 opening_temp_C: float = 82.0,
                 full_open_temp_C: float = 92.0,
                 config_path: Optional[str] = None):
        """
        Initialize thermostat model.

        Args:
            opening_temp_C: Temperature (°C) thermostat starts opening.
            full_open_temp_C: Temperature (°C) thermostat is fully open.
            config_path: Optional path to YAML config file.
        """
        self.opening_temp_C = opening_temp_C
        self.full_open_temp_C = full_open_temp_C

        if config_path and os.path.exists(config_path):
             self._load_config(config_path)

        if self.full_open_temp_C <= self.opening_temp_C:
            logger.warning("Thermostat full_open_temp <= opening_temp. Adjusting full_open.")
            self.full_open_temp_C = self.opening_temp_C + 10.0

        # State variable
        self.current_opening_fraction: float = 0.0 # 0 (closed) to 1 (fully open)

        logger.info(f"Thermostat initialized: Opens {self.opening_temp_C:.1f}°C, Fully Open {self.full_open_temp_C:.1f}°C")

    def _load_config(self, config_path: str):
        """Load parameters from YAML config file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            thermo_config = config.get('thermostat', {})
            self.opening_temp_C = float(thermo_config.get('opening_temp', self.opening_temp_C))
            self.full_open_temp_C = float(thermo_config.get('full_open_temp', self.full_open_temp_C))
            logger.info(f"Thermostat config loaded from {config_path}")
        except Exception as e:
             logger.error(f"Error loading thermostat config from {config_path}: {e}. Using existing values.")

    def update_state(self, coolant_temp_C: float):
        """Update thermostat opening fraction based on coolant temperature."""
        if coolant_temp_C <= self.opening_temp_C:
            self.current_opening_fraction = 0.0
        elif coolant_temp_C >= self.full_open_temp_C:
            self.current_opening_fraction = 1.0
        else:
            # Linear interpolation within the opening range
            self.current_opening_fraction = (coolant_temp_C - self.opening_temp_C) / (self.full_open_temp_C - self.opening_temp_C)

    def get_opening_fraction(self) -> float:
        """Return the current opening fraction (0-1)."""
        return self.current_opening_fraction

    def get_flow_fraction_to_radiator(self) -> float:
        """Return the fraction of total coolant flow directed to the radiator."""
        # Assumes flow resistance is proportional to (1 - opening fraction) for bypass
        # and opening fraction for radiator. Simplified model.
        return self.current_opening_fraction # Directly use opening fraction

    def get_thermostat_state(self) -> Dict:
        """Get current thermostat state."""
        return {
            'opening_fraction': self.current_opening_fraction,
            'opening_temp_C': self.opening_temp_C,
            'full_open_temp_C': self.full_open_temp_C
        }


class CoolingSystem:
    """Integrates cooling system components."""
    def __init__(self,
                 radiator: Radiator,
                 water_pump: WaterPump,
                 cooling_fan: Optional[CoolingFan] = None, # Fan is optional
                 thermostat: Thermostat = None, # Use default if not provided
                 coolant_volume_L: float = 2.5, # Total system volume
                 config_path: Optional[str] = None):
        """
        Initialize the complete cooling system.

        Args:
            radiator: Radiator component.
            water_pump: Water pump component.
            cooling_fan: Optional CoolingFan component.
            thermostat: Thermostat component.
            coolant_volume_L: Total coolant volume in the system (L).
            config_path: Optional path to YAML config file for system parameters.
        """
        self.radiator = radiator
        self.water_pump = water_pump
        self.cooling_fan = cooling_fan
        self.thermostat = thermostat or Thermostat() # Create default thermostat if none provided

        self.coolant_volume_L = coolant_volume_L
        self.coolant_density_kg_L = WATER_DENSITY / 1000.0
        self.coolant_specific_heat_J_kgK = WATER_SPECIFIC_HEAT
        self.system_pressure_cap_bar = 1.3 # Default pressure cap

        if config_path and os.path.exists(config_path):
            self._load_config(config_path)

        # State variables
        self.coolant_temp_C: float = 25.0
        self.ambient_temp_C: float = 25.0
        self.vehicle_speed_mps: float = 0.0
        self.engine_rpm: float = 0.0
        self.engine_load: float = 0.0 # Example, might not be directly used here
        self.engine_heat_input_W: float = 0.0 # Heat transferred from engine to coolant (W)
        self.radiator_heat_rejection_W: float = 0.0

        # Control targets (can be set externally)
        self.target_coolant_temp_C: float = 90.0
        self.fan_control_signal: float = 0.0 # External control override
        self.pump_control_signal: float = 1.0 # External control override (for electric pump)
        self.use_automatic_control: bool = True # Flag to enable internal automatic control

        # Calculate total coolant mass
        self.total_coolant_mass_kg = self.coolant_volume_L * self.coolant_density_kg_L
        self.total_thermal_capacity_J_K = self.total_coolant_mass_kg * self.coolant_specific_heat_J_kgK
        if self.total_thermal_capacity_J_K <= 0:
             logger.warning("Total thermal capacity is zero or negative. Temperature simulation may be unstable.")
             self.total_thermal_capacity_J_K = 1e-3 # Prevent division by zero

        logger.info(f"Cooling System initialized: Volume={self.coolant_volume_L:.1f}L, Cap={self.system_pressure_cap_bar:.1f}bar")

    def _load_config(self, config_path: str):
         """Load system parameters from YAML config file."""
         try:
             with open(config_path, 'r') as f:
                 config = yaml.safe_load(f)
             system_config = config.get('system', {})
             self.coolant_volume_L = float(system_config.get('coolant_volume', self.coolant_volume_L))
             self.coolant_density_kg_L = float(system_config.get('coolant_density', self.coolant_density_kg_L * 1000)) / 1000.0 # Handle density in kg/m3 or kg/L
             self.coolant_specific_heat_J_kgK = float(system_config.get('coolant_specific_heat', self.coolant_specific_heat_J_kgK))
             self.system_pressure_cap_bar = float(system_config.get('system_pressure_cap', self.system_pressure_cap_bar))
             logger.info(f"Cooling system parameters loaded from {config_path}")
         except Exception as e:
             logger.error(f"Error loading system config from {config_path}: {e}. Using existing values.")

    def update_ambient_conditions(self, ambient_temp_C: float, vehicle_speed_mps: float):
        """Update ambient temperature and vehicle speed."""
        self.ambient_temp_C = ambient_temp_C
        self.vehicle_speed_mps = max(0, vehicle_speed_mps) # Ensure non-negative speed

    def update_engine_state(self, engine_rpm: float, engine_load: float, engine_heat_input_W: float):
        """Update engine operating conditions affecting the cooling system."""
        self.engine_rpm = max(0, engine_rpm)
        self.engine_load = np.clip(engine_load, 0.0, 1.0)
        self.engine_heat_input_W = max(0, engine_heat_input_W) # Heat input must be positive

    def set_control_targets(self, target_temp: Optional[float]=None, auto_control: Optional[bool]=None):
        """Set control targets and enable/disable automatic control."""
        if target_temp is not None: self.target_coolant_temp_C = target_temp
        if auto_control is not None: self.use_automatic_control = auto_control

    def _run_automatic_control(self):
        """Internal method to calculate control signals based on temperature."""
        # Fan Control (simple P-controller based on temp exceeding target)
        temp_error = self.coolant_temp_C - self.target_coolant_temp_C
        # Activate fan above target, ramp up to max over a 10C range
        fan_signal = np.clip(temp_error / 10.0, 0.0, 1.0)
        # Reduce fan need at higher speeds
        speed_reduction_factor = max(0.0, 1.0 - self.vehicle_speed_mps / 20.0) # Fan less needed above 20 m/s
        self.fan_control_signal = fan_signal * speed_reduction_factor

        # Pump Control (for electric pump) - keep it simple: full speed if engine running
        self.pump_control_signal = 1.0 if self.engine_rpm > 0 else 0.0

        # Apply controls to components
        if self.cooling_fan:
            self.cooling_fan.update_control(self.fan_control_signal)
        if self.water_pump.pump_type == PumpType.ELECTRIC:
            self.water_pump.update_pump_speed(control_signal=self.pump_control_signal)

    def update_system_state(self, dt: float):
        """Update the cooling system state over a time step dt."""
        # 1. Update component states based on inputs/controls
        self.thermostat.update_state(self.coolant_temp_C)
        if self.water_pump.pump_type == PumpType.MECHANICAL:
            self.water_pump.update_pump_speed(engine_rpm=self.engine_rpm)
        if self.use_automatic_control:
             self._run_automatic_control()
        # Fan/Pump states are now updated

        # 2. Calculate coolant flow rate
        # Estimate system pressure drop (sum of components)
        # This is iterative in reality, simplified here
        flow_guess = self.water_pump.max_flow_rate_lpm / 2.0 # Initial guess
        system_pressure_drop = self.radiator.calculate_pressure_drop_coolant_bar(flow_guess)
        # Add pressure drop for engine block, hoses etc. (estimated)
        system_pressure_drop += 0.1 # Assume 0.1 bar drop elsewhere

        coolant_flow_lpm = self.water_pump.calculate_flow_rate_lpm(system_pressure_drop)

        # 3. Calculate flow distribution
        flow_to_radiator_lpm = coolant_flow_lpm * self.thermostat.get_flow_fraction_to_radiator()

        # 4. Calculate air flow through radiator
        radiator_air_pressure_drop_pa = self.radiator.calculate_pressure_drop_air_pa(
             self.cooling_fan.current_airflow_m3s if self.cooling_fan else 0.0 # Estimate based on fan max
        )
        fan_airflow_actual_m3s = self.cooling_fan.get_airflow_m3s(radiator_air_pressure_drop_pa) if self.cooling_fan else 0.0

        ram_air_mps = self.vehicle_speed_mps # Assume vehicle speed is effective air speed at inlet
        # Simple addition of ram air and fan air (can be refined)
        total_air_flow_m3s = (ram_air_mps * self.radiator.core_area * 0.8) + fan_airflow_actual_m3s # 0.8 ram air efficiency factor

        # 5. Calculate heat rejection
        self.radiator_heat_rejection_W = self.radiator.calculate_heat_rejection(
            self.coolant_temp_C, self.ambient_temp_C, flow_to_radiator_lpm, total_air_flow_m3s
        )

        # 6. Update coolant temperature
        net_heat_W = self.engine_heat_input_W - self.radiator_heat_rejection_W
        delta_temp = (net_heat_W * dt) / self.total_thermal_capacity_J_K
        self.coolant_temp_C += delta_temp

        # Clamp temperature (e.g., can't go below ambient easily)
        self.coolant_temp_C = max(self.ambient_temp_C - 5, self.coolant_temp_C) # Allow slightly below ambient due to potential inaccuracies

        # Update power consumptions
        self.water_pump.calculate_power_consumption_W()
        #if self.cooling_fan: self.cooling_fan.calculate_power_consumption()


    def simulate_step(self, ambient_temp_C: float, vehicle_speed_mps: float,
                    engine_rpm: float, engine_load: float,
                    engine_heat_input_W: float, dt: float) -> Dict:
        """Perform a single simulation step with given conditions."""
        self.update_ambient_conditions(ambient_temp_C, vehicle_speed_mps)
        self.update_engine_state(engine_rpm, engine_load, engine_heat_input_W)
        self.update_system_state(dt)
        return self.get_system_state()

    def get_system_state(self) -> Dict:
        """Get current state of the cooling system."""
        state = {
            'coolant_temp_C': self.coolant_temp_C,
            'ambient_temp_C': self.ambient_temp_C,
            'vehicle_speed_mps': self.vehicle_speed_mps,
            'engine_rpm': self.engine_rpm,
            'engine_load': self.engine_load,
            'engine_heat_input_W': self.engine_heat_input_W,
            'radiator_heat_rejection_W': self.radiator_heat_rejection_W,
            'net_heat_rate_W': self.engine_heat_input_W - self.radiator_heat_rejection_W,
            'water_pump': self.water_pump.get_pump_state(),
            'thermostat': self.thermostat.get_thermostat_state(),
            'fan_control_signal': self.fan_control_signal,
            'pump_control_signal': self.pump_control_signal
        }
        if self.cooling_fan:
            state['cooling_fan'] = self.cooling_fan.get_fan_state()
        return state

    def get_system_specs(self) -> Dict:
        """Get specifications of the cooling system components."""
        specs = {
            'radiator': self.radiator.get_radiator_specs(),
            'water_pump': self.water_pump.get_pump_specs(),
            'thermostat': self.thermostat.get_thermostat_state(), # Includes thresholds
            'system': {
                'coolant_volume_L': self.coolant_volume_L,
                'coolant_density_kg_L': self.coolant_density_kg_L,
                'coolant_specific_heat_J_kgK': self.coolant_specific_heat_J_kgK,
                'system_pressure_cap_bar': self.system_pressure_cap_bar
            }
        }
        if self.cooling_fan:
            specs['cooling_fan'] = self.cooling_fan.get_fan_specs()
        return specs

    def calculate_system_performance(self, ambient_temps_C: List[float],
                                  engine_heats_W: List[float],
                                  vehicle_speed_mps: float = 15.0,
                                  engine_rpm: float = 8000,
                                  engine_load: float = 0.8) -> Dict:
        """
        Calculate steady-state performance across ranges of ambient temps and heat loads.

        Returns:
            Dict with performance map data (steady state coolant temps, rejection rates, etc.).
        """
        n_ambient = len(ambient_temps_C)
        n_heat = len(engine_heats_W)
        steady_coolant_temps = np.zeros((n_ambient, n_heat))
        steady_rejection_rates = np.zeros((n_ambient, n_heat))
        steady_fan_duties = np.zeros((n_ambient, n_heat))

        logger.info(f"Calculating system performance map ({n_ambient} ambients x {n_heat} heat loads)...")

        # Store initial state
        initial_coolant_temp = self.coolant_temp_C

        for i, ambient in enumerate(ambient_temps_C):
            for j, heat in enumerate(engine_heats_W):
                # Reset temp for each point, start near ambient
                self.coolant_temp_C = ambient + 10.0
                last_temp = -999 # Force first check pass
                # Simulate until steady state (or timeout)
                max_iter = 500
                for k in range(max_iter):
                    state = self.simulate_step(ambient, vehicle_speed_mps, engine_rpm, engine_load, heat, dt=1.0) # 1s step
                    temp_change = abs(self.coolant_temp_C - last_temp)
                    if k > 10 and temp_change < 0.01: # Check after 10s, tolerance 0.01 C/s
                         break
                    last_temp = self.coolant_temp_C
                else:
                     logger.warning(f"Steady state not reached for ambient={ambient}, heat={heat/1000:.1f}kW")

                steady_coolant_temps[i, j] = self.coolant_temp_C
                steady_rejection_rates[i, j] = self.radiator_heat_rejection_W
                steady_fan_duties[i, j] = self.fan_control_signal

        # Restore initial state
        self.coolant_temp_C = initial_coolant_temp

        logger.info("Performance map calculation complete.")
        return {
            'ambient_temps_C': np.array(ambient_temps_C),
            'engine_heats_W': np.array(engine_heats_W),
            'coolant_temps_C': steady_coolant_temps,
            'heat_rejection_W': steady_rejection_rates,
            'fan_duty_cycles': steady_fan_duties,
            'conditions': {'speed': vehicle_speed_mps, 'rpm': engine_rpm, 'load': engine_load}
        }

    # --- Plotting Wrappers ---
    def plot_performance_map(self, performance_data: Dict, save_path: Optional[str] = None):
        """Plot the cooling system performance map."""
        from ..utils.plotting import plot_cooling_system_map, save_plot
        # Prepare data in the format expected by the plotting function
        plot_data = {
            'speeds': performance_data['ambient_temps_C'], # X-axis is ambient temp
            'engine_loads': performance_data['engine_heats_W'] / 1000.0, # Y-axis is heat load in kW
            'temperature_map': performance_data['coolant_temps_C'], # Z-axis is coolant temp
            'ambient_temperature': performance_data['conditions']['ambient'], # Add context
            # Include limits if needed
            'coolant_warning_temp': self.thermostat.full_open_temp_C + 8, # Example warning
            'coolant_critical_temp': self.thermostat.full_open_temp_C + 18 # Example critical
        }
        fig = plot_cooling_system_map(plot_data, title='Cooling System Performance Map')
        # Adjust labels for this specific plot
        if fig:
            axes = fig.get_axes()
            if axes:
                axes[0].set_xlabel('Ambient Temperature (°C)')
                axes[0].set_ylabel('Engine Heat Input (kW)')
                plt.tight_layout(rect=[0, 0, 1, 0.95])
            if save_path: save_plot(fig, save_path)
            else: plt.show()
            plt.close(fig)

    def plot_cooling_capacity(self, performance_data: Dict, save_path: Optional[str] = None):
        """Plot cooling capacity vs. ambient temperature."""
        from ..utils.plotting import save_plot # Local import

        ambient_temps = performance_data['ambient_temps_C']
        engine_heats = performance_data['engine_heats_W']
        coolant_temps = performance_data['coolant_temps_C']
        max_allowed_temp = 105.0 # Example limit

        cooling_capacities_kw = []
        for i in range(len(ambient_temps)):
            # Find max heat where coolant temp is <= max_allowed_temp
            valid_heat_indices = np.where(coolant_temps[i, :] <= max_allowed_temp)[0]
            if len(valid_heat_indices) > 0:
                max_heat_idx = valid_heat_indices[-1]
                # Interpolate if possible between last valid and first invalid
                if max_heat_idx + 1 < len(engine_heats):
                     t1, t2 = coolant_temps[i, max_heat_idx], coolant_temps[i, max_heat_idx + 1]
                     h1, h2 = engine_heats[max_heat_idx], engine_heats[max_heat_idx + 1]
                     if t2 > t1: # Ensure valid interpolation range
                          interp_heat = h1 + (h2 - h1) * (max_allowed_temp - t1) / (t2 - t1)
                          cooling_capacities_kw.append(interp_heat / 1000.0)
                     else:
                          cooling_capacities_kw.append(engine_heats[max_heat_idx] / 1000.0)
                else: # Can handle max tested heat
                     cooling_capacities_kw.append(engine_heats[max_heat_idx] / 1000.0)
            else: # Cannot handle even the lowest heat load
                cooling_capacities_kw.append(0.0)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(ambient_temps, cooling_capacities_kw, 'b-o', linewidth=DEFAULT_LINE_WIDTH)
        _apply_common_ax_settings(ax, xlabel='Ambient Temperature (°C)', ylabel='Cooling Capacity (kW)',
                                  title=f'System Cooling Capacity (Limit: {max_allowed_temp}°C Coolant)')

        plt.tight_layout()
        if save_path: save_plot(fig, save_path)
        plt.show()
        plt.close(fig)

# --- Factory Functions ---

def create_cbr600f4i_cooling_system(config_dir: str = "configs/thermal") -> CoolingSystem:
    """Create a cooling system based on default CBR600F4i parameters."""
    # Assumes separate config files exist for each component
    try:
        rad_path = os.path.join(config_dir, "radiator_cbr600.yaml") # Example path
        pump_path = os.path.join(config_dir, "pump_cbr600_mech.yaml")
        fan_path = os.path.join(config_dir, "fan_cbr600.yaml")
        thermo_path = os.path.join(config_dir, "thermostat_cbr600.yaml")
        system_path = os.path.join(config_dir, "system_cbr600.yaml")

        # Create components from potentially specific files
        radiator = Radiator(config_path=rad_path) if os.path.exists(rad_path) else Radiator()
        pump = WaterPump(config_path=pump_path) if os.path.exists(pump_path) else WaterPump(pump_type=PumpType.MECHANICAL)
        fan = CoolingFan(config_path=fan_path) if os.path.exists(fan_path) else CoolingFan()
        thermostat = Thermostat(config_path=thermo_path) if os.path.exists(thermo_path) else Thermostat()

        # Create system, potentially loading system-level params
        system = CoolingSystem(radiator, pump, fan, thermostat, config_path=system_path)
        logger.info("Created CBR600F4i default cooling system.")
        return system
    except Exception as e:
         logger.error(f"Failed to create CBR600F4i system from configs: {e}. Returning basic default.")
         return CoolingSystem(Radiator(), WaterPump(pump_type=PumpType.MECHANICAL), CoolingFan(), Thermostat())


def create_formula_student_cooling_system(config_dir: str = "configs/thermal") -> CoolingSystem:
    """Create an optimized cooling system typical for Formula Student."""
    # Assumes config files are tailored for FS (e.g., electric pump, potentially larger radiator)
    try:
        # Use generic config names defined in the project structure
        rad_path = os.path.join(config_dir, "cooling_system.yaml") # Radiator params might be in main file
        pump_path = os.path.join(config_dir, "cooling_system.yaml") # Pump params might be in main file
        fan_path = os.path.join(config_dir, "cooling_system.yaml") # Fan params might be in main file
        thermo_path = os.path.join(config_dir, "cooling_system.yaml")
        system_path = os.path.join(config_dir, "cooling_system.yaml") # System params

        # Create components using the main cooling_system config file
        radiator = Radiator(config_path=rad_path) if os.path.exists(rad_path) else \
                   Radiator(radiator_type=RadiatorType.DOUBLE_CORE_ALUMINUM, core_area=0.18) # FS default
        pump = WaterPump(config_path=pump_path) if os.path.exists(pump_path) else \
               WaterPump(pump_type=PumpType.ELECTRIC, max_flow_rate_lpm=75) # FS default
        fan = CoolingFan(config_path=fan_path) if os.path.exists(fan_path) else \
              CoolingFan(fan_type=FanType.VARIABLE_SPEED, max_airflow_m3s=0.35) # FS default
        thermostat = Thermostat(config_path=thermo_path) if os.path.exists(thermo_path) else \
                     Thermostat(opening_temp_C=80, full_open_temp_C=90) # FS default

        system = CoolingSystem(radiator, pump, fan, thermostat, config_path=system_path)
        logger.info("Created Formula Student optimized cooling system.")
        return system
    except Exception as e:
         logger.error(f"Failed to create FS system from configs: {e}. Returning basic default.")
         return CoolingSystem(Radiator(), WaterPump(), CoolingFan(), Thermostat())


# Example Usage
if __name__ == "__main__":
    # Create and test the FS optimized system
    fs_system = create_formula_student_cooling_system()

    print("\n--- Formula Student Cooling System Specs ---")
    specs = fs_system.get_system_specs()
    print(yaml.dump(specs, default_flow_style=False))

    # Simulate a step
    print("\n--- Simulating Step ---")
    state = fs_system.simulate_step(
        ambient_temp_C=30.0,
        vehicle_speed_mps=5.0, # Low speed
        engine_rpm=7000,
        engine_load=0.6,
        engine_heat_input_W=25000, # 25kW heat
        dt=1.0
    )
    print("State after 1 second step:")
    print(f" Coolant Temp: {state['coolant_temp_C']:.1f}°C")
    print(f" Heat Rejected: {state['radiator_heat_rejection_W']/1000:.1f} kW")
    print(f" Fan Duty: {state.get('cooling_fan',{}).get('duty_cycle',0)*100:.0f}%")
    print(f" Pump Flow: {state['water_pump']['flow_lpm']:.1f} LPM")

    # Analyze performance
    print("\n--- Analyzing Performance Map ---")
    ambients = [20, 25, 30, 35, 40]
    heats = [10000, 20000, 30000, 40000, 50000]
    perf_data = fs_system.calculate_system_performance(ambients, heats, vehicle_speed_mps=10.0)

    # Plot performance map
    fs_system.plot_performance_map(perf_data)

    # Plot cooling capacity
    fs_system.plot_cooling_capacity(perf_data)