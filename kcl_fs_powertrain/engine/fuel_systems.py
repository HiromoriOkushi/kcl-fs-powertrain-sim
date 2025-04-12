"""
Fuel systems module for Formula Student powertrain simulation.

Models fuel properties, injectors, pumps, and calculates consumption.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Callable
from enum import Enum, auto
import yaml
from scipy.interpolate import interp1d
import logging

# Import constants (assuming it's one level up in utils)
try:
    from ..utils.constants import FuelPropertiesConstants as FuelPropertiesLookup, LITERS_TO_M3, M3_TO_LITERS
except ImportError:
    # Define fallbacks if utils not available
    class FuelPropertiesLookup: # Mock class
        _PROPERTIES = {
            'GASOLINE_98RON': [0.75, 44.4, 14.7, 350, 98],
            'E85': [0.78, 29.2, 9.8, 850, 105],
            'E100': [0.79, 26.8, 9.0, 920, 108],
            'METHANOL': [0.79, 19.7, 6.5, 1100, 109]
        }
    LITERS_TO_M3 = 0.001
    M3_TO_LITERS = 1000.0

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("FuelSystems")


class FuelType(Enum):
    """Enumeration of available fuel types matching the keys in constants."""
    GASOLINE_98RON = auto()
    E85 = auto()
    E100 = auto()
    METHANOL = auto()


class FuelProperties:
    """
    Class representing the physical and chemical properties of a specific fuel.
    Uses the FuelPropertiesConstants lookup from the constants module.
    """
    def __init__(self, fuel_type: FuelType = FuelType.E85, custom_properties: Optional[Dict] = None):
        """
        Initialize fuel properties using lookup from constants.

        Args:
            fuel_type: FuelType enum member.
            custom_properties: Optional dict to override specific properties.
        """
        self.fuel_type = fuel_type
        fuel_type_name = fuel_type.name # Get the string name for lookup

        if fuel_type_name not in FuelPropertiesLookup._PROPERTIES:
            raise ValueError(f"Fuel type '{fuel_type_name}' not found in constants lookup.")

        props = FuelPropertiesLookup._PROPERTIES[fuel_type_name]
        self.density_kg_per_L = props[0]
        self.energy_density_MJ_per_kg = props[1]
        self.stoich_afr = props[2]
        self.latent_heat_kJ_per_kg = props[3]
        self.octane_ron = props[4]

        # Apply custom properties if provided
        if custom_properties:
            self.density_kg_per_L = custom_properties.get('density_kg_per_L', self.density_kg_per_L)
            self.energy_density_MJ_per_kg = custom_properties.get('energy_density_MJ_per_kg', self.energy_density_MJ_per_kg)
            self.stoich_afr = custom_properties.get('stoich_afr', self.stoich_afr)
            self.latent_heat_kJ_per_kg = custom_properties.get('latent_heat_kJ_per_kg', self.latent_heat_kJ_per_kg)
            self.octane_ron = custom_properties.get('octane_ron', self.octane_ron)

        # Derived properties in standard SI units (J/kg)
        self.energy_density_J_per_kg = self.energy_density_MJ_per_kg * 1e6
        self.latent_heat_J_per_kg = self.latent_heat_kJ_per_kg * 1000
        # Density in kg/m^3
        self.density_kg_per_m3 = self.density_kg_per_L * 1000

        logger.debug(f"Initialized FuelProperties for {fuel_type.name}")

    @classmethod
    def from_config(cls, config: Dict) -> 'FuelProperties':
        """
        Create a FuelProperties instance from a dictionary configuration.

        Args:
            config: Dictionary containing 'fuel_type' and optional property overrides.

        Returns:
            FuelProperties instance.
        """
        fuel_type_str = config.get('fuel_type', 'E85').upper()
        try:
            fuel_type = FuelType[fuel_type_str]
        except KeyError:
            raise ValueError(f"Invalid fuel type in config: {fuel_type_str}")

        custom_props = config.get('custom_properties', {})
        return cls(fuel_type, custom_props)

    def get_volumetric_energy_density_MJ_per_L(self) -> float:
        """Calculate volumetric energy density in MJ/L."""
        return self.density_kg_per_L * self.energy_density_MJ_per_kg

    def get_theoretical_power_kw(self, mass_flow_rate_g_s: float) -> float:
        """Calculate theoretical chemical power from fuel mass flow rate."""
        return (mass_flow_rate_g_s / 1000.0) * self.energy_density_J_per_kg / 1000.0 # Result in kW

    def get_cooling_effect_kw(self, mass_flow_rate_g_s: float) -> float:
        """Calculate cooling effect (power) due to fuel evaporation."""
        return (mass_flow_rate_g_s / 1000.0) * self.latent_heat_J_per_kg / 1000.0 # Result in kW

    def compare_with(self, other: 'FuelProperties') -> Dict:
        """Compare properties relative to another fuel."""
        if not isinstance(other, FuelProperties): raise TypeError("Comparison requires another FuelProperties instance.")
        return {
            'density_ratio': self.density_kg_per_L / other.density_kg_per_L,
            'energy_density_ratio': self.energy_density_MJ_per_kg / other.energy_density_MJ_per_kg,
            'volumetric_energy_ratio': self.get_volumetric_energy_density_MJ_per_L() / other.get_volumetric_energy_density_MJ_per_L(),
            'stoich_afr_ratio': self.stoich_afr / other.stoich_afr,
            'latent_heat_ratio': self.latent_heat_kJ_per_kg / other.latent_heat_kJ_per_kg,
            'octane_ratio': self.octane_ron / other.octane_ron
        }

    def to_dict(self) -> Dict:
        """Convert fuel properties to a dictionary."""
        return {
            'fuel_type': self.fuel_type.name,
            'density_kg_per_L': self.density_kg_per_L,
            'energy_density_MJ_per_kg': self.energy_density_MJ_per_kg,
            'stoich_afr': self.stoich_afr,
            'latent_heat_kJ_per_kg': self.latent_heat_kJ_per_kg,
            'octane_ron': self.octane_ron,
            'volumetric_energy_density_MJ_per_L': self.get_volumetric_energy_density_MJ_per_L()
        }


class FuelInjector:
    """Models a fuel injector."""

    def __init__(self, flow_rate_cc_min: float = 550.0, opening_time_ms: float = 1.0,
                 ref_pressure_bar: float = 3.0, min_pulse_width_ms: float = 0.9,
                 max_duty_cycle: float = 0.90): # Max duty cycle slightly higher
        """
        Initialize a fuel injector model.

        Args:
            flow_rate_cc_min: Injector static flow rate (cc/min) at reference pressure.
            opening_time_ms: Injector dead time/opening time (ms).
            ref_pressure_bar: Reference fuel pressure (gauge) for flow rate (bar).
            min_pulse_width_ms: Minimum effective pulse width (ms).
            max_duty_cycle: Maximum allowable duty cycle (0-1).
        """
        self.static_flow_rate_cc_min = flow_rate_cc_min
        self.opening_time_ms = opening_time_ms
        self.ref_pressure_bar = ref_pressure_bar
        self.min_pulse_width_ms = min_pulse_width_ms
        self.max_duty_cycle = max_duty_cycle

        # Operating conditions (can be updated)
        self.current_pressure_bar: float = ref_pressure_bar
        # Injector performance might also depend on voltage, but simplified here.

        logger.debug(f"Injector initialized: {flow_rate_cc_min} cc/min @ {ref_pressure_bar} bar")

    @classmethod
    def from_config(cls, config: Dict) -> 'FuelInjector':
        """Create FuelInjector from configuration dictionary."""
        return cls(
            flow_rate_cc_min=config.get('flow_rate_cc_min', 550.0),
            opening_time_ms=config.get('opening_time_ms', 1.0),
            ref_pressure_bar=config.get('ref_pressure_bar', 3.0),
            min_pulse_width_ms=config.get('min_pulse_width_ms', 0.9),
            max_duty_cycle=config.get('max_duty_cycle', 0.90)
        )

    def set_operating_conditions(self, pressure_bar: float):
        """Set the current operating fuel pressure."""
        self.current_pressure_bar = pressure_bar

    def get_dynamic_flow_rate_cc_min(self) -> float:
        """Calculate the dynamic flow rate based on current pressure."""
        # Flow rate scales with sqrt of pressure ratio relative to reference
        # Assuming negligible pressure drop across injector nozzle itself
        pressure_ratio = max(0, self.current_pressure_bar / self.ref_pressure_bar)
        return self.static_flow_rate_cc_min * np.sqrt(pressure_ratio)

    def get_flow_rate_g_s(self, fuel_density_kg_per_L: float) -> float:
        """Calculate dynamic flow rate in grams per second."""
        flow_cc_min = self.get_dynamic_flow_rate_cc_min()
        # Convert cc/min to L/s: (cc/min) * (1 L / 1000 cc) * (1 min / 60 s)
        flow_L_s = flow_cc_min / 60000.0
        # Convert L/s to kg/s: flow_L_s * density_kg_per_L
        flow_kg_s = flow_L_s * fuel_density_kg_per_L
        # Convert kg/s to g/s
        flow_g_s = flow_kg_s * 1000.0
        return flow_g_s

    def get_required_pulse_width_ms(self, fuel_mass_mg: float, fuel_density_kg_per_L: float) -> float:
        """Calculate required total pulse width (including dead time) for a fuel mass."""
        # Flow rate in mg/ms
        flow_g_s = self.get_flow_rate_g_s(fuel_density_kg_per_L)
        flow_mg_ms = flow_g_s # 1 g/s = 1 mg/ms

        if flow_mg_ms <= 1e-6: # Avoid division by zero
            return self.min_pulse_width_ms # Return min if no flow

        # Time needed to inject the fuel mass (injection time)
        injection_time_ms = fuel_mass_mg / flow_mg_ms

        # Total pulse width = injection time + opening time (dead time)
        total_pulse_width_ms = injection_time_ms + self.opening_time_ms

        # Ensure minimum pulse width is met
        return max(total_pulse_width_ms, self.min_pulse_width_ms)

    def get_max_fuel_mass_mg_per_cycle(self, engine_rpm: float, cylinders: int, fuel_density_kg_per_L: float) -> float:
        """Calculate the maximum fuel mass (mg) injectable per cycle at given RPM."""
        if engine_rpm <= 0: return 0.0

        # Time per engine cycle (ms) - depends on injection strategy (e.g., per intake stroke in 4-stroke)
        # Assume sequential injection, one injection per cylinder per 2 revolutions (720 degrees)
        cycle_time_ms = (120.0 * 1000.0) / engine_rpm # ms per 720 degrees crank angle

        # Maximum time the injector can be open per cycle
        max_effective_time_ms = cycle_time_ms * self.max_duty_cycle

        # Actual injection time possible
        injection_time_ms = max(0.0, max_effective_time_ms - self.opening_time_ms)

        # Flow rate in mg/ms
        flow_g_s = self.get_flow_rate_g_s(fuel_density_kg_per_L)
        flow_mg_ms = flow_g_s

        # Max mass = flow rate * injection time
        max_mass_mg = flow_mg_ms * injection_time_ms

        return max_mass_mg

    def calculate_duty_cycle(self, fuel_mass_mg: float, engine_rpm: float, cylinders: int, fuel_density_kg_per_L: float) -> float:
        """Calculate the required duty cycle for a given fuel mass and RPM."""
        if engine_rpm <= 0: return 0.0

        required_pulse_width_ms = self.get_required_pulse_width_ms(fuel_mass_mg, fuel_density_kg_per_L)

        # Time per engine cycle (ms)
        cycle_time_ms = (120.0 * 1000.0) / engine_rpm

        if cycle_time_ms <= 1e-6: return self.max_duty_cycle # Avoid division by zero at extreme RPM

        duty_cycle = required_pulse_width_ms / cycle_time_ms

        return min(duty_cycle, self.max_duty_cycle) # Cap at max duty cycle

    def is_adequate(self, required_flow_g_s: float, max_engine_rpm: float, cylinders: int, fuel_density_kg_per_L: float) -> bool:
        """Check if the injector can supply the required fuel flow at max RPM."""
        # Calculate max possible fuel mass per cycle at max RPM
        max_fuel_mass_mg_cycle = self.get_max_fuel_mass_mg_per_cycle(max_engine_rpm, cylinders, fuel_density_kg_per_L)

        # Calculate required fuel mass per cycle
        # Time per cycle (s) = 120 / max_engine_rpm
        cycle_time_s = 120.0 / max_engine_rpm
        required_fuel_mass_g_cycle = required_flow_g_s * cycle_time_s
        required_fuel_mass_mg_cycle = required_fuel_mass_g_cycle * 1000.0

        return max_fuel_mass_mg_cycle >= required_fuel_mass_mg_cycle

    def to_dict(self) -> Dict:
        """Convert injector properties to a dictionary."""
        return {
            'static_flow_rate_cc_min': self.static_flow_rate_cc_min,
            'opening_time_ms': self.opening_time_ms,
            'ref_pressure_bar': self.ref_pressure_bar,
            'min_pulse_width_ms': self.min_pulse_width_ms,
            'max_duty_cycle': self.max_duty_cycle,
            'current_pressure_bar': self.current_pressure_bar,
            'dynamic_flow_rate_cc_min': self.get_dynamic_flow_rate_cc_min()
        }


class FuelPump:
    """Models a fuel pump."""

    def __init__(self, max_pressure_bar: float = 4.5, max_flow_lph: float = 150.0,
                 nominal_voltage: float = 13.5, current_draw_A: float = 6.0,
                 min_voltage: float = 9.0):
        """
        Initialize a fuel pump model.

        Args:
            max_pressure_bar: Max pressure (deadhead) the pump can generate (bar).
            max_flow_lph: Max flow rate (at zero pressure) in Liters Per Hour (LPH).
            nominal_voltage: Voltage at which performance is rated (V).
            current_draw_A: Current draw at nominal voltage and typical operating point (A).
            min_voltage: Minimum voltage for pump operation (V).
        """
        self.max_pressure_bar = max_pressure_bar
        self.max_flow_lph = max_flow_lph
        self.nominal_voltage = nominal_voltage
        self.base_current_draw_A = current_draw_A
        self.min_voltage = min_voltage

        # Operating conditions
        self.current_voltage: float = nominal_voltage
        self.duty_cycle: float = 1.0 # For PWM controlled pumps (0-1)

        self._create_pressure_flow_curve()

    def _create_pressure_flow_curve(self):
        """Create a simplified linear pressure-flow curve at nominal voltage."""
        # Assumes flow decreases linearly from max flow at 0 pressure
        # to 0 flow at max pressure. Real curves are more complex.
        self._pressures_bar = np.array([0.0, self.max_pressure_bar])
        self._flows_lph = np.array([self.max_flow_lph, 0.0])
        self._pressure_flow_func = interp1d(self._pressures_bar, self._flows_lph,
                                           kind='linear', bounds_error=False,
                                           fill_value=(self.max_flow_lph, 0.0))

    @classmethod
    def from_config(cls, config: Dict) -> 'FuelPump':
        """Create FuelPump from configuration dictionary."""
        return cls(
            max_pressure_bar=config.get('max_pressure_bar', 4.5),
            max_flow_lph=config.get('max_flow_lph', 150.0),
            nominal_voltage=config.get('nominal_voltage', 13.5),
            current_draw_A=config.get('current_draw_A', 6.0),
            min_voltage=config.get('min_voltage', 9.0)
        )

    def set_operating_conditions(self, voltage_V: float, duty_cycle: float = 1.0):
        """Set the current operating voltage and duty cycle."""
        self.current_voltage = max(self.min_voltage, voltage_V)
        self.duty_cycle = np.clip(duty_cycle, 0.0, 1.0)

    def get_flow_rate_lph(self, system_pressure_bar: float) -> float:
        """Calculate the flow rate (LPH) at the given system pressure."""
        # Voltage effect on flow and pressure (simplified: scales linearly with voltage ratio)
        voltage_ratio = max(0, (self.current_voltage - self.min_voltage)) / max(1e-3, (self.nominal_voltage - self.min_voltage))
        voltage_factor = np.clip(voltage_ratio, 0.0, 1.5) # Allow some over-voltage benefit, cap it

        # Scale the pump curve based on voltage and duty cycle
        effective_max_flow = self.max_flow_lph * voltage_factor * self.duty_cycle
        effective_max_pressure = self.max_pressure_bar * voltage_factor * self.duty_cycle

        # Use the scaled curve to find flow at the system pressure
        # Recreate interp function with scaled values
        scaled_pressures = np.array([0.0, effective_max_pressure])
        scaled_flows = np.array([effective_max_flow, 0.0])
        scaled_func = interp1d(scaled_pressures, scaled_flows,
                               kind='linear', bounds_error=False,
                               fill_value=(effective_max_flow, 0.0))

        flow_rate = float(scaled_func(system_pressure_bar))
        return max(0.0, flow_rate) # Ensure non-negative flow

    def get_flow_rate_g_s(self, system_pressure_bar: float, fuel_density_kg_per_L: float) -> float:
        """Calculate flow rate in grams per second."""
        flow_lph = self.get_flow_rate_lph(system_pressure_bar)
        # Convert L/hr to kg/s: (L/hr) * (1 hr / 3600 s) * density_kg_per_L
        flow_kg_s = flow_lph / 3600.0 * fuel_density_kg_per_L
        # Convert kg/s to g/s
        flow_g_s = flow_kg_s * 1000.0
        return flow_g_s

    def get_power_consumption_W(self, system_pressure_bar: float) -> float:
        """Estimate electrical power consumption in Watts."""
        # Current draw often increases with pressure (load)
        pressure_ratio = np.clip(system_pressure_bar / self.max_pressure_bar, 0, 1)
        # Simple model: current increases slightly with pressure
        current_factor = 1.0 + 0.3 * pressure_ratio
        current_A = self.base_current_draw_A * current_factor * self.duty_cycle

        # Power = V * I
        power_W = self.current_voltage * current_A
        return power_W

    def is_adequate(self, required_flow_lph: float, system_pressure_bar: float) -> bool:
        """Check if pump can supply the required flow at system pressure."""
        max_flow_at_pressure = self.get_flow_rate_lph(system_pressure_bar)
        return max_flow_at_pressure >= required_flow_lph

    def to_dict(self) -> Dict:
        """Convert pump properties to a dictionary."""
        return {
            'max_pressure_bar': self.max_pressure_bar,
            'max_flow_lph': self.max_flow_lph,
            'nominal_voltage': self.nominal_voltage,
            'base_current_draw_A': self.base_current_draw_A,
            'min_voltage': self.min_voltage,
            'current_voltage': self.current_voltage,
            'duty_cycle': self.duty_cycle
        }

class FuelConsumption:
    """Analyzes and predicts fuel consumption."""

    def __init__(self, fuel_properties: FuelProperties, engine = None): # Engine type hint requires forward ref or import
        """
        Initialize the fuel consumption analyzer.

        Args:
            fuel_properties: FuelProperties instance.
            engine: Optional engine model instance (needed for map-based calculations).
        """
        self.fuel_properties = fuel_properties
        self.engine = engine # Store engine reference
        if engine is None:
            logger.warning("FuelConsumption initialized without an engine model. Some methods may fail.")

        # Storage for results
        self.consumption_map_gs = None # g/s map
        self.bsfc_map_gkwh = None # g/kWh map
        self.lap_consumption_history = [] # Store consumption per lap
        self.event_consumption = {} # Store estimated consumption for events

    def calculate_fuel_mass_flow_g_s(self, power_kw: float, bsfc_g_kwh: Optional[float] = None) -> float:
        """
        Calculate fuel mass flow rate (g/s) required for a given power output.

        Args:
            power_kw: Power output in kW.
            bsfc_g_kwh: Optional Brake Specific Fuel Consumption (g/kWh). If None, uses a default estimate.

        Returns:
            Fuel mass flow rate in g/s.
        """
        if power_kw <= 0: return 0.0 # No fuel needed for zero or negative power

        if bsfc_g_kwh is None:
            # Estimate BSFC based on fuel type if not provided
            # These are rough estimates, real BSFC varies significantly with RPM/load
            if self.fuel_properties.fuel_type == FuelType.E85: bsfc_g_kwh = 400.0
            elif self.fuel_properties.fuel_type == FuelType.GASOLINE_98RON: bsfc_g_kwh = 280.0
            elif self.fuel_properties.fuel_type == FuelType.E100: bsfc_g_kwh = 430.0
            elif self.fuel_properties.fuel_type == FuelType.METHANOL: bsfc_g_kwh = 600.0
            else: bsfc_g_kwh = 350.0 # Generic default

        # Calculate fuel mass flow rate: (g/kWh) * kW / (s/h) = g/s
        fuel_mass_g_s = (bsfc_g_kwh * power_kw) / 3600.0
        return fuel_mass_g_s

    def calculate_consumption_map(self, rpm_range: np.ndarray, throttle_range: np.ndarray) -> Optional[np.ndarray]:
        """
        Calculate a fuel consumption map (g/s) over RPM and throttle. Requires engine model.

        Args:
            rpm_range: Array of RPM points.
            throttle_range: Array of throttle positions (0-1).

        Returns:
            2D numpy array of fuel consumption in g/s, or None if engine model unavailable.
        """
        if self.engine is None:
            logger.error("Engine model required to calculate consumption map.")
            return None

        consumption_map = np.zeros((len(rpm_range), len(throttle_range)))
        bsfc_map = np.zeros_like(consumption_map) # Also calculate BSFC

        for i, rpm in enumerate(rpm_range):
            for j, throttle in enumerate(throttle_range):
                power_kw = self.engine.get_power(rpm, throttle) # Uses current engine temp

                # Use engine's fuel consumption method if available, otherwise estimate BSFC
                if hasattr(self.engine, 'get_fuel_consumption'):
                     consumption_g_s = self.engine.get_fuel_consumption(rpm, throttle)
                else:
                     # Fallback: use estimated BSFC - less accurate
                     consumption_g_s = self.calculate_fuel_mass_flow_g_s(power_kw)

                consumption_map[i, j] = consumption_g_s
                # Calculate BSFC
                bsfc_map[i,j] = (consumption_g_s * 3600.0 / power_kw) if power_kw > 0.1 else 0 # g/kWh

        self.consumption_map_gs = consumption_map
        self.bsfc_map_gkwh = bsfc_map
        return consumption_map

    def get_consumption_rate_g_s(self, rpm: float, throttle: float) -> float:
        """Get interpolated consumption rate (g/s) from the map."""
        if self.consumption_map_gs is None or self.engine is None:
            # Fallback: Calculate directly if map not generated
            power_kw = self.engine.get_power(rpm, throttle) if self.engine else 0
            return self.calculate_fuel_mass_flow_g_s(power_kw)

        # Need RPM and Throttle ranges used to generate the map
        if not hasattr(self, 'map_rpm_range') or not hasattr(self, 'map_throttle_range'):
             logger.warning("Consumption map axes ranges not stored. Cannot interpolate.")
             # Fallback
             power_kw = self.engine.get_power(rpm, throttle) if self.engine else 0
             return self.calculate_fuel_mass_flow_g_s(power_kw)

        # Interpolate (requires storing ranges used in calculate_consumption_map)
        # Placeholder for 2D interpolation logic (e.g., using scipy.interpolate.interp2d or RegularGridInterpolator)
        # For simplicity, find nearest points or use direct calculation for now
        power_kw = self.engine.get_power(rpm, throttle) if self.engine else 0
        return self.calculate_fuel_mass_flow_g_s(power_kw)


    def calculate_track_consumption(self, time_s: np.ndarray, rpm: np.ndarray, throttle: np.ndarray) -> Dict:
        """
        Calculate fuel consumption over a drive cycle (time series data).

        Args:
            time_s: Array of time points (seconds).
            rpm: Array of engine RPMs.
            throttle: Array of throttle positions (0-1).

        Returns:
            Dictionary with consumption results (total mass, volume, rate array).
        """
        if self.engine is None:
             logger.error("Engine model required for track consumption calculation.")
             return {'error': "Engine model required"}
        if not (len(time_s) == len(rpm) == len(throttle)):
             raise ValueError("Input arrays (time, rpm, throttle) must have the same length.")

        consumption_rate_gs = np.zeros_like(time_s)
        power_kw = np.zeros_like(time_s)

        for i in range(len(time_s)):
            # Get power using the engine model
            power_kw[i] = self.engine.get_power(rpm[i], throttle[i])
            # Get consumption rate (g/s)
            if hasattr(self.engine, 'get_fuel_consumption'):
                 consumption_rate_gs[i] = self.engine.get_fuel_consumption(rpm[i], throttle[i])
            else:
                 consumption_rate_gs[i] = self.calculate_fuel_mass_flow_g_s(power_kw[i])

        # Integrate consumption rate over time
        dt = np.diff(time_s, prepend=time_s[0]) # Calculate time steps
        fuel_mass_g = np.sum(consumption_rate_gs * dt)
        fuel_volume_L = fuel_mass_g / (self.fuel_properties.density_kg_per_L * 1000.0)

        # Store history for potential later use (e.g., per lap)
        self.lap_consumption_history.append({'mass_g': fuel_mass_g, 'volume_L': fuel_volume_L})

        return {
            'total_mass_g': fuel_mass_g,
            'total_volume_L': fuel_volume_L,
            'average_rate_gs': np.mean(consumption_rate_gs),
            'consumption_rate_gs': consumption_rate_gs, # Store the array
            'power_kw': power_kw # Store power array
        }

    def _estimate_event_consumption(self, avg_power_kw: float, duration_s: float, num_runs: int, safety_factor: float) -> Tuple[float, float]:
        """Helper to estimate consumption for timed events."""
        consumption_rate_gs = self.calculate_fuel_mass_flow_g_s(avg_power_kw)
        mass_per_run_g = consumption_rate_gs * duration_s
        total_mass_g = mass_per_run_g * num_runs * safety_factor
        total_volume_L = total_mass_g / (self.fuel_properties.density_kg_per_L * 1000.0)
        return total_mass_g, total_volume_L

    def estimate_all_event_requirements(self, params: Dict, safety_factors: Dict = None) -> Dict:
        """
        Estimate fuel requirements for all standard FS dynamic events.

        Args:
            params: Dictionary containing event parameters like:
                'endurance_laps', 'endurance_avg_lap_time_s', 'endurance_avg_power_kw',
                'accel_runs', 'accel_duration_s', 'accel_avg_power_kw',
                'autocross_laps', 'autocross_avg_lap_time_s', 'autocross_avg_power_kw',
                'skidpad_runs', 'skidpad_duration_s', 'skidpad_avg_power_kw'.
            safety_factors: Optional dict of safety factors per event (e.g., {'endurance': 1.2}).

        Returns:
            Dictionary containing estimated fuel volume (L) for each event and total.
        """
        if safety_factors is None:
            safety_factors = {'endurance': 1.15, 'acceleration': 1.5, 'autocross': 1.2, 'skidpad': 1.3}

        self.event_consumption = {}
        total_volume_L = 0.0

        # Endurance
        if 'endurance_laps' in params and 'endurance_avg_power_kw' in params and 'endurance_avg_lap_time_s' in params:
            mass, vol = self._estimate_event_consumption(
                params['endurance_avg_power_kw'],
                params['endurance_avg_lap_time_s'],
                params['endurance_laps'],
                safety_factors.get('endurance', 1.15)
            )
            self.event_consumption['endurance'] = {'mass_g': mass, 'volume_L': vol}
            total_volume_L += vol

        # Acceleration
        if 'accel_runs' in params and 'accel_duration_s' in params and 'accel_avg_power_kw' in params:
             mass, vol = self._estimate_event_consumption(
                params['accel_avg_power_kw'],
                params['accel_duration_s'],
                params['accel_runs'],
                safety_factors.get('acceleration', 1.5)
            )
             self.event_consumption['acceleration'] = {'mass_g': mass, 'volume_L': vol}
             total_volume_L += vol

        # Autocross
        if 'autocross_laps' in params and 'autocross_avg_lap_time_s' in params and 'autocross_avg_power_kw' in params:
             mass, vol = self._estimate_event_consumption(
                params['autocross_avg_power_kw'],
                params['autocross_avg_lap_time_s'],
                params['autocross_laps'],
                safety_factors.get('autocross', 1.2)
            )
             self.event_consumption['autocross'] = {'mass_g': mass, 'volume_L': vol}
             total_volume_L += vol

        # Skidpad
        if 'skidpad_runs' in params and 'skidpad_duration_s' in params and 'skidpad_avg_power_kw' in params:
            mass, vol = self._estimate_event_consumption(
                params['skidpad_avg_power_kw'],
                params['skidpad_duration_s'],
                params['skidpad_runs'],
                safety_factors.get('skidpad', 1.3)
            )
            self.event_consumption['skidpad'] = {'mass_g': mass, 'volume_L': vol}
            total_volume_L += vol

        self.event_consumption['total_volume_L'] = total_volume_L
        logger.info(f"Estimated total fuel needed: {total_volume_L:.2f} L")
        return self.event_consumption

    # --- Plotting Methods ---
    # (Implementations will use the centralized plotting functions from utils)

    def plot_consumption_profile(self, consumption_results: Dict, save_path: Optional[str] = None):
        """Plot time-series consumption data from calculate_track_consumption."""
        if 'time' not in consumption_results or 'consumption_rate_gs' not in consumption_results:
            logger.error("Cannot plot consumption profile: Missing required data.")
            return

        from ..utils.plotting import save_plot # Local import

        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

        time = consumption_results['time']
        power = consumption_results.get('power_kw', [])
        cons_rate = consumption_results['consumption_rate_gs']
        cum_cons = np.cumsum(cons_rate * np.diff(time, prepend=time[0]))

        # Plot Power
        if len(power) == len(time):
             axes[0].plot(time, power, color=COLOR_SCHEMES['default'][1], label='Power (kW)')
             axes[0].set_ylabel('Power (kW)', color=COLOR_SCHEMES['default'][1])
             axes[0].tick_params(axis='y', labelcolor=COLOR_SCHEMES['default'][1])
        _apply_common_ax_settings(axes[0], title='Power Profile')

        # Plot Consumption Rate
        axes[1].plot(time, cons_rate, color=COLOR_SCHEMES['default'][2], label='Fuel Rate (g/s)')
        _apply_common_ax_settings(axes[1], ylabel='Fuel Rate (g/s)', title='Fuel Consumption Rate')

        # Plot Cumulative Consumption
        axes[2].plot(time, cum_cons, color=COLOR_SCHEMES['default'][3], label='Cumulative Fuel (g)')
        _apply_common_ax_settings(axes[2], xlabel='Time (s)', ylabel='Cumulative Fuel (g)', title='Cumulative Fuel Use')

        fig.suptitle(f'Drive Cycle Fuel Consumption ({self.fuel_properties.fuel_type.name})')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        if save_path: save_plot(fig, save_path)
        plt.show()
        plt.close(fig)

    def plot_bsfc_map(self, rpm_range: Optional[np.ndarray] = None, throttle_range: Optional[np.ndarray] = None,
                    save_path: Optional[str] = None):
        """Plot the BSFC map (g/kWh)."""
        if self.bsfc_map_gkwh is None:
             if rpm_range is None or throttle_range is None:
                 logger.error("BSFC map not calculated and ranges not provided.")
                 return
             # Try calculating the map now
             self.calculate_consumption_map(rpm_range, throttle_range)
             if self.bsfc_map_gkwh is None:
                 logger.error("Failed to calculate BSFC map.")
                 return
        # Retrieve stored ranges if calculation was done previously
        rpm_map = getattr(self, 'map_rpm_range', rpm_range)
        throttle_map = getattr(self, 'map_throttle_range', throttle_range)
        if rpm_map is None or throttle_map is None:
            logger.error("RPM/Throttle ranges for BSFC map are missing.")
            return

        from ..utils.plotting import save_plot # Local import

        fig, ax = plt.subplots(figsize=(10, 8))
        X, Y = np.meshgrid(rpm_map, throttle_map * 100) # Throttle as %

        # Plot filled contour
        # Limit contours for better visualization, clip extreme BSFC values
        bsfc_clipped = np.clip(self.bsfc_map_gkwh, 250, 800)
        contour = ax.contourf(X, Y, bsfc_clipped.T, levels=15, cmap='viridis_r') # Lower is better
        cbar = plt.colorbar(contour)
        cbar.set_label('BSFC (g/kWh)')

        # Add contour lines
        contour_lines = ax.contour(X, Y, self.bsfc_map_gkwh.T, levels=[300, 350, 400, 500, 600], colors='white', alpha=0.7)
        ax.clabel(contour_lines, inline=True, fontsize=8, fmt='%1.0f')

        _apply_common_ax_settings(ax, xlabel='Engine RPM', ylabel='Throttle (%)', title=f'BSFC Map ({self.fuel_properties.fuel_type.name})')

        plt.tight_layout()
        if save_path: save_plot(fig, save_path)
        plt.show()
        plt.close(fig)

    def plot_event_requirements(self, save_path: Optional[str] = None):
        """Plot estimated fuel requirements by event."""
        if not self.event_consumption or 'total_volume_L' not in self.event_consumption:
            logger.error("Event consumption not estimated. Run estimate_all_event_requirements first.")
            return

        from ..utils.plotting import save_plot # Local import

        events = [e for e in self.event_consumption if e != 'total_volume_L']
        volumes = [self.event_consumption[e]['volume_L'] for e in events]
        total_volume = self.event_consumption['total_volume_L']

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(events, volumes, color=COLOR_SCHEMES['default'][:len(events)])

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.05 * total_volume,
                    f'{height:.2f} L', ha='center', va='bottom', fontsize=9)

        ax.axhline(total_volume, color='r', linestyle='--', label=f'Total: {total_volume:.2f} L')
        _apply_common_ax_settings(ax, xlabel='Event', ylabel='Estimated Fuel (L)', title=f'Estimated Fuel Requirements ({self.fuel_properties.fuel_type.name})')
        ax.legend()

        plt.tight_layout()
        if save_path: save_plot(fig, save_path)
        plt.show()
        plt.close(fig)

    def to_dict(self) -> Dict:
        """Convert consumption analysis data to a dictionary."""
        # Basic structure, can be expanded
        return {
            'fuel_properties': self.fuel_properties.to_dict(),
            'event_consumption_estimates': self.event_consumption,
            'lap_consumption_history': self.lap_consumption_history
            # Optionally add map data if needed, but can be large
            # 'bsfc_map_g_kwh': self.bsfc_map_gkwh.tolist() if self.bsfc_map_gkwh is not None else None,
        }


class FuelSystem:
    """Represents the complete fuel system."""

    def __init__(self, fuel_properties: FuelProperties, fuel_pump: FuelPump,
                 injectors: List[FuelInjector], tank_capacity_L: float = 7.0,
                 pressure_regulator_bar: float = 3.5): # Common FS pressure
        """
        Initialize the fuel system.

        Args:
            fuel_properties: FuelProperties instance.
            fuel_pump: FuelPump instance.
            injectors: List of FuelInjector instances (one per cylinder).
            tank_capacity_L: Fuel tank capacity in Liters.
            pressure_regulator_bar: Fuel pressure regulator setting (gauge, bar).
        """
        if not injectors: raise ValueError("Fuel system must have at least one injector.")

        self.fuel_properties = fuel_properties
        self.fuel_pump = fuel_pump
        self.injectors = injectors
        self.num_injectors = len(injectors)
        self.tank_capacity_L = tank_capacity_L
        self.pressure_regulator_bar = pressure_regulator_bar

        # State variables
        self.current_fuel_level_L: float = tank_capacity_L
        self.current_pressure_bar: float = pressure_regulator_bar
        self.current_voltage_V: float = 13.5 # Assume nominal voltage initially

        self.set_operating_conditions(self.current_voltage_V) # Initialize component states
        logger.info(f"Fuel System initialized: {self.fuel_properties.fuel_type.name}, {tank_capacity_L}L tank, {self.num_injectors} injectors.")

    @classmethod
    def from_config(cls, config: Dict) -> 'FuelSystem':
        """Create FuelSystem from configuration dictionary."""
        # Create FuelProperties
        fuel_props = FuelProperties.from_config(config.get('fuel_properties', {}))

        # Create FuelPump
        pump = FuelPump.from_config(config.get('fuel_pump', {}))

        # Create Injectors
        injector_config = config.get('injector', {})
        num_cylinders = config.get('num_cylinders', 4)
        injectors = [FuelInjector.from_config(injector_config) for _ in range(num_cylinders)]

        return cls(
            fuel_properties=fuel_props,
            fuel_pump=pump,
            injectors=injectors,
            tank_capacity_L=config.get('tank_capacity_L', 7.0),
            pressure_regulator_bar=config.get('pressure_regulator_bar', 3.5)
        )

    def set_operating_conditions(self, voltage_V: float, pump_duty_cycle: float = 1.0):
        """Set voltage and pump duty cycle, updating system pressure."""
        self.current_voltage_V = voltage_V
        self.fuel_pump.set_operating_conditions(voltage_V, pump_duty_cycle)

        # Pressure is set by the regulator, assuming pump can supply it
        # A more complex model could check if pump pressure < regulator setting
        self.current_pressure_bar = self.pressure_regulator_bar

        # Update injectors with current pressure
        for injector in self.injectors:
            injector.set_operating_conditions(self.current_pressure_bar)

    def calculate_required_flow_g_s(self, required_power_kw: float, bsfc_g_kwh: Optional[float] = None) -> float:
        """Calculate the fuel flow (g/s) needed for a given power output."""
        cons = FuelConsumption(self.fuel_properties)
        return cons.calculate_fuel_mass_flow_g_s(required_power_kw, bsfc_g_kwh)

    def calculate_max_supported_power_kw(self, engine_rpm: float, bsfc_g_kwh: Optional[float] = None) -> float:
        """Calculate the maximum power (kW) the fuel system can support at a given RPM."""
        # Max flow rate is limited by either pump or total injector capacity

        # 1. Pump limit
        pump_flow_lph = self.fuel_pump.get_flow_rate_lph(self.current_pressure_bar)
        pump_flow_g_s = pump_flow_lph / 3600.0 * self.fuel_properties.density_kg_per_L * 1000.0

        # 2. Injector limit
        max_injector_mass_mg_cycle = self.injectors[0].get_max_fuel_mass_mg_per_cycle(
            engine_rpm, self.num_injectors, self.fuel_properties.density_kg_per_L
        )
        # Convert max mass per cycle per injector to total g/s
        # Cycles per second = engine_rpm / 120 (for 4-stroke)
        cycles_per_sec = engine_rpm / 120.0
        injector_flow_g_s = (max_injector_mass_mg_cycle / 1000.0) * cycles_per_sec * self.num_injectors

        # The system is limited by the minimum of the two
        max_system_flow_g_s = min(pump_flow_g_s, injector_flow_g_s)

        # Estimate BSFC if not provided
        if bsfc_g_kwh is None:
            if self.fuel_properties.fuel_type == FuelType.E85: bsfc_g_kwh = 400.0
            else: bsfc_g_kwh = 350.0 # Generic default

        # Calculate max power from max flow rate: kW = (g/s) * 3600 / (g/kWh)
        max_power_kw = (max_system_flow_g_s * 3600.0) / bsfc_g_kwh if bsfc_g_kwh > 0 else 0.0
        return max_power_kw

    def update_fuel_level(self, consumption_rate_gs: float, dt_s: float) -> float:
        """Update fuel level based on consumption rate and time step."""
        consumed_mass_g = consumption_rate_gs * dt_s
        consumed_volume_L = consumed_mass_g / (self.fuel_properties.density_kg_per_L * 1000.0)
        self.current_fuel_level_L = max(0.0, self.current_fuel_level_L - consumed_volume_L)
        return self.current_fuel_level_L

    def is_fuel_sufficient(self, required_volume_L: float) -> bool:
        """Check if current fuel level is sufficient."""
        return self.current_fuel_level_L >= required_volume_L

    def validate_system(self, max_power_kw: float, max_rpm: float) -> Dict:
        """Validate if the fuel system components are adequate for the max requirements."""
        required_flow_g_s = self.calculate_required_flow_g_s(max_power_kw)
        required_flow_lph = required_flow_g_s * 3600.0 / (self.fuel_properties.density_kg_per_L * 1000.0)

        # Check pump adequacy
        pump_adequate = self.fuel_pump.is_adequate(required_flow_lph, self.pressure_regulator_bar)
        pump_max_flow = self.fuel_pump.get_flow_rate_lph(self.pressure_regulator_bar)

        # Check injector adequacy
        injectors_adequate = self.injectors[0].is_adequate(
            required_flow_g_s / self.num_injectors, # g/s per injector
            max_rpm,
            self.num_injectors,
            self.fuel_properties.density_kg_per_L
        )
        max_injector_mass_cycle = self.injectors[0].get_max_fuel_mass_mg_per_cycle(max_rpm, self.num_injectors, self.fuel_properties.density_kg_per_L)
        max_injector_flow_g_s = (max_injector_mass_cycle / 1000.0) * (max_rpm / 120.0) * self.num_injectors

        system_adequate = pump_adequate and injectors_adequate
        limiting_factor = "None"
        if not system_adequate:
             limiting_factor = "Pump" if not pump_adequate else "Injectors"

        return {
            'system_adequate': system_adequate,
            'pump_adequate': pump_adequate,
            'injectors_adequate': injectors_adequate,
            'required_flow_lph': required_flow_lph,
            'pump_max_flow_lph': pump_max_flow,
            'injectors_max_flow_g_s': max_injector_flow_g_s,
            'limiting_factor': limiting_factor
        }

    def to_dict(self) -> Dict:
        """Convert fuel system state and specs to a dictionary."""
        return {
            'fuel_properties': self.fuel_properties.to_dict(),
            'fuel_pump': self.fuel_pump.to_dict(),
            'injectors': [inj.to_dict() for inj in self.injectors],
            'num_injectors': self.num_injectors,
            'tank_capacity_L': self.tank_capacity_L,
            'pressure_regulator_bar': self.pressure_regulator_bar,
            'current_fuel_level_L': self.current_fuel_level_L,
            'current_pressure_bar': self.current_pressure_bar,
            'current_voltage_V': self.current_voltage_V
        }

# Example Usage
if __name__ == "__main__":
    # --- Fuel Properties Example ---
    e85_props = FuelProperties(FuelType.E85)
    gasoline_props = FuelProperties(fuel_type=FuelType.GASOLINE_98RON)
    print("E85 Properties:", e85_props.to_dict())
    print("\nE85 vs Gasoline Comparison:", e85_props.compare_with(gasoline_props))
    print(f"E85 Volumetric Energy: {e85_props.get_volumetric_energy_density_MJ_per_L():.2f} MJ/L")
    print(f"Gasoline Volumetric Energy: {gasoline_props.get_volumetric_energy_density_MJ_per_L():.2f} MJ/L")

    # --- Injector Example ---
    injector = FuelInjector(flow_rate_cc_min=650, opening_time_ms=0.8)
    injector.set_operating_conditions(pressure_bar=3.5)
    print("\nInjector Example:")
    print(f"  Dynamic Flow Rate: {injector.get_dynamic_flow_rate_cc_min():.1f} cc/min")
    fuel_mass_mg = 15.0 # mg per injection
    pulse_width = injector.get_required_pulse_width_ms(fuel_mass_mg, e85_props.density_kg_per_L)
    print(f"  Pulse width for {fuel_mass_mg}mg E85: {pulse_width:.2f} ms")
    duty_cycle = injector.calculate_duty_cycle(fuel_mass_mg, 12000, 4, e85_props.density_kg_per_L)
    print(f"  Duty cycle at 12000 RPM: {duty_cycle:.1%}")
    max_mass = injector.get_max_fuel_mass_mg_per_cycle(14000, 4, e85_props.density_kg_per_L)
    print(f"  Max fuel per cycle at 14000 RPM: {max_mass:.1f} mg")

    # --- Pump Example ---
    pump = FuelPump(max_pressure_bar=5.0, max_flow_lph=200)
    pump.set_operating_conditions(voltage_V=13.8)
    print("\nPump Example:")
    pressure = 3.5 # bar
    flow = pump.get_flow_rate_lph(pressure)
    power = pump.get_power_consumption_W(pressure)
    print(f"  Flow at {pressure} bar: {flow:.1f} LPH")
    print(f"  Power consumption: {power:.1f} W")
    print(f"  Is pump adequate for 100 LPH? {pump.is_adequate(100, pressure)}")

    # --- Fuel System Example ---
    # Assume we create a basic FuelSystem instance here (as in the class example)
    # system_config = {'fuel_type':'E85', 'fuel_pump': {}, 'injector':{}, 'num_cylinders':4}
    # fuel_system = FuelSystem.from_config(system_config)
    # validation = fuel_system.validate_system(max_power_kw=70, max_rpm=14000)
    # print("\nFuel System Validation (70kW @ 14k RPM):", validation)
    # fuel_system.update_fuel_level(consumption_rate_gs=5.0, dt_s=60) # Consume fuel for 60s
    # print(f"Fuel level after 60s: {fuel_system.current_fuel_level_L:.2f} L")

    # --- Consumption Example ---
    # Requires an engine model instance
    # cons_analyzer = FuelConsumption(e85_props, engine_instance)
    # event_params = { 'endurance_laps': 22, ... } # Define avg power/time for events
    # fuel_reqs = cons_analyzer.estimate_all_event_requirements(event_params)
    # cons_analyzer.plot_event_requirements()