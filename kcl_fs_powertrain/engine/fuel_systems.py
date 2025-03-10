"""
Fuel systems module for Formula Student powertrain simulation.

This module provides classes and functions for modeling fuel systems in Formula Student
vehicles, including different fuel types, injectors, fuel pumps, and consumption patterns.
It integrates with the engine module to provide comprehensive fuel delivery and consumption
analysis capabilities.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Callable
from enum import Enum, auto
import yaml
from scipy.interpolate import interp1d

# Define module exports
__all__ = [
    'FuelType', 'FuelProperties', 'FuelInjector', 
    'FuelPump', 'FuelConsumption', 'FuelSystem'
]


class FuelType(Enum):
    """Enumeration of available fuel types for Formula Student engines."""
    GASOLINE = auto()
    E85 = auto()
    E100 = auto()
    METHANOL = auto()


class FuelProperties:
    """Class representing the physical and chemical properties of different fuels."""
    
    # Default fuel properties dictionary
    # Values: [density (kg/L), energy density (MJ/kg), stoichiometric AFR, latent heat (kJ/kg), octane rating]
    _DEFAULT_PROPERTIES = {
        FuelType.GASOLINE: [0.745, 44.4, 14.7, 380, 98],
        FuelType.E85: [0.781, 29.2, 9.8, 825, 105],
        FuelType.E100: [0.789, 26.8, 9.0, 920, 108.6],
        FuelType.METHANOL: [0.792, 19.9, 6.4, 1100, 108]
    }
    
    def __init__(self, fuel_type: FuelType = FuelType.E85, custom_properties: Optional[Dict] = None):
        """
        Initialize fuel properties for the specified fuel type.
        
        Args:
            fuel_type: Type of fuel from FuelType enum
            custom_properties: Optional custom properties to override defaults
        """
        self.fuel_type = fuel_type
        
        # Set default properties for the selected fuel
        props = self._DEFAULT_PROPERTIES[fuel_type]
        self.density = props[0]  # kg/L
        self.energy_density = props[1]  # MJ/kg
        self.stoich_afr = props[2]  # Air-fuel ratio
        self.latent_heat = props[3]  # kJ/kg
        self.octane_rating = props[4]  # RON
        
        # Override with custom properties if provided
        if custom_properties:
            for key, value in custom_properties.items():
                if key in self.__dict__ and key != "fuel_type":
                    setattr(self, key, value)
    
    @classmethod
    def from_config(cls, config_path: str) -> 'FuelProperties':
        """
        Create a FuelProperties instance from a YAML configuration file.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            FuelProperties instance
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Fuel configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Get fuel type
        fuel_type_str = config.get('fuel_type', 'E85').upper()
        try:
            fuel_type = FuelType[fuel_type_str]
        except KeyError:
            raise ValueError(f"Invalid fuel type: {fuel_type_str}")
        
        # Get custom properties if any
        custom_props = {}
        for prop in ['density', 'energy_density', 'stoich_afr', 'latent_heat', 'octane_rating']:
            if prop in config:
                custom_props[prop] = config[prop]
        
        return cls(fuel_type, custom_props)
    
    def get_volumetric_energy_density(self) -> float:
        """
        Calculate volumetric energy density.
        
        Returns:
            Volumetric energy density in MJ/L
        """
        return self.density * self.energy_density
    
    def get_theoretical_power(self, mass_flow_rate: float) -> float:
        """
        Calculate theoretical power from fuel mass flow rate.
        
        Args:
            mass_flow_rate: Fuel mass flow rate in g/s
            
        Returns:
            Theoretical power in kW
        """
        # Convert g/s to kg/s and MJ/kg to kJ/kg
        return (mass_flow_rate / 1000) * self.energy_density * 1000
    
    def get_cooling_effect(self, mass_flow_rate: float) -> float:
        """
        Calculate cooling effect due to fuel evaporation.
        
        Args:
            mass_flow_rate: Fuel mass flow rate in g/s
            
        Returns:
            Cooling power in kW
        """
        # Convert g/s to kg/s and return kW
        return (mass_flow_rate / 1000) * self.latent_heat
    
    def compare_with(self, other: 'FuelProperties') -> Dict:
        """
        Compare properties with another fuel.
        
        Args:
            other: Another FuelProperties instance
            
        Returns:
            Dictionary with comparison results
        """
        return {
            'density_ratio': self.density / other.density,
            'energy_density_ratio': self.energy_density / other.energy_density,
            'volumetric_energy_ratio': self.get_volumetric_energy_density() / other.get_volumetric_energy_density(),
            'stoich_afr_ratio': self.stoich_afr / other.stoich_afr,
            'latent_heat_ratio': self.latent_heat / other.latent_heat,
            'octane_ratio': self.octane_rating / other.octane_rating
        }
    
    def to_dict(self) -> Dict:
        """
        Convert fuel properties to dictionary.
        
        Returns:
            Dictionary with fuel properties
        """
        return {
            'fuel_type': self.fuel_type.name,
            'density': self.density,
            'energy_density': self.energy_density,
            'stoich_afr': self.stoich_afr,
            'latent_heat': self.latent_heat,
            'octane_rating': self.octane_rating,
            'volumetric_energy_density': self.get_volumetric_energy_density()
        }


class FuelInjector:
    """Class modeling a fuel injector for a Formula Student engine."""
    
    def __init__(self, flow_rate: float = 550.0, opening_time: float = 1.0, 
                battery_voltage: float = 13.8, min_pulse_width: float = 0.9,
                max_duty_cycle: float = 0.85):
        """
        Initialize a fuel injector model.
        
        Args:
            flow_rate: Injector flow rate in cc/min at reference pressure
            opening_time: Injector opening time in ms
            battery_voltage: Battery voltage in V
            min_pulse_width: Minimum pulse width in ms
            max_duty_cycle: Maximum duty cycle (0-1)
        """
        self.flow_rate = flow_rate  # cc/min
        self.opening_time = opening_time  # ms
        self.battery_voltage = battery_voltage  # V
        self.min_pulse_width = min_pulse_width  # ms
        self.max_duty_cycle = max_duty_cycle  # 0-1
        
        # Reference pressure for flow rate rating (typically 3 bar)
        self.reference_pressure = 3.0  # bar
        
        # Current state
        self.current_pressure = 3.0  # bar
        self.current_voltage = battery_voltage  # V
    
    @classmethod
    def from_config(cls, config_path: str) -> 'FuelInjector':
        """
        Create a FuelInjector instance from a YAML configuration file.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            FuelInjector instance
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Injector configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Create injector with configuration values
        return cls(
            flow_rate=config.get('flow_rate', 550.0),
            opening_time=config.get('opening_time', 1.0),
            battery_voltage=config.get('battery_voltage', 13.8),
            min_pulse_width=config.get('min_pulse_width', 0.9),
            max_duty_cycle=config.get('max_duty_cycle', 0.85)
        )
    
    def set_operating_conditions(self, pressure: float, voltage: float):
        """
        Set the current operating conditions for the injector.
        
        Args:
            pressure: Fuel pressure in bar
            voltage: Battery voltage in V
        """
        self.current_pressure = pressure
        self.current_voltage = voltage
    
    def get_flow_rate(self, pressure: Optional[float] = None) -> float:
        """
        Calculate the adjusted flow rate based on current pressure.
        
        Args:
            pressure: Optional fuel pressure in bar (uses current_pressure if None)
            
        Returns:
            Adjusted flow rate in cc/min
        """
        if pressure is None:
            pressure = self.current_pressure
        
        # Flow rate scales with square root of pressure differential
        pressure_factor = np.sqrt(pressure / self.reference_pressure)
        return self.flow_rate * pressure_factor
    
    def get_pulse_width(self, fuel_mass: float, fuel_density: float = 0.781) -> float:
        """
        Calculate required pulse width to inject a specific mass of fuel.
        
        Args:
            fuel_mass: Fuel mass to inject in mg
            fuel_density: Fuel density in g/cc (default is E85)
            
        Returns:
            Pulse width in ms
        """
        # Convert flow rate from cc/min to mg/ms
        flow_rate_mg_ms = self.get_flow_rate() * fuel_density / 60
        
        # Calculate base pulse width
        base_pulse_width = fuel_mass / flow_rate_mg_ms
        
        # Adjust for opening time
        adjusted_pulse_width = base_pulse_width + self.opening_time
        
        # Ensure minimum pulse width
        return max(adjusted_pulse_width, self.min_pulse_width)
    
    def get_max_flow_rate(self, engine_rpm: float, cylinders: int = 4) -> float:
        """
        Calculate maximum possible flow rate at given RPM.
        
        Args:
            engine_rpm: Engine speed in RPM
            cylinders: Number of engine cylinders
            
        Returns:
            Maximum flow rate in cc/min
        """
        # Calculate injections per minute
        injections_per_min = engine_rpm * (cylinders / 2)  # for 4-stroke
        
        # Calculate maximum injection time per cycle
        cycle_time_ms = 60000 / injections_per_min  # ms per cycle
        max_injection_time = cycle_time_ms * self.max_duty_cycle  # ms
        
        # Calculate effective flow rate
        effective_flow_rate = self.get_flow_rate() * (max_injection_time - self.opening_time) / max_injection_time
        
        return effective_flow_rate
    
    def get_max_fuel_mass(self, engine_rpm: float, cylinders: int = 4, 
                         fuel_density: float = 0.781) -> float:
        """
        Calculate maximum injectable fuel mass per cycle.
        
        Args:
            engine_rpm: Engine speed in RPM
            cylinders: Number of engine cylinders
            fuel_density: Fuel density in g/cc
            
        Returns:
            Maximum fuel mass in mg per injection
        """
        # Calculate injections per minute
        injections_per_min = engine_rpm * (cylinders / 2)  # for 4-stroke
        
        # Calculate maximum injection time per cycle
        cycle_time_ms = 60000 / injections_per_min  # ms per cycle
        max_injection_time = cycle_time_ms * self.max_duty_cycle  # ms
        
        # Calculate adjustable injection time
        adjustable_time = max_injection_time - self.opening_time
        
        # Calculate maximum mass
        flow_rate_mg_ms = self.get_flow_rate() * fuel_density / 60  # mg/ms
        max_mass = flow_rate_mg_ms * adjustable_time  # mg
        
        return max_mass
    
    def calculate_duty_cycle(self, fuel_mass: float, engine_rpm: float, 
                           cylinders: int = 4, fuel_density: float = 0.781) -> float:
        """
        Calculate duty cycle for a given fuel mass and RPM.
        
        Args:
            fuel_mass: Fuel mass to inject in mg
            engine_rpm: Engine speed in RPM
            cylinders: Number of engine cylinders
            fuel_density: Fuel density in g/cc
            
        Returns:
            Duty cycle (0-1)
        """
        # Calculate pulse width
        pulse_width = self.get_pulse_width(fuel_mass, fuel_density)
        
        # Calculate cycle time
        injections_per_min = engine_rpm * (cylinders / 2)  # for 4-stroke
        cycle_time_ms = 60000 / injections_per_min  # ms per cycle
        
        # Calculate duty cycle
        duty_cycle = pulse_width / cycle_time_ms
        
        return min(duty_cycle, self.max_duty_cycle)
    
    def is_adequate(self, required_flow: float, engine_rpm: float = 14000, 
                  cylinders: int = 4) -> bool:
        """
        Check if injector is adequate for required flow.
        
        Args:
            required_flow: Required flow rate in cc/min
            engine_rpm: Maximum engine speed in RPM
            cylinders: Number of engine cylinders
            
        Returns:
            True if injector is adequate, False otherwise
        """
        max_flow = self.get_max_flow_rate(engine_rpm, cylinders)
        return max_flow >= required_flow
    
    def to_dict(self) -> Dict:
        """
        Convert injector properties to dictionary.
        
        Returns:
            Dictionary with injector properties
        """
        return {
            'flow_rate': self.flow_rate,
            'opening_time': self.opening_time,
            'battery_voltage': self.battery_voltage,
            'min_pulse_width': self.min_pulse_width,
            'max_duty_cycle': self.max_duty_cycle,
            'reference_pressure': self.reference_pressure,
            'current_pressure': self.current_pressure,
            'current_voltage': self.current_voltage
        }


class FuelPump:
    """Class modeling a fuel pump for a Formula Student engine."""
    
    def __init__(self, max_pressure: float = 5.0, max_flow_rate: float = 340.0,
                nominal_voltage: float = 13.8, current_draw: float = 5.0,
                min_voltage: float = 8.0):
        """
        Initialize a fuel pump model.
        
        Args:
            max_pressure: Maximum pump pressure in bar
            max_flow_rate: Maximum flow rate in L/hr
            nominal_voltage: Nominal operating voltage in V
            current_draw: Current draw at nominal voltage in A
            min_voltage: Minimum operating voltage in V
        """
        self.max_pressure = max_pressure  # bar
        self.max_flow_rate = max_flow_rate  # L/hr
        self.nominal_voltage = nominal_voltage  # V
        self.current_draw = current_draw  # A
        self.min_voltage = min_voltage  # V
        
        # Current state
        self.current_voltage = nominal_voltage  # V
        self.current_pressure = max_pressure  # bar
        self.duty_cycle = 1.0  # 0-1
        
        # Create pressure-flow curve
        self._create_pressure_flow_curve()
    
    def _create_pressure_flow_curve(self):
        """Create pressure-flow curve for the pump."""
        # Typical pressure-flow relationship for fuel pumps
        # Flow decreases linearly with pressure
        pressures = np.array([0.0, self.max_pressure])
        flows = np.array([self.max_flow_rate, 0.0])
        
        # Create interpolation function
        self.pressure_flow_function = interp1d(
            pressures, flows, kind='linear', 
            bounds_error=False, fill_value=(self.max_flow_rate, 0.0)
        )
    
    @classmethod
    def from_config(cls, config_path: str) -> 'FuelPump':
        """
        Create a FuelPump instance from a YAML configuration file.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            FuelPump instance
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Pump configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Create pump with configuration values
        return cls(
            max_pressure=config.get('max_pressure', 5.0),
            max_flow_rate=config.get('max_flow_rate', 340.0),
            nominal_voltage=config.get('nominal_voltage', 13.8),
            current_draw=config.get('current_draw', 5.0),
            min_voltage=config.get('min_voltage', 8.0)
        )
    
    def set_operating_conditions(self, voltage: float, duty_cycle: float = 1.0):
        """
        Set the current operating conditions for the pump.
        
        Args:
            voltage: Battery voltage in V
            duty_cycle: Pump duty cycle (0-1)
        """
        self.current_voltage = voltage
        self.duty_cycle = max(0.0, min(1.0, duty_cycle))
        
        # Update current pressure based on voltage and duty cycle
        voltage_factor = min(1.0, max(0.0, (voltage - self.min_voltage) / 
                                    (self.nominal_voltage - self.min_voltage)))
        self.current_pressure = self.max_pressure * voltage_factor * self.duty_cycle
    
    def get_flow_rate(self, pressure: Optional[float] = None) -> float:
        """
        Calculate the flow rate at given pressure.
        
        Args:
            pressure: Optional system pressure in bar (uses current_pressure if None)
            
        Returns:
            Flow rate in L/hr
        """
        if pressure is None:
            pressure = self.current_pressure
        
        # Get base flow rate from pressure-flow curve
        base_flow = float(self.pressure_flow_function(pressure))
        
        # Adjust for voltage and duty cycle
        voltage_factor = min(1.0, max(0.0, (self.current_voltage - self.min_voltage) / 
                                     (self.nominal_voltage - self.min_voltage)))
        
        return base_flow * voltage_factor * self.duty_cycle
    
    def get_power_consumption(self) -> float:
        """
        Calculate electrical power consumption.
        
        Returns:
            Power consumption in W
        """
        # Current scales approximately linearly with voltage and duty cycle
        voltage_factor = self.current_voltage / self.nominal_voltage
        current = self.current_draw * voltage_factor * self.duty_cycle
        
        return self.current_voltage * current
    
    def is_adequate(self, required_flow: float, system_pressure: float) -> bool:
        """
        Check if pump is adequate for required flow at given pressure.
        
        Args:
            required_flow: Required flow rate in L/hr
            system_pressure: System pressure in bar
            
        Returns:
            True if pump is adequate, False otherwise
        """
        max_flow = self.get_flow_rate(system_pressure)
        return max_flow >= required_flow
    
    def to_dict(self) -> Dict:
        """
        Convert pump properties to dictionary.
        
        Returns:
            Dictionary with pump properties
        """
        return {
            'max_pressure': self.max_pressure,
            'max_flow_rate': self.max_flow_rate,
            'nominal_voltage': self.nominal_voltage,
            'current_draw': self.current_draw,
            'min_voltage': self.min_voltage,
            'current_voltage': self.current_voltage,
            'current_pressure': self.current_pressure,
            'duty_cycle': self.duty_cycle
        }


class FuelConsumption:
    """Class for analyzing and predicting fuel consumption in Formula Student vehicles."""
    
    def __init__(self, fuel_properties: FuelProperties, engine=None):
        """
        Initialize the fuel consumption analyzer.
        
        Args:
            fuel_properties: FuelProperties instance
            engine: Optional MotorcycleEngine instance
        """
        self.fuel_properties = fuel_properties
        self.engine = engine
        
        # Storage for consumption data
        self.consumption_data = None
        self.lap_consumption = None
        self.event_consumption = {}
    
    def calculate_fuel_mass(self, power_kw: float, thermal_efficiency: float = 0.32) -> float:
        """
        Calculate fuel mass flow rate required to produce given power.
        
        Args:
            power_kw: Power output in kW
            thermal_efficiency: Engine thermal efficiency (0-1)
            
        Returns:
            Fuel mass flow rate in g/s
        """
        # Calculate required fuel energy input
        energy_input_kw = power_kw / thermal_efficiency
        
        # Convert to fuel mass flow rate
        # energy_input_kw = fuel_mass_g_s * energy_density_MJ_kg / 1000
        fuel_mass_g_s = energy_input_kw * 1000 / self.fuel_properties.energy_density
        
        return fuel_mass_g_s
    
    def calculate_consumption_map(self, rpm_range: np.ndarray, throttle_range: np.ndarray) -> np.ndarray:
        """
        Calculate fuel consumption map for different RPM and throttle positions.
        
        Args:
            rpm_range: Array of RPM points
            throttle_range: Array of throttle positions (0-1)
            
        Returns:
            2D array of fuel consumption in g/s
        """
        if self.engine is None:
            raise ValueError("Engine not provided")
        
        # Create empty consumption map
        consumption_map = np.zeros((len(rpm_range), len(throttle_range)))
        
        # Fill consumption map
        for i, rpm in enumerate(rpm_range):
            for j, throttle in enumerate(throttle_range):
                power_kw = self.engine.get_power(rpm, throttle)
                consumption_map[i, j] = self.calculate_fuel_mass(power_kw)
        
        return consumption_map
    
    def calculate_specific_consumption(self, rpm_range: np.ndarray, throttle_range: np.ndarray) -> np.ndarray:
        """
        Calculate brake specific fuel consumption (BSFC) map.
        
        Args:
            rpm_range: Array of RPM points
            throttle_range: Array of throttle positions (0-1)
            
        Returns:
            2D array of BSFC in g/kWh
        """
        if self.engine is None:
            raise ValueError("Engine not provided")
        
        # Create empty BSFC map
        bsfc_map = np.zeros((len(rpm_range), len(throttle_range)))
        
        # Fill BSFC map
        for i, rpm in enumerate(rpm_range):
            for j, throttle in enumerate(throttle_range):
                power_kw = self.engine.get_power(rpm, throttle)
                
                if power_kw > 0.1:  # Avoid division by very small power
                    fuel_mass_g_s = self.calculate_fuel_mass(power_kw)
                    bsfc = fuel_mass_g_s * 3600 / power_kw  # g/kWh
                    bsfc_map[i, j] = bsfc
                else:
                    bsfc_map[i, j] = 0.0
        
        return bsfc_map
    
    def calculate_track_consumption(self, speed_profile: np.ndarray, rpm_profile: np.ndarray, 
                                  throttle_profile: np.ndarray, time_steps: np.ndarray) -> Dict:
        """
        Calculate fuel consumption over a track.
        
        Args:
            speed_profile: Array of vehicle speeds (km/h)
            rpm_profile: Array of engine RPMs
            throttle_profile: Array of throttle positions (0-1)
            time_steps: Array of time points (s)
            
        Returns:
            Dictionary with consumption results
        """
        if self.engine is None:
            raise ValueError("Engine not provided")
        
        # Initialize arrays
        power_profile = np.zeros_like(rpm_profile)
        consumption_profile = np.zeros_like(rpm_profile)
        cumulative_consumption = np.zeros_like(rpm_profile)
        
        # Calculate consumption at each point
        for i, (rpm, throttle) in enumerate(zip(rpm_profile, throttle_profile)):
            power_kw = self.engine.get_power(rpm, throttle)
            power_profile[i] = power_kw
            consumption_profile[i] = self.calculate_fuel_mass(power_kw)
        
        # Calculate cumulative consumption
        for i in range(1, len(time_steps)):
            dt = time_steps[i] - time_steps[i-1]
            fuel_used = consumption_profile[i-1] * dt  # g
            cumulative_consumption[i] = cumulative_consumption[i-1] + fuel_used
        
        # Calculate total consumption
        total_time = time_steps[-1] - time_steps[0]
        total_distance = np.trapz(speed_profile / 3600, time_steps)  # km
        total_consumption = cumulative_consumption[-1]  # g
        avg_consumption_per_km = total_consumption / total_distance if total_distance > 0 else 0  # g/km
        
        # Store results
        self.consumption_data = {
            'time': time_steps,
            'speed': speed_profile,
            'rpm': rpm_profile,
            'throttle': throttle_profile,
            'power': power_profile,
            'consumption_rate': consumption_profile,
            'cumulative_consumption': cumulative_consumption
        }
        
        # Return summary
        return {
            'total_time': total_time,  # s
            'total_distance': total_distance,  # km
            'total_consumption': total_consumption,  # g
            'avg_consumption_per_km': avg_consumption_per_km,  # g/km
            'avg_consumption_per_lap': total_consumption,  # g/lap (assuming one lap)
            'fuel_volume': total_consumption / (self.fuel_properties.density * 1000)  # L
        }
    
    def calculate_endurance_requirement(self, lap_consumption: float, num_laps: int, 
                                      safety_factor: float = 1.2) -> float:
        """
        Calculate required fuel for an endurance event.
        
        Args:
            lap_consumption: Fuel consumption per lap in g
            num_laps: Number of laps
            safety_factor: Safety factor to account for variations
            
        Returns:
            Required fuel volume in L
        """
        # Calculate total mass
        total_mass = lap_consumption * num_laps * safety_factor  # g
        
        # Convert to volume
        total_volume = total_mass / (self.fuel_properties.density * 1000)  # L
        
        # Store in event consumption
        self.event_consumption['endurance'] = {
            'fuel_mass': total_mass,  # g
            'fuel_volume': total_volume,  # L
            'num_laps': num_laps,
            'safety_factor': safety_factor
        }
        
        return total_volume
    
    def calculate_acceleration_requirement(self, distance: float = 75.0, duration: float = 4.0,
                                         num_runs: int = 4, safety_factor: float = 1.5) -> float:
        """
        Calculate required fuel for acceleration event.
        
        Args:
            distance: Acceleration distance in m
            duration: Approximate duration in s
            num_runs: Number of acceleration runs
            safety_factor: Safety factor
            
        Returns:
            Required fuel volume in L
        """
        if self.engine is None:
            raise ValueError("Engine not provided")
        
        # Estimate average power (assuming full throttle)
        avg_power = self.engine.get_power(self.engine.max_power_rpm * 0.8, 1.0)  # kW
        
        # Calculate fuel consumption for one run
        consumption_rate = self.calculate_fuel_mass(avg_power)  # g/s
        fuel_per_run = consumption_rate * duration  # g
        
        # Calculate total
        total_mass = fuel_per_run * num_runs * safety_factor  # g
        total_volume = total_mass / (self.fuel_properties.density * 1000)  # L
        
        # Store in event consumption
        self.event_consumption['acceleration'] = {
            'fuel_mass': total_mass,  # g
            'fuel_volume': total_volume,  # L
            'num_runs': num_runs,
            'distance': distance,
            'duration': duration,
            'safety_factor': safety_factor
        }
        
        return total_volume
    
    def calculate_autocross_requirement(self, lap_consumption: float, num_laps: int = 4,
                                      safety_factor: float = 1.2) -> float:
        """
        Calculate required fuel for autocross event.
        
        Args:
            lap_consumption: Fuel consumption per lap in g
            num_laps: Number of laps
            safety_factor: Safety factor
            
        Returns:
            Required fuel volume in L
        """
        # Calculate total mass
        total_mass = lap_consumption * num_laps * safety_factor  # g
        
        # Convert to volume
        total_volume = total_mass / (self.fuel_properties.density * 1000)  # L
        
        # Store in event consumption
        self.event_consumption['autocross'] = {
            'fuel_mass': total_mass,  # g
            'fuel_volume': total_volume,  # L
            'num_laps': num_laps,
            'safety_factor': safety_factor
        }
        
        return total_volume
    
    def calculate_skidpad_requirement(self, duration: float = 60.0, avg_throttle: float = 0.7,
                                    safety_factor: float = 1.2) -> float:
        """
        Calculate required fuel for skidpad event.
        
        Args:
            duration: Estimated event duration in s
            avg_throttle: Average throttle position (0-1)
            safety_factor: Safety factor
            
        Returns:
            Required fuel volume in L
        """
        if self.engine is None:
            raise ValueError("Engine not provided")
        
        # Estimate average RPM (typically mid-range)
        avg_rpm = (self.engine.idle_rpm + self.engine.redline) / 2
        
        # Calculate average power
        avg_power = self.engine.get_power(avg_rpm, avg_throttle)  # kW
        
        # Calculate fuel consumption
        consumption_rate = self.calculate_fuel_mass(avg_power)  # g/s
        total_mass = consumption_rate * duration * safety_factor  # g
        
        # Convert to volume
        total_volume = total_mass / (self.fuel_properties.density * 1000)  # L
        
        # Store in event consumption
        self.event_consumption['skidpad'] = {
            'fuel_mass': total_mass,  # g
            'fuel_volume': total_volume,  # L
            'duration': duration,
            'avg_throttle': avg_throttle,
            'safety_factor': safety_factor
        }
        
        return total_volume
    
    def calculate_total_event_requirement(self) -> float:
        """
        Calculate total fuel requirement for all events.
        
        Returns:
            Total required fuel volume in L
        """
        total_volume = sum(event['fuel_volume'] for event in self.event_consumption.values())
        return total_volume
    
    def plot_consumption_profile(self, save_path: Optional[str] = None):
        """
        Plot consumption profile over time.
        
        Args:
            save_path: Optional path to save the plot
        """
        if self.consumption_data is None:
            raise ValueError("No consumption data available. Run calculate_track_consumption first.")
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        
        # Plot speed profile
        ax1.plot(self.consumption_data['time'], self.consumption_data['speed'], 'b-')
        ax1.set_ylabel('Speed (km/h)')
        ax1.grid(True, alpha=0.3)
        ax1.set_title('Speed Profile')
        
        # Plot power and fuel consumption rate
        ax2.plot(self.consumption_data['time'], self.consumption_data['power'], 'r-', label='Power (kW)')
        ax2_twin = ax2.twinx()
        ax2_twin.plot(self.consumption_data['time'], self.consumption_data['consumption_rate'], 'g-', label='Fuel Rate (g/s)')
        ax2.set_ylabel('Power (kW)', color='r')
        ax2_twin.set_ylabel('Fuel Rate (g/s)', color='g')
        ax2.tick_params(axis='y', colors='r')
        ax2_twin.tick_params(axis='y', colors='g')
        
        # Add legend
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        ax2.grid(True, alpha=0.3)
        ax2.set_title('Power and Fuel Consumption Rate')
        
        # Plot cumulative consumption
        ax3.plot(self.consumption_data['time'], self.consumption_data['cumulative_consumption'], 'k-')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Cumulative Fuel (g)')
        ax3.grid(True, alpha=0.3)
        ax3.set_title('Cumulative Fuel Consumption')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_bsfc_map(self, rpm_range: np.ndarray, throttle_range: np.ndarray, 
                    save_path: Optional[str] = None):
        """
        Plot brake specific fuel consumption map.
        
        Args:
            rpm_range: Array of RPM points
            throttle_range: Array of throttle positions (0-1)
            save_path: Optional path to save the plot
        """
        # Calculate BSFC map
        bsfc_map = self.calculate_specific_consumption(rpm_range, throttle_range)
        
        # Create meshgrid for contour plot
        rpm_grid, throttle_grid = np.meshgrid(rpm_range, throttle_range)
        
        plt.figure(figsize=(10, 8))
        
        # Create contour plot
        contour = plt.contourf(rpm_grid, throttle_grid, bsfc_map.T, 20, cmap='viridis')
        plt.colorbar(contour, label='BSFC (g/kWh)')
        
        # Add labels and title
        plt.xlabel('Engine Speed (RPM)')
        plt.ylabel('Throttle Position')
        plt.title(f'Brake Specific Fuel Consumption Map - {self.fuel_properties.fuel_type.name}')
        
        # Add grid and improve layout
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_event_requirements(self, save_path: Optional[str] = None):
        """
        Plot fuel requirements for different events.
        
        Args:
            save_path: Optional path to save the plot
        """
        if not self.event_consumption:
            raise ValueError("No event consumption data. Calculate event requirements first.")
        
        events = list(self.event_consumption.keys())
        volumes = [event['fuel_volume'] for event in self.event_consumption.values()]
        
        plt.figure(figsize=(10, 6))
        
        # Create bar chart
        bars = plt.bar(events, volumes, color='skyblue')
        
        # Add labels and values on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{height:.2f} L', ha='center', va='bottom')
        
        # Add total
        plt.axhline(y=sum(volumes), color='r', linestyle='--', label=f'Total: {sum(volumes):.2f} L')
        
        # Add labels and title
        plt.xlabel('Event')
        plt.ylabel('Fuel Volume (L)')
        plt.title(f'Fuel Requirements by Event - {self.fuel_properties.fuel_type.name}')
        plt.legend()
        
        # Improve layout
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def to_dict(self) -> Dict:
        """
        Convert fuel consumption data to dictionary.
        
        Returns:
            Dictionary with fuel consumption data
        """
        result = {
            'fuel_properties': self.fuel_properties.to_dict(),
            'event_consumption': self.event_consumption
        }
        
        # Add track consumption data if available
        if self.consumption_data:
            result['track_consumption'] = {
                'total_consumption': self.consumption_data['cumulative_consumption'][-1],
                'avg_consumption_rate': np.mean(self.consumption_data['consumption_rate']),
                'peak_consumption_rate': np.max(self.consumption_data['consumption_rate']),
                'consumption_duration': self.consumption_data['time'][-1] - self.consumption_data['time'][0]
            }
        
        return result


class FuelSystem:
    """Class representing the complete fuel system for a Formula Student vehicle."""
    
    def __init__(self, fuel_properties: FuelProperties, fuel_pump: FuelPump, injectors: List[FuelInjector],
               tank_capacity: float = 7.0, pressure_regulator: float = 3.0):
        """
        Initialize the fuel system.
        
        Args:
            fuel_properties: FuelProperties instance
            fuel_pump: FuelPump instance
            injectors: List of FuelInjector instances (one per cylinder)
            tank_capacity: Fuel tank capacity in L
            pressure_regulator: Pressure regulator setting in bar
        """
        self.fuel_properties = fuel_properties
        self.fuel_pump = fuel_pump
        self.injectors = injectors
        self.tank_capacity = tank_capacity
        self.pressure_regulator = pressure_regulator
        
        # Current state
        self.current_fuel_level = tank_capacity  # L
        self.current_system_pressure = pressure_regulator  # bar
        self.current_battery_voltage = 13.8  # V
        
        # Set initial operating conditions
        self.fuel_pump.set_operating_conditions(self.current_battery_voltage)
        for injector in self.injectors:
            injector.set_operating_conditions(self.current_system_pressure, self.current_battery_voltage)
    
    @classmethod
    def from_config(cls, config_path: str) -> 'FuelSystem':
        """
        Create a FuelSystem instance from a YAML configuration file.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            FuelSystem instance
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Fuel system configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Create fuel properties
        fuel_type_str = config.get('fuel_type', 'E85').upper()
        fuel_props = FuelProperties(FuelType[fuel_type_str])
        
        # Create fuel pump
        pump_config = config.get('fuel_pump', {})
        fuel_pump = FuelPump(
            max_pressure=pump_config.get('max_pressure', 5.0),
            max_flow_rate=pump_config.get('max_flow_rate', 340.0),
            nominal_voltage=pump_config.get('nominal_voltage', 13.8),
            current_draw=pump_config.get('current_draw', 5.0),
            min_voltage=pump_config.get('min_voltage', 8.0)
        )
        
        # Create injectors
        injector_config = config.get('injector', {})
        num_injectors = config.get('num_cylinders', 4)
        injectors = [
            FuelInjector(
                flow_rate=injector_config.get('flow_rate', 550.0),
                opening_time=injector_config.get('opening_time', 1.0),
                battery_voltage=injector_config.get('battery_voltage', 13.8),
                min_pulse_width=injector_config.get('min_pulse_width', 0.9),
                max_duty_cycle=injector_config.get('max_duty_cycle', 0.85)
            )
            for _ in range(num_injectors)
        ]
        
        # Create fuel system
        return cls(
            fuel_properties=fuel_props,
            fuel_pump=fuel_pump,
            injectors=injectors,
            tank_capacity=config.get('tank_capacity', 7.0),
            pressure_regulator=config.get('pressure_regulator', 3.0)
        )
    
    def set_system_state(self, fuel_level: float, battery_voltage: float):
        """
        Set the current state of the fuel system.
        
        Args:
            fuel_level: Current fuel level in L
            battery_voltage: Current battery voltage in V
        """
        self.current_fuel_level = min(fuel_level, self.tank_capacity)
        self.current_battery_voltage = battery_voltage
        
        # Update pump
        self.fuel_pump.set_operating_conditions(battery_voltage)
        
        # Calculate actual pressure based on pump performance
        max_pressure = self.fuel_pump.current_pressure
        self.current_system_pressure = min(max_pressure, self.pressure_regulator)
        
        # Update injectors
        for injector in self.injectors:
            injector.set_operating_conditions(self.current_system_pressure, battery_voltage)
    
    def calculate_max_flow_rate(self, engine_rpm: float) -> float:
        """
        Calculate maximum possible fuel flow rate at current conditions.
        
        Args:
            engine_rpm: Engine speed in RPM
            
        Returns:
            Maximum fuel flow rate in cc/min
        """
        # Check pump flow rate
        pump_flow_lph = self.fuel_pump.get_flow_rate(self.current_system_pressure)
        pump_flow_ccpm = pump_flow_lph * 1000 / 60  # L/hr to cc/min
        
        # Check injector flow rate (for one cylinder)
        injector_flow = self.injectors[0].get_max_flow_rate(engine_rpm, len(self.injectors))
        
        # Total injector flow rate (all cylinders)
        total_injector_flow = injector_flow * len(self.injectors)
        
        # Return the limiting factor
        return min(pump_flow_ccpm, total_injector_flow)
    
    def calculate_max_power(self, engine_rpm: float, thermal_efficiency: float = 0.32) -> float:
        """
        Calculate maximum engine power possible with current fuel system.
        
        Args:
            engine_rpm: Engine speed in RPM
            thermal_efficiency: Engine thermal efficiency (0-1)
            
        Returns:
            Maximum power in kW
        """
        # Get maximum flow rate
        max_flow_ccpm = self.calculate_max_flow_rate(engine_rpm)
        
        # Convert to g/s
        max_flow_g_s = max_flow_ccpm * self.fuel_properties.density / 60
        
        # Calculate maximum power
        max_power_kw = max_flow_g_s * self.fuel_properties.energy_density * thermal_efficiency / 1000
        
        return max_power_kw
    
    def update_fuel_level(self, consumption_rate: float, duration: float) -> float:
        """
        Update fuel level based on consumption.
        
        Args:
            consumption_rate: Fuel consumption rate in g/s
            duration: Duration in s
            
        Returns:
            New fuel level in L
        """
        # Calculate fuel mass consumed
        fuel_mass = consumption_rate * duration  # g
        
        # Convert to volume
        fuel_volume = fuel_mass / (self.fuel_properties.density * 1000)  # L
        
        # Update level
        self.current_fuel_level = max(0, self.current_fuel_level - fuel_volume)
        
        return self.current_fuel_level
    
    def is_fuel_sufficient(self, required_mass: float) -> bool:
        """
        Check if current fuel level is sufficient for required mass.
        
        Args:
            required_mass: Required fuel mass in g
            
        Returns:
            True if sufficient, False otherwise
        """
        required_volume = required_mass / (self.fuel_properties.density * 1000)  # L
        return self.current_fuel_level >= required_volume
    
    def validate_system(self, max_power_kw: float, max_rpm: float, thermal_efficiency: float = 0.32) -> Dict:
        """
        Validate if the fuel system is adequate for the engine requirements.
        
        Args:
            max_power_kw: Maximum engine power in kW
            max_rpm: Maximum engine RPM
            thermal_efficiency: Engine thermal efficiency (0-1)
            
        Returns:
            Dictionary with validation results
        """
        # Calculate required fuel flow for max power
        fuel_consumption = FuelConsumption(self.fuel_properties)
        required_flow_g_s = fuel_consumption.calculate_fuel_mass(max_power_kw, thermal_efficiency)
        
        # Convert to cc/min
        required_flow_ccpm = required_flow_g_s * 60 / self.fuel_properties.density * 1000
        
        # Get maximum flow capability
        max_flow_ccpm = self.calculate_max_flow_rate(max_rpm)
        
        # Validate pump
        pump_flow_lph = self.fuel_pump.get_flow_rate(self.current_system_pressure)
        pump_flow_ccpm = pump_flow_lph * 1000 / 60
        pump_adequate = pump_flow_ccpm >= required_flow_ccpm
        
        # Validate injectors
        injector_flow = self.injectors[0].get_max_flow_rate(max_rpm, len(self.injectors))
        total_injector_flow = injector_flow * len(self.injectors)
        injectors_adequate = total_injector_flow >= required_flow_ccpm
        
        # Calculate margin
        flow_margin = max_flow_ccpm / required_flow_ccpm if required_flow_ccpm > 0 else float('inf')
        
        return {
            'system_adequate': pump_adequate and injectors_adequate,
            'pump_adequate': pump_adequate,
            'injectors_adequate': injectors_adequate,
            'required_flow_ccpm': required_flow_ccpm,
            'max_flow_ccpm': max_flow_ccpm,
            'flow_margin': flow_margin,
            'limiting_component': 'pump' if pump_flow_ccpm < total_injector_flow else 'injectors'
        }
    
    def to_dict(self) -> Dict:
        """
        Convert fuel system to dictionary.
        
        Returns:
            Dictionary with fuel system data
        """
        return {
            'fuel_properties': self.fuel_properties.to_dict(),
            'fuel_pump': self.fuel_pump.to_dict(),
            'injectors': [injector.to_dict() for injector in self.injectors],
            'tank_capacity': self.tank_capacity,
            'pressure_regulator': self.pressure_regulator,
            'current_fuel_level': self.current_fuel_level,
            'current_system_pressure': self.current_system_pressure,
            'current_battery_voltage': self.current_battery_voltage
        }


# Example usage
if __name__ == "__main__":
    import os
    import sys
    sys.path.append('..')  # Add parent directory to path
    
    try:
        from engine.motorcycle_engine import MotorcycleEngine
        
        # Create engine
        config_path = os.path.join("configs", "engine", "cbr600f4i.yaml")
        engine = MotorcycleEngine(config_path=config_path)
        
        # Create fuel properties for E85
        fuel_props = FuelProperties(FuelType.E85)
        
        # Create fuel injector
        injector = FuelInjector(flow_rate=550.0, opening_time=1.0)
        
        # Create fuel pump
        pump = FuelPump(max_pressure=5.0, max_flow_rate=340.0)
        
        # Create fuel system
        fuel_system = FuelSystem(
            fuel_properties=fuel_props,
            fuel_pump=pump,
            injectors=[injector] * 4,  # 4 injectors for 4 cylinders
            tank_capacity=7.0,
            pressure_regulator=3.0
        )
        
        # Validate fuel system for the engine
        validation = fuel_system.validate_system(
            max_power_kw=engine.max_power * 0.75,  # Convert hp to kW
            max_rpm=engine.redline,
            thermal_efficiency=0.32
        )
        
        print("Fuel System Validation:")
        for key, value in validation.items():
            print(f"  {key}: {value}")
        
        # Create fuel consumption analyzer
        consumption = FuelConsumption(fuel_props, engine)
        
        # Calculate endurance fuel requirement
        lap_consumption = 200.0  # g/lap (example value)
        endurance_fuel = consumption.calculate_endurance_requirement(
            lap_consumption=lap_consumption,
            num_laps=22,  # Typical Formula Student endurance
            safety_factor=1.2
        )
        
        print(f"\nEndurance Fuel Requirement: {endurance_fuel:.2f} L")
        
        # Calculate requirements for other events
        acceleration_fuel = consumption.calculate_acceleration_requirement()
        autocross_fuel = consumption.calculate_autocross_requirement(lap_consumption)
        skidpad_fuel = consumption.calculate_skidpad_requirement()
        
        # Calculate total requirement
        total_fuel = consumption.calculate_total_event_requirement()
        
        print(f"Total Fuel Requirement for all events: {total_fuel:.2f} L")
        print(f"Is tank capacity sufficient? {'Yes' if total_fuel <= fuel_system.tank_capacity else 'No'}")
        
        # Plot event requirements
        consumption.plot_event_requirements()
        
    except ImportError:
        print("MotorcycleEngine class not available, skipping engine-dependent examples")
        
        # Basic demonstration without engine
        fuel_props = FuelProperties(FuelType.E85)
        print("\nE85 Fuel Properties:")
        for key, value in fuel_props.to_dict().items():
            print(f"  {key}: {value}")
        
        # Compare with gasoline
        gasoline_props = FuelProperties(FuelType.GASOLINE)
        comparison = fuel_props.compare_with(gasoline_props)
        
        print("\nE85 vs Gasoline comparison:")
        for key, value in comparison.items():
            print(f"  {key}: {value:.2f}")
        
        # Create injector and pump
        injector = FuelInjector(flow_rate=550.0)
        pump = FuelPump(max_pressure=5.0, max_flow_rate=340.0)
        
        print(f"\nInjector flow rate at 3 bar: {injector.get_flow_rate(3.0):.1f} cc/min")
        print(f"Pump flow rate at 3 bar: {pump.get_flow_rate(3.0):.1f} L/hr")