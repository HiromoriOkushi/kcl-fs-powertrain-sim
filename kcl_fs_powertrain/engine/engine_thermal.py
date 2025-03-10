"""
Engine thermal management module for Formula Student powertrain simulation.

This module provides classes and functions to model the thermal behavior of a motorcycle 
engine in a Formula Student vehicle, including cooling system performance, heat generation,
thermal dynamics, and temperature-dependent performance effects.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Callable
from enum import Enum, auto
import yaml
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d, griddata

# Define module exports
__all__ = [
    'CoolingSystem', 'EngineHeatModel', 'ThermalSimulation', 
    'CoolingPerformance', 'ThermalConfig'
]


class ThermalConfig:
    """Configuration parameters for engine thermal model."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize thermal configuration with default values or from file.
        
        Args:
            config_path: Optional path to YAML configuration file
        """
        # Default thermal parameters
        # Component specific heat capacities (J/kg·K)
        self.specific_heat_engine_block = 450.0  # Cast aluminum
        self.specific_heat_engine_oil = 2000.0   # Engine oil
        self.specific_heat_coolant = 3700.0      # Water-glycol mix
        
        # Component masses (kg)
        self.mass_engine_block = 35.0            # Engine block mass
        self.mass_engine_oil = 2.5               # Engine oil mass
        self.mass_coolant = 3.0                  # Coolant mass
        
        # Heat transfer coefficients (W/m²·K)
        self.htc_oil_to_block = 1500.0
        self.htc_coolant_to_block = 3000.0
        self.htc_block_to_air = 50.0
        
        # Cooling system parameters
        self.radiator_effectiveness = 0.7        # Radiator effectiveness (0-1)
        self.radiator_area = 0.15                # Radiator area (m²)
        self.min_airflow_speed = 2.0             # Minimum airflow speed (m/s)
        self.fan_max_airflow = 0.3               # Fan max airflow (m³/s)
        
        # Thermal management setpoints
        self.thermostat_open_temp = 82.0         # Thermostat opening temp (°C)
        self.target_coolant_temp = 90.0          # Target coolant temp (°C)
        self.max_engine_temp = 120.0             # Maximum engine temp (°C)
        self.optimal_oil_temp = 100.0            # Optimal oil temp (°C)
        
        # Load from file if provided
        if config_path:
            self.load_from_file(config_path)
    
    def load_from_file(self, config_path: str):
        """
        Load thermal configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Thermal configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Update attributes from config
        if 'thermal' in config:
            thermal_config = config['thermal']
            
            # Update specific heats
            if 'specific_heat' in thermal_config:
                specific_heat = thermal_config['specific_heat']
                self.specific_heat_engine_block = specific_heat.get('engine_block', self.specific_heat_engine_block)
                self.specific_heat_engine_oil = specific_heat.get('engine_oil', self.specific_heat_engine_oil)
                self.specific_heat_coolant = specific_heat.get('coolant', self.specific_heat_coolant)
            
            # Update masses
            if 'mass' in thermal_config:
                mass = thermal_config['mass']
                self.mass_engine_block = mass.get('engine_block', self.mass_engine_block)
                self.mass_engine_oil = mass.get('engine_oil', self.mass_engine_oil)
                self.mass_coolant = mass.get('coolant', self.mass_coolant)
            
            # Update heat transfer coefficients
            if 'heat_transfer' in thermal_config:
                htc = thermal_config['heat_transfer']
                self.htc_oil_to_block = htc.get('oil_to_block', self.htc_oil_to_block)
                self.htc_coolant_to_block = htc.get('coolant_to_block', self.htc_coolant_to_block)
                self.htc_block_to_air = htc.get('block_to_air', self.htc_block_to_air)
            
            # Update cooling system parameters
            if 'cooling_system' in thermal_config:
                cooling = thermal_config['cooling_system']
                self.radiator_effectiveness = cooling.get('radiator_effectiveness', self.radiator_effectiveness)
                self.radiator_area = cooling.get('radiator_area', self.radiator_area)
                self.min_airflow_speed = cooling.get('min_airflow_speed', self.min_airflow_speed)
                self.fan_max_airflow = cooling.get('fan_max_airflow', self.fan_max_airflow)
            
            # Update thermal management setpoints
            if 'setpoints' in thermal_config:
                setpoints = thermal_config['setpoints']
                self.thermostat_open_temp = setpoints.get('thermostat_open', self.thermostat_open_temp)
                self.target_coolant_temp = setpoints.get('target_coolant', self.target_coolant_temp)
                self.max_engine_temp = setpoints.get('max_engine', self.max_engine_temp)
                self.optimal_oil_temp = setpoints.get('optimal_oil', self.optimal_oil_temp)
    
    def save_to_file(self, config_path: str):
        """
        Save thermal configuration to YAML file.
        
        Args:
            config_path: Path to save configuration
        """
        # Create configuration dictionary
        config = {
            'thermal': {
                'specific_heat': {
                    'engine_block': self.specific_heat_engine_block,
                    'engine_oil': self.specific_heat_engine_oil,
                    'coolant': self.specific_heat_coolant
                },
                'mass': {
                    'engine_block': self.mass_engine_block,
                    'engine_oil': self.mass_engine_oil,
                    'coolant': self.mass_coolant
                },
                'heat_transfer': {
                    'oil_to_block': self.htc_oil_to_block,
                    'coolant_to_block': self.htc_coolant_to_block,
                    'block_to_air': self.htc_block_to_air
                },
                'cooling_system': {
                    'radiator_effectiveness': self.radiator_effectiveness,
                    'radiator_area': self.radiator_area,
                    'min_airflow_speed': self.min_airflow_speed,
                    'fan_max_airflow': self.fan_max_airflow
                },
                'setpoints': {
                    'thermostat_open': self.thermostat_open_temp,
                    'target_coolant': self.target_coolant_temp,
                    'max_engine': self.max_engine_temp,
                    'optimal_oil': self.optimal_oil_temp
                }
            }
        }
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        # Save to file
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    
    def get_thermal_capacities(self) -> Dict[str, float]:
        """
        Calculate thermal capacities of engine components.
        
        Returns:
            Dictionary with thermal capacities in J/K
        """
        return {
            'engine_block': self.mass_engine_block * self.specific_heat_engine_block,
            'engine_oil': self.mass_engine_oil * self.specific_heat_engine_oil,
            'coolant': self.mass_coolant * self.specific_heat_coolant
        }
    
    def to_dict(self) -> Dict:
        """
        Convert configuration to dictionary.
        
        Returns:
            Dictionary with configuration parameters
        """
        return {
            'specific_heat_engine_block': self.specific_heat_engine_block,
            'specific_heat_engine_oil': self.specific_heat_engine_oil,
            'specific_heat_coolant': self.specific_heat_coolant,
            'mass_engine_block': self.mass_engine_block,
            'mass_engine_oil': self.mass_engine_oil,
            'mass_coolant': self.mass_coolant,
            'htc_oil_to_block': self.htc_oil_to_block,
            'htc_coolant_to_block': self.htc_coolant_to_block,
            'htc_block_to_air': self.htc_block_to_air,
            'radiator_effectiveness': self.radiator_effectiveness,
            'radiator_area': self.radiator_area,
            'min_airflow_speed': self.min_airflow_speed,
            'fan_max_airflow': self.fan_max_airflow,
            'thermostat_open_temp': self.thermostat_open_temp,
            'target_coolant_temp': self.target_coolant_temp,
            'max_engine_temp': self.max_engine_temp,
            'optimal_oil_temp': self.optimal_oil_temp
        }


class CoolingSystem:
    """Model of a motorcycle engine cooling system for Formula Student."""
    
    def __init__(self, config: ThermalConfig = None):
        """
        Initialize cooling system model.
        
        Args:
            config: ThermalConfig instance with cooling system parameters
        """
        self.config = config or ThermalConfig()
        
        # Current state
        self.fan_state = 0.0  # Fan duty cycle (0-1)
        self.pump_state = 0.0  # Pump duty cycle (0-1)
        self.thermostat_position = 0.0  # Thermostat position (0-1)
        self.radiator_airflow = 0.0  # Current airflow through radiator (m³/s)
        self.radiator_effectiveness = self.config.radiator_effectiveness
        
        # Initialize thermal model
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize thermal model parameters."""
        # Create thermostat opening curve (position vs temperature)
        # Typically opens gradually over a 10°C range
        temp_range = np.linspace(
            self.config.thermostat_open_temp,
            self.config.thermostat_open_temp + 10,
            100
        )
        position = np.clip(
            (temp_range - self.config.thermostat_open_temp) / 10.0,
            0.0, 1.0
        )
        self.thermostat_curve = interp1d(
            temp_range, position,
            bounds_error=False, fill_value=(0.0, 1.0)
        )
        
        # Create fan control curve (duty cycle vs temperature)
        # Typically ramps up over a 5°C range above target
        temp_range = np.linspace(
            self.config.target_coolant_temp,
            self.config.target_coolant_temp + 5,
            100
        )
        duty_cycle = np.clip(
            (temp_range - self.config.target_coolant_temp) / 5.0,
            0.0, 1.0
        )
        self.fan_curve = interp1d(
            temp_range, duty_cycle,
            bounds_error=False, fill_value=(0.0, 1.0)
        )
        
        # Create pump control curve (duty cycle vs temperature/engine load)
        # Base pump speed is dependent on engine RPM (mechanically driven)
        # Some systems use an electric pump with its own control curve
        self.pump_curve = lambda coolant_temp, engine_load: min(0.5 + 0.5 * engine_load, 1.0)
    
    def update_thermostat(self, coolant_temp: float):
        """
        Update thermostat position based on coolant temperature.
        
        Args:
            coolant_temp: Coolant temperature in °C
        """
        self.thermostat_position = float(self.thermostat_curve(coolant_temp))
    
    def update_fan(self, coolant_temp: float):
        """
        Update cooling fan state based on coolant temperature.
        
        Args:
            coolant_temp: Coolant temperature in °C
        """
        self.fan_state = float(self.fan_curve(coolant_temp))
    
    def update_pump(self, coolant_temp: float, engine_load: float):
        """
        Update coolant pump state based on temperature and engine load.
        
        Args:
            coolant_temp: Coolant temperature in °C
            engine_load: Engine load factor (0-1)
        """
        self.pump_state = self.pump_curve(coolant_temp, engine_load)
    
    def calculate_radiator_airflow(self, vehicle_speed: float):
        """
        Calculate airflow through the radiator based on vehicle speed and fan state.
        
        Args:
            vehicle_speed: Vehicle speed in m/s
            
        Returns:
            Airflow through radiator in m³/s
        """
        # Base airflow from vehicle speed
        # Simplified model assuming radiator cross-sectional area
        speed_airflow = max(vehicle_speed, self.config.min_airflow_speed) * self.config.radiator_area
        
        # Fan contribution
        fan_airflow = self.fan_state * self.config.fan_max_airflow
        
        # Total airflow
        self.radiator_airflow = speed_airflow + fan_airflow
        
        return self.radiator_airflow
    
    def calculate_heat_rejection(self, coolant_temp: float, ambient_temp: float, 
                               vehicle_speed: float) -> float:
        """
        Calculate heat rejection rate from the cooling system.
        
        Args:
            coolant_temp: Coolant temperature in °C
            ambient_temp: Ambient temperature in °C
            vehicle_speed: Vehicle speed in m/s
            
        Returns:
            Heat rejection rate in watts (W)
        """
        # Update system state
        self.update_thermostat(coolant_temp)
        self.update_fan(coolant_temp)
        self.calculate_radiator_airflow(vehicle_speed)
        
        # Calculate heat transfer through radiator
        # Q = ṁ·cp·ε·(Thot - Tcold)
        # where ṁ is mass flow rate, cp is specific heat of air
        
        # Air properties
        air_density = 1.2  # kg/m³
        air_specific_heat = 1005.0  # J/kg·K
        
        # Air mass flow rate
        air_mass_flow = air_density * self.radiator_airflow
        
        # Temperature difference
        temp_diff = max(0, coolant_temp - ambient_temp)
        
        # Effectiveness is modulated by thermostat position
        effective_radiator = self.radiator_effectiveness * self.thermostat_position
        
        # Heat rejection (W)
        heat_rejection = air_mass_flow * air_specific_heat * effective_radiator * temp_diff
        
        return heat_rejection
    
    def calculate_coolant_flow(self, engine_rpm: float, engine_load: float) -> float:
        """
        Calculate coolant flow rate through the engine.
        
        Args:
            engine_rpm: Engine speed in RPM
            engine_load: Engine load factor (0-1)
            
        Returns:
            Coolant flow rate in L/min
        """
        # Base flow proportional to engine RPM (mechanical pump)
        # Typical water pump flow is ~1.5 L/min per 1000 RPM at idle
        base_flow = engine_rpm / 1000 * 1.5
        
        # Update pump state
        self.update_pump(0, engine_load)  # Coolant temp doesn't matter for this calculation
        
        # For electric pumps, flow would be more dependent on pump_state
        # For mechanical pumps, flow is primarily dependent on engine RPM
        coolant_flow = base_flow * self.pump_state
        
        return coolant_flow
    
    def get_system_state(self) -> Dict:
        """
        Get current state of the cooling system.
        
        Returns:
            Dictionary with current state
        """
        return {
            'fan_state': self.fan_state,
            'pump_state': self.pump_state,
            'thermostat_position': self.thermostat_position,
            'radiator_airflow': self.radiator_airflow,
            'radiator_effectiveness': self.radiator_effectiveness
        }


class EngineHeatModel:
    """Heat generation and thermal model for a motorcycle engine."""
    
    def __init__(self, config: ThermalConfig = None, engine=None):
        """
        Initialize engine heat model.
        
        Args:
            config: ThermalConfig instance
            engine: Optional MotorcycleEngine instance
        """
        self.config = config or ThermalConfig()
        self.engine = engine
        
        # Initialize state
        self.engine_temp = 25.0  # °C
        self.oil_temp = 25.0     # °C
        self.coolant_temp = 25.0 # °C
        
        # Heat generation parameters
        self.combustion_efficiency = 0.30  # Typical efficiency for motorcycle engines
        self.friction_coefficient = 0.15   # Portion of fuel energy converted to friction heat
        self.mechanical_efficiency = 0.85  # Mechanical efficiency of the engine
        
        # Heat distribution parameters (fraction of total waste heat)
        self.heat_to_coolant = 0.60        # Fraction of waste heat to coolant
        self.heat_to_oil = 0.25            # Fraction of waste heat to oil
        self.heat_to_exhaust = 0.10        # Fraction of waste heat to exhaust
        self.heat_to_ambient = 0.05        # Fraction of waste heat directly to ambient
    
    def calculate_total_heat(self, fuel_power: float, engine_power: float) -> float:
        """
        Calculate total heat generated by the engine.
        
        Args:
            fuel_power: Fuel chemical power in W
            engine_power: Engine mechanical power output in W
            
        Returns:
            Total waste heat in W
        """
        # Calculate efficiency
        if fuel_power > 0:
            current_efficiency = engine_power / fuel_power
        else:
            current_efficiency = self.combustion_efficiency
        
        # Total waste heat is fuel energy not converted to useful work
        waste_heat = fuel_power * (1 - current_efficiency)
        
        return waste_heat
    
    def calculate_heat_sources(self, fuel_power: float, engine_power: float,
                             engine_rpm: float, engine_load: float) -> Dict[str, float]:
        """
        Calculate heat generation for different engine components.
        
        Args:
            fuel_power: Fuel chemical power in W
            engine_power: Engine mechanical power output in W
            engine_rpm: Engine speed in RPM
            engine_load: Engine load factor (0-1)
            
        Returns:
            Dictionary with heat generation rates for each component
        """
        # Calculate total waste heat
        total_heat = self.calculate_total_heat(fuel_power, engine_power)
        
        # Calculate fraction going to each component
        # These fractions can vary with RPM and load
        coolant_factor = self.heat_to_coolant * (1.0 + 0.1 * engine_load)
        oil_factor = self.heat_to_oil * (1.0 + 0.2 * engine_load)
        exhaust_factor = self.heat_to_exhaust * (1.0 + 0.3 * engine_load)
        
        # Normalize factors
        total_factor = coolant_factor + oil_factor + exhaust_factor + self.heat_to_ambient
        coolant_fraction = coolant_factor / total_factor
        oil_fraction = oil_factor / total_factor
        exhaust_fraction = exhaust_factor / total_factor
        ambient_fraction = self.heat_to_ambient / total_factor
        
        # Calculate heat to each component
        heat_to_coolant = total_heat * coolant_fraction
        heat_to_oil = total_heat * oil_fraction
        heat_to_exhaust = total_heat * exhaust_fraction
        heat_to_ambient = total_heat * ambient_fraction
        
        return {
            'total': total_heat,
            'coolant': heat_to_coolant,
            'oil': heat_to_oil,
            'exhaust': heat_to_exhaust,
            'ambient': heat_to_ambient
        }
    
    def calculate_fuel_power(self, fuel_mass_flow: float, fuel_energy_density: float = 42.5) -> float:
        """
        Calculate chemical power from fuel.
        
        Args:
            fuel_mass_flow: Fuel mass flow rate in g/s
            fuel_energy_density: Fuel energy density in MJ/kg
            
        Returns:
            Fuel chemical power in W
        """
        # Convert g/s to kg/s and MJ/kg to J/kg
        return (fuel_mass_flow / 1000) * fuel_energy_density * 1e6
    
    def calculate_engine_power(self, engine_rpm: float, engine_torque: float) -> float:
        """
        Calculate engine power output.
        
        Args:
            engine_rpm: Engine speed in RPM
            engine_torque: Engine torque in Nm
            
        Returns:
            Engine power in W
        """
        # Power (W) = Torque (Nm) * Angular velocity (rad/s)
        return engine_torque * engine_rpm * 2 * np.pi / 60
    
    def calculate_thermal_transfer(self, cooling_system: CoolingSystem,
                                 ambient_temp: float, vehicle_speed: float,
                                 engine_rpm: float, engine_load: float) -> Dict[str, float]:
        """
        Calculate heat transfer between components and to ambient.
        
        Args:
            cooling_system: CoolingSystem instance
            ambient_temp: Ambient temperature in °C
            vehicle_speed: Vehicle speed in m/s
            engine_rpm: Engine speed in RPM
            engine_load: Engine load factor (0-1)
            
        Returns:
            Dictionary with heat transfer rates
        """
        # Heat transfer between engine and oil
        temp_diff_engine_oil = self.engine_temp - self.oil_temp
        q_engine_oil = temp_diff_engine_oil * self.config.htc_oil_to_block
        
        # Heat transfer between engine and coolant
        temp_diff_engine_coolant = self.engine_temp - self.coolant_temp
        q_engine_coolant = temp_diff_engine_coolant * self.config.htc_coolant_to_block
        
        # Heat transfer from engine to ambient air (convection)
        temp_diff_engine_ambient = self.engine_temp - ambient_temp
        # Convection coefficient increases with vehicle speed
        convection_factor = 1.0 + 0.1 * vehicle_speed
        q_engine_ambient = temp_diff_engine_ambient * self.config.htc_block_to_air * convection_factor
        
        # Heat rejection from radiator
        q_radiator = cooling_system.calculate_heat_rejection(
            self.coolant_temp, ambient_temp, vehicle_speed
        )
        
        # Heat transfer from oil (simplified oil cooler model)
        # Assume oil cooler effectiveness proportional to speed and temp difference
        oil_cooler_factor = 0.005 * vehicle_speed  # W/K·(m/s)
        temp_diff_oil_ambient = self.oil_temp - ambient_temp
        q_oil_ambient = temp_diff_oil_ambient * oil_cooler_factor
        
        return {
            'engine_oil': q_engine_oil,
            'engine_coolant': q_engine_coolant,
            'engine_ambient': q_engine_ambient,
            'radiator': q_radiator,
            'oil_ambient': q_oil_ambient
        }
    
    def update_temperatures(self, heat_sources: Dict[str, float], heat_transfers: Dict[str, float], 
                          dt: float) -> Dict[str, float]:
        """
        Update component temperatures based on heat sources and transfers.
        
        Args:
            heat_sources: Dictionary with heat source rates
            heat_transfers: Dictionary with heat transfer rates
            dt: Time step in seconds
            
        Returns:
            Dictionary with updated temperatures
        """
        # Get thermal capacities
        thermal_capacities = self.config.get_thermal_capacities()
        
        # Calculate net heat for each component
        q_net_engine = (
            heat_sources['coolant'] +    # Heat to coolant (goes through engine)
            heat_sources['oil'] -        # Heat to oil (goes through engine)
            heat_transfers['engine_oil'] -      # Transfer from engine to oil
            heat_transfers['engine_coolant'] -  # Transfer from engine to coolant
            heat_transfers['engine_ambient']    # Transfer from engine to ambient
        )
        
        q_net_oil = (
            heat_transfers['engine_oil'] -      # Transfer from engine to oil
            heat_transfers['oil_ambient']       # Transfer from oil to ambient
        )
        
        q_net_coolant = (
            heat_transfers['engine_coolant'] -  # Transfer from engine to coolant
            heat_transfers['radiator']          # Transfer from coolant to ambient via radiator
        )
        
        # Calculate temperature changes
        dT_engine = q_net_engine * dt / thermal_capacities['engine_block']
        dT_oil = q_net_oil * dt / thermal_capacities['engine_oil']
        dT_coolant = q_net_coolant * dt / thermal_capacities['coolant']
        
        # Update temperatures
        self.engine_temp += dT_engine
        self.oil_temp += dT_oil
        self.coolant_temp += dT_coolant
        
        # Return updated temperatures
        return {
            'engine': self.engine_temp,
            'oil': self.oil_temp,
            'coolant': self.coolant_temp
        }
    
    def get_temperature_state(self) -> Dict[str, float]:
        """
        Get current temperature state.
        
        Returns:
            Dictionary with current temperatures
        """
        return {
            'engine': self.engine_temp,
            'oil': self.oil_temp,
            'coolant': self.coolant_temp
        }
    
    def get_temperature_effects(self) -> Dict[str, float]:
        """
        Calculate temperature effects on engine performance.
        
        Returns:
            Dictionary with temperature effect factors
        """
        # Engine temperature effect on power
        # Optimal around 90-100°C, reduced when too cold or too hot
        engine_temp_factor = 0.5 + 0.5 * np.exp(-0.002 * (self.engine_temp - 95)**2)
        
        # Oil temperature effect on friction
        # Optimal around 90-110°C, higher friction when too cold
        if self.oil_temp < 60:
            oil_temp_factor = 0.7 + 0.3 * self.oil_temp / 60  # Cold oil has high friction
        elif self.oil_temp < 100:
            oil_temp_factor = 1.0  # Optimal range
        else:
            # Slight increase in friction with very hot oil
            oil_temp_factor = 1.0 + 0.05 * (self.oil_temp - 100) / 20
        
        # Coolant temperature effect on volumetric efficiency
        # Cooler intake charge is denser (better efficiency)
        # But too cold means poor fuel vaporization
        if self.coolant_temp < 70:
            coolant_temp_factor = 0.85 + 0.15 * self.coolant_temp / 70
        elif self.coolant_temp < 90:
            coolant_temp_factor = 1.0  # Optimal range
        else:
            # Decreased volumetric efficiency with hot coolant
            coolant_temp_factor = 1.0 - 0.1 * (self.coolant_temp - 90) / 20
        
        return {
            'power': engine_temp_factor,
            'friction': oil_temp_factor,
            'volumetric_efficiency': coolant_temp_factor
        }


class ThermalSimulation:
    """Simulation of engine thermal dynamics over time."""
    
    def __init__(self, engine_model: EngineHeatModel, cooling_system: CoolingSystem):
        """
        Initialize thermal simulation.
        
        Args:
            engine_model: EngineHeatModel instance
            cooling_system: CoolingSystem instance
        """
        self.engine_model = engine_model
        self.cooling_system = cooling_system
        
        # Simulation data storage
        self.time_points = []
        self.temperature_data = []
        self.heat_flow_data = []
        self.cooling_data = []
    
    def reset(self, initial_temps: Dict[str, float] = None):
        """
        Reset simulation to initial state.
        
        Args:
            initial_temps: Optional dictionary with initial temperatures
        """
        # Reset data storage
        self.time_points = []
        self.temperature_data = []
        self.heat_flow_data = []
        self.cooling_data = []
        
        # Reset engine model temperatures
        if initial_temps:
            self.engine_model.engine_temp = initial_temps.get('engine', 25.0)
            self.engine_model.oil_temp = initial_temps.get('oil', 25.0)
            self.engine_model.coolant_temp = initial_temps.get('coolant', 25.0)
        else:
            self.engine_model.engine_temp = 25.0
            self.engine_model.oil_temp = 25.0
            self.engine_model.coolant_temp = 25.0
    
    def run_step(self, engine_rpm: float, engine_torque: float, fuel_mass_flow: float,
               ambient_temp: float, vehicle_speed: float, dt: float) -> Dict[str, float]:
        """
        Run a single simulation step.
        
        Args:
            engine_rpm: Engine speed in RPM
            engine_torque: Engine torque in Nm
            fuel_mass_flow: Fuel mass flow rate in g/s
            ambient_temp: Ambient temperature in °C
            vehicle_speed: Vehicle speed in m/s
            dt: Time step in seconds
            
        Returns:
            Dictionary with updated temperatures
        """
        # Calculate engine load (approximation)
        if engine_rpm > 0:
            engine_load = min(1.0, engine_torque / 70.0)  # Assuming 70 Nm as max torque
        else:
            engine_load = 0.0
        
        # Calculate engine and fuel power
        engine_power = self.engine_model.calculate_engine_power(engine_rpm, engine_torque)
        fuel_power = self.engine_model.calculate_fuel_power(fuel_mass_flow)
        
        # Calculate heat sources
        heat_sources = self.engine_model.calculate_heat_sources(
            fuel_power, engine_power, engine_rpm, engine_load
        )
        
        # Calculate heat transfers
        heat_transfers = self.engine_model.calculate_thermal_transfer(
            self.cooling_system, ambient_temp, vehicle_speed, engine_rpm, engine_load
        )
        
        # Update temperatures
        temps = self.engine_model.update_temperatures(heat_sources, heat_transfers, dt)
        
        # Store data
        self.time_points.append(self.time_points[-1] + dt if self.time_points else dt)
        self.temperature_data.append(temps)
        self.heat_flow_data.append({**heat_sources, **heat_transfers})
        self.cooling_data.append(self.cooling_system.get_system_state())
        
        return temps
    
    def run_profile(self, profile_data: Dict[str, np.ndarray], dt: float = 0.1) -> Dict[str, np.ndarray]:
        """
        Run simulation over a time profile.
        
        Args:
            profile_data: Dictionary with arrays for time, engine_rpm, engine_torque, etc.
            dt: Time step in seconds
            
        Returns:
            Dictionary with simulation results
        """
        # Reset simulation
        self.reset()
        
        # Extract profile data
        time = profile_data['time']
        engine_rpm = profile_data['engine_rpm']
        engine_torque = profile_data['engine_torque']
        fuel_mass_flow = profile_data.get('fuel_mass_flow', np.zeros_like(time))
        ambient_temp = profile_data.get('ambient_temp', np.full_like(time, 25.0))
        vehicle_speed = profile_data.get('vehicle_speed', np.zeros_like(time))
        
        # Run simulation for each time step
        temp_engine = []
        temp_oil = []
        temp_coolant = []
        
        # Calculate number of steps
        total_time = time[-1]
        num_steps = int(total_time / dt) + 1
        
        # Create interpolation functions for input variables
        f_rpm = interp1d(time, engine_rpm, bounds_error=False, fill_value="extrapolate")
        f_torque = interp1d(time, engine_torque, bounds_error=False, fill_value="extrapolate")
        f_fuel = interp1d(time, fuel_mass_flow, bounds_error=False, fill_value="extrapolate")
        f_ambient = interp1d(time, ambient_temp, bounds_error=False, fill_value="extrapolate")
        f_speed = interp1d(time, vehicle_speed, bounds_error=False, fill_value="extrapolate")
        
        # Run simulation
        for i in range(num_steps):
            t = i * dt
            
            # Interpolate inputs
            rpm = float(f_rpm(t))
            torque = float(f_torque(t))
            fuel = float(f_fuel(t))
            ambient = float(f_ambient(t))
            speed = float(f_speed(t))
            
            # Run step
            temps = self.run_step(rpm, torque, fuel, ambient, speed, dt)
            
            # Store results
            temp_engine.append(temps['engine'])
            temp_oil.append(temps['oil'])
            temp_coolant.append(temps['coolant'])
        
        # Create result arrays
        result_time = np.array(self.time_points)
        result_temp_engine = np.array(temp_engine)
        result_temp_oil = np.array(temp_oil)
        result_temp_coolant = np.array(temp_coolant)
        
        return {
            'time': result_time,
            'temp_engine': result_temp_engine,
            'temp_oil': result_temp_oil,
            'temp_coolant': result_temp_coolant
        }
    
    def run_steady_state(self, engine_rpm: float, engine_torque: float, fuel_mass_flow: float,
                       ambient_temp: float, vehicle_speed: float, 
                       max_time: float = 600.0, tolerance: float = 0.1) -> Dict[str, float]:
        """
        Run simulation until steady state is reached.
        
        Args:
            engine_rpm: Engine speed in RPM
            engine_torque: Engine torque in Nm
            fuel_mass_flow: Fuel mass flow rate in g/s
            ambient_temp: Ambient temperature in °C
            vehicle_speed: Vehicle speed in m/s
            max_time: Maximum simulation time in seconds
            tolerance: Temperature change tolerance for steady state in °C/s
            
        Returns:
            Dictionary with steady state temperatures
        """
        # Reset simulation
        self.reset()
        
        # Run simulation steps until steady state or max time
        time = 0.0
        dt = 1.0  # 1 second time step for steady state calculation
        
        while time < max_time:
            # Run step
            temps = self.run_step(
                engine_rpm, engine_torque, fuel_mass_flow,
                ambient_temp, vehicle_speed, dt
            )
            
            time += dt
            
            # Check if steady state reached (last 30 seconds)
            if len(self.time_points) > 30:
                # Calculate temperature change rates over last 30 seconds
                engine_rate = abs(temps['engine'] - self.temperature_data[-30]['engine']) / 30
                oil_rate = abs(temps['oil'] - self.temperature_data[-30]['oil']) / 30
                coolant_rate = abs(temps['coolant'] - self.temperature_data[-30]['coolant']) / 30
                
                if engine_rate < tolerance and oil_rate < tolerance and coolant_rate < tolerance:
                    break
        
        return {
            'engine': temps['engine'],
            'oil': temps['oil'],
            'coolant': temps['coolant'],
            'time_to_steady': time
        }
    
    def plot_temperature_profile(self, save_path: Optional[str] = None):
        """
        Plot temperature profile over time.
        
        Args:
            save_path: Optional path to save the plot
        """
        if not self.time_points:
            raise ValueError("No simulation data available")
        
        # Extract temperature data
        time = np.array(self.time_points)
        temp_engine = np.array([data['engine'] for data in self.temperature_data])
        temp_oil = np.array([data['oil'] for data in self.temperature_data])
        temp_coolant = np.array([data['coolant'] for data in self.temperature_data])
        
        plt.figure(figsize=(10, 6))
        
        # Plot temperatures
        plt.plot(time, temp_engine, 'r-', label='Engine Block')
        plt.plot(time, temp_oil, 'g-', label='Oil')
        plt.plot(time, temp_coolant, 'b-', label='Coolant')
        
        # Add labels and legend
        plt.xlabel('Time (s)')
        plt.ylabel('Temperature (°C)')
        plt.title('Engine Thermal Simulation')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Set reasonable y-axis limits
        plt.ylim(20, max(max(temp_engine), max(temp_oil), max(temp_coolant)) * 1.1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_heat_flow(self, save_path: Optional[str] = None):
        """
        Plot heat flow over time.
        
        Args:
            save_path: Optional path to save the plot
        """
        if not self.time_points:
            raise ValueError("No simulation data available")
        
        # Extract heat flow data
        time = np.array(self.time_points)
        heat_total = np.array([data['total'] for data in self.heat_flow_data])
        heat_coolant = np.array([data['coolant'] for data in self.heat_flow_data])
        heat_oil = np.array([data['oil'] for data in self.heat_flow_data])
        heat_radiator = np.array([data['radiator'] for data in self.heat_flow_data])
        
        plt.figure(figsize=(10, 6))
        
        # Plot heat flows
        plt.plot(time, heat_total / 1000, 'k-', label='Total Waste Heat')
        plt.plot(time, heat_coolant / 1000, 'r-', label='To Coolant')
        plt.plot(time, heat_oil / 1000, 'g-', label='To Oil')
        plt.plot(time, heat_radiator / 1000, 'b-', label='Radiator Rejection')
        
        # Add labels and legend
        plt.xlabel('Time (s)')
        plt.ylabel('Heat Flow (kW)')
        plt.title('Engine Heat Flow')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_cooling_system(self, save_path: Optional[str] = None):
        """
        Plot cooling system state over time.
        
        Args:
            save_path: Optional path to save the plot
        """
        if not self.time_points:
            raise ValueError("No simulation data available")
        
        # Extract cooling system data
        time = np.array(self.time_points)
        fan_state = np.array([data['fan_state'] for data in self.cooling_data])
        thermostat = np.array([data['thermostat_position'] for data in self.cooling_data])
        airflow = np.array([data['radiator_airflow'] for data in self.cooling_data])
        
        # Create plot with two y-axes
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # Plot fan state and thermostat position
        ax1.plot(time, fan_state, 'r-', label='Fan Duty Cycle')
        ax1.plot(time, thermostat, 'g-', label='Thermostat Position')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('State (0-1)')
        ax1.set_ylim(0, 1.1)
        
        # Create secondary y-axis for airflow
        ax2 = ax1.twinx()
        ax2.plot(time, airflow, 'b-', label='Radiator Airflow')
        ax2.set_ylabel('Airflow (m³/s)')
        
        # Add labels and legend
        plt.title('Cooling System State')
        ax1.grid(True, alpha=0.3)
        
        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


class CoolingPerformance:
    """Analysis of cooling system performance for Formula Student applications."""
    
    def __init__(self, engine_model: EngineHeatModel, cooling_system: CoolingSystem):
        """
        Initialize cooling performance analyzer.
        
        Args:
            engine_model: EngineHeatModel instance
            cooling_system: CoolingSystem instance
        """
        self.engine_model = engine_model
        self.cooling_system = cooling_system
        self.simulation = ThermalSimulation(engine_model, cooling_system)
        
        # Performance data storage
        self.steady_state_map = {}
        self.transient_responses = {}
    
    def generate_steady_state_map(self, rpm_range: List[float], torque_range: List[float],
                                ambient_temp: float = 25.0, vehicle_speed: float = 10.0,
                                fuel_consumption=None) -> Dict:
        """
        Generate steady state temperature map for different operating points.
        
        Args:
            rpm_range: List of engine speeds in RPM
            torque_range: List of engine torques in Nm
            ambient_temp: Ambient temperature in °C
            vehicle_speed: Vehicle speed in m/s
            fuel_consumption: Optional function to calculate fuel consumption (g/s)
                              from RPM and torque
            
        Returns:
            Dictionary with steady state temperature maps
        """
        # Initialize result matrices
        n_rpm = len(rpm_range)
        n_torque = len(torque_range)
        
        temp_engine = np.zeros((n_rpm, n_torque))
        temp_oil = np.zeros((n_rpm, n_torque))
        temp_coolant = np.zeros((n_rpm, n_torque))
        time_to_steady = np.zeros((n_rpm, n_torque))
        
        # Calculate steady state temperature for each operating point
        for i, rpm in enumerate(rpm_range):
            for j, torque in enumerate(torque_range):
                print(f"Calculating steady state for {rpm} RPM, {torque} Nm...")
                
                # Calculate fuel consumption if function provided, otherwise estimate
                if fuel_consumption:
                    fuel_flow = fuel_consumption(rpm, torque)
                else:
                    # Rough estimate: 250 g/kWh BSFC
                    power_kw = self.engine_model.calculate_engine_power(rpm, torque) / 1000
                    fuel_flow = power_kw * 250 / 3600 if power_kw > 0 else 0.1
                
                # Run steady state simulation
                result = self.simulation.run_steady_state(
                    rpm, torque, fuel_flow, ambient_temp, vehicle_speed
                )
                
                # Store results
                temp_engine[i, j] = result['engine']
                temp_oil[i, j] = result['oil']
                temp_coolant[i, j] = result['coolant']
                time_to_steady[i, j] = result['time_to_steady']
        
        # Store results
        self.steady_state_map = {
            'rpm': np.array(rpm_range),
            'torque': np.array(torque_range),
            'temp_engine': temp_engine,
            'temp_oil': temp_oil,
            'temp_coolant': temp_coolant,
            'time_to_steady': time_to_steady,
            'ambient_temp': ambient_temp,
            'vehicle_speed': vehicle_speed
        }
        
        return self.steady_state_map
    
    def analyze_transient_response(self, rpm_step: Tuple[float, float], 
                                 torque_step: Tuple[float, float],
                                 ambient_temp: float = 25.0, 
                                 vehicle_speed: float = 10.0,
                                 fuel_consumption=None,
                                 duration: float = 300.0) -> Dict:
        """
        Analyze transient thermal response to step changes.
        
        Args:
            rpm_step: Tuple of (initial_rpm, final_rpm)
            torque_step: Tuple of (initial_torque, final_torque)
            ambient_temp: Ambient temperature in °C
            vehicle_speed: Vehicle speed in m/s
            fuel_consumption: Optional function to calculate fuel consumption
            duration: Simulation duration after step in seconds
            
        Returns:
            Dictionary with transient response data
        """
        # Reset simulation
        self.simulation.reset()
        
        # Initial conditions - run to steady state
        initial_rpm, final_rpm = rpm_step
        initial_torque, final_torque = torque_step
        
        # Calculate fuel consumption
        if fuel_consumption:
            initial_fuel = fuel_consumption(initial_rpm, initial_torque)
            final_fuel = fuel_consumption(final_rpm, final_torque)
        else:
            # Rough estimate
            initial_power = self.engine_model.calculate_engine_power(initial_rpm, initial_torque) / 1000
            final_power = self.engine_model.calculate_engine_power(final_rpm, final_torque) / 1000
            initial_fuel = initial_power * 250 / 3600 if initial_power > 0 else 0.1
            final_fuel = final_power * 250 / 3600 if final_power > 0 else 0.1
        
        # Run initial steady state
        print("Running initial steady state...")
        initial_state = self.simulation.run_steady_state(
            initial_rpm, initial_torque, initial_fuel, ambient_temp, vehicle_speed
        )
        
        # Reset simulation but keep final temperatures
        self.simulation.reset(initial_state)
        
        # Create time profile for step change
        dt = 0.1  # 0.1 second time step
        time = np.arange(0, duration + dt, dt)
        n_steps = len(time)
        
        # Create step profiles
        rpm_profile = np.full_like(time, final_rpm)
        torque_profile = np.full_like(time, final_torque)
        fuel_profile = np.full_like(time, final_fuel)
        ambient_profile = np.full_like(time, ambient_temp)
        speed_profile = np.full_like(time, vehicle_speed)
        
        # Run simulation
        print("Running transient response...")
        profile_data = {
            'time': time,
            'engine_rpm': rpm_profile,
            'engine_torque': torque_profile,
            'fuel_mass_flow': fuel_profile,
            'ambient_temp': ambient_profile,
            'vehicle_speed': speed_profile
        }
        
        results = self.simulation.run_profile(profile_data, dt)
        
        # Calculate response characteristics
        temp_engine = results['temp_engine']
        temp_oil = results['temp_oil']
        temp_coolant = results['temp_coolant']
        
        # Find 63% rise time (time constant)
        engine_final = temp_engine[-1]
        engine_initial = temp_engine[0]
        engine_delta = engine_final - engine_initial
        engine_target = engine_initial + 0.63 * engine_delta
        
        oil_final = temp_oil[-1]
        oil_initial = temp_oil[0]
        oil_delta = oil_final - oil_initial
        oil_target = oil_initial + 0.63 * oil_delta
        
        coolant_final = temp_coolant[-1]
        coolant_initial = temp_coolant[0]
        coolant_delta = coolant_final - coolant_initial
        coolant_target = coolant_initial + 0.63 * coolant_delta
        
        # Find closest points to 63% rise
        engine_tc_idx = np.argmin(np.abs(temp_engine - engine_target))
        oil_tc_idx = np.argmin(np.abs(temp_oil - oil_target))
        coolant_tc_idx = np.argmin(np.abs(temp_coolant - coolant_target))
        
        engine_time_constant = time[engine_tc_idx]
        oil_time_constant = time[oil_tc_idx]
        coolant_time_constant = time[coolant_tc_idx]
        
        # Store results
        response_data = {
            'time': time,
            'temp_engine': temp_engine,
            'temp_oil': temp_oil,
            'temp_coolant': temp_coolant,
            'engine_initial': engine_initial,
            'engine_final': engine_final,
            'oil_initial': oil_initial,
            'oil_final': oil_final,
            'coolant_initial': coolant_initial,
            'coolant_final': coolant_final,
            'engine_time_constant': engine_time_constant,
            'oil_time_constant': oil_time_constant,
            'coolant_time_constant': coolant_time_constant
        }
        
        # Store in instance for later analysis
        step_key = f"{initial_rpm}-{final_rpm}_{initial_torque}-{final_torque}"
        self.transient_responses[step_key] = response_data
        
        return response_data
    
    def analyze_cooling_system_sizing(self, reference_rpm: float, reference_torque: float,
                                    ambient_range: List[float], 
                                    radiator_sizes: List[float],
                                    fuel_consumption=None) -> Dict:
        """
        Analyze cooling system sizing requirements.
        
        Args:
            reference_rpm: Reference engine RPM for analysis
            reference_torque: Reference engine torque for analysis
            ambient_range: List of ambient temperatures to analyze
            radiator_sizes: List of radiator sizes (relative to baseline)
            fuel_consumption: Optional function to calculate fuel consumption
            
        Returns:
            Dictionary with cooling system sizing analysis
        """
        # Initialize results
        n_ambient = len(ambient_range)
        n_sizes = len(radiator_sizes)
        
        max_coolant_temp = np.zeros((n_ambient, n_sizes))
        cooling_margin = np.zeros((n_ambient, n_sizes))
        
        # Save original radiator size
        original_size = self.cooling_system.config.radiator_area
        
        # Calculate fuel consumption
        if fuel_consumption:
            fuel_flow = fuel_consumption(reference_rpm, reference_torque)
        else:
            # Rough estimate
            power_kw = self.engine_model.calculate_engine_power(reference_rpm, reference_torque) / 1000
            fuel_flow = power_kw * 250 / 3600 if power_kw > 0 else 0.1
        
        vehicle_speed = 10.0  # m/s (consistent test condition)
        
        # Run analysis for each combination
        for i, ambient in enumerate(ambient_range):
            for j, size_factor in enumerate(radiator_sizes):
                print(f"Analyzing ambient {ambient}°C, radiator size factor {size_factor}...")
                
                # Adjust radiator size
                self.cooling_system.config.radiator_area = original_size * size_factor
                
                # Run steady state simulation
                result = self.simulation.run_steady_state(
                    reference_rpm, reference_torque, fuel_flow, ambient, vehicle_speed
                )
                
                # Store results
                max_coolant_temp[i, j] = result['coolant']
                cooling_margin[i, j] = self.cooling_system.config.max_engine_temp - result['engine']
        
        # Restore original radiator size
        self.cooling_system.config.radiator_area = original_size
        
        # Store and return results
        sizing_results = {
            'ambient_temps': np.array(ambient_range),
            'radiator_sizes': np.array(radiator_sizes),
            'max_coolant_temp': max_coolant_temp,
            'cooling_margin': cooling_margin,
            'reference_rpm': reference_rpm,
            'reference_torque': reference_torque
        }
        
        return sizing_results
    
    def plot_steady_state_map(self, temp_type: str = 'coolant', save_path: Optional[str] = None):
        """
        Plot steady state temperature map.
        
        Args:
            temp_type: Type of temperature to plot ('engine', 'oil', or 'coolant')
            save_path: Optional path to save the plot
        """
        if not self.steady_state_map:
            raise ValueError("Steady state map not generated")
        
        # Get data
        rpm = self.steady_state_map['rpm']
        torque = self.steady_state_map['torque']
        
        if temp_type == 'engine':
            temp_data = self.steady_state_map['temp_engine']
            title = 'Engine Block Temperature (°C)'
        elif temp_type == 'oil':
            temp_data = self.steady_state_map['temp_oil']
            title = 'Oil Temperature (°C)'
        elif temp_type == 'coolant':
            temp_data = self.steady_state_map['temp_coolant']
            title = 'Coolant Temperature (°C)'
        else:
            raise ValueError("temp_type must be 'engine', 'oil', or 'coolant'")
        
        # Create meshgrid for contour plot
        rpm_grid, torque_grid = np.meshgrid(rpm, torque)
        
        plt.figure(figsize=(10, 8))
        
        # Create contour plot
        contour = plt.contourf(rpm_grid, torque_grid, temp_data.T, 20, cmap='hot')
        plt.colorbar(contour, label='Temperature (°C)')
        
        # Add contour lines with labels
        contour_lines = plt.contour(rpm_grid, torque_grid, temp_data.T, 10, colors='black', alpha=0.5)
        plt.clabel(contour_lines, inline=True, fontsize=8)
        
        # Add labels and title
        plt.xlabel('Engine Speed (RPM)')
        plt.ylabel('Engine Torque (Nm)')
        plt.title(f'Steady State {title}')
        
        # Add annotation with conditions
        ambient = self.steady_state_map['ambient_temp']
        speed = self.steady_state_map['vehicle_speed']
        plt.annotate(f'Ambient: {ambient}°C, Vehicle Speed: {speed} m/s',
                    xy=(0.05, 0.05), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
        
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_transient_response(self, step_key: Optional[str] = None, save_path: Optional[str] = None):
        """
        Plot transient thermal response.
        
        Args:
            step_key: Optional key for specific response to plot
            save_path: Optional path to save the plot
        """
        if not self.transient_responses:
            raise ValueError("No transient responses available")
        
        # Use the most recent response if no key provided
        if step_key is None:
            step_key = list(self.transient_responses.keys())[-1]
        
        if step_key not in self.transient_responses:
            raise ValueError(f"Step key {step_key} not found in transient responses")
        
        # Get data
        response = self.transient_responses[step_key]
        time = response['time']
        temp_engine = response['temp_engine']
        temp_oil = response['temp_oil']
        temp_coolant = response['temp_coolant']
        
        plt.figure(figsize=(10, 6))
        
        # Plot temperatures
        plt.plot(time, temp_engine, 'r-', label='Engine Block')
        plt.plot(time, temp_oil, 'g-', label='Oil')
        plt.plot(time, temp_coolant, 'b-', label='Coolant')
        
        # Plot time constants
        engine_tc = response['engine_time_constant']
        oil_tc = response['oil_time_constant']
        coolant_tc = response['coolant_time_constant']
        
        plt.axvline(x=engine_tc, color='r', linestyle='--', alpha=0.5,
                   label=f'Engine τ = {engine_tc:.1f}s')
        plt.axvline(x=oil_tc, color='g', linestyle='--', alpha=0.5,
                   label=f'Oil τ = {oil_tc:.1f}s')
        plt.axvline(x=coolant_tc, color='b', linestyle='--', alpha=0.5,
                   label=f'Coolant τ = {coolant_tc:.1f}s')
        
        # Add labels and legend
        plt.xlabel('Time (s)')
        plt.ylabel('Temperature (°C)')
        plt.title(f'Transient Thermal Response: {step_key}')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_cooling_sizing(self, sizing_results: Dict, save_path: Optional[str] = None):
        """
        Plot cooling system sizing analysis.
        
        Args:
            sizing_results: Results from analyze_cooling_system_sizing
            save_path: Optional path to save the plot
        """
        ambient_temps = sizing_results['ambient_temps']
        radiator_sizes = sizing_results['radiator_sizes']
        max_coolant_temp = sizing_results['max_coolant_temp']
        cooling_margin = sizing_results['cooling_margin']
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Create meshgrid
        ambient_grid, size_grid = np.meshgrid(ambient_temps, radiator_sizes)
        
        # Plot max coolant temperature
        contour1 = ax1.contourf(ambient_grid, size_grid, max_coolant_temp.T, 20, cmap='hot')
        fig.colorbar(contour1, ax=ax1, label='Max Coolant Temp (°C)')
        
        # Add contour lines with labels
        contour_lines1 = ax1.contour(ambient_grid, size_grid, max_coolant_temp.T, 
                                    levels=[85, 90, 95, 100, 105, 110], colors='black', alpha=0.5)
        ax1.clabel(contour_lines1, inline=True, fontsize=8)
        
        # Plot cooling margin
        contour2 = ax2.contourf(ambient_grid, size_grid, cooling_margin.T, 20, cmap='viridis')
        fig.colorbar(contour2, ax=ax2, label='Cooling Margin (°C)')
        
        # Add contour lines with labels
        contour_lines2 = ax2.contour(ambient_grid, size_grid, cooling_margin.T, 
                                   levels=[5, 10, 15, 20, 25], colors='black', alpha=0.5)
        ax2.clabel(contour_lines2, inline=True, fontsize=8)
        
        # Add red line for minimum acceptable margin (10°C)
        safe_level = np.ones_like(ambient_temps) * 10
        ax2.plot(ambient_temps, safe_level, 'r--', linewidth=2, label='Min Safe Margin')
        
        # Add labels
        ax1.set_xlabel('Ambient Temperature (°C)')
        ax1.set_ylabel('Radiator Size Factor')
        ax1.set_title('Maximum Coolant Temperature')
        
        ax2.set_xlabel('Ambient Temperature (°C)')
        ax2.set_ylabel('Radiator Size Factor')
        ax2.set_title('Engine Cooling Margin')
        ax2.legend()
        
        # Add overall title
        rpm = sizing_results['reference_rpm']
        torque = sizing_results['reference_torque']
        plt.suptitle(f'Cooling System Sizing Analysis: {rpm} RPM, {torque} Nm', fontsize=14)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for title
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


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
        
        # Create thermal configuration
        thermal_config = ThermalConfig()
        
        # Create cooling system
        cooling_system = CoolingSystem(thermal_config)
        
        # Create engine heat model
        heat_model = EngineHeatModel(thermal_config, engine)
        
        # Create thermal simulation
        simulation = ThermalSimulation(heat_model, cooling_system)
        
        print("Running thermal simulation...")
        
        # Create a simple test profile
        # Start with idle, then high load, then cool down
        total_time = 600  # 10 minutes
        time = np.arange(0, total_time, 1)  # 1 second steps
        
        # Create profile segments
        idle_time = 60
        high_load_time = 300
        cool_down_time = total_time - idle_time - high_load_time
        
        # RPM profile: idle -> high RPM -> idle
        rpm_profile = np.zeros_like(time)
        rpm_profile[:idle_time] = engine.idle_rpm
        rpm_profile[idle_time:idle_time+high_load_time] = 10000
        rpm_profile[idle_time+high_load_time:] = engine.idle_rpm
        
        # Torque profile
        torque_profile = np.zeros_like(time)
        torque_profile[:idle_time] = 5  # Low torque at idle
        torque_profile[idle_time:idle_time+high_load_time] = 60  # High torque
        torque_profile[idle_time+high_load_time:] = 5  # Back to idle
        
        # Simple fuel consumption estimate
        fuel_profile = np.zeros_like(time)
        for i, (rpm, torque) in enumerate(zip(rpm_profile, torque_profile)):
            power_kw = heat_model.calculate_engine_power(rpm, torque) / 1000
            fuel_profile[i] = max(0.1, power_kw * 250 / 3600)  # Rough BSFC-based estimate
        
        # Ambient conditions
        ambient_temp = np.full_like(time, 25.0)  # Constant ambient temperature
        vehicle_speed = np.zeros_like(time)
        vehicle_speed[:idle_time] = 0  # Stationary
        vehicle_speed[idle_time:idle_time+high_load_time] = 15  # Moving at 15 m/s (~54 km/h)
        vehicle_speed[idle_time+high_load_time:] = 0  # Stationary again
        
        # Create profile data
        profile_data = {
            'time': time,
            'engine_rpm': rpm_profile,
            'engine_torque': torque_profile,
            'fuel_mass_flow': fuel_profile,
            'ambient_temp': ambient_temp,
            'vehicle_speed': vehicle_speed
        }
        
        # Run simulation
        results = simulation.run_profile(profile_data)
        
        # Plot results
        simulation.plot_temperature_profile()
        simulation.plot_heat_flow()
        simulation.plot_cooling_system()
        
        # Analyze cooling performance
        cooling_performance = CoolingPerformance(heat_model, cooling_system)
        
        # Run a transient response analysis (simplified for example)
        response = cooling_performance.analyze_transient_response(
            rpm_step=(engine.idle_rpm, 10000),
            torque_step=(5, 60),
            ambient_temp=25.0,
            vehicle_speed=15.0,
            duration=300
        )
        
        # Plot transient response
        cooling_performance.plot_transient_response()
        
        print("Thermal simulation completed.")
        
    except ImportError:
        print("MotorcycleEngine class not available, running with simplified models")
        
        # Create configuration
        thermal_config = ThermalConfig()
        
        # Print configuration
        print("Thermal Configuration:")
        for key, value in thermal_config.to_dict().items():
            print(f"  {key}: {value}")
        
        # Create cooling system and engine heat model
        cooling_system = CoolingSystem(thermal_config)
        heat_model = EngineHeatModel(thermal_config)
        
        # Run a simple test
        engine_rpm = 6000
        engine_torque = 40
        fuel_flow = 1.2  # g/s
        ambient_temp = 25
        vehicle_speed = 10
        
        # Calculate heat rejection
        heat_rejection = cooling_system.calculate_heat_rejection(
            coolant_temp=90, ambient_temp=ambient_temp, vehicle_speed=vehicle_speed
        )
        
        print(f"\nAt {vehicle_speed} m/s, radiator can reject {heat_rejection/1000:.2f} kW of heat")
        
        # Calculate heat generation
        engine_power = heat_model.calculate_engine_power(engine_rpm, engine_torque)
        fuel_power = heat_model.calculate_fuel_power(fuel_flow)
        
        heat_sources = heat_model.calculate_heat_sources(
            fuel_power, engine_power, engine_rpm, 0.7
        )
        
        print("\nHeat Generation at 6000 RPM, 40 Nm:")
        print(f"  Engine Power: {engine_power/1000:.2f} kW")
        print(f"  Fuel Power: {fuel_power/1000:.2f} kW")
        print(f"  Total Waste Heat: {heat_sources['total']/1000:.2f} kW")
        print(f"  Heat to Coolant: {heat_sources['coolant']/1000:.2f} kW")
        print(f"  Heat to Oil: {heat_sources['oil']/1000:.2f} kW")