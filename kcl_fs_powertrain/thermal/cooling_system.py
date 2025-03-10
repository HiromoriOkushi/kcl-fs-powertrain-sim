"""
Cooling system module for Formula Student powertrain simulation.

This module provides detailed models of cooling system components for Formula Student
car applications, including radiators, water pumps, cooling fans, and thermostats.
It calculates heat rejection capacity, coolant flow, and temperature distribution
throughout the cooling system under various operating conditions.

The module is designed to work with the engine thermal model to provide comprehensive
thermal management simulation for the Honda CBR600F4i engine adapted for Formula Student.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Callable
import logging
from enum import Enum, auto

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("Cooling_System")


class RadiatorType(Enum):
    """Enumeration of different radiator types used in Formula Student cars."""
    SINGLE_CORE_ALUMINUM = auto()  # Standard aluminum single-core radiator
    DOUBLE_CORE_ALUMINUM = auto()  # High-performance dual-core aluminum radiator
    SINGLE_CORE_COPPER = auto()    # Copper-based single-core radiator
    CUSTOM = auto()                # Custom specification radiator


class PumpType(Enum):
    """Enumeration of different water pump types."""
    MECHANICAL = auto()  # Engine-driven mechanical water pump
    ELECTRIC = auto()    # Electric water pump with variable flow control
    CUSTOM = auto()      # Custom specification pump


class FanType(Enum):
    """Enumeration of different cooling fan types."""
    SINGLE_SPEED = auto()    # Basic single-speed cooling fan
    VARIABLE_SPEED = auto()  # Variable speed cooling fan with PWM control
    DUAL_FAN = auto()        # Dual fan configuration for enhanced airflow
    CUSTOM = auto()          # Custom specification fan arrangement


class Radiator:
    """
    Radiator model for Formula Student cooling system.
    
    This class models the radiator's thermal behavior, including heat transfer
    characteristics and airflow effects. It calculates heat rejection capacity
    based on coolant flow, air flow, and temperature differentials.
    """
    
    def __init__(self, 
                 radiator_type: RadiatorType = RadiatorType.SINGLE_CORE_ALUMINUM,
                 core_area: float = 0.15,        # m²
                 core_thickness: float = 0.045,  # m
                 fin_density: float = 14,        # fins/inch
                 tube_rows: int = 2,
                 max_pressure: float = 1.5,      # bar
                 custom_params: Optional[Dict] = None):
        """
        Initialize the radiator model with specific parameters.
        
        Args:
            radiator_type: Type of radiator construction
            core_area: Radiator core area in m²
            core_thickness: Radiator core thickness in m
            fin_density: Number of cooling fins per inch
            tube_rows: Number of tube rows in the core
            max_pressure: Maximum system pressure in bar
            custom_params: Optional dictionary with custom parameters
        """
        self.radiator_type = radiator_type
        self.core_area = core_area
        self.core_thickness = core_thickness
        self.fin_density = fin_density
        self.tube_rows = tube_rows
        self.max_pressure = max_pressure
        
        # Initialize derived properties
        self.total_volume = core_area * core_thickness  # m³
        self.air_side_surface_area = self._calculate_surface_area()  # m²
        
        # Default thermal properties based on radiator type
        if radiator_type == RadiatorType.SINGLE_CORE_ALUMINUM:
            self.base_effectiveness = 0.68
            self.thermal_conductivity = 205  # W/(m·K) for aluminum
            self.weight = core_area * 5.0  # kg, estimated based on size
        elif radiator_type == RadiatorType.DOUBLE_CORE_ALUMINUM:
            self.base_effectiveness = 0.75
            self.thermal_conductivity = 205  # W/(m·K) for aluminum
            self.weight = core_area * 7.5  # kg, estimated based on size
        elif radiator_type == RadiatorType.SINGLE_CORE_COPPER:
            self.base_effectiveness = 0.72
            self.thermal_conductivity = 385  # W/(m·K) for copper
            self.weight = core_area * 8.5  # kg, estimated based on size
        elif radiator_type == RadiatorType.CUSTOM:
            # Use custom parameters if provided
            if custom_params:
                self.base_effectiveness = custom_params.get('effectiveness', 0.70)
                self.thermal_conductivity = custom_params.get('thermal_conductivity', 205)
                self.weight = custom_params.get('weight', core_area * 6.0)
            else:
                # Default values for custom type
                self.base_effectiveness = 0.70
                self.thermal_conductivity = 205
                self.weight = core_area * 6.0
        
        # Current state parameters
        self.current_effectiveness = self.base_effectiveness
        
        logger.info(f"Radiator initialized: {radiator_type.name}, {core_area:.2f}m² core area")
    
    def _calculate_surface_area(self) -> float:
        """
        Calculate the air side surface area of the radiator.
        
        Returns:
            Surface area in m²
        """
        # Convert fin density from fins/inch to fins/m
        fins_per_meter = self.fin_density * 39.37
        
        # Estimate fin area based on core dimensions and fin density
        # This is a simplified model that can be refined with specific radiator data
        fin_thickness = 0.0001  # m, typical fin thickness
        fin_height = self.core_thickness * 0.8  # m, estimate
        fin_count = self.core_area * fins_per_meter
        
        # Area contribution from fins
        fin_area = fin_count * 2 * fin_height * (self.core_area / fin_count)**0.5
        
        # Area contribution from tubes
        tube_area = self.core_area * self.tube_rows * 1.2  # Factor for tube surface exposure
        
        # Total air-side surface area
        return fin_area + tube_area
    
    def calculate_heat_rejection(self, coolant_temp: float, ambient_temp: float, 
                               coolant_flow_rate: float, air_flow_rate: float) -> float:
        """
        Calculate heat rejected by the radiator under given conditions.
        
        Args:
            coolant_temp: Coolant temperature in °C
            ambient_temp: Ambient air temperature in °C
            coolant_flow_rate: Coolant flow rate in L/min
            air_flow_rate: Air flow rate through radiator in m³/s
            
        Returns:
            Heat rejection rate in watts (W)
        """
        # Update effectiveness based on flow rates
        self._update_effectiveness(coolant_flow_rate, air_flow_rate)
        
        # Calculate temperature difference
        delta_t = coolant_temp - ambient_temp
        
        # Calculate coolant properties
        coolant_density = 1000  # kg/m³ for water-based coolant
        coolant_specific_heat = 3900  # J/(kg·K) for typical coolant
        
        # Convert coolant flow from L/min to kg/s
        coolant_mass_flow = coolant_flow_rate * (coolant_density / 60000)
        
        # Calculate air properties
        air_density = 1.2  # kg/m³
        air_specific_heat = 1005  # J/(kg·K)
        
        # Calculate air mass flow
        air_mass_flow = air_flow_rate * air_density
        
        # Calculate capacity rates
        c_coolant = coolant_mass_flow * coolant_specific_heat
        c_air = air_mass_flow * air_specific_heat
        
        # Use minimum capacity rate for heat transfer calculation
        c_min = min(c_coolant, c_air)
        
        # Calculate heat rejection using effectiveness-NTU method
        heat_rejection = self.current_effectiveness * c_min * delta_t
        
        return max(0, heat_rejection)  # Can't reject heat if delta_t <= 0
    
    def _update_effectiveness(self, coolant_flow_rate: float, air_flow_rate: float):
        """
        Update radiator effectiveness based on flow rates.
        
        Args:
            coolant_flow_rate: Coolant flow rate in L/min
            air_flow_rate: Air flow rate in m³/s
        """
        # This is a simplified model that adjusts effectiveness based on flow rates
        # Flow rate effects on effectiveness
        if coolant_flow_rate < 10:  # Low coolant flow
            coolant_factor = 0.8 + 0.2 * (coolant_flow_rate / 10)
        else:  # Normal to high coolant flow
            coolant_factor = 1.0 + 0.05 * min(1.0, (coolant_flow_rate - 10) / 20)
        
        # Air flow effects
        if air_flow_rate < 0.3:  # Low air flow
            air_factor = 0.7 + 0.3 * (air_flow_rate / 0.3)
        else:  # Normal to high air flow
            air_factor = 1.0 + 0.1 * min(1.0, (air_flow_rate - 0.3) / 0.5)
        
        # Update effectiveness
        self.current_effectiveness = self.base_effectiveness * coolant_factor * air_factor
    
    def calculate_pressure_drop(self, coolant_flow_rate: float) -> float:
        """
        Calculate pressure drop across the radiator at given flow rate.
        
        Args:
            coolant_flow_rate: Coolant flow rate in L/min
            
        Returns:
            Pressure drop in bar
        """
        # Simplified quadratic model for pressure drop
        # Coefficients can be tuned based on specific radiator data
        base_flow = 15  # L/min, reference flow rate
        base_drop = 0.2  # bar, pressure drop at reference flow
        
        # Scale quadratically with flow rate
        factor = (coolant_flow_rate / base_flow)**2
        pressure_drop = base_drop * factor
        
        return pressure_drop
    
    def calculate_coolant_exit_temp(self, inlet_temp: float, heat_rejection: float, 
                                  coolant_flow_rate: float) -> float:
        """
        Calculate coolant exit temperature after passing through radiator.
        
        Args:
            inlet_temp: Coolant inlet temperature in °C
            heat_rejection: Heat rejected by radiator in watts
            coolant_flow_rate: Coolant flow rate in L/min
            
        Returns:
            Coolant exit temperature in °C
        """
        # Calculate coolant properties
        coolant_density = 1000  # kg/m³
        coolant_specific_heat = 3900  # J/(kg·K)
        
        # Convert flow rate to kg/s
        mass_flow = coolant_flow_rate * (coolant_density / 60000)
        
        # Calculate temperature change (Q = m * c * ΔT)
        if mass_flow > 0:
            delta_t = heat_rejection / (mass_flow * coolant_specific_heat)
        else:
            delta_t = 0
        
        # Calculate exit temperature
        exit_temp = inlet_temp - delta_t
        
        return exit_temp
    
    def get_radiator_specs(self) -> Dict:
        """
        Get specifications of the radiator.
        
        Returns:
            Dictionary with radiator specifications
        """
        return {
            'type': self.radiator_type.name,
            'core_area': self.core_area,
            'core_thickness': self.core_thickness,
            'fin_density': self.fin_density,
            'tube_rows': self.tube_rows,
            'max_pressure': self.max_pressure,
            'base_effectiveness': self.base_effectiveness,
            'thermal_conductivity': self.thermal_conductivity,
            'weight': self.weight,
            'air_side_surface_area': self.air_side_surface_area
        }


class WaterPump:
    """
    Water pump model for Formula Student cooling system.
    
    This class models the water pump's flow characteristics based on pump type,
    including pressure-flow relationships and power consumption. It can represent
    either mechanical (engine-driven) or electric water pumps.
    """
    
    def __init__(self, 
                 pump_type: PumpType = PumpType.MECHANICAL,
                 max_flow_rate: float = 80.0,     # L/min
                 max_pressure: float = 1.8,       # bar
                 nominal_speed: float = 3600.0,   # RPM
                 mechanical_efficiency: float = 0.65,
                 custom_params: Optional[Dict] = None):
        """
        Initialize water pump model.
        
        Args:
            pump_type: Type of water pump
            max_flow_rate: Maximum flow rate in L/min
            max_pressure: Maximum pressure in bar
            nominal_speed: Nominal pump speed in RPM
            mechanical_efficiency: Pump mechanical efficiency (0-1)
            custom_params: Optional dictionary with custom parameters
        """
        self.pump_type = pump_type
        self.max_flow_rate = max_flow_rate
        self.max_pressure = max_pressure
        self.nominal_speed = nominal_speed
        self.mechanical_efficiency = mechanical_efficiency
        
        # Set pump-specific parameters based on type
        if pump_type == PumpType.MECHANICAL:
            self.speed_ratio = 1.0  # Speed ratio relative to engine speed
            self.power_consumption_max = 500.0  # W
            self.weight = 0.8  # kg
        elif pump_type == PumpType.ELECTRIC:
            self.speed_ratio = 0.0  # Not tied to engine speed
            self.power_consumption_max = 150.0  # W
            self.weight = 0.5  # kg
            self.voltage = 12.0  # V
            self.current_draw_max = 12.5  # A
        elif pump_type == PumpType.CUSTOM:
            # Use custom parameters if provided
            if custom_params:
                self.speed_ratio = custom_params.get('speed_ratio', 0.0)
                self.power_consumption_max = custom_params.get('power_consumption_max', 250.0)
                self.weight = custom_params.get('weight', 0.6)
                self.voltage = custom_params.get('voltage', 12.0)
                self.current_draw_max = custom_params.get('current_draw_max', 10.0)
            else:
                # Default values for custom type
                self.speed_ratio = 0.5
                self.power_consumption_max = 250.0
                self.weight = 0.6
                self.voltage = 12.0
                self.current_draw_max = 10.0
        
        # Current operating state
        self.current_speed = 0.0  # RPM
        self.current_flow_rate = 0.0  # L/min
        self.current_pressure = 0.0  # bar
        self.current_power = 0.0  # W
        
        logger.info(f"Water pump initialized: {pump_type.name}, max flow: {max_flow_rate} L/min")
    
    def update_pump_speed(self, engine_rpm: Optional[float] = None, 
                         control_signal: Optional[float] = None):
        """
        Update pump speed based on engine RPM or control signal.
        
        Args:
            engine_rpm: Engine speed in RPM (for mechanical pumps)
            control_signal: Control signal (0-1) for electric pumps
        """
        if self.pump_type == PumpType.MECHANICAL and engine_rpm is not None:
            # Mechanical pump tied to engine speed
            self.current_speed = engine_rpm * self.speed_ratio
        elif self.pump_type == PumpType.ELECTRIC and control_signal is not None:
            # Electric pump controlled by PWM signal
            control_signal = max(0.0, min(1.0, control_signal))  # Clamp to 0-1
            self.current_speed = self.nominal_speed * control_signal
        else:
            # No change if appropriate input not provided
            logger.warning(f"Inappropriate pump control input for {self.pump_type.name} pump")
    
    def calculate_flow_rate(self, system_pressure: float) -> float:
        """
        Calculate flow rate based on current pump speed and system pressure.
        
        Args:
            system_pressure: System backpressure in bar
            
        Returns:
            Flow rate in L/min
        """
        # Calculate max theoretical flow and pressure at current speed
        speed_factor = self.current_speed / self.nominal_speed
        theoretical_max_flow = self.max_flow_rate * speed_factor
        theoretical_max_pressure = self.max_pressure * speed_factor**2
        
        # Handle case where pump can't overcome system pressure
        if system_pressure >= theoretical_max_pressure:
            self.current_flow_rate = 0.0
            self.current_pressure = system_pressure
            return 0.0
        
        # Calculate flow rate using affinity laws and pump curve
        # This is a simplified quadratic model for the pump curve
        # Flow decreases linearly with pressure
        flow_factor = 1.0 - (system_pressure / theoretical_max_pressure)
        self.current_flow_rate = theoretical_max_flow * flow_factor
        self.current_pressure = system_pressure
        
        return self.current_flow_rate
    
    def calculate_power_consumption(self) -> float:
        """
        Calculate power consumption of the pump.
        
        Returns:
            Power consumption in watts
        """
        if self.current_speed == 0.0:
            self.current_power = 0.0
            return 0.0
        
        # Calculate hydraulic power
        # P_hydraulic = flow_rate * pressure * conversion_factor
        hydraulic_power = (self.current_flow_rate / 60000) * (self.current_pressure * 100000)
        
        # Apply mechanical efficiency
        self.current_power = hydraulic_power / self.mechanical_efficiency
        
        # Add baseline power consumption (mechanical losses, etc.)
        baseline_power = self.power_consumption_max * 0.1 * (self.current_speed / self.nominal_speed)
        self.current_power += baseline_power
        
        return self.current_power
    
    def get_pump_state(self) -> Dict:
        """
        Get current state of the pump.
        
        Returns:
            Dictionary with current pump state
        """
        return {
            'type': self.pump_type.name,
            'current_speed': self.current_speed,
            'current_flow_rate': self.current_flow_rate,
            'current_pressure': self.current_pressure,
            'current_power': self.current_power
        }
    
    def get_pump_specs(self) -> Dict:
        """
        Get specifications of the pump.
        
        Returns:
            Dictionary with pump specifications
        """
        specs = {
            'type': self.pump_type.name,
            'max_flow_rate': self.max_flow_rate,
            'max_pressure': self.max_pressure,
            'nominal_speed': self.nominal_speed,
            'mechanical_efficiency': self.mechanical_efficiency,
            'weight': self.weight
        }
        
        # Add type-specific parameters
        if self.pump_type == PumpType.MECHANICAL:
            specs['speed_ratio'] = self.speed_ratio
        elif self.pump_type == PumpType.ELECTRIC:
            specs['voltage'] = self.voltage
            specs['current_draw_max'] = self.current_draw_max
            
        return specs


class CoolingFan:
    """
    Cooling fan model for Formula Student car.
    
    This class models the behavior of electric cooling fans, including
    airflow generation, power consumption, and control characteristics.
    """
    
    def __init__(self, 
                 fan_type: FanType = FanType.SINGLE_SPEED,
                 max_airflow: float = 0.3,         # m³/s
                 diameter: float = 0.25,           # m
                 max_power: float = 90.0,          # W
                 voltage: float = 12.0,            # V
                 custom_params: Optional[Dict] = None):
        """
        Initialize cooling fan model.
        
        Args:
            fan_type: Type of cooling fan
            max_airflow: Maximum airflow in m³/s
            diameter: Fan diameter in m
            max_power: Maximum power consumption in W
            voltage: Operating voltage in V
            custom_params: Optional dictionary with custom parameters
        """
        self.fan_type = fan_type
        self.max_airflow = max_airflow
        self.diameter = diameter
        self.max_power = max_power
        self.voltage = voltage
        
        # Calculate derived properties
        self.area = np.pi * (diameter / 2)**2
        
        # Set fan-specific parameters based on type
        if fan_type == FanType.SINGLE_SPEED:
            self.control_type = "on_off"
            self.num_fans = 1
            self.weight = 0.5  # kg
        elif fan_type == FanType.VARIABLE_SPEED:
            self.control_type = "pwm"
            self.num_fans = 1
            self.weight = 0.55  # kg
        elif fan_type == FanType.DUAL_FAN:
            self.control_type = "on_off"
            self.num_fans = 2
            self.weight = 1.0  # kg
            self.max_airflow *= 1.8  # Not quite double due to interference
            self.max_power *= 2.0
        elif fan_type == FanType.CUSTOM:
            # Use custom parameters if provided
            if custom_params:
                self.control_type = custom_params.get('control_type', "pwm")
                self.num_fans = custom_params.get('num_fans', 1)
                self.weight = custom_params.get('weight', 0.6)
            else:
                # Default values for custom type
                self.control_type = "pwm"
                self.num_fans = 1
                self.weight = 0.6
        
        # Current operating state
        self.current_duty_cycle = 0.0  # 0-1 for PWM control, 0 or 1 for on/off
        self.current_airflow = 0.0  # m³/s
        self.current_power = 0.0  # W
        self.is_active = False
        
        logger.info(f"Cooling fan initialized: {fan_type.name}, max airflow: {max_airflow} m³/s")
    
    def update_control(self, control_signal: float):
        """
        Update fan control state based on control signal.
        
        Args:
            control_signal: Control signal (0-1)
        """
        control_signal = max(0.0, min(1.0, control_signal))  # Clamp to 0-1
        
        if self.control_type == "on_off":
            # On/off control with hysteresis
            # Turn on if signal > 0.6, turn off if signal < 0.4
            if control_signal > 0.6:
                self.current_duty_cycle = 1.0
                self.is_active = True
            elif control_signal < 0.4:
                self.current_duty_cycle = 0.0
                self.is_active = False
                
        elif self.control_type == "pwm":
            # PWM control (proportional)
            self.current_duty_cycle = control_signal
            self.is_active = control_signal > 0.05  # Minimum threshold for activation
        
        # Update airflow and power based on duty cycle
        self._update_outputs()
    
    def _update_outputs(self):
        """Update airflow and power consumption based on current duty cycle."""
        # Calculate airflow (approximately cubic relationship with duty cycle)
        if self.is_active:
            # Use cubic function for variable speed fans, step function for on/off
            if self.control_type == "pwm":
                self.current_airflow = self.max_airflow * self.current_duty_cycle**3
            else:  # on_off
                self.current_airflow = self.max_airflow * self.current_duty_cycle
                
            # Calculate power consumption (approximately cubic relationship)
            self.current_power = self.max_power * self.current_duty_cycle**3
        else:
            self.current_airflow = 0.0
            self.current_power = 0.0
    
    def calculate_power_consumption(self) -> float:
        """
        Calculate current power consumption.
        
        Returns:
            Power consumption in watts
        """
        return self.current_power
    
    def calculate_back_pressure_effect(self, system_back_pressure: float) -> float:
        """
        Calculate reduction in airflow due to system back pressure.
        
        Args:
            system_back_pressure: System back pressure in pascals
            
        Returns:
            Airflow reduction factor (0-1)
        """
        # This is a simplified model of fan performance vs system pressure
        # Actual fans would have a more complex pressure-flow curve
        
        # Typical pressure-flow relationship
        # As back pressure increases, airflow decreases
        max_pressure_capability = 250.0  # Pa, typical for automotive fans
        
        if system_back_pressure >= max_pressure_capability:
            return 0.0  # No flow if back pressure too high
        
        # Linear model for simplicity
        reduction_factor = 1.0 - (system_back_pressure / max_pressure_capability)
        
        return reduction_factor
    
    def get_fan_state(self) -> Dict:
        """
        Get current state of the fan.
        
        Returns:
            Dictionary with current fan state
        """
        return {
            'type': self.fan_type.name,
            'is_active': self.is_active,
            'current_duty_cycle': self.current_duty_cycle,
            'current_airflow': self.current_airflow,
            'current_power': self.current_power
        }
    
    def get_fan_specs(self) -> Dict:
        """
        Get specifications of the fan.
        
        Returns:
            Dictionary with fan specifications
        """
        return {
            'type': self.fan_type.name,
            'max_airflow': self.max_airflow,
            'diameter': self.diameter,
            'max_power': self.max_power,
            'voltage': self.voltage,
            'area': self.area,
            'control_type': self.control_type,
            'num_fans': self.num_fans,
            'weight': self.weight
        }


class Thermostat:
    """
    Thermostat model for Formula Student cooling system.
    
    This class models the behavior of the engine thermostat, which regulates
    coolant flow through the radiator to maintain optimal engine temperature.
    """
    
    def __init__(self, 
                 opening_temp: float = 82.0,   # °C
                 full_open_temp: float = 92.0, # °C
                 bypass_flow_max: float = 15.0, # L/min
                 custom_params: Optional[Dict] = None):
        """
        Initialize thermostat model.
        
        Args:
            opening_temp: Temperature at which thermostat begins to open (°C)
            full_open_temp: Temperature at which thermostat is fully open (°C)
            bypass_flow_max: Maximum flow through bypass when thermostat closed (L/min)
            custom_params: Optional dictionary with custom parameters
        """
        self.opening_temp = opening_temp
        self.full_open_temp = full_open_temp
        self.bypass_flow_max = bypass_flow_max
        
        # Apply custom parameters if provided
        if custom_params:
            for key, value in custom_params.items():
                if hasattr(self, key):
                    setattr(self, key, value)
        
        # Current state
        self.current_opening = 0.0  # 0-1, fraction of full open
        self.current_temperature = 25.0  # °C
        
        logger.info(f"Thermostat initialized: opening range {opening_temp}°C - {full_open_temp}°C")
    
    def update_temperature(self, coolant_temp: float):
        """
        Update thermostat opening based on coolant temperature.
        
        Args:
            coolant_temp: Current coolant temperature in °C
        """
        self.current_temperature = coolant_temp
        
        # Calculate opening fraction based on temperature
        if coolant_temp <= self.opening_temp:
            self.current_opening = 0.0
        elif coolant_temp >= self.full_open_temp:
            self.current_opening = 1.0
        else:
            # Linear interpolation for partial opening
            self.current_opening = (coolant_temp - self.opening_temp) / (self.full_open_temp - self.opening_temp)
    
    def calculate_flow_distribution(self, total_flow: float) -> Tuple[float, float]:
        """
        Calculate flow distribution between radiator and bypass.
        
        Args:
            total_flow: Total coolant flow rate in L/min
            
        Returns:
            Tuple of (radiator_flow, bypass_flow) in L/min
        """
        # When thermostat is closed, flow goes through bypass
        # When thermostat is open, flow goes through radiator
        # During transition, flow is distributed based on opening fraction
        
        if self.current_opening <= 0.0:
            # Fully closed - all flow through bypass up to its maximum capacity
            bypass_flow = min(total_flow, self.bypass_flow_max)
            radiator_flow = 0.0
        elif self.current_opening >= 1.0:
            # Fully open - all flow through radiator
            radiator_flow = total_flow
            bypass_flow = 0.0
        else:
            # Partially open - flow distributed based on opening fraction
            # and the relative resistance of each path
            
            # This is a simplified model - in reality the flow distribution
            # would depend on the detailed hydraulic characteristics
            
            # Calculate base flow balance
            radiator_fraction = self.current_opening**2  # Non-linear relationship
            radiator_flow = total_flow * radiator_fraction
            bypass_flow = total_flow - radiator_flow
            
            # Limit bypass flow to its maximum capacity
            if bypass_flow > self.bypass_flow_max:
                bypass_flow = self.bypass_flow_max
                radiator_flow = total_flow - bypass_flow
        
        return radiator_flow, bypass_flow
    
    def get_thermostat_state(self) -> Dict:
        """
        Get current state of the thermostat.
        
        Returns:
            Dictionary with current thermostat state
        """
        return {
            'current_temperature': self.current_temperature,
            'current_opening': self.current_opening,
            'opening_temp': self.opening_temp,
            'full_open_temp': self.full_open_temp
        }


class CoolingSystem:
    """
    Complete cooling system for Formula Student car.
    
    This class integrates all cooling system components (radiator, water pump,
    cooling fan, thermostat) into a complete system model for simulating
    thermal management of the Formula Student car.
    """
    
    def __init__(self, 
                 radiator: Optional[Radiator] = None,
                 water_pump: Optional[WaterPump] = None,
                 cooling_fan: Optional[CoolingFan] = None,
                 thermostat: Optional[Thermostat] = None):
        """
        Initialize the complete cooling system.
        
        Args:
            radiator: Radiator component
            water_pump: Water pump component
            cooling_fan: Cooling fan component
            thermostat: Thermostat component
        """
        # Initialize components with defaults if not provided
        self.radiator = radiator or Radiator()
        self.water_pump = water_pump or WaterPump()
        self.cooling_fan = cooling_fan or CoolingFan()
        self.thermostat = thermostat or Thermostat()
        
        # System parameters
        self.coolant_volume = 3.0  # L, total system volume
        self.coolant_density = 1050.0  # kg/m³
        self.coolant_specific_heat = 3900.0  # J/(kg·K)
        self.system_pressure_cap = 1.2  # bar
        
        # Current state
        self.coolant_temp = 25.0  # °C
        self.engine_temp = 25.0  # °C
        self.ambient_temp = 25.0  # °C
        self.fan_control_signal = 0.0  # 0-1
        self.pump_control_signal = 0.0  # 0-1 (for electric pump)
        self.vehicle_speed = 0.0  # m/s
        self.engine_rpm = 0.0  # RPM
        self.engine_load = 0.0  # 0-1
        self.radiator_heat_rejection = 0.0  # W
        self.engine_heat_input = 0.0  # W
        
        # System dynamics
        self.last_update_time = None
        
        logger.info("Complete cooling system initialized")
    
    def update_ambient_conditions(self, ambient_temp: float, vehicle_speed: float):
        """
        Update ambient conditions affecting the cooling system.
        
        Args:
            ambient_temp: Ambient temperature in °C
            vehicle_speed: Vehicle speed in m/s
        """
        self.ambient_temp = ambient_temp
        self.vehicle_speed = vehicle_speed
    
    def update_engine_state(self, engine_temp: float, engine_rpm: float, 
                          engine_load: float, engine_heat_input: float):
        """
        Update engine state parameters affecting the cooling system.
        
        Args:
            engine_temp: Engine temperature in °C
            engine_rpm: Engine speed in RPM
            engine_load: Engine load factor (0-1)
            engine_heat_input: Heat input to cooling system from engine in W
        """
        self.engine_temp = engine_temp
        self.engine_rpm = engine_rpm
        self.engine_load = engine_load
        self.engine_heat_input = engine_heat_input
    
    def update_control_signals(self, fan_control: float, pump_control: Optional[float] = None):
        """
        Update control signals for cooling system components.
        
        Args:
            fan_control: Fan control signal (0-1)
            pump_control: Pump control signal (0-1), only used for electric pumps
        """
        self.fan_control_signal = max(0.0, min(1.0, fan_control))
        self.cooling_fan.update_control(self.fan_control_signal)
        
        # Update pump control if electric pump
        if pump_control is not None and self.water_pump.pump_type == PumpType.ELECTRIC:
            self.pump_control_signal = max(0.0, min(1.0, pump_control))
            self.water_pump.update_pump_speed(control_signal=self.pump_control_signal)
    
    def create_automatic_control(self, target_temp: float = 90.0, hysteresis: float = 5.0):
        """
        Configure automatic control based on temperature.
        
        Args:
            target_temp: Target coolant temperature in °C
            hysteresis: Temperature hysteresis for fan control in °C
        """
        # Fan control based on coolant temperature
        fan_control = 0.0
        
        if self.coolant_temp > target_temp + hysteresis:
            fan_control = 1.0  # Full fan if temp is very high
        elif self.coolant_temp > target_temp:
            # Linear ramp from 0 to 1 over the hysteresis range
            fan_control = (self.coolant_temp - target_temp) / hysteresis
        
        # Apply control signals
        self.update_control_signals(fan_control, None)
    
    def update_system_state(self, dt: float):
        """
        Update the complete cooling system state for a time step.
        
        Args:
            dt: Time step in seconds
        """
        # Update thermostat opening based on coolant temperature
        self.thermostat.update_temperature(self.coolant_temp)
        
        # Update water pump speed based on engine RPM or control signal
        if self.water_pump.pump_type == PumpType.MECHANICAL:
            self.water_pump.update_pump_speed(engine_rpm=self.engine_rpm)
        # For electric pump, already updated in update_control_signals
        
        # Calculate system pressure based on pump, thermostat, and radiator
        # This is a simplified system pressure calculation
        system_pressure = 0.5  # bar, base pressure
        
        # Calculate pump flow rate against system pressure
        total_flow_rate = self.water_pump.calculate_flow_rate(system_pressure)
        
        # Calculate flow distribution through thermostat
        radiator_flow, bypass_flow = self.thermostat.calculate_flow_distribution(total_flow_rate)
        
        # Calculate airflow through radiator
        # Base airflow from vehicle speed (ram air effect)
        ram_air_factor = 0.5  # Efficiency factor for ram air
        radiator_area = self.radiator.core_area
        speed_airflow = self.vehicle_speed * radiator_area * ram_air_factor
        
        # Add fan airflow
        fan_airflow = self.cooling_fan.current_airflow
        
        # Total airflow through radiator
        total_airflow = speed_airflow + fan_airflow
        
        # Calculate heat rejection by radiator
        self.radiator_heat_rejection = self.radiator.calculate_heat_rejection(
            self.coolant_temp, self.ambient_temp, radiator_flow, total_airflow
        )
        
        # Calculate engine heat input (from external model)
        net_heat = self.engine_heat_input - self.radiator_heat_rejection
        
        # Calculate temperature change (simplified thermal model)
        coolant_mass = self.coolant_volume * self.coolant_density / 1000  # kg
        coolant_heat_capacity = coolant_mass * self.coolant_specific_heat  # J/K
        
        # Temperature change (Q = m * c * ΔT)
        if coolant_heat_capacity > 0:
            delta_temp = net_heat * dt / coolant_heat_capacity
        else:
            delta_temp = 0
        
        # Update coolant temperature
        self.coolant_temp += delta_temp
        
        # Calculate power consumption
        self.water_pump.calculate_power_consumption()
        self.cooling_fan.calculate_power_consumption()
    
    def simulate_step(self, ambient_temp: float, vehicle_speed: float,
                    engine_temp: float, engine_rpm: float, engine_load: float,
                    engine_heat_input: float, dt: float) -> Dict:
        """
        Perform a single simulation step with the provided conditions.
        
        Args:
            ambient_temp: Ambient temperature in °C
            vehicle_speed: Vehicle speed in m/s
            engine_temp: Engine temperature in °C
            engine_rpm: Engine speed in RPM
            engine_load: Engine load factor (0-1)
            engine_heat_input: Heat input to cooling system from engine in W
            dt: Time step in seconds
            
        Returns:
            Dictionary with updated system state
        """
        # Update external conditions
        self.update_ambient_conditions(ambient_temp, vehicle_speed)
        self.update_engine_state(engine_temp, engine_rpm, engine_load, engine_heat_input)
        
        # Create and apply automatic control
        self.create_automatic_control()
        
        # Update system state
        self.update_system_state(dt)
        
        # Return current state
        return self.get_system_state()
    
    def get_system_state(self) -> Dict:
        """
        Get current state of the complete cooling system.
        
        Returns:
            Dictionary with current cooling system state
        """
        return {
            'coolant_temp': self.coolant_temp,
            'engine_temp': self.engine_temp,
            'ambient_temp': self.ambient_temp,
            'vehicle_speed': self.vehicle_speed,
            'engine_rpm': self.engine_rpm,
            'engine_load': self.engine_load,
            'engine_heat_input': self.engine_heat_input,
            'radiator_heat_rejection': self.radiator_heat_rejection,
            'net_heat': self.engine_heat_input - self.radiator_heat_rejection,
            'water_pump': self.water_pump.get_pump_state(),
            'cooling_fan': self.cooling_fan.get_fan_state(),
            'thermostat': self.thermostat.get_thermostat_state(),
            'fan_control_signal': self.fan_control_signal,
            'pump_control_signal': self.pump_control_signal
        }
    
    def get_system_specs(self) -> Dict:
        """
        Get specifications of the complete cooling system.
        
        Returns:
            Dictionary with cooling system specifications
        """
        return {
            'radiator': self.radiator.get_radiator_specs(),
            'water_pump': self.water_pump.get_pump_specs(),
            'cooling_fan': self.cooling_fan.get_fan_specs(),
            'thermostat': {
                'opening_temp': self.thermostat.opening_temp,
                'full_open_temp': self.thermostat.full_open_temp,
                'bypass_flow_max': self.thermostat.bypass_flow_max
            },
            'system': {
                'coolant_volume': self.coolant_volume,
                'coolant_density': self.coolant_density,
                'coolant_specific_heat': self.coolant_specific_heat,
                'system_pressure_cap': self.system_pressure_cap
            }
        }
    
    def calculate_system_performance(self, ambient_temp_range: List[float], 
                                  engine_heat_range: List[float]) -> Dict:
        """
        Calculate cooling system performance across a range of conditions.
        
        Args:
            ambient_temp_range: List of ambient temperatures to test (°C)
            engine_heat_range: List of engine heat inputs to test (W)
            
        Returns:
            Dictionary with performance results
        """
        # Initialize results arrays
        n_ambient = len(ambient_temp_range)
        n_heat = len(engine_heat_range)
        
        coolant_temps = np.zeros((n_ambient, n_heat))
        rejection_rates = np.zeros((n_ambient, n_heat))
        fan_duties = np.zeros((n_ambient, n_heat))
        pump_flows = np.zeros((n_ambient, n_heat))
        
        # Test conditions
        test_rpm = 6000
        test_load = 0.7
        test_vehicle_speed = 10.0  # m/s
        
        # Run simulations
        for i, ambient in enumerate(ambient_temp_range):
            for j, heat in enumerate(engine_heat_range):
                # Reset system state
                self.coolant_temp = ambient + 10.0  # Start slightly above ambient
                self.engine_temp = ambient + 15.0
                
                # Run simulation until steady state
                max_iterations = 100
                for iteration in range(max_iterations):
                    # Update state for 1 second
                    state = self.simulate_step(
                        ambient, test_vehicle_speed,
                        self.engine_temp, test_rpm, test_load,
                        heat, 1.0
                    )
                    
                    # Check for steady state (temperature change < 0.1°C)
                    if iteration > 0 and abs(state['coolant_temp'] - coolant_temps[i, j]) < 0.1:
                        break
                    
                    coolant_temps[i, j] = state['coolant_temp']
                    rejection_rates[i, j] = state['radiator_heat_rejection']
                    fan_duties[i, j] = state['fan_control_signal']
                    pump_flows[i, j] = state['water_pump']['current_flow_rate']
        
        # Calculate performance metrics
        cooling_capacities = np.zeros(n_ambient)
        for i in range(n_ambient):
            # Find maximum heat input that keeps coolant below 100°C
            max_heat_idx = np.argmax(coolant_temps[i, :] >= 100.0)
            if max_heat_idx > 0:
                cooling_capacities[i] = engine_heat_range[max_heat_idx - 1]
            elif max_heat_idx == 0:
                cooling_capacities[i] = 0.0  # Can't handle even minimum heat
            else:
                cooling_capacities[i] = engine_heat_range[-1]  # Can handle all tested heat levels
        
        return {
            'ambient_temps': ambient_temp_range,
            'engine_heats': engine_heat_range,
            'coolant_temps': coolant_temps,
            'rejection_rates': rejection_rates,
            'fan_duties': fan_duties,
            'pump_flows': pump_flows,
            'cooling_capacities': cooling_capacities
        }
    
    def plot_performance_map(self, performance_data: Dict, save_path: Optional[str] = None):
        """
        Plot cooling system performance map.
        
        Args:
            performance_data: Performance data from calculate_system_performance
            save_path: Optional path to save the plot
        """
        ambient_temps = performance_data['ambient_temps']
        engine_heats = performance_data['engine_heats']
        coolant_temps = performance_data['coolant_temps']
        
        # Create meshgrid for contour plot
        X, Y = np.meshgrid(ambient_temps, engine_heats)
        
        plt.figure(figsize=(10, 8))
        
        # Create contour plot of coolant temperatures
        contour = plt.contourf(X, Y, coolant_temps.T, 20, cmap='hot')
        plt.colorbar(contour, label='Coolant Temperature (°C)')
        
        # Add contour line for critical temperature (100°C)
        critical_contour = plt.contour(X, Y, coolant_temps.T, [100], colors='white', linestyles='dashed', linewidths=2)
        plt.clabel(critical_contour, inline=True, fontsize=10, fmt='%.0f°C')
        
        # Add labels and title
        plt.xlabel('Ambient Temperature (°C)')
        plt.ylabel('Engine Heat Input (W)')
        plt.title('Cooling System Performance Map')
        
        # Add grid for better readability
        plt.grid(True, linestyle='--', alpha=0.3)
        
        # Save plot if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        plt.tight_layout()
        plt.show()
    
    def plot_cooling_capacity(self, performance_data: Dict, save_path: Optional[str] = None):
        """
        Plot cooling capacity vs ambient temperature.
        
        Args:
            performance_data: Performance data from calculate_system_performance
            save_path: Optional path to save the plot
        """
        ambient_temps = performance_data['ambient_temps']
        cooling_capacities = performance_data['cooling_capacities']
        
        plt.figure(figsize=(10, 6))
        
        # Plot cooling capacity
        plt.plot(ambient_temps, cooling_capacities / 1000, 'b-', linewidth=2)
        
        # Add labels and title
        plt.xlabel('Ambient Temperature (°C)')
        plt.ylabel('Cooling Capacity (kW)')
        plt.title('Maximum Cooling Capacity vs Ambient Temperature')
        
        # Add grid for better readability
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Save plot if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        plt.tight_layout()
        plt.show()


def create_cbr600f4i_cooling_system() -> CoolingSystem:
    """
    Create a cooling system configured for the Honda CBR600F4i engine.
    
    Returns:
        CoolingSystem configured for CBR600F4i
    """
    # Create radiator
    radiator = Radiator(
        radiator_type=RadiatorType.SINGLE_CORE_ALUMINUM,
        core_area=0.16,       # m²
        core_thickness=0.04,  # m
        fin_density=15,       # fins/inch
        tube_rows=2
    )
    
    # Create water pump (mechanical, engine-driven)
    water_pump = WaterPump(
        pump_type=PumpType.MECHANICAL,
        max_flow_rate=60.0,   # L/min
        max_pressure=1.6,     # bar
        nominal_speed=5000.0, # RPM
        mechanical_efficiency=0.7
    )
    
    # Create cooling fan
    cooling_fan = CoolingFan(
        fan_type=FanType.VARIABLE_SPEED,
        max_airflow=0.25,     # m³/s
        diameter=0.28,        # m
        max_power=80.0,       # W
        voltage=12.0          # V
    )
    
    # Create thermostat
    thermostat = Thermostat(
        opening_temp=82.0,    # °C
        full_open_temp=92.0,  # °C
        bypass_flow_max=10.0  # L/min
    )
    
    # Create complete cooling system
    cooling_system = CoolingSystem(
        radiator=radiator,
        water_pump=water_pump,
        cooling_fan=cooling_fan,
        thermostat=thermostat
    )
    
    # Set additional system parameters
    cooling_system.coolant_volume = 2.4  # L, typical for CBR600F4i
    cooling_system.system_pressure_cap = 1.1  # bar
    
    return cooling_system


def create_formula_student_cooling_system() -> CoolingSystem:
    """
    Create an optimized cooling system for Formula Student application.
    
    Returns:
        CoolingSystem optimized for Formula Student
    """
    # Create high-performance radiator
    radiator = Radiator(
        radiator_type=RadiatorType.DOUBLE_CORE_ALUMINUM,
        core_area=0.18,       # m², increased for FS application
        core_thickness=0.045, # m
        fin_density=16,       # fins/inch
        tube_rows=2
    )
    
    # Create water pump (electric for better control)
    water_pump = WaterPump(
        pump_type=PumpType.ELECTRIC,
        max_flow_rate=75.0,   # L/min
        max_pressure=1.8,     # bar
        nominal_speed=6000.0, # RPM
        mechanical_efficiency=0.75
    )
    
    # Create dual cooling fans
    cooling_fan = CoolingFan(
        fan_type=FanType.DUAL_FAN,
        max_airflow=0.4,      # m³/s
        diameter=0.22,        # m, smaller fans for better packaging
        max_power=160.0,      # W
        voltage=12.0          # V
    )
    
    # Create optimal thermostat
    thermostat = Thermostat(
        opening_temp=80.0,    # °C, opening earlier for better control
        full_open_temp=88.0,  # °C
        bypass_flow_max=12.0  # L/min
    )
    
    # Create complete cooling system
    cooling_system = CoolingSystem(
        radiator=radiator,
        water_pump=water_pump,
        cooling_fan=cooling_fan,
        thermostat=thermostat
    )
    
    # Set optimized system parameters
    cooling_system.coolant_volume = 2.2  # L, reduced for weight savings
    cooling_system.system_pressure_cap = 1.3  # bar, increased for better boiling point
    
    return cooling_system


# Example usage
if __name__ == "__main__":
    # Create a Formula Student cooling system
    cooling_system = create_formula_student_cooling_system()
    
    print("Formula Student Cooling System Specifications:")
    specs = cooling_system.get_system_specs()
    
    # Print radiator specs
    print("\nRadiator:")
    for key, value in specs['radiator'].items():
        print(f"  {key}: {value}")
    
    # Print water pump specs
    print("\nWater Pump:")
    for key, value in specs['water_pump'].items():
        print(f"  {key}: {value}")
    
    # Print cooling fan specs
    print("\nCooling Fan:")
    for key, value in specs['cooling_fan'].items():
        print(f"  {key}: {value}")
    
    # Print thermostat specs
    print("\nThermostat:")
    for key, value in specs['thermostat'].items():
        print(f"  {key}: {value}")
    
    # Run a simple simulation
    print("\nRunning simulation...")
    
    # Initial conditions
    ambient_temp = 30.0  # °C
    vehicle_speed = 15.0  # m/s (~54 km/h)
    engine_rpm = 8000.0  # RPM
    engine_load = 0.7    # 70% load
    engine_heat = 30000.0  # W (30 kW heat into coolant)
    
    # Simulate for 60 seconds with 1 second steps
    for i in range(60):
        state = cooling_system.simulate_step(
            ambient_temp, vehicle_speed,
            cooling_system.coolant_temp, engine_rpm, engine_load,
            engine_heat, 1.0
        )
        
        # Print every 10 seconds
        if i % 10 == 0:
            print(f"\nTime: {i}s")
            print(f"  Coolant Temp: {state['coolant_temp']:.1f}°C")
            print(f"  Heat Rejection: {state['radiator_heat_rejection']/1000:.1f} kW")
            print(f"  Fan Duty: {state['fan_control_signal']*100:.0f}%")
            print(f"  Pump Flow: {state['water_pump']['current_flow_rate']:.1f} L/min")
    
    # Calculate and plot system performance
    print("\nCalculating performance map...")
    
    # Define test ranges
    ambient_temps = np.linspace(20, 40, 5)  # 20-40°C
    engine_heats = np.linspace(10000, 50000, 5)  # 10-50 kW
    
    # Calculate performance
    performance = cooling_system.calculate_system_performance(ambient_temps, engine_heats)
    
    # Print cooling capacities
    print("\nCooling Capacities:")
    for temp, capacity in zip(ambient_temps, performance['cooling_capacities']):
        print(f"  At {temp:.1f}°C: {capacity/1000:.1f} kW")
    
    print("\nSimulation complete!")