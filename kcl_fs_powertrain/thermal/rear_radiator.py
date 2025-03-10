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

# Import base cooling system module
from .cooling_system import (
    Radiator, RadiatorType, CoolingFan, FanType, Thermostat, CoolingSystem
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("Rear_Radiator")


class MountingPosition(Enum):
    """Enumeration of possible rear radiator mounting positions."""
    ABOVE_DIFFUSER = auto()      # Mounted above the diffuser
    BEHIND_DRIVER = auto()       # Mounted directly behind the driver
    SIDE_POD_REAR = auto()       # Mounted in rear of side pods
    ANGLED_UPWARD = auto()       # Angled upward for better natural convection
    CUSTOM = auto()              # Custom mounting position


class DuctType(Enum):
    """Enumeration of different duct types for rear radiators."""
    NACA_INLET = auto()          # NACA duct inlet
    SIDE_SCOOP = auto()          # Side scoop inlet
    TOP_INLET = auto()           # Top inlet duct
    DIFFUSER_INTEGRATED = auto() # Integrated with diffuser
    CUSTOM = auto()              # Custom duct configuration


class RearRadiator:
    """
    Specialized radiator class for rear-mounted cooling solutions in Formula Student.
    
    This class extends the base Radiator class with specific properties and behaviors
    relevant to rear-mounted radiator configurations.
    """
    
    def __init__(self, 
                 radiator: Radiator,
                 mounting_position: MountingPosition = MountingPosition.ABOVE_DIFFUSER,
                 angle_degrees: float = 15.0,        # Installation angle in degrees
                 distance_from_diffuser: float = 0.1, # Meters
                 custom_params: Optional[Dict] = None):
        """
        Initialize a rear-mounted radiator configuration.
        
        Args:
            radiator: Base Radiator object
            mounting_position: Mounting position enum
            angle_degrees: Installation angle in degrees from vertical
            distance_from_diffuser: Distance from diffuser in meters
            custom_params: Optional dictionary with custom parameters
        """
        self.base_radiator = radiator
        self.mounting_position = mounting_position
        self.angle_degrees = angle_degrees
        self.distance_from_diffuser = distance_from_diffuser
        
        # Calculate derived properties
        self.angle_radians = np.radians(angle_degrees)
        self.effective_area = radiator.core_area * np.cos(self.angle_radians)
        
        # Default airflow properties based on mounting position
        if mounting_position == MountingPosition.ABOVE_DIFFUSER:
            self.natural_airflow_factor = 0.6  # Good airflow from diffuser
            self.stagnation_factor = 0.3       # Moderate stagnation risk
        elif mounting_position == MountingPosition.BEHIND_DRIVER:
            self.natural_airflow_factor = 0.4  # Reduced airflow behind driver
            self.stagnation_factor = 0.5       # Higher stagnation risk
        elif mounting_position == MountingPosition.SIDE_POD_REAR:
            self.natural_airflow_factor = 0.7  # Good side airflow
            self.stagnation_factor = 0.2       # Lower stagnation risk
        elif mounting_position == MountingPosition.ANGLED_UPWARD:
            self.natural_airflow_factor = 0.5  # Moderate airflow
            self.stagnation_factor = 0.2       # Lower stagnation from convection
        elif mounting_position == MountingPosition.CUSTOM:
            # Use custom parameters if provided
            if custom_params:
                self.natural_airflow_factor = custom_params.get('natural_airflow_factor', 0.5)
                self.stagnation_factor = custom_params.get('stagnation_factor', 0.3)
            else:
                self.natural_airflow_factor = 0.5
                self.stagnation_factor = 0.3
        
        # Low-speed airflow characteristics
        self.convection_factor = 0.2 + (angle_degrees / 90.0) * 0.3  # Natural convection factor
        
        # High-speed performance characteristics
        self.drag_impact = self._calculate_drag_impact()
        
        logger.info(f"Rear radiator initialized: {mounting_position.name} position, {angle_degrees}° angle")
    
    def _calculate_drag_impact(self) -> float:
        """
        Calculate the drag impact of the rear radiator installation.
        
        Returns:
            Drag coefficient contribution
        """
        # Base drag contribution
        base_drag = 0.01
        
        # Modify based on position and angle
        if self.mounting_position == MountingPosition.ABOVE_DIFFUSER:
            position_factor = 1.2  # Higher drag impact above diffuser
        elif self.mounting_position == MountingPosition.BEHIND_DRIVER:
            position_factor = 0.8  # Lower drag impact (already in wake)
        elif self.mounting_position == MountingPosition.SIDE_POD_REAR:
            position_factor = 1.1  # Moderate drag impact
        elif self.mounting_position == MountingPosition.ANGLED_UPWARD:
            position_factor = 1.3  # Higher drag from angled surface
        else:
            position_factor = 1.0
        
        # Angle effect - more perpendicular to airflow = more drag
        angle_factor = 1.0 + (90.0 - self.angle_degrees) / 90.0
        
        # Size effect
        size_factor = self.base_radiator.core_area / 0.15  # Normalized to 0.15m²
        
        return base_drag * position_factor * angle_factor * size_factor
    
    def calculate_effective_airflow(self, vehicle_speed: float, fan_airflow: float) -> float:
        """
        Calculate effective airflow through the rear-mounted radiator.
        
        Args:
            vehicle_speed: Vehicle speed in m/s
            fan_airflow: Additional airflow from fans in m³/s
            
        Returns:
            Effective airflow through radiator in m³/s
        """
        # Base ram air from vehicle speed
        if vehicle_speed < 5.0:
            # At very low speeds, ram air is minimal, rely more on natural convection
            ram_air = vehicle_speed * self.base_radiator.core_area * self.natural_airflow_factor * 0.5
            # Add natural convection effect which increases with radiator angle
            convection_air = self.convection_factor * 0.05  # Base convection flow
        else:
            # At higher speeds, ram air is more effective
            speed_factor = min(1.0, vehicle_speed / 20.0)  # Normalize to 20 m/s
            ram_air = vehicle_speed * self.base_radiator.core_area * self.natural_airflow_factor * speed_factor
            convection_air = 0.0  # Negligible compared to ram air
        
        # Add fan contribution
        # Fan effectiveness depends on the installation
        if self.mounting_position == MountingPosition.BEHIND_DRIVER:
            fan_factor = 0.8  # Slightly reduced effectiveness
        elif self.mounting_position == MountingPosition.ANGLED_UPWARD:
            fan_factor = 0.9  # Good effectiveness due to angle
        else:
            fan_factor = 0.85  # Standard effectiveness
            
        effective_fan_airflow = fan_airflow * fan_factor
        
        # Total airflow
        total_airflow = ram_air + convection_air + effective_fan_airflow
        
        # Account for stagnation at low speeds
        if vehicle_speed < 8.0:
            # Stagnation factor increases as speed decreases
            stagnation_effect = self.stagnation_factor * (1.0 - vehicle_speed / 8.0)
            total_airflow *= (1.0 - stagnation_effect)
        
        return total_airflow
    
    def calculate_heat_rejection(self, coolant_temp: float, ambient_temp: float, 
                               coolant_flow_rate: float, vehicle_speed: float, 
                               fan_airflow: float) -> float:
        """
        Calculate heat rejected by the rear-mounted radiator.
        
        Args:
            coolant_temp: Coolant temperature in °C
            ambient_temp: Ambient air temperature in °C
            coolant_flow_rate: Coolant flow rate in L/min
            vehicle_speed: Vehicle speed in m/s
            fan_airflow: Airflow from cooling fans in m³/s
            
        Returns:
            Heat rejection rate in watts (W)
        """
        # Calculate effective airflow
        effective_airflow = self.calculate_effective_airflow(vehicle_speed, fan_airflow)
        
        # Adjust for heat soak in stationary conditions
        heat_soak_factor = 1.0
        if vehicle_speed < 3.0:
            # Heat soak reduces effectiveness at low speeds
            # More significant for certain mounting positions
            if self.mounting_position == MountingPosition.BEHIND_DRIVER:
                heat_soak_factor = 0.8 - 0.1 * (3.0 - vehicle_speed) / 3.0
            elif self.mounting_position == MountingPosition.ABOVE_DIFFUSER:
                heat_soak_factor = 0.9 - 0.1 * (3.0 - vehicle_speed) / 3.0
            else:
                heat_soak_factor = 0.9 - 0.05 * (3.0 - vehicle_speed) / 3.0
        
        # Use base radiator to calculate heat rejection with effective airflow
        base_rejection = self.base_radiator.calculate_heat_rejection(
            coolant_temp, ambient_temp, coolant_flow_rate, effective_airflow
        )
        
        # Apply heat soak factor
        actual_rejection = base_rejection * heat_soak_factor
        
        return actual_rejection
    
    def get_radiator_specs(self) -> Dict:
        """
        Get specifications of the rear radiator.
        
        Returns:
            Dictionary with radiator specifications
        """
        # Get base radiator specs
        base_specs = self.base_radiator.get_radiator_specs()
        
        # Add rear-specific specs
        rear_specs = {
            'mounting_position': self.mounting_position.name,
            'angle_degrees': self.angle_degrees,
            'distance_from_diffuser': self.distance_from_diffuser,
            'effective_area': self.effective_area,
            'natural_airflow_factor': self.natural_airflow_factor,
            'stagnation_factor': self.stagnation_factor,
            'convection_factor': self.convection_factor,
            'drag_impact': self.drag_impact
        }
        
        # Combine the dictionaries
        return {**base_specs, **rear_specs}


class RearRadiatorDuct:
    """
    Ducting system for rear-mounted radiators.
    
    This class models the air intake and exhaust ducting for a rear-mounted
    radiator, including airflow calculations and pressure effects.
    """
    
    def __init__(self, 
                 duct_type: DuctType = DuctType.NACA_INLET,
                 inlet_area: float = 0.025,         # m²
                 outlet_area: float = 0.03,         # m²
                 duct_length: float = 0.4,          # m
                 duct_efficiency: float = 0.8,      # 0-1 efficiency factor
                 custom_params: Optional[Dict] = None):
        """
        Initialize a radiator duct system.
        
        Args:
            duct_type: Type of duct system
            inlet_area: Inlet area in m²
            outlet_area: Outlet area in m²
            duct_length: Duct length in m
            duct_efficiency: Duct efficiency factor (0-1)
            custom_params: Optional dictionary with custom parameters
        """
        self.duct_type = duct_type
        self.inlet_area = inlet_area
        self.outlet_area = outlet_area
        self.duct_length = duct_length
        self.duct_efficiency = duct_efficiency
        
        # Derived properties
        self.expansion_ratio = outlet_area / inlet_area
        
        # Set duct-specific properties based on type
        if duct_type == DuctType.NACA_INLET:
            self.inlet_efficiency = 0.85  # Good efficiency for NACA ducts
            self.pressure_recovery = 0.80 # Good pressure recovery
            self.drag_coefficient = 0.03  # Low drag
        elif duct_type == DuctType.SIDE_SCOOP:
            self.inlet_efficiency = 0.75  # Moderate efficiency
            self.pressure_recovery = 0.65 # Lower pressure recovery
            self.drag_coefficient = 0.08  # Higher drag
        elif duct_type == DuctType.TOP_INLET:
            self.inlet_efficiency = 0.70  # Moderate efficiency
            self.pressure_recovery = 0.60 # Lower pressure recovery
            self.drag_coefficient = 0.06  # Moderate drag
        elif duct_type == DuctType.DIFFUSER_INTEGRATED:
            self.inlet_efficiency = 0.90  # Very good efficiency
            self.pressure_recovery = 0.85 # Very good pressure recovery
            self.drag_coefficient = 0.02  # Very low drag
        elif duct_type == DuctType.CUSTOM:
            # Use custom parameters if provided
            if custom_params:
                self.inlet_efficiency = custom_params.get('inlet_efficiency', 0.75)
                self.pressure_recovery = custom_params.get('pressure_recovery', 0.70)
                self.drag_coefficient = custom_params.get('drag_coefficient', 0.05)
            else:
                self.inlet_efficiency = 0.75
                self.pressure_recovery = 0.70
                self.drag_coefficient = 0.05
        
        # Calculate flow resistance
        self.flow_resistance = self._calculate_flow_resistance()
        
        logger.info(f"Radiator duct initialized: {duct_type.name}, {inlet_area:.3f}m² inlet area")
    
    def _calculate_flow_resistance(self) -> float:
        """
        Calculate flow resistance of the duct system.
        
        Returns:
            Flow resistance coefficient
        """
        # Base resistance from duct length and expansion
        # This is a simplified model - a full CFD analysis would be more accurate
        base_resistance = 1.5 * self.duct_length / (self.inlet_area ** 0.5)
        
        # Adjust for expansion ratio - expanding ducts increase resistance
        if self.expansion_ratio > 1.0:
            expansion_factor = 1.0 + 0.2 * (self.expansion_ratio - 1.0)
        else:
            expansion_factor = 1.0
        
        # Adjust for duct type
        if self.duct_type == DuctType.NACA_INLET:
            type_factor = 0.8  # Low resistance
        elif self.duct_type == DuctType.DIFFUSER_INTEGRATED:
            type_factor = 0.7  # Very low resistance
        elif self.duct_type == DuctType.SIDE_SCOOP:
            type_factor = 1.1  # Higher resistance
        else:
            type_factor = 1.0  # Baseline
        
        # Calculate overall resistance
        resistance = base_resistance * expansion_factor * type_factor
        
        # Apply efficiency factor (lower efficiency = higher resistance)
        resistance = resistance / self.duct_efficiency
        
        return resistance
    
    def calculate_airflow(self, vehicle_speed: float) -> float:
        """
        Calculate airflow through the duct at given vehicle speed.
        
        Args:
            vehicle_speed: Vehicle speed in m/s
            
        Returns:
            Airflow through duct in m³/s
        """
        if vehicle_speed <= 0:
            return 0.0
        
        # Base flow rate assuming ideal conditions
        # Q = A * v * efficiency
        ideal_flow = self.inlet_area * vehicle_speed * self.inlet_efficiency
        
        # Apply resistance effects - flow decreases with resistance
        # This is a simplified model based on pressure loss principles
        air_density = 1.2  # kg/m³
        dynamic_pressure = 0.5 * air_density * vehicle_speed ** 2
        
        # Pressure loss due to resistance
        pressure_loss = dynamic_pressure * (1.0 - self.pressure_recovery)
        
        # Flow reduction factor based on pressure loss
        # In reality, this would be calculated from the system curve
        flow_factor = 1.0 / (1.0 + (self.flow_resistance * pressure_loss / 100))
        
        # Apply flow factor to ideal flow
        actual_flow = ideal_flow * flow_factor
        
        return actual_flow
    
    def calculate_drag(self, vehicle_speed: float) -> float:
        """
        Calculate drag force contributed by the duct system.
        
        Args:
            vehicle_speed: Vehicle speed in m/s
            
        Returns:
            Drag force in Newtons
        """
        if vehicle_speed <= 0:
            return 0.0
        
        # Calculate dynamic pressure
        air_density = 1.2  # kg/m³
        dynamic_pressure = 0.5 * air_density * vehicle_speed ** 2
        
        # Calculate drag force (F = C_d * A * q)
        drag_force = self.drag_coefficient * self.inlet_area * dynamic_pressure
        
        return drag_force
    
    def get_duct_specs(self) -> Dict:
        """
        Get specifications of the duct system.
        
        Returns:
            Dictionary with duct specifications
        """
        return {
            'duct_type': self.duct_type.name,
            'inlet_area': self.inlet_area,
            'outlet_area': self.outlet_area,
            'duct_length': self.duct_length,
            'expansion_ratio': self.expansion_ratio,
            'duct_efficiency': self.duct_efficiency,
            'inlet_efficiency': self.inlet_efficiency,
            'pressure_recovery': self.pressure_recovery,
            'drag_coefficient': self.drag_coefficient,
            'flow_resistance': self.flow_resistance
        }


class RearRadiatorSystem:
    """
    Complete rear radiator system for Formula Student car.
    
    This class integrates the rear-mounted radiator, ducting, and fans into
    a complete system for cooling simulation.
    """
    
    def __init__(self, 
                 rear_radiator: RearRadiator,
                 inlet_duct: RearRadiatorDuct,
                 outlet_duct: Optional[RearRadiatorDuct] = None,
                 cooling_fan: Optional[CoolingFan] = None):
        """
        Initialize the complete rear radiator system.
        
        Args:
            rear_radiator: RearRadiator object
            inlet_duct: Inlet duct system
            outlet_duct: Optional outlet duct system
            cooling_fan: Optional cooling fan
        """
        self.radiator = rear_radiator
        self.inlet_duct = inlet_duct
        self.outlet_duct = outlet_duct
        self.cooling_fan = cooling_fan
        
        # If no outlet duct specified, create a default one
        if self.outlet_duct is None:
            self.outlet_duct = RearRadiatorDuct(
                duct_type=DuctType.CUSTOM,
                inlet_area=inlet_duct.outlet_area,  # Match inlet duct outlet
                outlet_area=inlet_duct.outlet_area * 1.2,  # Slightly larger
                duct_length=0.2,
                duct_efficiency=0.9
            )
        
        # Current state
        self.current_airflow = 0.0  # m³/s through radiator
        self.fan_control_signal = 0.0  # 0-1
        self.heat_rejection = 0.0  # W
        
        logger.info("Rear radiator system initialized")
    
    def update_fan_control(self, control_signal: float):
        """
        Update cooling fan control.
        
        Args:
            control_signal: Fan control signal (0-1)
        """
        if self.cooling_fan:
            self.fan_control_signal = max(0.0, min(1.0, control_signal))
            self.cooling_fan.update_control(self.fan_control_signal)
    
    def calculate_system_airflow(self, vehicle_speed: float) -> float:
        """
        Calculate airflow through the complete system.
        
        Args:
            vehicle_speed: Vehicle speed in m/s
            
        Returns:
            Total system airflow in m³/s
        """
        # Calculate duct airflow
        duct_airflow = self.inlet_duct.calculate_airflow(vehicle_speed)
        
        # Add fan airflow if present
        fan_airflow = 0.0
        if self.cooling_fan:
            fan_airflow = self.cooling_fan.current_airflow
        
        # Total airflow is the sum of duct and fan airflow
        # In reality, the interaction is more complex and would need CFD analysis
        total_airflow = duct_airflow + fan_airflow
        
        # Limit airflow by outlet duct capacity
        outlet_capacity = self.outlet_duct.calculate_airflow(vehicle_speed) * 1.2  # Allow some backpressure
        total_airflow = min(total_airflow, outlet_capacity)
        
        self.current_airflow = total_airflow
        return total_airflow
    
    def calculate_system_drag(self, vehicle_speed: float) -> float:
        """
        Calculate total drag from the rear radiator system.
        
        Args:
            vehicle_speed: Vehicle speed in m/s
            
        Returns:
            Total drag force in Newtons
        """
        # Calculate duct drag
        inlet_drag = self.inlet_duct.calculate_drag(vehicle_speed)
        outlet_drag = self.outlet_duct.calculate_drag(vehicle_speed)
        
        # Calculate radiator drag (pressure drop effect)
        # This is a simplified model
        air_density = 1.2  # kg/m³
        if self.current_airflow > 0:
            flow_velocity = self.current_airflow / self.radiator.effective_area
            radiator_drag = 0.5 * air_density * flow_velocity**2 * self.radiator.effective_area * 0.5
        else:
            radiator_drag = 0.0
        
        # Add drag from mounting position
        position_drag = 0.5 * air_density * vehicle_speed**2 * self.radiator.drag_impact
        
        # Total drag
        total_drag = inlet_drag + outlet_drag + radiator_drag + position_drag
        
        return total_drag
    
    def calculate_heat_rejection(self, coolant_temp: float, ambient_temp: float, 
                               coolant_flow_rate: float, vehicle_speed: float) -> float:
        """
        Calculate heat rejected by the rear radiator system.
        
        Args:
            coolant_temp: Coolant temperature in °C
            ambient_temp: Ambient air temperature in °C
            coolant_flow_rate: Coolant flow rate in L/min
            vehicle_speed: Vehicle speed in m/s
            
        Returns:
            Heat rejection rate in watts (W)
        """
        # Get fan airflow if present
        fan_airflow = 0.0
        if self.cooling_fan:
            fan_airflow = self.cooling_fan.current_airflow
        
        # Calculate system airflow
        airflow = self.calculate_system_airflow(vehicle_speed)
        
        # Calculate heat rejection
        self.heat_rejection = self.radiator.calculate_heat_rejection(
            coolant_temp, ambient_temp, coolant_flow_rate, vehicle_speed, fan_airflow
        )
        
        return self.heat_rejection
    
    def calculate_coolant_exit_temp(self, inlet_temp: float, coolant_flow_rate: float) -> float:
        """
        Calculate coolant exit temperature.
        
        Args:
            inlet_temp: Coolant inlet temperature in °C
            coolant_flow_rate: Coolant flow rate in L/min
            
        Returns:
            Coolant exit temperature in °C
        """
        return self.radiator.base_radiator.calculate_coolant_exit_temp(
            inlet_temp, self.heat_rejection, coolant_flow_rate
        )
    
    def automatic_fan_control(self, coolant_temp: float, target_temp: float = 90.0, 
                            hysteresis: float = 5.0, vehicle_speed: float = 0.0):
        """
        Apply automatic fan control based on temperature.
        
        Args:
            coolant_temp: Current coolant temperature in °C
            target_temp: Target coolant temperature in °C
            hysteresis: Temperature hysteresis band in °C
            vehicle_speed: Current vehicle speed in m/s for conditional control
        """
        if self.cooling_fan is None:
            return
        
        # Base control on temperature difference from target
        if coolant_temp >= target_temp + hysteresis:
            fan_control = 1.0  # Full fan if temp is very high
        elif coolant_temp > target_temp:
            # Linear ramp from 0 to 1 over the hysteresis range
            fan_control = (coolant_temp - target_temp) / hysteresis
        else:
            fan_control = 0.0  # Fan off if below target
        
        # Modify control based on vehicle speed
        # At high speeds, ram air may be sufficient and fan can be reduced
        if vehicle_speed > 15.0:
            speed_factor = min(1.0, (30.0 - vehicle_speed) / 15.0)
            fan_control *= speed_factor
        
        # Apply control
        self.update_fan_control(fan_control)
    
    def get_system_state(self) -> Dict:
        """
        Get current state of the rear radiator system.
        
        Returns:
            Dictionary with current system state
        """
        state = {
            'current_airflow': self.current_airflow,
            'heat_rejection': self.heat_rejection,
            'fan_control_signal': self.fan_control_signal,
        }
        
        # Add fan state if present
        if self.cooling_fan:
            state['cooling_fan'] = self.cooling_fan.get_fan_state()
        
        return state
    
    def get_system_specs(self) -> Dict:
        """
        Get specifications of the complete rear radiator system.
        
        Returns:
            Dictionary with system specifications
        """
        specs = {
            'radiator': self.radiator.get_radiator_specs(),
            'inlet_duct': self.inlet_duct.get_duct_specs(),
            'outlet_duct': self.outlet_duct.get_duct_specs(),
        }
        
        # Add fan specs if present
        if self.cooling_fan:
            specs['cooling_fan'] = self.cooling_fan.get_fan_specs()
        
        return specs
    
    def analyze_performance(self, vehicle_speed_range: List[float], 
                          coolant_temp: float = 90.0,
                          ambient_temp: float = 25.0,
                          coolant_flow_rate: float = 50.0) -> Dict:
        """
        Analyze system performance across a range of vehicle speeds.
        
        Args:
            vehicle_speed_range: List of vehicle speeds to analyze (m/s)
            coolant_temp: Coolant temperature for analysis in °C
            ambient_temp: Ambient temperature for analysis in °C
            coolant_flow_rate: Coolant flow rate for analysis in L/min
            
        Returns:
            Dictionary with performance results
        """
        # Initialize result arrays
        n_speeds = len(vehicle_speed_range)
        airflows = np.zeros(n_speeds)
        heat_rejections = np.zeros(n_speeds)
        drags = np.zeros(n_speeds)
        
        # Apply fan control appropriate for analysis
        # Use full fan for a conservative analysis
        if self.cooling_fan:
            self.update_fan_control(1.0)
        
        # Run analysis for each speed
        for i, speed in enumerate(vehicle_speed_range):
            airflows[i] = self.calculate_system_airflow(speed)
            heat_rejections[i] = self.calculate_heat_rejection(
                coolant_temp, ambient_temp, coolant_flow_rate, speed
            )
            drags[i] = self.calculate_system_drag(speed)
        
        return {
            'vehicle_speeds': vehicle_speed_range,
            'airflows': airflows,
            'heat_rejections': heat_rejections,
            'drags': drags,
            'analysis_conditions': {
                'coolant_temp': coolant_temp,
                'ambient_temp': ambient_temp,
                'coolant_flow_rate': coolant_flow_rate
            }
        }
    
    def plot_performance_curves(self, analysis_results: Dict, save_path: Optional[str] = None):
        """
        Plot system performance curves from analysis results.
        
        Args:
            analysis_results: Results from analyze_performance
            save_path: Optional path to save the plot
        """
        # Extract data
        speeds = analysis_results['vehicle_speeds']
        airflows = analysis_results['airflows']
        heat_rejections = analysis_results['heat_rejections']
        drags = analysis_results['drags']
        
        # Create figure with multiple subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        
        # Plot airflow
        ax1.plot(speeds, airflows, 'b-', linewidth=2)
        ax1.set_ylabel('Airflow (m³/s)')
        ax1.set_title('Rear Radiator System Performance')
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Plot heat rejection
        ax2.plot(speeds, heat_rejections / 1000, 'r-', linewidth=2)
        ax2.set_ylabel('Heat Rejection (kW)')
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        # Plot drag
        ax3.plot(speeds, drags, 'g-', linewidth=2)
        ax3.set_xlabel('Vehicle Speed (m/s)')
        ax3.set_ylabel('Drag Force (N)')
        ax3.grid(True, linestyle='--', alpha=0.7)
        
        # Get analysis conditions for title
        conditions = analysis_results['analysis_conditions']
        plt.figtext(0.02, 0.02, f"Analysis Conditions: {conditions['coolant_temp']}°C coolant, "
                   f"{conditions['ambient_temp']}°C ambient, {conditions['coolant_flow_rate']} L/min flow",
                   fontsize=9)
        
        plt.tight_layout()
        
        # Save plot if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        plt.show()


def create_default_rear_radiator_system() -> RearRadiatorSystem:
    """
    Create a default rear radiator system configuration for Formula Student.
    
    Returns:
        Configured RearRadiatorSystem
    """
    from .cooling_system import Radiator, RadiatorType, CoolingFan, FanType
    
    # Create base radiator
    base_radiator = Radiator(
        radiator_type=RadiatorType.SINGLE_CORE_ALUMINUM,
        core_area=0.16,
        core_thickness=0.04,
        fin_density=16,
        tube_rows=2
    )
    
    # Create rear radiator
    rear_radiator = RearRadiator(
        radiator=base_radiator,
        mounting_position=MountingPosition.ABOVE_DIFFUSER,
        angle_degrees=20.0,
        distance_from_diffuser=0.08
    )
    
    # Create inlet duct
    inlet_duct = RearRadiatorDuct(
        duct_type=DuctType.NACA_INLET,
        inlet_area=0.03,
        outlet_area=0.04,
        duct_length=0.35
    )
    
    # Create outlet duct
    outlet_duct = RearRadiatorDuct(
        duct_type=DuctType.CUSTOM,
        inlet_area=0.04,
        outlet_area=0.05,
        duct_length=0.15,
        duct_efficiency=0.9
    )
    
    # Create cooling fan
    cooling_fan = CoolingFan(
        fan_type=FanType.VARIABLE_SPEED,
        max_airflow=0.3,
        diameter=0.28
    )
    
    # Create complete system
    system = RearRadiatorSystem(
        rear_radiator=rear_radiator,
        inlet_duct=inlet_duct,
        outlet_duct=outlet_duct,
        cooling_fan=cooling_fan
    )
    
    return system


def create_optimized_rear_radiator_system() -> RearRadiatorSystem:
    """
    Create an optimized rear radiator system for Formula Student racing.
    
    Returns:
        Optimized RearRadiatorSystem
    """
    from .cooling_system import Radiator, RadiatorType, CoolingFan, FanType
    
    # Create high-performance radiator
    base_radiator = Radiator(
        radiator_type=RadiatorType.DOUBLE_CORE_ALUMINUM,
        core_area=0.18,
        core_thickness=0.045,
        fin_density=18,
        tube_rows=2
    )
    
    # Create rear radiator with optimal mounting
    rear_radiator = RearRadiator(
        radiator=base_radiator,
        mounting_position=MountingPosition.SIDE_POD_REAR,  # Better airflow than directly behind driver
        angle_degrees=15.0,  # Moderate angle for balance of airflow and packaging
        distance_from_diffuser=0.1
    )
    
    # Create aerodynamically optimized inlet duct
    inlet_duct = RearRadiatorDuct(
        duct_type=DuctType.DIFFUSER_INTEGRATED,  # Integrated with diffuser for best aero performance
        inlet_area=0.035,
        outlet_area=0.05,
        duct_length=0.3,
        duct_efficiency=0.9
    )
    
    # Create optimized outlet duct
    outlet_duct = RearRadiatorDuct(
        duct_type=DuctType.CUSTOM,
        inlet_area=0.05,
        outlet_area=0.06,
        duct_length=0.12,  # Shorter for less resistance
        duct_efficiency=0.95
    )
    
    # Create high-performance cooling fan
    cooling_fan = CoolingFan(
        fan_type=FanType.DUAL_FAN,  # Dual fans for better airflow
        max_airflow=0.4,
        diameter=0.22  # Smaller diameter fans can fit better in rear packaging
    )
    
    # Create complete system
    system = RearRadiatorSystem(
        rear_radiator=rear_radiator,
        inlet_duct=inlet_duct,
        outlet_duct=outlet_duct,
        cooling_fan=cooling_fan
    )
    
    return system


def create_minimal_weight_rear_radiator_system() -> RearRadiatorSystem:
    """
    Create a minimal weight rear radiator system for Formula Student.
    
    Returns:
        Weight-optimized RearRadiatorSystem
    """
    from .cooling_system import Radiator, RadiatorType, CoolingFan, FanType
    
    # Create lightweight radiator (smaller but still adequate)
    base_radiator = Radiator(
        radiator_type=RadiatorType.SINGLE_CORE_ALUMINUM,
        core_area=0.14,  # Smaller core area
        core_thickness=0.035,  # Thinner core
        fin_density=20,  # Higher fin density for better efficiency
        tube_rows=1  # Single row for weight reduction
    )
    
    # Create angled rear radiator for better natural convection
    rear_radiator = RearRadiator(
        radiator=base_radiator,
        mounting_position=MountingPosition.ANGLED_UPWARD,
        angle_degrees=30.0,  # Steeper angle for better natural convection
        distance_from_diffuser=0.12
    )
    
    # Create minimal inlet duct
    inlet_duct = RearRadiatorDuct(
        duct_type=DuctType.TOP_INLET,  # Direct top inlet for simplicity
        inlet_area=0.025,
        outlet_area=0.03,
        duct_length=0.25,
        duct_efficiency=0.75
    )
    
    # Create minimal outlet duct
    outlet_duct = RearRadiatorDuct(
        duct_type=DuctType.CUSTOM,
        inlet_area=0.03,
        outlet_area=0.035,
        duct_length=0.1,
        duct_efficiency=0.85
    )
    
    # Create lightweight single fan
    cooling_fan = CoolingFan(
        fan_type=FanType.SINGLE_SPEED,  # Simpler fan control
        max_airflow=0.2,
        diameter=0.25
    )
    
    # Create complete system
    system = RearRadiatorSystem(
        rear_radiator=rear_radiator,
        inlet_duct=inlet_duct,
        outlet_duct=outlet_duct,
        cooling_fan=cooling_fan
    )
    
    return system


# Example usage
if __name__ == "__main__":
    # Create an optimized rear radiator system
    system = create_optimized_rear_radiator_system()
    
    print("Rear Radiator System Specifications:")
    specs = system.get_system_specs()
    
    # Print radiator specs
    print("\nRadiator:")
    for key, value in specs['radiator'].items():
        print(f"  {key}: {value}")
    
    # Print inlet duct specs
    print("\nInlet Duct:")
    for key, value in specs['inlet_duct'].items():
        print(f"  {key}: {value}")
    
    # Test performance at different speeds
    print("\nPerformance Analysis:")
    vehicle_speeds = np.linspace(0, 30, 7)  # 0-30 m/s (0-108 km/h)
    
    # Run performance analysis
    performance = system.analyze_performance(
        vehicle_speed_range=vehicle_speeds,
        coolant_temp=90.0,
        ambient_temp=25.0,
        coolant_flow_rate=50.0
    )
    
    # Print performance at key speeds
    print("\nAirflow and Heat Rejection:")
    print("  Speed (m/s) | Airflow (m³/s) | Heat Rejection (kW) | Drag (N)")
    print("  " + "-" * 65)
    
    for i, speed in enumerate(performance['vehicle_speeds']):
        airflow = performance['airflows'][i]
        heat_rej = performance['heat_rejections'][i] / 1000  # Convert to kW
        drag = performance['drags'][i]
        print(f"  {speed:6.1f}      | {airflow:7.3f}      | {heat_rej:10.2f}        | {drag:6.1f}")
    
    # Plot performance curves
    system.plot_performance_curves(performance)
    
    print("\nAnalysis complete!")