"""
Side pod module for Formula Student powertrain simulation.

This module provides classes and functions for modeling the aerodynamic and
thermal aspects of side pods in Formula Student cars. Side pods serve multiple
purposes, including housing radiators, managing airflow, and contributing to the
overall aerodynamic package of the vehicle.

The module includes detailed modeling of side pod geometry, ducting systems,
radiator integration, and aerodynamic effects, allowing for optimization of
both cooling performance and aerodynamic efficiency.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Callable
import logging
from enum import Enum, auto

# Import related modules
from .cooling_system import (
    Radiator, RadiatorType, CoolingFan, FanType, CoolingSystem
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("Side_Pod")


class SidePodType(Enum):
    """Enumeration of different side pod types used in Formula Student cars."""
    MINIMAL = auto()           # Minimalistic design focused on weight reduction
    STANDARD = auto()          # Standard side pod with balanced characteristics
    UNDERCUT = auto()          # Side pod with undercut for improved airflow to diffuser
    HIGH_DOWNFORCE = auto()    # Side pod optimized for downforce generation
    COOLING_FOCUSED = auto()   # Side pod optimized for maximum cooling
    CUSTOM = auto()            # Custom side pod configuration


class RadiatorOrientation(Enum):
    """Enumeration of radiator mounting orientations within side pods."""
    VERTICAL = auto()          # Radiator mounted vertically
    ANGLED = auto()            # Radiator angled for better airflow
    HORIZONTAL = auto()        # Radiator mounted horizontally
    SPLIT = auto()             # Multiple radiators in split configuration
    CUSTOM = auto()            # Custom radiator configuration


class SidePod:
    """
    Side pod model for Formula Student car.
    
    This class models the geometric and aerodynamic characteristics of a
    Formula Student side pod, including dimensions, shape factors, and
    aerodynamic performance data.
    """
    
    def __init__(self, 
                 pod_type: SidePodType = SidePodType.STANDARD,
                 length: float = 0.8,          # m
                 max_width: float = 0.3,       # m
                 max_height: float = 0.35,     # m
                 inlet_area: float = 0.04,     # m²
                 outlet_area: float = 0.05,    # m²
                 floor_clearance: float = 0.05, # m
                 radiator_orientation: RadiatorOrientation = RadiatorOrientation.VERTICAL,
                 custom_params: Optional[Dict] = None):
        """
        Initialize side pod model.
        
        Args:
            pod_type: Type of side pod design
            length: Length of side pod in meters
            max_width: Maximum width of side pod in meters
            max_height: Maximum height of side pod in meters
            inlet_area: Air inlet area in m²
            outlet_area: Air outlet area in m²
            floor_clearance: Distance from ground to bottom of side pod in meters
            radiator_orientation: Orientation of radiator within the side pod
            custom_params: Optional dictionary with custom parameters
        """
        self.pod_type = pod_type
        self.length = length
        self.max_width = max_width
        self.max_height = max_height
        self.inlet_area = inlet_area
        self.outlet_area = outlet_area
        self.floor_clearance = floor_clearance
        self.radiator_orientation = radiator_orientation
        
        # Calculate derived geometric properties
        self.volume = self._calculate_volume()
        self.surface_area = self._calculate_surface_area()
        self.frontal_area = self._calculate_frontal_area()
        
        # Set type-specific parameters
        if pod_type == SidePodType.MINIMAL:
            self.drag_coefficient = 0.22
            self.lift_coefficient = 0.05
            self.cooling_efficiency = 0.7
            self.weight = 3.0  # kg
        elif pod_type == SidePodType.STANDARD:
            self.drag_coefficient = 0.25
            self.lift_coefficient = -0.1  # Negative = downforce
            self.cooling_efficiency = 0.85
            self.weight = 4.5  # kg
        elif pod_type == SidePodType.UNDERCUT:
            self.drag_coefficient = 0.28
            self.lift_coefficient = -0.25
            self.cooling_efficiency = 0.8
            self.weight = 5.0  # kg
        elif pod_type == SidePodType.HIGH_DOWNFORCE:
            self.drag_coefficient = 0.32
            self.lift_coefficient = -0.4
            self.cooling_efficiency = 0.75
            self.weight = 6.0  # kg
        elif pod_type == SidePodType.COOLING_FOCUSED:
            self.drag_coefficient = 0.3
            self.lift_coefficient = -0.15
            self.cooling_efficiency = 0.95
            self.weight = 5.5  # kg
        elif pod_type == SidePodType.CUSTOM:
            # Use custom parameters if provided
            if custom_params:
                self.drag_coefficient = custom_params.get('drag_coefficient', 0.27)
                self.lift_coefficient = custom_params.get('lift_coefficient', -0.15)
                self.cooling_efficiency = custom_params.get('cooling_efficiency', 0.85)
                self.weight = custom_params.get('weight', 5.0)
            else:
                # Default values for custom type
                self.drag_coefficient = 0.27
                self.lift_coefficient = -0.15
                self.cooling_efficiency = 0.85
                self.weight = 5.0
        
        # Additional parameters
        self.radiator_volume_fraction = self._determine_radiator_volume_fraction()
        self.duct_loss_coefficient = self._calculate_duct_loss_coefficient()
        
        logger.info(f"Side pod initialized: {pod_type.name}, {length:.2f}m length")
    
    def _calculate_volume(self) -> float:
        """
        Calculate the approximate volume of the side pod.
        
        Returns:
            Volume in m³
        """
        # This is a simplified calculation - actual volume would depend on specific shape
        # Apply a form factor based on side pod type
        if self.pod_type == SidePodType.MINIMAL:
            form_factor = 0.5  # More streamlined shape
        elif self.pod_type == SidePodType.UNDERCUT:
            form_factor = 0.6  # Accounts for undercut
        elif self.pod_type == SidePodType.HIGH_DOWNFORCE:
            form_factor = 0.7  # More complex shape
        else:
            form_factor = 0.65  # Standard form factor
        
        # Approximate volume calculation
        volume = self.length * self.max_width * self.max_height * form_factor
        
        return volume
    
    def _calculate_surface_area(self) -> float:
        """
        Calculate the approximate surface area of the side pod.
        
        Returns:
            Surface area in m²
        """
        # Simplified surface area calculation
        # Base on a modified rectangular prism formula
        # This is an approximation - actual surface area depends on specific geometry
        
        # Adjust based on pod type
        if self.pod_type == SidePodType.MINIMAL:
            shape_factor = 1.1  # More streamlined shape = less surface area
        elif self.pod_type == SidePodType.HIGH_DOWNFORCE:
            shape_factor = 1.3  # More complex shape = more surface area
        else:
            shape_factor = 1.2  # Standard shape factor
        
        # Approximate surface area
        surface_area = 2 * (
            self.length * self.max_width +
            self.length * self.max_height +
            self.max_width * self.max_height
        ) * shape_factor
        
        return surface_area
    
    def _calculate_frontal_area(self) -> float:
        """
        Calculate the frontal area of the side pod.
        
        Returns:
            Frontal area in m²
        """
        # For side pods, the frontal area is typically the inlet area plus
        # some additional area depending on the shape
        
        # Base frontal area on inlet plus additional exposed area
        if self.pod_type == SidePodType.MINIMAL:
            additional_factor = 1.2
        elif self.pod_type == SidePodType.COOLING_FOCUSED:
            additional_factor = 1.5
        else:
            additional_factor = 1.3
        
        frontal_area = self.inlet_area * additional_factor
        
        return frontal_area
    
    def _determine_radiator_volume_fraction(self) -> float:
        """
        Determine the fraction of side pod volume available for radiator.
        
        Returns:
            Volume fraction (0-1)
        """
        # Different pod types have different space available for radiators
        if self.pod_type == SidePodType.MINIMAL:
            return 0.55  # Limited space for radiator
        elif self.pod_type == SidePodType.COOLING_FOCUSED:
            return 0.7   # Maximized space for radiator
        elif self.pod_type == SidePodType.HIGH_DOWNFORCE:
            return 0.5   # Some space sacrificed for aero
        else:
            return 0.6   # Standard allocation
    
    def _calculate_duct_loss_coefficient(self) -> float:
        """
        Calculate duct loss coefficient based on side pod geometry.
        
        Returns:
            Duct loss coefficient
        """
        # Baseline loss coefficient
        base_loss = 1.5
        
        # Adjust based on inlet/outlet area ratio
        area_ratio = self.outlet_area / self.inlet_area
        if area_ratio > 1.0:
            # Expanding duct - lower losses
            area_factor = 1.0 - 0.1 * min(3.0, area_ratio - 1.0)
        else:
            # Contracting duct - higher losses
            area_factor = 1.0 + 0.3 * min(2.0, 1.0 - area_ratio)
        
        # Adjust based on type
        if self.pod_type == SidePodType.COOLING_FOCUSED:
            type_factor = 0.8  # Optimized for low losses
        elif self.pod_type == SidePodType.MINIMAL:
            type_factor = 1.2  # Compromised for size
        else:
            type_factor = 1.0  # Standard
        
        return base_loss * area_factor * type_factor
    
    def calculate_max_radiator_size(self) -> Tuple[float, float, float]:
        """
        Calculate maximum radiator dimensions that can fit in the side pod.
        
        Returns:
            Tuple of (max_width, max_height, max_thickness) in meters
        """
        # Available volume for radiator
        available_volume = self.volume * self.radiator_volume_fraction
        
        # Radiator dimensions depend on orientation
        if self.radiator_orientation == RadiatorOrientation.VERTICAL:
            max_width = self.max_width * 0.9
            max_height = self.max_height * 0.9
            # Thickness calculated from available volume
            max_thickness = available_volume / (max_width * max_height)
            
        elif self.radiator_orientation == RadiatorOrientation.HORIZONTAL:
            max_width = self.max_width * 0.9
            max_thickness = min(0.05, self.max_height * 0.3)  # Limit thickness
            # Height calculated from available volume
            max_height = available_volume / (max_width * max_thickness)
            
        elif self.radiator_orientation == RadiatorOrientation.ANGLED:
            # For angled orientation, dimensions are adjusted
            max_width = self.max_width * 0.85
            max_height = self.max_height * 0.85
            max_thickness = available_volume / (max_width * max_height)
            
        else:  # SPLIT or CUSTOM
            # For split configuration, return dimensions for one radiator
            # assuming two radiators of equal size
            available_volume /= 2
            max_width = self.max_width * 0.8
            max_height = self.max_height * 0.5
            max_thickness = available_volume / (max_width * max_height)
        
        return max_width, max_height, max_thickness
    
    def calculate_airflow(self, vehicle_speed: float) -> float:
        """
        Calculate airflow through the side pod at given vehicle speed.
        
        Args:
            vehicle_speed: Vehicle speed in m/s
            
        Returns:
            Airflow in m³/s
        """
        if vehicle_speed <= 0:
            return 0.0
        
        # Base airflow calculation
        # Q = inlet_area * velocity * efficiency
        base_flow = self.inlet_area * vehicle_speed
        
        # Efficiency factor based on pod type and internal losses
        if self.pod_type == SidePodType.COOLING_FOCUSED:
            efficiency_factor = 0.85  # High efficiency
        elif self.pod_type == SidePodType.MINIMAL:
            efficiency_factor = 0.65  # Lower efficiency
        else:
            efficiency_factor = 0.75  # Standard efficiency
        
        # Account for duct losses
        loss_factor = 1.0 / (1.0 + self.duct_loss_coefficient * (vehicle_speed / 20.0)**2)
        
        # Calculate actual flow
        actual_flow = base_flow * efficiency_factor * loss_factor
        
        # Apply cooling efficiency as an overall factor
        return actual_flow * self.cooling_efficiency
    
    def calculate_drag(self, vehicle_speed: float, air_density: float = 1.225) -> float:
        """
        Calculate drag force from the side pod.
        
        Args:
            vehicle_speed: Vehicle speed in m/s
            air_density: Air density in kg/m³
            
        Returns:
            Drag force in Newtons
        """
        if vehicle_speed <= 0:
            return 0.0
        
        # Calculate dynamic pressure
        dynamic_pressure = 0.5 * air_density * vehicle_speed**2
        
        # Calculate drag force (F = Cd * A * q)
        drag_force = self.drag_coefficient * self.frontal_area * dynamic_pressure
        
        return drag_force
    
    def calculate_downforce(self, vehicle_speed: float, air_density: float = 1.225) -> float:
        """
        Calculate downforce (negative lift) from the side pod.
        
        Args:
            vehicle_speed: Vehicle speed in m/s
            air_density: Air density in kg/m³
            
        Returns:
            Downforce in Newtons (positive value represents downforce)
        """
        if vehicle_speed <= 0:
            return 0.0
        
        # Calculate dynamic pressure
        dynamic_pressure = 0.5 * air_density * vehicle_speed**2
        
        # Calculate lift/downforce (F = Cl * A * q)
        # Negative lift coefficient means downforce
        downforce = -self.lift_coefficient * self.surface_area * dynamic_pressure
        
        return downforce
    
    def get_side_pod_specs(self) -> Dict:
        """
        Get specifications of the side pod.
        
        Returns:
            Dictionary with side pod specifications
        """
        return {
            'type': self.pod_type.name,
            'radiator_orientation': self.radiator_orientation.name,
            'length': self.length,
            'max_width': self.max_width,
            'max_height': self.max_height,
            'inlet_area': self.inlet_area,
            'outlet_area': self.outlet_area,
            'floor_clearance': self.floor_clearance,
            'volume': self.volume,
            'surface_area': self.surface_area,
            'frontal_area': self.frontal_area,
            'drag_coefficient': self.drag_coefficient,
            'lift_coefficient': self.lift_coefficient,
            'cooling_efficiency': self.cooling_efficiency,
            'weight': self.weight,
            'max_radiator_dimensions': self.calculate_max_radiator_size()
        }


class SidePodRadiator:
    """
    Radiator specifically configured for side pod integration in Formula Student.
    
    This class extends the base Radiator with specific properties and behaviors
    for side pod mounting, including orientation effects and airflow characteristics.
    """
    
    def __init__(self, 
                 radiator: Radiator,
                 side_pod: SidePod,
                 orientation: RadiatorOrientation = RadiatorOrientation.VERTICAL,
                 tilt_angle: float = 0.0,          # degrees from vertical/horizontal
                 position_factor: float = 0.5,     # 0-1 position along side pod length
                 custom_params: Optional[Dict] = None):
        """
        Initialize a side pod radiator configuration.
        
        Args:
            radiator: Base Radiator object
            side_pod: SidePod object where radiator is mounted
            orientation: Mounting orientation of radiator
            tilt_angle: Tilt angle in degrees from reference plane
            position_factor: Position factor along side pod length (0=front, 1=rear)
            custom_params: Optional dictionary with custom parameters
        """
        self.base_radiator = radiator
        self.side_pod = side_pod
        self.orientation = orientation
        self.tilt_angle = tilt_angle
        self.position_factor = position_factor
        
        # Validate radiator dimensions against side pod constraints
        self._validate_dimensions()
        
        # Calculate derived properties
        self.tilt_radians = np.radians(tilt_angle)
        self.effective_area = self._calculate_effective_area()
        
        # Calculate orientation-specific efficiency factors
        self.orientation_efficiency = self._calculate_orientation_efficiency()
        self.position_efficiency = self._calculate_position_efficiency()
        
        # Custom parameters
        if custom_params:
            self.custom_efficiency = custom_params.get('custom_efficiency', 1.0)
        else:
            self.custom_efficiency = 1.0
        
        logger.info(f"Side pod radiator initialized: {orientation.name} orientation, {tilt_angle:.1f}° tilt")
    
    def _validate_dimensions(self):
        """Validate radiator dimensions against side pod constraints."""
        max_width, max_height, max_thickness = self.side_pod.calculate_max_radiator_size()
        
        if self.orientation == RadiatorOrientation.VERTICAL:
            # Check width and height
            if self.base_radiator.core_area / max_height > max_width:
                logger.warning("Radiator width exceeds side pod capacity")
            if self.base_radiator.core_thickness > max_thickness:
                logger.warning("Radiator thickness exceeds side pod capacity")
        
        elif self.orientation == RadiatorOrientation.HORIZONTAL:
            # Check width and thickness
            if self.base_radiator.core_area / max_width > max_height:
                logger.warning("Radiator height exceeds side pod capacity")
            if self.base_radiator.core_thickness > max_thickness:
                logger.warning("Radiator thickness exceeds side pod capacity")
        
        # For other orientations, just warn in general if dimensions seem too large
        elif self.base_radiator.core_area > max_width * max_height:
            logger.warning("Radiator dimensions may exceed side pod capacity")
    
    def _calculate_effective_area(self) -> float:
        """
        Calculate the effective radiator area based on orientation and tilt.
        
        Returns:
            Effective area in m²
        """
        if self.orientation == RadiatorOrientation.VERTICAL:
            # For vertical orientation, tilt is from vertical plane
            return self.base_radiator.core_area * np.cos(self.tilt_radians)
        
        elif self.orientation == RadiatorOrientation.HORIZONTAL:
            # For horizontal orientation, tilt is from horizontal plane
            return self.base_radiator.core_area * np.cos(self.tilt_radians)
        
        elif self.orientation == RadiatorOrientation.ANGLED:
            # For angled orientation (baseline is 45°), adjust based on tilt
            base_angle_rad = np.radians(45)
            total_angle_rad = base_angle_rad + self.tilt_radians
            return self.base_radiator.core_area * np.cos(total_angle_rad)
        
        else:  # SPLIT or CUSTOM
            # For split configuration, assume standard orientation
            return self.base_radiator.core_area
    
    def _calculate_orientation_efficiency(self) -> float:
        """
        Calculate efficiency factor based on radiator orientation.
        
        Returns:
            Orientation efficiency factor (0-1)
        """
        if self.orientation == RadiatorOrientation.VERTICAL:
            # Vertical is typically good for side pods
            base_efficiency = 0.95
            # Adjust for tilt - moderate tilt can improve, excessive reduces
            tilt_factor = 1.0 + 0.05 * np.sin(2 * self.tilt_radians)
            
        elif self.orientation == RadiatorOrientation.HORIZONTAL:
            # Horizontal can be less efficient due to airflow patterns
            base_efficiency = 0.85
            # Adjust for tilt - some tilt usually helps horizontal radiators
            tilt_factor = 1.0 + 0.1 * np.sin(self.tilt_radians)
            
        elif self.orientation == RadiatorOrientation.ANGLED:
            # Angled often provides good compromise
            base_efficiency = 0.9
            # Adjustment based on deviation from optimal angle
            optimal_angle_rad = np.radians(15)  # Assume 15° is optimal
            deviation = abs(self.tilt_radians - optimal_angle_rad)
            tilt_factor = 1.0 - 0.1 * min(1.0, deviation / np.radians(15))
            
        elif self.orientation == RadiatorOrientation.SPLIT:
            # Split configuration can be effective with good design
            base_efficiency = 0.92
            tilt_factor = 1.0
            
        else:  # CUSTOM
            base_efficiency = 0.9
            tilt_factor = 1.0
        
        return base_efficiency * tilt_factor
    
    def _calculate_position_efficiency(self) -> float:
        """
        Calculate efficiency factor based on radiator position in side pod.
        
        Returns:
            Position efficiency factor (0-1)
        """
        # Position efficiency depends on airflow distribution in side pod
        # Middle positions typically get more consistent airflow
        
        # Penalize extreme positions (front or rear)
        if self.position_factor < 0.2:
            # Front position - can get good airflow but may be affected by front wheel wake
            return 0.9
        elif self.position_factor > 0.8:
            # Rear position - airflow may be reduced or disturbed
            return 0.85
        else:
            # Middle position - typically optimal
            # Peak efficiency at position_factor = 0.4-0.6
            return 0.95 + 0.05 * (1 - abs(self.position_factor - 0.5) / 0.1)
    
    def calculate_effective_airflow(self, side_pod_airflow: float) -> float:
        """
        Calculate effective airflow through the radiator.
        
        Args:
            side_pod_airflow: Airflow through the side pod in m³/s
            
        Returns:
            Effective airflow through radiator in m³/s
        """
        # Calculate what fraction of side pod airflow goes through radiator
        # This depends on radiator area relative to side pod inlet/outlet
        area_ratio = min(1.0, self.effective_area / self.side_pod.inlet_area)
        
        # Apply efficiency factors
        effective_airflow = side_pod_airflow * area_ratio * self.orientation_efficiency * self.position_efficiency * self.custom_efficiency
        
        return effective_airflow
    
    def calculate_heat_rejection(self, coolant_temp: float, ambient_temp: float, 
                               coolant_flow_rate: float, vehicle_speed: float) -> float:
        """
        Calculate heat rejected by the side pod radiator.
        
        Args:
            coolant_temp: Coolant temperature in °C
            ambient_temp: Ambient air temperature in °C
            coolant_flow_rate: Coolant flow rate in L/min
            vehicle_speed: Vehicle speed in m/s
            
        Returns:
            Heat rejection rate in watts (W)
        """
        # Calculate airflow through side pod
        side_pod_airflow = self.side_pod.calculate_airflow(vehicle_speed)
        
        # Calculate effective airflow through radiator
        effective_airflow = self.calculate_effective_airflow(side_pod_airflow)
        
        # Use base radiator to calculate heat rejection
        heat_rejection = self.base_radiator.calculate_heat_rejection(
            coolant_temp, ambient_temp, coolant_flow_rate, effective_airflow
        )
        
        return heat_rejection
    
    def calculate_pressure_drop(self, coolant_flow_rate: float) -> float:
        """
        Calculate pressure drop in the radiator.
        
        Args:
            coolant_flow_rate: Coolant flow rate in L/min
            
        Returns:
            Pressure drop in bar
        """
        # Use base radiator calculation with adjustment for orientation
        base_drop = self.base_radiator.calculate_pressure_drop(coolant_flow_rate)
        
        # Orientation can affect pressure drop (e.g., bends in piping)
        if self.orientation == RadiatorOrientation.HORIZONTAL:
            orientation_factor = 1.1  # Slightly higher drop due to bends
        elif self.orientation == RadiatorOrientation.ANGLED:
            orientation_factor = 1.05  # Moderate impact
        elif self.orientation == RadiatorOrientation.SPLIT:
            orientation_factor = 1.2  # Higher drop due to split flow path
        else:
            orientation_factor = 1.0  # Standard vertical orientation
        
        return base_drop * orientation_factor
    
    def get_radiator_specs(self) -> Dict:
        """
        Get specifications of the side pod radiator.
        
        Returns:
            Dictionary with radiator specifications
        """
        # Get base radiator specs
        base_specs = self.base_radiator.get_radiator_specs()
        
        # Add side pod specific specs
        side_pod_specs = {
            'orientation': self.orientation.name,
            'tilt_angle': self.tilt_angle,
            'position_factor': self.position_factor,
            'effective_area': self.effective_area,
            'orientation_efficiency': self.orientation_efficiency,
            'position_efficiency': self.position_efficiency,
            'overall_efficiency': self.orientation_efficiency * self.position_efficiency * self.custom_efficiency
        }
        
        # Combine dictionaries
        return {**base_specs, **side_pod_specs}


class SidePodSystem:
    """
    Complete side pod system for Formula Student car.
    
    This class integrates the side pod, radiator, and optional cooling fan
    into a complete system for cooling and aerodynamic simulation.
    """
    
    def __init__(self, 
                 side_pod: SidePod,
                 radiator: SidePodRadiator,
                 cooling_fan: Optional[CoolingFan] = None,
                 is_left_side: bool = True):
        """
        Initialize the complete side pod system.
        
        Args:
            side_pod: SidePod object
            radiator: SidePodRadiator object
            cooling_fan: Optional CoolingFan object
            is_left_side: Whether this is the left side pod (True) or right (False)
        """
        self.side_pod = side_pod
        self.radiator = radiator
        self.cooling_fan = cooling_fan
        self.is_left_side = is_left_side
        
        # Current state variables
        self.current_airflow = 0.0  # m³/s
        self.fan_control_signal = 0.0  # 0-1
        self.heat_rejection = 0.0  # W
        self.drag_force = 0.0  # N
        self.downforce = 0.0  # N
        
        logger.info(f"Side pod system initialized for {'left' if is_left_side else 'right'} side")
    
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
        # Calculate side pod airflow
        side_pod_airflow = self.side_pod.calculate_airflow(vehicle_speed)
        
        # Add fan airflow if present
        fan_airflow = 0.0
        if self.cooling_fan:
            fan_airflow = self.cooling_fan.current_airflow
            
            # Fan effectiveness depends on installation
            if self.radiator.orientation == RadiatorOrientation.VERTICAL:
                fan_factor = 0.9  # Good effectiveness
            elif self.radiator.orientation == RadiatorOrientation.HORIZONTAL:
                fan_factor = 0.75  # Reduced effectiveness
            else:
                fan_factor = 0.8  # Standard effectiveness
                
            fan_airflow *= fan_factor
        
        # Total airflow is the combination of natural and fan-induced flow
        # This is a simplified model - in reality, the interaction is complex
        # For low speeds, fan contribution is additive
        # For high speeds, ram air dominates and fan contribution diminishes
        if vehicle_speed < 10.0:
            # Low speed - fan is more effective
            total_airflow = side_pod_airflow + fan_airflow
        else:
            # Higher speed - fan contribution diminishes
            fan_contribution = fan_airflow * (1.0 - min(1.0, (vehicle_speed - 10.0) / 20.0))
            total_airflow = side_pod_airflow + fan_contribution
        
        self.current_airflow = total_airflow
        return total_airflow
    
    def calculate_heat_rejection(self, coolant_temp: float, ambient_temp: float, 
                               coolant_flow_rate: float, vehicle_speed: float) -> float:
        """
        Calculate heat rejected by the side pod system.
        
        Args:
            coolant_temp: Coolant temperature in °C
            ambient_temp: Ambient air temperature in °C
            coolant_flow_rate: Coolant flow rate in L/min
            vehicle_speed: Vehicle speed in m/s
            
        Returns:
            Heat rejection rate in watts (W)
        """
        # Update system airflow
        self.calculate_system_airflow(vehicle_speed)
        
        # Calculate heat rejection
        self.heat_rejection = self.radiator.calculate_heat_rejection(
            coolant_temp, ambient_temp, coolant_flow_rate, vehicle_speed
        )
        
        return self.heat_rejection
    
    def calculate_aerodynamic_forces(self, vehicle_speed: float) -> Tuple[float, float]:
        """
        Calculate aerodynamic forces generated by the side pod.
        
        Args:
            vehicle_speed: Vehicle speed in m/s
            
        Returns:
            Tuple of (drag_force, downforce) in Newtons
        """
        self.drag_force = self.side_pod.calculate_drag(vehicle_speed)
        self.downforce = self.side_pod.calculate_downforce(vehicle_speed)
        
        return self.drag_force, self.downforce
    
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
        Get current state of the side pod system.
        
        Returns:
            Dictionary with current system state
        """
        state = {
            'current_airflow': self.current_airflow,
            'heat_rejection': self.heat_rejection,
            'fan_control_signal': self.fan_control_signal,
            'drag_force': self.drag_force,
            'downforce': self.downforce,
            'side': 'left' if self.is_left_side else 'right'
        }
        
        # Add fan state if present
        if self.cooling_fan:
            state['cooling_fan'] = self.cooling_fan.get_fan_state()
        
        return state
    
    def get_system_specs(self) -> Dict:
        """
        Get specifications of the complete side pod system.
        
        Returns:
            Dictionary with system specifications
        """
        specs = {
            'side_pod': self.side_pod.get_side_pod_specs(),
            'radiator': self.radiator.get_radiator_specs(),
            'side': 'left' if self.is_left_side else 'right'
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
        downforces = np.zeros(n_speeds)
        
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
            drag, downforce = self.calculate_aerodynamic_forces(speed)
            drags[i] = drag
            downforces[i] = downforce
        
        return {
            'vehicle_speeds': vehicle_speed_range,
            'airflows': airflows,
            'heat_rejections': heat_rejections,
            'drags': drags,
            'downforces': downforces,
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
        downforces = analysis_results['downforces']
        
        # Create figure with multiple subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        
        # Plot airflow and heat rejection
        ax1.plot(speeds, airflows, 'b-', linewidth=2, label='Airflow')
        ax1.set_ylabel('Airflow (m³/s)')
        ax1.legend(loc='upper left')
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        ax1_twin = ax1.twinx()
        ax1_twin.plot(speeds, heat_rejections / 1000, 'r-', linewidth=2, label='Heat Rejection')
        ax1_twin.set_ylabel('Heat Rejection (kW)')
        ax1_twin.legend(loc='upper right')
        
        # Plot drag force
        ax2.plot(speeds, drags, 'g-', linewidth=2)
        ax2.set_ylabel('Drag Force (N)')
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        # Plot downforce
        ax3.plot(speeds, downforces, 'm-', linewidth=2)
        ax3.set_xlabel('Vehicle Speed (m/s)')
        ax3.set_ylabel('Downforce (N)')
        ax3.grid(True, linestyle='--', alpha=0.7)
        
        # Add title
        side_text = 'Left' if self.is_left_side else 'Right'
        plt.suptitle(f'{side_text} Side Pod System Performance')
        
        # Get analysis conditions for subtitle
        conditions = analysis_results['analysis_conditions']
        plt.figtext(0.02, 0.02, f"Analysis Conditions: {conditions['coolant_temp']}°C coolant, "
                   f"{conditions['ambient_temp']}°C ambient, {conditions['coolant_flow_rate']} L/min flow",
                   fontsize=9)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        
        # Save plot if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        plt.show()


class DualSidePodSystem:
    """
    Dual side pod system for Formula Student car.
    
    This class integrates left and right side pod systems for complete
    vehicle cooling and aerodynamic analysis.
    """
    
    def __init__(self, 
                 left_side_pod: SidePodSystem,
                 right_side_pod: SidePodSystem):
        """
        Initialize the dual side pod system.
        
        Args:
            left_side_pod: Left side pod system
            right_side_pod: Right side pod system
        """
        self.left_system = left_side_pod
        self.right_system = right_side_pod
        
        # Ensure side pod systems are correctly identified
        self.left_system.is_left_side = True
        self.right_system.is_left_side = False
        
        logger.info("Dual side pod system initialized")
    
    def update_fan_control(self, left_control: float, right_control: float):
        """
        Update cooling fan control for both side pods.
        
        Args:
            left_control: Left side fan control signal (0-1)
            right_control: Right side fan control signal (0-1)
        """
        self.left_system.update_fan_control(left_control)
        self.right_system.update_fan_control(right_control)
    
    def calculate_total_airflow(self, vehicle_speed: float) -> float:
        """
        Calculate total airflow through both side pods.
        
        Args:
            vehicle_speed: Vehicle speed in m/s
            
        Returns:
            Total airflow in m³/s
        """
        left_airflow = self.left_system.calculate_system_airflow(vehicle_speed)
        right_airflow = self.right_system.calculate_system_airflow(vehicle_speed)
        
        return left_airflow + right_airflow
    
    def calculate_total_heat_rejection(self, coolant_temp: float, ambient_temp: float,
                                     coolant_flow_rate: float, vehicle_speed: float) -> float:
        """
        Calculate total heat rejection from both side pods.
        
        Args:
            coolant_temp: Coolant temperature in °C
            ambient_temp: Ambient air temperature in °C
            coolant_flow_rate: Coolant flow rate in L/min
            vehicle_speed: Vehicle speed in m/s
            
        Returns:
            Total heat rejection in watts (W)
        """
        # Assume coolant flow is split evenly between both radiators
        # In a real system, this would depend on the plumbing configuration
        left_flow = coolant_flow_rate / 2
        right_flow = coolant_flow_rate / 2
        
        left_rejection = self.left_system.calculate_heat_rejection(
            coolant_temp, ambient_temp, left_flow, vehicle_speed
        )
        
        right_rejection = self.right_system.calculate_heat_rejection(
            coolant_temp, ambient_temp, right_flow, vehicle_speed
        )
        
        return left_rejection + right_rejection
    
    def calculate_total_aerodynamic_forces(self, vehicle_speed: float) -> Tuple[float, float]:
        """
        Calculate total aerodynamic forces from both side pods.
        
        Args:
            vehicle_speed: Vehicle speed in m/s
            
        Returns:
            Tuple of (total_drag, total_downforce) in Newtons
        """
        left_drag, left_downforce = self.left_system.calculate_aerodynamic_forces(vehicle_speed)
        right_drag, right_downforce = self.right_system.calculate_aerodynamic_forces(vehicle_speed)
        
        return left_drag + right_drag, left_downforce + right_downforce
    
    def automatic_fan_control(self, coolant_temp: float, target_temp: float = 90.0,
                            hysteresis: float = 5.0, vehicle_speed: float = 0.0):
        """
        Apply automatic fan control to both side pods.
        
        Args:
            coolant_temp: Current coolant temperature in °C
            target_temp: Target coolant temperature in °C
            hysteresis: Temperature hysteresis band in °C
            vehicle_speed: Current vehicle speed in m/s
        """
        self.left_system.automatic_fan_control(coolant_temp, target_temp, hysteresis, vehicle_speed)
        self.right_system.automatic_fan_control(coolant_temp, target_temp, hysteresis, vehicle_speed)
    
    def get_system_state(self) -> Dict:
        """
        Get current state of the dual side pod system.
        
        Returns:
            Dictionary with current system state
        """
        left_state = self.left_system.get_system_state()
        right_state = self.right_system.get_system_state()
        
        # Calculate totals
        total_airflow = left_state['current_airflow'] + right_state['current_airflow']
        total_heat_rejection = left_state['heat_rejection'] + right_state['heat_rejection']
        total_drag = left_state['drag_force'] + right_state['drag_force']
        total_downforce = left_state['downforce'] + right_state['downforce']
        
        return {
            'left': left_state,
            'right': right_state,
            'total': {
                'current_airflow': total_airflow,
                'heat_rejection': total_heat_rejection,
                'drag_force': total_drag,
                'downforce': total_downforce
            }
        }
    
    def get_system_specs(self) -> Dict:
        """
        Get specifications of the dual side pod system.
        
        Returns:
            Dictionary with system specifications
        """
        left_specs = self.left_system.get_system_specs()
        right_specs = self.right_system.get_system_specs()
        
        return {
            'left': left_specs,
            'right': right_specs,
            'total_weight': left_specs['side_pod']['weight'] + right_specs['side_pod']['weight']
        }
    
    def analyze_system_performance(self, vehicle_speed_range: List[float],
                                 coolant_temp: float = 90.0,
                                 ambient_temp: float = 25.0,
                                 coolant_flow_rate: float = 50.0) -> Dict:
        """
        Analyze performance of the complete dual side pod system.
        
        Args:
            vehicle_speed_range: List of vehicle speeds to analyze (m/s)
            coolant_temp: Coolant temperature for analysis in °C
            ambient_temp: Ambient temperature for analysis in °C
            coolant_flow_rate: Coolant flow rate for analysis in L/min
            
        Returns:
            Dictionary with performance results
        """
        # Get individual analyses
        left_analysis = self.left_system.analyze_performance(
            vehicle_speed_range, coolant_temp, ambient_temp, coolant_flow_rate / 2
        )
        
        right_analysis = self.right_system.analyze_performance(
            vehicle_speed_range, coolant_temp, ambient_temp, coolant_flow_rate / 2
        )
        
        # Calculate combined performance
        speeds = left_analysis['vehicle_speeds']
        total_airflows = left_analysis['airflows'] + right_analysis['airflows']
        total_heat_rejections = left_analysis['heat_rejections'] + right_analysis['heat_rejections']
        total_drags = left_analysis['drags'] + right_analysis['drags']
        total_downforces = left_analysis['downforces'] + right_analysis['downforces']
        
        # Calculate aerodynamic efficiency (downforce/drag ratio)
        aero_efficiency = np.zeros_like(speeds)
        for i, (drag, downforce) in enumerate(zip(total_drags, total_downforces)):
            if drag > 0:
                aero_efficiency[i] = downforce / drag
        
        # Calculate cooling efficiency (heat rejection relative to total drag)
        cooling_efficiency = np.zeros_like(speeds)
        for i, (heat, drag) in enumerate(zip(total_heat_rejections, total_drags)):
            if drag > 0:
                cooling_efficiency[i] = heat / drag  # W/N
        
        return {
            'vehicle_speeds': speeds,
            'total_airflows': total_airflows,
            'total_heat_rejections': total_heat_rejections,
            'total_drags': total_drags,
            'total_downforces': total_downforces,
            'aero_efficiency': aero_efficiency,
            'cooling_efficiency': cooling_efficiency,
            'left_analysis': left_analysis,
            'right_analysis': right_analysis,
            'analysis_conditions': left_analysis['analysis_conditions']
        }
    
    def plot_combined_performance(self, analysis_results: Dict, save_path: Optional[str] = None):
        """
        Plot combined performance of both side pods.
        
        Args:
            analysis_results: Results from analyze_system_performance
            save_path: Optional path to save the plot
        """
        # Extract data
        speeds = analysis_results['vehicle_speeds']
        airflows = analysis_results['total_airflows']
        heat_rejections = analysis_results['total_heat_rejections']
        drags = analysis_results['total_drags']
        downforces = analysis_results['total_downforces']
        aero_efficiency = analysis_results['aero_efficiency']
        cooling_efficiency = analysis_results['cooling_efficiency']
        
        # Create figure with multiple subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 14), sharex=True)
        
        # Plot heat rejection and airflow
        color1 = 'tab:red'
        ax1.set_ylabel('Heat Rejection (kW)', color=color1)
        ax1.plot(speeds, heat_rejections / 1000, color=color1, linewidth=2)
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        color2 = 'tab:blue'
        ax1_twin = ax1.twinx()
        ax1_twin.set_ylabel('Airflow (m³/s)', color=color2)
        ax1_twin.plot(speeds, airflows, color=color2, linewidth=2)
        ax1_twin.tick_params(axis='y', labelcolor=color2)
        
        # Plot drag and downforce
        color3 = 'tab:green'
        ax2.set_ylabel('Forces (N)', color=color3)
        ax2.plot(speeds, drags, color=color3, linewidth=2, label='Drag')
        ax2.plot(speeds, downforces, color='tab:purple', linewidth=2, label='Downforce')
        ax2.tick_params(axis='y', labelcolor=color3)
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend()
        
        # Plot efficiency metrics
        color4 = 'tab:orange'
        ax3.set_ylabel('Aero Efficiency (L/D)', color=color4)
        ax3.plot(speeds, aero_efficiency, color=color4, linewidth=2)
        ax3.tick_params(axis='y', labelcolor=color4)
        ax3.grid(True, linestyle='--', alpha=0.7)
        
        color5 = 'tab:cyan'
        ax3_twin = ax3.twinx()
        ax3_twin.set_ylabel('Cooling Efficiency (W/N)', color=color5)
        ax3_twin.plot(speeds, cooling_efficiency, color=color5, linewidth=2)
        ax3_twin.tick_params(axis='y', labelcolor=color5)
        
        ax3.set_xlabel('Vehicle Speed (m/s)')
        
        # Add title
        plt.suptitle('Dual Side Pod System Performance')
        
        # Get analysis conditions for subtitle
        conditions = analysis_results['analysis_conditions']
        plt.figtext(0.02, 0.02, f"Analysis Conditions: {conditions['coolant_temp']}°C coolant, "
                   f"{conditions['ambient_temp']}°C ambient, {conditions['coolant_flow_rate']} L/min flow",
                   fontsize=9)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        
        # Save plot if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        plt.show()


def create_standard_side_pod_system() -> DualSidePodSystem:
    """
    Create a standard dual side pod system for Formula Student.
    
    Returns:
        Configured DualSidePodSystem
    """
    from .cooling_system import Radiator, RadiatorType, CoolingFan, FanType
    
    # Create base radiators
    left_radiator = Radiator(
        radiator_type=RadiatorType.SINGLE_CORE_ALUMINUM,
        core_area=0.14,
        core_thickness=0.036,
        fin_density=16,
        tube_rows=2
    )
    
    right_radiator = Radiator(
        radiator_type=RadiatorType.SINGLE_CORE_ALUMINUM,
        core_area=0.14,
        core_thickness=0.036,
        fin_density=16,
        tube_rows=2
    )
    
    # Create standard side pods
    left_pod = SidePod(
        pod_type=SidePodType.STANDARD,
        length=0.8,
        max_width=0.28,
        max_height=0.32,
        inlet_area=0.035,
        outlet_area=0.04,
        radiator_orientation=RadiatorOrientation.VERTICAL
    )
    
    right_pod = SidePod(
        pod_type=SidePodType.STANDARD,
        length=0.8,
        max_width=0.28,
        max_height=0.32,
        inlet_area=0.035,
        outlet_area=0.04,
        radiator_orientation=RadiatorOrientation.VERTICAL
    )
    
    # Create side pod radiators
    left_pod_radiator = SidePodRadiator(
        radiator=left_radiator,
        side_pod=left_pod,
        orientation=RadiatorOrientation.VERTICAL,
        tilt_angle=5.0,
        position_factor=0.4
    )
    
    right_pod_radiator = SidePodRadiator(
        radiator=right_radiator,
        side_pod=right_pod,
        orientation=RadiatorOrientation.VERTICAL,
        tilt_angle=5.0,
        position_factor=0.4
    )
    
    # Create fans
    left_fan = CoolingFan(
        fan_type=FanType.VARIABLE_SPEED,
        max_airflow=0.18,
        diameter=0.24
    )
    
    right_fan = CoolingFan(
        fan_type=FanType.VARIABLE_SPEED,
        max_airflow=0.18,
        diameter=0.24
    )
    
    # Create side pod systems
    left_system = SidePodSystem(
        side_pod=left_pod,
        radiator=left_pod_radiator,
        cooling_fan=left_fan,
        is_left_side=True
    )
    
    right_system = SidePodSystem(
        side_pod=right_pod,
        radiator=right_pod_radiator,
        cooling_fan=right_fan,
        is_left_side=False
    )
    
    # Create dual system
    return DualSidePodSystem(left_system, right_system)


def create_aero_optimized_side_pod_system() -> DualSidePodSystem:
    """
    Create an aerodynamically optimized dual side pod system.
    
    Returns:
        Configured DualSidePodSystem
    """
    from .cooling_system import Radiator, RadiatorType, CoolingFan, FanType
    
    # Create base radiators
    left_radiator = Radiator(
        radiator_type=RadiatorType.SINGLE_CORE_ALUMINUM,
        core_area=0.13,
        core_thickness=0.04,
        fin_density=18,
        tube_rows=2
    )
    
    right_radiator = Radiator(
        radiator_type=RadiatorType.SINGLE_CORE_ALUMINUM,
        core_area=0.13,
        core_thickness=0.04,
        fin_density=18,
        tube_rows=2
    )
    
    # Create undercut side pods for better aero
    left_pod = SidePod(
        pod_type=SidePodType.UNDERCUT,
        length=0.85,
        max_width=0.26,
        max_height=0.3,
        inlet_area=0.03,
        outlet_area=0.035,
        floor_clearance=0.04,  # Lower for better ground effect
        radiator_orientation=RadiatorOrientation.ANGLED
    )
    
    right_pod = SidePod(
        pod_type=SidePodType.UNDERCUT,
        length=0.85,
        max_width=0.26,
        max_height=0.3,
        inlet_area=0.03,
        outlet_area=0.035,
        floor_clearance=0.04,
        radiator_orientation=RadiatorOrientation.ANGLED
    )
    
    # Create side pod radiators with angled orientation
    left_pod_radiator = SidePodRadiator(
        radiator=left_radiator,
        side_pod=left_pod,
        orientation=RadiatorOrientation.ANGLED,
        tilt_angle=15.0,  # Angled for better airflow
        position_factor=0.45
    )
    
    right_pod_radiator = SidePodRadiator(
        radiator=right_radiator,
        side_pod=right_pod,
        orientation=RadiatorOrientation.ANGLED,
        tilt_angle=15.0,
        position_factor=0.45
    )
    
    # Create fans
    left_fan = CoolingFan(
        fan_type=FanType.VARIABLE_SPEED,
        max_airflow=0.15,
        diameter=0.22
    )
    
    right_fan = CoolingFan(
        fan_type=FanType.VARIABLE_SPEED,
        max_airflow=0.15,
        diameter=0.22
    )
    
    # Create side pod systems
    left_system = SidePodSystem(
        side_pod=left_pod,
        radiator=left_pod_radiator,
        cooling_fan=left_fan,
        is_left_side=True
    )
    
    right_system = SidePodSystem(
        side_pod=right_pod,
        radiator=right_pod_radiator,
        cooling_fan=right_fan,
        is_left_side=False
    )
    
    # Create dual system
    return DualSidePodSystem(left_system, right_system)


def create_cooling_optimized_side_pod_system() -> DualSidePodSystem:
    """
    Create a cooling-optimized dual side pod system.
    
    Returns:
        Configured DualSidePodSystem
    """
    from .cooling_system import Radiator, RadiatorType, CoolingFan, FanType
    
    # Create high-performance radiators
    left_radiator = Radiator(
        radiator_type=RadiatorType.DOUBLE_CORE_ALUMINUM,
        core_area=0.16,
        core_thickness=0.05,
        fin_density=20,
        tube_rows=2
    )
    
    right_radiator = Radiator(
        radiator_type=RadiatorType.DOUBLE_CORE_ALUMINUM,
        core_area=0.16,
        core_thickness=0.05,
        fin_density=20,
        tube_rows=2
    )
    
    # Create cooling-focused side pods
    left_pod = SidePod(
        pod_type=SidePodType.COOLING_FOCUSED,
        length=0.9,
        max_width=0.32,
        max_height=0.34,
        inlet_area=0.045,
        outlet_area=0.055,
        radiator_orientation=RadiatorOrientation.VERTICAL
    )
    
    right_pod = SidePod(
        pod_type=SidePodType.COOLING_FOCUSED,
        length=0.9,
        max_width=0.32,
        max_height=0.34,
        inlet_area=0.045,
        outlet_area=0.055,
        radiator_orientation=RadiatorOrientation.VERTICAL
    )
    
    # Create side pod radiators
    left_pod_radiator = SidePodRadiator(
        radiator=left_radiator,
        side_pod=left_pod,
        orientation=RadiatorOrientation.VERTICAL,
        tilt_angle=0.0,  # Straight vertical for maximum airflow
        position_factor=0.5  # Centered for optimal airflow
    )
    
    right_pod_radiator = SidePodRadiator(
        radiator=right_radiator,
        side_pod=right_pod,
        orientation=RadiatorOrientation.VERTICAL,
        tilt_angle=0.0,
        position_factor=0.5
    )
    
    # Create high-performance fans
    left_fan = CoolingFan(
        fan_type=FanType.DUAL_FAN,  # Dual fans for maximum airflow
        max_airflow=0.25,
        diameter=0.22
    )
    
    right_fan = CoolingFan(
        fan_type=FanType.DUAL_FAN,
        max_airflow=0.25,
        diameter=0.22
    )
    
    # Create side pod systems
    left_system = SidePodSystem(
        side_pod=left_pod,
        radiator=left_pod_radiator,
        cooling_fan=left_fan,
        is_left_side=True
    )
    
    right_system = SidePodSystem(
        side_pod=right_pod,
        radiator=right_pod_radiator,
        cooling_fan=right_fan,
        is_left_side=False
    )
    
    # Create dual system
    return DualSidePodSystem(left_system, right_system)


def create_minimum_weight_side_pod_system() -> DualSidePodSystem:
    """
    Create a minimum weight dual side pod system.
    
    Returns:
        Configured DualSidePodSystem
    """
    from .cooling_system import Radiator, RadiatorType, CoolingFan, FanType
    
    # Create lightweight radiators
    left_radiator = Radiator(
        radiator_type=RadiatorType.SINGLE_CORE_ALUMINUM,
        core_area=0.12,
        core_thickness=0.03,
        fin_density=16,
        tube_rows=1  # Single row for weight reduction
    )
    
    right_radiator = Radiator(
        radiator_type=RadiatorType.SINGLE_CORE_ALUMINUM,
        core_area=0.12,
        core_thickness=0.03,
        fin_density=16,
        tube_rows=1
    )
    
    # Create minimal side pods
    left_pod = SidePod(
        pod_type=SidePodType.MINIMAL,
        length=0.7,  # Shorter length
        max_width=0.24,  # Narrower
        max_height=0.28,  # Lower height
        inlet_area=0.025,
        outlet_area=0.03,
        radiator_orientation=RadiatorOrientation.VERTICAL
    )
    
    right_pod = SidePod(
        pod_type=SidePodType.MINIMAL,
        length=0.7,
        max_width=0.24,
        max_height=0.28,
        inlet_area=0.025,
        outlet_area=0.03,
        radiator_orientation=RadiatorOrientation.VERTICAL
    )
    
    # Create side pod radiators
    left_pod_radiator = SidePodRadiator(
        radiator=left_radiator,
        side_pod=left_pod,
        orientation=RadiatorOrientation.VERTICAL,
        tilt_angle=8.0,  # Slight angle for improved airflow
        position_factor=0.45
    )
    
    right_pod_radiator = SidePodRadiator(
        radiator=right_radiator,
        side_pod=right_pod,
        orientation=RadiatorOrientation.VERTICAL,
        tilt_angle=8.0,
        position_factor=0.45
    )
    
    # Create lightweight fans (single speed)
    left_fan = CoolingFan(
        fan_type=FanType.SINGLE_SPEED,  # Simpler, lighter fan
        max_airflow=0.15,
        diameter=0.2
    )
    
    right_fan = CoolingFan(
        fan_type=FanType.SINGLE_SPEED,
        max_airflow=0.15,
        diameter=0.2
    )
    
    # Create side pod systems
    left_system = SidePodSystem(
        side_pod=left_pod,
        radiator=left_pod_radiator,
        cooling_fan=left_fan,
        is_left_side=True
    )
    
    right_system = SidePodSystem(
        side_pod=right_pod,
        radiator=right_pod_radiator,
        cooling_fan=right_fan,
        is_left_side=False
    )
    
    # Create dual system
    return DualSidePodSystem(left_system, right_system)


# Example usage
if __name__ == "__main__":
    # Create a standard dual side pod system
    dual_system = create_standard_side_pod_system()
    
    print("Dual Side Pod System Specifications:")
    specs = dual_system.get_system_specs()
    
    # Print left side pod specs
    print("\nLeft Side Pod:")
    for key, value in specs['left']['side_pod'].items():
        if key != 'max_radiator_dimensions':  # Skip detailed dimensions
            print(f"  {key}: {value}")
    
    print("\nLeft Radiator:")
    for key, value in specs['left']['radiator'].items():
        if isinstance(value, (int, float, str)):  # Print only simple values
            print(f"  {key}: {value}")
    
    # Test performance at different speeds
    print("\nPerformance Analysis:")
    vehicle_speeds = np.linspace(0, 30, 7)  # 0-30 m/s (0-108 km/h)
    
    # Run performance analysis
    performance = dual_system.analyze_system_performance(
        vehicle_speed_range=vehicle_speeds,
        coolant_temp=90.0,
        ambient_temp=25.0,
        coolant_flow_rate=50.0
    )
    
    # Print total performance at key speeds
    print("\nTotal system performance:")
    print("  Speed (m/s) | Airflow (m³/s) | Heat Rejection (kW) | Drag (N) | Downforce (N)")
    print("  " + "-" * 80)
    
    for i, speed in enumerate(performance['vehicle_speeds']):
        airflow = performance['total_airflows'][i]
        heat_rej = performance['total_heat_rejections'][i] / 1000  # Convert to kW
        drag = performance['total_drags'][i]
        downforce = performance['total_downforces'][i]
        print(f"  {speed:6.1f}      | {airflow:7.3f}      | {heat_rej:10.2f}        | {drag:6.1f}  | {downforce:8.1f}")
    
    # Plot performance curves
    dual_system.plot_combined_performance(performance)
    
    print("\nAnalysis complete!")
    
    # Compare different side pod configurations
    print("\nComparing different side pod configurations:")
    
    # Create different configurations
    standard_system = create_standard_side_pod_system()
    aero_system = create_aero_optimized_side_pod_system()
    cooling_system = create_cooling_optimized_side_pod_system()
    lightweight_system = create_minimum_weight_side_pod_system()
    
    # Get weights
    standard_weight = standard_system.get_system_specs()['total_weight']
    aero_weight = aero_system.get_system_specs()['total_weight']
    cooling_weight = cooling_system.get_system_specs()['total_weight']
    lightweight_weight = lightweight_system.get_system_specs()['total_weight']
    
    print(f"  Standard configuration weight: {standard_weight:.1f} kg")
    print(f"  Aero-optimized configuration weight: {aero_weight:.1f} kg")
    print(f"  Cooling-optimized configuration weight: {cooling_weight:.1f} kg")
    print(f"  Lightweight configuration weight: {lightweight_weight:.1f} kg")