"""
Electric compressor module for Formula Student powertrain thermal simulation.

This module provides classes and functions for modeling an electric compressor used
to supplement airflow through the cooling system of a Formula Student car. The compressor
is particularly important at low vehicle speeds when natural airflow is insufficient for
adequate cooling. It ensures optimal air flow through radiators and heat exchangers is
maintained across all operating conditions.

The module includes detailed modeling of compressor performance characteristics, power
consumption, control strategies, and integration with the cooling system components.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Callable
import logging
from enum import Enum, auto

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("Electric_Compressor")


class CompressorType(Enum):
    """Enumeration of different electric compressor types used in Formula Student cars."""
    CENTRIFUGAL = auto()      # Centrifugal compressor with high efficiency
    AXIAL = auto()            # Axial flow compressor with compact design
    MIXED_FLOW = auto()       # Mixed flow design balancing efficiency and size
    TWIN_STAGE = auto()       # Two-stage compressor for higher pressure ratio
    CUSTOM = auto()           # Custom specification compressor


class CompressorControl(Enum):
    """Enumeration of control methods for electric compressors."""
    ON_OFF = auto()           # Simple on/off control based on temperature
    VARIABLE_SPEED = auto()   # Variable speed control with PWM
    PID_CONTROL = auto()      # PID control based on temperature difference
    ADAPTIVE = auto()         # Adaptive control based on multiple inputs
    CUSTOM = auto()           # Custom control strategy


class ElectricCompressor:
    """
    Electric compressor model for Formula Student cooling system.
    
    This class models an electric compressor that supplements airflow through the
    cooling system, particularly important at low vehicle speeds when natural
    airflow is insufficient for adequate cooling.
    """
    
    def __init__(self, 
                 compressor_type: CompressorType = CompressorType.CENTRIFUGAL,
                 control_method: CompressorControl = CompressorControl.VARIABLE_SPEED,
                 max_airflow: float = 0.3,        # m³/s
                 max_pressure: float = 200.0,     # Pa
                 max_power: float = 120.0,        # W
                 voltage: float = 12.0,           # V
                 diameter: float = 0.1,           # m
                 weight: float = 0.8,             # kg
                 response_time: float = 0.2,      # seconds to reach setpoint
                 custom_params: Optional[Dict] = None):
        """
        Initialize the electric compressor model.
        
        Args:
            compressor_type: Type of compressor
            control_method: Control method for the compressor
            max_airflow: Maximum airflow in m³/s
            max_pressure: Maximum pressure differential in Pa
            max_power: Maximum power consumption in W
            voltage: Operating voltage in V
            diameter: Compressor diameter in m
            weight: Compressor weight in kg
            response_time: Time to reach setpoint in seconds
            custom_params: Optional dictionary with custom parameters
        """
        self.compressor_type = compressor_type
        self.control_method = control_method
        self.max_airflow = max_airflow
        self.max_pressure = max_pressure
        self.max_power = max_power
        self.voltage = voltage
        self.diameter = diameter
        self.weight = weight
        self.response_time = response_time
        
        # Calculate inlet area
        self.inlet_area = np.pi * (diameter / 2)**2
        
        # Set type-specific parameters
        if compressor_type == CompressorType.CENTRIFUGAL:
            self.efficiency = 0.68
            self.pressure_ratio_max = 1.3
            self.noise_level = 65  # dB
        elif compressor_type == CompressorType.AXIAL:
            self.efficiency = 0.75
            self.pressure_ratio_max = 1.15
            self.noise_level = 70  # dB
        elif compressor_type == CompressorType.MIXED_FLOW:
            self.efficiency = 0.72
            self.pressure_ratio_max = 1.25
            self.noise_level = 68  # dB
        elif compressor_type == CompressorType.TWIN_STAGE:
            self.efficiency = 0.7
            self.pressure_ratio_max = 1.4
            self.noise_level = 72  # dB
        elif compressor_type == CompressorType.CUSTOM:
            # Use custom parameters if provided
            if custom_params:
                self.efficiency = custom_params.get('efficiency', 0.7)
                self.pressure_ratio_max = custom_params.get('pressure_ratio_max', 1.25)
                self.noise_level = custom_params.get('noise_level', 68)
            else:
                # Default values for custom type
                self.efficiency = 0.7
                self.pressure_ratio_max = 1.25
                self.noise_level = 68
        
        # Control parameters
        if control_method == CompressorControl.PID_CONTROL:
            self.kp = 0.8  # Proportional gain
            self.ki = 0.1  # Integral gain
            self.kd = 0.05  # Derivative gain
            self.integral_error = 0.0
            self.previous_error = 0.0
        elif control_method == CompressorControl.ADAPTIVE:
            self.temp_weight = 0.6  # Weight for temperature error
            self.speed_weight = 0.3  # Weight for vehicle speed
            self.load_weight = 0.1  # Weight for engine load
        
        # Current state
        self.current_control_signal = 0.0  # 0-1 control signal
        self.target_control_signal = 0.0  # Target control signal
        self.current_speed = 0.0  # Actual compressor speed (0-1)
        self.current_airflow = 0.0  # m³/s
        self.current_pressure = 0.0  # Pa
        self.current_power = 0.0  # W
        self.is_active = False
        self.run_time = 0.0  # Total run time in seconds
        
        # Performance curves (simplified models)
        # These would typically be based on actual compressor performance data
        self._initialize_performance_curves()
        
        logger.info(f"Electric compressor initialized: {compressor_type.name}, max airflow: {max_airflow} m³/s")
    
    def _initialize_performance_curves(self):
        """Initialize compressor performance curves for airflow, pressure, and power."""
        # Create simplified performance curves
        # In a real implementation, these would be based on measured data
        
        # Speed points for interpolation (0-1 range)
        self.speed_points = np.linspace(0, 1, 11)
        
        # Flow curve (roughly cubic relationship with speed)
        self.flow_curve = self.max_airflow * self.speed_points**3
        
        # Pressure curve (roughly quadratic relationship with speed)
        self.pressure_curve = self.max_pressure * self.speed_points**2
        
        # Power curve (combination of speed-dependent and flow-dependent factors)
        self.power_curve = self.max_power * (0.1 + 0.9 * self.speed_points**3)
        
        # Create interpolation functions
        self.flow_function = lambda x: np.interp(x, self.speed_points, self.flow_curve)
        self.pressure_function = lambda x: np.interp(x, self.speed_points, self.pressure_curve)
        self.power_function = lambda x: np.interp(x, self.speed_points, self.power_curve)
    
    def update_control(self, control_signal: float, dt: float):
        """
        Update compressor control state.
        
        Args:
            control_signal: Control signal (0-1)
            dt: Time step in seconds
        """
        # Bound control signal to valid range
        self.target_control_signal = max(0.0, min(1.0, control_signal))
        
        # Determine if compressor should be active
        if self.control_method == CompressorControl.ON_OFF:
            # Simple on/off control with hysteresis
            if control_signal > 0.6:
                self.is_active = True
                self.target_control_signal = 1.0
            elif control_signal < 0.4:
                self.is_active = False
                self.target_control_signal = 0.0
        else:
            # Variable speed or other control methods
            self.is_active = control_signal > 0.05  # Active if signal above minimum threshold
        
        # Model response time dynamics
        if self.response_time > 0 and dt > 0:
            # Simple first-order response model
            response_rate = dt / self.response_time
            speed_change = (self.target_control_signal - self.current_speed) * response_rate
            self.current_speed = self.current_speed + speed_change
        else:
            # Instant response
            self.current_speed = self.target_control_signal
        
        # Current control signal tracks speed for reporting
        self.current_control_signal = self.current_speed
        
        # Update run time if active
        if self.is_active:
            self.run_time += dt
        
        # Update performance values based on current speed
        self._update_performance_values()
    
    def _update_performance_values(self):
        """Update airflow, pressure, and power based on current speed."""
        if self.is_active and self.current_speed > 0:
            self.current_airflow = float(self.flow_function(self.current_speed))
            self.current_pressure = float(self.pressure_function(self.current_speed))
            self.current_power = float(self.power_function(self.current_speed))
        else:
            self.current_airflow = 0.0
            self.current_pressure = 0.0
            self.current_power = 0.0
    
    def calculate_pid_control(self, setpoint: float, current_value: float, dt: float) -> float:
        """
        Calculate control signal using PID control.
        
        Args:
            setpoint: Target temperature or other controlled variable
            current_value: Current temperature or other controlled variable
            dt: Time step in seconds
            
        Returns:
            Control signal (0-1)
        """
        if self.control_method != CompressorControl.PID_CONTROL:
            logger.warning("PID control method requested but not configured")
            return 0.0
        
        # Calculate error
        error = setpoint - current_value
        
        # Calculate integral term
        self.integral_error += error * dt
        # Anti-windup: limit integral term
        self.integral_error = max(-1.0, min(1.0, self.integral_error))
        
        # Calculate derivative term
        if dt > 0:
            derivative = (error - self.previous_error) / dt
        else:
            derivative = 0.0
        self.previous_error = error
        
        # Calculate PID output
        output = self.kp * error + self.ki * self.integral_error + self.kd * derivative
        
        # Bound output to valid range
        control_signal = max(0.0, min(1.0, output))
        
        return control_signal
    
    def calculate_adaptive_control(self, temp_error: float, vehicle_speed: float, 
                                 engine_load: float) -> float:
        """
        Calculate control signal using adaptive control.
        
        Args:
            temp_error: Temperature error (setpoint - actual)
            vehicle_speed: Vehicle speed in m/s
            engine_load: Engine load factor (0-1)
            
        Returns:
            Control signal (0-1)
        """
        if self.control_method != CompressorControl.ADAPTIVE:
            logger.warning("Adaptive control method requested but not configured")
            return 0.0
        
        # Normalize vehicle speed (assuming 30 m/s as reference)
        norm_speed = max(0.0, min(1.0, 1.0 - vehicle_speed / 30.0))
        
        # Calculate weighted sum
        weighted_sum = (
            self.temp_weight * max(0.0, temp_error / 10.0) +  # Normalize temp error to 0-1 range
            self.speed_weight * norm_speed +
            self.load_weight * engine_load
        )
        
        # Apply non-linear mapping for better low-speed response
        if weighted_sum < 0.3:
            control_signal = 0.0  # Off below threshold to save power
        else:
            control_signal = min(1.0, (weighted_sum - 0.3) * 1.5)  # Scale to full range
        
        return control_signal
    
    def calculate_power_consumption(self) -> float:
        """
        Calculate current power consumption.
        
        Returns:
            Power consumption in watts
        """
        return self.current_power
    
    def calculate_airflow_vs_backpressure(self, back_pressure: float) -> float:
        """
        Calculate actual airflow considering system back pressure.
        
        Args:
            back_pressure: System back pressure in Pa
            
        Returns:
            Actual airflow in m³/s
        """
        if not self.is_active or self.current_speed <= 0:
            return 0.0
        
        # Calculate pressure available to overcome back pressure
        available_pressure = self.current_pressure
        
        if back_pressure >= available_pressure:
            # Cannot overcome back pressure
            return 0.0
        
        # Simplified model of how airflow decreases with back pressure
        # In a real compressor, this would be based on the compressor map
        flow_factor = 1.0 - (back_pressure / available_pressure)**0.5
        
        return self.current_airflow * flow_factor
    
    def get_operational_envelope(self) -> Dict[str, np.ndarray]:
        """
        Get compressor operational envelope data.
        
        Returns:
            Dictionary with arrays for compressor map
        """
        # Create a simple compressor map
        speeds = np.linspace(0.2, 1.0, 5)  # 20% to 100% speed
        flow_rates = np.linspace(0.1, 1.0, 10)  # 10% to 100% flow rate
        
        # Initialize pressure arrays
        pressures = np.zeros((len(speeds), len(flow_rates)))
        efficiencies = np.zeros((len(speeds), len(flow_rates)))
        
        # Create simplified compressor map
        for i, speed in enumerate(speeds):
            max_flow = self.max_airflow * speed**3
            max_press = self.max_pressure * speed**2
            
            for j, flow_frac in enumerate(flow_rates):
                flow = max_flow * flow_frac
                
                # Pressure varies with flow in a roughly parabolic relationship
                # Peak pressure at around 50% flow
                flow_factor = 1.0 - 4.0 * (flow_frac - 0.5)**2
                pressures[i, j] = max_press * flow_factor
                
                # Efficiency also varies with flow in a parabolic relationship
                # Peak efficiency at around 70% flow
                eff_factor = 1.0 - 3.0 * (flow_frac - 0.7)**2
                efficiencies[i, j] = self.efficiency * eff_factor
        
        # Convert flow_rates to actual values
        actual_flows = np.outer(speeds, flow_rates) * self.max_airflow
        
        return {
            'speeds': speeds,
            'flow_rates': flow_rates,
            'actual_flows': actual_flows,
            'pressures': pressures,
            'efficiencies': efficiencies
        }
    
    def get_compressor_state(self) -> Dict:
        """
        Get current state of the compressor.
        
        Returns:
            Dictionary with current compressor state
        """
        return {
            'type': self.compressor_type.name,
            'control_method': self.control_method.name,
            'is_active': self.is_active,
            'current_control_signal': self.current_control_signal,
            'current_speed': self.current_speed,
            'current_airflow': self.current_airflow,
            'current_pressure': self.current_pressure,
            'current_power': self.current_power,
            'run_time': self.run_time
        }
    
    def get_compressor_specs(self) -> Dict:
        """
        Get specifications of the compressor.
        
        Returns:
            Dictionary with compressor specifications
        """
        return {
            'type': self.compressor_type.name,
            'control_method': self.control_method.name,
            'max_airflow': self.max_airflow,
            'max_pressure': self.max_pressure,
            'max_power': self.max_power,
            'voltage': self.voltage,
            'diameter': self.diameter,
            'weight': self.weight,
            'response_time': self.response_time,
            'efficiency': self.efficiency,
            'pressure_ratio_max': self.pressure_ratio_max,
            'noise_level': self.noise_level,
            'inlet_area': self.inlet_area
        }
    
    def plot_performance_curves(self, save_path: Optional[str] = None):
        """
        Plot compressor performance curves.
        
        Args:
            save_path: Optional path to save the plot
        """
        # Create figure with multiple subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        
        # Speed range for plotting
        speed_range = np.linspace(0, 1, 101)
        
        # Calculate performance curves
        airflow_curve = [self.flow_function(s) for s in speed_range]
        pressure_curve = [self.pressure_function(s) for s in speed_range]
        power_curve = [self.power_function(s) for s in speed_range]
        
        # Plot airflow curve
        ax1.plot(speed_range, airflow_curve, 'b-', linewidth=2)
        ax1.set_ylabel('Airflow (m³/s)')
        ax1.set_title('Electric Compressor Performance Curves')
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Plot pressure curve
        ax2.plot(speed_range, pressure_curve, 'r-', linewidth=2)
        ax2.set_ylabel('Pressure (Pa)')
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        # Plot power curve
        ax3.plot(speed_range, power_curve, 'g-', linewidth=2)
        ax3.set_xlabel('Compressor Speed (fraction of max)')
        ax3.set_ylabel('Power Consumption (W)')
        ax3.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        # Save plot if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        plt.show()
    
    def plot_compressor_map(self, save_path: Optional[str] = None):
        """
        Plot compressor map showing operational envelope.
        
        Args:
            save_path: Optional path to save the plot
        """
        # Get operational envelope data
        envelope = self.get_operational_envelope()
        
        # Create figure for compressor map
        plt.figure(figsize=(10, 8))
        
        # Create meshgrid for contour plot
        X, Y = np.meshgrid(envelope['actual_flows'][0], envelope['speeds'])
        
        # Create contour plot of pressures
        contour = plt.contourf(X, Y, envelope['pressures'], 20, cmap='viridis')
        plt.colorbar(contour, label='Pressure (Pa)')
        
        # Add efficiency contour lines
        eff_contour = plt.contour(X, Y, envelope['efficiencies'], 
                                 levels=[0.5, 0.6, 0.65, 0.7], colors='red')
        plt.clabel(eff_contour, inline=True, fontsize=8, fmt='%.2f')
        
        # Add constant speed lines
        for i, speed in enumerate(envelope['speeds']):
            plt.plot(envelope['actual_flows'][i], [speed] * len(envelope['flow_rates']), 
                    'k--', alpha=0.5, linewidth=1)
        
        # Add labels and title
        plt.xlabel('Airflow (m³/s)')
        plt.ylabel('Compressor Speed (fraction of max)')
        plt.title('Compressor Map with Efficiency Contours')
        
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        
        # Save plot if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        plt.show()


class CompressorControlModule:
    """
    Control module for electric compressor in Formula Student cooling system.
    
    This class implements various control strategies for the electric compressor
    based on vehicle conditions, temperature monitoring, and cooling requirements.
    """
    
    def __init__(self, compressor: ElectricCompressor,
                 control_strategy: CompressorControl = CompressorControl.ADAPTIVE,
                 target_temp: float = 90.0,
                 max_temp: float = 100.0,
                 min_temp: float = 80.0,
                 custom_params: Optional[Dict] = None):
        """
        Initialize compressor control module.
        
        Args:
            compressor: ElectricCompressor to control
            control_strategy: Control strategy to use
            target_temp: Target coolant temperature in °C
            max_temp: Maximum allowed coolant temperature in °C
            min_temp: Minimum temperature for compressor activation in °C
            custom_params: Optional dictionary with custom parameters
        """
        self.compressor = compressor
        self.control_strategy = control_strategy
        self.target_temp = target_temp
        self.max_temp = max_temp
        self.min_temp = min_temp
        
        # Control parameters
        if control_strategy == CompressorControl.PID_CONTROL:
            # PID parameters (can be overridden by custom_params)
            self.kp = 0.1  # Proportional gain
            self.ki = 0.01  # Integral gain
            self.kd = 0.05  # Derivative gain
            self.integral = 0.0
            self.prev_error = 0.0
        elif control_strategy == CompressorControl.ADAPTIVE:
            # Adaptive control parameters
            self.temp_weight = 0.7
            self.speed_weight = 0.2
            self.load_weight = 0.1
        
        # Apply custom parameters if provided
        if custom_params:
            for key, value in custom_params.items():
                if hasattr(self, key):
                    setattr(self, key, value)
        
        # Current state
        self.control_output = 0.0
        self.last_vehicle_speed = 0.0
        self.last_engine_load = 0.0
        
        logger.info(f"Compressor control module initialized with {control_strategy.name} strategy")
    
    def update_control(self, coolant_temp: float, vehicle_speed: float, 
                     engine_load: float, dt: float):
        """
        Update compressor control based on current conditions.
        
        Args:
            coolant_temp: Current coolant temperature in °C
            vehicle_speed: Current vehicle speed in m/s
            engine_load: Current engine load (0-1)
            dt: Time step in seconds
        """
        # Store current vehicle state
        self.last_vehicle_speed = vehicle_speed
        self.last_engine_load = engine_load
        
        # Calculate control signal based on strategy
        if self.control_strategy == CompressorControl.ON_OFF:
            # Simple on/off control with hysteresis
            if coolant_temp >= self.target_temp:
                self.control_output = 1.0
            elif coolant_temp <= self.min_temp:
                self.control_output = 0.0
            # Otherwise maintain previous state (hysteresis)
            
        elif self.control_strategy == CompressorControl.VARIABLE_SPEED:
            # Linear control based on temperature
            if coolant_temp <= self.min_temp:
                self.control_output = 0.0
            elif coolant_temp >= self.max_temp:
                self.control_output = 1.0
            else:
                # Linear ramp from min_temp to max_temp
                self.control_output = (coolant_temp - self.min_temp) / (self.max_temp - self.min_temp)
                
            # Adjust based on vehicle speed (less compressor at higher speeds)
            speed_factor = max(0.0, 1.0 - vehicle_speed / 20.0)  # Reduce to zero above 20 m/s
            self.control_output *= speed_factor
            
        elif self.control_strategy == CompressorControl.PID_CONTROL:
            # PID control based on temperature error
            error = coolant_temp - self.target_temp
            
            # Only activate if temperature is above minimum
            if coolant_temp < self.min_temp:
                self.control_output = 0.0
                self.integral = 0.0
                self.prev_error = 0.0
            else:
                # Calculate integral term
                self.integral += error * dt
                # Anti-windup: limit integral term
                self.integral = max(-10.0, min(10.0, self.integral))
                
                # Calculate derivative term
                if dt > 0:
                    derivative = (error - self.prev_error) / dt
                else:
                    derivative = 0.0
                self.prev_error = error
                
                # Calculate PID output
                output = self.kp * error + self.ki * self.integral + self.kd * derivative
                
                # Convert to control signal (0-1)
                # Scale based on temperature range
                scale_factor = 1.0 / (self.max_temp - self.target_temp)
                self.control_output = max(0.0, min(1.0, output * scale_factor))
                
                # Adjust based on vehicle speed
                speed_factor = max(0.0, 1.0 - vehicle_speed / 20.0)
                self.control_output *= speed_factor
                
        elif self.control_strategy == CompressorControl.ADAPTIVE:
            # Adaptive control based on multiple inputs
            
            # Temperature factor (0-1)
            if coolant_temp <= self.min_temp:
                temp_factor = 0.0
            elif coolant_temp >= self.max_temp:
                temp_factor = 1.0
            else:
                temp_factor = (coolant_temp - self.min_temp) / (self.max_temp - self.min_temp)
            
            # Vehicle speed factor (inverse, 1-0)
            speed_factor = max(0.0, min(1.0, 1.0 - vehicle_speed / 20.0))
            
            # Engine load factor (0-1)
            load_factor = engine_load
            
            # Weighted combination
            self.control_output = (
                self.temp_weight * temp_factor +
                self.speed_weight * speed_factor +
                self.load_weight * load_factor
            )
            
            # Apply minimum threshold to save power when minimal effect
            if self.control_output < 0.1:
                self.control_output = 0.0
        
        # Update compressor with calculated control signal
        self.compressor.update_control(self.control_output, dt)
    
    def override_control(self, control_signal: float, dt: float):
        """
        Override automatic control with manual control signal.
        
        Args:
            control_signal: Manual control signal (0-1)
            dt: Time step in seconds
        """
        self.control_output = max(0.0, min(1.0, control_signal))
        self.compressor.update_control(self.control_output, dt)
    
    def reset_controller(self):
        """Reset controller state (integral terms, etc.)."""
        self.control_output = 0.0
        if self.control_strategy == CompressorControl.PID_CONTROL:
            self.integral = 0.0
            self.prev_error = 0.0
        
        # Update compressor with zero control signal
        self.compressor.update_control(0.0, 0.1)  # Small dt for quick response
    
    def get_control_state(self) -> Dict:
        """
        Get current state of the controller.
        
        Returns:
            Dictionary with current controller state
        """
        state = {
            'strategy': self.control_strategy.name,
            'target_temp': self.target_temp,
            'min_temp': self.min_temp,
            'max_temp': self.max_temp,
            'control_output': self.control_output,
            'vehicle_speed': self.last_vehicle_speed,
            'engine_load': self.last_engine_load
        }
        
        # Add strategy-specific state
        if self.control_strategy == CompressorControl.PID_CONTROL:
            state.update({
                'kp': self.kp,
                'ki': self.ki,
                'kd': self.kd,
                'integral': self.integral,
                'prev_error': self.prev_error
            })
        elif self.control_strategy == CompressorControl.ADAPTIVE:
            state.update({
                'temp_weight': self.temp_weight,
                'speed_weight': self.speed_weight,
                'load_weight': self.load_weight
            })
        
        return state


class CoolingAssistSystem:
    """
    Integrated cooling assist system for Formula Student cars.
    
    This class integrates the electric compressor with the cooling system,
    providing a complete solution for supplementary airflow management,
    particularly at low vehicle speeds.
    """
    
    def __init__(self, 
                 compressor: ElectricCompressor,
                 control_module: CompressorControlModule,
                 duct_diameter: float = 0.08,      # m
                 duct_length: float = 0.5,         # m
                 discharge_area: float = 0.01,     # m²
                 custom_params: Optional[Dict] = None):
        """
        Initialize cooling assist system.
        
        Args:
            compressor: Electric compressor
            control_module: Compressor control module
            duct_diameter: Diameter of connecting ducts in m
            duct_length: Length of ducting system in m
            discharge_area: Area of discharge outlet in m²
            custom_params: Optional dictionary with custom parameters
        """
        self.compressor = compressor
        self.control_module = control_module
        self.duct_diameter = duct_diameter
        self.duct_length = duct_length
        self.discharge_area = discharge_area
        
        # Calculate duct parameters
        self.duct_area = np.pi * (duct_diameter / 2)**2
        self.expansion_ratio = discharge_area / self.duct_area
        
        # System performance parameters
        self.pressure_loss_coefficient = self._calculate_pressure_loss()
        
        # Apply custom parameters if provided
        if custom_params:
            for key, value in custom_params.items():
                if hasattr(self, key):
                    setattr(self, key, value)
        
        # Current state
        self.current_airflow = 0.0  # m³/s
        self.current_system_pressure = 0.0  # Pa
        self.current_power = 0.0  # W
        self.current_duct_velocity = 0.0  # m/s
        self.is_active = False
        
        logger.info("Cooling assist system initialized")
    
    def _calculate_pressure_loss(self) -> float:
        """
        Calculate pressure loss coefficient for the ducting system.
        
        Returns:
            Pressure loss coefficient
        """
        # Simple model for pressure loss in ducting system
        # Losses from pipe friction, bends, and expansion
        
        # Friction factor (assumed turbulent flow)
        friction_factor = 0.02
        
        # Loss from pipe friction
        k_friction = friction_factor * self.duct_length / self.duct_diameter
        
        # Loss from bends (assume two 90° bends)
        k_bends = 2 * 0.2  # 0.2 is typical loss coefficient for a smooth 90° bend
        
        # Loss from expansion at outlet
        if self.expansion_ratio > 1:
            k_expansion = (1 - 1/self.expansion_ratio)**2
        else:
            k_expansion = 0.0
        
        # Total loss coefficient
        return k_friction + k_bends + k_expansion
    
    def update_system(self, coolant_temp: float, vehicle_speed: float, 
                    engine_load: float, dt: float, radiator_backpressure: float = 0.0):
        """
        Update cooling assist system state.
        
        Args:
            coolant_temp: Current coolant temperature in °C
            vehicle_speed: Current vehicle speed in m/s
            engine_load: Current engine load (0-1)
            dt: Time step in seconds
            radiator_backpressure: Radiator backpressure in Pa
        """
        # Update control module
        self.control_module.update_control(coolant_temp, vehicle_speed, engine_load, dt)
        
        # Calculate system pressure including radiator backpressure
        self.current_system_pressure = radiator_backpressure
        
        # Calculate actual airflow considering system pressure
        self.current_airflow = self.compressor.calculate_airflow_vs_backpressure(self.current_system_pressure)
        
        # Calculate duct velocity
        if self.duct_area > 0:
            self.current_duct_velocity = self.current_airflow / self.duct_area
        else:
            self.current_duct_velocity = 0.0
        
        # Update active state and power consumption
        self.is_active = self.compressor.is_active
        self.current_power = self.compressor.calculate_power_consumption()
    
    def calculate_supplementary_airflow(self) -> float:
        """
        Calculate supplementary airflow provided to cooling system.
        
        Returns:
            Supplementary airflow in m³/s
        """
        return self.current_airflow
    
    def calculate_total_power(self) -> float:
        """
        Calculate total power consumption of the cooling assist system.
        
        Returns:
            Total power consumption in watts
        """
        return self.current_power
    
    def get_system_state(self) -> Dict:
        """
        Get current state of the cooling assist system.
        
        Returns:
            Dictionary with current system state
        """
        return {
            'is_active': self.is_active,
            'current_airflow': self.current_airflow,
            'current_system_pressure': self.current_system_pressure,
            'current_power': self.current_power,
            'current_duct_velocity': self.current_duct_velocity,
            'compressor': self.compressor.get_compressor_state(),
            'control': self.control_module.get_control_state()
        }
    
    def get_system_specs(self) -> Dict:
        """
        Get specifications of the cooling assist system.
        
        Returns:
            Dictionary with system specifications
        """
        return {
            'compressor': self.compressor.get_compressor_specs(),
            'duct_diameter': self.duct_diameter,
            'duct_length': self.duct_length,
            'duct_area': self.duct_area,
            'discharge_area': self.discharge_area,
            'expansion_ratio': self.expansion_ratio,
            'pressure_loss_coefficient': self.pressure_loss_coefficient
        }
    
    def plot_system_performance(self, vehicle_speed_range: List[float], 
                              coolant_temp: float = 90.0,
                              engine_load: float = 0.5,
                              save_path: Optional[str] = None):
        """
        Plot cooling assist system performance across vehicle speed range.
        
        Args:
            vehicle_speed_range: List of vehicle speeds to analyze (m/s)
            coolant_temp: Coolant temperature for analysis in °C
            engine_load: Engine load for analysis (0-1)
            save_path: Optional path to save the plot
        """
        # Initialize result arrays
        n_speeds = len(vehicle_speed_range)
        airflows = np.zeros(n_speeds)
        powers = np.zeros(n_speeds)
        control_signals = np.zeros(n_speeds)
        
        # Save current state to restore after analysis
        original_state = self.get_system_state()
        
        # Run analysis for each speed
        for i, speed in enumerate(vehicle_speed_range):
            # Update system with a small time step
            self.update_system(coolant_temp, speed, engine_load, 0.1)
            
            # Store results
            airflows[i] = self.current_airflow
            powers[i] = self.current_power
            control_signals[i] = self.control_module.control_output
        
        # Create figure with multiple subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        
        # Plot control signal
        ax1.plot(vehicle_speed_range, control_signals, 'b-', linewidth=2)
        ax1.set_ylabel('Control Signal (0-1)')
        ax1.set_title('Cooling Assist System Performance vs Vehicle Speed')
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Plot airflow
        ax2.plot(vehicle_speed_range, airflows, 'r-', linewidth=2)
        ax2.set_ylabel('Supplementary Airflow (m³/s)')
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        # Plot power consumption
        ax3.plot(vehicle_speed_range, powers, 'g-', linewidth=2)
        ax3.set_xlabel('Vehicle Speed (m/s)')
        ax3.set_ylabel('Power Consumption (W)')
        ax3.grid(True, linestyle='--', alpha=0.7)
        
        # Add analysis conditions as subtitle
        plt.figtext(0.5, 0.01, 
                   f"Analysis Conditions: Coolant Temp = {coolant_temp}°C, Engine Load = {engine_load}",
                   ha='center', fontsize=10)
        
        plt.tight_layout(rect=[0, 0.03, 1, 1])
        
        # Save plot if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        plt.show()


def create_default_cooling_assist_system() -> CoolingAssistSystem:
    """
    Create a default configuration cooling assist system for Formula Student.
    
    Returns:
        Configured CoolingAssistSystem
    """
    # Create compressor
    compressor = ElectricCompressor(
        compressor_type=CompressorType.CENTRIFUGAL,
        control_method=CompressorControl.VARIABLE_SPEED,
        max_airflow=0.25,
        max_pressure=180.0,
        max_power=100.0
    )
    
    # Create control module
    control_module = CompressorControlModule(
        compressor=compressor,
        control_strategy=CompressorControl.ADAPTIVE,
        target_temp=90.0,
        max_temp=98.0,
        min_temp=85.0
    )
    
    # Create cooling assist system
    cooling_assist = CoolingAssistSystem(
        compressor=compressor,
        control_module=control_module,
        duct_diameter=0.08,
        duct_length=0.5,
        discharge_area=0.01
    )
    
    return cooling_assist


def create_high_performance_cooling_assist_system() -> CoolingAssistSystem:
    """
    Create a high-performance cooling assist system for Formula Student.
    
    Returns:
        Configured high-performance CoolingAssistSystem
    """
    # Create high-performance compressor
    compressor = ElectricCompressor(
        compressor_type=CompressorType.TWIN_STAGE,
        control_method=CompressorControl.VARIABLE_SPEED,
        max_airflow=0.35,
        max_pressure=250.0,
        max_power=150.0,
        efficiency=0.75
    )
    
    # Create optimized control module
    control_module = CompressorControlModule(
        compressor=compressor,
        control_strategy=CompressorControl.PID_CONTROL,
        target_temp=88.0,  # Slightly lower target for better performance
        max_temp=95.0,
        min_temp=82.0,
        custom_params={
            'kp': 0.12,
            'ki': 0.02,
            'kd': 0.08
        }
    )
    
    # Create high-performance cooling assist system
    cooling_assist = CoolingAssistSystem(
        compressor=compressor,
        control_module=control_module,
        duct_diameter=0.09,  # Larger ducts for better flow
        duct_length=0.4,     # Shorter ducts for less pressure drop
        discharge_area=0.015 # Larger discharge area
    )
    
    return cooling_assist


def create_lightweight_cooling_assist_system() -> CoolingAssistSystem:
    """
    Create a lightweight cooling assist system for Formula Student.
    
    Returns:
        Configured lightweight CoolingAssistSystem
    """
    # Create lightweight compressor
    compressor = ElectricCompressor(
        compressor_type=CompressorType.AXIAL,
        control_method=CompressorControl.ON_OFF,  # Simpler control for weight saving
        max_airflow=0.2,
        max_pressure=150.0,
        max_power=80.0,
        weight=0.6,  # Lighter weight
        diameter=0.08
    )
    
    # Create simple control module
    control_module = CompressorControlModule(
        compressor=compressor,
        control_strategy=CompressorControl.ON_OFF,
        target_temp=92.0,  # Higher target to save power
        max_temp=100.0,
        min_temp=88.0
    )
    
    # Create lightweight cooling assist system
    cooling_assist = CoolingAssistSystem(
        compressor=compressor,
        control_module=control_module,
        duct_diameter=0.07,  # Smaller ducts for weight saving
        duct_length=0.45,
        discharge_area=0.008
    )
    
    return cooling_assist


def create_integrated_cooling_system(cooling_assist: CoolingAssistSystem) -> Dict:
    """
    Create an integrated cooling system configuration that combines the cooling assist
    system with radiators in a Formula Student car.
    
    Args:
        cooling_assist: CoolingAssistSystem to integrate
        
    Returns:
        Dictionary with integrated system configuration
    """
    from .cooling_system import (
        CoolingSystem, Radiator, RadiatorType, WaterPump, 
        PumpType, CoolingFan, FanType, Thermostat
    )
    
    # Create base cooling system components
    radiator = Radiator(
        radiator_type=RadiatorType.SINGLE_CORE_ALUMINUM,
        core_area=0.16,
        core_thickness=0.04,
        fin_density=16,
        tube_rows=2
    )
    
    water_pump = WaterPump(
        pump_type=PumpType.ELECTRIC,
        max_flow_rate=70.0,
        max_pressure=1.7,
        nominal_speed=6000.0
    )
    
    cooling_fan = CoolingFan(
        fan_type=FanType.VARIABLE_SPEED,
        max_airflow=0.28,
        diameter=0.27
    )
    
    thermostat = Thermostat(
        opening_temp=82.0,
        full_open_temp=92.0
    )
    
    # Create complete cooling system
    cooling_system = CoolingSystem(
        radiator=radiator,
        water_pump=water_pump,
        cooling_fan=cooling_fan,
        thermostat=thermostat
    )
    
    # Return integrated configuration
    return {
        'cooling_system': cooling_system,
        'cooling_assist': cooling_assist,
        'integration_notes': [
            "Connect compressor discharge to radiator inlet via ducting",
            "Mount compressor in sidepod with adequate ventilation",
            "Ensure electrical connections are waterproof",
            "Add relay for compressor control",
            "Include fuse protection (15A recommended)",
            "Connect control module to engine management system"
        ],
        'recommended_settings': {
            'target_temp': 90.0,
            'fan_activation_temp': 88.0,
            'compressor_activation_temp': 85.0,
            'max_coolant_temp': 98.0
        }
    }


# Example usage
if __name__ == "__main__":
    # Create a default cooling assist system
    cooling_assist = create_default_cooling_assist_system()
    
    # Print system specifications
    print("Cooling Assist System Specifications:")
    specs = cooling_assist.get_system_specs()
    
    # Print compressor specs
    print("\nCompressor:")
    for key, value in specs['compressor'].items():
        print(f"  {key}: {value}")
    
    # Test system performance at different vehicle speeds
    print("\nTesting system performance:")
    test_speeds = [0, 5, 10, 15, 20, 30]
    coolant_temp = 92.0  # °C
    engine_load = 0.7
    
    for speed in test_speeds:
        # Update system with a small time step
        cooling_assist.update_system(coolant_temp, speed, engine_load, 0.1)
        state = cooling_assist.get_system_state()
        
        # Print results
        print(f"\nAt {speed} m/s vehicle speed:")
        print(f"  Control Signal: {state['control']['control_output']:.2f}")
        print(f"  Compressor Active: {state['is_active']}")
        print(f"  Supplementary Airflow: {state['current_airflow']:.3f} m³/s")
        print(f"  Power Consumption: {state['current_power']:.1f} W")
    
    # Plot system performance curve
    print("\nPlotting system performance curve...")
    cooling_assist.plot_system_performance(
        vehicle_speed_range=np.linspace(0, 30, 31),
        coolant_temp=92.0,
        engine_load=0.7
    )
    
    # Plot compressor performance curves
    print("\nPlotting compressor performance curves...")
    cooling_assist.compressor.plot_performance_curves()
    
    print("\nAnalysis complete!")