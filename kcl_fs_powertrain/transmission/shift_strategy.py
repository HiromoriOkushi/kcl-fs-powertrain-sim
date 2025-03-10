"""
Shift strategy module for Formula Student powertrain simulation.

This module provides classes and functions for implementing optimal gear shifting
strategies for a Formula Student car, working in conjunction with the CAS system
and gearing system. It determines when to shift gears based on engine performance,
vehicle conditions, and race event requirements.

The module includes different shifting strategies optimized for:
- Maximum acceleration
- Maximum efficiency
- Endurance events
- Acceleration events
- Combined strategies
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum, auto
from typing import Dict, List, Tuple, Optional, Union, Callable
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("Shift_Strategy")


class StrategyType(Enum):
    """Enumeration of different shift strategy types."""
    MAX_ACCELERATION = auto()  # Strategy optimized for maximum acceleration
    MAX_EFFICIENCY = auto()    # Strategy optimized for fuel efficiency
    ENDURANCE = auto()         # Strategy balanced for endurance events
    SKIDPAD = auto()           # Strategy for skidpad event
    AUTOCROSS = auto()         # Strategy for autocross event
    ACCELERATION = auto()      # Strategy for acceleration run
    CUSTOM = auto()            # Custom user-defined strategy


class ShiftCondition(Enum):
    """Enumeration of conditions that can trigger a shift."""
    RPM_THRESHOLD = auto()     # Shift based on engine RPM
    SPEED_THRESHOLD = auto()   # Shift based on vehicle speed
    LOAD_THRESHOLD = auto()    # Shift based on engine load
    TORQUE_CURVE = auto()      # Shift based on torque curve
    POWER_CURVE = auto()       # Shift based on power curve
    TIME_BASED = auto()        # Shift after a specific time
    DISTANCE_BASED = auto()    # Shift after a specific distance
    CUSTOM_CONDITION = auto()  # Custom user-defined condition


class ShiftPoint:
    """
    Class representing a gear shift point with associated conditions.
    """
    
    def __init__(self, gear: int, target_gear: int, 
                 condition_type: ShiftCondition, threshold_value: float,
                 priority: int = 1, description: str = ""):
        """
        Initialize shift point with conditions.
        
        Args:
            gear: Current gear
            target_gear: Target gear to shift to
            condition_type: Type of condition that triggers shift
            threshold_value: Value at which shift is triggered
            priority: Priority level (higher number = higher priority)
            description: Human-readable description of shift point
        """
        self.gear = gear
        self.target_gear = target_gear
        self.condition_type = condition_type
        self.threshold_value = threshold_value
        self.priority = priority
        self.description = description
    
    def __str__(self) -> str:
        """String representation of shift point."""
        return (f"Shift from {self.gear} to {self.target_gear} when "
                f"{self.condition_type.name} reaches {self.threshold_value} "
                f"(Priority: {self.priority})")


class ShiftStrategy:
    """
    Base class for shift strategies.
    
    This class provides the framework for implementing different shift strategies
    and evaluating when to shift gears based on vehicle and engine conditions.
    """
    
    def __init__(self, strategy_type: StrategyType, name: str = ""):
        """
        Initialize the shift strategy.
        
        Args:
            strategy_type: Type of shift strategy
            name: Optional name for the strategy
        """
        self.strategy_type = strategy_type
        self.name = name or strategy_type.name.lower().replace("_", " ").title()
        
        # Initialize shift points storage
        self.upshift_points = {}  # Maps gear to list of ShiftPoint objects
        self.downshift_points = {}  # Maps gear to list of ShiftPoint objects
        
        # Performance metrics
        self.shift_history = []  # List of executed shifts
        self.strategy_performance = {}  # Performance metrics
        
        logger.info(f"Shift strategy initialized: {self.name}")
    
    def add_upshift_point(self, shift_point: ShiftPoint):
        """
        Add an upshift point to the strategy.
        
        Args:
            shift_point: ShiftPoint object defining upshift condition
        """
        gear = shift_point.gear
        if gear not in self.upshift_points:
            self.upshift_points[gear] = []
        
        self.upshift_points[gear].append(shift_point)
        # Sort by priority (descending)
        self.upshift_points[gear].sort(key=lambda sp: sp.priority, reverse=True)
        
        logger.debug(f"Added upshift point: {shift_point}")
    
    def add_downshift_point(self, shift_point: ShiftPoint):
        """
        Add a downshift point to the strategy.
        
        Args:
            shift_point: ShiftPoint object defining downshift condition
        """
        gear = shift_point.gear
        if gear not in self.downshift_points:
            self.downshift_points[gear] = []
        
        self.downshift_points[gear].append(shift_point)
        # Sort by priority (descending)
        self.downshift_points[gear].sort(key=lambda sp: sp.priority, reverse=True)
        
        logger.debug(f"Added downshift point: {shift_point}")
    
    def evaluate_shift(self, current_gear: int, engine_rpm: float, vehicle_speed: float,
                      engine_load: float, throttle_position: float, 
                      vehicle_state: Dict = None) -> Optional[int]:
        """
        Evaluate whether a shift is needed based on current conditions.
        
        Args:
            current_gear: Current gear
            engine_rpm: Current engine RPM
            vehicle_speed: Current vehicle speed in m/s
            engine_load: Current engine load (0-1)
            throttle_position: Current throttle position (0-1)
            vehicle_state: Optional dictionary with additional vehicle state data
            
        Returns:
            Target gear to shift to, or None if no shift is needed
        """
        # Don't shift if throttle is very low (likely preparing to stop)
        if throttle_position < 0.1 and engine_rpm < 3000:
            return None
        
        # Check upshift conditions
        if current_gear in self.upshift_points and current_gear > 0:
            for shift_point in self.upshift_points[current_gear]:
                if self._check_shift_condition(shift_point, engine_rpm, vehicle_speed, 
                                             engine_load, throttle_position, vehicle_state):
                    return shift_point.target_gear
        
        # Check downshift conditions
        if current_gear in self.downshift_points and current_gear > 0:
            for shift_point in self.downshift_points[current_gear]:
                if self._check_shift_condition(shift_point, engine_rpm, vehicle_speed, 
                                             engine_load, throttle_position, vehicle_state):
                    return shift_point.target_gear
        
        # No shift needed
        return None
    
    def _check_shift_condition(self, shift_point: ShiftPoint, engine_rpm: float, 
                              vehicle_speed: float, engine_load: float,
                              throttle_position: float, vehicle_state: Dict = None) -> bool:
        """
        Check if a specific shift condition is met.
        
        Args:
            shift_point: ShiftPoint object to evaluate
            engine_rpm: Current engine RPM
            vehicle_speed: Current vehicle speed in m/s
            engine_load: Current engine load (0-1)
            throttle_position: Current throttle position (0-1)
            vehicle_state: Optional dictionary with additional vehicle state data
            
        Returns:
            True if condition is met, False otherwise
        """
        condition = shift_point.condition_type
        threshold = shift_point.threshold_value
        
        # Evaluate based on condition type
        if condition == ShiftCondition.RPM_THRESHOLD:
            # For upshift, RPM should be above threshold
            # For downshift, RPM should be below threshold
            if shift_point.target_gear > shift_point.gear:  # Upshift
                return engine_rpm >= threshold
            else:  # Downshift
                return engine_rpm <= threshold
                
        elif condition == ShiftCondition.SPEED_THRESHOLD:
            # Similar logic for vehicle speed
            if shift_point.target_gear > shift_point.gear:  # Upshift
                return vehicle_speed >= threshold
            else:  # Downshift
                return vehicle_speed <= threshold
                
        elif condition == ShiftCondition.LOAD_THRESHOLD:
            # Engine load threshold
            if shift_point.target_gear > shift_point.gear:  # Upshift
                return engine_load >= threshold
            else:  # Downshift
                return engine_load <= threshold
        
        elif condition == ShiftCondition.TORQUE_CURVE:
            # This would require a more complex evaluation using the engine's torque curve
            # We would need that provided in the vehicle_state
            if vehicle_state and 'torque_curve' in vehicle_state:
                torque_curve = vehicle_state['torque_curve']
                current_torque = np.interp(engine_rpm, torque_curve['rpm'], torque_curve['torque'])
                predicted_torque_next_gear = self._predict_torque_in_gear(
                    engine_rpm, shift_point.target_gear, torque_curve, vehicle_state)
                
                # Compare torque multiplication through gears
                if shift_point.target_gear > shift_point.gear:  # Upshift
                    return predicted_torque_next_gear >= current_torque * threshold
                else:  # Downshift
                    return predicted_torque_next_gear >= current_torque * threshold
            
            return False
            
        elif condition == ShiftCondition.POWER_CURVE:
            # Similar to torque curve but with power
            if vehicle_state and 'power_curve' in vehicle_state:
                power_curve = vehicle_state['power_curve']
                current_power = np.interp(engine_rpm, power_curve['rpm'], power_curve['power'])
                predicted_power_next_gear = self._predict_power_in_gear(
                    engine_rpm, shift_point.target_gear, power_curve, vehicle_state)
                
                if shift_point.target_gear > shift_point.gear:  # Upshift
                    return predicted_power_next_gear >= current_power * threshold
                else:  # Downshift
                    return predicted_power_next_gear >= current_power * threshold
            
            return False
            
        elif condition == ShiftCondition.TIME_BASED:
            # Time-based shifts (e.g., for launch control)
            if vehicle_state and 'elapsed_time' in vehicle_state:
                return vehicle_state['elapsed_time'] >= threshold
            
            return False
            
        elif condition == ShiftCondition.DISTANCE_BASED:
            # Distance-based shifts
            if vehicle_state and 'distance_traveled' in vehicle_state:
                return vehicle_state['distance_traveled'] >= threshold
            
            return False
            
        elif condition == ShiftCondition.CUSTOM_CONDITION:
            # Evaluate custom condition if provided
            if vehicle_state and 'custom_condition_func' in vehicle_state:
                custom_func = vehicle_state['custom_condition_func']
                return custom_func(shift_point, engine_rpm, vehicle_speed, 
                                 engine_load, throttle_position, vehicle_state)
            
            return False
        
        return False
    
    def _predict_torque_in_gear(self, current_rpm: float, target_gear: int, 
                              torque_curve: Dict, vehicle_state: Dict) -> float:
        """
        Predict engine torque after shifting to target gear.
        
        Args:
            current_rpm: Current engine RPM
            target_gear: Target gear
            torque_curve: Engine torque curve data
            vehicle_state: Vehicle state data
            
        Returns:
            Predicted torque in target gear
        """
        # This requires gear ratios and vehicle speed to calculate
        if 'gear_ratios' not in vehicle_state or 'current_gear' not in vehicle_state:
            return 0.0
        
        gear_ratios = vehicle_state['gear_ratios']
        current_gear = vehicle_state['current_gear']
        
        # Ensure we have valid gear data
        if current_gear <= 0 or current_gear > len(gear_ratios) or target_gear <= 0 or target_gear > len(gear_ratios):
            return 0.0
        
        # Calculate RPM in target gear
        current_ratio = gear_ratios[current_gear - 1]
        target_ratio = gear_ratios[target_gear - 1]
        target_rpm = current_rpm * (current_ratio / target_ratio)
        
        # Interpolate torque at target RPM
        rpm_points = torque_curve['rpm']
        torque_points = torque_curve['torque']
        
        # Ensure target RPM is within range
        if target_rpm < min(rpm_points) or target_rpm > max(rpm_points):
            return 0.0
        
        # Interpolate torque
        predicted_torque = np.interp(target_rpm, rpm_points, torque_points)
        
        return predicted_torque
    
    def _predict_power_in_gear(self, current_rpm: float, target_gear: int, 
                              power_curve: Dict, vehicle_state: Dict) -> float:
        """
        Predict engine power after shifting to target gear.
        
        Args:
            current_rpm: Current engine RPM
            target_gear: Target gear
            power_curve: Engine power curve data
            vehicle_state: Vehicle state data
            
        Returns:
            Predicted power in target gear
        """
        # Similar to torque prediction but for power
        if 'gear_ratios' not in vehicle_state or 'current_gear' not in vehicle_state:
            return 0.0
        
        gear_ratios = vehicle_state['gear_ratios']
        current_gear = vehicle_state['current_gear']
        
        # Ensure we have valid gear data
        if current_gear <= 0 or current_gear > len(gear_ratios) or target_gear <= 0 or target_gear > len(gear_ratios):
            return 0.0
        
        # Calculate RPM in target gear
        current_ratio = gear_ratios[current_gear - 1]
        target_ratio = gear_ratios[target_gear - 1]
        target_rpm = current_rpm * (current_ratio / target_ratio)
        
        # Interpolate power at target RPM
        rpm_points = power_curve['rpm']
        power_points = power_curve['power']
        
        # Ensure target RPM is within range
        if target_rpm < min(rpm_points) or target_rpm > max(rpm_points):
            return 0.0
        
        # Interpolate power
        predicted_power = np.interp(target_rpm, rpm_points, power_points)
        
        return predicted_power
    
    def record_shift(self, from_gear: int, to_gear: int, engine_rpm: float, 
                   vehicle_speed: float, timestamp: float):
        """
        Record a shift for performance analysis.
        
        Args:
            from_gear: Starting gear
            to_gear: Target gear
            engine_rpm: Engine RPM at shift
            vehicle_speed: Vehicle speed at shift
            timestamp: Time of shift
        """
        shift_record = {
            'from_gear': from_gear,
            'to_gear': to_gear,
            'engine_rpm': engine_rpm,
            'vehicle_speed': vehicle_speed,
            'timestamp': timestamp,
            'shift_type': 'upshift' if to_gear > from_gear else 'downshift'
        }
        
        self.shift_history.append(shift_record)
        logger.debug(f"Recorded shift: {from_gear} -> {to_gear} at {engine_rpm} RPM")
    
    def analyze_performance(self) -> Dict:
        """
        Analyze shift strategy performance based on recorded shifts.
        
        Returns:
            Dictionary with performance metrics
        """
        if not self.shift_history:
            return {'shifts_analyzed': 0}
        
        # Count shifts by type
        upshifts = [s for s in self.shift_history if s['shift_type'] == 'upshift']
        downshifts = [s for s in self.shift_history if s['shift_type'] == 'downshift']
        
        # Calculate average RPM at shift
        avg_upshift_rpm = sum(s['engine_rpm'] for s in upshifts) / len(upshifts) if upshifts else 0
        avg_downshift_rpm = sum(s['engine_rpm'] for s in downshifts) / len(downshifts) if downshifts else 0
        
        # Calculate shift frequency
        if len(self.shift_history) >= 2:
            total_time = self.shift_history[-1]['timestamp'] - self.shift_history[0]['timestamp']
            shift_frequency = len(self.shift_history) / total_time if total_time > 0 else 0
        else:
            shift_frequency = 0
        
        # Store performance metrics
        self.strategy_performance = {
            'shifts_analyzed': len(self.shift_history),
            'upshifts': len(upshifts),
            'downshifts': len(downshifts),
            'avg_upshift_rpm': avg_upshift_rpm,
            'avg_downshift_rpm': avg_downshift_rpm,
            'shift_frequency': shift_frequency
        }
        
        return self.strategy_performance
    
    def plot_shift_points(self, engine_rpm_range: List[float], 
                        vehicle_state: Dict = None, save_path: Optional[str] = None):
        """
        Plot the shift points of the strategy.
        
        Args:
            engine_rpm_range: Range of engine RPM to plot
            vehicle_state: Optional vehicle state data for additional context
            save_path: Optional path to save the plot
        """
        plt.figure(figsize=(10, 6))
        
        # If we have gear ratios, we can calculate vehicle speeds
        has_gear_data = (vehicle_state is not None and 
                       'gear_ratios' in vehicle_state and 
                       'wheel_radius' in vehicle_state)
        
        if has_gear_data:
            gear_ratios = vehicle_state['gear_ratios']
            wheel_radius = vehicle_state['wheel_radius']
            final_drive = vehicle_state.get('final_drive_ratio', 1.0)
            
            # Plot speed profile for each gear
            for gear in range(1, len(gear_ratios) + 1):
                speeds = []
                for rpm in engine_rpm_range:
                    # Simple speed calculation: v = ω * r
                    # ω (rad/s) = engine_rpm * 2π / 60 / (gear_ratio * final_drive)
                    wheel_rpm = rpm / (gear_ratios[gear-1] * final_drive)
                    wheel_rads = wheel_rpm * 2 * np.pi / 60
                    speed = wheel_rads * wheel_radius
                    speeds.append(speed * 3.6)  # Convert to km/h
                
                plt.plot(engine_rpm_range, speeds, label=f"Gear {gear}")
        
        # Plot upshift points
        for gear, shift_points in self.upshift_points.items():
            for sp in shift_points:
                if sp.condition_type == ShiftCondition.RPM_THRESHOLD:
                    if has_gear_data and gear <= len(gear_ratios):
                        # Calculate vehicle speed at this RPM
                        wheel_rpm = sp.threshold_value / (gear_ratios[gear-1] * final_drive)
                        wheel_rads = wheel_rpm * 2 * np.pi / 60
                        speed = wheel_rads * wheel_radius * 3.6  # km/h
                        
                        plt.axvline(x=sp.threshold_value, color='g', linestyle='--', alpha=0.7)
                        plt.text(sp.threshold_value + 100, speed, 
                               f"Up {gear}→{sp.target_gear}", 
                               rotation=90, verticalalignment='bottom')
                    else:
                        plt.axvline(x=sp.threshold_value, color='g', linestyle='--', alpha=0.7)
                        plt.text(sp.threshold_value + 100, 0, 
                               f"Up {gear}→{sp.target_gear}", 
                               rotation=90, verticalalignment='bottom')
        
        # Plot downshift points
        for gear, shift_points in self.downshift_points.items():
            for sp in shift_points:
                if sp.condition_type == ShiftCondition.RPM_THRESHOLD:
                    if has_gear_data and gear <= len(gear_ratios):
                        # Calculate vehicle speed at this RPM
                        wheel_rpm = sp.threshold_value / (gear_ratios[gear-1] * final_drive)
                        wheel_rads = wheel_rpm * 2 * np.pi / 60
                        speed = wheel_rads * wheel_radius * 3.6  # km/h
                        
                        plt.axvline(x=sp.threshold_value, color='r', linestyle='--', alpha=0.7)
                        plt.text(sp.threshold_value - 200, speed, 
                               f"Down {gear}→{sp.target_gear}", 
                               rotation=90, verticalalignment='top')
                    else:
                        plt.axvline(x=sp.threshold_value, color='r', linestyle='--', alpha=0.7)
                        plt.text(sp.threshold_value - 200, 0, 
                               f"Down {gear}→{sp.target_gear}", 
                               rotation=90, verticalalignment='top')
        
        plt.xlabel("Engine Speed (RPM)")
        plt.ylabel("Vehicle Speed (km/h)" if has_gear_data else "Value")
        plt.title(f"Shift Strategy: {self.name}")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


class MaxAccelerationStrategy(ShiftStrategy):
    """
    Shift strategy optimized for maximum acceleration.
    
    This strategy is designed to keep the engine operating at or near the peak
    of its power band to maximize acceleration. It typically involves higher
    RPM shift points and more aggressive shifting behavior.
    """
    
    def __init__(self, engine_max_rpm: float, engine_peak_power_rpm: float, 
                gear_ratios: List[float], name: str = "Maximum Acceleration"):
        """
        Initialize max acceleration strategy.
        
        Args:
            engine_max_rpm: Maximum engine RPM
            engine_peak_power_rpm: RPM at peak engine power
            gear_ratios: List of gear ratios
            name: Strategy name
        """
        super().__init__(StrategyType.MAX_ACCELERATION, name)
        
        self.engine_max_rpm = engine_max_rpm
        self.engine_peak_power_rpm = engine_peak_power_rpm
        self.gear_ratios = gear_ratios
        
        # Configure upshift points - typically just below redline
        upshift_rpm = engine_max_rpm * 0.95
        
        # Configure downshift points - typically to keep engine near peak power
        for i in range(1, len(gear_ratios)):
            current_gear = i
            next_gear = i + 1
            
            # Set upshift point for this gear (if not the last gear)
            if current_gear < len(gear_ratios):
                self.add_upshift_point(ShiftPoint(
                    current_gear, next_gear, 
                    ShiftCondition.RPM_THRESHOLD, upshift_rpm,
                    description=f"Upshift from {current_gear} to {next_gear} at {upshift_rpm} RPM"
                ))
            
            # Set downshift point for next gear
            if next_gear <= len(gear_ratios):
                # Calculate RPM after downshift that would put us at peak power
                current_ratio = gear_ratios[current_gear - 1]
                next_ratio = gear_ratios[next_gear - 1]
                downshift_rpm = engine_peak_power_rpm * (next_ratio / current_ratio) * 0.9
                
                self.add_downshift_point(ShiftPoint(
                    next_gear, current_gear,
                    ShiftCondition.RPM_THRESHOLD, downshift_rpm,
                    description=f"Downshift from {next_gear} to {current_gear} at {downshift_rpm} RPM"
                ))
        
        logger.info(f"Max acceleration strategy configured with upshift at {upshift_rpm} RPM")


class MaxEfficiencyStrategy(ShiftStrategy):
    """
    Shift strategy optimized for maximum efficiency.
    
    This strategy is designed to keep the engine operating in its most efficient
    range to maximize fuel economy. It typically involves lower RPM shift points
    and gentler shifting behavior.
    """
    
    def __init__(self, engine_max_rpm: float, engine_peak_torque_rpm: float, 
                gear_ratios: List[float], name: str = "Maximum Efficiency"):
        """
        Initialize max efficiency strategy.
        
        Args:
            engine_max_rpm: Maximum engine RPM
            engine_peak_torque_rpm: RPM at peak engine torque
            gear_ratios: List of gear ratios
            name: Strategy name
        """
        super().__init__(StrategyType.MAX_EFFICIENCY, name)
        
        self.engine_max_rpm = engine_max_rpm
        self.engine_peak_torque_rpm = engine_peak_torque_rpm
        self.gear_ratios = gear_ratios
        
        # Configure upshift points - typically at or just above peak torque
        upshift_rpm = engine_peak_torque_rpm * 1.1
        
        # Configure downshift points - typically to prevent lugging the engine
        min_rpm = 2000  # Prevent engine lugging
        
        for i in range(1, len(gear_ratios)):
            current_gear = i
            next_gear = i + 1
            
            # Set upshift point for this gear (if not the last gear)
            if current_gear < len(gear_ratios):
                self.add_upshift_point(ShiftPoint(
                    current_gear, next_gear, 
                    ShiftCondition.RPM_THRESHOLD, upshift_rpm,
                    description=f"Upshift from {current_gear} to {next_gear} at {upshift_rpm} RPM"
                ))
            
            # Set downshift point for next gear
            if next_gear <= len(gear_ratios):
                # Calculate RPM after downshift that would prevent lugging
                current_ratio = gear_ratios[current_gear - 1]
                next_ratio = gear_ratios[next_gear - 1]
                downshift_rpm = min_rpm * (next_ratio / current_ratio)
                
                self.add_downshift_point(ShiftPoint(
                    next_gear, current_gear,
                    ShiftCondition.RPM_THRESHOLD, downshift_rpm,
                    description=f"Downshift from {next_gear} to {current_gear} at {downshift_rpm} RPM"
                ))
        
        logger.info(f"Max efficiency strategy configured with upshift at {upshift_rpm} RPM")


class EnduranceStrategy(ShiftStrategy):
    """
    Shift strategy optimized for endurance events.
    
    This strategy balances performance and efficiency to maximize the car's
    endurance capabilities. It typically involves moderate RPM shift points
    and smooth shifting behavior to reduce mechanical wear.
    """
    
    def __init__(self, engine_max_rpm: float, engine_peak_power_rpm: float, 
                engine_peak_torque_rpm: float, gear_ratios: List[float],
                name: str = "Endurance"):
        """
        Initialize endurance strategy.
        
        Args:
            engine_max_rpm: Maximum engine RPM
            engine_peak_power_rpm: RPM at peak engine power
            engine_peak_torque_rpm: RPM at peak engine torque
            gear_ratios: List of gear ratios
            name: Strategy name
        """
        super().__init__(StrategyType.ENDURANCE, name)
        
        self.engine_max_rpm = engine_max_rpm
        self.engine_peak_power_rpm = engine_peak_power_rpm
        self.engine_peak_torque_rpm = engine_peak_torque_rpm
        self.gear_ratios = gear_ratios
        
        # For endurance, we want to balance performance and efficiency
        # Upshift earlier than max acceleration but not as early as max efficiency
        upshift_rpm = (engine_peak_power_rpm + engine_peak_torque_rpm) / 2
        
        # Downshift to maintain reasonable torque
        min_rpm = 2500  # Slightly higher than efficiency to maintain responsiveness
        
        for i in range(1, len(gear_ratios)):
            current_gear = i
            next_gear = i + 1
            
            # Set upshift point for this gear (if not the last gear)
            if current_gear < len(gear_ratios):
                self.add_upshift_point(ShiftPoint(
                    current_gear, next_gear, 
                    ShiftCondition.RPM_THRESHOLD, upshift_rpm,
                    description=f"Upshift from {current_gear} to {next_gear} at {upshift_rpm} RPM"
                ))
                
                # Add load-based upshift for better efficiency
                self.add_upshift_point(ShiftPoint(
                    current_gear, next_gear,
                    ShiftCondition.LOAD_THRESHOLD, 0.4,  # Upshift at 40% load
                    priority=2,  # Lower priority than RPM-based
                    description=f"Upshift from {current_gear} to {next_gear} at 40% load"
                ))
            
            # Set downshift point for next gear
            if next_gear <= len(gear_ratios):
                # Calculate RPM after downshift
                current_ratio = gear_ratios[current_gear - 1]
                next_ratio = gear_ratios[next_gear - 1]
                downshift_rpm = min_rpm * (next_ratio / current_ratio)
                
                self.add_downshift_point(ShiftPoint(
                    next_gear, current_gear,
                    ShiftCondition.RPM_THRESHOLD, downshift_rpm,
                    description=f"Downshift from {next_gear} to {current_gear} at {downshift_rpm} RPM"
                ))
                
                # Add load-based downshift for better response
                self.add_downshift_point(ShiftPoint(
                    next_gear, current_gear,
                    ShiftCondition.LOAD_THRESHOLD, 0.8,  # Downshift at 80% load
                    priority=2,  # Lower priority than RPM-based
                    description=f"Downshift from {next_gear} to {current_gear} at 80% load"
                ))
        
        logger.info(f"Endurance strategy configured with balanced shift points")


class AccelerationEventStrategy(ShiftStrategy):
    """
    Shift strategy optimized for acceleration events.
    
    This strategy is specifically tuned for the Formula Student acceleration
    event, which requires maximum straight-line acceleration over a short distance.
    It includes launch control and precisely timed shifts for optimal acceleration.
    """
    
    def __init__(self, engine_max_rpm: float, engine_peak_power_rpm: float, 
                gear_ratios: List[float], wheel_radius: float, vehicle_mass: float,
                name: str = "Acceleration Event"):
        """
        Initialize acceleration event strategy.
        
        Args:
            engine_max_rpm: Maximum engine RPM
            engine_peak_power_rpm: RPM at peak engine power
            gear_ratios: List of gear ratios
            wheel_radius: Wheel radius in meters
            vehicle_mass: Vehicle mass in kg
            name: Strategy name
        """
        super().__init__(StrategyType.ACCELERATION, name)
        
        self.engine_max_rpm = engine_max_rpm
        self.engine_peak_power_rpm = engine_peak_power_rpm
        self.gear_ratios = gear_ratios
        self.wheel_radius = wheel_radius
        self.vehicle_mass = vehicle_mass
        
        # For acceleration event, we want maximum performance
        # Upshift very close to redline
        upshift_rpm = engine_max_rpm * 0.98
        
        # Configure shift points based on expected performance
        # These would typically be tuned based on testing data
        for i in range(1, len(gear_ratios)):
            current_gear = i
            next_gear = i + 1
            
            # Set upshift point for this gear (if not the last gear)
            if current_gear < len(gear_ratios):
                self.add_upshift_point(ShiftPoint(
                    current_gear, next_gear, 
                    ShiftCondition.RPM_THRESHOLD, upshift_rpm,
                    description=f"Upshift from {current_gear} to {next_gear} at {upshift_rpm} RPM"
                ))
        
        # Add launch control parameters
        self.launch_rpm = engine_peak_power_rpm * 0.8
        self.launch_slip_target = 0.2  # Target wheel slip for optimal launch
        
        logger.info(f"Acceleration event strategy configured with upshift at {upshift_rpm} RPM")
    
    def configure_launch_control(self, launch_rpm: float, slip_target: float):
        """
        Configure launch control parameters.
        
        Args:
            launch_rpm: Launch control RPM target
            slip_target: Target wheel slip ratio
        """
        self.launch_rpm = launch_rpm
        self.launch_slip_target = slip_target
        logger.info(f"Launch control configured: {launch_rpm} RPM, {slip_target} slip target")
    
    def get_launch_params(self) -> Dict:
        """
        Get launch control parameters.
        
        Returns:
            Dictionary with launch control parameters
        """
        return {
            'launch_rpm': self.launch_rpm,
            'slip_target': self.launch_slip_target,
            'initial_gear': 1
        }


class StrategyManager:
    """
    Manager class for shift strategies.
    
    This class manages multiple shift strategies and allows switching between
    them based on driving conditions or event requirements.
    """
    
    def __init__(self, default_strategy: ShiftStrategy = None):
        """
        Initialize strategy manager.
        
        Args:
            default_strategy: Default shift strategy to use
        """
        self.strategies = {}
        self.active_strategy = default_strategy
        self.vehicle_state = {}
        
        if default_strategy:
            self.add_strategy(default_strategy)
        
        logger.info("Strategy manager initialized")
    
    def add_strategy(self, strategy: ShiftStrategy):
        """
        Add a strategy to the manager.
        
        Args:
            strategy: ShiftStrategy to add
        """
        self.strategies[strategy.name] = strategy
        logger.info(f"Added strategy: {strategy.name}")
    
    def set_active_strategy(self, strategy_name: str) -> bool:
        """
        Set the active shift strategy.
        
        Args:
            strategy_name: Name of strategy to activate
            
        Returns:
            True if strategy was successfully activated, False otherwise
        """
        if strategy_name in self.strategies:
            self.active_strategy = self.strategies[strategy_name]
            logger.info(f"Activated strategy: {strategy_name}")
            return True
        else:
            logger.warning(f"Strategy not found: {strategy_name}")
            return False
    
    def update_vehicle_state(self, state_updates: Dict):
        """
        Update vehicle state data.
        
        Args:
            state_updates: Dictionary with vehicle state updates
        """
        self.vehicle_state.update(state_updates)
    
    def evaluate_shift(self, current_gear: int, engine_rpm: float, vehicle_speed: float,
                      engine_load: float, throttle_position: float) -> Optional[int]:
        """
        Evaluate whether a shift is needed using the active strategy.
        
        Args:
            current_gear: Current gear
            engine_rpm: Current engine RPM
            vehicle_speed: Current vehicle speed in m/s
            engine_load: Current engine load (0-1)
            throttle_position: Current throttle position (0-1)
            
        Returns:
            Target gear to shift to, or None if no shift is needed
        """
        if self.active_strategy:
            return self.active_strategy.evaluate_shift(
                current_gear, engine_rpm, vehicle_speed, 
                engine_load, throttle_position, self.vehicle_state
            )
        
        return None
    
    def record_shift(self, from_gear: int, to_gear: int, engine_rpm: float, 
                   vehicle_speed: float):
        """
        Record a shift in the active strategy.
        
        Args:
            from_gear: Starting gear
            to_gear: Target gear
            engine_rpm: Engine RPM at shift
            vehicle_speed: Vehicle speed at shift
        """
        if self.active_strategy:
            self.active_strategy.record_shift(
                from_gear, to_gear, engine_rpm, 
                vehicle_speed, time.time()
            )
    
    def get_strategy_performance(self, strategy_name: Optional[str] = None) -> Dict:
        """
        Get performance metrics for a specific strategy or all strategies.
        
        Args:
            strategy_name: Name of strategy to analyze, or None for all strategies
            
        Returns:
            Dictionary with performance metrics
        """
        if strategy_name:
            if strategy_name in self.strategies:
                return self.strategies[strategy_name].analyze_performance()
            else:
                logger.warning(f"Strategy not found: {strategy_name}")
                return {}
        else:
            # Analyze all strategies
            performance = {}
            for name, strategy in self.strategies.items():
                performance[name] = strategy.analyze_performance()
            
            return performance
    
    def get_active_strategy_name(self) -> Optional[str]:
        """
        Get the name of the active strategy.
        
        Returns:
            Name of active strategy, or None if no active strategy
        """
        if self.active_strategy:
            return self.active_strategy.name
        
        return None


def create_formula_student_strategies(
    engine_max_rpm: float, 
    engine_peak_power_rpm: float,
    engine_peak_torque_rpm: float,
    gear_ratios: List[float],
    wheel_radius: float,
    vehicle_mass: float
) -> StrategyManager:
    """
    Create a set of strategies optimized for Formula Student competition.
    
    Args:
        engine_max_rpm: Maximum engine RPM
        engine_peak_power_rpm: RPM at peak engine power
        engine_peak_torque_rpm: RPM at peak engine torque
        gear_ratios: List of gear ratios
        wheel_radius: Wheel radius in meters
        vehicle_mass: Vehicle mass in kg
        
    Returns:
        StrategyManager with Formula Student strategies
    """
    # Create individual strategies
    max_accel = MaxAccelerationStrategy(
        engine_max_rpm, engine_peak_power_rpm, gear_ratios
    )
    
    max_efficiency = MaxEfficiencyStrategy(
        engine_max_rpm, engine_peak_torque_rpm, gear_ratios
    )
    
    endurance = EnduranceStrategy(
        engine_max_rpm, engine_peak_power_rpm, engine_peak_torque_rpm, gear_ratios
    )
    
    acceleration = AccelerationEventStrategy(
        engine_max_rpm, engine_peak_power_rpm, gear_ratios, wheel_radius, vehicle_mass
    )
    
    # Create strategy manager
    manager = StrategyManager(default_strategy=max_accel)
    
    # Add strategies
    manager.add_strategy(max_accel)
    manager.add_strategy(max_efficiency)
    manager.add_strategy(endurance)
    manager.add_strategy(acceleration)
    
    # Create a custom skidpad strategy
    skidpad = ShiftStrategy(StrategyType.SKIDPAD, "Skidpad")
    
    # For skidpad, we want to stay in one gear to maintain consistent handling
    # Typically 2nd or 3rd gear depending on the car
    optimal_skidpad_gear = 2
    
    # Upshift to optimal gear if in lower gear
    for i in range(1, optimal_skidpad_gear):
        skidpad.add_upshift_point(ShiftPoint(
            i, optimal_skidpad_gear,
            ShiftCondition.RPM_THRESHOLD, engine_peak_torque_rpm,
            description=f"Upshift to optimal skidpad gear {optimal_skidpad_gear}"
        ))
    
    # Downshift to optimal gear if in higher gear
    for i in range(optimal_skidpad_gear + 1, len(gear_ratios) + 1):
        skidpad.add_downshift_point(ShiftPoint(
            i, optimal_skidpad_gear,
            ShiftCondition.RPM_THRESHOLD, engine_peak_torque_rpm,
            description=f"Downshift to optimal skidpad gear {optimal_skidpad_gear}"
        ))
    
    manager.add_strategy(skidpad)
    
    return manager


# Example usage
if __name__ == "__main__":
    # Honda CBR600F4i parameters
    engine_max_rpm = 14000
    engine_peak_power_rpm = 12500
    engine_peak_torque_rpm = 10500
    gear_ratios = [2.750, 2.000, 1.667, 1.444, 1.304, 1.208]
    wheel_radius = 0.2286  # 9-inch wheel radius (typical for 13-inch wheels)
    vehicle_mass = 230  # kg, typical for FS car
    
    # Create strategy manager with FS strategies
    strategy_manager = create_formula_student_strategies(
        engine_max_rpm, engine_peak_power_rpm, engine_peak_torque_rpm,
        gear_ratios, wheel_radius, vehicle_mass
    )
    
    # Print active strategy
    active_strategy = strategy_manager.get_active_strategy_name()
    print(f"Active strategy: {active_strategy}")
    
    # Test shift evaluation
    current_gear = 2
    engine_rpm = 13000
    vehicle_speed = 15.0  # m/s
    engine_load = 0.8
    throttle_position = 0.9
    
    target_gear = strategy_manager.evaluate_shift(
        current_gear, engine_rpm, vehicle_speed, 
        engine_load, throttle_position
    )
    
    if target_gear:
        print(f"Shift recommended: {current_gear} -> {target_gear}")
        strategy_manager.record_shift(current_gear, target_gear, engine_rpm, vehicle_speed)
    else:
        print("No shift recommended")
    
    # Switch to endurance strategy
    strategy_manager.set_active_strategy("Endurance")
    print(f"Switched to strategy: {strategy_manager.get_active_strategy_name()}")
    
    # Test shift evaluation with endurance strategy
    target_gear = strategy_manager.evaluate_shift(
        current_gear, engine_rpm, vehicle_speed, 
        engine_load, throttle_position
    )
    
    if target_gear:
        print(f"Shift recommended: {current_gear} -> {target_gear}")
        strategy_manager.record_shift(current_gear, target_gear, engine_rpm, vehicle_speed)
    else:
        print("No shift recommended")
    
    # Plot shift points for active strategy
    vehicle_state = {
        'gear_ratios': gear_ratios,
        'wheel_radius': wheel_radius,
        'final_drive_ratio': 53/14  # 14:53 sprocket ratio
    }
    strategy_manager.update_vehicle_state(vehicle_state)
    
    # Get the active strategy and plot its shift points
    active_strategy_obj = strategy_manager.active_strategy
    active_strategy_obj.plot_shift_points(
        list(range(1000, engine_max_rpm + 1000, 500)),
        vehicle_state
    )