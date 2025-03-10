"""
Clutch-less Automatic Shifter (CAS) system for Formula Student powertrain.

This module provides classes and functions for modeling and simulating the
Clutch-less Automatic Shifter (CAS) system used in the KCL Formula Student car.
The CAS system enables rapid, clutch-less gear shifts by momentarily cutting ignition
and controlling the throttle during gear changes, resulting in faster shift times
and improved vehicle performance.

The system includes:
- Electronic throttle control for precise engine management during shifts
- Ignition cut functionality for seamless gear transitions
- Optimized shift point calculation based on engine performance data
- Over-rev protection to prevent engine damage
- Fail-safe neutral engagement for vehicle startup and shutdown
"""

import time
import numpy as np
from enum import Enum, auto
import logging
from typing import Dict, List, Tuple, Optional, Union, Callable

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("CAS_System")


class ShiftState(Enum):
    """Enumeration of possible shift states for the CAS system."""
    IDLE = auto()           # No shifting activity
    PREPARING = auto()       # Preparing for gear shift
    IGNITION_CUT = auto()    # Cutting ignition during shift
    SHIFTING = auto()        # Actuating the shift mechanism
    RECOVERING = auto()      # Restoring normal operation post-shift
    COOLING = auto()         # Cooling period between shifts
    ERROR = auto()           # Error state


class ShiftDirection(Enum):
    """Enumeration of possible shift directions."""
    UP = auto()         # Upshift to higher gear
    DOWN = auto()       # Downshift to lower gear
    NEUTRAL = auto()    # Shift to neutral


class CASSystem:
    """
    Clutch-less Automatic Shifter (CAS) system for Formula Student car.
    
    This class models the behavior of a CAS system, which enables rapid gear shifts
    without using a traditional clutch. The system uses electronic throttle control
    and ignition cut to facilitate smooth gear changes.
    """
    
    def __init__(self, gear_ratios: List[float], engine=None):
        """
        Initialize the CAS system with configuration parameters.
        
        Args:
            gear_ratios: List of gear ratios (ordered from first to last gear)
            engine: Optional reference to the engine object for integration
        """
        # System configuration
        self.gear_ratios = gear_ratios
        self.num_gears = len(gear_ratios)
        self.engine = engine
        
        # Default timing parameters (in milliseconds)
        self.ignition_cut_time = 20  # ms - duration to cut ignition
        self.shift_actuation_time = 15  # ms - time to physically shift gear
        self.throttle_blip_time = 25  # ms - time for throttle blip on downshift
        self.recovery_time = 10  # ms - time to recover after shift
        self.minimum_shift_interval = 200  # ms - minimum time between shifts
        
        # Default actuation parameters
        self.upshift_rpm_threshold = 500  # RPM below redline for upshift
        self.downshift_rpm_threshold = 1500  # RPM above idle for downshift
        self.throttle_cut_percentage = 80  # % reduction during upshift
        self.throttle_blip_percentage = 40  # % increase during downshift
        
        # Default safety parameters
        self.max_shifts_per_minute = 30  # Limit to prevent system overheating
        self.neutral_safety_enabled = True  # Require neutral at startup
        self.overrev_protection_enabled = True  # Prevent shifting that would overrev
        
        # Current state
        self.current_gear = 0  # 0 = neutral, 1-N = gear number
        self.last_shift_time = 0  # Timestamp of last shift
        self.shifts_in_last_minute = 0  # Counter for shift frequency
        self.state = ShiftState.IDLE
        self.error_code = 0
        self.error_message = ""
        
        # Shift optimization data
        self.optimal_shift_points = self._calculate_initial_shift_points()
        
        # Initialize performance metrics
        self.shift_times = []  # List of recorded shift durations
        self.shift_counts = {i: 0 for i in range(self.num_gears + 1)}  # Count shifts by gear
        
        logger.info(f"CAS System initialized with {self.num_gears} gears")
    
    def _calculate_initial_shift_points(self) -> Dict[int, Tuple[int, int]]:
        """
        Calculate initial optimal shift points based on gear ratios.
        This is a simplified placeholder model that would be tuned with real data.
        
        Returns:
            Dictionary mapping gear numbers to (upshift_rpm, downshift_rpm) tuples
        """
        # Default values if no engine data available
        default_max_rpm = 14000 if self.engine is None else self.engine.redline
        default_idle_rpm = 1300 if self.engine is None else self.engine.idle_rpm
        default_peak_torque_rpm = 10500  # Typical for CBR600F4i
        
        # Simple model: upshift near redline, downshift to keep in power band
        shift_points = {}
        
        # For each gear, calculate optimal shift points
        for gear in range(1, self.num_gears + 1):
            # Upshift point: RPM at which to shift up from this gear
            if gear < self.num_gears:
                upshift_rpm = default_max_rpm - self.upshift_rpm_threshold
            else:
                upshift_rpm = default_max_rpm  # No upshift from highest gear
            
            # Downshift point: RPM at which to shift down to this gear from gear+1
            if gear > 1:
                # Calculate RPM after downshift to keep engine in power band
                next_ratio = self.gear_ratios[gear-1]
                current_ratio = self.gear_ratios[gear-2]
                target_rpm = default_peak_torque_rpm
                downshift_rpm = target_rpm * (current_ratio / next_ratio)
            else:
                downshift_rpm = default_idle_rpm + 1000  # Arbitrary for first gear
            
            shift_points[gear] = (upshift_rpm, downshift_rpm)
        
        return shift_points
    
    def update_shift_points(self, engine_torque_curve=None, vehicle_speed=None):
        """
        Update shift points based on current engine performance data and vehicle conditions.
        This would use more sophisticated algorithms in a real implementation.
        
        Args:
            engine_torque_curve: Optional torque curve data for optimizing shift points
            vehicle_speed: Optional current vehicle speed for context
        """
        # This would implement a more complex algorithm using engine performance data
        # to dynamically optimize shift points based on conditions
        # Placeholder for demonstration purposes
        
        if engine_torque_curve is not None:
            logger.info("Updating shift points based on engine torque curve")
            # Example: Find peak torque RPM and set shift points to stay near it
            # Implementation would depend on the format of the torque curve data
        
        if vehicle_speed is not None:
            logger.info(f"Contextualizing shift points for vehicle speed: {vehicle_speed} km/h")
            # Example: Adjust shift points based on vehicle speed (e.g., different
            # strategies for acceleration vs. cruising)
    
    def request_shift(self, direction: ShiftDirection, target_gear: Optional[int] = None) -> bool:
        """
        Request a gear shift in the specified direction.
        
        Args:
            direction: Shift direction (UP, DOWN, or NEUTRAL)
            target_gear: Optional specific gear to shift to
            
        Returns:
            True if shift request was accepted, False otherwise
        """
        current_time = time.time() * 1000  # Current time in ms
        
        # Check if system is ready for another shift
        if self.state != ShiftState.IDLE:
            logger.warning(f"Shift rejected: System busy in state {self.state}")
            return False
        
        # Check minimum interval between shifts
        if current_time - self.last_shift_time < self.minimum_shift_interval:
            logger.warning("Shift rejected: Too soon after previous shift")
            return False
        
        # Check shift frequency limit
        if self.shifts_in_last_minute >= self.max_shifts_per_minute:
            logger.warning("Shift rejected: Maximum shift frequency exceeded")
            return False
        
        # Determine target gear based on direction if not specified
        if target_gear is None:
            if direction == ShiftDirection.UP and self.current_gear < self.num_gears:
                target_gear = self.current_gear + 1
            elif direction == ShiftDirection.DOWN and self.current_gear > 1:
                target_gear = self.current_gear - 1
            elif direction == ShiftDirection.NEUTRAL:
                target_gear = 0
        
        # Validate target gear
        if target_gear is None or target_gear < 0 or target_gear > self.num_gears:
            logger.warning(f"Shift rejected: Invalid target gear {target_gear}")
            return False
        
        # Check for unnecessary shift
        if target_gear == self.current_gear:
            logger.info(f"Shift ignored: Already in gear {target_gear}")
            return True  # Not an error, just no action needed
        
        # Check for overrev protection
        if self.overrev_protection_enabled and self.engine is not None:
            if direction == ShiftDirection.DOWN:
                # Calculate engine RPM after downshift
                speed_factor = self.gear_ratios[self.current_gear-1] / self.gear_ratios[target_gear-1]
                new_rpm = self.engine.current_rpm * speed_factor
                
                if new_rpm > self.engine.redline:
                    logger.warning(f"Shift rejected: Downshift would cause overrev to {new_rpm} RPM")
                    return False
        
        logger.info(f"Shift request accepted: {direction.name} to gear {target_gear}")
        # Execute the shift
        return self._execute_shift(direction, target_gear)
    
    def _execute_shift(self, direction: ShiftDirection, target_gear: int) -> bool:
        """
        Execute the actual gear shift operation.
        
        Args:
            direction: Shift direction (UP, DOWN, or NEUTRAL)
            target_gear: Target gear to shift to
            
        Returns:
            True if shift was successful, False otherwise
        """
        shift_start_time = time.time() * 1000  # Start time in ms
        
        try:
            # 1. Prepare for shift
            self.state = ShiftState.PREPARING
            
            # 2. Adjust throttle based on shift direction
            if direction == ShiftDirection.UP:
                self._reduce_throttle()
            elif direction == ShiftDirection.DOWN:
                # For downshifts, we'll blip the throttle later during the shift sequence
                pass
            
            # 3. Cut ignition
            self.state = ShiftState.IGNITION_CUT
            self._cut_ignition()
            time.sleep(self.ignition_cut_time / 1000)  # Convert ms to seconds
            
            # 4. Actuate the shift
            self.state = ShiftState.SHIFTING
            
            # For downshifts, blip the throttle to match revs
            if direction == ShiftDirection.DOWN:
                self._blip_throttle()
                time.sleep(self.throttle_blip_time / 1000)
            
            # Simulating the physical gear change
            time.sleep(self.shift_actuation_time / 1000)
            
            # 5. Recover normal operation
            self.state = ShiftState.RECOVERING
            self._restore_ignition()
            self._restore_throttle()
            time.sleep(self.recovery_time / 1000)
            
            # 6. Update state
            self.current_gear = target_gear
            self.state = ShiftState.IDLE
            
            # 7. Record metrics
            shift_end_time = time.time() * 1000
            shift_duration = shift_end_time - shift_start_time
            self.shift_times.append(shift_duration)
            self.shift_counts[target_gear] += 1
            self.shifts_in_last_minute += 1
            self.last_shift_time = shift_end_time
            
            logger.info(f"Shift completed: now in gear {self.current_gear}, took {shift_duration:.1f}ms")
            return True
            
        except Exception as e:
            self.state = ShiftState.ERROR
            self.error_code = 1
            self.error_message = f"Shift error: {str(e)}"
            logger.error(self.error_message)
            
            # Try to restore normal operation
            self._restore_ignition()
            self._restore_throttle()
            return False
    
    def _reduce_throttle(self):
        """Reduce throttle for upshift."""
        if self.engine is not None:
            current_throttle = self.engine.throttle_position
            reduced_throttle = current_throttle * (1 - self.throttle_cut_percentage / 100)
            logger.debug(f"Reducing throttle from {current_throttle:.2f} to {reduced_throttle:.2f}")
            # In a real implementation, this would command the engine throttle
        else:
            logger.debug("Simulating throttle reduction")
    
    def _blip_throttle(self):
        """Blip throttle for downshift."""
        if self.engine is not None:
            current_throttle = self.engine.throttle_position
            blipped_throttle = min(1.0, current_throttle + self.throttle_blip_percentage / 100)
            logger.debug(f"Blipping throttle from {current_throttle:.2f} to {blipped_throttle:.2f}")
            # In a real implementation, this would command the engine throttle
        else:
            logger.debug("Simulating throttle blip")
    
    def _restore_throttle(self):
        """Restore normal throttle operation."""
        logger.debug("Restoring throttle control")
        # In a real implementation, this would restore normal throttle control
    
    def _cut_ignition(self):
        """Cut ignition for shift."""
        logger.debug("Cutting ignition")
        # In a real implementation, this would send a signal to the engine control
        # to temporarily interrupt ignition
    
    def _restore_ignition(self):
        """Restore ignition after shift."""
        logger.debug("Restoring ignition")
        # In a real implementation, this would send a signal to restore ignition
    
    def engage_neutral(self) -> bool:
        """
        Engage neutral gear (safety function for startup/shutdown).
        
        Returns:
            True if neutral was successfully engaged, False otherwise
        """
        return self.request_shift(ShiftDirection.NEUTRAL)
    
    def get_optimal_rpm_for_shift(self, gear: int, direction: ShiftDirection) -> Optional[int]:
        """
        Get the optimal RPM for shifting from the current gear.
        
        Args:
            gear: Current gear
            direction: Shift direction
            
        Returns:
            Optimal RPM for shift or None if not applicable
        """
        if gear < 1 or gear > self.num_gears:
            return None
        
        if direction == ShiftDirection.UP and gear < self.num_gears:
            return self.optimal_shift_points[gear][0]
        elif direction == ShiftDirection.DOWN and gear > 1:
            return self.optimal_shift_points[gear-1][1]
        
        return None
    
    def should_shift(self, current_rpm: float, current_throttle: float) -> Optional[ShiftDirection]:
        """
        Determine if a shift should be made based on current conditions.
        
        Args:
            current_rpm: Current engine RPM
            current_throttle: Current throttle position (0-1)
            
        Returns:
            Recommended shift direction or None if no shift recommended
        """
        # Don't shift if in neutral or if throttle is very low (likely preparing to stop)
        if self.current_gear == 0 or current_throttle < 0.1:
            return None
        
        # Check for upshift condition
        if (self.current_gear < self.num_gears and 
            current_rpm >= self.optimal_shift_points[self.current_gear][0] and
            current_throttle > 0.5):  # Only upshift under load
            return ShiftDirection.UP
        
        # Check for downshift condition
        if (self.current_gear > 1 and 
            current_rpm <= self.optimal_shift_points[self.current_gear-1][1] and
            current_throttle > 0.3):  # Only downshift with some throttle applied
            return ShiftDirection.DOWN
        
        return None
    
    def get_system_status(self) -> Dict:
        """
        Get the current status of the CAS system.
        
        Returns:
            Dictionary with current CAS system status
        """
        avg_shift_time = sum(self.shift_times) / len(self.shift_times) if self.shift_times else 0
        
        return {
            'current_gear': self.current_gear,
            'state': self.state.name,
            'shifts_in_last_minute': self.shifts_in_last_minute,
            'last_shift_time': self.last_shift_time,
            'average_shift_time_ms': avg_shift_time,
            'shift_counts': self.shift_counts,
            'error_code': self.error_code,
            'error_message': self.error_message
        }
    
    def simulate_shift(self, direction: ShiftDirection, engine_rpm: float) -> Dict:
        """
        Simulate a shift for testing without actual hardware.
        
        Args:
            direction: Shift direction
            engine_rpm: Current engine RPM
            
        Returns:
            Dictionary with simulation results
        """
        target_gear = None
        if direction == ShiftDirection.UP and self.current_gear < self.num_gears:
            target_gear = self.current_gear + 1
        elif direction == ShiftDirection.DOWN and self.current_gear > 1:
            target_gear = self.current_gear - 1
        elif direction == ShiftDirection.NEUTRAL:
            target_gear = 0
        
        # Calculate expected RPM after shift
        expected_rpm = None
        if target_gear != 0 and self.current_gear != 0:  # Not involving neutral
            if direction == ShiftDirection.UP:
                ratio_change = self.gear_ratios[self.current_gear-1] / self.gear_ratios[target_gear-1]
                expected_rpm = engine_rpm * ratio_change
            elif direction == ShiftDirection.DOWN:
                ratio_change = self.gear_ratios[self.current_gear-1] / self.gear_ratios[target_gear-1]
                expected_rpm = engine_rpm * ratio_change
        
        # Simulate the shift
        shift_success = self.request_shift(direction, target_gear)
        
        return {
            'requested_direction': direction.name,
            'initial_gear': self.current_gear if not shift_success else target_gear,
            'target_gear': target_gear,
            'initial_rpm': engine_rpm,
            'expected_rpm': expected_rpm,
            'shift_success': shift_success,
            'shift_time_ms': self.shift_times[-1] if shift_success and self.shift_times else None
        }
    
    def reset_shift_statistics(self):
        """Reset shift statistics and counters."""
        self.shift_times = []
        self.shift_counts = {i: 0 for i in range(self.num_gears + 1)}
        self.shifts_in_last_minute = 0
        logger.info("Shift statistics reset")
    
    def update(self, dt: float):
        """
        Update the system state based on elapsed time.
        
        Args:
            dt: Time elapsed since last update in seconds
        """
        # Update shift frequency counter
        if self.shifts_in_last_minute > 0:
            # Simple decay of shift counter over time (60 seconds)
            decay_rate = dt / 60.0
            self.shifts_in_last_minute = max(0, self.shifts_in_last_minute - decay_rate * self.max_shifts_per_minute)
        
        # Add cooling period if needed
        if self.state == ShiftState.COOLING:
            if time.time() * 1000 - self.last_shift_time > self.minimum_shift_interval:
                self.state = ShiftState.IDLE
                logger.debug("Cooling period complete, system ready")


# Example usage
if __name__ == "__main__":
    # Define gear ratios for Honda CBR600F4i (example values)
    gear_ratios = [2.750, 2.000, 1.667, 1.444, 1.304, 1.208]
    
    # Create CAS system
    cas = CASSystem(gear_ratios)
    
    # Simulate a series of shifts
    print("Simulating upshifts through all gears...")
    
    # Start in first gear
    cas.current_gear = 1
    
    # Simulate sequential upshifts
    for i in range(1, len(gear_ratios)):
        # Simulate engine at optimal shift point
        engine_rpm = cas.get_optimal_rpm_for_shift(i, ShiftDirection.UP)
        
        print(f"\nAt {engine_rpm} RPM in gear {i}:")
        result = cas.simulate_shift(ShiftDirection.UP, engine_rpm)
        
        print(f"  Shifted from gear {i} to {result['target_gear']}")
        print(f"  Engine RPM changed from {result['initial_rpm']} to approximately {result['expected_rpm']:.0f}")
        print(f"  Shift took {result['shift_time_ms']:.1f}ms")
        
        # Add a small delay between shifts
        time.sleep(0.3)
    
    print("\nSystem status after all shifts:")
    status = cas.get_system_status()
    for key, value in status.items():
        print(f"  {key}: {value}")