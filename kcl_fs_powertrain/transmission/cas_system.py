"""
Clutch-less Automatic Shifter (CAS) system model.

Simulates the behavior and timing of a CAS system, including ignition cuts,
throttle blips (optional), and shift actuation for rapid gear changes.
"""

import time
import numpy as np
from enum import Enum, auto
import logging
from typing import Dict, List, Tuple, Optional, Union, Callable
import os
import yaml

logger = logging.getLogger("__name__")
# Assuming MotorcycleEngine might be needed for properties like redline
try:
    from ..engine.motorcycle_engine import MotorcycleEngine
except ImportError:
    class MotorcycleEngine: pass # Placeholder
    MotorcycleEngine = None



class ShiftState(Enum):
    """Possible states of the CAS during a shift sequence."""
    IDLE = auto()           # Ready for shift command
    PREPARE_UPSHIFT = auto() # Reducing throttle/preparing cut
    PREPARE_DOWNSHIFT = auto()# Preparing blip/cut
    IGNITION_CUT = auto()    # Ignition is cut
    ACTUATING_SHIFT = auto() # Solenoid/actuator is moving forks
    SHIFT_IN_PROGRESS = auto()
    THROTTLE_BLIP = auto()   # Throttle is blipped (downshift only)
    RECOVERY = auto()        # Ignition/throttle restored, stabilizing
    COOLDOWN = auto()        # Minimum interval between shifts
    ERROR = auto()           # System error


class ShiftDirection(Enum):
    """Direction of the requested shift."""
    UP = 1
    DOWN = -1
    NEUTRAL = 0


class CASSystem:
    """
    Models a Clutch-less Automatic Shifter (CAS) system.
    Simulates timing and control actions for rapid gear changes.
    """
    def __init__(self, gear_ratios: List[float], engine: Optional[MotorcycleEngine] = None,
                 config_path: Optional[str] = None):
        """
        Initialize the CAS system.

        Args:
            gear_ratios: List of transmission gear ratios.
            engine: Optional MotorcycleEngine instance for RPM limits etc.
            config_path: Optional path to YAML config file for CAS parameters.
        """
        self.gear_ratios = gear_ratios
        self.num_gears = len(gear_ratios)
        self.engine = engine # Store reference for potential use (e.g., redline)

        # --- Default Timing Parameters (ms) ---
        self.ignition_cut_time_ms: float = 25.0
        self.shift_actuation_time_ms: float = 18.0
        self.throttle_blip_time_ms: float = 30.0
        self.recovery_time_ms: float = 15.0
        self.prepare_time_ms: float = 5.0 # Time before cut/blip starts

        # --- Default Control Parameters ---
        self.throttle_cut_percent: float = 80.0 # % reduction for upshift
        self.throttle_blip_increase_percent: float = 40.0 # % increase for downshift blip

        # --- Default Constraints ---
        self.min_shift_interval_ms: float = 150.0 # Cooldown between shifts
        self.max_shifts_per_minute: int = 40
        self.overrev_protection_rpm_margin: float = 300 # RPM below redline allowed on downshift

        # Load from config if provided
        if config_path and os.path.exists(config_path):
            self._load_config(config_path)
        elif engine and hasattr(engine, 'config') and 'cas' in engine.config:
             # Try loading from engine's main config if 'cas' section exists
             self._load_config_dict(engine.config.get('cas', {}))

        # State Variables
        self.current_gear: int = 0 # Start in Neutral
        self.system_state: ShiftState = ShiftState.IDLE
        self.last_shift_finish_time_ms: float = -self.min_shift_interval_ms # Allow immediate first shift
        self.shift_start_time_ms: float = 0.0
        self.shift_log: List[Dict] = [] # Store details of each shift
        # Basic shift frequency tracking
        self._shift_timestamps: List[float] = []

        logger.info(f"CAS System initialized. Ignition Cut: {self.ignition_cut_time_ms}ms, Actuation: {self.shift_actuation_time_ms}ms")

    def _load_config(self, config_path: str):
        """Load CAS parameters from YAML file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            cas_config = config.get('cas', {}) # Look for 'cas' section
            self._load_config_dict(cas_config)
            logger.info(f"CAS config loaded from {config_path}")
        except Exception as e:
            logger.error(f"Error loading CAS config from {config_path}: {e}. Using defaults.")

    def _load_config_dict(self, config: Dict):
         """Load CAS parameters from dictionary."""
         self.ignition_cut_time_ms = float(config.get('ignition_cut_time', self.ignition_cut_time_ms))
         self.shift_actuation_time_ms = float(config.get('shift_actuation_time', self.shift_actuation_time_ms))
         self.throttle_blip_time_ms = float(config.get('throttle_blip_time', self.throttle_blip_time_ms))
         self.recovery_time_ms = float(config.get('recovery_time', self.recovery_time_ms))
         self.prepare_time_ms = float(config.get('prepare_time', self.prepare_time_ms))
         self.min_shift_interval_ms = float(config.get('minimum_shift_interval', self.min_shift_interval_ms))
         self.throttle_cut_percent = float(config.get('throttle_cut_percentage', self.throttle_cut_percent))
         self.throttle_blip_increase_percent = float(config.get('throttle_blip_percentage', self.throttle_blip_increase_percent))
         self.max_shifts_per_minute = int(config.get('max_shifts_per_minute', self.max_shifts_per_minute))
         self.overrev_protection_rpm_margin = float(config.get('overrev_protection_margin', self.overrev_protection_rpm_margin))
         # Note: Safety enables like neutral_safety, overrev_protection are usually handled by calling code

    def _check_overrev(self, target_gear: int, current_rpm: float) -> bool:
         """Check if shifting to target_gear would cause overrev."""
         if self.engine is None or current_rpm <= 0 or self.current_gear <= 0 or target_gear >= self.current_gear:
             return False # Cannot check or not a downshift

         if target_gear < 1: return False # Shifting to neutral is fine

         # Calculate expected RPM after downshift
         current_ratio = self.gear_ratios[self.current_gear - 1]
         target_ratio = self.gear_ratios[target_gear - 1]
         expected_rpm = current_rpm * (current_ratio / target_ratio)

         redline = self.engine.redline_rpm
         if expected_rpm > (redline - self.overrev_protection_rpm_margin):
              logger.warning(f"Overrev Protection: Shift {self.current_gear}->{target_gear} rejected. "
                             f"Current RPM {current_rpm:.0f}, Expected RPM {expected_rpm:.0f} > Limit {redline - self.overrev_protection_rpm_margin:.0f}")
              return True # Overrev detected
         return False # No overrev predicted

    def request_shift(self, direction: ShiftDirection, current_rpm: float, target_gear_override: Optional[int] = None) -> bool:
        """
        Request a gear shift. Checks readiness and constraints.
        Updates internal state but does not block. Returns True if shift initiated.
        The calling simulator MUST schedule a completion event.

        Args:
            direction: UP, DOWN, or NEUTRAL.
            current_rpm: The current engine RPM (needed for overrev check). # <-- Added to docstring
            target_gear_override: Optional specific gear number to shift to.

        Returns:
            True if the shift process was initiated, False otherwise.
        """
        current_time_s = time.monotonic() # Use seconds for system time
        # --- Use monotonic time consistently ---
        if not self._check_shift_readiness(current_time_s * 1000.0): # Convert to ms for check
            return False
        # ------------------------------------

        # Determine target gear
        target_gear = target_gear_override
        if target_gear is None:
            if direction == ShiftDirection.UP and self.current_gear < self.num_gears:
                target_gear = self.current_gear + 1
            elif direction == ShiftDirection.DOWN and self.current_gear > 0: # Allow shift from 1 to N
                target_gear = self.current_gear - 1
            elif direction == ShiftDirection.NEUTRAL:
                target_gear = 0
            else:
                logger.debug(f"Shift request ignored: Cannot shift {direction.name} from gear {self.current_gear}.")
                return False # Invalid shift direction from current gear

        # Validate target gear
        if target_gear < 0 or target_gear > self.num_gears:
            logger.warning(f"Shift rejected: Invalid target gear {target_gear}.")
            return False
        if target_gear == self.current_gear:
             logger.debug(f"Shift ignored: Already in gear {target_gear}.")
             return True # No error, just no action needed

        # Check overrev on downshifts using the passed current_rpm
        if direction == ShiftDirection.DOWN:
             # Removed the check for self.engine as current_rpm is now passed directly
             # if self.engine:
             #    current_rpm = getattr(self.engine, 'current_rpm', 0) # No longer needed
             if self._check_overrev(target_gear, current_rpm):
                 return False # Overrev prevented shift

        # --- If all checks pass, initiate shift ---
        self.shift_start_time_s = current_time_s
        self.system_state = ShiftState.SHIFT_IN_PROGRESS # Use a clear state for busy
        self.target_gear_during_shift = target_gear # Store target gear

        # Log timestamp list using seconds
        # (Ensure this list exists - initialize in __init__)
        if not hasattr(self, '_shift_timestamps_s'): self._shift_timestamps_s = []
        # self._shift_timestamps_s.append(current_time_s) # Append when initiating

        logger.info(f"CAS Initiating shift: {self.current_gear} -> {target_gear}")

        # The shift process is initiated. The external simulator is responsible
        # for scheduling the completion event based on get_total_shift_time_ms()
        # and calling complete_shift().
        return True
    def complete_shift(self, current_time_s: float):
        """Mark the current shift as complete and update state."""
        # Ensure consistent time units (seconds)
        if self.system_state == ShiftState.SHIFT_IN_PROGRESS:
             shift_duration_ms = (current_time_s - self.shift_start_time_s) * 1000.0
             from_gear = self.current_gear
             to_gear = getattr(self, 'target_gear_during_shift', -1)
             if to_gear == -1:
                  logger.error("Cannot complete shift: Target gear was not stored.")
                  self.system_state = ShiftState.ERROR
                  return

             self.current_gear = to_gear
             self.system_state = ShiftState.IDLE
             # Use seconds for last shift time
             self.last_shift_finish_time_s = current_time_s
             # Log the shift (RPM at completion might need update logic)
             rpm_at_completion = getattr(self.engine, 'current_rpm', None) if self.engine else None
             self._log_shift(from_gear, to_gear, shift_duration_ms, rpm_at_completion)
             logger.info(f"CAS shift completed: {from_gear} -> {to_gear} in {shift_duration_ms:.1f} ms.")
             if hasattr(self, 'target_gear_during_shift'):
                  delattr(self, 'target_gear_during_shift')
        else:
             logger.warning(f"complete_shift called when not in SHIFT_IN_PROGRESS state (current state: {self.system_state.name}).")


    def _check_shift_readiness(self, current_time_ms: float) -> bool:
         """Check if the system is ready for a new shift command."""
         if self.system_state != ShiftState.IDLE:
             logger.debug(f"Shift rejected: System busy ({self.system_state.name}).")
             return False

         # Use seconds internally now
         current_time_s = current_time_ms / 1000.0
         last_finish_s = getattr(self, 'last_shift_finish_time_s', -float('inf')) # Initialize if needed
         min_interval_s = self.min_shift_interval_ms / 1000.0

         time_since_last_s = current_time_s - last_finish_s
         if time_since_last_s < min_interval_s:
             logger.debug(f"Shift rejected: Cooldown active ({time_since_last_s*1000:.0f} < {self.min_shift_interval_ms:.0f} ms).")
             return False

         # Check shift frequency (shifts in the last 60 seconds)
         sixty_seconds_ago_s = current_time_s - 60.0
         # Ensure timestamp list uses seconds
         if not hasattr(self, '_shift_timestamps_s'): self._shift_timestamps_s = []
         self._shift_timestamps_s = [t for t in self._shift_timestamps_s if t > sixty_seconds_ago_s]
         if len(self._shift_timestamps_s) >= self.max_shifts_per_minute:
             logger.warning(f"Shift rejected: Exceeded max shifts per minute ({self.max_shifts_per_minute}).")
             return False

         return True

    def _log_shift(self, from_gear: int, to_gear: int, duration_ms: float, rpm_at_completion: Optional[float]):
        """Log details of a completed shift."""
        record = {
            'timestamp_s': time.monotonic(), # Use monotonic seconds for log
            'from_gear': from_gear,
            'to_gear': to_gear,
            'duration_ms': duration_ms,
            'engine_rpm_at_completion': rpm_at_completion
        }
        self.shift_log.append(record)
    
    def reset(self):
         """Reset CAS state to initial conditions."""
         self.current_gear = 0 # Start in Neutral
         self.system_state = ShiftState.IDLE
         self.last_shift_finish_time_s = -self.min_shift_interval_ms / 1000.0 # Allow immediate first shift
         self.shift_start_time_s = 0.0
         self.shift_log = []
         self._shift_timestamps_s = []
         if hasattr(self, 'target_gear_during_shift'):
             delattr(self, 'target_gear_during_shift') # Ensure cleared on reset
         logger.info("CAS system state reset.")

    def _log_shift(self, from_gear: int, to_gear: int, duration_ms: float):
         """Log details of a completed shift."""
         record = {
             'timestamp': time.monotonic(),
             'from_gear': from_gear,
             'to_gear': to_gear,
             'duration_ms': duration_ms,
             'engine_rpm': self.engine.current_rpm if self.engine else None # Log RPM at completion
         }
         self.shift_log.append(record)

    def get_total_shift_time_ms(self, direction: ShiftDirection) -> float:
        """Calculate the total theoretical time for a shift."""
        total_time = self.prepare_time_ms + \
                     self.ignition_cut_time_ms + \
                     self.shift_actuation_time_ms + \
                     self.recovery_time_ms
        if direction == ShiftDirection.DOWN:
            total_time += self.throttle_blip_time_ms
        return total_time

    def get_status(self) -> Dict:
        """Get current status and basic stats."""
        current_time_ms = time.monotonic() * 1000.0
        time_since_last = current_time_ms - self.last_shift_finish_time_ms
        ready_to_shift = time_since_last >= self.min_shift_interval_ms and self.system_state == ShiftState.IDLE

        # Recalculate shifts in last minute
        sixty_seconds_ago = current_time_ms - 60000.0
        self._shift_timestamps = [t for t in self._shift_timestamps if t > sixty_seconds_ago]
        shifts_last_minute = len(self._shift_timestamps)

        avg_shift_time = np.mean([s['duration_ms'] for s in self.shift_log]) if self.shift_log else 0

        return {
            'current_gear': self.current_gear,
            'system_state': self.system_state.name,
            'ready_to_shift': ready_to_shift,
            'time_since_last_shift_ms': time_since_last if self.last_shift_finish_time_ms > 0 else None,
            'shifts_last_minute': shifts_last_minute,
            'avg_shift_time_ms': avg_shift_time,
            'total_shifts_logged': len(self.shift_log)
        }

    def reset_statistics(self):
         """Reset shift logs and counters."""
         self.shift_log = []
         self._shift_timestamps = []
         logger.info("CAS statistics reset.")

# Example Usage
if __name__ == "__main__":
    gears = [2.9, 2.1, 1.6, 1.3, 1.1, 0.95] # Example ratios
    # Mock engine for RPM limits
    mock_engine = type('MockEngine', (object,), {'redline_rpm': 13000, 'current_rpm': 8000})()
    cas = CASSystem(gears, mock_engine)

    print("Initial Status:", cas.get_status())

    # Simulate requesting an upshift
    print("\nRequesting UPSHIFT from Neutral...")
    success = cas.request_shift(ShiftDirection.UP, target_gear_override=1) # Specify target for N->1
    print(f"Shift success: {success}")
    print("Status after shift:", cas.get_status())

    # Simulate another upshift too quickly
    print("\nRequesting UPSHIFT immediately...")
    mock_engine.current_rpm = 11000 # Set RPM for readiness check
    success = cas.request_shift(ShiftDirection.UP)
    print(f"Shift success: {success}") # Expected: False (too soon)

    # Wait and try again
    print("\nWaiting for cooldown...")
    time.sleep(cas.min_shift_interval_ms / 1000.0 + 0.01)
    print("Requesting UPSHIFT from 1 to 2...")
    success = cas.request_shift(ShiftDirection.UP)
    print(f"Shift success: {success}")
    print("Status after shift:", cas.get_status())

    # Simulate downshift causing overrev
    print("\nRequesting DOWNSHIFT from 2 to 1 (likely overrev)...")
    mock_engine.current_rpm = 12500 # High RPM in 2nd
    success = cas.request_shift(ShiftDirection.DOWN)
    print(f"Shift success: {success}") # Expected: False (overrev)
    print("Status after failed shift:", cas.get_status())

    # Simulate downshift from higher gear
    print("\nRequesting DOWNSHIFT from 3 to 2...")
    cas.current_gear = 3
    mock_engine.current_rpm = 6000
    success = cas.request_shift(ShiftDirection.DOWN)
    print(f"Shift success: {success}")
    print("Status after shift:", cas.get_status())

    print("\nShift Log:")
    for log in cas.shift_log:
        print(f" - {log['timestamp']:.1f}: {log['from_gear']}->{log['to_gear']} ({log['duration_ms']:.1f}ms)")