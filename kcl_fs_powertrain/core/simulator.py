"""
Core simulation engine for time-stepping vehicle dynamics.

Provides the base Simulator class responsible for managing the simulation loop,
time steps, event handling, state integration, and data logging. Event-specific
simulators (like Acceleration, LapTime) will build upon or utilize this core simulator.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
import logging
import time
import heapq # For priority queue event system
from enum import Enum, auto
import yaml
import os
import pandas as pd
from scipy.integrate import solve_ivp # For potentially more advanced integration

# Import necessary components from the package
try:
    from .vehicle import Vehicle
    from .track import Track
    from ..utils.constants import GRAVITY
except ImportError:
    # Fallbacks for standalone execution or testing
    class Vehicle: pass
    class Track: pass
    GRAVITY = 9.81
    logger = logging.getLogger("Simulator_Fallback")
    logger.warning("Could not import core vehicle/track components. Using fallbacks.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("CoreSimulator")

# --- Enums and Helper Classes ---

class IntegrationMethod(Enum):
    """Numerical integration methods."""
    EULER = auto()
    RK4 = auto() # 4th Order Runge-Kutta
    # SCIPY_RK45 = auto() # Using SciPy's adaptive RK45

class EventType(Enum):
    """Types of discrete events during simulation."""
    SIMULATION_END = auto() # Signal to stop the simulation
    TARGET_TIME_REACHED = auto()
    TARGET_DISTANCE_REACHED = auto()
    GEAR_SHIFT_REQUEST = auto() # Request from strategy
    GEAR_SHIFT_COMPLETE = auto() # Notification from CAS/Transmission
    THERMAL_WARNING = auto()
    THERMAL_CRITICAL = auto()
    CUSTOM = auto()

class EnvironmentConditions:
    """Represents environmental conditions."""
    def __init__(self, ambient_temp_C: float = 25.0, air_pressure_Pa: float = 101325.0,
                 air_density_kg_m3: Optional[float] = None):
        self.ambient_temp_C = ambient_temp_C
        self.air_pressure_Pa = air_pressure_Pa
        # Calculate density if not provided
        self.air_density_kg_m3 = air_density_kg_m3 if air_density_kg_m3 is not None else self._calculate_air_density()

    def _calculate_air_density(self) -> float:
        """Calculate air density using ideal gas law (simplified)."""
        R_specific = 287.058 # J/(kg*K)
        temp_K = self.ambient_temp_C + 273.15
        density = self.air_pressure_Pa / (R_specific * temp_K)
        return density

class ControlInputs:
    """Represents driver/control system inputs."""
    def __init__(self, throttle: float = 0.0, brake: float = 0.0,
                 steering_rad: float = 0.0, gear_request: Optional[int] = None):
        self.throttle = np.clip(throttle, 0.0, 1.0)
        self.brake = np.clip(brake, 0.0, 1.0)
        self.steering_rad = steering_rad # Radians
        self.gear_request = gear_request # Target gear, None means hold current

class SimulationEvent:
    """Represents a discrete event to be processed by the simulator."""
    def __init__(self, event_type: EventType, time: float,
                 data: Optional[Dict] = None, priority: int = 0):
        self.event_type = event_type
        self.time = time
        self.data = data if data else {}
        self.priority = priority # Higher value = higher priority if times are equal

    def __lt__(self, other: 'SimulationEvent') -> bool:
        """Comparison for heapq (min-heap based on time, max-heap on priority)."""
        if self.time == other.time:
            return self.priority > other.priority # Higher priority processed first at same time
        return self.time < other.time

    def __repr__(self) -> str:
        return f"Event({self.event_type.name} @ {self.time:.4f}s, Prio={self.priority}, Data={self.data})"


class DataLogger:
    """Logs simulation data at specified intervals."""
    def __init__(self, variables: List[str], log_interval_s: float = 0.05):
        """
        Args:
            variables: List of variable paths (e.g., 'vehicle.current_speed_mps', 'time').
            log_interval_s: Time interval between logging entries (s).
        """
        self.variables = variables
        self.log_interval_s = max(1e-6, log_interval_s) # Ensure positive interval
        self.data = {var: [] for var in variables}
        self._last_log_time = -np.inf # Ensure first step is logged

    def log(self, current_time: float, state_data: Dict):
        """Log data if the interval has passed."""
        if current_time >= self._last_log_time + self.log_interval_s:
            for var in self.variables:
                value = self._get_value_from_path(state_data, var)
                self.data[var].append(value)
            self._last_log_time = current_time

    def _get_value_from_path(self, data_dict: Dict, path: str) -> Any:
        """Retrieve nested value using dot notation."""
        keys = path.split('.')
        value = data_dict
        try:
            for key in keys:
                 if isinstance(value, dict):
                     value = value.get(key)
                 elif hasattr(value, key):
                      value = getattr(value, key)
                 else:
                      logger.warning(f"Could not resolve path '{path}' at key '{key}' in logger.")
                      return None # Or np.nan
                 if value is None: break # Stop if any part of path is None
            return value
        except Exception as e:
             logger.warning(f"Error resolving path '{path}' in logger: {e}")
             return None

    def get_dataframe(self) -> pd.DataFrame:
        """Return logged data as a pandas DataFrame."""
        # Ensure all lists have the same length (important if logging stopped early)
        min_len = min(len(v) for v in self.data.values()) if self.data else 0
        data_truncated = {k: v[:min_len] for k, v in self.data.items()}
        return pd.DataFrame(data_truncated)

    def clear(self):
        """Clear logged data."""
        self.data = {var: [] for var in self.variables}
        self._last_log_time = -np.inf

# --- Core Simulator ---

class Simulator:
    """Core time-stepping simulation engine."""
    def __init__(self, vehicle: Vehicle, track: Optional[Track] = None,
                 environment: Optional[EnvironmentConditions] = None,
                 config: Optional[Dict] = None):
        """
        Args:
            vehicle: The vehicle instance to simulate.
            track: Optional track instance for path-following sims.
            environment: Optional environment conditions instance.
            config: Optional simulation configuration dictionary.
        """
        self.vehicle = vehicle
        self.track = track
        self.environment = environment or EnvironmentConditions() # Default environment
        self.config = config or {} # Store config

        # --- Simulation State ---
        self.current_time_s: float = 0.0
        self.step_count: int = 0
        self.simulation_stop_reason: Optional[str] = None

        # --- Simulation Parameters ---
        self.dt_base: float = 0.01 # Base time step
        self.adaptive_stepping: bool = False
        self.min_dt: float = 0.001
        self.max_dt: float = 0.05
        self.integration_method: IntegrationMethod = IntegrationMethod.RK4

        # --- Event Management ---
        self.event_queue: List[SimulationEvent] = []
        self.event_handlers: Dict[EventType, List[Callable]] = {e_type: [] for e_type in EventType}

        # --- Control ---
        # Control inputs can be set externally or by a controller/strategy object
        self.current_controls = ControlInputs()
        # TODO: Add mechanism for external controller/driver model to set inputs

        # --- Data Logging ---
        self.loggers: List[DataLogger] = []
        self._setup_default_logger() # Add a default logger

        # --- Initialization ---
        self.reset() # Initialize state

        logger.info("Core Simulator initialized.")

    def configure(self, config: Dict):
         """Apply configuration settings from a dictionary."""
         self.config.update(config)
         sim_cfg = self.config.get('simulation', {})
         self.dt_base = float(sim_cfg.get('time_step', self.dt_base))
         self.adaptive_stepping = bool(sim_cfg.get('adaptive_stepping', self.adaptive_stepping))
         self.min_dt = float(sim_cfg.get('min_time_step', self.min_dt))
         self.max_dt = float(sim_cfg.get('max_time_step', self.max_dt))
         method_str = sim_cfg.get('integration_method', self.integration_method.name).upper()
         self.integration_method = IntegrationMethod[method_str] if method_str in IntegrationMethod.__members__ else IntegrationMethod.RK4

         env_cfg = self.config.get('environment', {})
         self.environment = EnvironmentConditions(**env_cfg)

         log_cfg = self.config.get('logging', {})
         if 'default_logger' in log_cfg:
              self.loggers = [] # Clear existing loggers
              logger_cfg = log_cfg['default_logger']
              self.add_data_logger(
                   variables=logger_cfg.get('variables', []),
                   log_interval_s=1.0 / logger_cfg.get('sampling_rate', 20.0) # Convert Hz to interval
              )
         logger.info("Simulator configured.")

    def _setup_default_logger(self):
         """Setup a default logger with common vehicle states."""
         default_vars = [
             'time', 'step',
             'vehicle.current_speed_mps', 'vehicle.current_acceleration_mpss',
             'vehicle.current_position_m', 'vehicle.current_gear',
             'vehicle.current_engine_rpm', 'vehicle.throttle_input', 'vehicle.brake_input',
             'vehicle.engine_temperature', 'vehicle.coolant_temperature', 'vehicle.oil_temperature',
             'vehicle.thermal_factor'
         ]
         self.add_data_logger(default_vars, log_interval_s=0.05) # Log at 20 Hz

    def add_data_logger(self, variables: List[str], log_interval_s: float = 0.05) -> DataLogger:
         """Add a data logger."""
         logger_instance = DataLogger(variables, log_interval_s)
         self.loggers.append(logger_instance)
         return logger_instance

    def register_event_handler(self, event_type: EventType, handler: Callable[[SimulationEvent], None]):
        """Register a callback function for a specific event type."""
        self.event_handlers[event_type].append(handler)
        logger.debug(f"Registered handler for {event_type.name}")

    def _queue_event(self, event: SimulationEvent):
        """Add an event to the priority queue."""
        heapq.heappush(self.event_queue, event)
        logger.debug(f"Queued event: {event}")

    def _process_events(self):
        """Process all events scheduled at or before the current time."""
        while self.event_queue and self.event_queue[0].time <= self.current_time_s:
            event = heapq.heappop(self.event_queue)
            logger.debug(f"Processing event: {event}")

            # Call registered handlers
            if event.event_type in self.event_handlers:
                for handler in self.event_handlers[event.event_type]:
                    try:
                        handler(event)
                    except Exception as e:
                        logger.error(f"Error in event handler for {event.event_type.name}: {e}", exc_info=True)

            # Handle built-in simulation control events
            if event.event_type in [EventType.SIMULATION_END, EventType.TARGET_TIME_REACHED, EventType.TARGET_DISTANCE_REACHED]:
                 self.simulation_stop_reason = event.data.get('reason', event.event_type.name)
                 logger.info(f"Simulation stop triggered by event: {self.simulation_stop_reason}")
                 # The main run loop will check self.simulation_stop_reason

            # Handle gear shift completion if CAS system modeled with events
            if event.event_type == EventType.GEAR_SHIFT_COMPLETE:
                 target_gear = event.data.get('target_gear')
                 if target_gear is not None and hasattr(self.vehicle, 'cas_system') and self.vehicle.cas_system:
                      # Finalize gear change in vehicle model and CAS state
                      self.vehicle.cas_system.complete_shift(self.current_time_s)
                      # Vehicle's current_gear is now updated by cas_system.complete_shift
                      logger.info(f"Gear shift completed to gear {self.vehicle.current_gear}")
                 else:
                      logger.warning("GEAR_SHIFT_COMPLETE event missing target_gear data or CAS system.")

    def _calculate_time_step(self) -> float:
         """Determine the time step for the next iteration."""
         if not self.adaptive_stepping:
             return self.dt_base

         # Adaptive step calculation (simplified example)
         # Base step on speed and acceleration magnitude
         speed = self.vehicle.current_speed_mps
         accel = abs(self.vehicle.current_acceleration_mpss)

         # Smaller steps at low speed or high acceleration
         if speed < 1.0 or accel > 10.0:
             dt = self.min_dt
         elif speed < 5.0 or accel > 5.0:
             dt = max(self.min_dt, self.dt_base * 0.5)
         else:
              # Increase step size slightly during steady conditions
              dt = min(self.max_dt, self.dt_base * 1.2)

         # Ensure dt doesn't overshoot the next event time
         if self.event_queue:
              next_event_time = self.event_queue[0].time
              dt = min(dt, next_event_time - self.current_time_s + 1e-9) # Add epsilon

         return max(self.min_dt, dt) # Ensure dt is at least min_dt

    def _integrate_state(self, dt: float):
        """Integrate the vehicle state over dt using the chosen method."""
        # Euler integration is performed by calling the vehicle's update method.
        # This method calculates forces based on the *start* of the step state
        # and updates velocity/position based on that constant acceleration over dt.
        if self.integration_method == IntegrationMethod.EULER:
            # Ensure ambient temp is passed correctly
            self.vehicle.update_vehicle_state(dt, self.environment.ambient_temp_C)
        elif self.integration_method == IntegrationMethod.RK4:
             # RK4 requires a state vector and derivative function, which is complex
             # to implement generically here without strict state management in Vehicle.
             # Falling back to Euler via vehicle.update_vehicle_state.
             logger.log(logging.DEBUG if self.step_count % 100 != 0 else logging.WARNING, # Log warning periodically
                        "RK4 integration selected but not implemented in CoreSimulator; using Euler via vehicle.update_vehicle_state.")
             self.vehicle.update_vehicle_state(dt, self.environment.ambient_temp_C)
        else:
             logger.error(f"Unsupported integration method: {self.integration_method}")
             # Fallback to Euler
             self.vehicle.update_vehicle_state(dt, self.environment.ambient_temp_C)
    def _log_simulation_data(self):
         """Log data from the current simulation state."""
         # Create a snapshot of the state for logging
         current_state_data = {
             'time': self.current_time_s,
             'step': self.step_count,
             'vehicle': self.vehicle, # Pass the whole object or specific attributes
             'environment': self.environment,
             'controls': self.current_controls
             # Add track properties if track exists
         }
         if self.track and hasattr(self.track, 'get_properties_at_distance'):
              # Need to calculate current track distance
              # Placeholder: Use vehicle's longitudinal position
              track_props = self.track.get_properties_at_distance(self.vehicle.current_position_m)
              current_state_data['track'] = track_props

         for logger_instance in self.loggers:
             logger_instance.log(self.current_time_s, current_state_data)


    def step(self) -> bool:
        """
        Execute a single simulation step.

        Returns:
            True if simulation should continue, False if stop condition met.
        """
        if self.simulation_stop_reason: return False # Stop if already flagged

        # 1. Process Events at current time
        # This handles scheduled events like shift completions, simulation end times etc.
        self._process_events()
        if self.simulation_stop_reason: return False # Event might trigger stop

        # 2. Determine Time Step
        dt = self._calculate_time_step()
        if dt <= 0: # Avoid getting stuck
             logger.warning(f"Zero or negative dt ({dt:.2e}) calculated at time {self.current_time_s:.3f}s. Stopping.")
             self.simulation_stop_reason = "Zero dt"
             return False

        # 3. Get Control Inputs (Assume self.current_controls is set externally)
        # Check for gear change request
        requested_gear = self.current_controls.gear_request
        if requested_gear is not None and requested_gear != self.vehicle.current_gear:
             # Check if CAS is currently shifting
             can_initiate_shift = True
             if self.vehicle.cas_system and self.vehicle.cas_system.system_state != ShiftState.IDLE:
                  can_initiate_shift = False
                  logger.debug(f"Ignoring gear request {requested_gear}: CAS is busy ({self.vehicle.cas_system.system_state.name})")

             if can_initiate_shift:
                 # Initiate gear change via vehicle method
                 success, shift_duration_s = self.vehicle.change_gear(requested_gear)
                 if success and shift_duration_s > 0:
                      # Schedule a GEAR_SHIFT_COMPLETE event
                      completion_time = self.current_time_s + shift_duration_s
                      self._queue_event(SimulationEvent(EventType.GEAR_SHIFT_COMPLETE, completion_time,
                                                       {'target_gear': requested_gear}))
                      logger.debug(f"Scheduled GEAR_SHIFT_COMPLETE for gear {requested_gear} at {completion_time:.3f}s")
                 # Reset the request regardless of success to avoid repeated attempts
                 self.current_controls.gear_request = None

        # Apply current controls to vehicle state (needed before integration)
        self.vehicle.throttle_input = self.current_controls.throttle
        self.vehicle.brake_input = self.current_controls.brake
        # Steering might be handled differently depending on dynamics model
        # self.vehicle.steering_angle_rad = self.current_controls.steering_rad

        # 4. Integrate Vehicle State
        # The vehicle's update method handles physics integration
        # It updates speed, position, RPM, temps based on current state and dt
        self._integrate_state(dt)

        # 5. Log Data
        self._log_simulation_data()

        # 6. Advance Time
        self.current_time_s += dt
        self.step_count += 1

        # 7. Check basic stop conditions (e.g., negative speed error)
        if self.vehicle.current_speed_mps < -0.1:
            logger.error(f"Simulation stopped: Negative speed detected ({self.vehicle.current_speed_mps:.2f} m/s)")
            self.simulation_stop_reason = "Negative Speed"
            return False

        return True # Continue simulation

    def run(self, duration_s: Optional[float] = None, distance_m: Optional[float] = None,
            stop_condition: Optional[Callable[[], bool]] = None) -> Dict:
        """
        Run simulation until a stopping condition is met.

        Args:
            duration_s: Maximum simulation time (seconds).
            distance_m: Maximum distance to simulate (meters).
            stop_condition: Custom function returning True to stop simulation.

        Returns:
            Dictionary containing simulation results (e.g., from loggers).
        """
        logger.info("Starting simulation run...")
        # Reset state if starting a new run (optional, maybe handled externally)
        # self.reset()

        # --- Setup Stop Conditions ---
        # Add events for time/distance limits
        if duration_s is not None:
             self._queue_event(SimulationEvent(EventType.TARGET_TIME_REACHED, duration_s, {'reason': f'Duration {duration_s}s reached'}))
        if distance_m is not None and self.track is None: # Only use distance if no track (simple accel)
             self._queue_event(SimulationEvent(EventType.TARGET_DISTANCE_REACHED, float('inf'), {'target_dist': distance_m, 'reason': f'Distance {distance_m}m reached'}))
             # Distance event needs dynamic time, handled in step/event check

        start_real_time = time.monotonic()

        # --- Simulation Loop ---
        while True:
             # Check distance condition here as it depends on state
             if distance_m is not None and self.vehicle.current_position_m >= distance_m:
                  self.simulation_stop_reason = f"Distance {distance_m}m reached"
                  logger.info(self.simulation_stop_reason)
                  break # Exit loop

             # Check custom stop condition
             if stop_condition and stop_condition():
                  self.simulation_stop_reason = "Custom stop condition met"
                  logger.info(self.simulation_stop_reason)
                  break

             # Execute step and check if it signals stop
             if not self.step():
                  break

        # --- Finalize ---
        end_real_time = time.monotonic()
        sim_duration = end_real_time - start_real_time
        logger.info(f"Simulation run finished. Reason: {self.simulation_stop_reason or 'Completed'}")
        logger.info(f" Simulated Time: {self.current_time_s:.3f} s")
        logger.info(f" Real Time Elapsed: {sim_duration:.3f} s")
        logger.info(f" Steps: {self.step_count}")

        return self.get_results()


    def reset(self):
        """Reset simulation state to initial conditions."""
        self.current_time_s = 0.0
        self.step_count = 0
        self.simulation_stop_reason = None
        self.event_queue = []
        # Reset vehicle state (position, speed, gear etc.)
        if self.vehicle:
            self.vehicle.current_speed_mps = 0.0
            self.vehicle.current_acceleration_mpss = 0.0
            self.vehicle.current_position_m = 0.0
            self.vehicle.change_gear(0) # Start in Neutral
            self.vehicle.current_engine_rpm = self.vehicle.engine.idle_rpm if self.vehicle.engine else 0.0
            # Reset temperatures to ambient
            self.vehicle.engine_temperature = self.environment.ambient_temp_C
            self.vehicle.coolant_temperature = self.environment.ambient_temp_C
            self.vehicle.oil_temperature = self.environment.ambient_temp_C
            self.vehicle.thermal_factor = 1.0
        # Reset controls
        self.current_controls = ControlInputs()
        # Clear loggers
        for logger_instance in self.loggers:
            logger_instance.clear()
        logger.info("Simulator state reset.")

    def get_results(self) -> Dict:
        """Compile and return results, primarily from loggers."""
        results = {
            'final_time_s': self.current_time_s,
            'steps': self.step_count,
            'stop_reason': self.simulation_stop_reason,
            'log_data': {}
        }
        if self.loggers:
             # Combine data from all loggers if multiple exist? Or just use first?
             # For now, assume first logger is primary
             results['log_data'] = self.loggers[0].get_dataframe()
        return results

    def export_results(self, filepath: str, logger_index: int = 0, format: str = 'csv'):
         """Export results from a specific logger."""
         if logger_index < 0 or logger_index >= len(self.loggers):
              logger.error(f"Invalid logger index: {logger_index}")
              return False
         try:
              df = self.loggers[logger_index].get_dataframe()
              output_dir = os.path.dirname(filepath)
              if output_dir: os.makedirs(output_dir, exist_ok=True)

              if format.lower() == 'csv':
                   df.to_csv(filepath, index=False, float_format='%.5f')
              elif format.lower() == 'json':
                   df.to_json(filepath, orient='records', indent=2)
              # Add other formats like parquet if needed
              else:
                   logger.error(f"Unsupported export format: {format}")
                   return False
              logger.info(f"Logger {logger_index} data exported to {filepath} ({format})")
              return True
         except Exception as e:
              logger.error(f"Failed to export logger data: {e}")
              return False

# Example usage (illustrative)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Core Simulator Demo")
    print("-" * 20)
    try:
        # Create a vehicle (requires necessary sub-modules to be available)
        from .vehicle import create_formula_student_vehicle
        vehicle = create_formula_student_vehicle()

        # Create simulator
        simulator = Simulator(vehicle)

        # Configure simulation
        simulator.configure({'simulation': {'time_step': 0.01}})

        # --- Example: Simple Acceleration Run ---
        print("\nSimulating 5 seconds of acceleration...")
        simulator.reset()
        simulator.current_controls = ControlInputs(throttle=1.0, gear_request=1) # Full throttle, 1st gear

        def stop_condition():
             # Example custom stop condition
             if simulator.vehicle.current_speed_mps > 30: # Stop if speed exceeds 30 m/s
                  simulator.simulation_stop_reason = "Speed > 30 m/s"
                  return True
             return False

        results = simulator.run(duration_s=5.0, stop_condition=stop_condition)

        print(f"\nSimulation Finished. Stop Reason: {results['stop_reason']}")
        print(f" Final Time: {results['final_time_s']:.3f} s")
        print(f" Final Speed: {vehicle.current_speed_mps * MS_TO_KMH:.1f} km/h")
        print(f" Final Position: {vehicle.current_position_m:.1f} m")

        # Export results
        output_dir = "../../plots/core_demo" # Relative path for example
        os.makedirs(output_dir, exist_ok=True)
        simulator.export_results(os.path.join(output_dir, "accel_demo_log.csv"))

        # Plot basic results
        if not results['log_data'].empty:
             df = results['log_data']
             fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 8))
             axes[0].plot(df['time'], df['vehicle.current_speed_mps'] * MS_TO_KMH)
             _apply_common_ax_settings(axes[0], ylabel='Speed (km/h)', title='Simulator Demo: Acceleration')
             axes[1].plot(df['time'], df['vehicle.current_position_m'])
             _apply_common_ax_settings(axes[1], xlabel='Time (s)', ylabel='Position (m)')
             plt.tight_layout()
             save_plot(fig, "demo_simulator_run", directory=output_dir)
             # plt.show()


    except ImportError as e:
        print(f"\nError: Could not import necessary modules for demo ({e}).")
    except FileNotFoundError as e:
         print(f"\nError: Config file not found. {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()