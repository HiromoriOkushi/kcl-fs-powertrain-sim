"""
Advanced lap time optimization using numerical methods.

Implements optimization routines to find the minimum lap time by adjusting
racing line and control inputs (throttle, brake, gear), using Runge-Kutta
integration for vehicle dynamics.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
import logging
import time
import os
import yaml
from scipy.optimize import minimize, Bounds
from scipy.interpolate import interp1d, CubicSpline

# Import core components
try:
    from ..core.vehicle import Vehicle
    from ..core.track_integration import TrackProfile
    from ..utils.track_utils import preprocess_track_points, ensure_unique_values
    from ..utils.constants import GRAVITY, KW_TO_HP
    from ..utils.plotting import plot_racing_line_analysis, save_plot, _apply_common_ax_settings
except ImportError:
    # Fallbacks
    GRAVITY = 9.81; KW_TO_HP=1.341
    class Vehicle: pass
    class TrackProfile: pass
    def preprocess_track_points(d): return d
    def ensure_unique_values(x): return x
    def plot_racing_line_analysis(*args, **kwargs): plt.figure(); plt.plot([0,1]); plt.title("Fallback Plot"); plt.show(); plt.close(); return plt.gcf()
    def save_plot(fig, path, **kwargs): pass
    def _apply_common_ax_settings(ax, **kwargs): pass
    logger = logging.getLogger("OptimalLapTime_Fallback")
    logger.warning("Could not import all necessary modules. Using fallbacks.")


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("OptimalLapTime")


class VehicleState:
    """Represents the detailed state vector for RK4 integration."""
    # State vector indices
    _X, _Y, _HEADING, _VX, _VY, _YAW_RATE, _ENG_RPM = 0, 1, 2, 3, 4, 5, 6
    _COOLANT_TEMP, _OIL_TEMP, _ENG_BLOCK_TEMP = 7, 8, 9 # Optional thermal states
    _STATE_SIZE_BASE = 7
    _STATE_SIZE_THERMAL = 10

    def __init__(self, size: int):
        self.state = np.zeros(size)
        self.size = size

    @property
    def x(self): return self.state[self._X]
    @x.setter
    def x(self, value): self.state[self._X] = value

    @property
    def y(self): return self.state[self._Y]
    @y.setter
    def y(self, value): self.state[self._Y] = value

    @property
    def heading(self): return self.state[self._HEADING]
    @heading.setter
    def heading(self, value): self.state[self._HEADING] = value

    @property
    def vx(self): return self.state[self._VX]
    @vx.setter
    def vx(self, value): self.state[self._VX] = value

    @property
    def vy(self): return self.state[self._VY]
    @vy.setter
    def vy(self, value): self.state[self._VY] = value

    @property
    def yaw_rate(self): return self.state[self._YAW_RATE]
    @yaw_rate.setter
    def yaw_rate(self, value): self.state[self._YAW_RATE] = value

    @property
    def engine_rpm(self): return self.state[self._ENG_RPM]
    @engine_rpm.setter
    def engine_rpm(self, value): self.state[self._ENG_RPM] = value

    @property
    def coolant_temp(self): return self.state[self._COOLANT_TEMP] if self.size > self._STATE_SIZE_BASE else 25.0
    @coolant_temp.setter
    def coolant_temp(self, value):
        if self.size > self._STATE_SIZE_BASE: self.state[self._COOLANT_TEMP] = value

    @property
    def oil_temp(self): return self.state[self._OIL_TEMP] if self.size > self._STATE_SIZE_BASE else 25.0
    @oil_temp.setter
    def oil_temp(self, value):
        if self.size > self._STATE_SIZE_BASE: self.state[self._OIL_TEMP] = value

    @property
    def engine_block_temp(self): return self.state[self._ENG_BLOCK_TEMP] if self.size > self._STATE_SIZE_BASE else 25.0
    @engine_block_temp.setter
    def engine_block_temp(self, value):
         if self.size > self._STATE_SIZE_BASE: self.state[self._ENG_BLOCK_TEMP] = value

    @property
    def speed(self): return np.sqrt(self.vx**2 + self.vy**2)

    def get_vector(self) -> np.ndarray:
        return self.state

    @classmethod
    def from_vector(cls, vector: np.ndarray) -> 'VehicleState':
        state = cls(len(vector))
        state.state = vector.copy()
        return state


class ControlInputs:
    """Represents control inputs (throttle, brake, steering, gear)."""
    def __init__(self, throttle: float = 0.0, brake: float = 0.0, steering: float = 0.0, gear: int = 1):
        self.throttle = np.clip(throttle, 0.0, 1.0)
        self.brake = np.clip(brake, 0.0, 1.0)
        self.steering = steering # Radians
        self.gear = int(round(gear)) # Ensure integer gear

    def to_vector(self) -> np.ndarray:
        return np.array([self.throttle, self.brake, self.steering, float(self.gear)])

    @classmethod
    def from_vector(cls, vector: np.ndarray) -> 'ControlInputs':
        return cls(vector[0], vector[1], vector[2], int(round(vector[3])))


class OptimalLapTimeOptimizer:
    """Optimizes lap time using numerical integration and optimization."""

    def __init__(self, vehicle: Vehicle, track_profile: TrackProfile, config: Optional[Dict] = None):
        """
        Args:
            vehicle: Vehicle model instance.
            track_profile: TrackProfile instance.
            config: Optional configuration dictionary for optimization settings.
        """
        self.vehicle = vehicle
        self.track_profile = track_profile
        self.track_data = self._load_and_preprocess_track(track_profile)
        if self.track_data is None:
             raise ValueError("Failed to load or preprocess track data.")

        # --- Default Settings ---
        self.dt: float = 0.02 # Time step for integration
        self.max_sim_time: float = 180.0 # Max time allowed for lap
        self.include_thermal: bool = True # Include engine thermal effects
        self.optimization_method: str = 'SLSQP' # SciPy optimization method
        self.max_iterations: int = 30 # Max iterations for optimizer
        self.tolerance: float = 1e-3 # Optimizer tolerance
        self.num_control_points: int = 50 # Number of points to parameterize line/controls

        # --- Vehicle Dynamics Params ---
        self.tire_model_type: str = 'simple' # Placeholder for tire model choice
        self.max_lat_g: float = 2.0
        self.max_lon_accel_g: float = 1.5
        self.max_braking_g: float = 1.8
        self.control_rate_limit = {'throttle': 8.0, 'brake': 12.0, 'steering': 15.0} # Units/sec

        # --- Track Parameterization ---
        self.track_position_limit: float = 0.95 # Max deviation from centerline (-1 to 1)

        # --- Load Configuration ---
        if config:
             self._load_config(config)

        # Initialize track interpolators
        self._initialize_track_interpolation()

        # State size depends on thermal model inclusion
        self.state_size = VehicleState._STATE_SIZE_THERMAL if self.include_thermal else VehicleState._STATE_SIZE_BASE

        # Results
        self.optimal_params: Optional[np.ndarray] = None
        self.optimal_lap_time: Optional[float] = None
        self.optimal_states: Optional[List[Dict]] = None # Store full state history
        self.optimal_racing_line: Optional[np.ndarray] = None

        logger.info("OptimalLapTimeOptimizer initialized.")
        logger.info(f" State size: {self.state_size}, Num Control Points: {self.num_control_points}")

    def _load_and_preprocess_track(self, track_profile: TrackProfile) -> Optional[Dict]:
         """Loads and preprocesses track data from TrackProfile."""
         # This mirrors the logic in LapTimeSimulator for consistency
         try:
             data = track_profile.get_track_data()
             if not data or 'points' not in data or len(data['points']) < 3: return None
             processed_data = preprocess_track_points(data)
             if 'points' not in processed_data or 'distance' not in processed_data: return None
             if 'curvature' not in processed_data or len(processed_data['curvature']) != len(processed_data['points']):
                  logger.info("Calculating track curvature for optimization...")
                  points = processed_data['points']
                  dx = np.gradient(points[:, 0]); dy = np.gradient(points[:, 1])
                  d2x = np.gradient(dx); d2y = np.gradient(dy)
                  curv = np.abs(dx * d2y - dy * d2x) / np.maximum((dx**2 + dy**2)**1.5, 1e-9)
                  processed_data['curvature'] = curv
             return processed_data
         except Exception as e:
             logger.error(f"Error loading/preprocessing track for optimizer: {e}")
             return None

    def _load_config(self, config: Dict):
        """Load settings from configuration dictionary."""
        opt_cfg = config.get('optimization', {})
        self.dt = float(opt_cfg.get('dt', self.dt))
        self.max_sim_time = float(opt_cfg.get('max_time', self.max_sim_time))
        self.include_thermal = bool(opt_cfg.get('include_thermal', self.include_thermal))
        self.optimization_method = opt_cfg.get('method', self.optimization_method)
        self.max_iterations = int(opt_cfg.get('max_iterations', self.max_iterations))
        self.tolerance = float(opt_cfg.get('tolerance', self.tolerance))
        self.num_control_points = int(opt_cfg.get('num_control_points', self.num_control_points))

        dyn_cfg = config.get('vehicle_dynamics', {})
        self.tire_model_type = dyn_cfg.get('tire_model', self.tire_model_type)
        self.max_lat_g = float(dyn_cfg.get('max_lateral_accel_g', self.max_lat_g))
        self.max_lon_accel_g = float(dyn_cfg.get('max_longitudinal_accel_g', self.max_lon_accel_g))
        self.max_braking_g = float(dyn_cfg.get('max_braking_decel_g', self.max_braking_g))
        self.control_rate_limit['throttle'] = float(dyn_cfg.get('throttle_rate_limit', self.control_rate_limit['throttle']))
        # ... load other rate limits ...

        line_cfg = config.get('racing_line', {})
        self.track_position_limit = float(line_cfg.get('track_position_limit', self.track_position_limit))
        logger.info("Optimizer configuration loaded.")

    def _initialize_track_interpolation(self):
        """Create interpolation functions for track properties."""
        dist = ensure_unique_values(self.track_data['distance'])
        points = self.track_data['points']
        width = self.track_data.get('width', np.full(len(dist), 3.0))
        curvature = self.track_data.get('curvature', np.zeros(len(dist)))

        # Ensure arrays match the unique distances length
        if len(points) != len(dist): points = interp1d(self.track_data['distance'], points, axis=0)(dist)
        if len(width) != len(dist): width = interp1d(self.track_data['distance'], width)(dist)
        if len(curvature) != len(dist): curvature = interp1d(self.track_data['distance'], curvature)(dist)


        # Close the loop for periodic splines
        loop_dist = dist[-1] + (dist[1] - dist[0]) # Estimate next point distance
        dist_loop = np.append(dist, loop_dist)
        points_loop = np.vstack([points, points[1]]) # Use second point as approx for first after loop
        width_loop = np.append(width, width[1])
        curvature_loop = np.append(curvature, curvature[1])


        interp_kind = 'cubic' # Use cubic for smoother track representation
        self.track_x_interp = CubicSpline(dist_loop, points_loop[:, 0], bc_type='periodic')
        self.track_y_interp = CubicSpline(dist_loop, points_loop[:, 1], bc_type='periodic')
        self.track_width_interp = interp1d(dist_loop, width_loop, kind='linear', fill_value="extrapolate") # Width can be linear
        self.track_curvature_interp = interp1d(dist_loop, curvature_loop, kind='linear', fill_value="extrapolate") # Curvature linear

        self.track_length = dist[-1] # Original track length
        logger.debug("Track interpolation functions created.")

    def _vehicle_dynamics_derivatives(self, t: float, y: np.ndarray,
                                     control_func: Callable[[float], ControlInputs],
                                     vehicle: Vehicle) -> np.ndarray:
        """
        Calculate state derivatives dy/dt = f(t, y, u).

        Args:
            t: Current time (not explicitly used in this simplified model).
            y: Current state vector (from VehicleState).
            control_func: Function that returns ControlInputs for time t.
            vehicle: The Vehicle object instance.

        Returns:
            Array of state derivatives.
        """
        state = VehicleState.from_vector(y)
        controls = control_func(t) # Get controls for this time

        # --- Extract State ---
        x, y, heading, vx, vy, yaw_rate, eng_rpm = state.state[:VehicleState._STATE_SIZE_BASE]
        speed = state.speed

        # --- Calculate Forces (Simplified Point Mass + Basic Aero/Tire) ---
        F_tractive = 0.0
        F_brake = 0.0
        F_cornering = 0.0
        torque_engine = 0.0

        # Longitudinal Forces
        if controls.gear > 0 and vehicle.drivetrain:
            # Calculate engine torque (consider thermal effects if enabled)
            current_temp = state.engine_block_temp if self.include_thermal else 90.0
            torque_engine = vehicle.engine.get_torque(eng_rpm, controls.throttle, current_temp)
            # Calculate total wheel torque
            torque_wheel_total = vehicle.drivetrain.calculate_total_wheel_torque(torque_engine, controls.gear)
            # Tractive force
            F_tractive = torque_wheel_total / vehicle.wheel_radius_m if vehicle.wheel_radius_m > 0 else 0

        # Braking force
        # Estimate max braking force based on grip limit (e.g., 1.8g)
        max_brake_force = vehicle.mass * self.max_braking_g * GRAVITY
        F_brake = controls.brake * max_brake_force

        # Rolling resistance
        F_rolling = vehicle.mass * GRAVITY * vehicle.rolling_resistance

        # Aerodynamic Drag
        F_drag = 0.5 * AIR_DENSITY_SEA_LEVEL * vehicle.drag_coefficient * vehicle.frontal_area_m2 * speed**2

        # Total Longitudinal Force (in vehicle frame)
        Fx_vehicle = F_tractive - F_brake - F_rolling - F_drag

        # Lateral Forces (Simplified Bicycle Model)
        # Calculate front/rear slip angles based on steering, yaw rate, vx, vy
        # For simplicity, use steering input to generate a proportional lateral force, limited by grip
        if speed > 0.5: # Need some speed for steering to work
             # Estimate lateral force demand from steering input
             # Max lateral force based on current speed
             max_lat_force = vehicle.mass * self.max_lat_g * GRAVITY # Static limit here, refine later
             downforce = vehicle.cornering.calculate_downforce_N(speed)
             max_lat_force_dynamic = vehicle.cornering.calculate_max_lateral_acceleration(speed) * vehicle.mass

             # Simplified steering model: Fy = Stiffness * steer_angle * factor
             # Assume steering input directly relates to desired lateral g for simplicity here
             # A proper model needs slip angles.
             # Let's assume steering input maps roughly to lateral acceleration demand.
             # Steering limit +/- 15 deg (approx +/- 0.26 rad)
             max_steer_rad = 0.26
             lat_accel_demand = (controls.steering / max_steer_rad) * max_lat_force_dynamic / vehicle.mass
             Fy_vehicle = vehicle.mass * lat_accel_demand
             # Clamp by actual max lateral force
             Fy_vehicle = np.clip(Fy_vehicle, -max_lat_force_dynamic, max_lat_force_dynamic)
        else:
             Fy_vehicle = 0.0

        # --- Calculate Accelerations (Global Frame) ---
        cos_h, sin_h = np.cos(heading), np.sin(heading)
        ax_global = (Fx_vehicle * cos_h - Fy_vehicle * sin_h) / vehicle.mass
        ay_global = (Fx_vehicle * sin_h + Fy_vehicle * cos_h) / vehicle.mass

        # --- Calculate State Derivatives ---
        dydt = np.zeros(self.state_size)
        dydt[self._X] = vx
        dydt[self._Y] = vy
        dydt[self._HEADING] = yaw_rate
        dydt[self._VX] = ax_global # Assuming ax_global is dvx/dt
        dydt[self._VY] = ay_global # Assuming ay_global is dvy/dt

        # Yaw rate dynamics (simplified: yaw_rate proportional to lat_accel/speed)
        if speed > 0.5:
            dydt[self._YAW_RATE] = Fy_vehicle / (vehicle.mass * speed) # Simplified yaw acceleration
        else:
            dydt[self._YAW_RATE] = 0.0

        # Engine RPM dynamics (simplified: tries to match wheel speed via drivetrain)
        if controls.gear > 0 and vehicle.drivetrain:
            target_rpm = vehicle.drivetrain.calculate_engine_speed_rpm(speed, controls.gear)
            # Clamp target RPM
            target_rpm = np.clip(target_rpm, vehicle.engine.idle_rpm, vehicle.engine.redline_rpm)
            # First order lag towards target rpm
            rpm_tau = 0.1 # Time constant for RPM change
            dydt[self._ENG_RPM] = (target_rpm - eng_rpm) / rpm_tau
        else: # Neutral or invalid gear
             dydt[self._ENG_RPM] = -(eng_rpm - vehicle.engine.idle_rpm) / 0.5 # Decay to idle

        # --- Thermal Dynamics (Optional) ---
        if self.include_thermal:
            # Use the EngineHeatModel from the vehicle
            if hasattr(vehicle, 'heat_model') and vehicle.heat_model:
                # Calculate heat generation
                heat_gen = vehicle.heat_model.calculate_heat_generation(eng_rpm, controls.throttle)
                # Calculate transfers (need ambient temp, coolant flow - estimate them)
                ambient_temp = 25.0 # Get from environment later
                coolant_flow = 50.0 # Estimate
                # Need cooling system effectiveness calculation here
                cooling_effectiveness = 0.7 # Estimate

                temps = {'engine': state.engine_block_temp, 'oil': state.oil_temp, 'coolant': state.coolant_temp}
                heat_transfer = vehicle.heat_model.calculate_internal_heat_transfer(temps)
                ambient_loss = vehicle.heat_model.calculate_ambient_heat_loss(temps, ambient_temp, speed)
                radiator_rejection = cooling_effectiveness * (temps['coolant'] - ambient_temp) * 150 # Simplified radiator heat rejection

                # Calculate net heat flows
                q_net_engine = heat_gen['to_ambient'] - heat_transfer['oil_to_block'] - heat_transfer['coolant_to_block'] - ambient_loss['block_to_ambient']
                q_net_oil = heat_gen['to_oil'] + heat_transfer['oil_to_block'] - ambient_loss['oil_to_ambient']
                q_net_coolant = heat_gen['to_coolant'] + heat_transfer['coolant_to_block'] - radiator_rejection

                # Calculate temperature derivatives dT/dt = Q / C
                capacities = vehicle.thermal_config.get_thermal_capacities()
                # Add external coolant capacity estimate
                total_coolant_capacity = capacities.get('coolant_engine', 1e-3) + 1.5 * 3800 # Engine + external estimate
                dydt[self._ENG_BLOCK_TEMP] = q_net_engine / max(1e-3, capacities.get('engine_block', 1e-3))
                dydt[self._OIL_TEMP] = q_net_oil / max(1e-3, capacities.get('engine_oil', 1e-3))
                dydt[self._COOLANT_TEMP] = q_net_coolant / max(1e-3, total_coolant_capacity)
            else:
                # No detailed thermal model, keep temps constant or use simple update
                dydt[self._ENG_BLOCK_TEMP] = 0.0
                dydt[self._OIL_TEMP] = 0.0
                dydt[self._COOLANT_TEMP] = 0.0


        return dydt


    def _integrate_rk4(self, y0: np.ndarray, t0: float, dt: float,
                     control_func: Callable[[float], ControlInputs],
                     vehicle: Vehicle) -> np.ndarray:
        """Integrate one step using RK4."""
        f = lambda t, y: self._vehicle_dynamics_derivatives(t, y, control_func, vehicle)

        k1 = dt * f(t0, y0)
        k2 = dt * f(t0 + 0.5*dt, y0 + 0.5*k1)
        k3 = dt * f(t0 + 0.5*dt, y0 + 0.5*k2)
        k4 = dt * f(t0 + dt, y0 + k3)

        y1 = y0 + (k1 + 2*k2 + 2*k3 + k4) / 6.0
        return y1

    def _create_control_interpolators(self, parameters: np.ndarray) -> Tuple[Callable, Callable, Callable]:
         """Create interpolation functions for controls based on distance."""
         num_pts = self.num_control_points
         control_distances = np.linspace(0, self.track_length, num_pts)

         throttle_params = np.clip(parameters[num_pts : 2*num_pts], 0.0, 1.0)
         brake_params = np.clip(parameters[2*num_pts : 3*num_pts], 0.0, 1.0)
         steering_params = np.clip(parameters[3*num_pts : 4*num_pts], -0.7, 0.7) # Limit steer

         # Use linear interpolation for controls between points
         interp_kind = 'linear'
         throttle_func = interp1d(control_distances, throttle_params, kind=interp_kind, bounds_error=False, fill_value=(throttle_params[0], throttle_params[-1]))
         brake_func = interp1d(control_distances, brake_params, kind=interp_kind, bounds_error=False, fill_value=(brake_params[0], brake_params[-1]))
         steering_func = interp1d(control_distances, steering_params, kind=interp_kind, bounds_error=False, fill_value=(steering_params[0], steering_params[-1]))

         return throttle_func, brake_func, steering_func

    def _parameterize_racing_line(self, parameters: np.ndarray) -> Callable[[float], Tuple[float, float]]:
        """Create racing line interpolation function from parameters."""
        num_pts = self.num_control_points
        control_distances = np.linspace(0, self.track_length, num_pts)
        track_positions = np.clip(parameters[0 : num_pts], -self.track_position_limit, self.track_position_limit)

        # Interpolate track positions along the full track length
        position_func = interp1d(control_distances, track_positions, kind='cubic', bounds_error=False, fill_value='extrapolate')

        def get_racing_line_point(s: float) -> Tuple[float, float]:
             """Get x, y coordinates on racing line for distance s."""
             s_norm = s % self.track_length # Wrap distance for closed track

             x_center = self.track_x_interp(s_norm)
             y_center = self.track_y_interp(s_norm)
             width = self.track_width_interp(s_norm)
             track_pos = position_func(s_norm) # Interpolated position offset

             # Calculate normal vector
             ds_small = 0.1
             x_next = self.track_x_interp(s_norm + ds_small)
             y_next = self.track_y_interp(s_norm + ds_small)
             tx, ty = x_next - x_center, y_next - y_center
             norm = np.sqrt(tx**2 + ty**2)
             if norm > 1e-6: tx /= norm; ty /= norm
             nx, ny = -ty, tx # Normal vector

             # Calculate point
             rx = x_center + track_pos * (width / 2.0) * nx
             ry = y_center + track_pos * (width / 2.0) * ny
             return rx, ry

        return get_racing_line_point


    def _simulate_lap_with_params(self, parameters: np.ndarray) -> Tuple[float, List[VehicleState], np.ndarray]:
        """
        Simulate a lap using RK4 integration with parameterized line and controls.

        Args:
            parameters: Combined vector [track_pos_params, throttle_params, brake_params, steering_params].

        Returns:
            Tuple (lap_time, states_history, racing_line_points)
        """
        num_pts = self.num_control_points
        if len(parameters) != 4 * num_pts:
             raise ValueError(f"Incorrect number of parameters. Expected {4*num_pts}, got {len(parameters)}")

        # Create interpolators for controls and racing line
        throttle_func, brake_func, steering_func = self._create_control_interpolators(parameters)
        racing_line_func = self._parameterize_racing_line(parameters)

        # --- Simulation Setup ---
        state = VehicleState(self.state_size)
        # Initial conditions (start of track, minimal speed)
        state.x, state.y = racing_line_func(0.0)
        x_next, y_next = racing_line_func(0.1) # Point slightly ahead for heading
        state.heading = np.arctan2(y_next - state.y, x_next - state.x)
        state.vx = 0.1 * np.cos(state.heading) # Start with minimal speed
        state.vy = 0.1 * np.sin(state.heading)
        state.engine_rpm = self.vehicle.engine.idle_rpm
        state.gear = 1
        # Initialize temps near ambient if thermal included
        if self.include_thermal:
             state.coolant_temp = 30.0; state.oil_temp = 30.0; state.engine_block_temp = 30.0

        t = 0.0
        dt = self.dt
        distance_travelled = 0.0 # Distance along the *actual path driven*
        track_distance_laps = 0.0 # Distance along the reference track centerline (for controls)
        states_history = [state]
        racing_line_points_driven = [[state.x, state.y]] # Store actual path

        max_steps = int(self.max_sim_time / dt)

        # --- Simulation Loop ---
        for step in range(max_steps):
            # 1. Determine current distance along reference track for controls
            # Find closest point on centerline to current x,y
            # This is computationally expensive, maybe approximate based on previous step?
            # Simple approximation: assume distance travelled matches track distance increment
            track_distance_laps += state.speed * dt
            track_distance_controls = track_distance_laps % self.track_length

            # 2. Get Controls for this distance/time
            controls = ControlInputs(
                throttle=throttle_func(track_distance_controls),
                brake=brake_func(track_distance_controls),
                steering=steering_func(track_distance_controls),
                # Basic gear logic (improve with strategy manager)
                gear=self._estimate_optimal_gear(state.speed, state.engine_rpm)
            )
            # Prevent simultaneous throttle/brake
            if controls.throttle > 0.1 and controls.brake > 0.1:
                controls.brake = 0.0 # Prioritize throttle

            # 3. Integrate state using RK4
            try:
                # Pass the control function that depends on time (or distance if controls are parameterized by distance)
                # Here we pass a lambda that gets controls based on the *current* track distance estimate
                current_controls = controls # Use controls calculated for this step
                y_next = self._integrate_rk4(state.get_vector(), t, dt, lambda time: current_controls, self.vehicle)
                next_state = VehicleState.from_vector(y_next)
            except Exception as e:
                logger.error(f"RK4 integration failed at t={t:.3f}s: {e}", exc_info=False)
                return self.max_sim_time * 2, states_history, np.array(racing_line_points_driven) # Penalty

            # 4. Update distance travelled and time
            step_dist = np.sqrt((next_state.x - state.x)**2 + (next_state.y - state.y)**2)
            distance_travelled += step_dist
            t += dt

            # 5. Update state and store history
            state = next_state
            states_history.append(state)
            racing_line_points_driven.append([state.x, state.y])

            # 6. Check Lap Completion
            # Check if we have crossed the start/finish line *after* travelling most of the track
            # Requires defining the start line more formally
            if distance_travelled > self.track_length * 0.9:
                # Simple check: Have we gotten close to the start point again?
                dist_to_start = np.sqrt((state.x - racing_line_points_driven[0][0])**2 + (state.y - racing_line_points_driven[0][1])**2)
                if dist_to_start < 1.0 and step > 100: # Close to start and not immediately after starting
                     logger.info(f"Lap completed at t={t:.3f}s, distance={distance_travelled:.1f}m")
                     # Optional: Interpolate exact crossing time
                     return t, states_history, np.array(racing_line_points_driven)

        # If loop finishes due to max time or steps
        logger.warning(f"Lap simulation did not complete within limits. Time: {t:.2f}s, Dist: {distance_travelled:.1f}m")
        return self.max_sim_time * 2, states_history, np.array(racing_line_points_driven) # Penalty

    def _objective_function(self, parameters: np.ndarray) -> float:
        """Objective function: Minimize lap time."""
        lap_time, _, _ = self._simulate_lap_with_params(parameters)

        # Add penalty for invalid simulations (e.g., going off track - needs constraints)
        # For now, large lap times act as penalty.
        # Add constraints to the optimizer instead.

        logger.debug(f"Objective function evaluated. Lap Time: {lap_time:.4f}")
        return lap_time

    def _create_constraints(self) -> List[Dict]:
         """Create constraints for the optimization problem."""
         constraints = []
         num_params_per_type = self.num_control_points
         num_total_params = 4 * num_params_per_type

         # --- Add constraints here ---
         # Example: Constraint on maximum steering rate change between control points
         # This requires defining how steering params map to actual steering rates
         # For now, let's add a simple constraint on the difference between adjacent steering params

         # Constraint: Limit rate of change for steering
         # |steering[i+1] - steering[i]| <= max_rate * delta_s
         max_steer_rate_per_dist = 0.1 # rad/m (example)
         delta_s = self.track_length / (self.num_control_points - 1)
         max_steer_diff = max_steer_rate_per_dist * delta_s

         steering_start_idx = 3 * num_params_per_type
         for i in range(num_params_per_type - 1):
             idx1 = steering_start_idx + i
             idx2 = steering_start_idx + i + 1
             # Constraint: steering[i+1] - steering[i] <= max_steer_diff
             A_upper = np.zeros(num_total_params); A_upper[idx2] = 1; A_upper[idx1] = -1
             constraints.append(LinearConstraint(A_upper, -np.inf, max_steer_diff))
             # Constraint: steering[i] - steering[i+1] <= max_steer_diff
             A_lower = np.zeros(num_total_params); A_lower[idx1] = 1; A_lower[idx2] = -1
             constraints.append(LinearConstraint(A_lower, -np.inf, max_steer_diff))

         # --- Add constraints for Throttle/Brake Rate ---
         # Similar logic can be applied to throttle and brake parameters

         logger.info(f"Created {len(constraints)} constraints for optimization.")
         return constraints


    def optimize_lap_time(self) -> Dict:
        """Run the numerical optimization to find the minimum lap time."""
        n_track = self.num_control_points
        n_throttle = self.num_control_points
        n_brake = self.num_control_points
        n_steering = self.num_control_points
        n_params = n_track + n_throttle + n_brake + n_steering

        # --- Initial Guess ---
        # Start with centerline, moderate constant throttle, minimal braking, zero steering
        initial_track_pos = np.zeros(n_track)
        initial_throttle = np.full(n_throttle, 0.6)
        initial_brake = np.full(n_brake, 0.05)
        initial_steering = np.zeros(n_steering)
        initial_params = np.concatenate([initial_track_pos, initial_throttle, initial_brake, initial_steering])

        # --- Bounds ---
        lower_bounds = np.concatenate([
            np.full(n_track, -self.track_position_limit),
            np.full(n_throttle, 0.0),
            np.full(n_brake, 0.0),
            np.full(n_steering, -0.8) # Limit max steering angle param
        ])
        upper_bounds = np.concatenate([
            np.full(n_track, self.track_position_limit),
            np.full(n_throttle, 1.0),
            np.full(n_brake, 1.0),
            np.full(n_steering, 0.8)
        ])
        bounds = Bounds(lower_bounds, upper_bounds)

        # --- Constraints ---
        constraints = self._create_constraints()

        # --- Optimization ---
        logger.info(f"Starting optimization ({self.optimization_method}) with {n_params} parameters...")
        start_opt_time = time.time()

        # Use try-except block for robustness
        try:
            result = minimize(
                self._objective_function,
                initial_params,
                method=self.optimization_method,
                bounds=bounds,
                constraints=constraints,
                options={
                    'maxiter': self.max_iterations,
                    'ftol': self.tolerance, # Changed from 'tol' to 'ftol' for SLSQP/L-BFGS-B
                    'disp': True # Display convergence messages
                }
            )
            success = result.success
            message = result.message
            optimized_params = result.x
            final_lap_time = result.fun # Objective function value at minimum

        except Exception as e:
            logger.error(f"Optimization failed with error: {e}", exc_info=True)
            success = False
            message = f"Optimization error: {e}"
            optimized_params = initial_params # Fallback to initial guess
            final_lap_time = self._objective_function(initial_params) # Recalculate initial time

        optimization_time = time.time() - start_opt_time
        logger.info(f"Optimization finished in {optimization_time:.2f}s. Success: {success}. Message: {message}")
        logger.info(f"Final Lap Time: {final_lap_time:.4f}s")

        # --- Post-Optimization Simulation ---
        logger.info("Running final simulation with optimized parameters...")
        final_lap_time_check, states_history, racing_line_points = self._simulate_lap_with_params(optimized_params)

        # Convert states history to list of dicts for easier use/saving
        states_list = []
        for state_obj in states_history:
             state_dict = {
                  'time': t, # Need to store time alongside state
                  'x': state_obj.x, 'y': state_obj.y, 'heading': state_obj.heading,
                  'vx': state_obj.vx, 'vy': state_obj.vy, 'yaw_rate': state_obj.yaw_rate,
                  'speed': state_obj.speed, 'engine_rpm': state_obj.engine_rpm,
                  'distance': state_obj.distance # Assuming distance is stored
             }
             if self.include_thermal:
                  state_dict.update({
                       'coolant_temp': state_obj.coolant_temp,
                       'oil_temp': state_obj.oil_temp,
                       'engine_block_temp': state_obj.engine_block_temp
                  })
             # Find corresponding controls (approximate by time/distance) - this is tricky
             # For now, just store state
             states_list.append(state_dict)
             # Need to reconstruct time correctly here if not stored in state
             # This requires modifying _simulate_lap_with_params to return time points


        # Store optimal results
        self.optimal_params = optimized_params
        self.optimal_lap_time = final_lap_time_check # Use the time from the final simulation
        self.optimal_states = states_list # Store the list of dicts
        self.optimal_racing_line = racing_line_points

        return {
            'lap_time': self.optimal_lap_time,
            'racing_line': self.optimal_racing_line,
            'parameters': self.optimal_params,
            'optimization_success': success,
            'optimization_message': message,
            'optimization_time': optimization_time,
            'vehicle_states': self.optimal_states # Return the structured list
        }

    def visualize_optimization_results(self, results: Optional[Dict] = None, save_path: Optional[str] = None):
         """Visualize the optimized racing line and performance."""
         if results is None: results = {} # Use internal results if none provided

         # Prepare data for plotting function
         plot_data = {
             'line': results.get('racing_line', self.optimal_racing_line),
             'distances': np.array([s['distance'] for s in results.get('vehicle_states', [])]) if results.get('vehicle_states') else None,
             'speed_profile': np.array([s['speed'] for s in results.get('vehicle_states', [])]) if results.get('vehicle_states') else None,
             'time_profile': np.array([s['time'] for s in results.get('vehicle_states', [])]) if results.get('vehicle_states') else None,
             'track_points': self.track_data.get('points'),
             'track_width': self.track_data.get('width'),
             'lap_time': results.get('lap_time', self.optimal_lap_time)
             # Curvature needs to be calculated for the optimal line
         }

         if plot_data['line'] is not None and len(plot_data['line']) > 2:
             dx = np.gradient(plot_data['line'][:, 0])
             dy = np.gradient(plot_data['line'][:, 1])
             d2x = np.gradient(dx)
             d2y = np.gradient(dy)
             plot_data['curvature'] = np.abs(dx * d2y - dy * d2x) / np.maximum((dx**2 + dy**2)**1.5, 1e-9)
         else:
             plot_data['curvature'] = None


         fig = plot_racing_line_analysis(plot_data, title="Optimal Lap Time Analysis", save_path=save_path)
         # if fig: plt.close(fig) # Close after saving/showing


# --- Standalone Runner ---
def run_advanced_lap_optimization(vehicle: Vehicle, track_file: str,
                                config_file: Optional[str] = None,
                                save_dir: Optional[str] = None) -> Dict:
    """High-level function to run advanced optimization."""
    try:
        logger.info(f"Running Advanced Lap Optimization for track: {os.path.basename(track_file)}")
        track_profile = TrackProfile(track_file)

        # Load config if provided
        config = None
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r') as f:
                 config = yaml.safe_load(f)

        optimizer = OptimalLapTimeOptimizer(vehicle, track_profile, config=config)
        results = optimizer.optimize_lap_time()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            logger.info(f"Saving optimization results to: {save_dir}")
            # Save visualization
            optimizer.visualize_optimization_results(results, save_path=os.path.join(save_dir, "optimal_lap_visualization.png"))
            # Save optimal parameters
            if results.get('parameters') is not None:
                 np.savetxt(os.path.join(save_dir, "optimal_parameters.csv"), results['parameters'], delimiter=',')
            # Save racing line
            if results.get('racing_line') is not None:
                 np.savetxt(os.path.join(save_dir, "optimal_racing_line.csv"), results['racing_line'], delimiter=',', header='x,y', comments='')
            # Save state history (optional, can be large)
            if results.get('vehicle_states'):
                 try:
                     pd.DataFrame(results['vehicle_states']).to_csv(os.path.join(save_dir, "optimal_states.csv"), index=False, float_format='%.5f')
                 except Exception as e:
                      logger.warning(f"Could not save vehicle states to CSV: {e}")
            # Save summary JSON
            summary = {k: v for k, v in results.items() if k not in ['parameters', 'racing_line', 'vehicle_states']}
            summary['parameters_shape'] = results['parameters'].shape if results.get('parameters') is not None else None
            summary['racing_line_points'] = len(results['racing_line']) if results.get('racing_line') is not None else 0
            summary['num_states'] = len(results.get('vehicle_states', []))
            with open(os.path.join(save_dir, "optimal_summary.json"), 'w') as f:
                 json.dump(summary, f, indent=2)


        logger.info(f"Advanced Optimization Result: Lap Time = {results.get('lap_time', 'N/A'):.4f}s")
        return results

    except Exception as e:
        logger.error(f"Error during advanced lap optimization: {e}", exc_info=True)
        return {'error': str(e), 'lap_time': None}

# Example Usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        from ..core.vehicle import create_formula_student_vehicle
        from .lap_time import create_example_track # Use lap_time's version
        import tempfile

        print("Advanced Lap Time Optimization Demo")
        print("-" * 35)

        vehicle = create_formula_student_vehicle()
        output_dir = tempfile.mkdtemp()
        track_file = os.path.join(output_dir, "optim_track.yaml")
        config_file = os.path.join(output_dir, "optim_config.yaml")

        print(f"Output directory: {output_dir}")
        create_example_track(track_file, difficulty='easy') # Easy track for faster demo

        # Create a minimal config for faster run
        optim_config = {
             'optimization': {
                  'max_iterations': 15,
                  'num_control_points': 30,
                  'dt': 0.03,
                  'tolerance': 5e-3
             }
        }
        with open(config_file, 'w') as f: yaml.dump(optim_config, f)

        print("\nRunning advanced optimization (reduced settings)...")
        results = run_advanced_lap_optimization(vehicle, track_file, config_file=config_file, save_dir=output_dir)

        if results.get('lap_time') is not None:
            print(f"\nOptimization Finished:")
            print(f" Lap Time: {results['lap_time']:.3f} s")
            print(f" Success: {results.get('optimization_success')}")
            print(f" Message: {results.get('optimization_message')}")
            print(f" Runtime: {results.get('optimization_time', 0):.1f} s")
        else:
            print("\nOptimization Failed.")
            print(f" Error: {results.get('error')}")

    except ImportError as e:
        print(f"\nError: Could not import necessary modules ({e}). Run from project root or ensure package is installed.")
    except FileNotFoundError as e:
         print(f"\nError: Configuration file not found. Make sure default configs exist. {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()