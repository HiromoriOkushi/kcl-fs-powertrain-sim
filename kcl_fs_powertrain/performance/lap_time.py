"""
Lap time simulation module for Formula Student powertrain.

Simulates vehicle performance around a defined track, calculating lap times,
speed profiles, and analyzing performance metrics considering vehicle dynamics
and powertrain limits.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Callable
import logging
import time
import yaml
from scipy.interpolate import interp1d

# Import core components (handle potential errors for standalone execution)
try:
    from ..core.vehicle import Vehicle
    from ..core.track import TrackSegmentType # Use TrackSegmentType from core.track
    from ..core.track_integration import TrackProfile, calculate_optimal_racing_line
    from ..utils.constants import GRAVITY, MS_TO_KMH, MS_TO_MPH
    from ..utils.plotting import plot_lap_time_results as plot_lap_unified # Use unified plotter
    from ..utils.plotting import plot_lap_time_comparison as plot_lap_comp_unified
    from ..utils.plotting import save_plot, _apply_common_ax_settings, COLOR_SCHEMES
    from ..utils.track_utils import preprocess_track_points, ensure_unique_values
except ImportError:
    # Fallbacks
    GRAVITY = 9.81
    MS_TO_KMH = 3.6
    MS_TO_MPH = 2.23694
    class Vehicle: pass
    class TrackProfile: pass
    class TrackSegmentType: STRAIGHT=0; CORNER_LEFT=1; CORNER_RIGHT=2 # Mock Enum
    def calculate_optimal_racing_line(*args, **kwargs): return None
    def plot_lap_unified(*args, **kwargs): plt.figure(); plt.plot([0,1],[0,1]); plt.title("Fallback Plot"); plt.show(); plt.close(); return plt.gcf()
    def plot_lap_comp_unified(*args, **kwargs): plt.figure(); plt.plot([0,1],[0,1]); plt.title("Fallback Comparison Plot"); plt.show(); plt.close(); return plt.gcf()
    def save_plot(fig, path, **kwargs): pass
    def _apply_common_ax_settings(ax, **kwargs): pass
    COLOR_SCHEMES = {'default': plt.cm.tab10.colors}
    def preprocess_track_points(data): return data # Passthrough
    def ensure_unique_values(x): return x # Passthrough
    logger = logging.getLogger("LapTime_Fallback")
    logger.warning("Could not import all necessary modules. Using fallbacks.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("LapTimeSimulation")

class CorneringPerformance:
    """Calculates vehicle cornering performance limits."""
    def __init__(self, vehicle: Vehicle):
        self.vehicle = vehicle
        # Extract relevant parameters with defaults
        self.mass_kg = getattr(vehicle, 'mass', 230.0)
        self.cg_height_m = getattr(vehicle, 'cg_height', 0.28)
        self.track_width_m = getattr(vehicle, 'track_width_rear', 1.15) # Use rear track width typically
        self.wheelbase_m = getattr(vehicle, 'wheelbase', 1.6)
        self.weight_dist_front = getattr(vehicle, 'weight_distribution_front', 0.48)
        # Aero properties
        self.lift_coefficient = getattr(vehicle, 'lift_coefficient', -2.0) # Negative = downforce
        self.frontal_area_m2 = getattr(vehicle, 'frontal_area', 1.1)
        # Tire properties (simplified)
        self.base_friction_coeff = 1.6 # High grip slicks
        self.load_sensitivity = 0.1 # How much grip increases with load (simplified)

        logger.debug("CorneringPerformance initialized.")

    def calculate_downforce_N(self, speed_mps: float) -> float:
        """Calculate aerodynamic downforce at a given speed."""
        air_density = 1.225 # kg/m^3
        # Downforce = -Lift = - Cl * 0.5 * rho * A * v^2
        downforce = -self.lift_coefficient * 0.5 * air_density * self.frontal_area_m2 * speed_mps**2
        return max(0, downforce) # Ensure non-negative downforce

    def calculate_max_lateral_acceleration(self, speed_mps: Optional[float] = None) -> float:
        """Calculate maximum sustainable lateral acceleration (m/s²)."""
        # Total vertical load = Weight + Downforce
        weight_N = self.mass_kg * GRAVITY
        downforce_N = self.calculate_downforce_N(speed_mps) if speed_mps is not None else 0
        total_vertical_load = weight_N + downforce_N

        # Effective friction coefficient increases with load (simplified)
        effective_mu = self.base_friction_coeff * (1 + self.load_sensitivity * (total_vertical_load / weight_N - 1)) if weight_N > 0 else self.base_friction_coeff
        effective_mu = min(effective_mu, 2.5) # Cap effective friction

        # Max lateral force = mu * Normal Force
        max_lateral_force = effective_mu * total_vertical_load

        # Max lateral acceleration = F_lat / mass
        max_lat_accel = max_lateral_force / self.mass_kg
        return max_lat_accel

    def calculate_max_cornering_speed(self, radius_m: float) -> float:
        """Calculate maximum speed (m/s) for a given corner radius."""
        if radius_m <= 0: return 0.0

        # Iteratively solve for speed since max_lat_accel depends on speed (via downforce)
        # v^2 = a_lat(v) * r
        max_speed_mps = 0.0
        # Start with static calculation
        a_lat_static = self.calculate_max_lateral_acceleration(speed_mps=0.0)
        speed_guess = np.sqrt(a_lat_static * radius_m)

        for _ in range(5): # Iterate a few times for convergence
             a_lat_dynamic = self.calculate_max_lateral_acceleration(speed_guess)
             max_speed_mps = np.sqrt(a_lat_dynamic * radius_m)
             # Update guess (use average or simply the new value)
             speed_guess = (speed_guess + max_speed_mps) / 2.0

        return max_speed_mps

    def calculate_lateral_weight_transfer(self, lateral_accel_mps2: float) -> float:
        """Calculate lateral weight transfer (N) from inside to outside wheels."""
        # Weight Transfer = (Mass * LatAccel * CG_Height) / TrackWidth
        weight_transfer_N = (self.mass_kg * abs(lateral_accel_mps2) * self.cg_height_m) / self.track_width_m
        return weight_transfer_N

    def calculate_roll_angle_deg(self, lateral_accel_mps2: float) -> float:
        """Estimate roll angle (degrees) based on lateral acceleration."""
        # Simplified: Linear roll gradient (needs tuning from suspension model)
        roll_gradient_deg_per_g = 1.0 # Example value
        lateral_g = lateral_accel_mps2 / GRAVITY
        roll_angle = roll_gradient_deg_per_g * abs(lateral_g)
        return roll_angle

    def get_cornering_metrics(self, speed_mps: float, radius_m: float) -> Dict:
        """Calculate comprehensive cornering metrics."""
        if radius_m <= 0: # Straight line
            lat_accel = 0.0
        else:
            lat_accel = speed_mps**2 / radius_m

        max_speed = self.calculate_max_cornering_speed(radius_m)
        max_lat_accel = self.calculate_max_lateral_acceleration(speed_mps)
        weight_transfer = self.calculate_lateral_weight_transfer(lat_accel)
        roll_angle = self.calculate_roll_angle_deg(lat_accel)

        # Check if speed exceeds limit
        unstable = speed_mps > max_speed * 1.01 # Add 1% tolerance

        return {
            'speed_mps': speed_mps,
            'radius_m': radius_m,
            'lateral_accel_mps2': lat_accel,
            'lateral_g': lat_accel / GRAVITY,
            'max_possible_speed_mps': max_speed,
            'max_possible_lateral_g': max_lat_accel / GRAVITY,
            'lateral_weight_transfer_N': weight_transfer,
            'roll_angle_deg': roll_angle,
            'is_stable': not unstable
        }


class LapTimeSimulator:
    """Simulates lap time around a track."""
    def __init__(self, vehicle: Vehicle, track_profile: Optional[TrackProfile] = None, track_file: Optional[str] = None):
        if not isinstance(vehicle, Vehicle):
             if "MockVehicle" not in str(type(vehicle)):
                 raise TypeError("vehicle must be an instance of the Vehicle class.")
        self.vehicle = vehicle
        self.track_profile: Optional[TrackProfile] = None
        self.track_data: Optional[Dict] = None
        self.racing_line: Optional[np.ndarray] = None
        self.speed_profile_mps: Optional[np.ndarray] = None # Speed profile along racing line
        self.time_profile_s: Optional[np.ndarray] = None # Cumulative time along racing line
        self.distances_m: Optional[np.ndarray] = None # Cumulative distance along racing line
        self.lap_time_s: Optional[float] = None
        self.sector_times: Optional[List[Dict]] = None

        self.cornering = CorneringPerformance(vehicle)

        # Simulation parameters
        self.include_thermal: bool = True

        # Load track if provided
        if track_profile:
            self.track_profile = track_profile
            self.track_data = self._load_and_preprocess_track(track_profile)
        elif track_file:
            self.load_track(track_file)

        logger.info("LapTimeSimulator initialized.")

    def _load_and_preprocess_track(self, track_input: Union[str, TrackProfile]) -> Optional[Dict]:
        """Loads and preprocesses track data."""
        try:
            if isinstance(track_input, str):
                profile = TrackProfile(track_input)
            elif isinstance(track_input, TrackProfile):
                profile = track_input
            else:
                raise TypeError("track_input must be a file path or TrackProfile object.")

            data = profile.get_track_data()
            if not data or 'points' not in data or len(data['points']) < 3:
                 logger.error("Track data is invalid or missing points.")
                 return None

            processed_data = preprocess_track_points(data)
            # Ensure essential arrays exist after preprocessing
            if 'points' not in processed_data or 'distance' not in processed_data:
                 logger.error("Track data preprocessing failed to produce required arrays.")
                 return None
            # Calculate curvature if not present
            if 'curvature' not in processed_data or len(processed_data['curvature']) != len(processed_data['points']):
                 logger.info("Calculating track curvature...")
                 points = processed_data['points']
                 # Use a robust curvature calculation method
                 dx = np.gradient(points[:, 0])
                 dy = np.gradient(points[:, 1])
                 d2x = np.gradient(dx)
                 d2y = np.gradient(dy)
                 curvature = np.abs(dx * d2y - dy * d2x) / np.maximum((dx**2 + dy**2)**1.5, 1e-9)
                 processed_data['curvature'] = curvature # Store absolute curvature

            self.track_profile = profile # Store the profile object
            logger.info(f"Track loaded and preprocessed. Length: {processed_data['distance'][-1]:.1f} m")
            return processed_data
        except Exception as e:
            logger.error(f"Error loading/preprocessing track: {e}")
            return None

    def load_track(self, track_file: str):
        """Load and preprocess track data from file."""
        self.track_data = self._load_and_preprocess_track(track_file)
        # Reset dependent calculations
        self.racing_line = None
        self.speed_profile_mps = None
        self.lap_time_s = None

    def calculate_racing_line(self, method: str = 'geometric', optimize: bool = True) -> Optional[np.ndarray]:
        """Calculate the racing line (defaults to centerline if optimization fails)."""
        if not self.track_data:
             logger.error("Cannot calculate racing line: Track data not loaded.")
             return None

        logger.info(f"Calculating racing line using '{method}' method...")
        try:
             # Currently, only a simplified geometric approach or centerline fallback
             # The advanced optimization is in optimal_lap_time.py
             if optimize:
                  # Placeholder for a simple geometric line based on curvature
                  curvature = self.track_data.get('curvature', np.zeros(len(self.track_data['points'])))
                  track_pos = -np.tanh(curvature * 5.0) * 0.8 # Move towards inside, max 80%
                  # Smooth the positions
                  if len(track_pos) > 10:
                      window_size = max(5, len(track_pos)//20)
                      track_pos = np.convolve(track_pos, np.ones(window_size)/window_size, mode='same')

                  # Calculate line points from positions
                  points = self.track_data['points']
                  width = self.track_data.get('width', np.full(len(points), 3.0))
                  # Recalculate normals robustly
                  normals = np.zeros_like(points)
                  tangents = np.gradient(points, axis=0)
                  norms = np.linalg.norm(tangents, axis=1)
                  valid = norms > 1e-6
                  normals[valid, 0] = -tangents[valid, 1] / norms[valid]
                  normals[valid, 1] = tangents[valid, 0] / norms[valid]
                  if np.allclose(points[0], points[-1]): # Handle loop closure
                       normals[0] = normals[-1] = (normals[1] + normals[-2]) / 2.0
                       normals[0] /= np.linalg.norm(normals[0])
                       normals[-1] = normals[0]

                  self.racing_line = points + normals * (track_pos * width / 2.0)[:, np.newaxis]

             else:
                  self.racing_line = self.track_data['points'] # Use centerline

             # Calculate distances and curvature along the *new* racing line
             if self.racing_line is not None:
                 self.distances_m = np.zeros(len(self.racing_line))
                 for i in range(1, len(self.racing_line)):
                     self.distances_m[i] = self.distances_m[i-1] + np.linalg.norm(self.racing_line[i] - self.racing_line[i-1])
                 # Recalculate curvature for the racing line
                 dx = np.gradient(self.racing_line[:, 0])
                 dy = np.gradient(self.racing_line[:, 1])
                 d2x = np.gradient(dx)
                 d2y = np.gradient(dy)
                 curvature_rl = np.abs(dx * d2y - dy * d2x) / np.maximum((dx**2 + dy**2)**1.5, 1e-9)
                 self.racing_line_curvature = curvature_rl # Store racing line curvature

                 logger.info(f"Racing line calculated. Length: {self.distances_m[-1]:.1f} m")
                 return self.racing_line

        except Exception as e:
             logger.error(f"Error calculating racing line: {e}. Falling back to centerline.")
             self.racing_line = self.track_data['points']
             self.distances_m = self.track_data['distance']
             self.racing_line_curvature = self.track_data['curvature']
             return self.racing_line
        return None


    def calculate_speed_profile(self) -> Optional[np.ndarray]:
        """Calculate speed profile (m/s) along the current racing line."""
        if self.racing_line is None or self.distances_m is None or self.racing_line_curvature is None:
            logger.info("Racing line not calculated, calculating it first...")
            if self.calculate_racing_line() is None:
                logger.error("Failed to calculate racing line, cannot generate speed profile.")
                return None

        n_points = len(self.racing_line)
        speeds = np.zeros(n_points)
        curvature = self.racing_line_curvature
        distances = self.distances_m

        logger.info("Calculating speed profile...")

        # --- Pass 1: Cornering Speed Limit ---
        max_vehicle_speed = self.vehicle.calculate_max_speed() if hasattr(self.vehicle, 'calculate_max_speed') else 50.0 # Default 50 m/s (~180kph)
        for i in range(n_points):
            if abs(curvature[i]) > 1e-6:
                radius = 1.0 / abs(curvature[i])
                # Use cornering calculator, iteratively finding speed limit
                corner_speed_limit = self.cornering.calculate_max_cornering_speed(radius)
                speeds[i] = min(corner_speed_limit, max_vehicle_speed)
            else:
                speeds[i] = max_vehicle_speed # Straight

        # --- Pass 2: Braking Limit (Backward Pass) ---
        # Assume max braking g (negative value)
        max_braking_g = -1.8 # Example
        max_braking_accel = max_braking_g * GRAVITY

        # Iterate backwards (use modulo for closed track)
        for i in range(n_points - 2, -1, -1):
            # Calculate distance to next point
            ds = distances[i+1] - distances[i]
            if ds < 1e-6: continue # Skip zero-length segments

            # Speed limit from braking: v_current^2 <= v_next^2 - 2 * a_brake * ds
            v_next = speeds[i+1]
            speed_limit_brake_sq = v_next**2 - 2 * max_braking_accel * ds
            if speed_limit_brake_sq < 0: speed_limit_brake_sq = 0 # Cannot be negative squared speed
            speed_limit_brake = np.sqrt(speed_limit_brake_sq)

            # Update speed if braking limit is lower
            speeds[i] = min(speeds[i], speed_limit_brake)

        # --- Pass 3: Acceleration Limit (Forward Pass) ---
        # Assume max acceleration g
        max_accel_g = 1.2 # Example
        max_accel = max_accel_g * GRAVITY

        # Iterate forwards
        for i in range(n_points - 1):
             # Calculate distance to next point
             ds = distances[i+1] - distances[i]
             if ds < 1e-6:
                  speeds[i+1] = min(speeds[i+1], speeds[i]) # Ensure non-increasing speed over zero distance
                  continue

             # Speed limit from accelerating: v_next^2 <= v_current^2 + 2 * a_accel * ds
             v_current = speeds[i]
             speed_limit_accel_sq = v_current**2 + 2 * max_accel * ds
             speed_limit_accel = np.sqrt(speed_limit_accel_sq)

             # Update speed if acceleration limit is lower
             speeds[i+1] = min(speeds[i+1], speed_limit_accel)

        self.speed_profile_mps = speeds
        logger.info(f"Speed profile calculated. Max Speed: {np.max(speeds):.1f} m/s")

        # Calculate time profile based on speed profile
        self._calculate_time_profile()

        return self.speed_profile_mps

    def _calculate_time_profile(self):
        """Calculate cumulative time along the racing line."""
        if self.speed_profile_mps is None or self.distances_m is None or len(self.speed_profile_mps) < 2:
             logger.warning("Cannot calculate time profile: Speed profile or distances missing.")
             self.time_profile_s = None
             self.lap_time_s = None
             return

        n_points = len(self.speed_profile_mps)
        times = np.zeros(n_points)
        distances = self.distances_m
        speeds = self.speed_profile_mps

        for i in range(n_points - 1):
             ds = distances[i+1] - distances[i]
             # Use average speed over the segment
             avg_speed = (speeds[i] + speeds[i+1]) / 2.0
             if avg_speed < 1e-3: # Avoid division by zero, assume minimum time step
                  dt = 0.1 # Assign a small time step if speed is near zero
             else:
                  dt = ds / avg_speed
             times[i+1] = times[i] + dt

        self.time_profile_s = times
        self.lap_time_s = times[-1]
        logger.info(f"Time profile calculated. Lap Time: {self.lap_time_s:.3f} s")

    def simulate_lap(self, include_thermal: bool = True) -> Dict:
        """Simulate a lap based on pre-calculated speed profile."""
        if self.speed_profile_mps is None or self.time_profile_s is None or self.distances_m is None:
             logger.info("Speed/Time profile not calculated, calculating now...")
             if self.calculate_speed_profile() is None:
                 logger.error("Failed to calculate speed profile for lap simulation.")
                 return {'error': "Failed to calculate speed profile"}

        n_points = len(self.distances_m)
        results = {
            'lap_time': self.lap_time_s,
            'time': self.time_profile_s,
            'distance': self.distances_m,
            'speed': self.speed_profile_mps,
            'acceleration': np.gradient(self.speed_profile_mps, self.time_profile_s, edge_order=2) if self.lap_time_s > 0 else np.zeros(n_points),
            'lateral_g': (self.speed_profile_mps**2 * np.abs(self.racing_line_curvature)) / GRAVITY if self.racing_line_curvature is not None else np.zeros(n_points),
            'engine_rpm': np.zeros(n_points),
            'gear': np.zeros(n_points, dtype=int),
            'include_thermal': include_thermal
        }

        # Calculate gear and RPM for each point
        for i in range(n_points):
            gear = self._estimate_optimal_gear(results['speed'][i])
            results['gear'][i] = gear
            results['engine_rpm'][i] = self.vehicle.drivetrain.calculate_engine_speed_rpm(results['speed'][i], gear)

        # --- Thermal Simulation (Optional) ---
        if include_thermal:
            logger.info("Including thermal effects in lap simulation...")
            # Initialize thermal state
            ambient_temp = 25.0 # Get from environment if available
            self.vehicle.engine_temperature = ambient_temp + 50 # Start warm
            self.vehicle.coolant_temperature = self.vehicle.engine_temperature - 5
            self.vehicle.oil_temperature = self.vehicle.engine_temperature - 10
            self.vehicle.thermal_factor = 1.0

            temps_engine = np.zeros(n_points)
            temps_coolant = np.zeros(n_points)
            temps_oil = np.zeros(n_points)
            power_factors = np.zeros(n_points)
            thermal_limited = False

            for i in range(n_points):
                dt = self.time_profile_s[i] - self.time_profile_s[i-1] if i > 0 else self.time_profile_s[0]
                if dt <= 0: dt = 0.01 # Avoid zero/negative dt

                # Update vehicle thermal state
                # Need engine torque/power for heat calculation
                throttle_est = min(1.0, results['acceleration'][i] / 5.0 + 0.3) # Very rough throttle estimate
                torque_est = self.vehicle.engine.get_torque(results['engine_rpm'][i], throttle_est, self.vehicle.engine_temperature)
                power_kw_est = torque_est * results['engine_rpm'][i] * (2*np.pi/60) / 1000.0

                # Simplified heat input: proportional to power + base friction heat
                heat_input_W = power_kw_est * 1000 * 2.0 + 5000 # Assume 2W heat per W power + 5kW base

                # Update cooling system (requires coolant flow estimate)
                # This requires a more integrated simulation step, simplified here
                coolant_flow_est = 50.0 # LPM estimate
                cooling_effectiveness_est = 0.7 # Estimate

                if hasattr(self.vehicle, 'update_thermal_state') and callable(self.vehicle.update_thermal_state):
                    # Use vehicle's integrated thermal update if available
                    self.vehicle.update_thermal_state(dt) # Assuming it uses internal state
                elif hasattr(self.vehicle.engine, 'update_thermal_state'):
                    # Use engine's thermal update
                    self.vehicle.engine.update_thermal_state(
                        ambient_temp=ambient_temp,
                        cooling_effectiveness=cooling_effectiveness_est,
                        dt=dt
                    )
                else:
                    # Basic fallback thermal update (less accurate)
                    net_heat = heat_input_W - (cooling_effectiveness_est * (self.vehicle.engine_temperature - ambient_temp) * 100)
                    self.vehicle.engine_temperature += net_heat * dt / 50000 # Simplified capacity

                # Store temperatures and factor
                temps_engine[i] = self.vehicle.engine_temperature
                temps_coolant[i] = self.vehicle.coolant_temperature
                temps_oil[i] = self.vehicle.oil_temperature
                power_factors[i] = self.vehicle.thermal_factor

                # Check if thermal limits were hit (potentially reduce speed profile here in a more advanced sim)
                if self.vehicle.thermal_factor < 0.95: # If performance is derated
                    thermal_limited = True

            results['engine_temp'] = temps_engine
            results['coolant_temp'] = temps_coolant
            results['oil_temp'] = temps_oil
            results['power_factor'] = power_factors
            results['thermal_limited'] = thermal_limited
            if thermal_limited: logger.warning("Lap simulation indicates thermal limiting.")

        logger.info(f"Lap simulated. Lap Time: {self.lap_time_s:.3f} s")
        return results

    def calculate_sector_times(self) -> List[Dict]:
        """Calculate time spent in predefined track sections (straights, corners)."""
        if not self.track_profile or not hasattr(self.track_profile, 'sections') or not self.track_profile.sections:
            logger.warning("No track sections defined in TrackProfile.")
            return []
        if self.time_profile_s is None or self.distances_m is None:
            logger.error("Lap simulation must be run before calculating sector times.")
            return []

        self.sector_times = []
        sections = self.track_profile.sections

        # Interpolate time based on distance
        time_interp = interp1d(self.distances_m, self.time_profile_s, bounds_error=False, fill_value='extrapolate')

        last_section_end_dist = 0.0
        for i, section in enumerate(sections):
            start_dist = self.distances_m[section['start_idx']]
            end_dist = self.distances_m[section['end_idx']]

            start_time = float(time_interp(start_dist))
            end_time = float(time_interp(end_dist))

            # Ensure times are monotonic
            if i > 0: start_time = max(start_time, self.sector_times[-1]['end_time'])
            end_time = max(end_time, start_time)

            sector_time = end_time - start_time
            sector_len = end_dist - start_dist

            self.sector_times.append({
                'sector': i + 1,
                'type': section['type'],
                'start_dist_m': start_dist,
                'end_dist_m': end_dist,
                'length_m': sector_len,
                'start_time_s': start_time,
                'end_time_s': end_time,
                'duration_s': sector_time
            })
            last_section_end_dist = end_dist

        # Adjust last sector end time to match total lap time
        if self.sector_times:
            self.sector_times[-1]['end_time'] = self.lap_time_s
            self.sector_times[-1]['duration_s'] = self.sector_times[-1]['end_time'] - self.sector_times[-1]['start_time_s']

        return self.sector_times

    def analyze_lap_performance(self, results: Dict) -> Dict:
        """Analyze lap performance metrics."""
        if 'lap_time' not in results or results['lap_time'] is None:
             logger.error("Lap simulation results incomplete or missing lap_time.")
             return {'error': 'Incomplete lap results'}

        lap_time = results['lap_time']
        speed_mps = results['speed']
        distances_m = results['distance']
        lateral_g = results.get('lateral_g', np.zeros_like(speed_mps))
        time_s = results['time']

        track_length = distances_m[-1]
        avg_speed_mps = track_length / lap_time if lap_time > 0 else 0
        max_speed_mps = np.max(speed_mps)
        max_lateral_g = np.max(np.abs(lateral_g))

        # Time in corners vs straights (using curvature from racing line)
        curvature = self.racing_line_curvature if self.racing_line_curvature is not None else np.zeros_like(distances_m)
        corner_threshold = 1.0 / 30.0 # Consider radius < 30m a corner
        corner_mask = np.abs(curvature) > corner_threshold
        dt = np.diff(time_s, prepend=0)
        time_in_corners = np.sum(dt[corner_mask])
        time_in_straights = lap_time - time_in_corners

        metrics = {
            'lap_time': lap_time,
            'track_length_m': track_length,
            'avg_speed_mps': avg_speed_mps,
            'avg_speed_kph': avg_speed_mps * MS_TO_KMH,
            'max_speed_mps': max_speed_mps,
            'max_speed_kph': max_speed_mps * MS_TO_KMH,
            'max_lateral_g': max_lateral_g,
            'time_in_corners_s': time_in_corners,
            'time_in_straights_s': time_in_straights,
            'corner_time_percent': time_in_corners / lap_time * 100 if lap_time > 0 else 0,
            'thermal_limited': results.get('thermal_limited', False)
        }

        return metrics

    def visualize_lap(self, results: Dict, save_path: Optional[str] = None):
        """Visualize lap results using the unified plotting function."""
        if 'lap_time' not in results or results['lap_time'] is None:
            logger.error("Cannot visualize lap: Simulation results are missing.")
            return

        # Prepare data for the plotting function
        plot_data = results.copy()
        if self.track_data:
            plot_data['track_points'] = self.track_data.get('points')
            plot_data['track_width'] = self.track_data.get('width')
        if self.racing_line is not None:
             plot_data['racing_line'] = self.racing_line
        if self.sector_times:
             plot_data['sector_times'] = self.sector_times

        fig = plot_lap_unified(plot_data, save_path=save_path)
        # Optional: Close figure after saving/showing
        # if fig: plt.close(fig)

    def compare_vehicle_configs(self, vehicle_configs: List[Vehicle], labels: List[str],
                              include_thermal: bool = True, save_path: Optional[str] = None) -> Dict:
        """Compare lap times for different vehicle configurations."""
        if not self.track_data: raise ValueError("Track not loaded.")
        if len(vehicle_configs) != len(labels): raise ValueError("Mismatch between vehicles and labels.")

        original_vehicle = self.vehicle # Store current vehicle
        comparison_results = []
        all_detailed_results = []

        logger.info(f"Comparing {len(vehicle_configs)} vehicle configurations...")
        for i, (vehicle, label) in enumerate(zip(vehicle_configs, labels)):
            logger.info(f" Simulating config '{label}'...")
            self.vehicle = vehicle # Set current vehicle
            # Re-initialize cornering calculator for the new vehicle
            self.cornering = CorneringPerformance(self.vehicle)
            # Reset results for this vehicle
            self.racing_line = None
            self.speed_profile_mps = None
            self.lap_time_s = None

            # Run simulation
            lap_results = self.simulate_lap(include_thermal=include_thermal)
            lap_metrics = self.analyze_lap_performance(lap_results)

            # Store results
            all_detailed_results.append({'label': label, **lap_results})
            comparison_results.append({
                'label': label,
                'lap_time': lap_metrics['lap_time'],
                'avg_speed': lap_metrics['avg_speed_mps'],
                'max_speed': lap_metrics['max_speed_mps'],
                'max_lateral_g': lap_metrics['max_lateral_g'],
                'thermal_limited': lap_metrics['thermal_limited']
                # Add more metrics if needed
            })

        self.vehicle = original_vehicle # Restore original vehicle
        self.cornering = CorneringPerformance(self.vehicle) # Restore cornering calc

        # Create comparison plot
        if save_path:
             self._plot_lap_time_comparison(all_detailed_results, labels, save_path) # Use internal plotter

        logger.info("Vehicle configuration comparison complete.")
        return {'summary': comparison_results, 'details': all_detailed_results}

    def _plot_lap_time_comparison(self, comparison_results: List[Dict], labels: List[str], save_path: str):
        """Internal helper to plot comparison using the unified plotter."""
        fig = plot_lap_comp_unified(comparison_results, labels=labels, save_path=save_path)
        # if fig: plt.close(fig) # Close after saving

# --- Factory Functions ---
def create_lap_time_simulator(vehicle: Vehicle, track_file: str) -> LapTimeSimulator:
    """Factory function to create and load a LapTimeSimulator."""
    return LapTimeSimulator(vehicle, track_file=track_file)

def run_fs_lap_simulation(vehicle: Vehicle, track_file: str,
                         include_thermal: bool = True,
                         save_dir: Optional[str] = None) -> Dict:
    """High-level function to run lap simulation and generate outputs."""
    try:
        logger.info(f"Running FS Lap Simulation for track: {os.path.basename(track_file)}")
        simulator = create_lap_time_simulator(vehicle, track_file)
        lap_results = simulator.simulate_lap(include_thermal=include_thermal)
        lap_metrics = simulator.analyze_lap_performance(lap_results)

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            logger.info(f"Saving lap simulation results to: {save_dir}")
            # Save plots
            simulator.visualize_lap(lap_results, save_path=os.path.join(save_dir, "lap_visualization.png"))
            # Save detailed data
            try:
                 # Combine results and metrics into one dict for potential saving
                 full_output = {'results': lap_results, 'metrics': lap_metrics}
                 # Example: Save metrics to JSON
                 metrics_path = os.path.join(save_dir, "lap_metrics.json")
                 with open(metrics_path, 'w') as f:
                      # Convert numpy arrays for JSON
                      serializable_metrics = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in lap_metrics.items()}
                      json.dump(serializable_metrics, f, indent=2)
                 # Example: Save time series data to CSV
                 timeseries_path = os.path.join(save_dir, "lap_timeseries.csv")
                 timeseries_data = {k:v for k,v in lap_results.items() if isinstance(v, np.ndarray) and len(v) == len(lap_results['time'])}
                 pd.DataFrame(timeseries_data).to_csv(timeseries_path, index=False, float_format='%.4f')
            except Exception as e:
                 logger.error(f"Error saving detailed results: {e}")

        return {'lap_time': lap_metrics['lap_time'], 'metrics': lap_metrics, 'results': lap_results}

    except Exception as e:
        logger.error(f"Error running FS lap simulation: {e}", exc_info=True)
        return {'error': str(e)}

def create_example_track(output_file: str, difficulty: str = 'medium') -> str:
    """Creates a simple example track YAML file."""
    # This function might be better placed in track_utils or examples,
    # but included here as per original breakdown.
    logger.info(f"Creating example track file: {output_file} (Difficulty: {difficulty})")
    # Use the track generator if available, otherwise fallback
    try:
        from ..track_generator.generator import FSTrackGenerator
        from ..track_generator.enums import SimType, TrackMode

        gen_params = {}
        if difficulty == 'easy': gen_params = {'min_length': 600, 'max_length': 800, 'curvature_threshold': 0.20}
        elif difficulty == 'medium': gen_params = {'min_length': 900, 'max_length': 1200, 'curvature_threshold': 0.25}
        else: gen_params = {'min_length': 1100, 'max_length': 1400, 'curvature_threshold': 0.33}

        # Point generator to parent dir of output_file for metadata etc.
        base_output_dir = os.path.dirname(os.path.dirname(output_file)) # Assume output_file is in output/tracks
        track_output_dir = os.path.dirname(output_file)

        generator = FSTrackGenerator(base_dir=base_output_dir, output_dir_override=track_output_dir, **gen_params)
        metadata = generator.generate_track(mode=TrackMode.EXTEND) # Generate CSV first

        if metadata:
             # Export specifically to the requested output_file path and format (YAML)
             yaml_path = output_file # Path already includes desired name
             if generator.export_track(yaml_path, SimType.FSSIM):
                  logger.info(f"Example track generated and saved as FSSIM YAML: {yaml_path}")
                  return yaml_path
             else:
                  logger.error("Failed to export generated track to YAML.")
        else:
             logger.error("Track generator failed to create a track.")

    except ImportError:
        logger.warning("Track generator not found. Creating very basic manual track.")
    except Exception as e:
         logger.error(f"Error using track generator: {e}. Creating basic manual track.")

    # --- Fallback: Manual Basic Track ---
    track_data = {
        'metadata': {'name': f'Fallback Example {difficulty} Track', 'length': 500},
        'points': [[0,0], [100,0], [100,50], [0,50], [0,0]], # Simple rectangle
        'width': [3.0] * 5
    }
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        yaml.dump(track_data, f)
    logger.info(f"Saved basic fallback track to {output_file}")
    return output_file

# Example Usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        from ..core.vehicle import create_formula_student_vehicle
        import tempfile

        print("Lap Time Simulation Demo")
        print("-" * 26)

        vehicle = create_formula_student_vehicle()
        output_dir = tempfile.mkdtemp()
        track_file = os.path.join(output_dir, "demo_track.yaml")
        create_example_track(track_file, difficulty='medium')

        print(f"Output directory: {output_dir}")
        print(f"Track file: {track_file}")

        # Run simulation
        simulation_output = run_fs_lap_simulation(vehicle, track_file, save_dir=output_dir)

        if 'error' in simulation_output:
             print(f"\nSimulation failed: {simulation_output['error']}")
        else:
             print("\nSimulation Successful:")
             print(f" Lap Time: {simulation_output['lap_time']:.3f} s")
             print(f" Avg Speed: {simulation_output['metrics']['avg_speed_kph']:.1f} km/h")
             print(f" Max Speed: {simulation_output['metrics']['max_speed_kph']:.1f} km/h")
             print(f" Max Lateral G: {simulation_output['metrics']['max_lateral_g']:.2f} g")
             print(f" Thermally Limited: {simulation_output['metrics']['thermal_limited']}")

    except ImportError as e:
        print(f"\nError: Could not import necessary modules ({e}). Run from project root or ensure package is installed.")
    except FileNotFoundError as e:
         print(f"\nError: Configuration file not found. Make sure default configs exist. {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()