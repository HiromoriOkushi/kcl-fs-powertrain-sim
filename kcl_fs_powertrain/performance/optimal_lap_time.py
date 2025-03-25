"""
Advanced optimization for lap time simulation.

This module extends the lap time simulator with advanced numerical methods
for finding the optimal racing line and control inputs to minimize lap time.
It implements Runge-Kutta integration for accurate vehicle dynamics and
gradient-based optimization for finding the optimal solution.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
import logging
from scipy.optimize import minimize, Bounds, LinearConstraint
from scipy.interpolate import CubicSpline, interp1d
import time

from .lap_time import LapTimeSimulator, CorneringPerformance
from ..core.vehicle import Vehicle
from ..core.track_integration import TrackProfile
from ..utils.track_utils import preprocess_track_points, ensure_unique_values

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("Optimal_Lap_Time")


class VehicleState:
    """
    Represents the complete state of the vehicle at a point in time.
    This includes position, velocity, orientation, and other state variables.
    """
    
    def __init__(self):
        # Position and orientation
        self.x = 0.0  # m
        self.y = 0.0  # m
        self.heading = 0.0  # rad
        
        # Velocity components
        self.vx = 0.0  # m/s
        self.vy = 0.0  # m/s
        self.yaw_rate = 0.0  # rad/s
        
        # Accelerations
        self.ax = 0.0  # m/s²
        self.ay = 0.0  # m/s²
        
        # Powertrain state
        self.engine_rpm = 1000.0  # RPM
        self.gear = 1  # Current gear
        self.throttle = 0.0  # 0-1
        self.brake = 0.0  # 0-1
        self.steering = 0.0  # rad
        
        # Thermal state (optional)
        self.engine_temp = 85.0  # °C
        self.coolant_temp = 85.0  # °C
        
        # Derived values
        self.speed = 0.0  # m/s
        self.lateral_accel = 0.0  # m/s²
        self.longitudinal_accel = 0.0  # m/s²
        
        # Track-related state
        self.distance = 0.0  # Distance along track
        self.track_position = 0.0  # -1 to 1, where 0 is centerline
    
    def update_derived_values(self):
        """Update derived values based on primary state variables."""
        self.speed = np.sqrt(self.vx**2 + self.vy**2)
        
        if self.speed > 0.1:  # Avoid division by zero
            # Calculate accelerations in vehicle frame
            heading_sin = np.sin(self.heading)
            heading_cos = np.cos(self.heading)
            
            # Rotate accelerations to vehicle frame
            self.longitudinal_accel = self.ax * heading_cos + self.ay * heading_sin
            self.lateral_accel = -self.ax * heading_sin + self.ay * heading_cos
    
    def to_array(self) -> np.ndarray:
        """Convert state to numpy array for integration."""
        return np.array([
            self.x, self.y, self.heading,
            self.vx, self.vy, self.yaw_rate,
            self.engine_rpm, self.throttle, self.brake, self.steering
        ])
    
    @classmethod
    def from_array(cls, array: np.ndarray) -> 'VehicleState':
        """Create state from numpy array."""
        state = cls()
        state.x = array[0]
        state.y = array[1]
        state.heading = array[2]
        state.vx = array[3]
        state.vy = array[4]
        state.yaw_rate = array[5]
        state.engine_rpm = array[6]
        state.throttle = array[7]
        state.brake = array[8]
        state.steering = array[9]
        state.update_derived_values()
        return state


class ControlInputs:
    """Represents control inputs to the vehicle at a point in time."""
    
    def __init__(self):
        self.throttle = 0.0  # 0-1
        self.brake = 0.0  # 0-1
        self.steering = 0.0  # rad
        self.gear = 1  # Current gear
    
    def to_array(self) -> np.ndarray:
        """Convert controls to numpy array."""
        return np.array([self.throttle, self.brake, self.steering, self.gear])
    
    @classmethod
    def from_array(cls, array: np.ndarray) -> 'ControlInputs':
        """Create controls from numpy array."""
        controls = cls()
        controls.throttle = array[0]
        controls.brake = array[1]
        controls.steering = array[2]
        controls.gear = int(array[3])
        return controls


class OptimalLapTimeOptimizer:
    """
    Advanced optimizer for finding the optimal racing line and control inputs
    to minimize lap time using numerical optimization methods.
    """
    
    def __init__(self, vehicle: Vehicle, track_profile: TrackProfile):
        """
        Initialize the optimizer with a vehicle model and track profile.
        
        Args:
            vehicle: Vehicle model to use for simulation
            track_profile: Track profile object
        """
        self.vehicle = vehicle
        self.track_profile = track_profile
        self.track_data = track_profile.get_track_data()
        
        # Integration settings
        self.dt = 0.01  # s
        self.max_time = 180.0  # s
        self.include_thermal = True
        
        # Optimization settings
        self.max_iterations = 50
        self.tolerance = 1e-4
        self.optimization_method = 'SLSQP'  # or 'Newton-CG', 'trust-constr', etc.
        
        # Initialize vehicle dynamics model
        self.cornering = CorneringPerformance(vehicle)
        
        # Initialize track interpolation
        self._initialize_track_interpolation()
        
        # Parameterization
        self.num_control_points = 50  # Number of control points for optimization
        
        # Results
        self.optimal_racing_line = None
        self.optimal_controls = None
        self.optimal_lap_time = None
        
        logger.info("Optimal lap time optimizer initialized")
    
    def _initialize_track_interpolation(self):
        """Initialize track centerline and width interpolation functions."""
        # Preprocess track data to remove duplicates
        self.track_data = preprocess_track_points(self.track_data)
        
        # Get track points and width
        points = self.track_data['points']
        width = self.track_data.get('width', np.full(len(points), 3.0))
        distances = self.track_data['distance']
        
        # Calculate curvature if not present in track data
        if 'curvature' not in self.track_data:
            # Calculate curvature (simple 3-point method)
            curvature = np.zeros(len(points))
            for i in range(1, len(points) - 1):
                # Get three consecutive points
                p1 = points[i-1]
                p2 = points[i]
                p3 = points[i+1]
                
                # Calculate vectors
                v1 = p2 - p1
                v2 = p3 - p2
                
                # Calculate angle change
                dot = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                dot = np.clip(dot, -1.0, 1.0)  # Ensure valid input for arccos
                angle = np.arccos(dot)
                
                # Calculate distance
                ds = (np.linalg.norm(v1) + np.linalg.norm(v2)) / 2
                
                # Curvature is angle change per distance
                curvature[i] = angle / ds
                
            # Handle endpoints
            curvature[0] = curvature[1]
            curvature[-1] = curvature[-2]
            
            # Store in track data
            self.track_data['curvature'] = curvature
        else:
            curvature = self.track_data['curvature']
        
        # Ensure track is closed (last point connects to first)
        if np.linalg.norm(points[0] - points[-1]) > 1e-3:
            points = np.vstack([points, points[0]])
            width = np.append(width, width[0])
            curvature = np.append(curvature, curvature[0])
            distances = np.append(distances, distances[-1] + np.linalg.norm(points[0] - points[-1]))
        
        # Ensure distances are unique for interpolation
        distances = ensure_unique_values(distances)
        
        # Create interpolation functions
        self.track_x_interp = interp1d(distances, points[:, 0], kind='cubic', bounds_error=False, fill_value='extrapolate')
        self.track_y_interp = interp1d(distances, points[:, 1], kind='cubic', bounds_error=False, fill_value='extrapolate')
        self.track_width_interp = interp1d(distances, width, kind='linear', bounds_error=False, fill_value='extrapolate')
        self.track_curvature_interp = interp1d(distances, curvature, kind='linear', bounds_error=False, fill_value='extrapolate')
        
        # Store track length
        self.track_length = distances[-1]
    
    def _vehicle_dynamics_derivatives(self, state: VehicleState, controls: ControlInputs) -> np.ndarray:
        """
        Calculate the derivatives of the vehicle state for integration.
        
        Args:
            state: Current vehicle state
            controls: Control inputs
            
        Returns:
            Array of derivatives for integration
        """
        # Extract state variables
        x, y, heading = state.x, state.y, state.heading
        vx, vy, yaw_rate = state.vx, state.vy, state.yaw_rate
        engine_rpm = state.engine_rpm
        
        # Extract control inputs
        throttle = controls.throttle
        brake = controls.brake
        steering = controls.steering
        gear = controls.gear
        
        # Calculate vehicle speed
        speed = np.sqrt(vx**2 + vy**2)
        
        # Position derivatives are simply velocity components
        dx_dt = vx
        dy_dt = vy
        dheading_dt = yaw_rate
        
        # Calculate forces
        # For a more detailed model, you would compute tire forces, aerodynamic forces, etc.
        # This is a simplified model
        
        # Longitudinal force (throttle and brake)
        if speed > 0.1:  # Avoid division by zero
            # Calculate engine torque
            engine_torque = self.vehicle.engine.get_torque(engine_rpm, throttle)
            
            # Calculate wheel torque
            wheel_torque = self.vehicle.drivetrain.calculate_wheel_torque(engine_torque, gear)
            
            # Calculate longitudinal force
            F_longitudinal = wheel_torque / self.vehicle.tire_radius
            
            # Add brake force (simplified)
            F_brake = -brake * self.vehicle.mass * 9.81 * 1.5  # Assuming 1.5g max deceleration
            
            # Total longitudinal force
            F_total_long = F_longitudinal + F_brake
        else:
            # Simplified starting force
            F_total_long = throttle * self.vehicle.mass * 5.0  # Simplified starting acceleration
        
        # Lateral force from steering
        # Simplified bicycle model
        if speed > 0.1:
            # Calculate slip angle (simplified)
            slip_angle = steering * self.vehicle.wheelbase / speed
            
            # Calculate lateral force (simplified tire model)
            cornering_stiffness = 20000.0  # N/rad, typical value
            F_lateral = -cornering_stiffness * slip_angle
            
            # Limit lateral force by tire grip
            max_lateral_force = self.vehicle.mass * self.cornering.calculate_max_lateral_acceleration(speed)
            F_lateral = np.clip(F_lateral, -max_lateral_force, max_lateral_force)
        else:
            F_lateral = 0.0
        
        # Calculate accelerations
        # Rotate forces to global coordinates
        F_x = F_total_long * np.cos(heading) - F_lateral * np.sin(heading)
        F_y = F_total_long * np.sin(heading) + F_lateral * np.cos(heading)
        
        # Add drag force
        if speed > 0.1:
            drag_direction_x = vx / speed
            drag_direction_y = vy / speed
            
            drag_coefficient = self.vehicle.drag_coefficient
            frontal_area = self.vehicle.frontal_area
            air_density = 1.225  # kg/m³
            
            drag_magnitude = 0.5 * air_density * drag_coefficient * frontal_area * speed**2
            
            F_x -= drag_magnitude * drag_direction_x
            F_y -= drag_magnitude * drag_direction_y
        
        # Calculate acceleration
        dvx_dt = F_x / self.vehicle.mass
        dvy_dt = F_y / self.vehicle.mass
        
        # Yaw acceleration (simplified)
        if speed > 0.1:
            dyaw_rate_dt = F_lateral * self.vehicle.wheelbase / (self.vehicle.mass * speed)
        else:
            dyaw_rate_dt = 0.0
        
        # Engine RPM derivative
        if gear > 0:
            # Calculate wheel speed
            wheel_rpm = speed / (2 * np.pi * self.vehicle.tire_radius) * 60
            
            # Calculate engine speed
            target_rpm = wheel_rpm * self.vehicle.drivetrain.get_overall_ratio(gear)
            
            # RPM approaches target with some lag
            rpm_lag = 0.5  # s
            dengine_rpm_dt = (target_rpm - engine_rpm) / rpm_lag
        else:
            # In neutral, engine RPM decays
            dengine_rpm_dt = -engine_rpm / 2.0
        
        # Throttle, brake, and steering derivatives (these would come from control inputs)
        dthrottle_dt = 0.0
        dbrake_dt = 0.0
        dsteering_dt = 0.0
        
        # Return the derivatives as an array
        return np.array([
            dx_dt, dy_dt, dheading_dt,
            dvx_dt, dvy_dt, dyaw_rate_dt,
            dengine_rpm_dt, dthrottle_dt, dbrake_dt, dsteering_dt
        ])
    
    def _integrate_rk4(self, state: VehicleState, controls: ControlInputs, dt: float) -> VehicleState:
        """
        Integrate vehicle dynamics using 4th order Runge-Kutta method.
        
        Args:
            state: Current vehicle state
            controls: Control inputs
            dt: Time step
            
        Returns:
            New vehicle state after integration
        """
        # Convert to numpy arrays for integration
        y = state.to_array()
        
        # RK4 integration
        k1 = self._vehicle_dynamics_derivatives(state, controls)
        
        # K2 step
        state2 = VehicleState.from_array(y + 0.5 * dt * k1)
        k2 = self._vehicle_dynamics_derivatives(state2, controls)
        
        # K3 step
        state3 = VehicleState.from_array(y + 0.5 * dt * k2)
        k3 = self._vehicle_dynamics_derivatives(state3, controls)
        
        # K4 step
        state4 = VehicleState.from_array(y + dt * k3)
        k4 = self._vehicle_dynamics_derivatives(state4, controls)
        
        # Final integration step
        y_new = y + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        # Convert back to state object
        new_state = VehicleState.from_array(y_new)
        
        # Update derived values
        new_state.update_derived_values()
        
        return new_state
    
    def _racing_line_from_parameters(self, parameters: np.ndarray) -> np.ndarray:
        """
        Convert optimization parameters to a racing line.
        
        Args:
            parameters: Optimization parameters
            
        Returns:
            Racing line as array of (x, y) points
        """
        # Parameters represent track position (-1 to 1) at control points
        track_positions = parameters[:self.num_control_points]
        
        # Clamp values to valid range
        track_positions = np.clip(track_positions, -0.9, 0.9)
        
        # Create distance points along track
        distances = np.linspace(0, self.track_length, self.num_control_points)
        # Ensure distances are unique
        distances = ensure_unique_values(distances)
        # Interpolate track positions for a dense set of points
        dense_distances = np.linspace(0, self.track_length, 500)
        position_interp = interp1d(distances, track_positions, kind='cubic',
                                  bounds_error=False, fill_value='extrapolate')
        dense_positions = position_interp(dense_distances)
        
        # Convert to x, y coordinates
        racing_line = np.zeros((len(dense_distances), 2))
        
        for i, (s, pos) in enumerate(zip(dense_distances, dense_positions)):
            # Get centerline point and width
            x_center = self.track_x_interp(s)
            y_center = self.track_y_interp(s)
            width = self.track_width_interp(s)
            
            # Calculate normal vector
            ds = 0.1  # Small distance for numerical derivative
            x_next = self.track_x_interp(s + ds)
            y_next = self.track_y_interp(s + ds)
            
            # Track direction
            tx = x_next - x_center
            ty = y_next - y_center
            
            # Normalize
            norm = np.sqrt(tx**2 + ty**2)
            if norm > 1e-6:
                tx /= norm
                ty /= norm
            
            # Normal vector (90 degrees to track direction)
            nx = -ty
            ny = tx
            
            # Calculate racing line point
            racing_line[i, 0] = x_center + pos * width/2 * nx
            racing_line[i, 1] = y_center + pos * width/2 * ny
        
        return racing_line
    
    def _controls_from_parameters(self, parameters: np.ndarray) -> List[ControlInputs]:
        """
        Convert optimization parameters to control inputs.
        
        Args:
            parameters: Optimization parameters
            
        Returns:
            List of control inputs along the racing line
        """
        # Extract control parameters
        throttle_params = parameters[self.num_control_points:2*self.num_control_points]
        brake_params = parameters[2*self.num_control_points:3*self.num_control_points]
        
        # Clamp values to valid range
        throttle_params = np.clip(throttle_params, 0.0, 1.0)
        brake_params = np.clip(brake_params, 0.0, 1.0)
        
        # Create distance points along track
        distances = np.linspace(0, self.track_length, self.num_control_points)
        
        # Interpolate control inputs for a dense set of points
        dense_distances = np.linspace(0, self.track_length, 500)
        
        throttle_interp = interp1d(distances, throttle_params, kind='linear',
                                  bounds_error=False, fill_value='extrapolate')
        brake_interp = interp1d(distances, brake_params, kind='linear',
                              bounds_error=False, fill_value='extrapolate')
        
        dense_throttle = throttle_interp(dense_distances)
        dense_brake = brake_interp(dense_distances)
        
        # Create control inputs
        controls = []
        for i, (s, throttle, brake) in enumerate(zip(dense_distances, dense_throttle, dense_brake)):
            # Calculate steering based on racing line curvature (simplified)
            control = ControlInputs()
            control.throttle = throttle
            control.brake = brake
            
            # Steering will be calculated during simulation based on racing line
            control.steering = 0.0
            
            # Gear will be calculated during simulation
            control.gear = 1
            
            controls.append(control)
        
        return controls
    
    def _simulate_lap_with_params(self, parameters: np.ndarray) -> Tuple[float, List[VehicleState]]:
        """
        Simulate a lap with the given parameters.
        
        Args:
            parameters: Optimization parameters
            
        Returns:
            Tuple of (lap time, list of vehicle states)
        """
        # Convert parameters to racing line and controls
        racing_line = self._racing_line_from_parameters(parameters)
        base_controls = self._controls_from_parameters(parameters)
        
        # Initial state
        state = VehicleState()
        state.x = racing_line[0, 0]
        state.y = racing_line[0, 1]
        
        # Calculate initial heading
        dx = racing_line[1, 0] - racing_line[0, 0]
        dy = racing_line[1, 1] - racing_line[0, 1]
        state.heading = np.arctan2(dy, dx)
        
        # Storage for states
        states = [state]
        
        # Simulation parameters
        dt = self.dt
        current_time = 0.0
        current_distance = 0.0
        
        # Lap is complete when we reach the track length
        while current_distance < self.track_length and current_time < self.max_time:
            # Find closest point on racing line
            distances = np.linalg.norm(racing_line - np.array([state.x, state.y]), axis=1)
            closest_idx = np.argmin(distances)
            
            # Get track distance
            track_distance = (closest_idx / len(racing_line)) * self.track_length
            
            # Calculate look-ahead point
            look_ahead_idx = min(closest_idx + 10, len(racing_line) - 1)
            target_x = racing_line[look_ahead_idx, 0]
            target_y = racing_line[look_ahead_idx, 1]
            
            # Calculate heading to target
            dx = target_x - state.x
            dy = target_y - state.y
            target_heading = np.arctan2(dy, dx)
            
            # Calculate steering to achieve target heading
            # Simplified proportional controller
            heading_error = target_heading - state.heading
            # Normalize to -pi to pi
            while heading_error > np.pi:
                heading_error -= 2*np.pi
            while heading_error < -np.pi:
                heading_error += 2*np.pi
            
            # Calculate steering angle
            steering = 0.5 * heading_error  # Proportional gain
            
            # Get base control inputs
            control = base_controls[closest_idx].to_array()
            
            # Override steering
            control_obj = ControlInputs.from_array(control)
            control_obj.steering = steering
            
            # Determine optimal gear based on speed
            if state.speed > 0.1:
                # Get optimal gear
                optimal_gear = self._estimate_optimal_gear(state.speed, state.engine_rpm)
                control_obj.gear = optimal_gear
            
            # Ensure we don't both throttle and brake simultaneously
            if control_obj.throttle > 0 and control_obj.brake > 0:
                if control_obj.throttle > control_obj.brake:
                    control_obj.brake = 0
                else:
                    control_obj.throttle = 0
            
            # Integrate dynamics
            new_state = self._integrate_rk4(state, control_obj, dt)
            
            # Update time and distance
            current_time += dt
            distance_step = np.sqrt((new_state.x - state.x)**2 + (new_state.y - state.y)**2)
            current_distance += distance_step
            
            # Store distance in state
            new_state.distance = current_distance
            
            # Check if we've run off track
            track_position = self._calculate_track_position(new_state.x, new_state.y)
            if abs(track_position) > 1.0:
                # Off track - large penalty
                return current_time + 1000.0, states
            
            # Update state
            state = new_state
            states.append(state)
        
        return current_time, states
    
    def _estimate_optimal_gear(self, speed: float, current_rpm: float) -> int:
        """
        Estimate the optimal gear for a given speed.
        
        Args:
            speed: Vehicle speed in m/s
            current_rpm: Current engine RPM
            
        Returns:
            Optimal gear
        """
        # Similar to the method in LapTimeSimulator but simplified
        
        # Default to first gear
        optimal_gear = 1
        
        # Only calculate if speed is significant
        if speed < 0.5:
            return optimal_gear
        
        # Check if current RPM is below a threshold for upshift or above for downshift
        if current_rpm > 0.9 * self.vehicle.engine.redline:
            # Need to upshift if not in highest gear
            if optimal_gear < self.vehicle.drivetrain.transmission.num_gears:
                optimal_gear += 1
        elif current_rpm < 0.6 * self.vehicle.engine.max_power_rpm:
            # Need to downshift if not in first gear
            if optimal_gear > 1:
                optimal_gear -= 1
        
        # More sophisticated logic would look at expected RPM in each gear
        for gear in range(1, self.vehicle.drivetrain.transmission.num_gears + 1):
            # Calculate engine RPM in this gear at the current speed
            engine_rpm = self.vehicle.drivetrain.calculate_engine_speed(speed, gear)
            
            # Skip if RPM is below idle or above redline
            if engine_rpm < self.vehicle.engine.idle_rpm or engine_rpm > self.vehicle.engine.redline:
                continue
            
            # Check if this gear keeps RPM near the power band
            if self.vehicle.engine.max_torque_rpm * 0.8 <= engine_rpm <= self.vehicle.engine.max_power_rpm * 1.1:
                optimal_gear = gear
                break
        
        return optimal_gear
    
    def _calculate_track_position(self, x: float, y: float) -> float:
        """
        Calculate the position relative to track centerline (-1 to 1).
        
        Args:
            x: X-coordinate
            y: Y-coordinate
            
        Returns:
            Track position (-1 to 1, where 0 is centerline)
        """
        # Find closest point on centerline
        min_dist = float('inf')
        min_idx = 0
        
        # Get track points
        track_points = self.track_data['points']
        
        for i, point in enumerate(track_points):
            dist = np.sqrt((x - point[0])**2 + (y - point[1])**2)
            if dist < min_dist:
                min_dist = dist
                min_idx = i
        
        # Get adjacent points
        next_idx = (min_idx + 1) % len(track_points)
        prev_idx = (min_idx - 1) % len(track_points)
        
        # Calculate track direction
        dx = track_points[next_idx, 0] - track_points[min_idx, 0]
        dy = track_points[next_idx, 1] - track_points[min_idx, 1]
        track_direction = np.array([dx, dy])
        track_direction = track_direction / np.linalg.norm(track_direction)
        
        # Calculate normal vector
        track_normal = np.array([-track_direction[1], track_direction[0]])
        
        # Calculate vector from track point to position
        pos_vector = np.array([x - track_points[min_idx, 0], y - track_points[min_idx, 1]])
        
        # Project onto normal vector
        distance_from_centerline = np.dot(pos_vector, track_normal)
        
        # Normalize by track width
        track_width = self.track_data.get('width', np.full(len(track_points), 3.0))[min_idx]
        normalized_position = 2.0 * distance_from_centerline / track_width
        
        return normalized_position
    
    def _objective_function(self, parameters: np.ndarray) -> float:
        """
        Objective function for optimization (lap time).
        
        Args:
            parameters: Optimization parameters
            
        Returns:
            Lap time
        """
        try:
            lap_time, _ = self._simulate_lap_with_params(parameters)
            logger.debug(f"Lap time: {lap_time:.3f}s")
            return lap_time
        except Exception as e:
            logger.error(f"Error in simulation: {str(e)}")
            return 1000.0  # Large penalty for failed simulations
    
    def optimize_lap_time(self) -> Dict:
        """
        Find the optimal racing line and control inputs to minimize lap time.
        
        Returns:
            Dictionary with optimization results
        """
        # Set up initial parameters
        # Initialize racing line to track centerline (0) with some noise
        racing_line_params = np.random.normal(0, 0.1, self.num_control_points)
        
        # Initialize controls based on track curvature
        distances = np.linspace(0, self.track_length, self.num_control_points)
        curvature = np.abs([self.track_curvature_interp(d) for d in distances])
        
        # High throttle on straights, low on corners
        throttle_params = 1.0 - 0.7 * curvature / np.max(curvature)
        throttle_params = np.clip(throttle_params, 0.2, 1.0)
        
        # Brake before corners
        brake_positions = np.roll(curvature, -3)  # Shift to brake before curves
        brake_params = 0.8 * brake_positions / np.max(brake_positions)
        brake_params = np.clip(brake_params, 0.0, 0.8)
        
        # Combine parameters
        initial_params = np.concatenate([racing_line_params, throttle_params, brake_params])
        
        # Set up bounds for parameters
        lower_bounds = np.concatenate([
            np.full(self.num_control_points, -0.9),  # Racing line (-0.9 to 0.9)
            np.full(self.num_control_points, 0.0),   # Throttle (0 to 1)
            np.full(self.num_control_points, 0.0)    # Brake (0 to 1)
        ])
        
        upper_bounds = np.concatenate([
            np.full(self.num_control_points, 0.9),  # Racing line (-0.9 to 0.9)
            np.full(self.num_control_points, 1.0),  # Throttle (0 to 1)
            np.full(self.num_control_points, 1.0)   # Brake (0 to 1)
        ])
        
        bounds = Bounds(lower_bounds, upper_bounds)
        
        # Set up optimization
        logger.info("Starting lap time optimization...")
        start_time = time.time()
        
        try:
            # Run optimization
            result = minimize(
                self._objective_function,
                initial_params,
                method=self.optimization_method,
                bounds=bounds,
                options={
                    'maxiter': self.max_iterations,
                    'ftol': self.tolerance,
                    'disp': True
                }
            )
            
            optimization_time = time.time() - start_time
            logger.info(f"Optimization completed in {optimization_time:.1f}s")
            
            # Get optimized parameters
            optimized_params = result.x
            
            # Get racing line and controls
            racing_line = self._racing_line_from_parameters(optimized_params)
            controls = self._controls_from_parameters(optimized_params)
            
            # Run final simulation
            lap_time, states = self._simulate_lap_with_params(optimized_params)
            
            # Store results
            self.optimal_racing_line = racing_line
            self.optimal_controls = controls
            self.optimal_lap_time = lap_time
            
            # Create results dictionary
            results = {
                'lap_time': lap_time,
                'racing_line': racing_line,
                'parameters': optimized_params,
                'optimization_success': result.success,
                'optimization_message': result.message,
                'optimization_time': optimization_time,
                'vehicle_states': states
            }
            
            logger.info(f"Optimal lap time: {lap_time:.3f}s")
            
            return results
            
        except Exception as e:
            logger.error(f"Optimization failed: {str(e)}")
            return {
                'lap_time': None,
                'racing_line': None,
                'parameters': None,
                'optimization_success': False,
                'optimization_message': str(e),
                'optimization_time': time.time() - start_time
            }
    
    def visualize_optimization_results(self, results: Dict, save_path: Optional[str] = None):
        """
        Visualize optimization results.
        
        Args:
            results: Results from optimize_lap_time
            save_path: Optional path to save the visualization
        """
        if not results['racing_line'] is not None:
            logger.error("No valid optimization results to visualize")
            return
        
        # Extract data
        racing_line = results['racing_line']
        lap_time = results['lap_time']
        states = results.get('vehicle_states', [])
        
        # Create figure
        plt.figure(figsize=(15, 10))
        
        # Plot track and racing line
        ax1 = plt.subplot(221)
        
        # Plot track centerline
        track_points = self.track_data['points']
        ax1.plot(track_points[:, 0], track_points[:, 1], 'k--', alpha=0.5, label='Track Centerline')
        
        # Plot optimal racing line
        ax1.plot(racing_line[:, 0], racing_line[:, 1], 'r-', linewidth=2, label='Optimal Racing Line')
        
        # Plot vehicle positions if available
        if states:
            # Sample states for cleaner visualization
            sample_rate = max(1, len(states) // 100)
            sampled_states = states[::sample_rate]
            
            # Extract positions
            x_pos = [s.x for s in sampled_states]
            y_pos = [s.y for s in sampled_states]
            
            # Plot vehicle path
            ax1.plot(x_pos, y_pos, 'b.', markersize=4, alpha=0.7, label='Vehicle Path')
        
        ax1.set_aspect('equal')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_title(f'Optimal Racing Line - Lap Time: {lap_time:.3f}s')
        ax1.legend()
        ax1.grid(True)
        
        # Plot speed profile
        if states:
            ax2 = plt.subplot(222)
            
            # Extract data
            distances = [s.distance for s in states]
            speeds = [s.speed * 3.6 for s in states]  # Convert to km/h
            
            ax2.plot(distances, speeds, 'g-', linewidth=2)
            ax2.set_xlabel('Distance (m)')
            ax2.set_ylabel('Speed (km/h)')
            ax2.set_title('Speed Profile')
            ax2.grid(True)
            
            # Plot controls
            ax3 = plt.subplot(223)
            
            # Extract control inputs
            throttle = [s.throttle for s in states]
            brake = [s.brake for s in states]
            
            ax3.plot(distances, throttle, 'g-', linewidth=2, label='Throttle')
            ax3.plot(distances, brake, 'r-', linewidth=2, label='Brake')
            ax3.set_xlabel('Distance (m)')
            ax3.set_ylabel('Control Input')
            ax3.set_title('Control Inputs')
            ax3.set_ylim(0, 1.05)
            ax3.legend()
            ax3.grid(True)
            
            # Plot accelerations
            ax4 = plt.subplot(224)
            
            # Extract accelerations
            longitudinal_accel = [s.longitudinal_accel for s in states]
            lateral_accel = [s.lateral_accel for s in states]
            
            # Convert to g
            longitudinal_g = [a / 9.81 for a in longitudinal_accel]
            lateral_g = [a / 9.81 for a in lateral_accel]
            
            ax4.plot(distances, longitudinal_g, 'b-', linewidth=2, label='Longitudinal G')
            ax4.plot(distances, lateral_g, 'm-', linewidth=2, label='Lateral G')
            ax4.set_xlabel('Distance (m)')
            ax4.set_ylabel('Acceleration (g)')
            ax4.set_title('Vehicle Dynamics')
            ax4.legend()
            ax4.grid(True)
        
        plt.tight_layout()
        
        # Save figure if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Visualization saved to {save_path}")
        
        plt.show()


def run_advanced_lap_optimization(vehicle: Vehicle, track_profile: TrackProfile,
                                save_dir: Optional[str] = None) -> Dict:
    """
    Run advanced lap time optimization.
    
    Args:
        vehicle: Vehicle model
        track_profile: Track profile
        save_dir: Optional directory to save results
        
    Returns:
        Dictionary with optimization results
    """
    # Create optimizer
    optimizer = OptimalLapTimeOptimizer(vehicle, track_profile)
    
    # Run optimization
    results = optimizer.optimize_lap_time()
    
    # Save results if directory provided
    if save_dir and results['racing_line'] is not None:
        import os
        
        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Save visualization
        optimizer.visualize_optimization_results(
            results,
            save_path=os.path.join(save_dir, "optimal_lap.png")
        )
        
        # Save racing line to CSV
        if results['racing_line'] is not None:
            np.savetxt(
                os.path.join(save_dir, "optimal_racing_line.csv"),
                results['racing_line'],
                delimiter=',',
                header='x,y'
            )
        
        # Save parameters
        if results['parameters'] is not None:
            np.savetxt(
                os.path.join(save_dir, "optimal_parameters.csv"),
                results['parameters'],
                delimiter=','
            )
    
    return results


# Example usage
if __name__ == "__main__":
    from ..core.vehicle import create_formula_student_vehicle
    from ..core.track_integration import TrackProfile, create_example_track
    import tempfile
    import os.path
    
    print("Advanced Lap Time Optimization Demo")
    print("-----------------------------------")
    
    # Create a Formula Student vehicle
    vehicle = create_formula_student_vehicle()
    
    # Create a temporary directory for outputs
    output_dir = tempfile.mkdtemp()
    print(f"Creating output directory: {output_dir}")
    
    # Create an example track
    track_file = os.path.join(output_dir, "example_track.yaml")
    print("Generating example track...")
    create_example_track(track_file, difficulty='easy')  # Use easy track for faster optimization
    
    # Load track profile
    track_profile = TrackProfile(track_file)
    
    # Basic vehicle specs
    print("\nVehicle Specifications:")
    print(f"  Engine: {vehicle.engine.make} {vehicle.engine.model}")
    print(f"  Power: {vehicle.engine.max_power} hp @ {vehicle.engine.max_power_rpm} RPM")
    print(f"  Mass: {vehicle.mass} kg")
    
    # Run optimization with reduced complexity for demo
    print("\nRunning advanced lap time optimization (this may take several minutes)...")
    optimizer = OptimalLapTimeOptimizer(vehicle, track_profile)
    optimizer.num_control_points = 20  # Reduce complexity for faster demo
    optimizer.max_iterations = 10      # Reduce iterations for faster demo
    
    results = optimizer.optimize_lap_time()
    
    # Print results
    print("\nOptimization Results:")
    if results['lap_time'] is not None:
        print(f"  Optimal Lap Time: {results['lap_time']:.3f} seconds")
        print(f"  Optimization Success: {results['optimization_success']}")
        print(f"  Optimization Time: {results['optimization_time']:.1f} seconds")
    else:
        print("  Optimization failed")
    
    # Visualize results
    if results['racing_line'] is not None:
        save_path = os.path.join(output_dir, "optimal_lap.png")
        optimizer.visualize_optimization_results(results, save_path=save_path)
        print(f"\nVisualization saved to: {save_path}")
    
    print(f"\nAll results saved to: {output_dir}")