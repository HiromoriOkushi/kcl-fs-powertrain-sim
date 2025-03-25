"""
Lap time simulation module for Formula Student powertrain.

This module provides classes and functions for simulating, analyzing, and optimizing
the lap time performance of a Formula Student vehicle. It includes specialized
tools for racing line optimization, cornering performance analysis, and
visualization of lap time data.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Callable
import logging
from scipy.interpolate import interp1d, CubicSpline
from scipy.optimize import minimize
import yaml

from ..core.vehicle import Vehicle
from ..core.track_integration import TrackProfile, calculate_optimal_racing_line
from ..transmission import CASSystem, ShiftDirection, StrategyType, ShiftStrategy
from ..engine import TorqueCurve
from ..utils.track_utils import preprocess_track_points

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("Lap_Time_Simulation")


class LapTimeSimulator:
    """
    Simulator for Formula Student lap time performance.
    
    This class provides tools for simulating a vehicle's performance around a track,
    optimizing racing lines, and analyzing lap times. It can account for vehicle
    dynamics, powertrain characteristics, and thermal management effects.
    """
    
    def __init__(self, vehicle: Vehicle, track_profile: Optional[TrackProfile] = None, track_file: Optional[str] = None):
        """
        Initialize the lap time simulator with a vehicle model and track.
        
        Args:
            vehicle: Vehicle model to use for simulation
            track_profile: Optional TrackProfile object
            track_file: Optional path to track file
        """
        self.vehicle = vehicle
        self.track_profile = track_profile
        
        # Track data storage
        self.track_data = None
        self.racing_line = None
        self.speed_profile = None
        self.sector_times = None
        self.lap_time = None
        self.time_profile = None
        
        # Simulation parameters
        self.time_step = 0.01  # s
        self.max_time = 120.0  # s
        self.include_thermal = True
        
        # Initialize cornering performance calculator
        self.cornering = CorneringPerformance(vehicle)
        
        # Load track if file provided
        if track_file and not track_profile:
            self.load_track(track_file)
        elif track_profile:
            self.track_profile = track_profile
            self.track_data = track_profile.get_track_data()
        
        logger.info("Lap time simulator initialized")
    
    def load_track(self, track_file: str):
        """
        Load track data from file.
        
        Args:
            track_file: Path to track file
        """
        try:
            self.track_profile = TrackProfile(track_file)
            self.track_data = self.track_profile.get_track_data()
            # Preprocess track data to remove duplicates
            self.track_data = preprocess_track_points(self.track_data)
            logger.info(f"Loaded track from {track_file}")
        except Exception as e:
            logger.error(f"Error loading track: {str(e)}")
            raise
    
    def calculate_racing_line(self):
        """
        Calculate optimized racing line for the track.
        
        Returns:
            NumPy array of racing line points
        """
        if not self.track_profile:
            raise ValueError("No track loaded")
        
        try:
            # Preprocess track data to remove duplicates
            if self.track_data:
                self.track_data = preprocess_track_points(self.track_data)
            
            # Ensure track profile data is also preprocessed
            if hasattr(self.track_profile, 'track_data'):
                self.track_profile.track_data = preprocess_track_points(self.track_profile.track_data)
            
            # Use the optimal racing line function from track_integration
            self.racing_line = calculate_optimal_racing_line(self.track_profile)
            logger.info("Racing line optimized")
            return self.racing_line
        except Exception as e:
            logger.error(f"Error calculating racing line: {str(e)}")
            # Fall back to centerline if racing line can't be calculated
            self.racing_line = self.track_data['points']
            return self.racing_line
        
    def calculate_speed_profile(self, use_optimized_shifts: bool = True) -> np.ndarray:
        """
        Calculate speed profile along the track.
        
        Args:
            use_optimized_shifts: Whether to use optimized shift points
            
        Returns:
            Array of speeds at each track point
        """
        if not self.track_profile:
            raise ValueError("No track loaded")
        
        if self.racing_line is None:
            # Calculate racing line first
            self.calculate_racing_line()
        
        # Extract track data
        points = self.track_data['points']
        curvature = self.track_data['curvature']
        distances = self.track_data['distance']
        
        # Number of points
        n_points = len(points)
        
        # Initialize speed profile
        speed_profile = np.zeros(n_points)
        
        # Vehicle parameters for cornering
        max_lat_accel = self.cornering.calculate_max_lateral_acceleration()
        
        # First pass - calculate maximum speed based on cornering
        for i in range(n_points):
            if abs(curvature[i]) > 1e-6:  # Avoid division by zero
                # Calculate corner radius (r = 1/curvature)
                radius = 1.0 / abs(curvature[i])
                
                # Calculate maximum cornering speed
                max_corner_speed = self.cornering.calculate_max_cornering_speed(radius)
            else:
                # Straight section - limited by powertrain
                max_corner_speed = float('inf')
            
            # Set initial speed profile
            speed_profile[i] = max_corner_speed
        
        # Apply maximum speed limit of the vehicle
        max_speed = self.calculate_max_speed()
        speed_profile = np.minimum(speed_profile, max_speed)
        
        # Second pass - forward simulation with acceleration limits
        for i in range(1, n_points):
            # Distance to next point
            distance = distances[i] - distances[i-1]
            
            # Maximum achievable speed with acceleration
            current_speed = speed_profile[i-1]
            
            # Determine appropriate gear
            if use_optimized_shifts and hasattr(self.vehicle, 'get_optimal_gear'):
                gear = self.vehicle.get_optimal_gear(current_speed)
            else:
                gear = self._estimate_optimal_gear(current_speed)
            
            # Calculate maximum acceleration
            throttle = 1.0  # Full throttle when accelerating
            brake = 0.0     # No braking when accelerating
            
            # Calculate acceleration at current state
            acceleration = self.calculate_max_acceleration(current_speed, gear)
            
            # Calculate maximum achievable speed
            # v² = u² + 2as
            max_accel_speed = np.sqrt(current_speed**2 + 2*acceleration*distance)
            
            # Update speed profile, not exceeding cornering limit
            speed_profile[i] = min(max_accel_speed, speed_profile[i])
        
        # Third pass - backward simulation with braking limits
        for i in range(n_points-2, -1, -1):
            # Distance between points
            distance = distances[i+1] - distances[i]
            
            # Maximum speed at current point based on braking to next point
            next_speed = speed_profile[i+1]
            
            # Calculate maximum deceleration
            max_decel = self.calculate_max_deceleration(next_speed)
            
            # Calculate speed limit based on braking
            # v² = u² - 2as
            braking_limited_speed = np.sqrt(next_speed**2 + 2*max_decel*distance)
            
            # Update speed profile
            speed_profile[i] = min(speed_profile[i], braking_limited_speed)
        
        # Store the calculated speed profile
        self.speed_profile = speed_profile
        
        logger.info("Speed profile calculated")
        
        return speed_profile
    
    def _estimate_optimal_gear(self, speed: float) -> int:
        """
        Estimate the optimal gear for a given speed.
        
        Args:
            speed: Vehicle speed in m/s
            
        Returns:
            Optimal gear
        """
        # Simple gear selection heuristic
        # This could be replaced with more sophisticated logic
        
        # Default to first gear
        optimal_gear = 1
        
        # Iterate through all gears to find the one that puts the engine
        # in the optimal RPM range (near max power)
        best_rpm_score = float('inf')
        target_rpm = self.vehicle.engine.max_power_rpm  # Target RPM for best power
        
        for gear in range(1, self.vehicle.drivetrain.transmission.num_gears + 1):
            # Calculate engine RPM in this gear at the current speed
            engine_rpm = self.vehicle.drivetrain.calculate_engine_speed(speed, gear)
            
            # Skip if RPM is below idle or above redline
            if engine_rpm < self.vehicle.engine.idle_rpm or engine_rpm > self.vehicle.engine.redline:
                continue
            
            # Score based on closeness to target RPM
            rpm_score = abs(engine_rpm - target_rpm)
            
            if rpm_score < best_rpm_score:
                best_rpm_score = rpm_score
                optimal_gear = gear
        
        return optimal_gear
    
    def calculate_max_speed(self) -> float:
        """
        Calculate maximum achievable speed based on vehicle parameters.
        
        Returns:
            Maximum speed in m/s
        """
        # Get vehicle parameters
        mass = self.vehicle.mass
        drag_coefficient = self.vehicle.drag_coefficient
        frontal_area = self.vehicle.frontal_area
        
        # Air density
        air_density = 1.225  # kg/m³
        
        # Get maximum engine power
        max_power = self.vehicle.engine.max_power * 745.7  # Convert hp to watts
        
        # Calculate maximum speed (solving: P = F_drag * v and F_drag = 0.5 * rho * Cd * A * v²)
        # This gives v³ = 2 * P / (rho * Cd * A)
        if drag_coefficient * frontal_area > 0:
            max_speed = (2 * max_power / (air_density * drag_coefficient * frontal_area)) ** (1/3)
        else:
            max_speed = 100.0  # Default if parameters are invalid
        
        # Apply efficiency factor for real-world conditions
        efficiency_factor = 0.9
        max_speed *= efficiency_factor
        
        logger.info(f"Calculated maximum speed: {max_speed*2.23694:.1f} mph")
        
        return max_speed
    
    def calculate_max_acceleration(self, speed: float, gear: int) -> float:
        """
        Calculate maximum acceleration at current speed and gear.
        
        Args:
            speed: Current speed in m/s
            gear: Current gear
            
        Returns:
            Maximum acceleration in m/s²
        """
        # Calculate engine RPM at current speed and gear
        engine_rpm = self.vehicle.drivetrain.calculate_engine_speed(speed, gear)
        
        # Get engine torque at current RPM
        engine_torque = self.vehicle.engine.get_torque(engine_rpm)
        
        # Calculate wheel torque
        wheel_torque = self.vehicle.drivetrain.calculate_wheel_torque(engine_torque, gear)
        
        # Calculate tractive force
        tractive_force = wheel_torque / self.vehicle.tire_radius
        
        # Calculate drag force
        air_density = 1.225  # kg/m³
        drag_force = 0.5 * air_density * self.vehicle.drag_coefficient * self.vehicle.frontal_area * speed**2
        
        # Calculate rolling resistance
        rolling_resistance = self.vehicle.mass * 9.81 * self.vehicle.rolling_resistance
        
        # Net force
        net_force = tractive_force - drag_force - rolling_resistance
        
        # Calculate acceleration (F = ma)
        acceleration = net_force / self.vehicle.mass
        
        return max(0, acceleration)  # Can't be negative
    
    def calculate_max_deceleration(self, speed: float) -> float:
        """
        Calculate maximum deceleration at current speed.
        
        Args:
            speed: Current speed in m/s
            
        Returns:
            Maximum deceleration in m/s² (negative value)
        """
        # For simplicity, assume a constant maximum deceleration
        # This could be replaced with a more sophisticated model based on
        # brake system characteristics, tire grip, etc.
        
        # Typical values for Formula Student car with good brakes
        base_deceleration = -20.0  # m/s²
        
        # Modify based on speed (less effective at very low speeds)
        if speed < 5.0:
            return base_deceleration * (0.5 + 0.5 * speed / 5.0)
        
        return base_deceleration
    
    def simulate_lap(self, include_thermal: bool = True) -> Dict:
        """
        Simulate a complete lap around the track.
        
        Args:
            include_thermal: Whether to include thermal effects
            
        Returns:
            Dictionary with simulation results
        """
        if not self.track_profile:
            raise ValueError("No track loaded")
        
        # Store thermal simulation parameter
        self.include_thermal = include_thermal
        
        # Calculate racing line if not already done
        if self.racing_line is None:
            self.calculate_racing_line()
        
        # Calculate speed profile if not already done
        if self.speed_profile is None:
            self.calculate_speed_profile()
        
        # Get track data
        distances = self.track_data['distance']
        curvature = self.track_data['curvature']
        
        # Reset vehicle state
        self.vehicle.current_speed = 0.0
        self.vehicle.current_position = 0.0
        self.vehicle.current_acceleration = 0.0
        self.vehicle.current_gear = 1
        self.vehicle.current_engine_rpm = self.vehicle.engine.idle_rpm
        
        # Initialize thermal state if including thermal effects
        if include_thermal and hasattr(self.vehicle, 'initialize_thermal_state'):
            self.vehicle.initialize_thermal_state()
        
        # Initialize storage for time-based simulation
        n_points = len(distances)
        time_points = np.zeros(n_points)
        speed_points = np.zeros(n_points)
        acceleration_points = np.zeros(n_points)
        rpm_points = np.zeros(n_points)
        gear_points = np.zeros(n_points)
        lateral_g_points = np.zeros(n_points)
        
        # Thermal state storage if included
        if include_thermal:
            engine_temp_points = np.zeros(n_points)
            coolant_temp_points = np.zeros(n_points)
            oil_temp_points = np.zeros(n_points)
            power_factor_points = np.zeros(n_points)
        
        # First point
        speed_points[0] = self.speed_profile[0]
        gear_points[0] = 1
        rpm_points[0] = self.vehicle.engine.idle_rpm
        
        # Time-based simulation along the lap
        current_time = 0.0
        current_dist_idx = 0
        
        while current_dist_idx < n_points - 1:
            # Current state
            current_speed = self.speed_profile[current_dist_idx]
            next_speed = self.speed_profile[current_dist_idx + 1]
            
            # Distance to next point
            current_distance = distances[current_dist_idx]
            next_distance = distances[current_dist_idx + 1]
            delta_distance = next_distance - current_distance
            
            # Average speed for this segment
            avg_speed = (current_speed + next_speed) / 2.0
            
            # Skip if speed is too low to avoid division by zero
            if avg_speed < 0.1:
                current_dist_idx += 1
                continue
            
            # Time to travel this segment
            segment_time = delta_distance / avg_speed
            
            # Update time
            next_time = current_time + segment_time
            
            # Store time and speed
            time_points[current_dist_idx + 1] = next_time
            speed_points[current_dist_idx + 1] = next_speed
            
            # Determine appropriate gear for this speed
            current_gear = gear_points[current_dist_idx]
            next_gear = self._estimate_optimal_gear(next_speed)
            
            # Check if shift is needed
            if next_gear != current_gear:
                # Record shift (in a real simulation, we might add shift time penalties)
                pass
            
            gear_points[current_dist_idx + 1] = next_gear
            
            # Calculate engine RPM
            engine_rpm = self.vehicle.drivetrain.calculate_engine_speed(next_speed, next_gear)
            rpm_points[current_dist_idx + 1] = engine_rpm
            
            # Calculate longitudinal acceleration
            if current_dist_idx > 0:
                # v² = u² + 2as, so a = (v² - u²) / 2s
                prev_speed = speed_points[current_dist_idx - 1]
                accel = (next_speed**2 - prev_speed**2) / (2 * delta_distance)
                acceleration_points[current_dist_idx] = accel
            
            # Calculate lateral acceleration based on curvature
            curve = curvature[current_dist_idx]
            if abs(curve) > 1e-6:
                lateral_accel = next_speed**2 * abs(curve)
                lateral_g = lateral_accel / 9.81
            else:
                lateral_g = 0.0
            
            lateral_g_points[current_dist_idx + 1] = lateral_g
            
            # Update thermal state if thermal simulation is enabled
            if include_thermal:
                # Use current track position as a progress indicator (0-1)
                progress = current_distance / track_length
                
                # More heat is generated in corners due to engine load
                local_curvature = abs(current_curvature)
                heat_factor = 1.0 + 2.0 * local_curvature  # Generate more heat in corners
                
                # Get cooling effectiveness based on vehicle configuration
                cooling_effectiveness = 0.5  # Default value
                
                # Apply cooling system type effect (make this impact significant)
                if hasattr(self.vehicle, 'cooling_system') and hasattr(self.vehicle.cooling_system, 'radiator'):
                    rad_type = self.vehicle.cooling_system.radiator.radiator_type.name
                    if rad_type == "DOUBLE_CORE_ALUMINUM":
                        cooling_effectiveness = 0.8  # 80% effectiveness
                    elif rad_type == "SINGLE_CORE_COPPER":
                        cooling_effectiveness = 0.7  # 70% effectiveness
                    elif rad_type == "SINGLE_CORE_ALUMINUM":
                        cooling_effectiveness = 0.5  # 50% effectiveness (worst)
                
                # Apply side pods effect
                if hasattr(self.vehicle, 'side_pod_system') and self.vehicle.side_pod_system is not None:
                    cooling_effectiveness *= 1.2  # 20% improvement
                
                # Apply rear radiator effect
                if hasattr(self.vehicle, 'rear_radiator') and self.vehicle.rear_radiator is not None:
                    cooling_effectiveness *= 1.15  # 15% improvement
                
                # Current temperature affects engine performance directly
                # Update engine temperature based on heat and cooling
                base_temp_rise = 0.2 * heat_factor  # Base temperature rise per step
                cooling_effect = 0.15 * cooling_effectiveness * current_speed / 20.0  # Cooling increases with speed
                
                # Net temperature change 
                temp_change = base_temp_rise - cooling_effect
                
                # Update engine temperature (prevent NaN by clamping values)
                if not hasattr(self.vehicle.engine, 'engine_temperature'):
                    self.vehicle.engine.engine_temperature = 60.0  # Initialize if not present
                
                self.vehicle.engine.engine_temperature = min(150.0, max(60.0, 
                                                                        self.vehicle.engine.engine_temperature + temp_change))
                
                # Update thermal factor - this directly affects engine torque
                if self.vehicle.engine.engine_temperature < 80.0:  # Cold
                    self.vehicle.engine.thermal_factor = 0.9
                elif self.vehicle.engine.engine_temperature < 100.0:  # Optimal
                    self.vehicle.engine.thermal_factor = 1.0
                elif self.vehicle.engine.engine_temperature < 110.0:  # Hot
                    self.vehicle.engine.thermal_factor = 0.9
                elif self.vehicle.engine.engine_temperature < 120.0:  # Very hot
                    self.vehicle.engine.thermal_factor = 0.8
                else:  # Overheating
                    self.vehicle.engine.thermal_factor = 0.65
                    thermal_limited = True
            
            # Move to next point
            current_time = next_time
            current_dist_idx += 1
        
        # Calculate lap time
        self.lap_time = time_points[-1]
        self.time_profile = time_points
        
        # Calculate sector times
        self.calculate_sector_times()
        
        # Prepare results dictionary
        results = {
            'lap_time': self.lap_time,
            'sector_times': self.sector_times,
            'time': time_points,
            'distance': distances,
            'speed': speed_points,
            'acceleration': acceleration_points,
            'engine_rpm': rpm_points,
            'gear': gear_points,
            'lateral_g': lateral_g_points,
            'include_thermal': include_thermal
        }
        
        # Add thermal data if included
        if include_thermal:
            results['engine_temp'] = engine_temp_points
            results['coolant_temp'] = coolant_temp_points
            results['oil_temp'] = oil_temp_points
            results['power_factor'] = power_factor_points
        
        logger.info(f"Lap simulation completed - Lap time: {self.lap_time:.3f}s")
        
        return results
    
    def calculate_sector_times(self) -> List[Dict]:
        """
        Calculate sector times based on track sections.
        
        Returns:
            List of sector time dictionaries
        """
        if not self.track_profile or self.time_profile is None:
            raise ValueError("Lap simulation must be run first")
        
        # Get track sections
        sections = self.track_data.get('sections', [])
        if not sections:
            logger.warning("No track sections defined, cannot calculate sector times")
            return []
        
        # Calculate time at each section boundary
        sector_times = []
        
        for i, section in enumerate(sections):
            start_idx = section['start_idx']
            end_idx = section['end_idx']
            
            # Calculate time spent in this section
            start_time = self.time_profile[start_idx]
            end_time = self.time_profile[end_idx]
            section_time = end_time - start_time
            
            sector_times.append({
                'sector': i + 1,
                'type': section['type'],
                'length': section['length'],
                'time': section_time
            })
        
        self.sector_times = sector_times
        
        return sector_times
    
    def analyze_lap_performance(self, results: Dict) -> Dict:
        """
        Analyze lap performance metrics.
        
        Args:
            results: Results from simulate_lap
            
        Returns:
            Dictionary with performance metrics
        """
        if not self.track_profile or self.lap_time is None:
            raise ValueError("Lap simulation must be run first")
        
        # Extract data
        time = results['time']
        speed = results['speed']
        lateral_g = results['lateral_g']
        
        # Calculate track length - use last distance value or calculate from track_data
        if 'distance' in results and len(results['distance']) > 0:
            track_length = results['distance'][-1]  # Use the last distance point
        elif 'length' in self.track_data:
            track_length = self.track_data['length']
        elif 'distance' in self.track_data and len(self.track_data['distance']) > 0:
            track_length = self.track_data['distance'][-1]
        else:
            # Fallback - calculate from points
            points = self.track_data['points']
            track_length = 0
            for i in range(1, len(points)):
                track_length += np.linalg.norm(points[i] - points[i-1])
        
        # Calculate average speed
        avg_speed = track_length / self.lap_time
        
        # Calculate maximum speed
        max_speed = np.max(speed)
        max_speed_idx = np.argmax(speed)
        
        # Calculate maximum lateral G
        max_lat_g = np.max(lateral_g)
        max_lat_g_idx = np.argmax(lateral_g)
        
        # Calculate time spent in different speed ranges
        speed_ranges = {
            '0-50 km/h': (0, 50/3.6),
            '50-100 km/h': (50/3.6, 100/3.6),
            '100+ km/h': (100/3.6, float('inf'))
        }
        
        time_in_range = {}
        for range_name, (min_speed, max_speed_range) in speed_ranges.items():
            mask = (speed >= min_speed) & (speed < max_speed_range)
            if np.any(mask[:-1]):
                time_in_range[range_name] = np.sum(np.diff(time)[mask[:-1]])
            else:
                time_in_range[range_name] = 0.0
        
        # Calculate time spent in corners vs straights
        if self.sector_times:
            time_in_corners = sum(s['time'] for s in self.sector_times 
                                if s['type'] in ['left_turn', 'right_turn'])
            time_in_straights = sum(s['time'] for s in self.sector_times 
                                if s['type'] == 'straight')
        else:
            # Approximate based on lateral G
            corner_mask = lateral_g >= 0.4
            time_in_corners = np.sum(np.diff(time)[corner_mask[:-1]])
            time_in_straights = self.lap_time - time_in_corners
        
        # Check for thermal limitations
        thermal_limited = False
        if 'engine_temp' in results:
            engine_temp = results['engine_temp']
            coolant_temp = results.get('coolant_temp', np.zeros_like(engine_temp))
            
            # Maximum allowed temperatures (from engine specs)
            max_engine_temp = 120.0  # Typical limit
            max_coolant_temp = 105.0  # Typical limit
            
            # Check if temperatures are limiting performance
            engine_temp_limited = np.any(engine_temp > max_engine_temp)
            coolant_temp_limited = np.any(coolant_temp > max_coolant_temp)
            
            thermal_limited = engine_temp_limited or coolant_temp_limited
        
        # Simplified estimate of potential improvement
        potential_improvement = 0.0
        
        if thermal_limited:
            # If thermally limited, estimate improvement with better cooling
            potential_improvement += 0.05 * self.lap_time
        
        # Store track length for future reference
        if 'length' not in self.track_data:
            self.track_data['length'] = track_length
        
        # Compile metrics
        metrics = {
            'lap_time': self.lap_time,
            'track_length': track_length,
            'avg_speed': avg_speed,
            'avg_speed_kph': avg_speed * 3.6,
            'max_speed': max_speed,
            'max_speed_kph': max_speed * 3.6,
            'max_lateral_g': max_lat_g,
            'time_in_speed_ranges': time_in_range,
            'time_in_corners': time_in_corners,
            'time_in_straights': time_in_straights,
            'thermal_limited': thermal_limited,
            'potential_improvement': potential_improvement
        }
        
        return metrics
    
    def visualize_lap(self, results: Dict, save_path: Optional[str] = None):
        """
        Visualize lap simulation results.
        
        Args:
            results: Results from simulate_lap
            save_path: Optional path to save the plot
        """
        if not self.track_profile or self.lap_time is None:
            raise ValueError("Lap simulation must be run first")
        
        # Extract data
        time = results['time']
        distance = results['distance']
        speed = results['speed']
        gear = results['gear']
        rpm = results['engine_rpm']
        lateral_g = results['lateral_g']
        
        # Convert speed to km/h for display
        speed_kph = speed * 3.6
        
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(15, 10))
        gs = plt.GridSpec(3, 2, height_ratios=[2, 1, 1])
        
        # Plot track layout with speed colors
        ax_track = plt.subplot(gs[0, 0])
        track_points = self.track_data['points']
        
        # Create colormap based on speed
        cmap = plt.cm.jet
        norm = plt.Normalize(np.min(speed_kph), np.max(speed_kph))
        colors = cmap(norm(speed_kph))
        
        # Plot track with color based on speed
        for i in range(len(track_points) - 1):
            ax_track.plot([track_points[i, 0], track_points[i+1, 0]], 
                        [track_points[i, 1], track_points[i+1, 1]], 
                        color=colors[i], linewidth=2)
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax_track)
        cbar.set_label('Speed (km/h)')
        
        # Add start/finish markers
        ax_track.plot(track_points[0, 0], track_points[0, 1], 'go', markersize=10, label='Start/Finish')
        
        # Set equal aspect ratio and grid
        ax_track.set_aspect('equal')
        ax_track.grid(True)
        ax_track.set_title(f'Track Layout - Lap Time: {self.lap_time:.3f}s')
        ax_track.set_xlabel('X (m)')
        ax_track.set_ylabel('Y (m)')
        ax_track.legend()
        
        # Plot speed vs distance
        ax_speed = plt.subplot(gs[0, 1])
        ax_speed.plot(distance, speed_kph, 'b-', linewidth=2)
        ax_speed.set_ylabel('Speed (km/h)')
        ax_speed.set_xlabel('Distance (m)')
        ax_speed.set_title('Speed Profile')
        ax_speed.grid(True)
        
        # Plot sector times
        ax_sectors = plt.subplot(gs[1, 0])
        if self.sector_times:
            sector_numbers = [s['sector'] for s in self.sector_times]
            sector_times = [s['time'] for s in self.sector_times]
            sector_types = [s['type'] for s in self.sector_times]
            
            # Color bars based on section type
            colors = ['green' if t == 'straight' else 'blue' if t == 'left_turn' else 'red' 
                     for t in sector_types]
            
            ax_sectors.bar(sector_numbers, sector_times, color=colors)
            ax_sectors.set_xlabel('Sector')
            ax_sectors.set_ylabel('Time (s)')
            ax_sectors.set_title('Sector Times')
            ax_sectors.grid(True, axis='y')
            
            # Add color legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='green', label='Straight'),
                Patch(facecolor='blue', label='Left Turn'),
                Patch(facecolor='red', label='Right Turn')
            ]
            ax_sectors.legend(handles=legend_elements, loc='upper right')
        
        # Plot lateral G
        ax_lat_g = plt.subplot(gs[1, 1])
        ax_lat_g.plot(distance, lateral_g, 'r-', linewidth=2)
        ax_lat_g.set_ylabel('Lateral G')
        ax_lat_g.set_xlabel('Distance (m)')
        ax_lat_g.set_title('Lateral Acceleration')
        ax_lat_g.grid(True)
        
        # Mark maximum lateral G
        max_lat_g_idx = np.argmax(lateral_g)
        ax_lat_g.plot(distance[max_lat_g_idx], lateral_g[max_lat_g_idx], 'ro')
        ax_lat_g.text(distance[max_lat_g_idx], lateral_g[max_lat_g_idx], 
                     f"{lateral_g[max_lat_g_idx]:.2f}g", color='red')
        
        # Plot gear and RPM
        ax_gear = plt.subplot(gs[2, 0])
        ax_gear.step(distance, gear, 'g-', linewidth=2)
        ax_gear.set_ylabel('Gear')
        ax_gear.set_xlabel('Distance (m)')
        ax_gear.set_title('Gear Selection')
        ax_gear.grid(True)
        ax_gear.set_yticks(range(1, self.vehicle.drivetrain.transmission.num_gears + 1))
        
        # Plot RPM
        ax_rpm = plt.subplot(gs[2, 1])
        ax_rpm.plot(distance, rpm, 'b-', linewidth=2)
        ax_rpm.set_ylabel('Engine RPM')
        ax_rpm.set_xlabel('Distance (m)')
        ax_rpm.set_title('Engine RPM')
        ax_rpm.grid(True)
        
        # Add redline
        ax_rpm.axhline(y=self.vehicle.engine.redline, color='r', linestyle='--', label='Redline')
        ax_rpm.legend()
        
        # Add thermal plots if available
        if self.include_thermal and 'engine_temp' in results:
            # Create a new figure for thermal data
            fig_thermal = plt.figure(figsize=(12, 8))
            
            # Plot engine and coolant temperatures
            ax_temp = plt.subplot(211)
            ax_temp.plot(distance, results['engine_temp'], 'r-', linewidth=2, label='Engine Temp')
            ax_temp.plot(distance, results['coolant_temp'], 'b-', linewidth=2, label='Coolant Temp')
            
            if 'oil_temp' in results:
                ax_temp.plot(distance, results['oil_temp'], 'g-', linewidth=2, label='Oil Temp')
            
            ax_temp.set_ylabel('Temperature (°C)')
            ax_temp.set_title('Thermal Performance')
            ax_temp.grid(True)
            ax_temp.legend()
            
            # Add warning thresholds
            ax_temp.axhline(y=105.0, color='r', linestyle='--', alpha=0.5, label='Max Coolant')
            ax_temp.axhline(y=120.0, color='darkred', linestyle='--', alpha=0.5, label='Max Engine')
            
            # Plot relative power factor due to temperature
            ax_power = plt.subplot(212, sharex=ax_temp)
            
            # If power factor is calculated, use it, otherwise calculate it
            if 'power_factor' in results:
                power_factor = results['power_factor']
            else:
                # Calculate power factor based on temperature (simplified model)
                engine_temp = results['engine_temp']
                optimal_temp = 90.0  # Optimal engine temperature
                power_factor = 0.5 + 0.5 * np.exp(-0.001 * (engine_temp - optimal_temp) ** 2)
                power_factor = np.clip(power_factor, 0.7, 1.0)  # Limit to reasonable range
            
            ax_power.plot(distance, power_factor, 'k-', linewidth=2)
            ax_power.set_ylabel('Power Factor')
            ax_power.set_xlabel('Distance (m)')
            ax_power.set_ylim(0.7, 1.05)
            ax_power.grid(True)
            
            # Add reference line at 100% power
            ax_power.axhline(y=1.0, color='g', linestyle='--', alpha=0.5)
            
            # Show thermal figure
            plt.tight_layout()
            
            # Save thermal plot if path provided
            if save_path:
                thermal_save_path = save_path.replace('.png', '_thermal.png')
                fig_thermal.savefig(thermal_save_path, dpi=300, bbox_inches='tight')
        
        # Adjust main figure layout
        plt.tight_layout()
        
        # Save main plot if path provided
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            
        plt.show()
    
    def compare_vehicle_configs(self, vehicle_configs: List[Vehicle], 
                              labels: List[str],
                              include_thermal: bool = True,
                              save_path: Optional[str] = None) -> Dict:
        """
        Compare lap times for different vehicle configurations.
        
        Args:
            vehicle_configs: List of vehicle models
            labels: List of labels for each configuration
            include_thermal: Whether to include thermal effects
            save_path: Optional path to save the plot
            
        Returns:
            Dictionary with comparison results
        """
        if not self.track_profile:
            raise ValueError("No track loaded")
        
        if len(vehicle_configs) != len(labels):
            raise ValueError("Number of vehicle configs must match number of labels")
        
        # Save current vehicle
        original_vehicle = self.vehicle
        
        # Results storage
        comparison_results = []
        
        # Simulate each configuration
        for vehicle, label in zip(vehicle_configs, labels):
            # Set vehicle
            self.vehicle = vehicle
            
            # Reset simulation results
            self.speed_profile = None
            self.lap_time = None
            self.time_profile = None
            
            # Recalculate speed profile and simulate lap
            self.calculate_speed_profile()
            lap_results = self.simulate_lap(include_thermal=include_thermal)
            
            # Analyze performance metrics
            lap_metrics = self.analyze_lap_performance(lap_results)
            
            # Store results with label
            comparison_result = {
                'label': label,
                'lap_time': self.lap_time,
                'results': lap_results,
                'metrics': lap_metrics
            }
            
            comparison_results.append(comparison_result)
            
            logger.info(f"Configuration '{label}' lap time: {self.lap_time:.3f}s")
        
        # Restore original vehicle
        self.vehicle = original_vehicle
        
        # Create comparison plots
        if save_path:
            self._plot_lap_time_comparison(comparison_results, save_path)
        
        # Compile comparison data
        comparison_data = {
            'configs': labels,
            'lap_times': [r['lap_time'] for r in comparison_results],
            'avg_speeds': [r['metrics']['avg_speed'] for r in comparison_results],
            'max_speeds': [r['metrics']['max_speed'] for r in comparison_results],
            'thermal_limited': [r['metrics']['thermal_limited'] for r in comparison_results],
            'detailed_results': comparison_results
        }
        
        return comparison_data
    
    def _plot_lap_time_comparison(self, comparison_results: List[Dict], save_path: str):
        """
        Plot comparison of lap times for different vehicle configurations.
        
        Args:
            comparison_results: List of comparison result dictionaries
            save_path: Path to save the plot
        """
        # Extract data
        labels = [r['label'] for r in comparison_results]
        lap_times = [r['lap_time'] for r in comparison_results]
        
        # Sort by lap time (fastest first)
        sort_idx = np.argsort(lap_times)
        sorted_labels = [labels[i] for i in sort_idx]
        sorted_times = [lap_times[i] for i in sort_idx]
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Plot lap times
        ax1 = plt.subplot(211)
        bars = ax1.bar(sorted_labels, sorted_times)
        
        # Add lap time labels
        for bar, time in zip(bars, sorted_times):
            ax1.text(bar.get_x() + bar.get_width()/2., time + 0.05,
                   f'{time:.3f}s', ha='center', va='bottom')
        
        ax1.set_ylabel('Lap Time (s)')
        ax1.set_title('Lap Time Comparison')
        ax1.grid(True, axis='y')
        
        # Plot speed profiles
        ax2 = plt.subplot(212)
        
        # Colors for different configurations
        colors = plt.cm.tab10.colors
        
        # Get maximum distance for plot
        max_distance = 0
        for result in comparison_results:
            distance = result['results']['distance']
            max_distance = max(max_distance, distance[-1])
        
        # Plot speed profile for each configuration
        for i, result in enumerate(comparison_results):
            distance = result['results']['distance']
            speed = result['results']['speed'] * 3.6  # Convert to km/h
            label = result['label']
            color = colors[i % len(colors)]
            
            ax2.plot(distance, speed, color=color, linewidth=2, label=label)
        
        ax2.set_xlabel('Distance (m)')
        ax2.set_ylabel('Speed (km/h)')
        ax2.set_title('Speed Profiles')
        ax2.grid(True)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create a separate figure for sector time comparison
        plt.figure(figsize=(14, 8))
        
        # Check if we have sector times
        if all('sector_times' in r['results'] for r in comparison_results):
            # Get all unique sector numbers
            all_sectors = set()
            for result in comparison_results:
                sectors = result['results'].get('sector_times', [])
                for sector in sectors:
                    all_sectors.add(sector['sector'])
            
            # Sort sectors
            all_sectors = sorted(all_sectors)
            
            # Create sector time arrays for each configuration
            sector_data = []
            
            for result in comparison_results:
                sectors = result['results'].get('sector_times', [])
                sector_times = np.zeros(len(all_sectors))
                
                for sector in sectors:
                    idx = all_sectors.index(sector['sector'])
                    sector_times[idx] = sector['time']
                
                sector_data.append(sector_times)
            
            # Plot sector times as grouped bars
            width = 0.8 / len(comparison_results)
            for i, (times, label) in enumerate(zip(sector_data, labels)):
                x = np.arange(len(all_sectors)) + i * width - width * len(comparison_results) / 2 + width / 2
                plt.bar(x, times, width, label=label, color=colors[i % len(colors)])
            
            plt.xlabel('Sector')
            plt.ylabel('Time (s)')
            plt.title('Sector Time Comparison')
            plt.grid(True, axis='y')
            plt.xticks(np.arange(len(all_sectors)), all_sectors)
            plt.legend()
            
            # Save sector comparison
            sector_save_path = save_path.replace('.png', '_sectors.png')
            plt.savefig(sector_save_path, dpi=300, bbox_inches='tight')
            plt.close()
    
    def optimize_vehicle_setup(self, param_ranges: Dict[str, Tuple[float, float]], 
                             max_iterations: int = 20,
                             include_thermal: bool = True) -> Dict:
        """
        Optimize vehicle setup parameters for fastest lap time.
        
        Args:
            param_ranges: Dictionary of parameter names to (min, max) tuples
            max_iterations: Maximum number of optimization iterations
            include_thermal: Whether to include thermal effects
            
        Returns:
            Dictionary with optimization results
        """
        if not self.track_profile:
            raise ValueError("No track loaded")
        
        # Save original vehicle parameters
        original_params = {}
        for param in param_ranges:
            if hasattr(self.vehicle, param):
                original_params[param] = getattr(self.vehicle, param)
            elif '.' in param:
                # Handle nested attributes like "drivetrain.final_drive.ratio"
                obj = self.vehicle
                path = param.split('.')
                for attr in path[:-1]:
                    if hasattr(obj, attr):
                        obj = getattr(obj, attr)
                    else:
                        logger.warning(f"Object {obj} has no attribute {attr}")
                        break
                else:
                    if hasattr(obj, path[-1]):
                        original_params[param] = getattr(obj, path[-1])
        
        # Define objective function for optimization
        def objective(param_values):
            # Set parameter values
            for param, value in zip(param_ranges.keys(), param_values):
                if hasattr(self.vehicle, param):
                    setattr(self.vehicle, param, value)
                elif '.' in param:
                    # Handle nested attributes
                    obj = self.vehicle
                    path = param.split('.')
                    for attr in path[:-1]:
                        if hasattr(obj, attr):
                            obj = getattr(obj, attr)
                        else:
                            break
                    else:
                        if hasattr(obj, path[-1]):
                            setattr(obj, path[-1], value)
            
            # Reset simulation state
            self.speed_profile = None
            
            # Run lap simulation and return lap time
            try:
                self.calculate_speed_profile()
                lap_results = self.simulate_lap(include_thermal=include_thermal)
                return self.lap_time
            except Exception as e:
                logger.error(f"Error in lap simulation: {str(e)}")
                return float('inf')  # Return infinity for failed simulations
        
        # Initial parameter values (middle of ranges)
        initial_values = [(min_val + max_val) / 2 for min_val, max_val in param_ranges.values()]
        
        # Parameter bounds
        bounds = list(param_ranges.values())
        
        # Simple grid search optimization
        current_best = objective(initial_values)
        best_params = initial_values.copy()
        
        # Tracking progress
        all_evaluations = [{'params': dict(zip(param_ranges.keys(), initial_values)), 'lap_time': current_best}]
        
        logger.info(f"Starting optimization with {len(param_ranges)} parameters. Initial lap time: {current_best:.3f}s")
        
        # Run optimization iterations
        for iteration in range(max_iterations):
            # Try adjusting each parameter individually
            for i, param_name in enumerate(param_ranges.keys()):
                for direction in [-1, 1]:  # Try decreasing and increasing
                    # Copy current best parameters
                    test_params = best_params.copy()
                    
                    # Adjust the parameter by 10% of its range
                    param_range = bounds[i][1] - bounds[i][0]
                    adjustment = direction * 0.1 * param_range
                    test_params[i] += adjustment
                    
                    # Keep within bounds
                    test_params[i] = max(bounds[i][0], min(bounds[i][1], test_params[i]))
                    
                    # Evaluate
                    lap_time = objective(test_params)
                    
                    # Record evaluation
                    all_evaluations.append({
                        'params': dict(zip(param_ranges.keys(), test_params)),
                        'lap_time': lap_time
                    })
                    
                    # Update if better
                    if lap_time < current_best:
                        current_best = lap_time
                        best_params = test_params.copy()
                        logger.info(f"Iteration {iteration+1}, improved {param_name} to {test_params[i]:.3f}, lap time: {lap_time:.3f}s")
            
            # Early termination if no improvement
            if best_params == initial_values:
                logger.info(f"No improvement after adjusting parameters, terminating early")
                break
            
            # Update initial values for next iteration
            initial_values = best_params.copy()
        
        # Restore original parameters
        for param, value in original_params.items():
            if hasattr(self.vehicle, param):
                setattr(self.vehicle, param, value)
            elif '.' in param:
                # Handle nested attributes
                obj = self.vehicle
                path = param.split('.')
                for attr in path[:-1]:
                    if hasattr(obj, attr):
                        obj = getattr(obj, attr)
                    else:
                        break
                else:
                    if hasattr(obj, path[-1]):
                        setattr(obj, path[-1], value)
        
        # Create results dictionary
        results = {
            'best_params': dict(zip(param_ranges.keys(), best_params)),
            'best_lap_time': current_best,
            'original_params': original_params,
            'all_evaluations': all_evaluations
        }
        
        return results
    
    def compare_vehicle_configs(self, vehicle_configs: List[Vehicle], 
                          labels: List[str],
                          include_thermal: bool = True,
                          save_path: Optional[str] = None) -> Dict:
        """
        Compare lap times for different vehicle configurations.
        
        Args:
            vehicle_configs: List of vehicle models
            labels: List of labels for each configuration
            include_thermal: Whether to include thermal effects
            save_path: Optional path to save the plot
            
        Returns:
            Dictionary with comparison results
        """
        if not self.track_profile:
            raise ValueError("No track loaded")
        
        if len(vehicle_configs) != len(labels):
            raise ValueError("Number of vehicle configs must match number of labels")
        
        # Save current vehicle
        original_vehicle = self.vehicle
        
        # Results storage
        comparison_results = []
        
        # Simulate each configuration
        for vehicle, label in zip(vehicle_configs, labels):
            # Set vehicle
            self.vehicle = vehicle
            
            # Reset simulation results
            self.speed_profile = None
            self.lap_time = None
            self.time_profile = None
            
            # Recalculate speed profile and simulate lap
            self.calculate_speed_profile()
            lap_results = self.simulate_lap(include_thermal=include_thermal)
            
            # Analyze performance metrics
            lap_metrics = self.analyze_lap_performance(lap_results)
            
            # Store results with label
            comparison_result = {
                'label': label,
                'lap_time': self.lap_time,
                'results': lap_results,
                'metrics': lap_metrics
            }
            
            comparison_results.append(comparison_result)
            
            logger.info(f"Configuration '{label}' lap time: {self.lap_time:.3f}s")
        
        # Restore original vehicle
        self.vehicle = original_vehicle
        
        # Create comparison plots
        if save_path:
            self._plot_lap_time_comparison(comparison_results, save_path)
        
        # Compile comparison data
        comparison_data = {
            'configs': labels,
            'lap_times': [r['lap_time'] for r in comparison_results],
            'avg_speeds': [r['metrics']['avg_speed'] for r in comparison_results],
            'max_speeds': [r['metrics']['max_speed'] for r in comparison_results],
            'thermal_limited': [r['metrics']['thermal_limited'] for r in comparison_results],
            'detailed_results': comparison_results
        }
        
        return comparison_data

    def _plot_lap_time_comparison(self, comparison_results: List[Dict], save_path: str):
        """
        Plot comparison of lap times for different vehicle configurations.
        
        Args:
            comparison_results: List of comparison result dictionaries
            save_path: Path to save the plot
        """
        # Extract data
        labels = [r['label'] for r in comparison_results]
        lap_times = [r['lap_time'] for r in comparison_results]
        
        # Sort by lap time (fastest first)
        sort_idx = np.argsort(lap_times)
        sorted_labels = [labels[i] for i in sort_idx]
        sorted_times = [lap_times[i] for i in sort_idx]
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Plot lap times
        ax1 = plt.subplot(211)
        bars = ax1.bar(sorted_labels, sorted_times)
        
        # Add lap time labels
        for bar, time in zip(bars, sorted_times):
            ax1.text(bar.get_x() + bar.get_width()/2., time + 0.05,
                   f'{time:.3f}s', ha='center', va='bottom')
        
        ax1.set_ylabel('Lap Time (s)')
        ax1.set_title('Lap Time Comparison')
        ax1.grid(True, axis='y')
        
        # Plot speed profiles
        ax2 = plt.subplot(212)
        
        # Colors for different configurations
        colors = plt.cm.tab10.colors
        
        # Get maximum distance for plot
        max_distance = 0
        for result in comparison_results:
            distance = result['results']['distance']
            max_distance = max(max_distance, distance[-1])
        
        # Plot speed profile for each configuration
        for i, result in enumerate(comparison_results):
            distance = result['results']['distance']
            speed = result['results']['speed'] * 3.6  # Convert to km/h
            label = result['label']
            color = colors[i % len(colors)]
            
            ax2.plot(distance, speed, color=color, linewidth=2, label=label)
        
        ax2.set_xlabel('Distance (m)')
        ax2.set_ylabel('Speed (km/h)')
        ax2.set_title('Speed Profiles')
        ax2.grid(True)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create a separate figure for sector time comparison
        plt.figure(figsize=(14, 8))
        
        # Check if we have sector times
        if all('sector_times' in r['results'] for r in comparison_results):
            # Get all unique sector numbers
            all_sectors = set()
            for result in comparison_results:
                sectors = result['results'].get('sector_times', [])
                for sector in sectors:
                    all_sectors.add(sector['sector'])
            
            # Sort sectors
            all_sectors = sorted(all_sectors)
            
            # Create sector time arrays for each configuration
            sector_data = []
            
            for result in comparison_results:
                sectors = result['results'].get('sector_times', [])
                sector_times = np.zeros(len(all_sectors))
                
                for sector in sectors:
                    idx = all_sectors.index(sector['sector'])
                    sector_times[idx] = sector['time']
                
                sector_data.append(sector_times)
            
            # Plot sector times as grouped bars
            width = 0.8 / len(comparison_results)
            for i, (times, label) in enumerate(zip(sector_data, labels)):
                x = np.arange(len(all_sectors)) + i * width - width * len(comparison_results) / 2 + width / 2
                plt.bar(x, times, width, label=label, color=colors[i % len(colors)])
            
            plt.xlabel('Sector')
            plt.ylabel('Time (s)')
            plt.title('Sector Time Comparison')
            plt.grid(True, axis='y')
            plt.xticks(np.arange(len(all_sectors)), all_sectors)
            plt.legend()
            
            # Save sector comparison
            sector_save_path = save_path.replace('.png', '_sectors.png')
            plt.savefig(sector_save_path, dpi=300, bbox_inches='tight')
            plt.close()
    
    def optimize_vehicle_setup(self, param_ranges: Dict, 
                             max_iterations: int = 20,
                             include_thermal: bool = True) -> Dict:
        """
        Optimize vehicle setup parameters for fastest lap time.
        
        Args:
            param_ranges: Dictionary of parameter names to (min, max) tuples
            max_iterations: Maximum number of optimization iterations
            include_thermal: Whether to include thermal effects
            
        Returns:
            Dictionary with optimization results
        """
        if not self.track_profile:
            raise ValueError("No track loaded")
        
        # Save original vehicle parameters
        original_params = {}
        for param in param_ranges:
            if hasattr(self.vehicle, param):
                original_params[param] = getattr(self.vehicle, param)
            elif '.' in param:
                # Handle nested attributes like "drivetrain.final_drive.ratio"
                obj = self.vehicle
                path = param.split('.')
                for attr in path[:-1]:
                    if hasattr(obj, attr):
                        obj = getattr(obj, attr)
                    else:
                        logger.warning(f"Object {obj} has no attribute {attr}")
                        break
                else:
                    if hasattr(obj, path[-1]):
                        original_params[param] = getattr(obj, path[-1])
        
        # Define objective function for optimization
        def objective(param_values):
            # Set parameter values
            for param, value in zip(param_ranges.keys(), param_values):
                if hasattr(self.vehicle, param):
                    setattr(self.vehicle, param, value)
                elif '.' in param:
                    # Handle nested attributes
                    obj = self.vehicle
                    path = param.split('.')
                    for attr in path[:-1]:
                        if hasattr(obj, attr):
                            obj = getattr(obj, attr)
                        else:
                            break
                    else:
                        if hasattr(obj, path[-1]):
                            setattr(obj, path[-1], value)
            
            # Reset simulation state
            self.speed_profile = None
            
            # Run lap simulation and return lap time
            try:
                self.calculate_speed_profile()
                lap_results = self.simulate_lap(include_thermal=include_thermal)
                return self.lap_time
            except Exception as e:
                logger.error(f"Error in lap simulation: {str(e)}")
                return float('inf')  # Return infinity for failed simulations
        
        # Initial parameter values (middle of ranges)
        initial_values = [(min_val + max_val) / 2 for min_val, max_val in param_ranges.values()]
        
        # Parameter bounds
        bounds = list(param_ranges.values())
        
        # Simple grid search optimization
        current_best = objective(initial_values)
        best_params = initial_values.copy()
        
        # Tracking progress
        all_evaluations = [{'params': dict(zip(param_ranges.keys(), initial_values)), 'lap_time': current_best}]
        
        logger.info(f"Starting optimization with {len(param_ranges)} parameters. Initial lap time: {current_best:.3f}s")
        
        # Run optimization iterations
        for iteration in range(max_iterations):
            # Try adjusting each parameter individually
            for i, param_name in enumerate(param_ranges.keys()):
                for direction in [-1, 1]:  # Try decreasing and increasing
                    # Copy current best parameters
                    test_params = best_params.copy()
                    
                    # Adjust the parameter by 10% of its range
                    param_range = bounds[i][1] - bounds[i][0]
                    adjustment = direction * 0.1 * param_range
                    test_params[i] += adjustment
                    
                    # Keep within bounds
                    test_params[i] = max(bounds[i][0], min(bounds[i][1], test_params[i]))
                    
                    # Evaluate
                    lap_time = objective(test_params)
                    
                    # Record evaluation
                    all_evaluations.append({
                        'params': dict(zip(param_ranges.keys(), test_params)),
                        'lap_time': lap_time
                    })
                    
                    # Update if better
                    if lap_time < current_best:
                        current_best = lap_time
                        best_params = test_params.copy()
                        logger.info(f"Iteration {iteration+1}, improved {param_name} to {test_params[i]:.3f}, lap time: {lap_time:.3f}s")
            
            # Early termination if no improvement
            if best_params == initial_values:
                logger.info(f"No improvement after adjusting parameters, terminating early")
                break
            
            # Update initial values for next iteration
            initial_values = best_params.copy()
        
        # Restore original parameters
        for param, value in original_params.items():
            if hasattr(self.vehicle, param):
                setattr(self.vehicle, param, value)
            elif '.' in param:
                # Handle nested attributes
                obj = self.vehicle
                path = param.split('.')
                for attr in path[:-1]:
                    if hasattr(obj, attr):
                        obj = getattr(obj, attr)
                    else:
                        break
                else:
                    if hasattr(obj, path[-1]):
                        setattr(obj, path[-1], value)
        
        # Create results dictionary
        results = {
            'best_params': dict(zip(param_ranges.keys(), best_params)),
            'best_lap_time': current_best,
            'original_params': original_params,
            'all_evaluations': all_evaluations
        }
        
        return results


class CorneringPerformance:
    """
    Calculator for vehicle cornering performance.
    
    This class provides tools for calculating maximum cornering speeds, lateral
    acceleration, and other cornering performance metrics for a Formula Student car.
    """
    
    def __init__(self, vehicle: Vehicle):
        """
        Initialize the cornering performance calculator.
        
        Args:
            vehicle: Vehicle model
        """
        self.vehicle = vehicle
        
        # Default parameters if not provided by vehicle
        self.tire_lateral_stiffness = getattr(vehicle, 'tire_lateral_stiffness', 15.0)  # N/deg
        self.cg_height = getattr(vehicle, 'cg_height', 0.3)  # m
        self.track_width = getattr(vehicle, 'track_width', 1.2)  # m
        self.wheelbase = getattr(vehicle, 'wheelbase', 1.6)  # m
        self.weight_distribution = getattr(vehicle, 'weight_distribution', 0.45)  # fraction on front axle
        
        # Calculate derived parameters
        self.mass = vehicle.mass
        self.weight = self.mass * 9.81  # N
        
        # Aero parameters for downforce
        self.cl = getattr(vehicle, 'lift_coefficient', -1.5)  # Negative for downforce
        self.frontal_area = vehicle.frontal_area
        
        logger.info("Cornering performance calculator initialized")
    
    def calculate_max_lateral_acceleration(self, speed: Optional[float] = None) -> float:
        """
        Calculate maximum lateral acceleration.
        
        Args:
            speed: Vehicle speed in m/s, for aero-dependent calculations
            
        Returns:
            Maximum lateral acceleration in m/s²
        """
        # Base coefficient of friction for tires
        mu = 1.3  # Typical for racing slicks
        
        # If speed is provided, add aero effects
        if speed is not None and speed > 0:
            # Calculate downforce
            air_density = 1.225  # kg/m³
            downforce = -0.5 * air_density * self.cl * self.frontal_area * speed**2
            
            # Calculate effective weight with downforce
            effective_weight = self.weight + downforce
            
            # Calculate effective friction coefficient
            effective_mu = mu * effective_weight / self.weight
            
            # Cap effective mu to reasonable values
            effective_mu = min(2.0, effective_mu)
        else:
            effective_mu = mu
        
        # Calculate lateral acceleration (simplified model)
        max_lat_accel = effective_mu * 9.81  # m/s²
        
        return max_lat_accel
    
    def calculate_max_cornering_speed(self, radius: float) -> float:
        """
        Calculate maximum cornering speed for a given radius.
        
        Args:
            radius: Corner radius in meters
            
        Returns:
            Maximum cornering speed in m/s
        """
        # Initial estimate based on steady-state cornering
        # v² = a_lat * r
        initial_speed = np.sqrt(9.81 * 1.3 * radius)  # Initial estimate with constant mu
        
        # Refine with aero effects
        max_speed = initial_speed
        
        # Iterative estimation to account for speed-dependent effects
        for _ in range(3):  # 3 iterations should be enough for convergence
            max_lat_accel = self.calculate_max_lateral_acceleration(max_speed)
            max_speed = np.sqrt(max_lat_accel * radius)
        
        return max_speed
    
    def calculate_weight_transfer(self, lateral_accel: float) -> Tuple[float, float]:
        """
        Calculate lateral weight transfer during cornering.
        
        Args:
            lateral_accel: Lateral acceleration in m/s²
            
        Returns:
            Tuple of (weight on inside wheels, weight on outside wheels)
        """
        # Total weight
        total_weight = self.weight
        
        # Static weight on each side
        static_weight_per_side = total_weight / 2
        
        # Calculate lateral weight transfer
        # Transfer = (mass * lateral_accel * cg_height) / track_width
        weight_transfer = (self.mass * lateral_accel * self.cg_height) / self.track_width
        
        # Calculate dynamic weights
        inside_weight = static_weight_per_side - weight_transfer
        outside_weight = static_weight_per_side + weight_transfer
        
        return inside_weight, outside_weight
    
    def calculate_roll_angle(self, lateral_accel: float) -> float:
        """
        Calculate roll angle during cornering.
        
        Args:
            lateral_accel: Lateral acceleration in m/s²
            
        Returns:
            Roll angle in degrees
        """
        # Simplified roll angle calculation
        # Assumes linear relationship between lateral acceleration and roll angle
        # Would need vehicle roll stiffness for more accurate calculation
        
        # Typical roll gradient for Formula Student car (degrees per g)
        roll_gradient = 1.0  # deg/g
        
        # Calculate roll angle
        roll_angle = roll_gradient * lateral_accel / 9.81
        
        return roll_angle
    
    def get_cornering_metrics(self, speed: float, radius: float) -> Dict:
        """
        Calculate comprehensive cornering metrics for a given speed and radius.
        
        Args:
            speed: Vehicle speed in m/s
            radius: Corner radius in meters
            
        Returns:
            Dictionary with cornering metrics
        """
        # Calculate lateral acceleration
        lateral_accel = speed**2 / radius
        lateral_g = lateral_accel / 9.81
        
        # Calculate weight transfer
        inside_weight, outside_weight = self.calculate_weight_transfer(lateral_accel)
        
        # Check for wheel lift (negative weight on inside wheels)
        wheel_lift = inside_weight <= 0
        
        # Calculate roll angle
        roll_angle = self.calculate_roll_angle(lateral_accel)
        
        # Calculate slip angle (simplified estimate)
        slip_angle = lateral_accel / (self.tire_lateral_stiffness * 4)  # deg
        
        # Calculate maximum speed for this radius
        max_speed = self.calculate_max_cornering_speed(radius)
        
        # Calculate speed fraction (how close to the limit)
        speed_fraction = speed / max_speed if max_speed > 0 else 1.0
        
        # Check if speed is above maximum (unstable)
        unstable = speed > max_speed
        
        # Compile metrics
        metrics = {
            'lateral_accel': lateral_accel,
            'lateral_g': lateral_g,
            'inside_weight': inside_weight,
            'outside_weight': outside_weight,
            'wheel_lift': wheel_lift,
            'roll_angle': roll_angle,
            'slip_angle': slip_angle,
            'max_speed': max_speed,
            'speed_fraction': speed_fraction,
            'unstable': unstable
        }
        
        return metrics


def create_lap_time_simulator(vehicle: Vehicle, track_file: str) -> LapTimeSimulator:
    """
    Create and configure a lap time simulator for a Formula Student vehicle.
    
    Args:
        vehicle: Vehicle model to use for simulation
        track_file: Path to track file
        
    Returns:
        Configured LapTimeSimulator
    """
    simulator = LapTimeSimulator(vehicle)
    
    # Load track
    simulator.load_track(track_file)
    
    # Configure with reasonable parameters
    simulator.time_step = 0.01
    simulator.max_time = 120.0
    
    return simulator


def run_fs_lap_simulation(vehicle: Vehicle, track_file: str, 
                         include_thermal: bool = True,
                         save_dir: Optional[str] = None) -> Dict:
    """
    Run a complete Formula Student lap time simulation.
    
    Args:
        vehicle: Vehicle model to use for simulation
        track_file: Path to track file
        include_thermal: Whether to include thermal effects
        save_dir: Optional directory to save results
        
    Returns:
        Dictionary with simulation results and metrics
    """
    # Create simulator
    simulator = create_lap_time_simulator(vehicle, track_file)
    
    # Calculate racing line
    simulator.calculate_racing_line()
    
    # Calculate speed profile
    simulator.calculate_speed_profile()
    
    # Run lap simulation
    lap_results = simulator.simulate_lap(include_thermal=include_thermal)
    
    # Analyze performance
    lap_metrics = simulator.analyze_lap_performance(lap_results)
    
    # Create output directory if specified
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        
        # Save lap visualization
        simulator.visualize_lap(lap_results, save_path=os.path.join(save_dir, "lap_visualization.png"))
        
        # Save metrics to CSV
        metrics_df = pd.DataFrame({
            'Metric': list(lap_metrics.keys()),
            'Value': list(lap_metrics.values())
        })
        metrics_df.to_csv(os.path.join(save_dir, "lap_metrics.csv"), index=False)
        
        # Save lap data
        lap_data = {
            'distance': lap_results['distance'],
            'time': lap_results['time'],
            'speed': lap_results['speed'],
            'acceleration': lap_results['acceleration'],
            'lateral_g': lap_results['lateral_g'],
            'gear': lap_results['gear'],
            'engine_rpm': lap_results['engine_rpm']
        }
        
        # Add thermal data if included
        if include_thermal and 'engine_temp' in lap_results:
            lap_data['engine_temp'] = lap_results['engine_temp']
            lap_data['coolant_temp'] = lap_results['coolant_temp']
            if 'oil_temp' in lap_results:
                lap_data['oil_temp'] = lap_results['oil_temp']
            if 'power_factor' in lap_results:
                lap_data['power_factor'] = lap_results['power_factor']
        
        # Convert to DataFrame and save
        lap_df = pd.DataFrame(lap_data)
        lap_df.to_csv(os.path.join(save_dir, "lap_data.csv"), index=False)
    
    # Print key results
    lap_time = lap_metrics['lap_time']
    avg_speed = lap_metrics['avg_speed_kph']
    max_speed = lap_metrics['max_speed_kph']
    
    logger.info(f"Lap simulation complete:")
    logger.info(f"  Lap time: {lap_time:.2f}s")
    logger.info(f"  Avg speed: {avg_speed:.1f} km/h")
    logger.info(f"  Max speed: {max_speed:.1f} km/h")
    
    if lap_metrics['thermal_limited']:
        logger.info(f"  Note: Vehicle was thermally limited during the lap")
    
    # Return results
    return {
        'lap_time': lap_time,
        'metrics': lap_metrics,
        'results': lap_results
    }


def create_example_track(output_file: str, difficulty: str = 'medium') -> str:
    """
    Create an example track for lap time simulation.
    
    Args:
        output_file: Path to save the track file
        difficulty: Track difficulty ('easy', 'medium', 'hard')
        
    Returns:
        Path to the created track file
    """
    try:
        # Try to use the track generator module if available
        from ..track_generator.generator import FSTrackGenerator
        from ..track_generator.enums import SimType
        
        # Track parameters based on difficulty
        if difficulty == 'easy':
            min_length = 600
            max_length = 800
            curvature_threshold = 0.2
        elif difficulty == 'medium':
            min_length = 1000
            max_length = 1200
            curvature_threshold = 0.25
        else:  # hard
            min_length = 1300
            max_length = 1500
            curvature_threshold = 0.33
        
        # Create track generator
        generator = FSTrackGenerator(
        base_dir=os.path.dirname(output_file),
        track_width=3.0,
        min_length=min_length,
        max_length=max_length,
        curvature_threshold=curvature_threshold
        )
        
        # Generate track
        generator.generate_track()
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Export track
        generator.export_track(output_file, SimType.FSSIM)
        
        logger.info(f"Created {difficulty} track using generator at {output_file}")
        
        return output_file
        
    except (ImportError, ModuleNotFoundError):
        # Fallback to creating a simple track manually
        logger.info("Track generator not available, creating simplified track manually")
        
        # Track parameters based on difficulty
        if difficulty == 'easy':
            length = 800
            corners = 6
            min_radius = 12
        elif difficulty == 'medium':
            length = 1200
            corners = 10
            min_radius = 9
        else:  # hard
            length = 1500
            corners = 15
            min_radius = 7
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Create a simple YAML track definition (very simplified)
        track_data = {
            'metadata': {
                'name': f"Example {difficulty} track",
                'length': length,
                'corners': corners,
                'min_radius': min_radius,
                'track_width': 3.0,
                'created': "Example track for lap time simulation"
            },
            'track': []
        }
        
        # Create a series of track points (very simplified oval for example)
        num_points = 100
        
        # Create oval shape
        for i in range(num_points):
            angle = 2 * np.pi * i / num_points
            
            # Oval shape
            a = length / 4  # semi-major axis
            b = length / 8  # semi-minor axis
            
            # Calculate position
            x = a * np.cos(angle)
            y = b * np.sin(angle)
            
            # Add some randomness for more interesting track
            if difficulty != 'easy':
                x += np.sin(angle * corners) * min_radius * 0.8
                y += np.cos(angle * corners) * min_radius * 0.8
            
            # Add to track
            track_data['track'].append({
                'x': float(x),
                'y': float(y),
                'width': 3.0
            })
        
        # Save to YAML file
        with open(output_file, 'w') as f:
            yaml.dump(track_data, f, default_flow_style=False)
        
        logger.info(f"Created example {difficulty} track at {output_file}")
        
        return output_file


# Example usage
if __name__ == "__main__":
    from ..core.vehicle import create_formula_student_vehicle
    import tempfile
    import os.path
    
    print("Formula Student Lap Time Simulation Demo")
    print("----------------------------------------")
    
    # Create a Formula Student vehicle
    vehicle = create_formula_student_vehicle()
    
    # Create a temporary directory for outputs
    output_dir = tempfile.mkdtemp()
    print(f"Creating output directory: {output_dir}")
    
    # Create an example track
    track_file = os.path.join(output_dir, "example_track.yaml")
    print("Generating example track...")
    create_example_track(track_file, difficulty='medium')
    
    # Basic vehicle specs
    print("\nVehicle Specifications:")
    print(f"  Engine: {vehicle.engine.make} {vehicle.engine.model}")
    print(f"  Power: {vehicle.engine.max_power} hp @ {vehicle.engine.max_power_rpm} RPM")
    print(f"  Torque: {vehicle.engine.max_torque} Nm @ {vehicle.engine.max_torque_rpm} RPM")
    print(f"  Mass: {vehicle.mass} kg")
    print(f"  Gears: {len(vehicle.drivetrain.transmission.gear_ratios)}")
    
    # Run lap time simulation
    print("\nRunning lap time simulation (this may take a moment)...")
    results = run_fs_lap_simulation(
        vehicle, 
        track_file, 
        include_thermal=True,
        save_dir=output_dir
    )
    
    # Print key results
    print("\nLap Time Simulation Results:")
    print(f"  Lap Time: {results['lap_time']:.3f} seconds")
    print(f"  Average Speed: {results['metrics']['avg_speed_kph']:.1f} km/h")
    print(f"  Maximum Speed: {results['metrics']['max_speed_kph']:.1f} km/h")
    print(f"  Time in Corners: {results['metrics']['time_in_corners']:.1f} seconds")
    print(f"  Time in Straights: {results['metrics']['time_in_straights']:.1f} seconds")
    
    if results['metrics']['thermal_limited']:
        print("  Note: Vehicle performance was thermally limited during the lap")
    
    # Run a comparison with a modified vehicle
    print("\nRunning comparison with modified vehicle setup...")
    
    # Create a modified vehicle with different parameters
    modified_vehicle = create_formula_student_vehicle()
    
    # Modify some parameters (just as an example)
    modified_vehicle.mass -= 20  # Reduce mass by 20 kg
    modified_vehicle.drag_coefficient *= 0.9  # Improve aero by 10%
    modified_vehicle.drivetrain.final_drive.ratio = 4.2  # Change final drive ratio
    
    # Create lap time simulator with both vehicles
    simulator = create_lap_time_simulator(vehicle, track_file)
    
    # Compare vehicles
    comparison_results = simulator.compare_vehicle_configs(
        [vehicle, modified_vehicle],
        ["Base Vehicle", "Modified Vehicle"],
        include_thermal=True,
        save_path=os.path.join(output_dir, "vehicle_comparison.png")
    )
    
    # Print comparison results
    print("\nVehicle Comparison Results:")
    for i, (config, lap_time) in enumerate(zip(comparison_results['configs'], 
                                             comparison_results['lap_times'])):
        print(f"  {config}: {lap_time:.3f} seconds")
    
    improvement = comparison_results['lap_times'][0] - comparison_results['lap_times'][1]
    if improvement > 0:
        print(f"  Improvement: {improvement:.3f} seconds ({improvement/comparison_results['lap_times'][0]*100:.1f}%)")
    else:
        print(f"  Degradation: {-improvement:.3f} seconds ({-improvement/comparison_results['lap_times'][0]*100:.1f}%)")
    
    # Display output location
    print(f"\nDetailed results saved to: {output_dir}")