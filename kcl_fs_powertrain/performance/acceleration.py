"""
Acceleration performance module for Formula Student powertrain simulation.

This module provides classes and functions for simulating, analyzing, and optimizing
the acceleration performance of a Formula Student vehicle. It includes specialized
tools for acceleration event simulations, launch control optimization, and
visualization of acceleration performance data.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Callable
import logging
from scipy.interpolate import interp1d

from ..core.vehicle import Vehicle
from ..transmission import CASSystem, ShiftDirection
from ..engine import TorqueCurve

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("Acceleration_Performance")


class AccelerationSimulator:
    """
    Simulator for Formula Student acceleration events.
    
    This class provides specialized tools for simulating acceleration events,
    optimizing launch control parameters, and analyzing acceleration performance.
    It builds on the vehicle model and adds specific functionality for acceleration
    events such as the 75m straight-line acceleration in Formula Student competitions.
    """
    
    def __init__(self, vehicle: Vehicle):
        """
        Initialize the acceleration simulator with a vehicle model.
        
        Args:
            vehicle: Vehicle model to use for simulation
        """
        self.vehicle = vehicle
        self.results_cache = {}  # Cache for simulation results
        
        # Default parameters
        self.distance = 75.0  # m (standard FS acceleration event)
        self.time_step = 0.01  # s
        self.max_time = 10.0  # s
        
        # Launch control parameters
        self.launch_rpm = 8000  # RPM for launch control
        self.launch_slip_target = 0.2  # Target wheel slip
        self.launch_duration = 0.5  # s duration of launch control phase
        
        logger.info("Acceleration simulator initialized")
    
    def configure(self, 
                distance: Optional[float] = None,
                time_step: Optional[float] = None,
                max_time: Optional[float] = None):
        """
        Configure simulation parameters.
        
        Args:
            distance: Distance in meters for acceleration run
            time_step: Time step in seconds for simulation
            max_time: Maximum time in seconds for simulation
        """
        if distance is not None:
            self.distance = distance
        
        if time_step is not None:
            self.time_step = time_step
        
        if max_time is not None:
            self.max_time = max_time
        
        logger.info(f"Configured simulator: distance={self.distance}m, time_step={self.time_step}s")
    
    def configure_launch_control(self,
                              launch_rpm: Optional[float] = None,
                              launch_slip_target: Optional[float] = None,
                              launch_duration: Optional[float] = None):
        """
        Configure launch control parameters.
        
        Args:
            launch_rpm: Engine RPM for launch control
            launch_slip_target: Target wheel slip ratio
            launch_duration: Duration of launch control phase in seconds
        """
        if launch_rpm is not None:
            self.launch_rpm = launch_rpm
        
        if launch_slip_target is not None:
            self.launch_slip_target = launch_slip_target
        
        if launch_duration is not None:
            self.launch_duration = launch_duration
        
        logger.info(f"Configured launch control: RPM={self.launch_rpm}, slip={self.launch_slip_target}")
    
    def simulate_acceleration(self, 
                           use_launch_control: bool = True,
                           optimized_shifts: bool = True) -> Dict:
        """
        Simulate an acceleration run.
        
        Args:
            use_launch_control: Whether to use launch control
            optimized_shifts: Whether to use optimized shift points
            
        Returns:
            Dictionary with simulation results
        """
        # Check if we have cached results with the current mass
        # Add the mass to the cache key
        vehicle_mass = self.vehicle.mass
        cache_key = f"accel_{use_launch_control}_{optimized_shifts}_{vehicle_mass}"
        
        if cache_key in self.results_cache:
            logger.info("Using cached acceleration results")
            return self.results_cache[cache_key]
        
        # For simple acceleration simulations, leverage the vehicle's built-in method
        if not use_launch_control and not optimized_shifts:
            # Use the vehicle's standard method
            return self.vehicle.simulate_acceleration_run(
                distance=self.distance,
                max_time=self.max_time,
                dt=self.time_step
            )
        
        # Check if we have cached results
        cache_key = f"accel_{use_launch_control}_{optimized_shifts}"
        if cache_key in self.results_cache:
            logger.info("Using cached acceleration results")
            return self.results_cache[cache_key]
        
        # Reset vehicle state
        self.vehicle.current_speed = 0.0
        self.vehicle.current_position = 0.0
        self.vehicle.current_acceleration = 0.0
        self.vehicle.current_gear = 1  # Start in first gear
        self.vehicle.current_engine_rpm = self.vehicle.engine.idle_rpm
        
        # Initialize results storage
        time_points = [0.0]
        speed_points = [0.0]
        position_points = [0.0]
        acceleration_points = [0.0]
        rpm_points = [self.vehicle.current_engine_rpm]
        gear_points = [self.vehicle.current_gear]
        wheel_slip_points = [0.0]
        
        # Get shift points
        if optimized_shifts and hasattr(self.vehicle, 'optimize_shift_points'):
            shift_point_data = self.vehicle.optimize_shift_points()
            shift_points = shift_point_data['upshift_points_by_gear'] if 'upshift_points_by_gear' in shift_point_data else {}
        else:
            # Default shift points at 95% of redline
            shift_points = {
                gear: self.vehicle.engine.redline * 0.95
                for gear in range(1, self.vehicle.drivetrain.transmission.num_gears)
            }
        
        # Launch control parameters
        in_launch_phase = use_launch_control
        launch_end_time = self.launch_duration if use_launch_control else 0
        
        # Simulation loop
        current_time = 0.0
        
        while current_time < self.max_time and self.vehicle.current_position < self.distance:
            # Apply full throttle
            throttle = 1.0
            
            # Launch control
            if in_launch_phase and current_time < launch_end_time:
                # During launch control, manage engine RPM
                if self.vehicle.current_engine_rpm > self.launch_rpm:
                    throttle = 0.8  # Reduce throttle to maintain launch RPM
                else:
                    throttle = 1.0
            else:
                in_launch_phase = False
            
            # Check for gear shift
            current_gear = self.vehicle.current_gear
            if current_gear in shift_points and self.vehicle.current_engine_rpm > shift_points[current_gear]:
                # Time to shift up
                next_gear = current_gear + 1
                
                # Use CAS system if available
                if hasattr(self.vehicle, 'cas_system') and self.vehicle.cas_system:
                    self.vehicle.cas_system.request_shift(ShiftDirection.UP, next_gear)
                else:
                    # Manual shift
                    self.vehicle.current_gear = next_gear
                    
                    # Update drivetrain too if necessary
                    if hasattr(self.vehicle, 'drivetrain') and hasattr(self.vehicle.drivetrain, 'change_gear'):
                        self.vehicle.drivetrain.change_gear(next_gear)
            
            # Calculate wheel slip (simplified)
            if self.vehicle.current_speed > 0.1:
                engine_speed_rad_s = self.vehicle.current_engine_rpm * 2 * np.pi / 60
                wheel_speed_rad_s = self.vehicle.current_speed / self.vehicle.tire_radius
                
                # Calculate wheel speed from engine through drivetrain
                if hasattr(self.vehicle.drivetrain, 'get_overall_ratio'):
                    overall_ratio = self.vehicle.drivetrain.get_overall_ratio(self.vehicle.current_gear)
                else:
                    # Simplified calculation
                    overall_ratio = self.vehicle.drivetrain.transmission.get_ratio(
                        self.vehicle.current_gear
                    ) * self.vehicle.drivetrain.final_drive.get_ratio()
                
                wheel_speed_from_engine = engine_speed_rad_s / overall_ratio
                wheel_slip = max(0, (wheel_speed_from_engine - wheel_speed_rad_s) / wheel_speed_rad_s)
            else:
                wheel_slip = 0.0
            
            # Calculate acceleration
            self.vehicle.calculate_acceleration(throttle, 0.0, self.vehicle.current_gear, self.vehicle.current_speed)
            
            # Update vehicle state
            self.vehicle.update_vehicle_state(self.time_step)
            
            # Update time
            current_time += self.time_step
            
            # Store results
            time_points.append(current_time)
            speed_points.append(self.vehicle.current_speed)
            position_points.append(self.vehicle.current_position)
            acceleration_points.append(self.vehicle.current_acceleration)
            rpm_points.append(self.vehicle.current_engine_rpm)
            gear_points.append(self.vehicle.current_gear)
            wheel_slip_points.append(wheel_slip)
        
        # Calculate results
        finish_time = None
        finish_speed = None
        
        # Interpolate to find exact finish time
        if self.vehicle.current_position >= self.distance:
            idx = np.searchsorted(position_points, self.distance)
            if idx > 0 and idx < len(time_points):
                # Linear interpolation
                t0, t1 = time_points[idx-1], time_points[idx]
                p0, p1 = position_points[idx-1], position_points[idx]
                finish_time = t0 + (t1 - t0) * (self.distance - p0) / (p1 - p0)
                
                # Interpolate finish speed
                s0, s1 = speed_points[idx-1], speed_points[idx]
                finish_speed = s0 + (s1 - s0) * (finish_time - t0) / (t1 - t0)
        
        # Calculate 0-60 mph time
        time_to_60mph = None
        mps_to_mph = 2.23694  # m/s to mph conversion
        target_speed = 60.0 / mps_to_mph  # 60 mph in m/s
        
        if max(speed_points) >= target_speed:
            idx = np.searchsorted(speed_points, target_speed)
            if idx > 0 and idx < len(time_points):
                # Linear interpolation
                t0, t1 = time_points[idx-1], time_points[idx]
                s0, s1 = speed_points[idx-1], speed_points[idx]
                time_to_60mph = t0 + (t1 - t0) * (target_speed - s0) / (s1 - s0)
        
        # Calculate 0-100 km/h time
        time_to_100kph = None
        mps_to_kph = 3.6  # m/s to km/h conversion
        target_speed_100 = 100.0 / mps_to_kph  # 100 km/h in m/s
        
        if max(speed_points) >= target_speed_100:
            idx = np.searchsorted(speed_points, target_speed_100)
            if idx > 0 and idx < len(time_points):
                # Linear interpolation
                t0, t1 = time_points[idx-1], time_points[idx]
                s0, s1 = speed_points[idx-1], speed_points[idx]
                time_to_100kph = t0 + (t1 - t0) * (target_speed_100 - s0) / (s1 - s0)
        
        # Compile results
        results = {
            'time': np.array(time_points),
            'speed': np.array(speed_points),
            'position': np.array(position_points),
            'acceleration': np.array(acceleration_points),
            'engine_rpm': np.array(rpm_points),
            'gear': np.array(gear_points),
            'wheel_slip': np.array(wheel_slip_points),
            'finish_time': finish_time,
            'finish_speed': finish_speed,
            'time_to_60mph': time_to_60mph,
            'time_to_100kph': time_to_100kph,
            'distance': self.distance,
            'used_launch_control': use_launch_control,
            'used_optimized_shifts': optimized_shifts
        }
        
        # Cache results
        self.results_cache[cache_key] = results
        
        logger.info(f"Acceleration simulation completed: time={finish_time:.2f}s, speed={finish_speed*2.23694:.1f}mph")
        
        return results
    
    def optimize_shift_points(self) -> Dict[int, float]:
        """
        Optimize gear shift points for maximum acceleration.
        
        Returns:
            Dictionary mapping gear numbers to optimal shift RPM
        """
        # Use the vehicle's built-in function if available
        if hasattr(self.vehicle, 'optimize_shift_points'):
            result = self.vehicle.optimize_shift_points()
            if 'upshift_points_by_gear' in result:
                return result['upshift_points_by_gear']
        
        # Fallback to a simpler optimization
        shift_points = {}
        
        # For each gear (except highest)
        for i in range(1, self.vehicle.drivetrain.transmission.num_gears):
            current_gear = i
            next_gear = i + 1
            
            # Get gear ratios
            current_ratio = self.vehicle.drivetrain.transmission.gear_ratios[current_gear - 1]
            next_ratio = self.vehicle.drivetrain.transmission.gear_ratios[next_gear - 1]
            
            # Calculate engine RPM after shift for a range of RPMs
            rpm_range = np.linspace(self.vehicle.engine.max_torque_rpm, self.vehicle.engine.redline, 50)
            best_shift_rpm = self.vehicle.engine.max_power_rpm  # Default to max power
            best_acceleration = 0.0
            
            for rpm in rpm_range:
                # Calculate engine torque at current RPM
                torque_current = self.vehicle.engine.get_torque(rpm)
                
                # Calculate engine RPM after shift
                rpm_after_shift = rpm * (current_ratio / next_ratio)
                
                # Calculate engine torque after shift
                torque_after_shift = self.vehicle.engine.get_torque(rpm_after_shift)
                
                # Calculate wheel torque and tractive force for both scenarios
                wheel_torque_current = self.vehicle.drivetrain.calculate_wheel_torque(torque_current, current_gear)
                wheel_torque_after = self.vehicle.drivetrain.calculate_wheel_torque(torque_after_shift, next_gear)
                
                tractive_force_current = wheel_torque_current / self.vehicle.tire_radius
                tractive_force_after = wheel_torque_after / self.vehicle.tire_radius
                
                # Calculate acceleration for both scenarios
                accel_current = tractive_force_current / self.vehicle.mass
                accel_after = tractive_force_after / self.vehicle.mass
                
                # If acceleration after shift is better, this is a good shift point
                if accel_after > accel_current and accel_after > best_acceleration:
                    best_acceleration = accel_after
                    best_shift_rpm = rpm
            
            shift_points[current_gear] = best_shift_rpm
        
        logger.info(f"Optimized shift points: {shift_points}")
        return shift_points
    
    def optimize_launch_control(self, 
                             rpm_range: Optional[List[float]] = None,
                             slip_range: Optional[List[float]] = None) -> Dict:
        """
        Optimize launch control parameters for best acceleration.
        
        Args:
            rpm_range: List of RPM values to test for launch control
            slip_range: List of wheel slip targets to test
            
        Returns:
            Dictionary with optimized launch control parameters
        """
        # Set up default ranges if not provided
        if rpm_range is None:
            # Default range from 60% to 90% of redline
            rpm_min = int(self.vehicle.engine.max_torque_rpm * 0.8)
            rpm_max = int(self.vehicle.engine.redline * 0.9)
            rpm_range = np.linspace(rpm_min, rpm_max, 5).tolist()
        
        if slip_range is None:
            # Default wheel slip range
            slip_range = [0.1, 0.15, 0.2, 0.25, 0.3]
        
        # Store results
        results = []
        
        # Test each combination
        for rpm in rpm_range:
            for slip in slip_range:
                # Configure launch control
                self.configure_launch_control(launch_rpm=rpm, launch_slip_target=slip)
                
                # Run simulation
                sim_result = self.simulate_acceleration(use_launch_control=True)
                
                # Store result
                results.append({
                    'launch_rpm': rpm,
                    'slip_target': slip,
                    'finish_time': sim_result['finish_time'],
                    'time_to_60mph': sim_result['time_to_60mph'],
                    'time_to_100kph': sim_result['time_to_100kph']
                })
        
        # Find best result based on finish time
        best_result = min(results, key=lambda x: x['finish_time'] if x['finish_time'] is not None else float('inf'))
        
        # Set best parameters
        self.configure_launch_control(
            launch_rpm=best_result['launch_rpm'],
            launch_slip_target=best_result['slip_target']
        )
        
        logger.info(f"Optimized launch control: RPM={best_result['launch_rpm']}, slip={best_result['slip_target']}")
        
        return {
            'best_parameters': best_result,
            'all_results': results,
            'rpm_range': rpm_range,
            'slip_range': slip_range
        }
    
    def analyze_performance_metrics(self, results: Dict) -> Dict:
        """
        Calculate and analyze acceleration performance metrics.
        
        Args:
            results: Results from simulate_acceleration
            
        Returns:
            Dictionary with analyzed performance metrics
        """
        # Extract data
        time = results['time']
        speed = results['speed']
        accel = results['acceleration']
        position = results['position']
        
        # Calculate peak values
        peak_acceleration = np.max(accel)
        peak_accel_time = time[np.argmax(accel)]
        
        # Calculate average acceleration
        if results['finish_time'] is not None:
            avg_acceleration = results['finish_speed'] / results['finish_time']
        else:
            avg_acceleration = None
        
        # Calculate split times (25m, 50m, 75m)
        split_times = {}
        for split in [25.0, 50.0, 75.0]:
            if split <= np.max(position):
                idx = np.searchsorted(position, split)
                if idx > 0 and idx < len(time):
                    # Linear interpolation
                    t0, t1 = time[idx-1], time[idx]
                    p0, p1 = position[idx-1], position[idx]
                    split_time = t0 + (t1 - t0) * (split - p0) / (p1 - p0)
                    split_times[split] = split_time
            else:
                split_times[split] = None
        
        # Calculate acceleration times (0-20, 0-40, 0-60, 0-100 km/h)
        accel_times = {}
        for target_kph in [20, 40, 60, 80, 100]:
            target_speed = target_kph / 3.6  # Convert to m/s
            if target_speed <= np.max(speed):
                idx = np.searchsorted(speed, target_speed)
                if idx > 0 and idx < len(time):
                    # Linear interpolation
                    t0, t1 = time[idx-1], time[idx]
                    s0, s1 = speed[idx-1], speed[idx]
                    accel_time = t0 + (t1 - t0) * (target_speed - s0) / (s1 - s0)
                    accel_times[target_kph] = accel_time
            else:
                accel_times[target_kph] = None
        
        # Performance grade (simplified A-F grade)
        grade = 'F'
        if results['finish_time'] is not None:
            if results['finish_time'] < 3.8:
                grade = 'A+'
            elif results['finish_time'] < 4.0:
                grade = 'A'
            elif results['finish_time'] < 4.3:
                grade = 'B'
            elif results['finish_time'] < 4.6:
                grade = 'C'
            elif results['finish_time'] < 5.0:
                grade = 'D'
            else:
                grade = 'F'
        
        # Compile metrics
        metrics = {
            'finish_time': results['finish_time'],
            'finish_speed': results['finish_speed'],
            'finish_speed_mph': results['finish_speed'] * 2.23694 if results['finish_speed'] is not None else None,
            'time_to_60mph': results['time_to_60mph'],
            'time_to_100kph': results['time_to_100kph'],
            'peak_acceleration': peak_acceleration,
            'peak_acceleration_g': peak_acceleration / 9.81,
            'peak_acceleration_time': peak_accel_time,
            'average_acceleration': avg_acceleration,
            'split_times': split_times,
            'acceleration_times': accel_times,
            'performance_grade': grade
        }
        
        return metrics
    
    def plot_acceleration_results(self, results: Dict, save_path: Optional[str] = None,
                                plot_wheel_slip: bool = False):
        """
        Plot acceleration simulation results.
        
        Args:
            results: Results from simulate_acceleration
            save_path: Optional path to save the plot
            plot_wheel_slip: Whether to plot wheel slip
        """
        # Extract data
        time = results['time']
        speed = results['speed']
        acceleration = results['acceleration']
        rpm = results['engine_rpm']
        gear = results['gear']
        
        # Convert speed to mph for display
        speed_mph = speed * 2.23694
        
        # Create figure with multiple subplots
        if plot_wheel_slip:
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 14), sharex=True)
        else:
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        
        # Plot speed
        ax1.plot(time, speed_mph, 'b-', linewidth=2)
        ax1.set_ylabel('Speed (mph)')
        ax1.set_title('Acceleration Performance Results')
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Add 0-60 mph time if available
        if results['time_to_60mph'] is not None:
            ax1.axhline(y=60, color='r', linestyle='--', alpha=0.5)
            ax1.axvline(x=results['time_to_60mph'], color='r', linestyle='--', alpha=0.5)
            ax1.text(
                results['time_to_60mph'] + 0.1, 
                62, 
                f"0-60 mph: {results['time_to_60mph']:.2f}s",
                color='r',
                fontweight='bold'
            )
        
        # Add 0-100 kph time if available
        if results['time_to_100kph'] is not None:
            # Convert 100kph to mph for plot
            kph100_in_mph = 100 / 1.60934
            ax1.axhline(y=kph100_in_mph, color='g', linestyle='--', alpha=0.5)
            ax1.axvline(x=results['time_to_100kph'], color='g', linestyle='--', alpha=0.5)
            ax1.text(
                results['time_to_100kph'] + 0.1, 
                kph100_in_mph + 2, 
                f"0-100 km/h: {results['time_to_100kph']:.2f}s",
                color='g',
                fontweight='bold'
            )
        
        # Plot acceleration
        ax2.plot(time, acceleration, 'g-', linewidth=2)
        ax2.set_ylabel('Acceleration (m/s²)')
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        # Add peak acceleration
        peak_idx = np.argmax(acceleration)
        ax2.plot(time[peak_idx], acceleration[peak_idx], 'ro')
        ax2.text(
            time[peak_idx] + 0.1,
            acceleration[peak_idx],
            f"Peak: {acceleration[peak_idx]:.2f} m/s² ({acceleration[peak_idx]/9.81:.2f}g)",
            color='r'
        )
        
        # Plot RPM and gear
        color1 = 'tab:blue'
        ax3.set_ylabel('Engine RPM', color=color1)
        ax3.plot(time, rpm, color=color1, linewidth=2)
        ax3.tick_params(axis='y', labelcolor=color1)
        ax3.grid(True, linestyle='--', alpha=0.7)
        
        # Add horizontal lines at shift points
        if 'used_optimized_shifts' in results and results['used_optimized_shifts']:
            shift_points = self.optimize_shift_points()
            for gear_num, shift_rpm in shift_points.items():
                ax3.axhline(y=shift_rpm, color='r', linestyle=':', alpha=0.3)
                ax3.text(0.1, shift_rpm + 100, f"Shift {gear_num}→{gear_num+1}", 
                       color='r', alpha=0.7, fontsize=8)
        
        color2 = 'tab:red'
        ax3_twin = ax3.twinx()
        ax3_twin.set_ylabel('Gear', color=color2)
        ax3_twin.step(time, gear, color=color2, linewidth=2, where='post')
        ax3_twin.tick_params(axis='y', labelcolor=color2)
        ax3_twin.set_yticks(range(0, self.vehicle.drivetrain.transmission.num_gears + 1))
        
        # Plot wheel slip if requested
        if plot_wheel_slip and 'wheel_slip' in results:
            wheel_slip = results['wheel_slip']
            ax4.plot(time, wheel_slip, 'b-', linewidth=2)
            ax4.set_ylabel('Wheel Slip Ratio')
            ax4.set_xlabel('Time (s)')
            ax4.grid(True, linestyle='--', alpha=0.7)
            
            # Add target slip line if using launch control
            if 'used_launch_control' in results and results['used_launch_control']:
                ax4.axhline(y=self.launch_slip_target, color='r', linestyle='--', alpha=0.5)
                ax4.text(0.1, self.launch_slip_target + 0.01, f"Target Slip: {self.launch_slip_target:.2f}", 
                       color='r', alpha=0.7)
        else:
            ax3.set_xlabel('Time (s)')
        
        # Add summary text
        if results['finish_time'] is not None:
            plt.figtext(
                0.5, 0.01,
                f"Distance: {results['distance']}m, Time: {results['finish_time']:.2f}s, " + 
                f"Final Speed: {results['finish_speed'] * 2.23694:.1f} mph",
                ha='center',
                fontsize=12,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray')
            )
        
        plt.tight_layout(rect=[0, 0.03, 1, 1])
        
        # Save plot if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        plt.show()
        
    def plot_acceleration_comparison(self, results_list: List[Dict], labels: List[str],
                                  save_path: Optional[str] = None):
        """
        Plot comparison of multiple acceleration simulations.
        
        Args:
            results_list: List of result dictionaries from simulate_acceleration
            labels: List of labels for each result set
            save_path: Optional path to save the plot
        """
        if not results_list or len(results_list) != len(labels):
            logger.error("Results list and labels must have the same length")
            return
        
        # Create figure with multiple subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
        
        # Colors for different runs
        colors = plt.cm.tab10.colors
        
        # Plot speed for each result
        for i, (results, label) in enumerate(zip(results_list, labels)):
            color = colors[i % len(colors)]
            
            # Extract data
            time = results['time']
            speed = results['speed'] * 2.23694  # Convert to mph
            accel = results['acceleration']
            position = results['position']
            
            # Plot speed vs time
            ax1.plot(time, speed, color=color, linewidth=2, label=f"{label}")
        
        ax1.set_ylabel('Speed (mph)')
        ax1.set_title('Acceleration Performance Comparison')
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend(loc='best')
        
        # Plot speed vs distance
        for i, (results, label) in enumerate(zip(results_list, labels)):
            color = colors[i % len(colors)]
            
            # Extract data
            position = results['position']
            speed = results['speed'] * 2.23694  # Convert to mph
            
            # Plot speed vs distance
            ax2.plot(position, speed, color=color, linewidth=2, label=f"{label}")
        
        ax2.set_ylabel('Speed (mph)')
        ax2.set_xlabel('Distance (m)')
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend(loc='best')
        
        # Plot acceleration vs time
        for i, (results, label) in enumerate(zip(results_list, labels)):
            color = colors[i % len(colors)]
            
            # Extract data
            time = results['time']
            accel = results['acceleration']
            
            # Plot acceleration
            ax3.plot(time, accel, color=color, linewidth=2, label=f"{label}")
        
        ax3.set_ylabel('Acceleration (m/s²)')
        ax3.set_xlabel('Time (s)')
        ax3.grid(True, linestyle='--', alpha=0.7)
        ax3.legend(loc='best')
        
        # Add summary table
        summary_text = "Performance Summary:\n"
        summary_text += f"{'Configuration':<20} {'Time (s)':<10} {'0-60 mph (s)':<13} {'Final Speed (mph)':<18}\n"
        summary_text += "-" * 65 + "\n"
        
        for i, (results, label) in enumerate(zip(results_list, labels)):
            finish_time = results['finish_time']
            time_to_60 = results['time_to_60mph']
            finish_speed = results['finish_speed'] * 2.23694 if results['finish_speed'] is not None else None
            
            finish_time_str = f"{finish_time:.2f}" if finish_time is not None else "N/A"
            time_to_60_str = f"{time_to_60:.2f}" if time_to_60 is not None else "N/A"
            finish_speed_str = f"{finish_speed:.1f}" if finish_speed is not None else "N/A"
            
            summary_text += f"{label:<20} {finish_time_str:<10} {time_to_60_str:<13} {finish_speed_str:<18}\n"
        
        plt.figtext(0.5, 0.01, summary_text, ha='center', fontsize=10, 
                  bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'),
                  family='monospace')
        
        plt.tight_layout(rect=[0, 0.15, 1, 1])
        
        # Save plot if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        plt.show()
    
    def generate_acceleration_report(self, save_dir: Optional[str] = None) -> Dict:
        """
        Generate a comprehensive acceleration performance report.
        
        Args:
            save_dir: Optional directory to save plots and data
            
        Returns:
            Dictionary with report data
        """
        # Create save directory if provided
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        # Run simulations with different configurations
        logger.info("Running acceleration simulations for report...")
        
        # Base case: No launch control, no optimized shifts
        base_results = self.simulate_acceleration(use_launch_control=False, optimized_shifts=False)
        
        # With optimized shifts
        optimized_shift_results = self.simulate_acceleration(use_launch_control=False, optimized_shifts=True)
        
        # With launch control
        launch_results = self.simulate_acceleration(use_launch_control=True, optimized_shifts=False)
        
        # With both launch control and optimized shifts
        full_optimized_results = self.simulate_acceleration(use_launch_control=True, optimized_shifts=True)
        
        # Analyze metrics for each simulation
        base_metrics = self.analyze_performance_metrics(base_results)
        optimized_shift_metrics = self.analyze_performance_metrics(optimized_shift_results)
        launch_metrics = self.analyze_performance_metrics(launch_results)
        full_optimized_metrics = self.analyze_performance_metrics(full_optimized_results)
        
        # Save plots if directory provided
        if save_dir:
            # Plot individual results
            self.plot_acceleration_results(
                base_results, 
                save_path=os.path.join(save_dir, "base_acceleration.png")
            )
            
            self.plot_acceleration_results(
                full_optimized_results, 
                plot_wheel_slip=True,
                save_path=os.path.join(save_dir, "optimized_acceleration.png")
            )
            
            # Plot comparison
            self.plot_acceleration_comparison(
                [base_results, optimized_shift_results, launch_results, full_optimized_results],
                ["Base", "Optimized Shifts", "Launch Control", "Full Optimization"],
                save_path=os.path.join(save_dir, "acceleration_comparison.png")
            )
            
            # Save results to CSV
            results_df = pd.DataFrame({
                "Metric": [
                    "Finish Time (s)",
                    "Finish Speed (mph)",
                    "0-60 mph Time (s)",
                    "0-100 km/h Time (s)",
                    "Peak Acceleration (g)",
                    "Average Acceleration (m/s²)",
                    "Performance Grade"
                ],
                "Base": [
                    base_metrics['finish_time'],
                    base_metrics['finish_speed_mph'],
                    base_metrics['time_to_60mph'],
                    base_metrics['time_to_100kph'],
                    base_metrics['peak_acceleration_g'],
                    base_metrics['average_acceleration'],
                    base_metrics['performance_grade']
                ],
                "Optimized Shifts": [
                    optimized_shift_metrics['finish_time'],
                    optimized_shift_metrics['finish_speed_mph'],
                    optimized_shift_metrics['time_to_60mph'],
                    optimized_shift_metrics['time_to_100kph'],
                    optimized_shift_metrics['peak_acceleration_g'],
                    optimized_shift_metrics['average_acceleration'],
                    optimized_shift_metrics['performance_grade']
                ],
                "Launch Control": [
                    launch_metrics['finish_time'],
                    launch_metrics['finish_speed_mph'],
                    launch_metrics['time_to_60mph'],
                    launch_metrics['time_to_100kph'],
                    launch_metrics['peak_acceleration_g'],
                    launch_metrics['average_acceleration'],
                    launch_metrics['performance_grade']
                ],
                "Full Optimization": [
                    full_optimized_metrics['finish_time'],
                    full_optimized_metrics['finish_speed_mph'],
                    full_optimized_metrics['time_to_60mph'],
                    full_optimized_metrics['time_to_100kph'],
                    full_optimized_metrics['peak_acceleration_g'],
                    full_optimized_metrics['average_acceleration'],
                    full_optimized_metrics['performance_grade']
                ]
            })
            
            results_df.to_csv(os.path.join(save_dir, "acceleration_metrics.csv"), index=False)
        
        # Compile report data
        report = {
            'simulations': {
                'base': base_results,
                'optimized_shifts': optimized_shift_results,
                'launch_control': launch_results,
                'full_optimized': full_optimized_results
            },
            'metrics': {
                'base': base_metrics,
                'optimized_shifts': optimized_shift_metrics,
                'launch_control': launch_metrics,
                'full_optimized': full_optimized_metrics
            },
            'improvement': {
                'optimized_shifts_vs_base': (base_metrics['finish_time'] - optimized_shift_metrics['finish_time']) / base_metrics['finish_time'] * 100 if base_metrics['finish_time'] else None,
                'launch_vs_base': (base_metrics['finish_time'] - launch_metrics['finish_time']) / base_metrics['finish_time'] * 100 if base_metrics['finish_time'] else None,
                'full_vs_base': (base_metrics['finish_time'] - full_optimized_metrics['finish_time']) / base_metrics['finish_time'] * 100 if base_metrics['finish_time'] else None
            },
            'vehicle_specs': {
                'mass': self.vehicle.mass,
                'power': self.vehicle.engine.max_power,
                'torque': self.vehicle.engine.max_torque,
                'gear_ratios': self.vehicle.drivetrain.transmission.gear_ratios,
                'final_drive': self.vehicle.drivetrain.final_drive.get_ratio()
            },
            'optimized_parameters': {
                'shift_points': self.optimize_shift_points(),
                'launch_rpm': self.launch_rpm,
                'launch_slip_target': self.launch_slip_target
            }
        }
        
        logger.info("Acceleration report generated")
        return report


def create_acceleration_simulator(vehicle: Vehicle) -> AccelerationSimulator:
    """
    Create and configure an acceleration simulator for a Formula Student vehicle.
    
    Args:
        vehicle: Vehicle model to use for simulation
        
    Returns:
        Configured AccelerationSimulator
    """
    simulator = AccelerationSimulator(vehicle)
    
    # Configure with standard FS acceleration event parameters
    simulator.configure(
        distance=75.0,
        time_step=0.01,
        max_time=8.0
    )
    
    # Configure launch control with reasonable defaults
    simulator.configure_launch_control(
        launch_rpm=vehicle.engine.max_torque_rpm * 1.1,
        launch_slip_target=0.2,
        launch_duration=0.5
    )
    
    return simulator


def run_fs_acceleration_simulation(vehicle: Vehicle, 
                                save_dir: Optional[str] = None) -> Dict:
    """
    Run a complete Formula Student acceleration event simulation.
    
    Args:
        vehicle: Vehicle model to use for simulation
        save_dir: Optional directory to save results
        
    Returns:
        Dictionary with simulation results and metrics
    """
    # Create simulator
    simulator = create_acceleration_simulator(vehicle)
    
    # Generate report
    report = simulator.generate_acceleration_report(save_dir)
    
    # Print key results
    best_time = report['metrics']['full_optimized']['finish_time']
    best_60mph = report['metrics']['full_optimized']['time_to_60mph']
    
    logger.info(f"Acceleration simulation complete:")
    logger.info(f"  75m time: {best_time:.2f}s")
    logger.info(f"  0-60 mph: {best_60mph:.2f}s")
    logger.info(f"  Grade: {report['metrics']['full_optimized']['performance_grade']}")
    
    # Calculate points based on Formula Student rules (simplified)
    max_points = 75.0  # Maximum points for acceleration event
    min_time = 3.5  # Theoretical minimum time (for max points)
    max_time = 6.0  # Maximum time (for min points)
    
    if best_time <= min_time:
        points = max_points
    elif best_time >= max_time:
        points = 0.0
    else:
        points = max_points * (max_time - best_time) / (max_time - min_time)
    
    logger.info(f"  Estimated FS points: {points:.1f}/75.0")
    
    return {
        'report': report,
        'best_time': best_time,
        'best_60mph': best_60mph,
        'points': points
    }


# Example usage
if __name__ == "__main__":
    from ..core.vehicle import create_formula_student_vehicle
    
    # Create a Formula Student vehicle
    vehicle = create_formula_student_vehicle()
    
    # Create acceleration simulator
    simulator = create_acceleration_simulator(vehicle)
    
    # Run and print basic acceleration test
    results = simulator.simulate_acceleration(use_launch_control=True, optimized_shifts=True)
    metrics = simulator.analyze_performance_metrics(results)
    
    print(f"Acceleration Test Results:")
    print(f"  75m Time: {metrics['finish_time']:.2f} seconds")
    print(f"  0-60 mph: {metrics['time_to_60mph']:.2f} seconds")
    print(f"  Peak Acceleration: {metrics['peak_acceleration_g']:.2f}g")
    print(f"  Grade: {metrics['performance_grade']}")
    
    # Plot results
    simulator.plot_acceleration_results(results, plot_wheel_slip=True)