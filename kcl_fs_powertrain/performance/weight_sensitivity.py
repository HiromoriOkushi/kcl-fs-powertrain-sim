"""
Weight sensitivity analysis module for Formula Student powertrain.

This module provides classes and functions for analyzing the sensitivity of
vehicle performance to changes in weight. It helps quantify the performance
impact of weight changes and identify optimal weight reduction targets.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Callable
import logging
from scipy.interpolate import interp1d
import time

from ..core.vehicle import Vehicle
from .acceleration import AccelerationSimulator, create_acceleration_simulator
from .lap_time import LapTimeSimulator, create_lap_time_simulator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("Weight_Sensitivity")


class WeightSensitivityAnalyzer:
    """
    Analyzer for vehicle weight sensitivity.
    
    This class provides tools for analyzing how vehicle performance metrics
    change with variations in vehicle weight. It can quantify the sensitivity
    of acceleration times, lap times, and other performance metrics to weight
    changes.
    """
    
    def __init__(self, vehicle: Vehicle):
        """
        Initialize the weight sensitivity analyzer with a vehicle model.
        
        Args:
            vehicle: Vehicle model to use for analysis
        """
        self.vehicle = vehicle
        self.base_weight = vehicle.mass
        
        # Store base performance metrics
        self.base_metrics = {}
        
        # Store sensitivity results
        self.acceleration_sensitivity = {}
        self.lap_time_sensitivity = {}
        
        # Create simulators
        self.acceleration_simulator = create_acceleration_simulator(vehicle)
        self.lap_time_simulator = None  # Will be created when needed with a track
        
        logger.info(f"Weight sensitivity analyzer initialized with base weight: {self.base_weight:.1f} kg")
    
    def analyze_acceleration_sensitivity(self, 
                                  weight_range: Tuple[float, float],
                                  num_points: int = 5,
                                  use_launch_control: bool = True,
                                  use_optimized_shifts: bool = True) -> Dict:
        """
        Analyze sensitivity of acceleration performance to weight changes.
        
        Args:
            weight_range: Tuple of (min_weight, max_weight) in kg
            num_points: Number of weight points to analyze
            use_launch_control: Whether to use launch control in acceleration tests
            use_optimized_shifts: Whether to use optimized shift points
            
        Returns:
            Dictionary with analysis results
        """
        logger.info(f"Analyzing acceleration sensitivity from {weight_range[0]:.1f} to {weight_range[1]:.1f} kg")
        
        # Clear the acceleration simulator's cache to force fresh calculations
        self.acceleration_simulator.results_cache = {}
        
        # Generate weight points
        weight_points = np.linspace(weight_range[0], weight_range[1], num_points)
        
        # Store results
        time_to_60mph = []
        time_to_100kph = []
        time_75m = []
        weights = []
        
        # Original weight
        original_weight = self.vehicle.mass
        
        # Run simulations at each weight point
        for weight in weight_points:
            # Set vehicle weight
            self.vehicle.mass = weight
            
            # Run acceleration simulation
            results = self.acceleration_simulator.simulate_acceleration(
                use_launch_control=use_launch_control,
                optimized_shifts=use_optimized_shifts
            )
            
            # Store results
            weights.append(weight)
            time_to_60mph.append(results['time_to_60mph'])
            time_to_100kph.append(results['time_to_100kph'])
            time_75m.append(results['finish_time'])
            
            # Safely format None values for logging
            time_60mph_str = f"{results['time_to_60mph']:.3f}" if results['time_to_60mph'] is not None else "N/A"
            time_100kph_str = f"{results['time_to_100kph']:.3f}" if results['time_to_100kph'] is not None else "N/A"
            time_75m_str = f"{results['finish_time']:.3f}" if results['finish_time'] is not None else "N/A"
            
            logger.info(f"Weight: {weight:.1f} kg, 0-60 mph: {time_60mph_str} s, "
                    f"0-100 kph: {time_100kph_str} s, 75m: {time_75m_str} s")
        
        # Restore original weight
        self.vehicle.mass = original_weight
        
        # Filter out None values for sensitivity calculations
        valid_weights_60mph = []
        valid_times_60mph = []
        for w, t in zip(weights, time_to_60mph):
            if t is not None:
                valid_weights_60mph.append(w)
                valid_times_60mph.append(t)
                
        valid_weights_100kph = []
        valid_times_100kph = []
        for w, t in zip(weights, time_to_100kph):
            if t is not None:
                valid_weights_100kph.append(w)
                valid_times_100kph.append(t)
                
        valid_weights_75m = []
        valid_times_75m = []
        for w, t in zip(weights, time_75m):
            if t is not None:
                valid_weights_75m.append(w)
                valid_times_75m.append(t)
        
        # Calculate sensitivity coefficients using only valid (non-None) values
        time_to_60mph_slope = self._calculate_sensitivity_coefficient(valid_weights_60mph, valid_times_60mph) if valid_times_60mph else 0
        time_to_100kph_slope = self._calculate_sensitivity_coefficient(valid_weights_100kph, valid_times_100kph) if valid_times_100kph else 0
        time_75m_slope = self._calculate_sensitivity_coefficient(valid_weights_75m, valid_times_75m) if valid_times_75m else 0
        
        # Calculate percentage improvements per 10kg if baseline values are valid
        seconds_per_10kg_60mph = time_to_60mph_slope * 10
        seconds_per_10kg_100kph = time_to_100kph_slope * 10
        seconds_per_10kg_75m = time_75m_slope * 10
        
        percent_improvement_per_10kg_60mph = 0
        percent_improvement_per_10kg_100kph = 0
        percent_improvement_per_10kg_75m = 0
        
        if valid_times_60mph:
            percent_improvement_per_10kg_60mph = (seconds_per_10kg_60mph / valid_times_60mph[0]) * 100
        
        if valid_times_100kph:
            percent_improvement_per_10kg_100kph = (seconds_per_10kg_100kph / valid_times_100kph[0]) * 100
            
        if valid_times_75m:
            percent_improvement_per_10kg_75m = (seconds_per_10kg_75m / valid_times_75m[0]) * 100
        
        # Store results
        sensitivity_results = {
            'weights': weights,
            'time_to_60mph': time_to_60mph,
            'time_to_100kph': time_to_100kph,
            'time_75m': time_75m,
            'sensitivity_60mph': time_to_60mph_slope,
            'sensitivity_100kph': time_to_100kph_slope,
            'sensitivity_75m': time_75m_slope,
            'seconds_per_10kg_60mph': seconds_per_10kg_60mph,
            'seconds_per_10kg_100kph': seconds_per_10kg_100kph,
            'seconds_per_10kg_75m': seconds_per_10kg_75m,
            'percent_improvement_per_10kg_60mph': percent_improvement_per_10kg_60mph,
            'percent_improvement_per_10kg_100kph': percent_improvement_per_10kg_100kph,
            'percent_improvement_per_10kg_75m': percent_improvement_per_10kg_75m
        }
        
        # Store in class variable
        self.acceleration_sensitivity = sensitivity_results
        
        logger.info("Acceleration sensitivity analysis completed")
        logger.info(f"0-60 mph sensitivity: {time_to_60mph_slope:.4f} seconds per kg "
                f"({time_to_60mph_slope * 10:.4f} seconds per 10 kg)")
        logger.info(f"0-100 kph sensitivity: {time_to_100kph_slope:.4f} seconds per kg "
                f"({time_to_100kph_slope * 10:.4f} seconds per 10 kg)")
        logger.info(f"75m time sensitivity: {time_75m_slope:.4f} seconds per kg "
                f"({time_75m_slope * 10:.4f} seconds per 10 kg)")
        
        return sensitivity_results
    
    def analyze_lap_time_sensitivity(self, 
                                  track_file: str,
                                  weight_range: Tuple[float, float],
                                  num_points: int = 5,
                                  include_thermal: bool = True) -> Dict:
        """
        Analyze sensitivity of lap time performance to weight changes.
        
        Args:
            track_file: Path to track file
            weight_range: Tuple of (min_weight, max_weight) in kg
            num_points: Number of weight points to analyze
            include_thermal: Whether to include thermal effects
            
        Returns:
            Dictionary with analysis results
        """
        logger.info(f"Analyzing lap time sensitivity from {weight_range[0]:.1f} to {weight_range[1]:.1f} kg")
        
        # Create lap time simulator if not already created
        if self.lap_time_simulator is None:
            self.lap_time_simulator = create_lap_time_simulator(self.vehicle, track_file)
        
        # Generate weight points
        weight_points = np.linspace(weight_range[0], weight_range[1], num_points)
        
        # Store results
        lap_times = []
        avg_speeds = []
        weights = []
        
        # Original weight
        original_weight = self.vehicle.mass
        
        # Run simulations at each weight point
        for weight in weight_points:
            # Set vehicle weight
            self.vehicle.mass = weight
            
            # Reset simulation state
            self.lap_time_simulator.speed_profile = None
            
            # Calculate speed profile
            self.lap_time_simulator.calculate_speed_profile()
            
            # Run lap simulation
            lap_results = self.lap_time_simulator.simulate_lap(include_thermal=include_thermal)
            
            # Analyze performance
            metrics = self.lap_time_simulator.analyze_lap_performance(lap_results)
            
            # Store results
            weights.append(weight)
            lap_times.append(metrics['lap_time'])
            avg_speeds.append(metrics['avg_speed_kph'])
            
            logger.info(f"Weight: {weight:.1f} kg, Lap time: {metrics['lap_time']:.3f} s, "
                       f"Avg speed: {metrics['avg_speed_kph']:.1f} kph")
        
        # Restore original weight
        self.vehicle.mass = original_weight
        
        # Calculate sensitivity coefficients
        lap_time_slope = self._calculate_sensitivity_coefficient(weights, lap_times)
        avg_speed_slope = self._calculate_sensitivity_coefficient(weights, avg_speeds)
        
        # Store results
        sensitivity_results = {
            'weights': weights,
            'lap_times': lap_times,
            'avg_speeds': avg_speeds,
            'sensitivity_lap_time': lap_time_slope,
            'sensitivity_avg_speed': avg_speed_slope,
            'seconds_per_10kg_lap': lap_time_slope * 10,
            'kph_per_10kg_avg_speed': avg_speed_slope * 10
        }
        
        # Store in class variable
        self.lap_time_sensitivity = sensitivity_results
        
        logger.info("Lap time sensitivity analysis completed")
        logger.info(f"Lap time sensitivity: {lap_time_slope:.4f} seconds per kg "
                   f"({lap_time_slope * 10:.4f} seconds per 10 kg)")
        logger.info(f"Average speed sensitivity: {avg_speed_slope:.4f} kph per kg "
                   f"({avg_speed_slope * 10:.4f} kph per 10 kg)")
        
        return sensitivity_results
    
    def analyze_lap_time_sensitivity(self, 
                              track_file: str,
                              weight_range: Tuple[float, float],
                              num_points: int = 5,
                              include_thermal: bool = True) -> Dict:
        """
        Analyze sensitivity of lap time performance to weight changes.
        
        Args:
            track_file: Path to track file
            weight_range: Tuple of (min_weight, max_weight) in kg
            num_points: Number of weight points to analyze
            include_thermal: Whether to include thermal effects
            
        Returns:
            Dictionary with analysis results
        """
        logger.info(f"Analyzing lap time sensitivity from {weight_range[0]:.1f} to {weight_range[1]:.1f} kg")
        
        # Create lap time simulator if not already created
        if self.lap_time_simulator is None:
            self.lap_time_simulator = create_lap_time_simulator(self.vehicle, track_file)
        
        # Generate weight points
        weight_points = np.linspace(weight_range[0], weight_range[1], num_points)
        
        # Store results
        lap_times = []
        avg_speeds = []
        weights = []
        
        # Original weight
        original_weight = self.vehicle.mass
        
        # Run simulations at each weight point
        for weight in weight_points:
            # Set vehicle weight
            self.vehicle.mass = weight
            
            # Reset simulation state
            self.lap_time_simulator.speed_profile = None
            
            # Calculate speed profile
            self.lap_time_simulator.calculate_speed_profile()
            
            # Run lap simulation
            lap_results = self.lap_time_simulator.simulate_lap(include_thermal=include_thermal)
            
            # Analyze performance
            metrics = self.lap_time_simulator.analyze_lap_performance(lap_results)
            
            # Store results
            weights.append(weight)
            lap_times.append(metrics['lap_time'])
            avg_speeds.append(metrics['avg_speed_kph'])
            
            logger.info(f"Weight: {weight:.1f} kg, Lap time: {metrics['lap_time']:.3f} s, "
                    f"Avg speed: {metrics['avg_speed_kph']:.1f} kph")
        
        # Restore original weight
        self.vehicle.mass = original_weight
        
        # Calculate sensitivity coefficients
        lap_time_slope = self._calculate_sensitivity_coefficient(weights, lap_times)
        avg_speed_slope = self._calculate_sensitivity_coefficient(weights, avg_speeds)
        
        # Calculate seconds per 10kg
        seconds_per_10kg_lap = lap_time_slope * 10
        kph_per_10kg_avg_speed = avg_speed_slope * 10
        
        # Calculate percentage improvements per 10kg
        percent_improvement_per_10kg_lap = 0
        percent_improvement_per_10kg_avg_speed = 0
        
        if lap_times and lap_times[0] > 0:
            percent_improvement_per_10kg_lap = (seconds_per_10kg_lap / lap_times[0]) * 100
            
        if avg_speeds and avg_speeds[0] > 0:
            percent_improvement_per_10kg_avg_speed = (kph_per_10kg_avg_speed / avg_speeds[0]) * 100
        
        # Store results
        sensitivity_results = {
            'weights': weights,
            'lap_times': lap_times,
            'avg_speeds': avg_speeds,
            'sensitivity_lap_time': lap_time_slope,
            'sensitivity_avg_speed': avg_speed_slope,
            'seconds_per_10kg_lap': seconds_per_10kg_lap,
            'kph_per_10kg_avg_speed': kph_per_10kg_avg_speed,
            'percent_improvement_per_10kg_lap': percent_improvement_per_10kg_lap,
            'percent_improvement_per_10kg_avg_speed': percent_improvement_per_10kg_avg_speed
        }
        
        # Store in class variable
        self.lap_time_sensitivity = sensitivity_results
        
        logger.info("Lap time sensitivity analysis completed")
        logger.info(f"Lap time sensitivity: {lap_time_slope:.4f} seconds per kg "
                f"({lap_time_slope * 10:.4f} seconds per 10 kg)")
        logger.info(f"Average speed sensitivity: {avg_speed_slope:.4f} kph per kg "
                f"({avg_speed_slope * 10:.4f} kph per 10 kg)")
        
        return sensitivity_results
    
    def calculate_weight_reduction_targets(self, 
                                        performance_target: float,
                                        sensitivity: Optional[float] = None,
                                        performance_type: str = 'acceleration') -> Dict:
        """
        Calculate required weight reduction to reach a performance target.
        
        Args:
            performance_target: Target performance value (e.g., 0-60 mph time in seconds)
            sensitivity: Sensitivity coefficient (seconds per kg), if None will use calculated value
            performance_type: Type of performance ('acceleration' or 'lap_time')
            
        Returns:
            Dictionary with weight reduction targets
        """
        # Get current performance and sensitivity
        if performance_type == 'acceleration':
            if not self.acceleration_sensitivity:
                logger.error("Acceleration sensitivity not analyzed. Call analyze_acceleration_sensitivity first.")
                return {}
                
            current_performance = self.acceleration_sensitivity['time_to_60mph'][0]
            if sensitivity is None:
                sensitivity = self.acceleration_sensitivity['sensitivity_60mph']
            
            performance_name = "0-60 mph time"
            
        elif performance_type == 'lap_time':
            if not self.lap_time_sensitivity:
                logger.error("Lap time sensitivity not analyzed. Call analyze_lap_time_sensitivity first.")
                return {}
                
            current_performance = self.lap_time_sensitivity['lap_times'][0]
            if sensitivity is None:
                sensitivity = self.lap_time_sensitivity['sensitivity_lap_time']
            
            performance_name = "Lap time"
        
        # Calculate required improvement
        required_improvement = current_performance - performance_target
        
        # Calculate required weight reduction
        if sensitivity > 0:
            required_weight_reduction = required_improvement / sensitivity
        else:
            required_weight_reduction = float('inf')
        
        # Current weight
        current_weight = self.base_weight
        
        # Target weight
        target_weight = current_weight - required_weight_reduction
        
        # Check if target is achievable
        is_achievable = target_weight > current_weight * 0.5  # Assuming 50% weight reduction is maximum possible
        
        # Create result
        result = {
            'current_performance': current_performance,
            'target_performance': performance_target,
            'required_improvement': required_improvement,
            'sensitivity': sensitivity,
            'current_weight': current_weight,
            'required_weight_reduction': required_weight_reduction,
            'target_weight': target_weight,
            'is_achievable': is_achievable,
            'performance_type': performance_type,
            'performance_name': performance_name
        }
        
        logger.info(f"Weight reduction target calculation for {performance_name}:")
        logger.info(f"Current: {current_performance:.3f} s, Target: {performance_target:.3f} s, "
                   f"Required improvement: {required_improvement:.3f} s")
        logger.info(f"Required weight reduction: {required_weight_reduction:.1f} kg "
                   f"({required_weight_reduction / current_weight * 100:.1f}%)")
        logger.info(f"Target weight: {target_weight:.1f} kg "
                   f"(achievable: {'Yes' if is_achievable else 'No'})")
        
        return result
    
    def _calculate_sensitivity_coefficient(self, x_data: List[float], y_data: List[float]) -> float:
        """
        Calculate sensitivity coefficient from data using linear regression.
        
        Args:
            x_data: Independent variable data (e.g., weights)
            y_data: Dependent variable data (e.g., times)
            
        Returns:
            Sensitivity coefficient (slope of linear regression)
        """
        if len(x_data) < 2 or len(y_data) < 2:
            logger.warning("Insufficient data for sensitivity calculation")
            return 0.0
            
        # Use numpy's polyfit for linear regression
        coefficients = np.polyfit(x_data, y_data, 1)
        
        # Return the slope (sensitivity coefficient)
        return coefficients[0]
    
    def plot_weight_sensitivity_curves(self, save_path: Optional[str] = None):
        """
        Plot weight sensitivity curves for acceleration and lap time.
        
        Args:
            save_path: Optional path to save the plot
        """
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Plot acceleration sensitivity if available
        if self.acceleration_sensitivity:
            plt.subplot(2, 2, 1)
            weights = self.acceleration_sensitivity['weights']
            times_60mph = self.acceleration_sensitivity['time_to_60mph']
            times_100kph = self.acceleration_sensitivity['time_to_100kph']
            
            # Filter out None values for plotting
            valid_60mph_data = [(w, t) for w, t in zip(weights, times_60mph) if t is not None]
            valid_100kph_data = [(w, t) for w, t in zip(weights, times_100kph) if t is not None]
            
            if valid_60mph_data:
                valid_weights_60mph, valid_times_60mph = zip(*valid_60mph_data)
                plt.plot(valid_weights_60mph, valid_times_60mph, 'b-o', label='0-60 mph')
            
            if valid_100kph_data:
                valid_weights_100kph, valid_times_100kph = zip(*valid_100kph_data)
                plt.plot(valid_weights_100kph, valid_times_100kph, 'r-o', label='0-100 kph')
            
            plt.xlabel('Weight (kg)')
            plt.ylabel('Time (s)')
            plt.title('Acceleration Performance vs. Weight')
            plt.grid(True)
            plt.legend()
            
            # Add trend line and equation if we have valid data
            if valid_60mph_data:
                fit_60mph = np.polyfit(valid_weights_60mph, valid_times_60mph, 1)
                fit_line_60mph = np.poly1d(fit_60mph)
                plt.plot(valid_weights_60mph, fit_line_60mph(valid_weights_60mph), 'b--')
                
                equation_60mph = f"y = {fit_60mph[0]:.4f}x + {fit_60mph[1]:.2f}"
                plt.text(min(valid_weights_60mph), max(valid_times_60mph), equation_60mph, color='b')
            
            # Plot 75m time
            plt.subplot(2, 2, 2)
            times_75m = self.acceleration_sensitivity['time_75m']
            
            # Filter out None values for plotting
            valid_75m_data = [(w, t) for w, t in zip(weights, times_75m) if t is not None]
            
            if valid_75m_data:
                valid_weights_75m, valid_times_75m = zip(*valid_75m_data)
                plt.plot(valid_weights_75m, valid_times_75m, 'g-o', label='75m Time')
                
                plt.xlabel('Weight (kg)')
                plt.ylabel('Time (s)')
                plt.title('75m Acceleration Time vs. Weight')
                plt.grid(True)
                
                # Add trend line and equation
                fit_75m = np.polyfit(valid_weights_75m, valid_times_75m, 1)
                fit_line_75m = np.poly1d(fit_75m)
                plt.plot(valid_weights_75m, fit_line_75m(valid_weights_75m), 'g--')
                
                equation_75m = f"y = {fit_75m[0]:.4f}x + {fit_75m[1]:.2f}"
                plt.text(min(valid_weights_75m), max(valid_times_75m), equation_75m, color='g')
        
        # Plot lap time sensitivity if available
        if self.lap_time_sensitivity:
            plt.subplot(2, 2, 3)
            weights = self.lap_time_sensitivity['weights']
            lap_times = self.lap_time_sensitivity['lap_times']
            
            # Filter out None values for plotting (if any)
            valid_lap_data = [(w, t) for w, t in zip(weights, lap_times) if t is not None]
            
            if valid_lap_data:
                valid_weights_lap, valid_times_lap = zip(*valid_lap_data)
                plt.plot(valid_weights_lap, valid_times_lap, 'm-o', label='Lap Time')
                
                plt.xlabel('Weight (kg)')
                plt.ylabel('Time (s)')
                plt.title('Lap Time vs. Weight')
                plt.grid(True)
                
                # Add trend line and equation
                fit_lap = np.polyfit(valid_weights_lap, valid_times_lap, 1)
                fit_line_lap = np.poly1d(fit_lap)
                plt.plot(valid_weights_lap, fit_line_lap(valid_weights_lap), 'm--')
                
                equation_lap = f"y = {fit_lap[0]:.4f}x + {fit_lap[1]:.2f}"
                plt.text(min(valid_weights_lap), max(valid_times_lap), equation_lap, color='m')
            
            # Plot average speed
            plt.subplot(2, 2, 4)
            avg_speeds = self.lap_time_sensitivity['avg_speeds']
            
            # Filter out None values for plotting (if any)
            valid_speed_data = [(w, s) for w, s in zip(weights, avg_speeds) if s is not None]
            
            if valid_speed_data:
                valid_weights_speed, valid_speeds = zip(*valid_speed_data)
                plt.plot(valid_weights_speed, valid_speeds, 'c-o', label='Average Speed')
                
                plt.xlabel('Weight (kg)')
                plt.ylabel('Speed (km/h)')
                plt.title('Average Speed vs. Weight')
                plt.grid(True)
                
                # Add trend line and equation
                fit_speed = np.polyfit(valid_weights_speed, valid_speeds, 1)
                fit_line_speed = np.poly1d(fit_speed)
                plt.plot(valid_weights_speed, fit_line_speed(valid_weights_speed), 'c--')
                
                equation_speed = f"y = {fit_speed[0]:.4f}x + {fit_speed[1]:.2f}"
                plt.text(min(valid_weights_speed), min(valid_speeds), equation_speed, color='c')
        
        plt.tight_layout()
        
        # Save plot if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Weight sensitivity curves saved to {save_path}")
        
        plt.show()
    
    def generate_weight_sensitivity_report(self, save_dir: Optional[str] = None) -> Dict:
        """
        Generate a comprehensive weight sensitivity analysis report.
        
        Args:
            save_dir: Optional directory to save plots and data
            
        Returns:
            Dictionary with report data
        """
        # Check if sensitivity analyses have been performed
        if not self.acceleration_sensitivity and not self.lap_time_sensitivity:
            logger.error("No sensitivity analyses have been performed. Call analyze_acceleration_sensitivity and/or analyze_lap_time_sensitivity first.")
            return {}
        
        # Create save directory if provided
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        # Create summary data
        summary = {
            'vehicle': {
                'base_weight': self.base_weight,
                'power': self.vehicle.engine.max_power,
                'power_to_weight': self.vehicle.engine.max_power / self.base_weight,
            }
        }
        
        # Add acceleration sensitivity data if available
        if self.acceleration_sensitivity:
            summary['acceleration'] = {
                'sensitivity_60mph': self.acceleration_sensitivity['sensitivity_60mph'],
                'sensitivity_100kph': self.acceleration_sensitivity['sensitivity_100kph'],
                'sensitivity_75m': self.acceleration_sensitivity['sensitivity_75m'],
                'seconds_per_10kg_60mph': self.acceleration_sensitivity['seconds_per_10kg_60mph'],
                'seconds_per_10kg_100kph': self.acceleration_sensitivity['seconds_per_10kg_100kph'],
                'seconds_per_10kg_75m': self.acceleration_sensitivity['seconds_per_10kg_75m'],
                'base_time_60mph': self.acceleration_sensitivity['time_to_60mph'][0] if self.acceleration_sensitivity['time_to_60mph'] else 0,
                'base_time_100kph': self.acceleration_sensitivity['time_to_100kph'][0] if self.acceleration_sensitivity['time_to_100kph'] else 0,
                'base_time_75m': self.acceleration_sensitivity['time_75m'][0] if self.acceleration_sensitivity['time_75m'] else 0
            }
            
            # Only add percent improvements if they're in the dictionary
            if 'percent_improvement_per_10kg_60mph' in self.acceleration_sensitivity:
                summary['acceleration']['percent_improvement_per_10kg_60mph'] = self.acceleration_sensitivity['percent_improvement_per_10kg_60mph']
            
            if 'percent_improvement_per_10kg_75m' in self.acceleration_sensitivity:
                summary['acceleration']['percent_improvement_per_10kg_75m'] = self.acceleration_sensitivity['percent_improvement_per_10kg_75m']
                
        # Add lap time sensitivity data if available
        if self.lap_time_sensitivity:
            summary['lap_time'] = {
                'sensitivity_lap_time': self.lap_time_sensitivity['sensitivity_lap_time'],
                'sensitivity_avg_speed': self.lap_time_sensitivity['sensitivity_avg_speed'],
                'seconds_per_10kg_lap': self.lap_time_sensitivity['seconds_per_10kg_lap'],
                'kph_per_10kg_avg_speed': self.lap_time_sensitivity['kph_per_10kg_avg_speed'],
                'base_lap_time': self.lap_time_sensitivity['lap_times'][0] if self.lap_time_sensitivity['lap_times'] else 0,
                'base_avg_speed': self.lap_time_sensitivity['avg_speeds'][0] if self.lap_time_sensitivity['avg_speeds'] else 0
            }
            
            # Only add percent improvements if they're in the dictionary
            if 'percent_improvement_per_10kg_lap' in self.lap_time_sensitivity:
                summary['lap_time']['percent_improvement_per_10kg_lap'] = self.lap_time_sensitivity['percent_improvement_per_10kg_lap']
            else:
                # Calculate it if not present but we have the necessary data
                if self.lap_time_sensitivity['lap_times'] and self.lap_time_sensitivity['lap_times'][0] > 0:
                    percent_imp = (self.lap_time_sensitivity['seconds_per_10kg_lap'] / self.lap_time_sensitivity['lap_times'][0]) * 100
                    summary['lap_time']['percent_improvement_per_10kg_lap'] = percent_imp
                
        # Create plots if save directory provided
        if save_dir:
            # Plot weight sensitivity curves
            self.plot_weight_sensitivity_curves(
                save_path=os.path.join(save_dir, "weight_sensitivity_curves.png")
            )
            
            # Create summary report as text
            summary_path = os.path.join(save_dir, "weight_sensitivity_summary.txt")
            with open(summary_path, 'w') as f:
                f.write("Formula Student Weight Sensitivity Analysis\n")
                f.write("==========================================\n\n")
                
                f.write(f"Vehicle Base Weight: {self.base_weight:.1f} kg\n")
                f.write(f"Engine Power: {self.vehicle.engine.max_power:.1f} hp\n")
                f.write(f"Power-to-Weight Ratio: {self.vehicle.engine.max_power/self.base_weight:.3f} hp/kg\n\n")
                
                if self.acceleration_sensitivity:
                    f.write("Acceleration Performance Sensitivity\n")
                    f.write("----------------------------------\n")
                    f.write(f"0-60 mph: {self.acceleration_sensitivity['sensitivity_60mph']:.4f} seconds per kg "
                        f"({self.acceleration_sensitivity['seconds_per_10kg_60mph']:.4f} seconds per 10 kg)\n")
                    f.write(f"0-100 kph: {self.acceleration_sensitivity['sensitivity_100kph']:.4f} seconds per kg "
                        f"({self.acceleration_sensitivity['seconds_per_10kg_100kph']:.4f} seconds per 10 kg)\n")
                    f.write(f"75m: {self.acceleration_sensitivity['sensitivity_75m']:.4f} seconds per kg "
                        f"({self.acceleration_sensitivity['seconds_per_10kg_75m']:.4f} seconds per 10 kg)\n\n")
                
                if self.lap_time_sensitivity:
                    f.write("Lap Time Performance Sensitivity\n")
                    f.write("------------------------------\n")
                    f.write(f"Lap Time: {self.lap_time_sensitivity['sensitivity_lap_time']:.4f} seconds per kg "
                        f"({self.lap_time_sensitivity['seconds_per_10kg_lap']:.4f} seconds per 10 kg)\n")
                    f.write(f"Average Speed: {self.lap_time_sensitivity['sensitivity_avg_speed']:.4f} kph per kg "
                        f"({self.lap_time_sensitivity['kph_per_10kg_avg_speed']:.4f} kph per 10 kg)\n\n")
                
                f.write("Performance Improvement per 1% Weight Reduction\n")
                f.write("-------------------------------------------\n")
                
                if self.acceleration_sensitivity:
                    # Safely calculate these values
                    if 'percent_improvement_per_10kg_60mph' in self.acceleration_sensitivity:
                        percent_per_percent_60mph = self.acceleration_sensitivity['percent_improvement_per_10kg_60mph'] / 10 * self.base_weight / 100
                        f.write(f"0-60 mph: {percent_per_percent_60mph:.4f}% improvement per 1% weight reduction\n")
                    else:
                        f.write("0-60 mph: Data not available\n")
                    
                    if 'percent_improvement_per_10kg_75m' in self.acceleration_sensitivity:
                        percent_per_percent_75m = self.acceleration_sensitivity['percent_improvement_per_10kg_75m'] / 10 * self.base_weight / 100
                        f.write(f"75m: {percent_per_percent_75m:.4f}% improvement per 1% weight reduction\n")
                    else:
                        f.write("75m: Data not available\n")
                
                if self.lap_time_sensitivity:
                    # Safely calculate lap time percent per percent improvement
                    if 'percent_improvement_per_10kg_lap' in self.lap_time_sensitivity:
                        percent_per_percent_lap = self.lap_time_sensitivity['percent_improvement_per_10kg_lap'] / 10 * self.base_weight / 100
                        f.write(f"Lap Time: {percent_per_percent_lap:.4f}% improvement per 1% weight reduction\n\n")
                    elif 'lap_times' in self.lap_time_sensitivity and self.lap_time_sensitivity['lap_times']:
                        # Calculate on the fly if needed
                        lap_time_value = self.lap_time_sensitivity['lap_times'][0]
                        if lap_time_value > 0:
                            seconds_per_10kg = self.lap_time_sensitivity['seconds_per_10kg_lap']
                            percent_imp = (seconds_per_10kg / lap_time_value) * 100
                            percent_per_percent_lap = percent_imp / 10 * self.base_weight / 100
                            f.write(f"Lap Time: {percent_per_percent_lap:.4f}% improvement per 1% weight reduction\n\n")
                        else:
                            f.write("Lap Time: Data not available\n\n")
                    else:
                        f.write("Lap Time: Data not available\n\n")
            
            # Save detailed data to CSV files
            if self.acceleration_sensitivity:
                accel_df = pd.DataFrame({
                    'Weight (kg)': self.acceleration_sensitivity['weights'],
                    '0-60 mph (s)': self.acceleration_sensitivity['time_to_60mph'],
                    '0-100 kph (s)': self.acceleration_sensitivity['time_to_100kph'],
                    '75m Time (s)': self.acceleration_sensitivity['time_75m']
                })
                accel_df.to_csv(os.path.join(save_dir, "acceleration_weight_sensitivity.csv"), index=False)
            
            if self.lap_time_sensitivity:
                lap_df = pd.DataFrame({
                    'Weight (kg)': self.lap_time_sensitivity['weights'],
                    'Lap Time (s)': self.lap_time_sensitivity['lap_times'],
                    'Average Speed (kph)': self.lap_time_sensitivity['avg_speeds']
                })
                lap_df.to_csv(os.path.join(save_dir, "lap_time_weight_sensitivity.csv"), index=False)
        
        logger.info("Weight sensitivity report generated")
        
        return summary