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
            
            logger.info(f"Weight: {weight:.1f} kg, 0-60 mph: {results['time_to_60mph']:.3f} s, "
                       f"0-100 kph: {results['time_to_100kph']:.3f} s, 75m: {results['finish_time']:.3f} s")
        
        # Restore original weight
        self.vehicle.mass = original_weight
        
        # Calculate sensitivity coefficients
        # We'll use linear regression to find the slope (seconds per kg)
        time_to_60mph_slope = self._calculate_sensitivity_coefficient(weights, time_to_60mph)
        time_to_100kph_slope = self._calculate_sensitivity_coefficient(weights, time_to_100kph)
        time_75m_slope = self._calculate_sensitivity_coefficient(weights, time_75m)
        
        # Store results
        sensitivity_results = {
            'weights': weights,
            'time_to_60mph': time_to_60mph,
            'time_to_100kph': time_to_100kph,
            'time_75m': time_75m,
            'sensitivity_60mph': time_to_60mph_slope,
            'sensitivity_100kph': time_to_100kph_slope,
            'sensitivity_75m': time_75m_slope,
            'seconds_per_10kg_60mph': time_to_60mph_slope * 10,
            'seconds_per_10kg_100kph': time_to_100kph_slope * 10,
            'seconds_per_10kg_75m': time_75m_slope * 10
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
    
    def analyze_weight_distribution_sensitivity(self, 
                                             distribution_range: Tuple[float, float] = (0.4, 0.6),
                                             num_points: int = 5,
                                             test_type: str = 'acceleration') -> Dict:
        """
        Analyze sensitivity to weight distribution changes (front/rear balance).
        
        Args:
            distribution_range: Tuple of (min_front_weight_fraction, max_front_weight_fraction)
            num_points: Number of distribution points to analyze
            test_type: Type of test ('acceleration' or 'lap_time')
            
        Returns:
            Dictionary with analysis results
        """
        logger.info(f"Analyzing weight distribution sensitivity from {distribution_range[0] * 100:.1f}% to "
                   f"{distribution_range[1] * 100:.1f}% front weight")
        
        # Generate distribution points
        distribution_points = np.linspace(distribution_range[0], distribution_range[1], num_points)
        
        # Store results
        performance_metrics = []
        distributions = []
        
        # Original distribution
        original_distribution = getattr(self.vehicle, 'weight_distribution', 0.5)
        
        # Run simulations at each distribution point
        for distribution in distribution_points:
            # Set vehicle weight distribution
            self.vehicle.weight_distribution = distribution
            
            # Run appropriate test
            if test_type == 'acceleration':
                # Run acceleration simulation
                results = self.acceleration_simulator.simulate_acceleration(
                    use_launch_control=True,
                    optimized_shifts=True
                )
                
                # Store results
                metric = results['time_to_60mph']
                metric_name = "0-60 mph time"
                
            elif test_type == 'lap_time':
                # Ensure lap time simulator is created
                if self.lap_time_simulator is None:
                    logger.error("Lap time simulator not initialized. Call analyze_lap_time_sensitivity first.")
                    return {}
                
                # Reset simulation state
                self.lap_time_simulator.speed_profile = None
                
                # Calculate speed profile
                self.lap_time_simulator.calculate_speed_profile()
                
                # Run lap simulation
                lap_results = self.lap_time_simulator.simulate_lap()
                
                # Analyze performance
                metrics = self.lap_time_simulator.analyze_lap_performance(lap_results)
                
                # Store results
                metric = metrics['lap_time']
                metric_name = "Lap time"
            
            distributions.append(distribution)
            performance_metrics.append(metric)
            
            logger.info(f"Front weight distribution: {distribution * 100:.1f}%, {metric_name}: {metric:.3f} s")
        
        # Restore original distribution
        self.vehicle.weight_distribution = original_distribution
        
        # Calculate sensitivity coefficient
        sensitivity = self._calculate_sensitivity_coefficient(distributions, performance_metrics)
        
        # Store results
        sensitivity_results = {
            'distributions': distributions,
            'performance_metrics': performance_metrics,
            'metric_name': metric_name,
            'sensitivity': sensitivity,
            'optimal_distribution': distributions[np.argmin(performance_metrics)]
        }
        
        logger.info(f"{metric_name} sensitivity to weight distribution: {sensitivity * 0.01:.4f} seconds per 1% shift")
        logger.info(f"Optimal front weight distribution: {sensitivity_results['optimal_distribution'] * 100:.1f}%")
        
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
            
            plt.plot(weights, times_60mph, 'b-o', label='0-60 mph')
            plt.plot(weights, times_100kph, 'r-o', label='0-100 kph')
            
            plt.xlabel('Weight (kg)')
            plt.ylabel('Time (s)')
            plt.title('Acceleration Performance vs. Weight')
            plt.grid(True)
            plt.legend()
            
            # Add trend line and equation
            fit_60mph = np.polyfit(weights, times_60mph, 1)
            fit_line_60mph = np.poly1d(fit_60mph)
            plt.plot(weights, fit_line_60mph(weights), 'b--')
            
            equation_60mph = f"y = {fit_60mph[0]:.4f}x + {fit_60mph[1]:.2f}"
            plt.text(weights[0], max(times_60mph), equation_60mph, color='b')
            
            # Plot 75m time
            plt.subplot(2, 2, 2)
            times_75m = self.acceleration_sensitivity['time_75m']
            
            plt.plot(weights, times_75m, 'g-o', label='75m Time')
            
            plt.xlabel('Weight (kg)')
            plt.ylabel('Time (s)')
            plt.title('75m Acceleration Time vs. Weight')
            plt.grid(True)
            
            # Add trend line and equation
            fit_75m = np.polyfit(weights, times_75m, 1)
            fit_line_75m = np.poly1d(fit_75m)
            plt.plot(weights, fit_line_75m(weights), 'g--')
            
            equation_75m = f"y = {fit_75m[0]:.4f}x + {fit_75m[1]:.2f}"
            plt.text(weights[0], max(times_75m), equation_75m, color='g')
        
        # Plot lap time sensitivity if available
        if self.lap_time_sensitivity:
            plt.subplot(2, 2, 3)
            weights = self.lap_time_sensitivity['weights']
            lap_times = self.lap_time_sensitivity['lap_times']
            
            plt.plot(weights, lap_times, 'm-o', label='Lap Time')
            
            plt.xlabel('Weight (kg)')
            plt.ylabel('Time (s)')
            plt.title('Lap Time vs. Weight')
            plt.grid(True)
            
            # Add trend line and equation
            fit_lap = np.polyfit(weights, lap_times, 1)
            fit_line_lap = np.poly1d(fit_lap)
            plt.plot(weights, fit_line_lap(weights), 'm--')
            
            equation_lap = f"y = {fit_lap[0]:.4f}x + {fit_lap[1]:.2f}"
            plt.text(weights[0], max(lap_times), equation_lap, color='m')
            
            # Plot average speed
            plt.subplot(2, 2, 4)
            avg_speeds = self.lap_time_sensitivity['avg_speeds']
            
            plt.plot(weights, avg_speeds, 'c-o', label='Average Speed')
            
            plt.xlabel('Weight (kg)')
            plt.ylabel('Speed (km/h)')
            plt.title('Average Speed vs. Weight')
            plt.grid(True)
            
            # Add trend line and equation
            fit_speed = np.polyfit(weights, avg_speeds, 1)
            fit_line_speed = np.poly1d(fit_speed)
            plt.plot(weights, fit_line_speed(weights), 'c--')
            
            equation_speed = f"y = {fit_speed[0]:.4f}x + {fit_speed[1]:.2f}"
            plt.text(weights[0], min(avg_speeds), equation_speed, color='c')
        
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
                'percent_improvement_per_10kg_60mph': (self.acceleration_sensitivity['seconds_per_10kg_60mph'] / 
                                                      self.acceleration_sensitivity['time_to_60mph'][0]) * 100,
                'percent_improvement_per_10kg_75m': (self.acceleration_sensitivity['seconds_per_10kg_75m'] / 
                                                    self.acceleration_sensitivity['time_75m'][0]) * 100,
                'base_time_60mph': self.acceleration_sensitivity['time_to_60mph'][0],
                'base_time_100kph': self.acceleration_sensitivity['time_to_100kph'][0],
                'base_time_75m': self.acceleration_sensitivity['time_75m'][0]
            }
            
        # Add lap time sensitivity data if available
        if self.lap_time_sensitivity:
            summary['lap_time'] = {
                'sensitivity_lap_time': self.lap_time_sensitivity['sensitivity_lap_time'],
                'sensitivity_avg_speed': self.lap_time_sensitivity['sensitivity_avg_speed'],
                'seconds_per_10kg_lap': self.lap_time_sensitivity['seconds_per_10kg_lap'],
                'kph_per_10kg_avg_speed': self.lap_time_sensitivity['kph_per_10kg_avg_speed'],
                'percent_improvement_per_10kg_lap': (self.lap_time_sensitivity['seconds_per_10kg_lap'] / 
                                                    self.lap_time_sensitivity['lap_times'][0]) * 100,
                'base_lap_time': self.lap_time_sensitivity['lap_times'][0],
                'base_avg_speed': self.lap_time_sensitivity['avg_speeds'][0]
            }
            
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
                    percent_per_percent_60mph = self.acceleration_sensitivity['percent_improvement_per_10kg_60mph'] / 10 * self.base_weight / 100
                    percent_per_percent_75m = self.acceleration_sensitivity['percent_improvement_per_10kg_75m'] / 10 * self.base_weight / 100
                    
                    f.write(f"0-60 mph: {percent_per_percent_60mph:.4f}% improvement per 1% weight reduction\n")
                    f.write(f"75m: {percent_per_percent_75m:.4f}% improvement per 1% weight reduction\n")
                
                if self.lap_time_sensitivity:
                    percent_per_percent_lap = self.lap_time_sensitivity['percent_improvement_per_10kg_lap'] / 10 * self.base_weight / 100
                    
                    f.write(f"Lap Time: {percent_per_percent_lap:.4f}% improvement per 1% weight reduction\n\n")
            
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