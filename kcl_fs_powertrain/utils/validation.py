"""
Validation utilities for Formula Student powertrain simulation.

This module provides functions for validating simulation results against theoretical
models, expected performance ranges, and real-world data. It helps ensure the
simulation produces realistic and accurate results.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import logging
import os
import json
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Import local modules
from ..utils.constants import (
    GRAVITY, FS_ACCELERATION_LENGTH, FS_SKIDPAD_RADIUS,
    MS_TO_KMH, MS_TO_MPH, KW_TO_HP
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("Validation")


# Expected performance ranges for Formula Student vehicles
FS_PERFORMANCE_RANGES = {
    # Acceleration metrics
    'acceleration_time': (3.5, 6.0),    # 75m acceleration time (s)
    'time_to_60mph': (3.0, 5.5),        # 0-60 mph time (s)
    'time_to_100kph': (2.8, 5.2),       # 0-100 km/h time (s)
    
    # Speed metrics
    'top_speed': (90.0, 150.0),         # Top speed (km/h)
    'speed_at_end_of_accel': (80.0, 130.0),  # Speed at end of acceleration event (km/h)
    
    # Lateral acceleration 
    'lateral_acceleration': (1.2, 2.2),  # Maximum lateral acceleration (g)
    'skidpad_time': (4.7, 5.7),         # Skidpad lap time (s)
    
    # Thermal metrics
    'max_engine_temp': (85.0, 130.0),   # Maximum engine temperature (°C)
    'max_coolant_temp': (80.0, 110.0),  # Maximum coolant temperature (°C)
    'max_oil_temp': (90.0, 140.0),      # Maximum oil temperature (°C)
    
    # Power metrics
    'power_to_weight': (0.15, 0.3),     # Power-to-weight ratio (kW/kg)
    'specific_power': (120.0, 190.0),   # Power per liter of displacement (HP/L)
    
    # Weight metrics
    'vehicle_mass': (180.0, 300.0),     # Vehicle mass (kg)
    'weight_distribution': (0.4, 0.5),   # Front weight distribution (fraction)
    
    # Efficiency metrics
    'fuel_consumption_rate': (3.0, 8.0)  # Fuel consumption rate (L/100km)
}

# Validation error thresholds
VALIDATION_THRESHOLDS = {
    'critical_error': 0.25,     # Relative error threshold for critical validation issues
    'warning': 0.15,            # Relative error threshold for validation warnings
    'acceptable': 0.05,         # Relative error threshold for acceptable validation
    'good': 0.01                # Relative error threshold for good validation
}


def validate_in_range(value: float, metric_name: str, 
                    custom_range: Optional[Tuple[float, float]] = None) -> Dict:
    """
    Validate if a value is within expected range for a metric.
    
    Args:
        value: Value to validate
        metric_name: Name of the metric to check
        custom_range: Optional custom range override
        
    Returns:
        Dictionary with validation results
    """
    if custom_range:
        expected_range = custom_range
    elif metric_name in FS_PERFORMANCE_RANGES:
        expected_range = FS_PERFORMANCE_RANGES[metric_name]
    else:
        logger.warning(f"No expected range found for metric: {metric_name}")
        return {
            'status': 'unknown',
            'metric': metric_name,
            'value': value,
            'expected_range': None,
            'message': f"No expected range defined for {metric_name}"
        }
    
    min_value, max_value = expected_range
    
    if value < min_value:
        # Value below minimum expected value
        relative_error = (min_value - value) / min_value
        status = _determine_validation_status(relative_error)
        
        return {
            'status': status,
            'metric': metric_name,
            'value': value,
            'expected_range': expected_range,
            'relative_error': relative_error,
            'message': f"{metric_name} ({value:.3f}) is below expected minimum ({min_value:.3f})"
        }
    elif value > max_value:
        # Value above maximum expected value
        relative_error = (value - max_value) / max_value
        status = _determine_validation_status(relative_error)
        
        return {
            'status': status,
            'metric': metric_name,
            'value': value,
            'expected_range': expected_range,
            'relative_error': relative_error,
            'message': f"{metric_name} ({value:.3f}) is above expected maximum ({max_value:.3f})"
        }
    else:
        # Value within expected range
        return {
            'status': 'valid',
            'metric': metric_name,
            'value': value,
            'expected_range': expected_range,
            'relative_error': 0.0,
            'message': f"{metric_name} ({value:.3f}) is within expected range ({min_value:.3f} - {max_value:.3f})"
        }


def _determine_validation_status(relative_error: float) -> str:
    """
    Determine validation status based on relative error.
    
    Args:
        relative_error: Calculated relative error
        
    Returns:
        Validation status string
    """
    if relative_error >= VALIDATION_THRESHOLDS['critical_error']:
        return 'critical_error'
    elif relative_error >= VALIDATION_THRESHOLDS['warning']:
        return 'warning'
    elif relative_error >= VALIDATION_THRESHOLDS['acceptable']:
        return 'acceptable'
    else:
        return 'good'


def validate_theoretical_model(measured_values: np.ndarray, 
                             theoretical_model: Callable[[np.ndarray], np.ndarray],
                             input_values: np.ndarray,
                             metric_name: str) -> Dict:
    """
    Validate measured values against a theoretical model.
    
    Args:
        measured_values: Array of measured values
        theoretical_model: Function that takes input_values and returns theoretical predictions
        input_values: Array of input values for the theoretical model
        metric_name: Name of the metric being validated
        
    Returns:
        Dictionary with validation results
    """
    if len(measured_values) != len(input_values):
        logger.error(f"Length mismatch: {len(measured_values)} measured values but {len(input_values)} input values")
        return {
            'status': 'error',
            'metric': metric_name,
            'message': f"Length mismatch in validation data for {metric_name}"
        }
    
    # Generate theoretical predictions
    theoretical_values = theoretical_model(input_values)
    
    # Calculate errors
    absolute_errors = np.abs(measured_values - theoretical_values)
    relative_errors = np.abs(absolute_errors / np.maximum(theoretical_values, 1e-10))
    
    # Calculate statistics
    mean_absolute_error = np.mean(absolute_errors)
    mean_relative_error = np.mean(relative_errors)
    max_relative_error = np.max(relative_errors)
    
    # Determine overall status
    status = _determine_validation_status(mean_relative_error)
    
    # Calculate R-squared (coefficient of determination)
    ss_total = np.sum((measured_values - np.mean(measured_values))**2)
    ss_residual = np.sum((measured_values - theoretical_values)**2)
    r_squared = 1 - (ss_residual / ss_total) if ss_total > 0 else 0
    
    return {
        'status': status,
        'metric': metric_name,
        'mean_absolute_error': mean_absolute_error,
        'mean_relative_error': mean_relative_error,
        'max_relative_error': max_relative_error,
        'r_squared': r_squared,
        'measured_values': measured_values,
        'theoretical_values': theoretical_values,
        'input_values': input_values,
        'message': (f"{metric_name} validation: mean relative error = {mean_relative_error:.3f}, "
                   f"R² = {r_squared:.3f}")
    }


def validate_acceleration_performance(acceleration_data: Dict, vehicle_specs: Dict) -> Dict:
    """
    Validate acceleration performance against theoretical models.
    
    Args:
        acceleration_data: Dictionary with acceleration test results
        vehicle_specs: Dictionary with vehicle specifications
        
    Returns:
        Dictionary with validation results
    """
    validation_results = {}
    
    # Check if we have enough data to validate
    if not acceleration_data or not vehicle_specs:
        logger.warning("Insufficient data for acceleration validation")
        return {'status': 'error', 'message': "Insufficient data for validation"}
    
    # Extract key metrics for validation
    time_75m = acceleration_data.get('finish_time')
    time_to_60mph = acceleration_data.get('time_to_60mph')
    time_to_100kph = acceleration_data.get('time_to_100kph')
    
    # Extract vehicle specifications
    mass = vehicle_specs.get('mass', 250.0)  # kg
    power = vehicle_specs.get('power', 60.0)  # kW
    drag_coefficient = vehicle_specs.get('drag_coefficient', 0.9)
    frontal_area = vehicle_specs.get('frontal_area', 1.2)  # m²
    
    # Basic validation against expected ranges
    if time_75m is not None:
        validation_results['acceleration_time'] = validate_in_range(
            time_75m, 'acceleration_time')
    
    if time_to_60mph is not None:
        validation_results['time_to_60mph'] = validate_in_range(
            time_to_60mph, 'time_to_60mph')
    
    if time_to_100kph is not None:
        validation_results['time_to_100kph'] = validate_in_range(
            time_to_100kph, 'time_to_100kph')
    
    # Theoretical model validation if we have detailed time-speed data
    if 'time' in acceleration_data and 'speed' in acceleration_data:
        time = np.array(acceleration_data['time'])
        speed = np.array(acceleration_data['speed'])
        
        # Simple power-limited acceleration model
        # a = (P/m/v) - (0.5*rho*Cd*A*v²/m) - Crr*g
        def theoretical_acceleration_model(time_points):
            # Initialize with starting speed > 0 to avoid division by zero
            speeds = [0.1]
            
            dt = time_points[1] - time_points[0] if len(time_points) > 1 else 0.01
            rho = 1.225  # Air density (kg/m³)
            crr = 0.015  # Rolling resistance coefficient
            
            # Convert power to watts
            power_watts = power * 1000
            
            # Initial acceleration
            a_initial = power_watts / (mass * speeds[0]) - (0.5 * rho * drag_coefficient * frontal_area * speeds[0]**2) / mass - crr * GRAVITY
            
            # Simple Euler integration
            for i in range(1, len(time_points)):
                v = speeds[-1]
                
                # Calculate acceleration at current speed
                if v > 0.1:
                    # Power-limited acceleration
                    a = power_watts / (mass * v) - (0.5 * rho * drag_coefficient * frontal_area * v**2) / mass - crr * GRAVITY
                else:
                    # Use initial acceleration for very low speeds
                    a = a_initial
                
                # Update speed
                new_speed = v + a * dt
                speeds.append(max(0.1, new_speed))  # Ensure speed is positive
            
            return np.array(speeds)
        
        # Validate against theoretical model
        validation_results['speed_vs_time'] = validate_theoretical_model(
            speed, theoretical_acceleration_model, time, 'acceleration_speed_profile')
    
    # Calculate overall validation status
    statuses = [r['status'] for r in validation_results.values() if 'status' in r]
    
    if 'critical_error' in statuses:
        overall_status = 'critical_error'
    elif 'warning' in statuses:
        overall_status = 'warning'
    elif 'acceptable' in statuses:
        overall_status = 'acceptable'
    elif 'good' in statuses:
        overall_status = 'good'
    else:
        overall_status = 'valid'
    
    return {
        'status': overall_status,
        'metric_validations': validation_results,
        'message': f"Acceleration validation overall status: {overall_status}"
    }


def validate_skidpad_performance(skidpad_data: Dict, vehicle_specs: Dict) -> Dict:
    """
    Validate skidpad performance against theoretical models.
    
    Args:
        skidpad_data: Dictionary with skidpad test results
        vehicle_specs: Dictionary with vehicle specifications
        
    Returns:
        Dictionary with validation results
    """
    validation_results = {}
    
    # Check if we have enough data to validate
    if not skidpad_data or not vehicle_specs:
        logger.warning("Insufficient data for skidpad validation")
        return {'status': 'error', 'message': "Insufficient data for validation"}
    
    # Extract key metrics for validation
    lateral_acceleration = skidpad_data.get('lateral_acceleration')  # g
    skidpad_time = skidpad_data.get('skidpad_time')  # s
    
    # Extract vehicle specifications
    mass = vehicle_specs.get('mass', 250.0)  # kg
    
    # Basic validation against expected ranges
    if lateral_acceleration is not None:
        validation_results['lateral_acceleration'] = validate_in_range(
            lateral_acceleration, 'lateral_acceleration')
    
    if skidpad_time is not None:
        validation_results['skidpad_time'] = validate_in_range(
            skidpad_time, 'skidpad_time')
    
    # Theoretical validation
    
    # For skidpad, check if the time and lateral acceleration are consistent
    if skidpad_time is not None and lateral_acceleration is not None:
        # Theoretical skidpad time based on lateral acceleration
        # T = 2π * sqrt(R/a) where R is skidpad radius and a is acceleration
        theoretical_time = 2 * np.pi * np.sqrt(FS_SKIDPAD_RADIUS / (lateral_acceleration * GRAVITY))
        
        relative_error = abs(skidpad_time - theoretical_time) / theoretical_time
        status = _determine_validation_status(relative_error)
        
        validation_results['skidpad_consistency'] = {
            'status': status,
            'metric': 'skidpad_consistency',
            'measured_time': skidpad_time,
            'theoretical_time': theoretical_time,
            'lateral_acceleration': lateral_acceleration,
            'relative_error': relative_error,
            'message': (f"Skidpad time consistency: measured={skidpad_time:.2f}s, "
                       f"theoretical={theoretical_time:.2f}s, error={relative_error:.3f}")
        }
    
    # Calculate overall validation status
    statuses = [r['status'] for r in validation_results.values() if 'status' in r]
    
    if 'critical_error' in statuses:
        overall_status = 'critical_error'
    elif 'warning' in statuses:
        overall_status = 'warning'
    elif 'acceptable' in statuses:
        overall_status = 'acceptable'
    elif 'good' in statuses:
        overall_status = 'good'
    else:
        overall_status = 'valid'
    
    return {
        'status': overall_status,
        'metric_validations': validation_results,
        'message': f"Skidpad validation overall status: {overall_status}"
    }


def validate_lap_time_performance(lap_data: Dict, vehicle_specs: Dict) -> Dict:
    """
    Validate lap time performance against theoretical models.
    
    Args:
        lap_data: Dictionary with lap time simulation results
        vehicle_specs: Dictionary with vehicle specifications
        
    Returns:
        Dictionary with validation results
    """
    validation_results = {}
    
    # Check if we have enough data to validate
    if not lap_data or not vehicle_specs:
        logger.warning("Insufficient data for lap time validation")
        return {'status': 'error', 'message': "Insufficient data for validation"}
    
    # Extract key metrics for validation
    lap_time = lap_data.get('lap_time')  # s
    avg_speed = lap_data.get('avg_speed')  # m/s
    max_speed = lap_data.get('max_speed')  # m/s
    
    # Extract vehicle specifications
    power = vehicle_specs.get('power', 60.0)  # kW
    mass = vehicle_specs.get('mass', 250.0)  # kg
    
    # Convert speeds to km/h for validation
    if avg_speed is not None:
        avg_speed_kph = avg_speed * MS_TO_KMH
    
    if max_speed is not None:
        max_speed_kph = max_speed * MS_TO_KMH
        # Validate top speed
        validation_results['top_speed'] = validate_in_range(
            max_speed_kph, 'top_speed')
    
    # Theoretical validation
    
    # For lap time, check if the average speed and lap time are consistent
    if lap_time is not None and avg_speed is not None and 'track_length' in lap_data:
        track_length = lap_data['track_length']  # m
        theoretical_time = track_length / avg_speed
        
        relative_error = abs(lap_time - theoretical_time) / theoretical_time
        status = _determine_validation_status(relative_error)
        
        validation_results['lap_time_consistency'] = {
            'status': status,
            'metric': 'lap_time_consistency',
            'measured_time': lap_time,
            'theoretical_time': theoretical_time,
            'average_speed': avg_speed,
            'track_length': track_length,
            'relative_error': relative_error,
            'message': (f"Lap time consistency: measured={lap_time:.2f}s, "
                       f"theoretical={theoretical_time:.2f}s, error={relative_error:.3f}")
        }
    
    # Power-to-weight and cornering validation
    if 'lateral_g' in lap_data and lap_data['lateral_g'] is not None:
        # Get maximum lateral acceleration
        max_lateral_g = np.max(lap_data['lateral_g'])
        
        validation_results['lateral_acceleration'] = validate_in_range(
            max_lateral_g, 'lateral_acceleration')
    
    # Calculate overall validation status
    statuses = [r['status'] for r in validation_results.values() if 'status' in r]
    
    if 'critical_error' in statuses:
        overall_status = 'critical_error'
    elif 'warning' in statuses:
        overall_status = 'warning'
    elif 'acceptable' in statuses:
        overall_status = 'acceptable'
    elif 'good' in statuses:
        overall_status = 'good'
    else:
        overall_status = 'valid'
    
    return {
        'status': overall_status,
        'metric_validations': validation_results,
        'message': f"Lap time validation overall status: {overall_status}"
    }


def validate_thermal_performance(thermal_data: Dict, vehicle_specs: Dict) -> Dict:
    """
    Validate thermal performance against theoretical models.
    
    Args:
        thermal_data: Dictionary with thermal simulation results
        vehicle_specs: Dictionary with vehicle specifications
        
    Returns:
        Dictionary with validation results
    """
    validation_results = {}
    
    # Check if we have enough data to validate
    if not thermal_data or not vehicle_specs:
        logger.warning("Insufficient data for thermal validation")
        return {'status': 'error', 'message': "Insufficient data for validation"}
    
    # Extract key metrics for validation
    max_engine_temp = thermal_data.get('max_engine_temp')  # °C
    max_coolant_temp = thermal_data.get('max_coolant_temp')  # °C
    max_oil_temp = thermal_data.get('max_oil_temp')  # °C
    
    # Basic validation against expected ranges
    if max_engine_temp is not None:
        validation_results['max_engine_temp'] = validate_in_range(
            max_engine_temp, 'max_engine_temp')
    
    if max_coolant_temp is not None:
        validation_results['max_coolant_temp'] = validate_in_range(
            max_coolant_temp, 'max_coolant_temp')
    
    if max_oil_temp is not None:
        validation_results['max_oil_temp'] = validate_in_range(
            max_oil_temp, 'max_oil_temp')
    
    # Theoretical validation if we have detailed temperature data
    if ('time' in thermal_data and 
        'engine_temp' in thermal_data and 
        'coolant_temp' in thermal_data and
        'heat_rejection' in thermal_data):
        
        time = np.array(thermal_data['time'])
        engine_temp = np.array(thermal_data['engine_temp'])
        coolant_temp = np.array(thermal_data['coolant_temp'])
        heat_rejection = np.array(thermal_data['heat_rejection'])
        
        # Check temperature difference consistency
        avg_temp_diff = np.mean(engine_temp - coolant_temp)
        
        if avg_temp_diff < 0:
            # Engine should generally be hotter than coolant
            validation_results['temp_diff_consistency'] = {
                'status': 'critical_error',
                'metric': 'temp_diff_consistency',
                'average_temp_diff': avg_temp_diff,
                'message': f"Average engine-coolant temperature difference ({avg_temp_diff:.1f}°C) is negative"
            }
        elif avg_temp_diff > 30:
            # Unusually large temperature difference
            validation_results['temp_diff_consistency'] = {
                'status': 'warning',
                'metric': 'temp_diff_consistency',
                'average_temp_diff': avg_temp_diff,
                'message': f"Average engine-coolant temperature difference ({avg_temp_diff:.1f}°C) is unusually large"
            }
        else:
            validation_results['temp_diff_consistency'] = {
                'status': 'valid',
                'metric': 'temp_diff_consistency',
                'average_temp_diff': avg_temp_diff,
                'message': f"Average engine-coolant temperature difference ({avg_temp_diff:.1f}°C) is reasonable"
            }
        
        # Check cooling system effectiveness vs vehicle speed if available
        if 'vehicle_speed' in thermal_data:
            vehicle_speed = np.array(thermal_data['vehicle_speed'])
            
            if len(vehicle_speed) == len(heat_rejection) and np.max(vehicle_speed) > 1.0:
                # Fit linear regression to heat rejection vs speed
                mask = vehicle_speed > 5.0  # Only consider when speed is significant
                
                if np.sum(mask) > 5:  # Need at least a few points for regression
                    slope, intercept, r_value, p_value, std_err = linregress(
                        vehicle_speed[mask], heat_rejection[mask])
                    
                    # Heat rejection should increase with speed
                    if slope <= 0:
                        validation_results['cooling_vs_speed'] = {
                            'status': 'critical_error',
                            'metric': 'cooling_vs_speed',
                            'slope': slope,
                            'message': f"Heat rejection does not increase with speed (slope={slope:.2f})"
                        }
                    else:
                        validation_results['cooling_vs_speed'] = {
                            'status': 'valid',
                            'metric': 'cooling_vs_speed',
                            'slope': slope,
                            'r_squared': r_value**2,
                            'message': f"Heat rejection increases with speed (slope={slope:.2f}, R²={r_value**2:.3f})"
                        }
    
    # Calculate overall validation status
    statuses = [r['status'] for r in validation_results.values() if 'status' in r]
    
    if 'critical_error' in statuses:
        overall_status = 'critical_error'
    elif 'warning' in statuses:
        overall_status = 'warning'
    elif 'acceptable' in statuses:
        overall_status = 'acceptable'
    elif 'good' in statuses:
        overall_status = 'good'
    else:
        overall_status = 'valid'
    
    return {
        'status': overall_status,
        'metric_validations': validation_results,
        'message': f"Thermal validation overall status: {overall_status}"
    }


def validate_vehicle_specs(vehicle_specs: Dict) -> Dict:
    """
    Validate vehicle specifications against expected ranges.
    
    Args:
        vehicle_specs: Dictionary with vehicle specifications
        
    Returns:
        Dictionary with validation results
    """
    validation_results = {}
    
    # Check if we have enough data to validate
    if not vehicle_specs:
        logger.warning("Insufficient data for vehicle specs validation")
        return {'status': 'error', 'message': "Insufficient data for validation"}
    
    # Extract key metrics for validation
    mass = vehicle_specs.get('mass')  # kg
    power = vehicle_specs.get('power')  # kW
    engine_displacement = vehicle_specs.get('engine_displacement')  # cc
    weight_distribution = vehicle_specs.get('weight_distribution')  # front weight fraction
    
    # Basic validation against expected ranges
    if mass is not None:
        validation_results['vehicle_mass'] = validate_in_range(
            mass, 'vehicle_mass')
    
    if weight_distribution is not None:
        validation_results['weight_distribution'] = validate_in_range(
            weight_distribution, 'weight_distribution')
    
    # Calculate and validate power-to-weight ratio
    if mass is not None and power is not None and mass > 0:
        power_to_weight = power / mass  # kW/kg
        validation_results['power_to_weight'] = validate_in_range(
            power_to_weight, 'power_to_weight')
    
    # Calculate and validate specific power
    if engine_displacement is not None and power is not None and engine_displacement > 0:
        # Convert power to HP and displacement to liters
        power_hp = power * KW_TO_HP
        displacement_l = engine_displacement / 1000
        
        specific_power = power_hp / displacement_l  # HP/L
        validation_results['specific_power'] = validate_in_range(
            specific_power, 'specific_power')
    
    # Calculate overall validation status
    statuses = [r['status'] for r in validation_results.values() if 'status' in r]
    
    if 'critical_error' in statuses:
        overall_status = 'critical_error'
    elif 'warning' in statuses:
        overall_status = 'warning'
    elif 'acceptable' in statuses:
        overall_status = 'acceptable'
    elif 'good' in statuses:
        overall_status = 'good'
    else:
        overall_status = 'valid'
    
    return {
        'status': overall_status,
        'metric_validations': validation_results,
        'message': f"Vehicle specifications validation overall status: {overall_status}"
    }


def validate_full_vehicle_performance(simulation_results: Dict) -> Dict:
    """
    Validate full vehicle performance across all simulation domains.
    
    Args:
        simulation_results: Dictionary with all simulation results
        
    Returns:
        Dictionary with comprehensive validation results
    """
    # Extract different result sets
    acceleration_data = simulation_results.get('acceleration', {})
    skidpad_data = simulation_results.get('skidpad', {})
    lap_data = simulation_results.get('lap', {})
    thermal_data = simulation_results.get('thermal', {})
    vehicle_specs = simulation_results.get('specs', {})
    
    # Perform domain-specific validations
    validation_results = {}
    
    if acceleration_data:
        validation_results['acceleration'] = validate_acceleration_performance(
            acceleration_data, vehicle_specs)
    
    if skidpad_data:
        validation_results['skidpad'] = validate_skidpad_performance(
            skidpad_data, vehicle_specs)
    
    if lap_data:
        validation_results['lap'] = validate_lap_time_performance(
            lap_data, vehicle_specs)
    
    if thermal_data:
        validation_results['thermal'] = validate_thermal_performance(
            thermal_data, vehicle_specs)
    
    if vehicle_specs:
        validation_results['vehicle_specs'] = validate_vehicle_specs(
            vehicle_specs)
    
    # Calculate overall validation status
    domain_statuses = [v['status'] for v in validation_results.values() if 'status' in v]
    
    if 'critical_error' in domain_statuses:
        overall_status = 'critical_error'
    elif 'warning' in domain_statuses:
        overall_status = 'warning'
    elif 'acceptable' in domain_statuses:
        overall_status = 'acceptable'
    elif 'good' in domain_statuses:
        overall_status = 'good'
    else:
        overall_status = 'valid'
    
    # Compile summary of issues
    issues = []
    
    for domain, domain_results in validation_results.items():
        if 'metric_validations' in domain_results:
            for metric, metric_result in domain_results['metric_validations'].items():
                if metric_result.get('status') in ['warning', 'critical_error']:
                    issues.append({
                        'domain': domain,
                        'metric': metric,
                        'status': metric_result.get('status'),
                        'message': metric_result.get('message')
                    })
    
    return {
        'status': overall_status,
        'domain_validations': validation_results,
        'issues': issues,
        'message': f"Overall validation status: {overall_status} with {len(issues)} issues"
    }


def compare_simulation_to_real_data(simulation_data: Dict, 
                                  real_data: Dict,
                                  metrics: Optional[List[str]] = None) -> Dict:
    """
    Compare simulation results to real-world data.
    
    Args:
        simulation_data: Dictionary with simulation results
        real_data: Dictionary with real-world measurements
        metrics: Optional list of metrics to compare
        
    Returns:
        Dictionary with comparison results
    """
    if not metrics:
        # Use all matching metrics
        metrics = [key for key in simulation_data if key in real_data]
    
    comparison_results = {}
    
    for metric in metrics:
        if metric in simulation_data and metric in real_data:
            sim_value = simulation_data[metric]
            real_value = real_data[metric]
            
            absolute_error = abs(sim_value - real_value)
            relative_error = absolute_error / abs(real_value) if abs(real_value) > 1e-10 else 0
            
            status = _determine_validation_status(relative_error)
            
            comparison_results[metric] = {
                'status': status,
                'simulation_value': sim_value,
                'real_value': real_value,
                'absolute_error': absolute_error,
                'relative_error': relative_error,
                'message': (f"{metric}: sim={sim_value:.3f}, real={real_value:.3f}, "
                           f"relative error={relative_error:.3f}")
            }
    
    # Calculate overall comparison status
    statuses = [r['status'] for r in comparison_results.values() if 'status' in r]
    
    if 'critical_error' in statuses:
        overall_status = 'critical_error'
    elif 'warning' in statuses:
        overall_status = 'warning'
    elif 'acceptable' in statuses:
        overall_status = 'acceptable'
    elif 'good' in statuses:
        overall_status = 'good'
    else:
        overall_status = 'valid'
    
    return {
        'status': overall_status,
        'metric_comparisons': comparison_results,
        'message': f"Simulation-real data comparison status: {overall_status}"
    }


def load_reference_data(file_path: str) -> Dict:
    """
    Load reference data from file.
    
    Args:
        file_path: Path to reference data file
        
    Returns:
        Dictionary with reference data
    """
    try:
        if file_path.endswith('.json'):
            with open(file_path, 'r') as f:
                return json.load(f)
        elif file_path.endswith('.csv'):
            return pd.read_csv(file_path).to_dict('list')
        else:
            logger.error(f"Unsupported reference data format: {file_path}")
            return {}
    except Exception as e:
        logger.error(f"Error loading reference data from {file_path}: {str(e)}")
        return {}


def save_validation_results(validation_results: Dict, file_path: str) -> bool:
    """
    Save validation results to file.
    
    Args:
        validation_results: Dictionary with validation results
        file_path: Path to save results
        
    Returns:
        Boolean indicating success
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Save as JSON
        with open(file_path, 'w') as f:
            json.dump(validation_results, f, indent=2)
        
        logger.info(f"Validation results saved to {file_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving validation results to {file_path}: {str(e)}")
        return False


def plot_validation_results(validation_results: Dict, save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot validation results for visual inspection.
    
    Args:
        validation_results: Dictionary with validation results
        save_path: Optional path to save the plot
        
    Returns:
        Matplotlib figure
    """
    # Create figure
    fig = plt.figure(figsize=(15, 10))
    gs = plt.GridSpec(2, 2)
    
    # Map validation status to color
    status_colors = {
        'valid': 'green',
        'good': 'green',
        'acceptable': 'yellow',
        'warning': 'orange',
        'critical_error': 'red',
        'error': 'gray',
        'unknown': 'gray'
    }
    
    # Extract domain results
    domain_validations = validation_results.get('domain_validations', {})
    
    # Plot domain statuses
    ax1 = fig.add_subplot(gs[0, 0])
    
    domains = list(domain_validations.keys())
    domain_statuses = [domain_validations[d].get('status', 'unknown') for d in domains]
    domain_colors = [status_colors[status] for status in domain_statuses]
    
    if domains:
        bars = ax1.barh(domains, [1] * len(domains), color=domain_colors)
        
        # Add status text
        for bar, status in zip(bars, domain_statuses):
            ax1.text(0.5, bar.get_y() + bar.get_height()/2, 
                   status.replace('_', ' ').title(), 
                   ha='center', va='center', color='black', fontweight='bold')
        
        ax1.set_xlim(0, 1)
        ax1.set_xticks([])
        ax1.set_xlabel('Status')
        ax1.set_title('Validation Status by Domain')
    
    # Plot metric validations
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Collect metric relative errors
    all_metrics = []
    all_errors = []
    all_colors = []
    
    for domain, domain_result in domain_validations.items():
        if 'metric_validations' in domain_result:
            for metric, metric_result in domain_result['metric_validations'].items():
                if 'relative_error' in metric_result:
                    all_metrics.append(f"{domain}: {metric}")
                    all_errors.append(metric_result['relative_error'])
                    all_colors.append(status_colors[metric_result.get('status', 'unknown')])
    
    if all_metrics:
        # Sort by error magnitude
        sorted_indices = np.argsort(all_errors)
        sorted_metrics = [all_metrics[i] for i in sorted_indices]
        sorted_errors = [all_errors[i] for i in sorted_indices]
        sorted_colors = [all_colors[i] for i in sorted_indices]
        
        bars = ax2.barh(sorted_metrics[-10:], sorted_errors[-10:], color=sorted_colors[-10:])
        
        for bar, error in zip(bars, sorted_errors[-10:]):
            ax2.text(error + 0.01, bar.get_y() + bar.get_height()/2, 
                   f"{error:.3f}", va='center')
        
        ax2.axvline(x=VALIDATION_THRESHOLDS['warning'], color='orange', linestyle='--', 
                  label=f"Warning ({VALIDATION_THRESHOLDS['warning']})")
        ax2.axvline(x=VALIDATION_THRESHOLDS['critical_error'], color='red', linestyle='--', 
                  label=f"Critical ({VALIDATION_THRESHOLDS['critical_error']})")
        
        ax2.set_xlabel('Relative Error')
        ax2.set_title('Top Relative Errors by Metric')
        ax2.legend()
    
    # Plot validation issues
    ax3 = fig.add_subplot(gs[1, :])
    
    issues = validation_results.get('issues', [])
    
    if issues:
        # Extract issue texts and statuses
        issue_texts = [f"{issue['domain']}: {issue['metric']} - {issue['message']}" for issue in issues]
        issue_statuses = [issue['status'] for issue in issues]
        issue_colors = [status_colors[status] for status in issue_statuses]
        
        # Only show top 10 issues if there are more
        if len(issues) > 10:
            issue_texts = issue_texts[:10]
            issue_colors = issue_colors[:10]
            ax3.set_title('Top 10 Validation Issues')
        else:
            ax3.set_title(f'All Validation Issues ({len(issues)})')
        
        # Plot colored lines with issue text
        for i, (text, color) in enumerate(zip(issue_texts, issue_colors)):
            ax3.text(0.01, 1 - (i + 1) * 0.1, text, 
                   color='black', backgroundcolor=color, alpha=0.7,
                   bbox=dict(facecolor=color, alpha=0.3, pad=5))
        
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1)
        ax3.set_xticks([])
        ax3.set_yticks([])
    else:
        ax3.text(0.5, 0.5, "No validation issues found", 
               ha='center', va='center', fontsize=14, 
               backgroundcolor='green', alpha=0.3,
               bbox=dict(facecolor='green', alpha=0.1, pad=10))
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1)
        ax3.set_xticks([])
        ax3.set_yticks([])
        ax3.set_title('Validation Issues')
    
    # Set overall title
    fig.suptitle(f"Validation Results - Overall Status: {validation_results.get('status', 'unknown').replace('_', ' ').title()}", 
               fontsize=16)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig