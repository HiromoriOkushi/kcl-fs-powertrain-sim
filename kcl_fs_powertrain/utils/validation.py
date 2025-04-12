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

# Import constants
from .constants import (
    GRAVITY, FS_ACCELERATION_LENGTH, FS_SKIDPAD_RADIUS,
    MS_TO_KMH, MS_TO_MPH, KW_TO_HP, KG_TO_LBS
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
    'acceleration_time': (3.6, 5.0),    # 75m acceleration time (s) - Updated range
    'time_to_60mph': (3.2, 5.0),        # 0-60 mph time (s) - Updated range
    'time_to_100kph': (3.0, 4.8),       # 0-100 km/h time (s) - Updated range

    # Speed metrics
    'top_speed': (95.0, 140.0),         # Top speed (km/h) - Updated range
    'speed_at_end_of_accel': (85.0, 125.0),  # Speed at end of acceleration event (km/h)

    # Lateral acceleration
    'max_lateral_g': (1.3, 2.3),       # Maximum lateral acceleration (g) - Updated range
    'skidpad_time': (4.8, 5.8),         # Skidpad lap time (s) - Updated range

    # Thermal metrics
    'max_engine_temp': (90.0, 125.0),   # Maximum engine temperature (°C) - Updated range
    'max_coolant_temp': (85.0, 115.0),  # Maximum coolant temperature (°C) - Updated range
    'max_oil_temp': (95.0, 145.0),      # Maximum oil temperature (°C) - Updated range

    # Power metrics
    'power_to_weight': (0.20, 0.40),     # Power-to-weight ratio (kW/kg) - Updated range
    'specific_power': (130.0, 200.0),   # Power per liter of displacement (HP/L)

    # Weight metrics
    'vehicle_mass': (170.0, 280.0),     # Vehicle mass with driver (kg) - Updated range
    'weight_distribution': (0.42, 0.52), # Front weight distribution (fraction)

    # Lap Time / Avg Speed
    'lap_time_autocross': (60.0, 90.0), # Typical autocross lap time (s) - Added
    'avg_speed_autocross': (40.0, 65.0), # Typical autocross avg speed (km/h) - Added

    # Efficiency metrics (highly variable, rough estimate)
    'fuel_efficiency_endurance': (0.3, 0.8)  # L/lap during endurance - Added
}

# Validation error thresholds
VALIDATION_THRESHOLDS = {
    'critical_error': 0.30,     # Relative error >= 30%
    'warning': 0.15,            # Relative error >= 15%
    'acceptable': 0.08,         # Relative error >= 8%
    'good': 0.03                # Relative error < 3%
}


def validate_in_range(value: float, metric_name: str,
                    custom_range: Optional[Tuple[float, float]] = None) -> Dict:
    """
    Validate if a value is within expected range for a metric.

    Args:
        value: Value to validate.
        metric_name: Name of the metric to check (must be in FS_PERFORMANCE_RANGES).
        custom_range: Optional custom range tuple (min_val, max_val) to override default.

    Returns:
        Dictionary with validation results:
            'status': 'valid', 'warning', 'critical_error', 'unknown'.
            'metric': Name of the metric.
            'value': The validated value.
            'expected_range': Tuple of (min_val, max_val).
            'relative_deviation': Deviation from the closest bound as a fraction of the range width (if outside).
            'message': Human-readable validation message.
    """
    if value is None or not np.isfinite(value):
         return {
            'status': 'error', 'metric': metric_name, 'value': value,
            'expected_range': None, 'relative_deviation': None,
            'message': f"Invalid or missing value for {metric_name}"
        }

    expected_range = custom_range if custom_range else FS_PERFORMANCE_RANGES.get(metric_name)

    if not expected_range:
        logger.warning(f"No expected range found for metric: {metric_name}")
        return {
            'status': 'unknown', 'metric': metric_name, 'value': value,
            'expected_range': None, 'relative_deviation': None,
            'message': f"No expected range defined for {metric_name}"
        }

    min_value, max_value = expected_range
    range_width = max_value - min_value
    if range_width <= 0: range_width = abs(min_value) * 0.1 if min_value != 0 else 1.0 # Handle zero range

    deviation = 0.0
    status = 'valid'
    message = f"{_format_metric_name(metric_name)} ({value:.3f}) is within expected range ({min_value:.3f} - {max_value:.3f})"

    if value < min_value:
        deviation = (min_value - value) / range_width
        status = _determine_validation_status_from_deviation(deviation)
        message = f"{_format_metric_name(metric_name)} ({value:.3f}) is below expected minimum ({min_value:.3f})"
    elif value > max_value:
        deviation = (value - max_value) / range_width
        status = _determine_validation_status_from_deviation(deviation)
        message = f"{_format_metric_name(metric_name)} ({value:.3f}) is above expected maximum ({max_value:.3f})"

    return {
        'status': status,
        'metric': metric_name,
        'value': value,
        'expected_range': expected_range,
        'relative_deviation': deviation, # Deviation relative to range width
        'message': message
    }

def _determine_validation_status_from_deviation(relative_deviation: float) -> str:
    """Determine validation status based on deviation relative to range width."""
    if relative_deviation >= VALIDATION_THRESHOLDS['critical_error']: return 'critical_error'
    if relative_deviation >= VALIDATION_THRESHOLDS['warning']: return 'warning'
    if relative_deviation >= VALIDATION_THRESHOLDS['acceptable']: return 'acceptable'
    return 'good' # If deviation is small relative to the range

def _determine_validation_status(relative_error: float) -> str:
    """Determine validation status based on relative error against a single value."""
    abs_error = abs(relative_error)
    if abs_error >= VALIDATION_THRESHOLDS['critical_error']: return 'critical_error'
    if abs_error >= VALIDATION_THRESHOLDS['warning']: return 'warning'
    if abs_error >= VALIDATION_THRESHOLDS['acceptable']: return 'acceptable'
    return 'good'

def validate_theoretical_model(measured_values: np.ndarray,
                             theoretical_model: Callable[[np.ndarray], np.ndarray],
                             input_values: np.ndarray,
                             metric_name: str) -> Dict:
    """
    Validate measured values against a theoretical model using basic statistics.

    Args:
        measured_values: Array of measured/simulated values.
        theoretical_model: Function that takes input_values and returns theoretical predictions.
        input_values: Array of input values for the theoretical model.
        metric_name: Name of the metric being validated.

    Returns:
        Dictionary with validation results including MAE, MRE, MaxRE, R².
    """
    measured_values = np.array(measured_values)
    input_values = np.array(input_values)

    if len(measured_values) != len(input_values):
        msg = f"Length mismatch: {len(measured_values)} measured vs {len(input_values)} input for {metric_name}"
        logger.error(msg)
        return {'status': 'error', 'metric': metric_name, 'message': msg}

    # Generate theoretical predictions
    try:
        theoretical_values = theoretical_model(input_values)
    except Exception as e:
        msg = f"Error executing theoretical model for {metric_name}: {e}"
        logger.error(msg)
        return {'status': 'error', 'metric': metric_name, 'message': msg}

    if len(theoretical_values) != len(measured_values):
         msg = f"Theoretical model output length mismatch for {metric_name}"
         logger.error(msg)
         return {'status': 'error', 'metric': metric_name, 'message': msg}

    # Calculate errors (handle potential division by zero in relative error)
    absolute_errors = np.abs(measured_values - theoretical_values)
    relative_errors = np.divide(absolute_errors, np.abs(theoretical_values),
                                out=np.zeros_like(absolute_errors),
                                where=np.abs(theoretical_values) > 1e-9) # Avoid division by zero

    # Calculate statistics
    mean_absolute_error = np.mean(absolute_errors)
    mean_relative_error = np.mean(relative_errors)
    max_relative_error = np.max(relative_errors)

    # Determine overall status based on mean relative error
    status = _determine_validation_status(mean_relative_error)

    # Calculate R-squared (coefficient of determination)
    ss_total = np.sum((measured_values - np.mean(measured_values))**2)
    ss_residual = np.sum(absolute_errors**2)
    r_squared = 1 - (ss_residual / ss_total) if ss_total > 1e-9 else 0.0 # Avoid division by zero

    return {
        'status': status,
        'metric': metric_name,
        'mean_absolute_error': mean_absolute_error,
        'mean_relative_error': mean_relative_error,
        'max_relative_error': max_relative_error,
        'r_squared': r_squared,
        'measured_values': measured_values.tolist(), # Convert for JSON
        'theoretical_values': theoretical_values.tolist(), # Convert for JSON
        'input_values': input_values.tolist(), # Convert for JSON
        'message': (f"{metric_name} validation vs theoretical: MRE={mean_relative_error:.3f}, "
                   f"R²={r_squared:.3f}")
    }

# --- Domain-Specific Validation Functions ---
# (Implementations based on descriptions, focusing on using validate_in_range and potentially validate_theoretical_model)

def validate_acceleration_performance(acceleration_data: Dict, vehicle_specs: Dict) -> Dict:
    """Validate acceleration performance against expected ranges."""
    validation_results = {}
    metrics_to_validate = ['acceleration_time', 'time_to_60mph', 'time_to_100kph', 'speed_at_end_of_accel']

    for metric in metrics_to_validate:
        if metric in acceleration_data and acceleration_data[metric] is not None:
             # Special handling for speed conversion if needed
             value = acceleration_data[metric]
             if metric == 'speed_at_end_of_accel':
                 value *= MS_TO_KMH # Convert m/s to km/h for range check
             validation_results[metric] = validate_in_range(value, metric)

    # Add other theoretical checks if needed, e.g., comparing peak acceleration to calculated limits

    statuses = [r['status'] for r in validation_results.values() if 'status' in r]
    overall_status = max(statuses, key=lambda s: ['good', 'acceptable', 'warning', 'critical_error', 'error', 'unknown'].index(s)) if statuses else 'unknown'

    return {
        'status': overall_status,
        'metric_validations': validation_results,
        'message': f"Acceleration validation status: {overall_status}"
    }


def validate_skidpad_performance(skidpad_data: Dict, vehicle_specs: Dict) -> Dict:
    """Validate skidpad performance against expected ranges."""
    validation_results = {}
    metrics_to_validate = ['skidpad_time', 'max_lateral_g'] # Use max_lateral_g if calculated

    # Adjust key name if needed
    if 'lateral_acceleration' in skidpad_data and 'max_lateral_g' not in skidpad_data:
         skidpad_data['max_lateral_g'] = skidpad_data['lateral_acceleration']

    for metric in metrics_to_validate:
         if metric in skidpad_data and skidpad_data[metric] is not None:
             validation_results[metric] = validate_in_range(skidpad_data[metric], metric)

    # Add consistency check between time and lateral_g if both exist
    if 'skidpad_time' in validation_results and 'max_lateral_g' in validation_results:
        time = skidpad_data['skidpad_time']
        lat_g = skidpad_data['max_lateral_g']
        if time > 0 and lat_g > 0:
            # Theoretical time T = 2 * pi * sqrt(R/a)
            theoretical_time = 2 * np.pi * np.sqrt(FS_SKIDPAD_RADIUS / (lat_g * GRAVITY))
            relative_error = abs(time - theoretical_time) / theoretical_time
            status = _determine_validation_status(relative_error)
            validation_results['skidpad_consistency'] = {
                 'status': status, 'metric': 'skidpad_consistency',
                 'message': f"Time/G consistency error: {relative_error:.2%}"
            }


    statuses = [r['status'] for r in validation_results.values() if 'status' in r]
    overall_status = max(statuses, key=lambda s: ['good', 'acceptable', 'warning', 'critical_error', 'error', 'unknown'].index(s)) if statuses else 'unknown'

    return {
        'status': overall_status,
        'metric_validations': validation_results,
        'message': f"Skidpad validation status: {overall_status}"
    }


def validate_lap_time_performance(lap_data: Dict, vehicle_specs: Dict) -> Dict:
    """Validate lap time performance against expected ranges."""
    validation_results = {}
    metrics_to_validate = ['lap_time_autocross', 'avg_speed_autocross', 'max_speed']

    # Adjust key names if needed
    if 'lap_time' in lap_data and 'lap_time_autocross' not in lap_data:
        lap_data['lap_time_autocross'] = lap_data['lap_time']
    if 'avg_speed_kph' in lap_data and 'avg_speed_autocross' not in lap_data:
        lap_data['avg_speed_autocross'] = lap_data['avg_speed_kph']
    if 'max_speed_kph' in lap_data and 'max_speed' not in lap_data:
         lap_data['max_speed'] = lap_data['max_speed_kph']

    for metric in metrics_to_validate:
         if metric in lap_data and lap_data[metric] is not None:
             validation_results[metric] = validate_in_range(lap_data[metric], metric)

    statuses = [r['status'] for r in validation_results.values() if 'status' in r]
    overall_status = max(statuses, key=lambda s: ['good', 'acceptable', 'warning', 'critical_error', 'error', 'unknown'].index(s)) if statuses else 'unknown'

    return {
        'status': overall_status,
        'metric_validations': validation_results,
        'message': f"Lap Time validation status: {overall_status}"
    }


def validate_thermal_performance(thermal_data: Dict, vehicle_specs: Dict) -> Dict:
    """Validate thermal performance against expected ranges."""
    validation_results = {}
    metrics_to_validate = ['max_engine_temp', 'max_coolant_temp', 'max_oil_temp']

    for metric in metrics_to_validate:
        if metric in thermal_data and thermal_data[metric] is not None:
            validation_results[metric] = validate_in_range(thermal_data[metric], metric)

    statuses = [r['status'] for r in validation_results.values() if 'status' in r]
    overall_status = max(statuses, key=lambda s: ['good', 'acceptable', 'warning', 'critical_error', 'error', 'unknown'].index(s)) if statuses else 'unknown'

    return {
        'status': overall_status,
        'metric_validations': validation_results,
        'message': f"Thermal validation status: {overall_status}"
    }


def validate_vehicle_specs(vehicle_specs: Dict) -> Dict:
    """Validate vehicle specifications against expected ranges."""
    validation_results = {}
    metrics_to_validate = ['vehicle_mass', 'power_to_weight', 'specific_power', 'weight_distribution']

    # Extract necessary sub-dictionaries if needed
    vehicle_params = vehicle_specs.get('vehicle', {})
    engine_params = vehicle_specs.get('engine', {})

    # Prepare data for validation
    mass = vehicle_params.get('mass')
    power_hp = engine_params.get('max_power_hp')
    displacement_cc = engine_params.get('displacement_cc')
    weight_dist = vehicle_params.get('weight_distribution_front')

    specs_to_validate = {'vehicle_mass': mass, 'weight_distribution': weight_dist}

    if mass and power_hp:
        specs_to_validate['power_to_weight'] = (power_hp * HP_TO_KW) / mass
    if displacement_cc and power_hp:
        specs_to_validate['specific_power'] = power_hp / (displacement_cc / 1000.0)

    for metric in metrics_to_validate:
        if metric in specs_to_validate and specs_to_validate[metric] is not None:
            validation_results[metric] = validate_in_range(specs_to_validate[metric], metric)

    statuses = [r['status'] for r in validation_results.values() if 'status' in r]
    overall_status = max(statuses, key=lambda s: ['good', 'acceptable', 'warning', 'critical_error', 'error', 'unknown'].index(s)) if statuses else 'unknown'

    return {
        'status': overall_status,
        'metric_validations': validation_results,
        'message': f"Vehicle Specs validation status: {overall_status}"
    }


def validate_full_vehicle_performance(simulation_results: Dict) -> Dict:
    """
    Validate full vehicle performance across all simulation domains.

    Args:
        simulation_results: Dictionary containing results for different domains
                            (e.g., 'acceleration', 'lap_time', 'thermal', 'specs').

    Returns:
        Dictionary with comprehensive validation results.
    """
    validation_results = {}
    vehicle_specs = simulation_results.get('specs', {}) # Specs needed by multiple validators

    # Perform domain-specific validations
    if 'acceleration' in simulation_results:
        validation_results['acceleration'] = validate_acceleration_performance(
            simulation_results['acceleration'], vehicle_specs)

    if 'skidpad' in simulation_results:
        validation_results['skidpad'] = validate_skidpad_performance(
            simulation_results['skidpad'], vehicle_specs)

    if 'lap_time' in simulation_results: # Assuming lap_time contains results from lap simulation
        validation_results['lap_time'] = validate_lap_time_performance(
            simulation_results['lap_time'], vehicle_specs)

    if 'thermal' in simulation_results:
        validation_results['thermal'] = validate_thermal_performance(
            simulation_results['thermal'], vehicle_specs)

    if vehicle_specs:
        validation_results['vehicle_specs'] = validate_vehicle_specs(vehicle_specs)

    # Calculate overall validation status
    domain_statuses = [v['status'] for v in validation_results.values() if 'status' in v]
    overall_status = 'unknown'
    if domain_statuses:
        overall_status = max(domain_statuses, key=lambda s: ['good', 'acceptable', 'warning', 'critical_error', 'error', 'unknown'].index(s))

    # Compile summary of issues
    issues = []
    for domain, domain_results in validation_results.items():
        if domain_results and 'metric_validations' in domain_results:
            for metric, metric_result in domain_results['metric_validations'].items():
                if metric_result and metric_result.get('status') not in ['valid', 'good', 'unknown', None]:
                    issues.append({
                        'domain': domain,
                        'metric': metric,
                        'status': metric_result.get('status'),
                        'message': metric_result.get('message', 'No message')
                    })

    return {
        'status': overall_status,
        'domain_validations': validation_results,
        'issues': issues,
        'message': f"Overall validation status: {overall_status} with {len(issues)} issues found."
    }


def compare_simulation_to_real_data(simulation_data: Dict,
                                  real_data: Dict,
                                  metrics: Optional[List[str]] = None) -> Dict:
    """
    Compare simulation results to real-world data.

    Args:
        simulation_data: Dictionary with simulation results (metric: value).
        real_data: Dictionary with real-world measurements (metric: value).
        metrics: Optional list of specific metric keys to compare. If None, compares all common keys.

    Returns:
        Dictionary with comparison results for each metric.
    """
    if metrics is None:
        # Find common metrics
        metrics = list(set(simulation_data.keys()) & set(real_data.keys()))
        if not metrics:
            logger.warning("No common metrics found between simulation and real data.")
            return {'status': 'unknown', 'metric_comparisons': {}, 'message': "No common metrics found."}

    comparison_results = {}
    relative_errors = []

    for metric in metrics:
        sim_value = simulation_data.get(metric)
        real_value = real_data.get(metric)

        if sim_value is None or real_value is None or not np.isfinite(sim_value) or not np.isfinite(real_value):
            comparison_results[metric] = {
                'status': 'unknown', 'message': f"Missing or invalid data for {metric}"
            }
            continue

        absolute_error = sim_value - real_value
        relative_error = absolute_error / real_value if abs(real_value) > 1e-9 else (0.0 if abs(absolute_error) < 1e-9 else float('inf'))
        relative_errors.append(abs(relative_error))

        status = _determine_validation_status(relative_error)

        comparison_results[metric] = {
            'status': status,
            'simulation_value': sim_value,
            'real_value': real_value,
            'absolute_error': absolute_error,
            'relative_error': relative_error,
            'message': (f"{_format_metric_name(metric)}: Sim={sim_value:.3f}, Real={real_value:.3f}, "
                       f"Rel. Error={relative_error:.2%}")
        }

    # Calculate overall comparison status based on average relative error
    if relative_errors:
        avg_relative_error = np.mean(relative_errors)
        overall_status = _determine_validation_status(avg_relative_error)
        overall_message = f"Simulation vs Real Data comparison status: {overall_status} (Avg Rel Err: {avg_relative_error:.2%})"
    else:
        overall_status = 'unknown'
        overall_message = "No valid metrics were compared."


    return {
        'status': overall_status,
        'metric_comparisons': comparison_results,
        'message': overall_message
    }


def load_reference_data(file_path: str) -> Dict:
    """
    Load reference data from file (JSON or CSV).

    Args:
        file_path: Path to reference data file.

    Returns:
        Dictionary with reference data, or empty dict if loading fails.
    """
    if not os.path.exists(file_path):
        logger.error(f"Reference data file not found: {file_path}")
        return {}

    try:
        if file_path.lower().endswith('.json'):
            with open(file_path, 'r') as f:
                data = json.load(f)
        elif file_path.lower().endswith('.csv'):
            df = pd.read_csv(file_path)
            # Convert to dict, handle potential single row/column cases
            if len(df) == 1 and len(df.columns) > 1:
                 data = df.iloc[0].to_dict() # Assume single row represents metrics
            elif len(df.columns) == 2 and df.columns[0].lower() == 'metric':
                 data = df.set_index(df.columns[0])[df.columns[1]].to_dict() # Metric, Value format
            else:
                 data = df.to_dict('list') # Default list format
        else:
            logger.error(f"Unsupported reference data format: {file_path}. Use JSON or CSV.")
            return {}

        logger.info(f"Reference data loaded successfully from {file_path}")
        return data

    except Exception as e:
        logger.error(f"Error loading reference data from {file_path}: {e}")
        return {}


def save_validation_results(validation_results: Dict, file_path: str) -> bool:
    """
    Save validation results to a JSON file.

    Args:
        validation_results: Dictionary with validation results.
        file_path: Path to save results JSON file.

    Returns:
        Boolean indicating success.
    """
    try:
        # Ensure directory exists
        output_dir = os.path.dirname(file_path)
        if output_dir: # Check if directory part exists
             os.makedirs(output_dir, exist_ok=True)

        # Convert numpy arrays to lists for JSON serialization if present
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                                np.int16, np.int32, np.int64, np.uint8,
                                np.uint16, np.uint32, np.uint64)):
                return int(obj)
            elif isinstance(obj, (np.float_, np.float16, np.float32,
                                np.float64)):
                return float(obj)
            elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
                return {'real': obj.real, 'imag': obj.imag}
            elif isinstance(obj, (np.bool_)):
                return bool(obj)
            elif isinstance(obj, (np.void)):
                return None
            return obj # Keep other types as is

        # Save as JSON with numpy conversion
        with open(file_path, 'w') as f:
            json.dump(validation_results, f, indent=2, default=convert_numpy)

        logger.info(f"Validation results saved to {file_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving validation results to {file_path}: {e}")
        return False


def plot_validation_results(validation_results: Dict, save_path: Optional[str] = None) -> Optional[plt.Figure]:
    """
    Plot validation results for visual inspection.

    Args:
        validation_results: Dictionary from validate_full_vehicle_performance.
        save_path: Optional path to save the plot.

    Returns:
        Matplotlib figure or None if error.
    """
    if not validation_results or 'domain_validations' not in validation_results:
        logger.error("Invalid or empty validation results provided.")
        return None

    fig = plt.figure(figsize=(16, 12)) # Slightly larger figure
    gs = plt.GridSpec(3, 2) # Adjusted grid

    # Map validation status to color and numerical score for sorting
    status_map = {
        'good': ('#2ca02c', 0), 'valid': ('#2ca02c', 0), # Green shades
        'acceptable': ('#ffdd7f', 1), # Light yellow
        'warning': ('#ff7f0e', 2), # Orange
        'critical_error': ('#d62728', 3), # Red
        'error': ('#7f7f7f', 4), # Grey
        'unknown': ('#c7c7c7', 5) # Light grey
    }

    # --- Plot Domain Statuses ---
    ax1 = fig.add_subplot(gs[0, 0])
    domain_validations = validation_results.get('domain_validations', {})
    domains = list(domain_validations.keys())
    domain_statuses = [domain_validations[d].get('status', 'unknown') for d in domains]

    # Sort domains by severity
    sorted_indices = sorted(range(len(domains)), key=lambda k: status_map[domain_statuses[k]][1])
    sorted_domains = [domains[i] for i in sorted_indices]
    sorted_statuses = [domain_statuses[i] for i in sorted_indices]
    sorted_colors = [status_map[s][0] for s in sorted_statuses]

    if sorted_domains:
        bars = ax1.barh(sorted_domains, [1] * len(sorted_domains), color=sorted_colors)
        for bar, status in zip(bars, sorted_statuses):
            ax1.text(0.5, bar.get_y() + bar.get_height()/2,
                   status.replace('_', ' ').title(),
                   ha='center', va='center', color='black', fontweight='bold', fontsize=9)
        ax1.set_xlim(0, 1)
        ax1.set_xticks([])
        _apply_common_ax_settings(ax1, xlabel='Status', title='Validation Status by Domain')
    else:
         ax1.text(0.5, 0.5, "No domain results", ha='center', va='center')
         _apply_common_ax_settings(ax1, title='Validation Status by Domain')

    # --- Plot Metric Relative Deviations/Errors ---
    ax2 = fig.add_subplot(gs[0, 1])
    all_metrics = []
    all_errors = [] # Use relative_deviation or relative_error
    all_colors = []

    for domain, domain_result in domain_validations.items():
        if 'metric_validations' in domain_result:
            for metric, metric_result in domain_result['metric_validations'].items():
                error = metric_result.get('relative_deviation', metric_result.get('relative_error'))
                status = metric_result.get('status', 'unknown')
                if error is not None and np.isfinite(error):
                    all_metrics.append(f"{domain}: {_format_metric_name(metric)}")
                    all_errors.append(abs(error)) # Plot absolute value
                    all_colors.append(status_map[status][0])

    if all_metrics:
        # Sort by error magnitude (highest first)
        sorted_indices = np.argsort(all_errors)[::-1]
        # Limit to top 15 metrics for readability
        num_to_plot = min(15, len(all_metrics))
        sorted_metrics = [all_metrics[i] for i in sorted_indices[:num_to_plot]]
        sorted_errors = [all_errors[i] for i in sorted_indices[:num_to_plot]]
        sorted_colors = [all_colors[i] for i in sorted_indices[:num_to_plot]]

        bars = ax2.barh(sorted_metrics, sorted_errors, color=sorted_colors)
        for bar, error in zip(bars, sorted_errors):
            ax2.text(error + 0.01 * max(sorted_errors, default=1), bar.get_y() + bar.get_height()/2,
                   f"{error:.2%}", va='center', fontsize=8)

        # Add threshold lines
        for thresh_name, thresh_val in VALIDATION_THRESHOLDS.items():
             if thresh_name != 'good': # Don't plot 'good' threshold
                 color = status_map[thresh_name][0] if thresh_name in status_map else 'gray'
                 ax2.axvline(x=thresh_val, color=color, linestyle='--', alpha=0.7,
                           label=f"{thresh_name.replace('_',' ').title()} ({thresh_val:.0%})")

        _apply_common_ax_settings(ax2, xlabel='Relative Error / Deviation', title='Top Metric Errors/Deviations')
        ax2.legend(fontsize=8)
        ax2.xaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(xmax=1.0))
    else:
        ax2.text(0.5, 0.5, "No metric errors calculated", ha='center', va='center')
        _apply_common_ax_settings(ax2, title='Top Metric Errors/Deviations')


    # --- Display Validation Issues ---
    ax3 = fig.add_subplot(gs[1:, :]) # Span bottom row
    issues = validation_results.get('issues', [])

    if issues:
         # Sort issues by severity
        sorted_issues = sorted(issues, key=lambda x: status_map[x.get('status', 'unknown')][1], reverse=True)
        # Limit number of displayed issues
        num_issues_display = min(15, len(sorted_issues))
        display_issues = sorted_issues[:num_issues_display]

        issue_texts = [f"[{issue.get('status', 'unknown').upper()}] {issue.get('message', 'No details')}" for issue in display_issues]
        issue_colors = [status_map[issue.get('status', 'unknown')][0] for issue in display_issues]

        # Display issues as text with background color
        for i, (text, color) in enumerate(zip(issue_texts, issue_colors)):
            ax3.text(0.01, 0.95 - i * (0.9 / num_issues_display), text, color='black',
                     bbox=dict(facecolor=color, alpha=0.5, pad=3, boxstyle='round,pad=0.3'),
                     verticalalignment='top', fontsize=8, wrap=True)

        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1)
        _apply_common_ax_settings(ax3, title=f'Validation Issues (Top {num_issues_display})')
        ax3.set_xticks([])
        ax3.set_yticks([])
    else:
        ax3.text(0.5, 0.5, "No validation issues found",
               ha='center', va='center', fontsize=14, color='green',
               bbox=dict(facecolor='lightgreen', alpha=0.3, pad=10, boxstyle='round,pad=0.5'))
        _apply_common_ax_settings(ax3, title='Validation Issues')
        ax3.set_xticks([])
        ax3.set_yticks([])


    # Set overall title
    overall_status = validation_results.get('status', 'unknown').replace('_', ' ').title()
    fig.suptitle(f"Validation Summary - Overall Status: {overall_status}", fontsize=16)

    plt.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust for suptitle

    # Save if requested
    if save_path:
        save_plot(fig, save_path)

    return fig