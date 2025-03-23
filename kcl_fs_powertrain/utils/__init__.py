"""
Utility modules for Formula Student powertrain simulation.

This package provides utility modules for constants, plotting, and validation
used throughout the Formula Student powertrain simulation project.
"""

# Import key functions and objects for easier access
from .constants import (
    # Physical constants
    GRAVITY, AIR_DENSITY_SEA_LEVEL, WATER_DENSITY, AIR_VISCOSITY,
    WATER_SPECIFIC_HEAT, AIR_SPECIFIC_HEAT, AIR_GAS_CONSTANT,
    STANDARD_PRESSURE, STEFAN_BOLTZMANN, ABSOLUTE_ZERO_C,
    
    # Unit conversion functions
    celsius_to_kelvin, kelvin_to_celsius, 
    celsius_to_fahrenheit, fahrenheit_to_celsius,
    
    # Unit conversion factors
    KMH_TO_MS, MS_TO_KMH, MPH_TO_MS, MS_TO_MPH,
    KW_TO_HP, HP_TO_KW, NM_TO_LBFT, LBFT_TO_NM,
    KG_TO_LBS, LBS_TO_KG, PSI_TO_PA, PA_TO_PSI,
    INCH_TO_M, M_TO_INCH, MM_TO_M, M_TO_MM,
    DEG_TO_RAD, RAD_TO_DEG, LITERS_TO_M3, M3_TO_LITERS,
    GAL_TO_LITERS, LITERS_TO_GAL,
    
    # Formula Student reference values
    FS_MAX_TRACK_WIDTH, FS_ACCELERATION_LENGTH, FS_SKIDPAD_RADIUS,
    FS_ENDURANCE_TYPICAL_LENGTH, FS_AUTOCROSS_TYPICAL_LENGTH,
    
    # Vehicle reference values
    DEFAULT_TIRE_RADIUS, DEFAULT_WEIGHT_DISTRIBUTION, DEFAULT_CG_HEIGHT,
    DEFAULT_FRONTAL_AREA, DEFAULT_DRAG_COEFFICIENT, DEFAULT_LIFT_COEFFICIENT,
    
    # Engine reference values
    DEFAULT_REDLINE, DEFAULT_IDLE_RPM, DEFAULT_POWER_TO_WEIGHT,
    
    # Thermal reference values
    DEFAULT_AMBIENT_TEMP, DEFAULT_ENGINE_OPERATING_TEMP,
    DEFAULT_COOLANT_OPERATING_TEMP, DEFAULT_OIL_OPERATING_TEMP,
    DEFAULT_RADIATOR_EFFICIENCY, DEFAULT_THERMOSTAT_OPENING_TEMP,
    DEFAULT_THERMOSTAT_FULLY_OPEN_TEMP,
    
    # Transmission reference values
    DEFAULT_SHIFT_TIME, DEFAULT_WHEEL_SLIP_RATIO,
    
    # Enumerations
    EventType, TireType, EngineType, ThermalWarningLevel,
    
    # Scoring functions
    calculate_acceleration_score, calculate_skidpad_score,
    calculate_autocross_score, calculate_endurance_score,
    calculate_efficiency_score,
    
    # Fuel properties
    FuelProperties
)

# Import plotting functions
from .plotting import (
    set_plot_style, save_plot,
    plot_engine_performance, plot_vehicle_performance_summary,
    plot_torque_curves_comparison, plot_track_layout,
    plot_racing_line_analysis, plot_thermal_performance,
    plot_thermal_comparison, plot_cooling_system_map,
    plot_weight_sensitivity, plot_weight_distribution_sensitivity,
    plot_endurance_results, plot_endurance_comparison,
    plot_acceleration_results, plot_acceleration_comparison,
    plot_lap_time_results, plot_lap_time_comparison
)

# Import validation functions
from .validation import (
    validate_in_range, validate_theoretical_model,
    validate_acceleration_performance, validate_skidpad_performance,
    validate_lap_time_performance, validate_thermal_performance,
    validate_vehicle_specs, validate_full_vehicle_performance,
    compare_simulation_to_real_data, load_reference_data,
    save_validation_results, plot_validation_results,
    FS_PERFORMANCE_RANGES, VALIDATION_THRESHOLDS
)

# Define what is exported by default
__all__ = [
    # Constants
    'GRAVITY', 'AIR_DENSITY_SEA_LEVEL', 'WATER_DENSITY', 'AIR_VISCOSITY',
    'WATER_SPECIFIC_HEAT', 'AIR_SPECIFIC_HEAT', 'AIR_GAS_CONSTANT',
    'STANDARD_PRESSURE', 'STEFAN_BOLTZMANN', 'ABSOLUTE_ZERO_C',
    
    # Unit conversion functions
    'celsius_to_kelvin', 'kelvin_to_celsius', 
    'celsius_to_fahrenheit', 'fahrenheit_to_celsius',
    
    # Unit conversion factors
    'KMH_TO_MS', 'MS_TO_KMH', 'MPH_TO_MS', 'MS_TO_MPH',
    'KW_TO_HP', 'HP_TO_KW', 'NM_TO_LBFT', 'LBFT_TO_NM',
    'KG_TO_LBS', 'LBS_TO_KG', 'PSI_TO_PA', 'PA_TO_PSI',
    'INCH_TO_M', 'M_TO_INCH', 'MM_TO_M', 'M_TO_MM',
    'DEG_TO_RAD', 'RAD_TO_DEG', 'LITERS_TO_M3', 'M3_TO_LITERS',
    'GAL_TO_LITERS', 'LITERS_TO_GAL',
    
    # Enumerations
    'EventType', 'TireType', 'EngineType', 'ThermalWarningLevel',
    
    # Reference values
    'FS_MAX_TRACK_WIDTH', 'FS_ACCELERATION_LENGTH', 'FS_SKIDPAD_RADIUS',
    'FS_ENDURANCE_TYPICAL_LENGTH', 'FS_AUTOCROSS_TYPICAL_LENGTH',
    
    # Plot functions
    'set_plot_style', 'save_plot',
    'plot_engine_performance', 'plot_vehicle_performance_summary',
    'plot_torque_curves_comparison', 'plot_track_layout',
    'plot_racing_line_analysis', 'plot_thermal_performance',
    'plot_thermal_comparison', 'plot_cooling_system_map',
    'plot_weight_sensitivity', 'plot_weight_distribution_sensitivity', 
    'plot_endurance_results', 'plot_endurance_comparison',
    'plot_acceleration_results', 'plot_acceleration_comparison',
    'plot_lap_time_results', 'plot_lap_time_comparison',
    
    # Validation functions
    'validate_in_range', 'validate_theoretical_model',
    'validate_acceleration_performance', 'validate_skidpad_performance',
    'validate_lap_time_performance', 'validate_thermal_performance',
    'validate_vehicle_specs', 'validate_full_vehicle_performance',
    'compare_simulation_to_real_data', 'load_reference_data',
    'save_validation_results', 'plot_validation_results',
    'FS_PERFORMANCE_RANGES', 'VALIDATION_THRESHOLDS',
    
    # Score calculation functions
    'calculate_acceleration_score', 'calculate_skidpad_score',
    'calculate_autocross_score', 'calculate_endurance_score',
    'calculate_efficiency_score',
    
    # Complex objects
    'FuelProperties'
]