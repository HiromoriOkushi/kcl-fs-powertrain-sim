"""
Plotting utilities for Formula Student powertrain simulation.

This module provides a comprehensive set of plotting functions for visualizing
simulation results, vehicle performance metrics, and component behavior.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import os
import logging

# Import local modules
from ..utils.constants import MS_TO_KMH, MS_TO_MPH, KG_TO_LBS, KW_TO_HP, LITERS_TO_GAL

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("Plotting")


# Default style settings for plots
DEFAULT_FIG_SIZE = (12, 8)
DEFAULT_DPI = 300
DEFAULT_LINE_WIDTH = 2
DEFAULT_MARKER_SIZE = 6
DEFAULT_FONT_SIZE = 10
DEFAULT_TITLE_SIZE = 14
DEFAULT_LABEL_SIZE = 12
DEFAULT_LEGEND_SIZE = 10
DEFAULT_GRID_ALPHA = 0.3
DEFAULT_SAVE_FORMAT = 'png'

# Color schemes
COLOR_SCHEMES = {
    'default': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'],
    'formula_student': ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf'],
    'thermal': [(0.0, '#313695'), (0.25, '#4575b4'), (0.5, '#74add1'), (0.75, '#fdae61'), (1.0, '#d73027')],  # Blue to Red
    'speed': [(0.0, '#2c7bb6'), (0.5, '#ffffbf'), (1.0, '#d7191c')],  # Blue to Yellow to Red
    'acceleration': [(0.0, '#1a9641'), (0.5, '#ffffbf'), (1.0, '#d7191c')],  # Green to Yellow to Red
}

# Create custom colormaps
THERMAL_CMAP = LinearSegmentedColormap.from_list('thermal', COLOR_SCHEMES['thermal'])
SPEED_CMAP = LinearSegmentedColormap.from_list('speed', COLOR_SCHEMES['speed'])
ACCELERATION_CMAP = LinearSegmentedColormap.from_list('acceleration', COLOR_SCHEMES['acceleration'])


#------------------------------------------------------------------------------
# Utility functions
#------------------------------------------------------------------------------

def set_plot_style(style: str = 'default') -> None:
    """
    Set global matplotlib style for consistent plots.
    
    Args:
        style: Style name ('default', 'clean', 'presentation', 'publication')
    """
    if style == 'default':
        plt.style.use('default')
    elif style == 'clean':
        plt.style.use('seaborn-v0_8-whitegrid')
    elif style == 'presentation':
        plt.style.use('seaborn-v0_8-talk')
    elif style == 'publication':
        plt.style.use('seaborn-v0_8-paper')
    else:
        logger.warning(f"Unknown style: {style}. Using default.")
        plt.style.use('default')
    
    # Set common parameters
    plt.rcParams['font.size'] = DEFAULT_FONT_SIZE
    plt.rcParams['axes.titlesize'] = DEFAULT_TITLE_SIZE
    plt.rcParams['axes.labelsize'] = DEFAULT_LABEL_SIZE
    plt.rcParams['xtick.labelsize'] = DEFAULT_FONT_SIZE
    plt.rcParams['ytick.labelsize'] = DEFAULT_FONT_SIZE
    plt.rcParams['legend.fontsize'] = DEFAULT_LEGEND_SIZE
    plt.rcParams['figure.figsize'] = DEFAULT_FIG_SIZE
    plt.rcParams['figure.dpi'] = DEFAULT_DPI
    plt.rcParams['lines.linewidth'] = DEFAULT_LINE_WIDTH
    plt.rcParams['lines.markersize'] = DEFAULT_MARKER_SIZE
    plt.rcParams['grid.alpha'] = DEFAULT_GRID_ALPHA


def save_plot(fig: plt.Figure, filename: str, directory: Optional[str] = None,
             format: str = DEFAULT_SAVE_FORMAT, dpi: int = DEFAULT_DPI) -> str:
    """
    Save a plot to file with proper directory handling.
    
    Args:
        fig: Matplotlib figure to save
        filename: Base filename (without extension)
        directory: Directory to save in (created if doesn't exist)
        format: File format ('png', 'pdf', 'svg', etc.)
        dpi: Resolution for raster formats
        
    Returns:
        Full path to saved file
    """
    # Process filename
    if '.' in filename:
        base, ext = os.path.splitext(filename)
        if ext[1:].lower() != format.lower():
            logger.warning(f"Filename extension ({ext}) doesn't match format ({format}). Using {format}.")
            filename = base
    
    # Ensure directory exists
    if directory:
        os.makedirs(directory, exist_ok=True)
        filepath = os.path.join(directory, f"{filename}.{format}")
    else:
        filepath = f"{filename}.{format}"
    
    # Save the figure
    fig.savefig(filepath, format=format, dpi=dpi, bbox_inches='tight')
    logger.info(f"Plot saved to {filepath}")
    
    return filepath


def _format_metric_name(metric: str) -> str:
    """Format metric name for display in plots."""
    if metric == 'time_to_60mph':
        return '0-60 mph'
    elif metric == 'time_to_100kph':
        return '0-100 km/h'
    elif metric == 'finish_time':
        return '75m Time'
    else:
        return metric.replace('_', ' ').title()


#------------------------------------------------------------------------------
# Engine plotting functions
#------------------------------------------------------------------------------

def plot_engine_performance(engine_data: Dict, title: Optional[str] = None,
                          unit_system: str = 'metric', show_efficiency: bool = False,
                          save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot engine performance curves (torque, power, efficiency).
    
    Args:
        engine_data: Dictionary with RPM, torque, and power data
        title: Plot title
        unit_system: Unit system ('metric' or 'imperial')
        show_efficiency: Whether to show efficiency curves
        save_path: Path to save plot (if None, not saved)
        
    Returns:
        Matplotlib figure
    """
    # Extract data
    rpm = engine_data.get('rpm', [])
    torque = engine_data.get('torque', [])
    power = engine_data.get('power', [])
    
    if not rpm or len(rpm) != len(torque) or len(rpm) != len(power):
        logger.error("Invalid engine data format")
        return None
    
    # Create figure
    fig, ax1 = plt.subplots(figsize=DEFAULT_FIG_SIZE)
    
    # Unit conversions
    if unit_system.lower() == 'imperial':
        torque_factor = 0.7376  # Convert Nm to lb-ft
        power_factor = KW_TO_HP  # Convert kW to HP
        torque_unit = "lb-ft"
        power_unit = "HP"
    else:
        torque_factor = 1.0
        power_factor = 1.0
        torque_unit = "Nm"
        power_unit = "kW"
    
    # Plot torque curve
    torque_line, = ax1.plot(rpm, np.array(torque) * torque_factor, 'b-', 
                         linewidth=DEFAULT_LINE_WIDTH, label=f"Torque ({torque_unit})")
    ax1.set_xlabel('Engine Speed (RPM)')
    ax1.set_ylabel(f'Torque ({torque_unit})')
    
    # Twin axis for power
    ax2 = ax1.twinx()
    power_line, = ax2.plot(rpm, np.array(power) * power_factor, 'r-', 
                        linewidth=DEFAULT_LINE_WIDTH, label=f"Power ({power_unit})")
    ax2.set_ylabel(f'Power ({power_unit})')
    
    # Plot efficiency if requested
    if show_efficiency and 'efficiency' in engine_data:
        efficiency = engine_data.get('efficiency', [])
        if len(efficiency) == len(rpm):
            ax3 = ax1.twinx()
            # Offset the axis
            ax3.spines['right'].set_position(('outward', 60))
            
            efficiency_line, = ax3.plot(rpm, efficiency, 'g-', 
                                     linewidth=DEFAULT_LINE_WIDTH, label="Efficiency (%)")
            ax3.set_ylabel('Efficiency (%)')
            ax3.set_ylim(0, 100)
    
    # Combine legends
    lines = [torque_line, power_line]
    labels = [torque_line.get_label(), power_line.get_label()]
    
    if show_efficiency and 'efficiency' in engine_data and len(efficiency) == len(rpm):
        lines.append(efficiency_line)
        labels.append(efficiency_line.get_label())
    
    ax1.legend(lines, labels, loc='best')
    
    # Set title
    if title:
        plt.title(title)
    else:
        plt.title('Engine Performance Curves')
    
    # Add grid
    ax1.grid(True, alpha=DEFAULT_GRID_ALPHA)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        save_plot(fig, save_path)
    
    return fig


#------------------------------------------------------------------------------
# Vehicle performance summary
#------------------------------------------------------------------------------

def plot_vehicle_performance_summary(vehicle_data: Dict, title: Optional[str] = None,
                                   unit_system: str = 'metric',
                                   save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot a comprehensive vehicle performance summary.
    
    Args:
        vehicle_data: Dictionary with vehicle performance data
        title: Plot title
        unit_system: Unit system ('metric' or 'imperial')
        save_path: Path to save plot (if None, not saved)
        
    Returns:
        Matplotlib figure
    """
    # Create figure
    fig = plt.figure(figsize=(15, 12))
    gs = gridspec.GridSpec(3, 2, height_ratios=[2, 1, 1])
    
    # Unit conversions
    if unit_system.lower() == 'imperial':
        speed_factor = MS_TO_MPH  # Convert m/s to mph
        speed_unit = "mph"
        distance_factor = 3.28084  # Convert meters to feet
        distance_unit = "ft"
        mass_factor = KG_TO_LBS  # Convert kg to lbs
        mass_unit = "lbs"
        power_factor = KW_TO_HP  # Convert kW to HP
        power_unit = "HP"
    else:
        speed_factor = MS_TO_KMH  # Convert m/s to km/h
        speed_unit = "km/h"
        distance_factor = 1.0
        distance_unit = "m"
        mass_factor = 1.0
        mass_unit = "kg"
        power_factor = 1.0
        power_unit = "kW"
    
    # Extract data
    acceleration_data = vehicle_data.get('acceleration', {})
    skidpad_data = vehicle_data.get('skidpad', {})
    lap_data = vehicle_data.get('lap', {})
    thermal_data = vehicle_data.get('thermal', {})
    specs = vehicle_data.get('specs', {})
    
    # Plot acceleration metrics
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Extract acceleration metrics
    accel_metrics = []
    accel_values = []
    
    if 'time_to_60mph' in acceleration_data:
        accel_metrics.append('0-60 mph')
        accel_values.append(acceleration_data['time_to_60mph'])
    
    if 'time_to_100kph' in acceleration_data:
        accel_metrics.append('0-100 km/h')
        accel_values.append(acceleration_data['time_to_100kph'])
    
    if 'finish_time' in acceleration_data:
        accel_metrics.append('75m Time')
        accel_values.append(acceleration_data['finish_time'])
    
    # Add additional acceleration metrics if available
    if 'time_to_30mph' in acceleration_data:
        accel_metrics.append('0-30 mph')
        accel_values.append(acceleration_data['time_to_30mph'])
    
    # Plot horizontal bars
    if accel_metrics:
        # Sort by value (fastest first for each metric)
        sorted_indices = np.argsort(accel_values)
        sorted_metrics = [accel_metrics[i] for i in sorted_indices]
        sorted_values = [accel_values[i] for i in sorted_indices]
        
        # Create bar chart
        bars = ax1.barh(sorted_metrics, sorted_values, color='blue')
        
        # Add time labels
        for bar, time in zip(bars, sorted_values):
            ax1.text(time + 0.05, bar.get_y() + bar.get_height()/2, 
                   f'{time:.2f}s', va='center')
        
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Metric')
        ax1.set_title('Acceleration Performance')
        ax1.grid(True, alpha=DEFAULT_GRID_ALPHA, axis='x')
    
    # Plot cornering and handling metrics
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Extract cornering metrics
    cornering_metrics = []
    cornering_values = []
    cornering_units = []
    
    if 'lateral_acceleration' in skidpad_data:
        cornering_metrics.append('Lateral Accel')
        cornering_values.append(skidpad_data['lateral_acceleration'])
        cornering_units.append('g')
    
    if 'skidpad_time' in skidpad_data:
        cornering_metrics.append('Skidpad Time')
        cornering_values.append(skidpad_data['skidpad_time'])
        cornering_units.append('s')
    
    if 'min_radius' in skidpad_data:
        radius = skidpad_data['min_radius']
        if unit_system.lower() == 'imperial':
            radius = radius * 3.28084  # Convert to feet
            cornering_metrics.append('Min Turn Radius')
            cornering_values.append(radius)
            cornering_units.append('ft')
        else:
            cornering_metrics.append('Min Turn Radius')
            cornering_values.append(radius)
            cornering_units.append('m')
    
    # Add lap time metrics if available
    if 'lap_time' in lap_data:
        cornering_metrics.append('Lap Time')
        cornering_values.append(lap_data['lap_time'])
        cornering_units.append('s')
    
    if 'avg_speed' in lap_data:
        cornering_metrics.append('Avg Speed')
        cornering_values.append(lap_data['avg_speed'] * speed_factor)
        cornering_units.append(speed_unit)
    
    # Plot horizontal bars
    if cornering_metrics:
        # For a cleaner chart, only show the first 5 metrics
        if len(cornering_metrics) > 5:
            cornering_metrics = cornering_metrics[:5]
            cornering_values = cornering_values[:5]
            cornering_units = cornering_units[:5]
        
        # Create bar chart
        bars = ax2.barh(cornering_metrics, cornering_values, color='green')
        
        # Add value labels with units
        for bar, value, unit in zip(bars, cornering_values, cornering_units):
            ax2.text(value + 0.05 * max(cornering_values), bar.get_y() + bar.get_height()/2, 
                   f'{value:.2f} {unit}', va='center')
        
        ax2.set_xlabel('Value')
        ax2.set_ylabel('Metric')
        ax2.set_title('Cornering and Handling Performance')
        ax2.grid(True, alpha=DEFAULT_GRID_ALPHA, axis='x')
    
    # Plot vehicle specifications
    ax3 = fig.add_subplot(gs[1, 0])
    
    # Extract key specifications
    spec_names = []
    spec_values = []
    spec_units = []
    
    if 'mass' in specs:
        mass = specs['mass'] * mass_factor
        spec_names.append('Mass')
        spec_values.append(mass)
        spec_units.append(mass_unit)
    
    if 'power' in specs:
        power = specs['power'] * power_factor
        spec_names.append('Power')
        spec_values.append(power)
        spec_units.append(power_unit)
    
    if 'torque' in specs:
        torque = specs['torque']
        if unit_system.lower() == 'imperial':
            torque = torque * 0.7376  # Convert to lb-ft
            spec_names.append('Torque')
            spec_values.append(torque)
            spec_units.append('lb-ft')
        else:
            spec_names.append('Torque')
            spec_values.append(torque)
            spec_units.append('Nm')
    
    if 'power_to_weight' in specs:
        ptw = specs['power_to_weight']
        if unit_system.lower() == 'imperial':
            # Convert kW/kg to HP/lb
            ptw = ptw * KW_TO_HP / KG_TO_LBS
            spec_names.append('Power-to-Weight')
            spec_values.append(ptw)
            spec_units.append('HP/lb')
        else:
            spec_names.append('Power-to-Weight')
            spec_values.append(ptw)
            spec_units.append('kW/kg')
    
    if 'weight_distribution' in specs:
        wd = specs['weight_distribution'] * 100  # Convert to percentage
        spec_names.append('Front Weight')
        spec_values.append(wd)
        spec_units.append('%')
    
    # Plot horizontal bars
    if spec_names:
        # Create bar chart
        bars = ax3.barh(spec_names, spec_values, color='purple')
        
        # Add value labels with units
        for bar, value, unit in zip(bars, spec_values, spec_units):
            ax3.text(value + 0.05 * max(spec_values), bar.get_y() + bar.get_height()/2, 
                   f'{value:.2f} {unit}', va='center')
        
        ax3.set_xlabel('Value')
        ax3.set_ylabel('Specification')
        ax3.set_title('Vehicle Specifications')
        ax3.grid(True, alpha=DEFAULT_GRID_ALPHA, axis='x')
    
    # Plot thermal performance
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Extract thermal metrics
    thermal_metrics = []
    thermal_values = []
    thermal_units = []
    
    if 'max_engine_temp' in thermal_data:
        temp = thermal_data['max_engine_temp']
        if unit_system.lower() == 'imperial':
            temp = temp * 9/5 + 32
            thermal_metrics.append('Max Engine Temp')
            thermal_values.append(temp)
            thermal_units.append('°F')
        else:
            thermal_metrics.append('Max Engine Temp')
            thermal_values.append(temp)
            thermal_units.append('°C')
    
    if 'max_coolant_temp' in thermal_data:
        temp = thermal_data['max_coolant_temp']
        if unit_system.lower() == 'imperial':
            temp = temp * 9/5 + 32
            thermal_metrics.append('Max Coolant Temp')
            thermal_values.append(temp)
            thermal_units.append('°F')
        else:
            thermal_metrics.append('Max Coolant Temp')
            thermal_values.append(temp)
            thermal_units.append('°C')
    
    if 'cooling_capacity' in thermal_data:
        thermal_metrics.append('Cooling Capacity')
        thermal_values.append(thermal_data['cooling_capacity'] / 1000)  # Convert to kW
        thermal_units.append('kW')
    
    if 'fan_duty_avg' in thermal_data:
        thermal_metrics.append('Avg Fan Duty')
        thermal_values.append(thermal_data['fan_duty_avg'] * 100)  # Convert to percentage
        thermal_units.append('%')
    
    # Plot horizontal bars
    if thermal_metrics:
        # Create bar chart
        bars = ax4.barh(thermal_metrics, thermal_values, color='red')
        
        # Add value labels with units
        for bar, value, unit in zip(bars, thermal_values, thermal_units):
            ax4.text(value + 0.05 * max(thermal_values), bar.get_y() + bar.get_height()/2, 
                   f'{value:.1f} {unit}', va='center')
        
        ax4.set_xlabel('Value')
        ax4.set_ylabel('Metric')
        ax4.set_title('Thermal Performance')
        ax4.grid(True, alpha=DEFAULT_GRID_ALPHA, axis='x')
    
    # Plot event scores
    ax5 = fig.add_subplot(gs[2, 0])
    
    # Extract event scores
    events = []
    scores = []
    max_scores = []
    
    if 'acceleration_score' in vehicle_data:
        events.append('Acceleration')
        scores.append(vehicle_data['acceleration_score'])
        max_scores.append(vehicle_data.get('max_acceleration_score', 75))
    
    if 'skidpad_score' in vehicle_data:
        events.append('Skidpad')
        scores.append(vehicle_data['skidpad_score'])
        max_scores.append(vehicle_data.get('max_skidpad_score', 75))
    
    if 'autocross_score' in vehicle_data:
        events.append('Autocross')
        scores.append(vehicle_data['autocross_score'])
        max_scores.append(vehicle_data.get('max_autocross_score', 100))
    
    if 'endurance_score' in vehicle_data:
        events.append('Endurance')
        scores.append(vehicle_data['endurance_score'])
        max_scores.append(vehicle_data.get('max_endurance_score', 325))
    
    if 'efficiency_score' in vehicle_data:
        events.append('Efficiency')
        scores.append(vehicle_data['efficiency_score'])
        max_scores.append(vehicle_data.get('max_efficiency_score', 100))
    
    # Plot horizontal bars
    if events:
        # Create bar chart with percentages of max score
        percentages = [s/m*100 for s, m in zip(scores, max_scores)]
        
        bars = ax5.barh(events, percentages, color='orange')
        
        # Add score labels
        for bar, score, max_score in zip(bars, scores, max_scores):
            ax5.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                   f'{score:.1f}/{max_score}', va='center')
        
        ax5.set_xlabel('Percentage of Maximum Score')
        ax5.set_ylabel('Event')
        ax5.set_title('Event Scores')
        ax5.grid(True, alpha=DEFAULT_GRID_ALPHA, axis='x')
        ax5.set_xlim(0, 105)  # Limit to 0-100% with room for labels
    
    # Plot radar chart of key metrics
    ax6 = fig.add_subplot(gs[2, 1], polar=True)
    
    # Define metrics for radar chart
    radar_metrics = ['Acceleration', 'Top Speed', 'Cornering', 'Braking', 'Efficiency']
    radar_values = [0.0, 0.0, 0.0, 0.0, 0.0]  # Default zeros
    
    # Extract normalized values (0-1 scale)
    
    # Acceleration (lower is better, so invert)
    if 'time_to_60mph' in acceleration_data:
        # Normalize: 2.5s->1.0, 5s->0.5, 10s->0.0
        time_60 = acceleration_data['time_to_60mph']
        radar_values[0] = max(0, min(1, (10 - time_60) / 7.5))
    
    # Top Speed
    if 'max_speed' in specs:
        # Normalize: 150km/h->0.5, 200km/h->0.75, 250km/h->1.0
        max_speed = specs['max_speed'] * speed_factor
        radar_values[1] = max(0, min(1, max_speed / 250))
    
    # Cornering
    if 'lateral_acceleration' in skidpad_data:
        # Normalize: 1g->0.5, 1.5g->0.75, 2g->1.0
        lat_g = skidpad_data['lateral_acceleration']
        radar_values[2] = max(0, min(1, lat_g / 2))
    
    # Braking
    if 'max_deceleration' in specs:
        # Normalize: 1g->0.5, 1.5g->0.75, 2g->1.0
        decel_g = abs(specs['max_deceleration']) / 9.81
        radar_values[3] = max(0, min(1, decel_g / 2))
    
    # Efficiency
    if 'efficiency_score' in vehicle_data and 'max_efficiency_score' in vehicle_data:
        # Normalize based on score percentage
        radar_values[4] = vehicle_data['efficiency_score'] / vehicle_data['max_efficiency_score']
    
    # Set up radar chart
    angles = np.linspace(0, 2*np.pi, len(radar_metrics), endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    radar_values += radar_values[:1]  # Close the loop
    
    ax6.fill(angles, radar_values, color='blue', alpha=0.25)
    ax6.plot(angles, radar_values, 'o-', linewidth=2, color='blue')
    
    # Set radar chart labels
    ax6.set_xticks(angles[:-1])
    ax6.set_xticklabels(radar_metrics)
    
    # Set radar chart grid and limits
    ax6.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax6.set_yticklabels(['0.25', '0.5', '0.75', '1.0'])
    ax6.set_ylim(0, 1)
    
    ax6.set_title('Performance Radar')
    
    # Set overall title
    if title:
        fig.suptitle(title, fontsize=DEFAULT_TITLE_SIZE+2)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    # Save if requested
    if save_path:
        save_plot(fig, save_path)
    
    return fig


#------------------------------------------------------------------------------
# Main function
#------------------------------------------------------------------------------

if __name__ == "__main__":
    # Set default plot style
    set_plot_style('clean')
    
    # Example usage (when module is run directly)
    logger.info("Running plotting module example...")
    
    # Create dummy engine data
    rpm = np.arange(1000, 15000, 100)
    torque = 70 * np.exp(-((rpm - 9000) / 4000) ** 2) + 10
    power = torque * rpm / 9549  # Convert to kW
    
    engine_data = {
        'rpm': rpm,
        'torque': torque,
        'power': power
    }
    
    # Plot engine performance
    fig = plot_engine_performance(engine_data, title="Honda CBR600F4i Engine Performance", 
                                unit_system='metric')
    save_plot(fig, "example_engine_performance")
    
    logger.info("Example engine performance plot saved as 'example_engine_performance.png'")
    
    logger.info("Plotting module ready to use. Import the module and call the appropriate functions.")


def plot_torque_curves_comparison(curves_data: List[Dict], title: Optional[str] = None,
                                unit_system: str = 'metric', 
                                save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot comparison of multiple torque curves.
    
    Args:
        curves_data: List of dictionaries with curve data and labels
        title: Plot title
        unit_system: Unit system ('metric' or 'imperial')
        save_path: Path to save plot (if None, not saved)
        
    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Unit conversions
    if unit_system.lower() == 'imperial':
        torque_factor = 0.7376  # Convert Nm to lb-ft
        power_factor = KW_TO_HP  # Convert kW to HP
        torque_unit = "lb-ft"
        power_unit = "HP"
    else:
        torque_factor = 1.0
        power_factor = 1.0
        torque_unit = "Nm"
        power_unit = "kW"
    
    # Colors for different curves
    colors = COLOR_SCHEMES['formula_student']
    
    # Plot each curve
    for i, curve in enumerate(curves_data):
        rpm = curve.get('rpm', [])
        torque = curve.get('torque', [])
        power = curve.get('power', [])
        label = curve.get('label', f'Curve {i+1}')
        color = curve.get('color', colors[i % len(colors)])
        
        if len(rpm) != len(torque) or len(rpm) != len(power):
            logger.warning(f"Skipping curve {i+1} due to data length mismatch")
            continue
        
        # Plot torque
        ax1.plot(rpm, np.array(torque) * torque_factor, '-', 
               color=color, linewidth=DEFAULT_LINE_WIDTH, label=label)
        
        # Plot power
        ax2.plot(rpm, np.array(power) * power_factor, '-', 
               color=color, linewidth=DEFAULT_LINE_WIDTH, label=label)
    
    # Set labels and grid
    ax1.set_xlabel('Engine Speed (RPM)')
    ax1.set_ylabel(f'Torque ({torque_unit})')
    ax1.grid(True, alpha=DEFAULT_GRID_ALPHA)
    ax1.legend(loc='best')
    
    # Plot lap time comparison bar chart
    ax2 = fig.add_subplot(gs[1, 0])
    bars = ax2.bar(config_names, lap_times, color='teal')
    
    # Add time labels
    for bar, time_val in zip(bars, lap_times):
        ax2.text(bar.get_x() + bar.get_width()/2., time_val + 0.05,
               f'{time_val:.2f}s', ha='center', va='bottom', fontsize=8)
    
    ax2.set_xlabel('Configuration')
    ax2.set_ylabel('Lap Time (s)')
    ax2.set_title('Lap Time Comparison')
    ax2.grid(True, alpha=DEFAULT_GRID_ALPHA, axis='y')
    ax2.set_xticklabels(config_names, rotation=45, ha='right')
    
    # Plot average speed comparison bar chart
    ax3 = fig.add_subplot(gs[1, 1])
    bars = ax3.bar(config_names, avg_speeds, color='blue')
    
    # Add speed labels
    for bar, speed_val in zip(bars, avg_speeds):
        ax3.text(bar.get_x() + bar.get_width()/2., speed_val + 0.5,
               f'{speed_val:.1f}', ha='center', va='bottom', fontsize=8)
    
    ax3.set_xlabel('Configuration')
    ax3.set_ylabel(f'Average Speed ({speed_unit})')
    ax3.set_title('Average Speed Comparison')
    ax3.grid(True, alpha=DEFAULT_GRID_ALPHA, axis='y')
    ax3.set_xticklabels(config_names, rotation=45, ha='right')
    
    # Plot maximum speed comparison bar chart
    ax4 = fig.add_subplot(gs[2, 0])
    bars = ax4.bar(config_names, max_speeds, color='red')
    
    # Add speed labels
    for bar, speed_val in zip(bars, max_speeds):
        ax4.text(bar.get_x() + bar.get_width()/2., speed_val + 0.5,
               f'{speed_val:.1f}', ha='center', va='bottom', fontsize=8)
    
    ax4.set_xlabel('Configuration')
    ax4.set_ylabel(f'Maximum Speed ({speed_unit})')
    ax4.set_title('Maximum Speed Comparison')
    ax4.grid(True, alpha=DEFAULT_GRID_ALPHA, axis='y')
    ax4.set_xticklabels(config_names, rotation=45, ha='right')
    
    # Plot track layout with racing lines if available
    ax5 = fig.add_subplot(gs[2, 1])
    track_plotted = False
    
    for i, data in enumerate(comparison_data):
        if 'track_points' in data and not track_plotted:
            track_points = data['track_points']
            ax5.plot(track_points[:, 0], track_points[:, 1], 'k-', 
                   alpha=0.5, linewidth=1, label='Track')
            track_plotted = True
        
        if 'racing_line' in data:
            racing_line = data['racing_line']
            color = data.get('color', colors[i % len(colors)])
            ax5.plot(racing_line[:, 0], racing_line[:, 1], '-', 
                   color=color, linewidth=2, label=data.get('label', f'Config {i+1}'))
    
    if track_plotted:
        # Set equal aspect ratio
        ax5.set_aspect('equal')
        ax5.set_xlabel('X (m)')
        ax5.set_ylabel('Y (m)')
        ax5.set_title('Racing Line Comparison')
        ax5.legend(loc='best')
    else:
        # If no track data, create a different plot (e.g., improvement percentages)
        # Calculate improvement percentage relative to first config
        if lap_times and lap_times[0] > 0:
            improvements = [(lap_times[0] - t) / lap_times[0] * 100 for t in lap_times]
            bars = ax5.bar(config_names, improvements, color='green')
            
            for bar, imp in zip(bars, improvements):
                ax5.text(bar.get_x() + bar.get_width()/2., imp + 0.1,
                       f'{imp:.2f}%', ha='center', va='bottom', fontsize=8)
            
            ax5.set_xlabel('Configuration')
            ax5.set_ylabel('Improvement (%)')
            ax5.set_title('Lap Time Improvement Relative to Baseline')
            ax5.grid(True, alpha=DEFAULT_GRID_ALPHA, axis='y')
            ax5.set_xticklabels(config_names, rotation=45, ha='right')
    
    # Set overall title
    if title:
        fig.suptitle(title, fontsize=DEFAULT_TITLE_SIZE+2)
    else:
        fig.suptitle('Lap Time Performance Comparison', fontsize=DEFAULT_TITLE_SIZE+2)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    # Save if requested
    if save_path:
        save_plot(fig, save_path)
    
    return fig


#------------------------------------------------------------------------------
# Track visualization functions
#------------------------------------------------------------------------------

def plot_track_layout(track_data: Dict, show_racing_line: bool = True,
                    show_segments: bool = True, show_elevation: bool = False,
                    title: Optional[str] = None, save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot track layout with optional racing line and segments.
    
    Args:
        track_data: Dictionary with track data
        show_racing_line: Whether to show racing line
        show_segments: Whether to show track segments
        show_elevation: Whether to show elevation profile
        title: Plot title
        save_path: Path to save plot (if None, not saved)
        
    Returns:
        Matplotlib figure
    """
    # Extract track data
    track_points = track_data.get('points', [])
    track_width = track_data.get('width', [])
    racing_line = track_data.get('racing_line', [])
    segments = track_data.get('segments', [])
    elevation = track_data.get('elevation', [])
    
    # Validate data
    if not track_points or len(track_points) < 2:
        logger.error("Invalid track data format")
        return None
    
    # Create figure
    if show_elevation:
        fig = plt.figure(figsize=(15, 12))
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
        ax1 = fig.add_subplot(gs[0])
    else:
        fig = plt.figure(figsize=(12, 10))
        ax1 = plt.gca()
    
    # Plot track centerline
    track_points = np.array(track_points)
    ax1.plot(track_points[:, 0], track_points[:, 1], 'k-', alpha=0.7, linewidth=1.5, label='Track Centerline')
    
    # Plot track boundaries if width is available
    if track_width:
        if np.isscalar(track_width):
            track_width = np.full(len(track_points), track_width)
        
        # Calculate normal vectors
        normals = np.zeros_like(track_points)
        for i in range(len(track_points)):
            # Get adjacent points (with wraparound for closed circuits)
            prev_idx = (i - 1) % len(track_points)
            next_idx = (i + 1) % len(track_points)
            
            # Calculate tangent vector
            tangent = track_points[next_idx] - track_points[prev_idx]
            
            # Normalize
            if np.linalg.norm(tangent) > 1e-6:
                tangent = tangent / np.linalg.norm(tangent)
            
            # Calculate normal (90 degree rotation)
            normals[i] = np.array([-tangent[1], tangent[0]])
        
        # Calculate left and right boundaries
        left_boundary = track_points + normals * np.column_stack((track_width, track_width)) / 2
        right_boundary = track_points - normals * np.column_stack((track_width, track_width)) / 2
        
        # Plot boundaries
        ax1.plot(left_boundary[:, 0], left_boundary[:, 1], 'k-', alpha=0.3, linewidth=1)
        ax1.plot(right_boundary[:, 0], right_boundary[:, 1], 'k-', alpha=0.3, linewidth=1)
    
    # Plot racing line if requested
    if show_racing_line and racing_line and len(racing_line) > 1:
        racing_line = np.array(racing_line)
        ax1.plot(racing_line[:, 0], racing_line[:, 1], 'r-', 
               linewidth=2, label='Racing Line')
    
    # Plot segments if requested
    if show_segments and segments:
        segment_colors = {
            'straight': 'green',
            'left_corner': 'blue',
            'right_corner': 'red',
            'chicane': 'purple',
            'hairpin': 'orange'
        }
        
        # Map segment types to colors
        segment_patches = []
        for segment in segments:
            segment_type = segment.get('type', 'straight')
            color = segment_colors.get(segment_type, 'gray')
            
            start_idx = segment.get('start_idx', 0)
            end_idx = segment.get('end_idx', 0)
            
            # Ensure indices are valid
            if start_idx < 0 or end_idx >= len(track_points) or start_idx > end_idx:
                continue
            
            # Get segment points
            segment_points = track_points[start_idx:end_idx+1]
            
            # Plot segment with appropriate color
            line = ax1.plot(segment_points[:, 0], segment_points[:, 1], '-', 
                          color=color, linewidth=3, alpha=0.7,
                          label=f"{segment_type.replace('_', ' ').title()}")
            
            # Store for legend
            if line and segment_type not in [p.get_label() for p in segment_patches]:
                segment_patches.append(line[0])
    
    # Add start/finish marker
    if 'start_position' in track_data:
        start_pos = track_data['start_position']
        ax1.scatter(start_pos[0], start_pos[1], color='green', marker='o', s=100, label='Start/Finish')
        
        # Add direction arrow if available
        if 'start_direction' in track_data:
            start_dir = track_data['start_direction']
            dir_vec = np.array([np.cos(start_dir), np.sin(start_dir)])
            ax1.arrow(start_pos[0], start_pos[1], dir_vec[0] * 10, dir_vec[1] * 10, 
                    head_width=3, head_length=5, fc='green', ec='green')
    
    # Set equal aspect ratio and grid
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=DEFAULT_GRID_ALPHA)
    
    # Set title and labels
    if 'name' in track_data:
        track_name = track_data['name']
        if 'length' in track_data:
            track_length = track_data['length']
            ax1_title = f"{track_name} (Length: {track_length:.1f}m)"
        else:
            ax1_title = track_name
    else:
        ax1_title = "Track Layout"
    
    ax1.set_title(ax1_title)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    
    # Create legend
    ax1.legend(loc='best')
    
    # Plot elevation profile if requested
    if show_elevation and elevation and len(elevation) == len(track_points):
        ax2 = fig.add_subplot(gs[1])
        
        # Calculate distance along track
        distances = np.zeros(len(track_points))
        for i in range(1, len(track_points)):
            distances[i] = distances[i-1] + np.linalg.norm(track_points[i] - track_points[i-1])
        
        # Plot elevation
        ax2.plot(distances, elevation, 'g-', linewidth=DEFAULT_LINE_WIDTH)
        
        # Calculate elevation gain/loss
        elevation_diff = np.diff(elevation)
        elevation_gain = np.sum(elevation_diff[elevation_diff > 0])
        elevation_loss = np.sum(elevation_diff[elevation_diff < 0])
        
        # Add elevation info
        info_text = f"Elevation Gain: {elevation_gain:.1f}m, Loss: {abs(elevation_loss):.1f}m"
        ax2.text(0.5, 0.9, info_text, transform=ax2.transAxes, ha='center')
        
        ax2.set_xlabel('Distance (m)')
        ax2.set_ylabel('Elevation (m)')
        ax2.set_title('Elevation Profile')
        ax2.grid(True, alpha=DEFAULT_GRID_ALPHA)
    
    # Set overall title
    if title:
        if show_elevation:
            plt.suptitle(title, fontsize=DEFAULT_TITLE_SIZE+2)
        else:
            ax1.set_title(title)
    
    # Adjust layout
    plt.tight_layout()
    if show_elevation:
        plt.subplots_adjust(top=0.92)
    
    # Save if requested
    if save_path:
        save_plot(fig, save_path)
    
    return fig


def plot_racing_line_analysis(racing_line_data: Dict, title: Optional[str] = None,
                            unit_system: str = 'metric', 
                            save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot racing line analysis including curvature and speed profiles.
    
    Args:
        racing_line_data: Dictionary with racing line data
        title: Plot title
        unit_system: Unit system ('metric' or 'imperial')
        save_path: Path to save plot (if None, not saved)
        
    Returns:
        Matplotlib figure
    """
    # Extract racing line data
    racing_line = racing_line_data.get('line', [])
    distances = racing_line_data.get('distances', [])
    curvature = racing_line_data.get('curvature', [])
    speed_profile = racing_line_data.get('speed_profile', [])
    time_profile = racing_line_data.get('time_profile', [])
    
    # Extract track data if available
    track_points = racing_line_data.get('track_points', [])
    track_width = racing_line_data.get('track_width', [])
    
    # Validate data
    if not racing_line or len(racing_line) < 2:
        logger.error("Invalid racing line data format")
        return None
    
    # Create figure
    fig = plt.figure(figsize=(15, 12))
    gs = gridspec.GridSpec(3, 2, height_ratios=[2, 1, 1])
    
    # Unit conversions
    if unit_system.lower() == 'imperial':
        speed_factor = MS_TO_MPH  # Convert m/s to mph
        speed_unit = "mph"
        distance_factor = 3.28084 / 5280  # Convert meters to miles
        distance_unit = "miles"
    else:
        speed_factor = MS_TO_KMH  # Convert m/s to km/h
        speed_unit = "km/h"
        distance_factor = 0.001  # Convert meters to kilometers
        distance_unit = "km"
    
    # Plot track layout with racing line
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Convert lists to numpy arrays if they aren't already
    racing_line = np.array(racing_line)
    
    # Plot track if available
    if len(track_points) > 1:
        track_points = np.array(track_points)
        ax1.plot(track_points[:, 0], track_points[:, 1], 'k-', 
               alpha=0.3, linewidth=1, label='Track Centerline')
        
        # Plot track boundaries if width is available
        if track_width:
            if np.isscalar(track_width):
                track_width = np.full(len(track_points), track_width)
            
            # Calculate normal vectors
            normals = np.zeros_like(track_points)
            for i in range(len(track_points)):
                # Get adjacent points (with wraparound for closed circuits)
                prev_idx = (i - 1) % len(track_points)
                next_idx = (i + 1) % len(track_points)
                
                # Calculate tangent vector
                tangent = track_points[next_idx] - track_points[prev_idx]
                
                # Normalize
                if np.linalg.norm(tangent) > 1e-6:
                    tangent = tangent / np.linalg.norm(tangent)
                
                # Calculate normal (90 degree rotation)
                normals[i] = np.array([-tangent[1], tangent[0]])
            
            # Calculate left and right boundaries
            left_boundary = track_points + normals * np.column_stack((track_width, track_width)) / 2
            right_boundary = track_points - normals * np.column_stack((track_width, track_width)) / 2
            
            # Plot boundaries
            ax1.plot(left_boundary[:, 0], left_boundary[:, 1], 'k-', alpha=0.3, linewidth=1)
            ax1.plot(right_boundary[:, 0], right_boundary[:, 1], 'k-', alpha=0.3, linewidth=1)
    
    # Plot racing line
    if speed_profile and len(speed_profile) == len(racing_line):
        # Color the racing line by speed
        points = racing_line.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        speed_array = np.array(speed_profile) * speed_factor
        norm = plt.Normalize(np.min(speed_array), np.max(speed_array))
        lc = plt.matplotlib.collections.LineCollection(segments, cmap=SPEED_CMAP, norm=norm)
        lc.set_array(speed_array)
        lc.set_linewidth(DEFAULT_LINE_WIDTH)
        ax1.add_collection(lc)
        
        # Add colorbar
        cbar = plt.colorbar(lc, ax=ax1)
        cbar.set_label(f'Speed ({speed_unit})')
    else:
        # Simple racing line plot
        ax1.plot(racing_line[:, 0], racing_line[:, 1], 'r-', 
               linewidth=DEFAULT_LINE_WIDTH, label='Racing Line')
    
    # Set equal aspect ratio and grid
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=DEFAULT_GRID_ALPHA)
    
    # Set title and labels
    ax1.set_title('Racing Line')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    
    # Plot speed profile
    ax2 = fig.add_subplot(gs[0, 1])
    
    if distances and speed_profile:
        ax2.plot(np.array(distances) * distance_factor, np.array(speed_profile) * speed_factor, 
               'b-', linewidth=DEFAULT_LINE_WIDTH)
        ax2.set_xlabel(f'Distance ({distance_unit})')
        ax2.set_ylabel(f'Speed ({speed_unit})')
        ax2.set_title('Speed Profile')
        ax2.grid(True, alpha=DEFAULT_GRID_ALPHA)
    
    # Plot curvature
    ax3 = fig.add_subplot(gs[1, 0])
    
    if distances and curvature:
        # Curvature can be positive or negative (left/right turns)
        ax3.plot(np.array(distances) * distance_factor, curvature, 
               'g-', linewidth=DEFAULT_LINE_WIDTH)
        
        # Add zero line
        ax3.axhline(y=0, color='k', linestyle='-', alpha=0.2)
        
        # Color regions by turn direction
        if len(distances) > 1:
            # Find transition points where curvature changes sign
            transitions = np.where(np.diff(np.signbit(curvature)))[0]
            
            # Fill regions
            for i in range(len(transitions) + 1):
                start = transitions[i-1] + 1 if i > 0 else 0
                end = transitions[i] if i < len(transitions) else len(curvature) - 1
                
                if start <= end:
                    x = np.array(distances)[start:end+1] * distance_factor
                    y = curvature[start:end+1]
                    
                    if y[0] > 0:  # Left turn
                        ax3.fill_between(x, y, 0, alpha=0.2, color='blue', label='Left Turn' if i == 0 else "")
                    else:  # Right turn
                        ax3.fill_between(x, y, 0, alpha=0.2, color='red', label='Right Turn' if i == 0 else "")
        
        ax3.set_xlabel(f'Distance ({distance_unit})')
        ax3.set_ylabel('Curvature (1/m)')
        ax3.set_title('Track Curvature')
        ax3.grid(True, alpha=DEFAULT_GRID_ALPHA)
        
        # Add legend if regions were filled
        handles, labels = ax3.get_legend_handles_labels()
        if handles:
            ax3.legend()
    
    # Plot time profile
    ax4 = fig.add_subplot(gs[1, 1])
    
    if distances and time_profile:
        ax4.plot(np.array(distances) * distance_factor, time_profile, 
               'purple', linewidth=DEFAULT_LINE_WIDTH)
        ax4.set_xlabel(f'Distance ({distance_unit})')
        ax4.set_ylabel('Time (s)')
        ax4.set_title('Time Profile')
        ax4.grid(True, alpha=DEFAULT_GRID_ALPHA)
        
        # Add total lap time
        if time_profile[-1] > 0:
            ax4.text(0.5, 0.9, f"Lap Time: {time_profile[-1]:.3f}s", 
                   transform=ax4.transAxes, ha='center',
                   bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
    
    # Plot radius (inverse of curvature)
    ax5 = fig.add_subplot(gs[2, 0])
    
    if distances and curvature:
        # Convert curvature to corner radius (ignoring zero curvature for straight sections)
        radius = []
        radius_distances = []
        
        for i, curve in enumerate(curvature):
            if abs(curve) > 0.001:  # Only for actual corners (non-zero curvature)
                radius.append(1.0 / abs(curve))
                radius_distances.append(distances[i])
        
        if radius:
            ax5.scatter(np.array(radius_distances) * distance_factor, radius, 
                      c=radius, cmap='viridis', alpha=0.7)
            ax5.set_xlabel(f'Distance ({distance_unit})')
            ax5.set_ylabel('Corner Radius (m)')
            ax5.set_title('Corner Radius')
            ax5.set_yscale('log')  # Log scale for better visualization
            ax5.grid(True, alpha=DEFAULT_GRID_ALPHA)
    
    # Plot acceleration/deceleration profile
    ax6 = fig.add_subplot(gs[2, 1])
    
    if distances and speed_profile and len(distances) > 1:
        # Calculate acceleration (derivative of speed)
        accel = np.zeros(len(speed_profile))
        
        for i in range(1, len(speed_profile) - 1):
            # Central difference for better accuracy
            delta_s = distances[i+1] - distances[i-1]
            delta_v = speed_profile[i+1] - speed_profile[i-1]
            
            if delta_s > 0:
                accel[i] = delta_v / delta_s
        
        # Forward difference for first point
        if len(distances) > 1:
            accel[0] = (speed_profile[1] - speed_profile[0]) / (distances[1] - distances[0])
        
        # Backward difference for last point
        if len(distances) > 1:
            accel[-1] = (speed_profile[-1] - speed_profile[-2]) / (distances[-1] - distances[-2])
        
        # Convert to g forces
        accel_g = accel / 9.81
        
        # Plot with colors for acceleration/deceleration
        acceleration_mask = accel_g > 0
        deceleration_mask = accel_g <= 0
        
        ax6.plot(np.array(distances)[acceleration_mask] * distance_factor, accel_g[acceleration_mask], 
               'g-', linewidth=DEFAULT_LINE_WIDTH, label='Acceleration')
        ax6.plot(np.array(distances)[deceleration_mask] * distance_factor, accel_g[deceleration_mask], 
               'r-', linewidth=DEFAULT_LINE_WIDTH, label='Deceleration')
        
        ax6.set_xlabel(f'Distance ({distance_unit})')
        ax6.set_ylabel('Acceleration (g)')
        ax6.set_title('Acceleration/Deceleration Profile')
        ax6.grid(True, alpha=DEFAULT_GRID_ALPHA)
        ax6.legend(loc='best')
    
    # Set overall title
    if title:
        fig.suptitle(title, fontsize=DEFAULT_TITLE_SIZE+2)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    # Add summary statistics
    summary_text = ""
    if 'total_time' in racing_line_data or (time_profile and len(time_profile) > 0):
        lap_time = racing_line_data.get('total_time', time_profile[-1] if time_profile else 0)
        summary_text += f"Lap Time: {lap_time:.3f}s | "
    
    if speed_profile:
        avg_speed = np.mean(speed_profile) * speed_factor
        max_speed = np.max(speed_profile) * speed_factor
        summary_text += f"Avg Speed: {avg_speed:.1f} {speed_unit} | "
        summary_text += f"Max Speed: {max_speed:.1f} {speed_unit}"
    
    if summary_text:
        plt.figtext(0.5, 0.01, summary_text, ha='center', fontsize=DEFAULT_LABEL_SIZE, 
                   bbox=dict(facecolor='white', alpha=0.8, edgecolor='lightgray'))
        plt.subplots_adjust(bottom=0.08)
    
    # Save if requested
    if save_path:
        save_plot(fig, save_path)
    
    return fig


#------------------------------------------------------------------------------
# Thermal system plotting functions
#------------------------------------------------------------------------------

def plot_thermal_performance(thermal_data: Dict, title: Optional[str] = None,
                           unit_system: str = 'metric', 
                           save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot thermal system performance.
    
    Args:
        thermal_data: Dictionary with thermal performance data
        title: Plot title
        unit_system: Unit system ('metric' or 'imperial')
        save_path: Path to save plot (if None, not saved)
        
    Returns:
        Matplotlib figure
    """
    # Extract data
    time = thermal_data.get('time', [])
    engine_temp = thermal_data.get('engine_temp', [])
    coolant_temp = thermal_data.get('coolant_temp', [])
    oil_temp = thermal_data.get('oil_temp', [])
    ambient_temp = thermal_data.get('ambient_temp', [])
    
    # Handle constant ambient temperature
    if not ambient_temp and 'ambient_temperature' in thermal_data:
        ambient_temp = [thermal_data['ambient_temperature']] * len(time)
    
    # Get vehicle speed if available
    vehicle_speed = thermal_data.get('vehicle_speed', [])
    
    # Get heat rejection rates if available
    heat_rejection = thermal_data.get('heat_rejection', [])
    radiator_effectiveness = thermal_data.get('radiator_effectiveness', [])
    
    # Get fan duty cycle if available
    fan_duty = thermal_data.get('fan_duty', [])
    pump_speed = thermal_data.get('pump_speed', [])
    
    # Validate data
    if not time or (not engine_temp and not coolant_temp and not oil_temp):
        logger.error("Invalid thermal data format")
        return None
    
    # Create figure
    fig = plt.figure(figsize=(14, 10))
    
    # Temperature conversion for imperial units
    if unit_system.lower() == 'imperial':
        temp_convert = lambda t: t * 9/5 + 32 if t is not None else None
        temp_unit = "°F"
        speed_factor = MS_TO_MPH  # Convert m/s to mph
        speed_unit = "mph"
    else:
        temp_convert = lambda t: t  # No conversion needed
        temp_unit = "°C"
        speed_factor = MS_TO_KMH  # Convert m/s to km/h
        speed_unit = "km/h"
    
    # Number of subplots depends on available data
    num_plots = 2  # At least temperatures and speed
    if heat_rejection:
        num_plots += 1
    if fan_duty or pump_speed:
        num_plots += 1
    
    gs = gridspec.GridSpec(num_plots, 1, height_ratios=[2] + [1] * (num_plots - 1))
    
    # Plot temperatures
    ax1 = fig.add_subplot(gs[0])
    
    if engine_temp:
        ax1.plot(time, [temp_convert(t) for t in engine_temp], 'r-', 
               linewidth=DEFAULT_LINE_WIDTH, label='Engine')
    
    if coolant_temp:
        ax1.plot(time, [temp_convert(t) for t in coolant_temp], 'b-', 
               linewidth=DEFAULT_LINE_WIDTH, label='Coolant')
    
    if oil_temp:
        ax1.plot(time, [temp_convert(t) for t in oil_temp], 'g-', 
               linewidth=DEFAULT_LINE_WIDTH, label='Oil')
    
    if ambient_temp:
        ax1.plot(time, [temp_convert(t) for t in ambient_temp], 'k--', 
               alpha=0.5, linewidth=1, label='Ambient')
    
    # Plot warning/critical thresholds if available
    warning_temps = thermal_data.get('warning_temps', {})
    critical_temps = thermal_data.get('critical_temps', {})
    
    if 'engine' in warning_temps:
        ax1.axhline(y=temp_convert(warning_temps['engine']), color='r', linestyle='--', 
                  alpha=0.5, label='Engine Warning')
    
    if 'engine' in critical_temps:
        ax1.axhline(y=temp_convert(critical_temps['engine']), color='r', linestyle='-', 
                  alpha=0.5, label='Engine Critical')
    
    if 'coolant' in warning_temps:
        ax1.axhline(y=temp_convert(warning_temps['coolant']), color='b', linestyle='--', 
                  alpha=0.5, label='Coolant Warning')
    
    if 'coolant' in critical_temps:
        ax1.axhline(y=temp_convert(critical_temps['coolant']), color='b', linestyle='-', 
                  alpha=0.5, label='Coolant Critical')
    
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel(f'Temperature ({temp_unit})')
    ax1.set_title('Temperature vs. Time')
    ax1.grid(True, alpha=DEFAULT_GRID_ALPHA)
    ax1.legend(loc='best')
    
    # Plot vehicle speed
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    
    if vehicle_speed:
        ax2.plot(time, np.array(vehicle_speed) * speed_factor, 'b-', 
               linewidth=DEFAULT_LINE_WIDTH)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel(f'Speed ({speed_unit})')
        ax2.set_title('Vehicle Speed')
        ax2.grid(True, alpha=DEFAULT_GRID_ALPHA)
    
    # Plot heat rejection if available
    current_plot = 2
    if heat_rejection:
        ax3 = fig.add_subplot(gs[current_plot], sharex=ax1)
        
        ax3.plot(time, heat_rejection, 'r-', linewidth=DEFAULT_LINE_WIDTH, label='Heat Rejection')
        
        if radiator_effectiveness:
            # Create a secondary axis for effectiveness
            ax3_twin = ax3.twinx()
            ax3_twin.plot(time, np.array(radiator_effectiveness) * 100, 'g--', 
                        linewidth=DEFAULT_LINE_WIDTH, label='Radiator Effectiveness')
            ax3_twin.set_ylabel('Effectiveness (%)')
            ax3_twin.set_ylim(0, 100)
            
            # Combine legends
            lines, labels = ax3.get_legend_handles_labels()
            lines2, labels2 = ax3_twin.get_legend_handles_labels()
            ax3.legend(lines + lines2, labels + labels2, loc='best')
        else:
            ax3.legend(loc='best')
        
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Heat Rejection (W)')
        ax3.set_title('Cooling System Performance')
        ax3.grid(True, alpha=DEFAULT_GRID_ALPHA)
        
        current_plot += 1
    
    # Plot fan duty and pump speed if available
    if fan_duty or pump_speed:
        ax4 = fig.add_subplot(gs[current_plot], sharex=ax1)
        
        if fan_duty:
            ax4.plot(time, np.array(fan_duty) * 100, 'b-', 
                   linewidth=DEFAULT_LINE_WIDTH, label='Fan Duty')
            ax4.set_ylabel('Fan Duty (%)')
        
        if pump_speed:
            if fan_duty:
                # Create a secondary axis for pump speed
                ax4_twin = ax4.twinx()
                ax4_twin.plot(time, pump_speed, 'g-', 
                            linewidth=DEFAULT_LINE_WIDTH, label='Pump Speed')
                ax4_twin.set_ylabel('Pump Speed (RPM)')
                
                # Combine legends
                lines, labels = ax4.get_legend_handles_labels()
                lines2, labels2 = ax4_twin.get_legend_handles_labels()
                ax4.legend(lines + lines2, labels + labels2, loc='best')
            else:
                ax4.plot(time, pump_speed, 'g-', 
                       linewidth=DEFAULT_LINE_WIDTH, label='Pump Speed')
                ax4.set_ylabel('Pump Speed (RPM)')
                ax4.legend(loc='best')
        else:
            ax4.legend(loc='best')
        
        ax4.set_xlabel('Time (s)')
        ax4.set_title('Cooling System Control')
        ax4.grid(True, alpha=DEFAULT_GRID_ALPHA)
    
    # Set overall title
    if title:
        fig.suptitle(title, fontsize=DEFAULT_TITLE_SIZE+2)
    else:
        fig.suptitle('Thermal System Performance', fontsize=DEFAULT_TITLE_SIZE+2)
    
    # Link x-axes
    plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, hspace=0.3)
    
    # Add summary statistics text
    summary_text = ""
    if engine_temp:
        max_engine = max(engine_temp)
        summary_text += f"Max Engine: {temp_convert(max_engine):.1f}{temp_unit} | "
    
    if coolant_temp:
        max_coolant = max(coolant_temp)
        summary_text += f"Max Coolant: {temp_convert(max_coolant):.1f}{temp_unit} | "
    
    if oil_temp:
        max_oil = max(oil_temp)
        summary_text += f"Max Oil: {temp_convert(max_oil):.1f}{temp_unit}"
    
    if summary_text:
        plt.figtext(0.5, 0.01, summary_text, ha='center', fontsize=DEFAULT_LABEL_SIZE, 
                   bbox=dict(facecolor='white', alpha=0.8, edgecolor='lightgray'))
        plt.subplots_adjust(bottom=0.08)
    
    # Save if requested
    if save_path:
        save_plot(fig, save_path)
    
    return fig


def plot_thermal_comparison(comparison_data: List[Dict], title: Optional[str] = None,
                          unit_system: str = 'metric',
                          save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot comparison of multiple thermal system configurations.
    
    Args:
        comparison_data: List of dictionaries with thermal data and labels
        title: Plot title
        unit_system: Unit system ('metric' or 'imperial')
        save_path: Path to save plot (if None, not saved)
        
    Returns:
        Matplotlib figure
    """
    if not comparison_data:
        logger.error("No comparison data provided")
        return None
    
    # Create figure
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 2, height_ratios=[2, 1])
    
    # Temperature conversion for imperial units
    if unit_system.lower() == 'imperial':
        temp_convert = lambda t: t * 9/5 + 32 if t is not None else None
        temp_unit = "°F"
        speed_factor = MS_TO_MPH  # Convert m/s to mph
        speed_unit = "mph"
    else:
        temp_convert = lambda t: t  # No conversion needed
        temp_unit = "°C"
        speed_factor = MS_TO_KMH  # Convert m/s to km/h
        speed_unit = "km/h"
    
    # Colors for different configurations
    colors = COLOR_SCHEMES['formula_student']
    
    # Plot engine temperatures
    ax1 = fig.add_subplot(gs[0, 0])
    
    config_names = []
    max_engine_temps = []
    max_coolant_temps = []
    
    for i, data in enumerate(comparison_data):
        config_name = data.get('label', f'Config {i+1}')
        config_names.append(config_name)
        
        time = data.get('time', [])
        engine_temp = data.get('engine_temp', [])
        
        if not time or not engine_temp:
            logger.warning(f"Skipping config {i+1} engine temp due to missing data")
            continue
        
        # Plot engine temperature
        color = data.get('color', colors[i % len(colors)])
        ax1.plot(time, [temp_convert(t) for t in engine_temp], 
               '-', color=color, linewidth=DEFAULT_LINE_WIDTH, label=config_name)
        
        # Store max temperature for bar chart
        max_engine_temps.append(temp_convert(max(engine_temp)))
    
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel(f'Engine Temperature ({temp_unit})')
    ax1.set_title('Engine Temperature Comparison')
    ax1.grid(True, alpha=DEFAULT_GRID_ALPHA)
    ax1.legend(loc='best')
    
    # Plot coolant temperatures
    ax2 = fig.add_subplot(gs[0, 1])
    
    for i, data in enumerate(comparison_data):
        time = data.get('time', [])
        coolant_temp = data.get('coolant_temp', [])
        
        if not time or not coolant_temp:
            logger.warning(f"Skipping config {i+1} coolant temp due to missing data")
            continue
        
        # Plot coolant temperature
        color = data.get('color', colors[i % len(colors)])
        ax2.plot(time, [temp_convert(t) for t in coolant_temp], 
               '-', color=color, linewidth=DEFAULT_LINE_WIDTH, label=data.get('label', f'Config {i+1}'))
        
        # Store max temperature for bar chart
        max_coolant_temps.append(temp_convert(max(coolant_temp)))
    
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel(f'Coolant Temperature ({temp_unit})')
    ax2.set_title('Coolant Temperature Comparison')
    ax2.grid(True, alpha=DEFAULT_GRID_ALPHA)
    ax2.legend(loc='best')
    
    # Plot maximum temperature comparison as bar chart
    ax3 = fig.add_subplot(gs[1, 0])
    
    x = np.arange(len(config_names))
    width = 0.35
    
    if max_engine_temps:
        bars1 = ax3.bar(x - width/2, max_engine_temps, width, label='Max Engine Temp')
        
        for bar, temp in zip(bars1, max_engine_temps):
            ax3.text(bar.get_x() + bar.get_width()/2., temp + 1,
                   f'{temp:.1f}', ha='center', va='bottom', fontsize=8)
    
    if max_coolant_temps:
        bars2 = ax3.bar(x + width/2, max_coolant_temps, width, label='Max Coolant Temp')
        
        for bar, temp in zip(bars2, max_coolant_temps):
            ax3.text(bar.get_x() + bar.get_width()/2., temp + 1,
                   f'{temp:.1f}', ha='center', va='bottom', fontsize=8)
    
    ax3.set_xlabel('Configuration')
    ax3.set_ylabel(f'Temperature ({temp_unit})')
    ax3.set_title('Maximum Temperature Comparison')
    ax3.set_xticks(x)
    ax3.set_xticklabels(config_names, rotation=45, ha='right')
    ax3.legend(loc='best')
    ax3.grid(True, alpha=DEFAULT_GRID_ALPHA, axis='y')
    
    # Plot cooling system performance metrics if available
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Check if heat rejection data is available
    heat_rejection_available = False
    for data in comparison_data:
        if 'heat_rejection' in data and data['heat_rejection']:
            heat_rejection_available = True
            break
    
    if heat_rejection_available:
        # Plot average heat rejection
        avg_heat_rejection = []
        
        for i, data in enumerate(comparison_data):
            heat_rejection = data.get('heat_rejection', [])
            
            if heat_rejection:
                avg_heat_rejection.append(np.mean(heat_rejection))
            else:
                avg_heat_rejection.append(0)
        
        bars = ax4.bar(config_names, avg_heat_rejection, color='purple')
        
        for bar, hr in zip(bars, avg_heat_rejection):
            ax4.text(bar.get_x() + bar.get_width()/2., hr + 100,
                   f'{hr:.0f}W', ha='center', va='bottom', fontsize=8)
        
        ax4.set_xlabel('Configuration')
        ax4.set_ylabel('Average Heat Rejection (W)')
        ax4.set_title('Cooling System Performance')
        ax4.set_xticklabels(config_names, rotation=45, ha='right')
        ax4.grid(True, alpha=DEFAULT_GRID_ALPHA, axis='y')
    else:
        # Plot something else if heat rejection is not available
        # For example, time above warning temperature
        if 'warning_temps' in comparison_data[0]:
            warning_temps = comparison_data[0].get('warning_temps', {})
            
            if 'engine' in warning_temps:
                engine_warning = warning_temps['engine']
                time_above_warning = []
                
                for data in comparison_data:
                    time = data.get('time', [])
                    engine_temp = data.get('engine_temp', [])
                    
                    if time and engine_temp:
                        # Count time above warning
                        count = sum(1 for t in engine_temp if t > engine_warning)
                        above_warning_pct = count / len(engine_temp) * 100
                        time_above_warning.append(above_warning_pct)
                    else:
                        time_above_warning.append(0)
                
                bars = ax4.bar(config_names, time_above_warning, color='orange')
                
                for bar, pct in zip(bars, time_above_warning):
                    ax4.text(bar.get_x() + bar.get_width()/2., pct + 1,
                           f'{pct:.1f}%', ha='center', va='bottom', fontsize=8)
                
                ax4.set_xlabel('Configuration')
                ax4.set_ylabel('Time Above Warning (%)')
                ax4.set_title(f'Time Above Engine Warning ({temp_convert(engine_warning)}{temp_unit})')
                ax4.set_xticklabels(config_names, rotation=45, ha='right')
                ax4.grid(True, alpha=DEFAULT_GRID_ALPHA, axis='y')
    
    # Set overall title
    if title:
        fig.suptitle(title, fontsize=DEFAULT_TITLE_SIZE+2)
    else:
        fig.suptitle('Thermal System Configuration Comparison', fontsize=DEFAULT_TITLE_SIZE+2)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # Save if requested
    if save_path:
        save_plot(fig, save_path)
    
    return fig


def plot_cooling_system_map(cooling_data: Dict, title: Optional[str] = None,
                          unit_system: str = 'metric',
                          save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot cooling system performance map.
    
    Args:
        cooling_data: Dictionary with cooling system performance data
        title: Plot title
        unit_system: Unit system ('metric' or 'imperial')
        save_path: Path to save plot (if None, not saved)
        
    Returns:
        Matplotlib figure
    """
    # Extract data
    speeds = cooling_data.get('speeds', [])
    engine_loads = cooling_data.get('engine_loads', [])
    temperature_map = cooling_data.get('temperature_map', [])
    heat_rejection_map = cooling_data.get('heat_rejection_map', [])
    
    # Validate data
    if not speeds or not engine_loads or not temperature_map:
        logger.error("Invalid cooling system data format")
        return None
    
    # Convert to numpy arrays if they aren't already
    speeds = np.array(speeds)
    engine_loads = np.array(engine_loads)
    temperature_map = np.array(temperature_map)
    
    if heat_rejection_map:
        heat_rejection_map = np.array(heat_rejection_map)
    
    # Create meshgrid for contour plots
    X, Y = np.meshgrid(speeds, engine_loads)
    
    # Unit conversions
    if unit_system.lower() == 'imperial':
        temp_convert = lambda t: t * 9/5 + 32 if t is not None else None
        temp_unit = "°F"
        speed_factor = MS_TO_MPH  # Convert m/s to mph
        speed_unit = "mph"
    else:
        temp_convert = lambda t: t  # No conversion needed
        temp_unit = "°C"
        speed_factor = MS_TO_KMH  # Convert m/s to km/h
        speed_unit = "km/h"
    
    # Convert speeds for display
    display_speeds = speeds * speed_factor
    
    # Convert temperatures for display
    if unit_system.lower() == 'imperial':
        display_temp_map = temperature_map * 9/5 + 32
    else:
        display_temp_map = temperature_map
    
    # Create figure
    fig = plt.figure(figsize=(15, 10))
    
    # Determine number of subplots
    if heat_rejection_map is not None:
        gs = gridspec.GridSpec(2, 2, height_ratios=[2, 1])
    else:
        gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
    
    # Plot temperature contour map
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Create filled contour plot
    contour = ax1.contourf(X * speed_factor, Y * 100, display_temp_map, 20, cmap=THERMAL_CMAP)
    
    # Add contour lines with labels
    contour_lines = ax1.contour(X * speed_factor, Y * 100, display_temp_map, 10, colors='black', alpha=0.5)
    plt.clabel(contour_lines, inline=True, fontsize=8, fmt='%.0f')
    
    # Add colorbar
    cbar = plt.colorbar(contour, ax=ax1)
    cbar.set_label(f'Temperature ({temp_unit})')
    
    # Add warning/critical thresholds if available
    warning_temps = cooling_data.get('warning_temps', {})
    critical_temps = cooling_data.get('critical_temps', {})
    
    if 'engine' in warning_temps:
        warning_temp = temp_convert(warning_temps['engine'])
        contour_warning = ax1.contour(X * speed_factor, Y * 100, display_temp_map, [warning_temp], 
                                    colors='yellow', linestyles='--', linewidths=2)
        plt.clabel(contour_warning, inline=True, fontsize=8, fmt='Warning: %.0f')
    
    if 'engine' in critical_temps:
        critical_temp = temp_convert(critical_temps['engine'])
        contour_critical = ax1.contour(X * speed_factor, Y * 100, display_temp_map, [critical_temp], 
                                     colors='red', linestyles='-', linewidths=2)
        plt.clabel(contour_critical, inline=True, fontsize=8, fmt='Critical: %.0f')
    
    ax1.set_xlabel(f'Vehicle Speed ({speed_unit})')
    ax1.set_ylabel('Engine Load (%)')
    ax1.set_title('Engine Temperature Map')
    ax1.grid(True, alpha=DEFAULT_GRID_ALPHA)
    
    # Plot heat rejection map if available
    if heat_rejection_map is not None:
        ax2 = fig.add_subplot(gs[0, 1])
        
        # Create filled contour plot
        contour2 = ax2.contourf(X * speed_factor, Y * 100, heat_rejection_map, 20, cmap='viridis')
        
        # Add contour lines with labels
        contour_lines2 = ax2.contour(X * speed_factor, Y * 100, heat_rejection_map, 
                                   10, colors='black', alpha=0.5)
        plt.clabel(contour_lines2, inline=True, fontsize=8, fmt='%.0f')
        
        # Add colorbar
        cbar2 = plt.colorbar(contour2, ax=ax2)
        cbar2.set_label('Heat Rejection (W)')
        
        ax2.set_xlabel(f'Vehicle Speed ({speed_unit})')
        ax2.set_ylabel('Engine Load (%)')
        ax2.set_title('Heat Rejection Map')
        ax2.grid(True, alpha=DEFAULT_GRID_ALPHA)
    
    # Plot temperature profiles at different loads
    ax3 = fig.add_subplot(gs[1, 0])
    
    # Select a few representative loads
    load_indices = [int(i) for i in np.linspace(0, len(engine_loads)-1, 4)]
    
    for i in load_indices:
        load = engine_loads[i]
        temp_profile = display_temp_map[i, :]
        ax3.plot(display_speeds, temp_profile, '-', 
               linewidth=DEFAULT_LINE_WIDTH, label=f"{load*100:.0f}% Load")
    
    # Add warning/critical thresholds
    if 'engine' in warning_temps:
        warning_temp = temp_convert(warning_temps['engine'])
        ax3.axhline(y=warning_temp, color='yellow', linestyle='--', 
                  label=f'Warning: {warning_temp:.0f}{temp_unit}')
    
    if 'engine' in critical_temps:
        critical_temp = temp_convert(critical_temps['engine'])
        ax3.axhline(y=critical_temp, color='red', linestyle='-', 
                  label=f'Critical: {critical_temp:.0f}{temp_unit}')
    
    ax3.set_xlabel(f'Vehicle Speed ({speed_unit})')
    ax3.set_ylabel(f'Temperature ({temp_unit})')
    ax3.set_title('Temperature vs. Speed at Different Loads')
    ax3.grid(True, alpha=DEFAULT_GRID_ALPHA)
    ax3.legend(loc='best')
    
    # Plot additional data if heat rejection map is available
    if heat_rejection_map is not None:
        ax4 = fig.add_subplot(gs[1, 1])
        
        # Plot heat rejection vs. temperature difference
        # We need to calculate temperature difference from ambient
        ambient_temp = cooling_data.get('ambient_temperature', 25.0)
        ambient_temp_display = temp_convert(ambient_temp)
        
        # Extract data points
        speed_heat_pairs = []
        for i in range(len(engine_loads)):
            for j in range(len(speeds)):
                temp_diff = display_temp_map[i, j] - ambient_temp_display
                if temp_diff > 0:  # Only consider positive temperature differences
                    speed_heat_pairs.append((
                        display_speeds[j],
                        heat_rejection_map[i, j],
                        temp_diff
                    ))
        
        if speed_heat_pairs:
            # Unzip data
            plot_speeds, plot_heat, plot_temp_diff = zip(*speed_heat_pairs)
            
            # Create scatter plot colored by temperature difference
            scatter = ax4.scatter(plot_speeds, plot_heat, c=plot_temp_diff, 
                                cmap=THERMAL_CMAP, alpha=0.7)
            
            # Add colorbar
            cbar3 = plt.colorbar(scatter, ax=ax4)
            cbar3.set_label(f'Temperature Difference ({temp_unit})')
            
            ax4.set_xlabel(f'Vehicle Speed ({speed_unit})')
            ax4.set_ylabel('Heat Rejection (W)')
            ax4.set_title('Heat Rejection vs. Speed')
            ax4.grid(True, alpha=DEFAULT_GRID_ALPHA)
    
    # Set overall title
    if title:
        fig.suptitle(title, fontsize=DEFAULT_TITLE_SIZE+2)
    else:
        fig.suptitle('Cooling System Performance Map', fontsize=DEFAULT_TITLE_SIZE+2)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # Save if requested
    if save_path:
        save_plot(fig, save_path)
    
    return fig


#------------------------------------------------------------------------------
# Weight sensitivity plotting functions
#------------------------------------------------------------------------------

def plot_weight_sensitivity(sensitivity_data: Dict, title: Optional[str] = None,
                          unit_system: str = 'metric', 
                          save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot weight sensitivity analysis.
    
    Args:
        sensitivity_data: Dictionary with weight sensitivity data
        title: Plot title
        unit_system: Unit system ('metric' or 'imperial')
        save_path: Path to save plot (if None, not saved)
        
    Returns:
        Matplotlib figure
    """
    # Extract data
    weights = sensitivity_data.get('weights', [])
    lap_times = sensitivity_data.get('lap_times', [])
    accel_times = sensitivity_data.get('acceleration_times', [])
    
    zero_to_sixty = sensitivity_data.get('zero_to_sixty', [])
    power_to_weight = sensitivity_data.get('power_to_weight', [])
    weight_distribution = sensitivity_data.get('weight_distribution', [])
    
    # Validate data
    if not weights or (not lap_times and not accel_times):
        logger.error("Invalid weight sensitivity data format")
        return None
    
    # Create figure
    fig = plt.figure(figsize=(15, 10))
    
    # Unit conversions
    if unit_system.lower() == 'imperial':
        weight_factor = KG_TO_LBS  # Convert kg to lbs
        weight_unit = "lbs"
    else:
        weight_factor = 1.0
        weight_unit = "kg"
    
    # Determine number of subplots based on available data
    plot_count = sum([
        bool(lap_times),
        bool(accel_times),
        bool(power_to_weight),
        bool(weight_distribution)
    ])
    
    if plot_count <= 2:
        subplot_rows, subplot_cols = 1, plot_count
    else:
        subplot_rows, subplot_cols = 2, 2
    
    # Convert weights for display
    display_weights = np.array(weights) * weight_factor
    
    # Current subplot index
    subplot_idx = 1
    
    # Plot lap time sensitivity
    if lap_times:
        ax1 = fig.add_subplot(subplot_rows, subplot_cols, subplot_idx)
        subplot_idx += 1
        
        # Fit linear regression
        coeffs = np.polyfit(weights, lap_times, 1)
        poly = np.poly1d(coeffs)
        
        # Calculate sensitivity coefficient (seconds per kg)
        sensitivity = coeffs[0]
        
        # Plot data points
        ax1.scatter(display_weights, lap_times, color='blue', s=40)
        
        # Plot regression line
        x_line = np.linspace(min(weights), max(weights), 100)
        y_line = poly(x_line)
        ax1.plot(x_line * weight_factor, y_line, 'r-', 
               linewidth=DEFAULT_LINE_WIDTH, 
               label=f"Sensitivity: {sensitivity:.4f}s/kg")
        
        ax1.set_xlabel(f'Vehicle Weight ({weight_unit})')
        ax1.set_ylabel('Lap Time (s)')
        ax1.set_title('Lap Time vs. Weight')
        ax1.grid(True, alpha=DEFAULT_GRID_ALPHA)
        ax1.legend(loc='best')
    
    # Plot fuel consumption
    ax2 = fig.add_subplot(gs[0, 1])
    
    if fuel_consumption:
        # Calculate cumulative consumption
        cumulative_fuel = np.cumsum(fuel_consumption)
        
        # Convert to selected units
        cumulative_fuel = cumulative_fuel * fuel_factor
        
        ax2.plot(lap_numbers, cumulative_fuel, 'g-o', 
               linewidth=DEFAULT_LINE_WIDTH, markersize=5)
        
        # Add per-lap consumption
        ax2_twin = ax2.twinx()
        per_lap_fuel = np.array(fuel_consumption) * fuel_factor
        ax2_twin.bar(lap_numbers, per_lap_fuel, alpha=0.3, color='green', 
                   label=f'Per Lap ({fuel_unit})')
        ax2_twin.set_ylabel(f'Per Lap Consumption ({fuel_unit})')
        
        # Set primary axis labels
        ax2.set_xlabel('Lap Number')
        ax2.set_ylabel(f'Cumulative Fuel ({fuel_unit})')
        ax2.set_title('Fuel Consumption')
        ax2.grid(True, alpha=DEFAULT_GRID_ALPHA)
        
        # Combine legends
        lines, labels = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='best')
    
    # Plot temperatures
    ax3 = fig.add_subplot(gs[1, 0])
    
    has_temps = False
    
    if engine_temps:
        has_temps = True
        ax3.plot(lap_numbers, [temp_convert(t) for t in engine_temps], 'r-o', 
               linewidth=DEFAULT_LINE_WIDTH, markersize=5, label='Engine')
    
    if coolant_temps:
        has_temps = True
        ax3.plot(lap_numbers, [temp_convert(t) for t in coolant_temps], 'b-o', 
               linewidth=DEFAULT_LINE_WIDTH, markersize=5, label='Coolant')
    
    if oil_temps:
        has_temps = True
        ax3.plot(lap_numbers, [temp_convert(t) for t in oil_temps], 'g-o', 
               linewidth=DEFAULT_LINE_WIDTH, markersize=5, label='Oil')
    
    if has_temps:
        # Plot warning/critical thresholds if available
        warning_temps = endurance_data.get('warning_temps', {})
        critical_temps = endurance_data.get('critical_temps', {})
        
        if 'engine' in warning_temps:
            ax3.axhline(y=temp_convert(warning_temps['engine']), color='r', linestyle='--', 
                      alpha=0.5, label='Engine Warning')
        
        if 'engine' in critical_temps:
            ax3.axhline(y=temp_convert(critical_temps['engine']), color='r', linestyle='-', 
                      alpha=0.5, label='Engine Critical')
        
        ax3.set_xlabel('Lap Number')
        ax3.set_ylabel(f'Temperature ({temp_unit})')
        ax3.set_title('Temperature Profile')
        ax3.grid(True, alpha=DEFAULT_GRID_ALPHA)
        ax3.legend(loc='best')
    
    # Plot component wear if available
    ax4 = fig.add_subplot(gs[1, 1])
    
    if component_wear and isinstance(component_wear, dict) and component_wear:
        # Extract component names and wear values
        components = list(component_wear.keys())
        wear_values = [component_wear[c] * 100 for c in components]  # Convert to percentage
        
        # Sort by wear (highest first)
        sorted_indices = np.argsort(wear_values)[::-1]
        sorted_components = [components[i] for i in sorted_indices]
        sorted_wear = [wear_values[i] for i in sorted_indices]
        
        # Create horizontal bar chart
        bars = ax4.barh(sorted_components, sorted_wear, color='orange')
        
        # Add percentage labels
        for bar, wear in zip(bars, sorted_wear):
            ax4.text(wear + 1, bar.get_y() + bar.get_height()/2, 
                   f'{wear:.1f}%', va='center')
        
        ax4.set_xlabel('Wear (%)')
        ax4.set_ylabel('Component')
        ax4.set_title('Component Wear')
        ax4.grid(True, alpha=DEFAULT_GRID_ALPHA, axis='x')
        ax4.set_xlim(0, 105)  # Limit to 0-100% with room for labels
    
    # Plot lap time consistency
    ax5 = fig.add_subplot(gs[2, 0])
    
    if lap_times and len(lap_times) > 1:
        # Calculate moving average
        window_size = min(5, len(lap_times))
        moving_avg = np.convolve(lap_times, np.ones(window_size)/window_size, mode='valid')
        
        # Calculate lap time deviation from average
        avg_lap_time = np.mean(lap_times)
        deviations = [(t - avg_lap_time) for t in lap_times]
        
        # Plot deviations as bars
        bars = ax5.bar(lap_numbers, deviations, color=['g' if d <= 0 else 'r' for d in deviations])
        
        # Plot moving average line
        moving_avg_x = np.arange(window_size//2 + 1, len(lap_times) - window_size//2 + 1)
        ax5.plot(moving_avg_x, moving_avg - avg_lap_time, 'b-', 
               linewidth=DEFAULT_LINE_WIDTH, label='Moving Avg')
        
        # Add zero line
        ax5.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        
        ax5.set_xlabel('Lap Number')
        ax5.set_ylabel('Deviation from Average (s)')
        ax5.set_title('Lap Time Consistency')
        ax5.grid(True, alpha=DEFAULT_GRID_ALPHA)
        ax5.legend(loc='best')
    
    # Plot efficiency score calculation
    ax6 = fig.add_subplot(gs[2, 1])
    
    if 'efficiency_score' in endurance_data and 'endurance_score' in endurance_data:
        efficiency_score = endurance_data['efficiency_score']
        endurance_score = endurance_data['endurance_score']
        
        # Create a pie chart
        labels = ['Endurance', 'Efficiency']
        sizes = [endurance_score, efficiency_score]
        colors = ['#ff9999', '#66b3ff']
        
        ax6.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', 
              startangle=90, explode=(0.1, 0))
        ax6.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        
        # Add total score
        total_score = endurance_score + efficiency_score
        max_score = endurance_data.get('max_endurance_score', 325) + endurance_data.get('max_efficiency_score', 100)
        
        ax6.text(0, -1.2, f'Total Score: {total_score:.1f}/{max_score}', 
               ha='center', fontsize=DEFAULT_TITLE_SIZE)
        
        ax6.set_title('Endurance + Efficiency Scores')
    
    # Set overall title
    if title:
        fig.suptitle(title, fontsize=DEFAULT_TITLE_SIZE+2)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    # Add summary stats
    if lap_times:
        total_time = np.sum(lap_times)
        avg_lap = np.mean(lap_times)
        best_lap = np.min(lap_times)
        
        summary_text = f"Total Time: {total_time:.1f}s | Avg Lap: {avg_lap:.2f}s | Best Lap: {best_lap:.2f}s"
        
        if fuel_consumption:
            total_fuel = np.sum(fuel_consumption) * fuel_factor
            summary_text += f" | Fuel Used: {total_fuel:.2f} {fuel_unit}"
        
        plt.figtext(0.5, 0.01, summary_text, ha='center', fontsize=DEFAULT_LABEL_SIZE, 
                   bbox=dict(facecolor='white', alpha=0.8, edgecolor='lightgray'))
        plt.subplots_adjust(bottom=0.08)
    
    # Save if requested
    if save_path:
        save_plot(fig, save_path)
    
    return fig


def plot_endurance_comparison(comparison_data: List[Dict], title: Optional[str] = None,
                            unit_system: str = 'metric',
                            save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot comparison of multiple endurance configurations.
    
    Args:
        comparison_data: List of dictionaries with endurance data and labels
        title: Plot title
        unit_system: Unit system ('metric' or 'imperial')
        save_path: Path to save plot (if None, not saved)
        
    Returns:
        Matplotlib figure
    """
    if not comparison_data:
        logger.error("No comparison data provided")
        return None
    
    # Create figure
    fig = plt.figure(figsize=(15, 12))
    gs = gridspec.GridSpec(3, 2, height_ratios=[2, 1, 1])
    
    # Unit conversions
    if unit_system.lower() == 'imperial':
        fuel_factor = LITERS_TO_GAL  # Convert liters to gallons
        fuel_unit = "gal"
    else:
        fuel_factor = 1.0
        fuel_unit = "L"
    
    # Extract configuration labels
    config_labels = [data.get('label', f'Config {i+1}') for i, data in enumerate(comparison_data)]
    
    # Extract key metrics
    total_times = []
    total_fuels = []
    avg_lap_times = []
    best_lap_times = []
    endurance_scores = []
    efficiency_scores = []
    reliability_counts = []
    
    for data in comparison_data:
        lap_times = data.get('lap_times', [])
        fuel_consumption = data.get('fuel_consumption', [])
        reliability_events = data.get('reliability_events', [])
        
        # Calculate metrics
        if lap_times:
            total_times.append(np.sum(lap_times))
            avg_lap_times.append(np.mean(lap_times))
            best_lap_times.append(np.min(lap_times))
        else:
            total_times.append(0)
            avg_lap_times.append(0)
            best_lap_times.append(0)
        
        if fuel_consumption:
            total_fuels.append(np.sum(fuel_consumption) * fuel_factor)
        else:
            total_fuels.append(0)
        
        if 'endurance_score' in data:
            endurance_scores.append(data['endurance_score'])
        else:
            endurance_scores.append(0)
        
        if 'efficiency_score' in data:
            efficiency_scores.append(data['efficiency_score'])
        else:
            efficiency_scores.append(0)
        
        if reliability_events:
            reliability_counts.append(len(reliability_events))
        else:
            reliability_counts.append(0)
    
    # Colors for different configurations
    colors = COLOR_SCHEMES['formula_student']
    
    # Plot lap time comparison
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Plot all lap times
    for i, data in enumerate(comparison_data):
        lap_times = data.get('lap_times', [])
        if lap_times:
            lap_numbers = np.arange(1, len(lap_times) + 1)
            ax1.plot(lap_numbers, lap_times, 'o-', 
                   color=colors[i % len(colors)], 
                   linewidth=1.5, markersize=4, 
                   label=config_labels[i])
    
    ax1.set_xlabel('Lap Number')
    ax1.set_ylabel('Lap Time (s)')
    ax1.set_title('Lap Time Comparison')
    ax1.grid(True, alpha=DEFAULT_GRID_ALPHA)
    ax1.legend(loc='best')
    
    # Plot total time and fuel comparison
    ax2 = fig.add_subplot(gs[0, 1])
    
    x = np.arange(len(config_labels))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, total_times, width, label='Total Time (s)', color='blue')
    
    # Add time labels
    for bar, time in zip(bars1, total_times):
        ax2.text(bar.get_x() + bar.get_width()/2., time + 10,
               f'{time:.1f}s', ha='center', va='bottom', fontsize=8)
    
    # Add secondary axis for fuel
    ax2_twin = ax2.twinx()
    bars2 = ax2_twin.bar(x + width/2, total_fuels, width, 
                       label=f'Total Fuel ({fuel_unit})', color='green')
    
    # Add fuel labels
    for bar, fuel in zip(bars2, total_fuels):
        ax2_twin.text(bar.get_x() + bar.get_width()/2., fuel + 0.1,
                    f'{fuel:.2f}', ha='center', va='bottom', fontsize=8)
    
    ax2.set_xlabel('Configuration')
    ax2.set_ylabel('Total Time (s)')
    ax2_twin.set_ylabel(f'Total Fuel ({fuel_unit})')
    ax2.set_title('Total Time and Fuel Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(config_labels, rotation=45, ha='right')
    
    # Combine legends
    lines, labels = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left')
    
    # Plot average and best lap times
    ax3 = fig.add_subplot(gs[1, 0])
    
    width = 0.35
    
    bars1 = ax3.bar(x - width/2, avg_lap_times, width, 
                  label='Average Lap Time', color='blue', alpha=0.7)
    
    bars2 = ax3.bar(x + width/2, best_lap_times, width, 
                  label='Best Lap Time', color='purple', alpha=0.7)
    
    # Add time labels
    for bar, time in zip(bars1, avg_lap_times):
        ax3.text(bar.get_x() + bar.get_width()/2., time + 0.1,
               f'{time:.2f}s', ha='center', va='bottom', fontsize=8)
    
    for bar, time in zip(bars2, best_lap_times):
        ax3.text(bar.get_x() + bar.get_width()/2., time + 0.1,
               f'{time:.2f}s', ha='center', va='bottom', fontsize=8)
    
    ax3.set_xlabel('Configuration')
    ax3.set_ylabel('Lap Time (s)')
    ax3.set_title('Average and Best Lap Times')
    ax3.set_xticks(x)
    ax3.set_xticklabels(config_labels, rotation=45, ha='right')
    ax3.legend(loc='best')
    
    # Plot scores
    ax4 = fig.add_subplot(gs[1, 1])
    
    width = 0.35
    
    bars1 = ax4.bar(x - width/2, endurance_scores, width, 
                  label='Endurance Score', color='blue', alpha=0.7)
    
    bars2 = ax4.bar(x + width/2, efficiency_scores, width, 
                  label='Efficiency Score', color='green', alpha=0.7)
    
    # Add score labels
    for bar, score in zip(bars1, endurance_scores):
        ax4.text(bar.get_x() + bar.get_width()/2., score + 1,
               f'{score:.1f}', ha='center', va='bottom', fontsize=8)
    
    for bar, score in zip(bars2, efficiency_scores):
        ax4.text(bar.get_x() + bar.get_width()/2., score + 1,
               f'{score:.1f}', ha='center', va='bottom', fontsize=8)
    
    ax4.set_xlabel('Configuration')
    ax4.set_ylabel('Points')
    ax4.set_title('Endurance and Efficiency Scores')
    ax4.set_xticks(x)
    ax4.set_xticklabels(config_labels, rotation=45, ha='right')
    ax4.legend(loc='best')
    
    # Plot total scores and reliability
    ax5 = fig.add_subplot(gs[2, 0])
    
    total_scores = [e + f for e, f in zip(endurance_scores, efficiency_scores)]
    
    bars = ax5.bar(config_labels, total_scores, color='purple')
    
    # Add score labels
    for bar, score in zip(bars, total_scores):
        ax5.text(bar.get_x() + bar.get_width()/2., score + 2,
               f'{score:.1f}', ha='center', va='bottom', fontsize=9)
    
    ax5.set_xlabel('Configuration')
    ax5.set_ylabel('Points')
    ax5.set_title('Total Score (Endurance + Efficiency)')
    ax5.set_xticklabels(config_labels, rotation=45, ha='right')
    
    # Plot fuel efficiency
    ax6 = fig.add_subplot(gs[2, 1])
    
    # Calculate fuel efficiency (distance per fuel)
    fuel_efficiency = []
    for i, data in enumerate(comparison_data):
        lap_times = data.get('lap_times', [])
        fuel_consumption = data.get('fuel_consumption', [])
        
        if lap_times and fuel_consumption:
            # Calculate total distance
            laps_completed = len(lap_times)
            track_length = data.get('track_length', 1000)  # Default if not provided
            total_distance = laps_completed * track_length
            
            # Calculate total fuel
            total_fuel = np.sum(fuel_consumption)
            
            if total_fuel > 0:
                # Calculate km/L or mpg depending on unit system
                if unit_system.lower() == 'imperial':
                    # Convert meters to miles and liters to gallons
                    miles = total_distance / 1609.34
                    gallons = total_fuel * LITERS_TO_GAL
                    efficiency = miles / gallons  # mpg
                else:
                    # Convert meters to km
                    km = total_distance / 1000
                    efficiency = km / total_fuel  # km/L
            else:
                efficiency = 0
        else:
            efficiency = 0
        
        fuel_efficiency.append(efficiency)
    
    # Create bar chart
    bars = ax6.bar(config_labels, fuel_efficiency, color='green')
    
    # Add efficiency labels
    for bar, eff in zip(bars, fuel_efficiency):
        ax6.text(bar.get_x() + bar.get_width()/2., eff + 0.1,
               f'{eff:.2f}', ha='center', va='bottom', fontsize=9)
    
    ax6.set_xlabel('Configuration')
    if unit_system.lower() == 'imperial':
        ax6.set_ylabel('Fuel Efficiency (mpg)')
    else:
        ax6.set_ylabel('Fuel Efficiency (km/L)')
    
    ax6.set_title('Fuel Efficiency')
    ax6.set_xticklabels(config_labels, rotation=45, ha='right')
    
    # Set overall title
    if title:
        fig.suptitle(title, fontsize=DEFAULT_TITLE_SIZE+2)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    # Save if requested
    if save_path:
        save_plot(fig, save_path)
    
    return fig
    
    # Plot acceleration sensitivity
    if accel_times:
        ax2 = fig.add_subplot(subplot_rows, subplot_cols, subplot_idx)
        subplot_idx += 1
        
        # Fit linear regression
        coeffs = np.polyfit(weights, accel_times, 1)
        poly = np.poly1d(coeffs)
        
        # Calculate sensitivity coefficient (seconds per kg)
        sensitivity = coeffs[0]
        
        # Plot data points
        ax2.scatter(display_weights, accel_times, color='green', s=40)
        
        # Plot regression line
        x_line = np.linspace(min(weights), max(weights), 100)
        y_line = poly(x_line)
        ax2.plot(x_line * weight_factor, y_line, 'r-', 
               linewidth=DEFAULT_LINE_WIDTH,
               label=f"Sensitivity: {sensitivity:.4f}s/kg")
        
        ax2.set_xlabel(f'Vehicle Weight ({weight_unit})')
        ax2.set_ylabel('Acceleration Time (s)')
        ax2.set_title('75m Acceleration Time vs. Weight')
        ax2.grid(True, alpha=DEFAULT_GRID_ALPHA)
        ax2.legend(loc='best')
    
    # Plot 0-60 mph time if available
    if zero_to_sixty:
        if subplot_idx <= subplot_rows * subplot_cols:
            ax3 = fig.add_subplot(subplot_rows, subplot_cols, subplot_idx)
            subplot_idx += 1
            
            # Fit linear regression
            coeffs = np.polyfit(weights, zero_to_sixty, 1)
            poly = np.poly1d(coeffs)
            
            # Calculate sensitivity coefficient (seconds per kg)
            sensitivity = coeffs[0]
            
            # Plot data points
            ax3.scatter(display_weights, zero_to_sixty, color='purple', s=40)
            
            # Plot regression line
            x_line = np.linspace(min(weights), max(weights), 100)
            y_line = poly(x_line)
            ax3.plot(x_line * weight_factor, y_line, 'r-', 
                   linewidth=DEFAULT_LINE_WIDTH,
                   label=f"Sensitivity: {sensitivity:.4f}s/kg")
            
            ax3.set_xlabel(f'Vehicle Weight ({weight_unit})')
            ax3.set_ylabel('0-60 mph Time (s)')
            ax3.set_title('0-60 mph Time vs. Weight')
            ax3.grid(True, alpha=DEFAULT_GRID_ALPHA)
            ax3.legend(loc='best')
    
    # Plot power-to-weight ratio if available
    if power_to_weight:
        if subplot_idx <= subplot_rows * subplot_cols:
            ax4 = fig.add_subplot(subplot_rows, subplot_cols, subplot_idx)
            subplot_idx += 1
            
            # Plot data points
            ax4.scatter(display_weights, power_to_weight, color='orange', s=40)
            
            # Plot smooth curve
            x_line = np.linspace(min(weights), max(weights), 100)
            
            # Power-to-weight varies with 1/weight
            engine_power = power_to_weight[0] * weights[0]  # Assumes constant engine power
            y_line = engine_power / x_line
            
            ax4.plot(x_line * weight_factor, y_line, 'r-', 
                   linewidth=DEFAULT_LINE_WIDTH)
            
            ax4.set_xlabel(f'Vehicle Weight ({weight_unit})')
            ax4.set_ylabel('Power-to-Weight (kW/kg)')
            ax4.set_title('Power-to-Weight Ratio vs. Weight')
            ax4.grid(True, alpha=DEFAULT_GRID_ALPHA)
    
    # Set overall title
    if title:
        fig.suptitle(title, fontsize=DEFAULT_TITLE_SIZE+2)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # Add summary text
    if lap_times and weights:
        # Calculate sensitivity coefficients
        lap_time_sensitivity = np.polyfit(weights, lap_times, 1)[0]
        
        if accel_times:
            accel_time_sensitivity = np.polyfit(weights, accel_times, 1)[0]
            summary_text = (f"Lap Time Sensitivity: {lap_time_sensitivity:.4f}s/kg | "
                           f"Acceleration Sensitivity: {accel_time_sensitivity:.4f}s/kg")
        else:
            summary_text = f"Lap Time Sensitivity: {lap_time_sensitivity:.4f}s/kg"
        
        plt.figtext(0.5, 0.01, summary_text, ha='center', fontsize=DEFAULT_LABEL_SIZE, 
                   bbox=dict(facecolor='white', alpha=0.8, edgecolor='lightgray'))
        plt.subplots_adjust(bottom=0.08)
    
    # Save if requested
    if save_path:
        save_plot(fig, save_path)
    
    return fig


def plot_weight_distribution_sensitivity(sensitivity_data: Dict, title: Optional[str] = None,
                                       unit_system: str = 'metric',
                                       save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot weight distribution sensitivity analysis.
    
    Args:
        sensitivity_data: Dictionary with weight distribution sensitivity data
        title: Plot title
        unit_system: Unit system ('metric' or 'imperial')
        save_path: Path to save plot (if None, not saved)
        
    Returns:
        Matplotlib figure
    """
    # Extract data
    front_weight_pct = sensitivity_data.get('front_weight_pct', [])
    lap_times = sensitivity_data.get('lap_times', [])
    accel_times = sensitivity_data.get('acceleration_times', [])
    lat_accel = sensitivity_data.get('lateral_acceleration', [])
    
    # Validate data
    if not front_weight_pct or (not lap_times and not accel_times and not lat_accel):
        logger.error("Invalid weight distribution sensitivity data format")
        return None
    
    # Create figure
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 2, height_ratios=[2, 1])
    
    # Plot lap time sensitivity
    ax1 = fig.add_subplot(gs[0, 0])
    
    if lap_times:
        # Plot data points
        ax1.scatter(front_weight_pct, lap_times, color='blue', s=40)
        
        # Fit polynomial regression (quadratic for weight distribution)
        if len(front_weight_pct) >= 3:
            coeffs = np.polyfit(front_weight_pct, lap_times, 2)
            poly = np.poly1d(coeffs)
            
            # Find optimal weight distribution
            # For a quadratic y = ax² + bx + c, minimum is at x = -b/2a
            if coeffs[0] > 0:  # Check that it's a minimum, not maximum
                optimal_distribution = -coeffs[1] / (2 * coeffs[0])
                
                # Only display if it's within a reasonable range
                if 0.3 <= optimal_distribution <= 0.7:
                    # Plot polynomial curve
                    x_line = np.linspace(min(front_weight_pct), max(front_weight_pct), 100)
                    y_line = poly(x_line)
                    ax1.plot(x_line, y_line, 'r-', linewidth=DEFAULT_LINE_WIDTH)
                    
                    # Mark optimal point
                    optimal_time = poly(optimal_distribution)
                    ax1.scatter([optimal_distribution], [optimal_time], color='red', s=80, zorder=5)
                    
                    # Add annotation
                    ax1.annotate(f'Optimal: {optimal_distribution:.1%}\n({optimal_time:.3f}s)', 
                               xy=(optimal_distribution, optimal_time),
                               xytext=(optimal_distribution, optimal_time + 0.05 * (max(lap_times) - min(lap_times))),
                               ha='center',
                               arrowprops=dict(arrowstyle='->', color='black'))
            else:
                # Just plot the curve without finding optimal (it's a maximum)
                x_line = np.linspace(min(front_weight_pct), max(front_weight_pct), 100)
                y_line = poly(x_line)
                ax1.plot(x_line, y_line, 'r-', linewidth=DEFAULT_LINE_WIDTH)
        
        ax1.set_xlabel('Front Weight Distribution (%)')
        ax1.set_ylabel('Lap Time (s)')
        ax1.set_title('Lap Time vs. Weight Distribution')
        ax1.grid(True, alpha=DEFAULT_GRID_ALPHA)
        
        # Convert x-axis to percentage
        ax1.set_xticklabels([f'{x:.0%}' for x in ax1.get_xticks()])
    
    # Plot acceleration sensitivity
    ax2 = fig.add_subplot(gs[0, 1])
    
    if accel_times:
        # Plot data points
        ax2.scatter(front_weight_pct, accel_times, color='green', s=40)
        
        # Fit polynomial regression (quadratic for weight distribution)
        if len(front_weight_pct) >= 3:
            coeffs = np.polyfit(front_weight_pct, accel_times, 2)
            poly = np.poly1d(coeffs)
            
            # Find optimal weight distribution
            if coeffs[0] > 0:  # Check that it's a minimum, not maximum
                optimal_distribution = -coeffs[1] / (2 * coeffs[0])
                
                # Only display if it's within a reasonable range
                if 0.3 <= optimal_distribution <= 0.7:
                    # Plot polynomial curve
                    x_line = np.linspace(min(front_weight_pct), max(front_weight_pct), 100)
                    y_line = poly(x_line)
                    ax2.plot(x_line, y_line, 'r-', linewidth=DEFAULT_LINE_WIDTH)
                    
                    # Mark optimal point
                    optimal_time = poly(optimal_distribution)
                    ax2.scatter([optimal_distribution], [optimal_time], color='red', s=80, zorder=5)
                    
                    # Add annotation
                    ax2.annotate(f'Optimal: {optimal_distribution:.1%}\n({optimal_time:.3f}s)', 
                               xy=(optimal_distribution, optimal_time),
                               xytext=(optimal_distribution, optimal_time + 0.05 * (max(accel_times) - min(accel_times))),
                               ha='center',
                               arrowprops=dict(arrowstyle='->', color='black'))
            else:
                # Just plot the curve without finding optimal (it's a maximum)
                x_line = np.linspace(min(front_weight_pct), max(front_weight_pct), 100)
                y_line = poly(x_line)
                ax2.plot(x_line, y_line, 'r-', linewidth=DEFAULT_LINE_WIDTH)
        
        ax2.set_xlabel('Front Weight Distribution (%)')
        ax2.set_ylabel('Acceleration Time (s)')
        ax2.set_title('75m Acceleration Time vs. Weight Distribution')
        ax2.grid(True, alpha=DEFAULT_GRID_ALPHA)
        
        # Convert x-axis to percentage
        ax2.set_xticklabels([f'{x:.0%}' for x in ax2.get_xticks()])
    
    # Plot lateral acceleration sensitivity
    ax3 = fig.add_subplot(gs[1, 0])
    
    if lat_accel:
        # Plot data points
        ax3.scatter(front_weight_pct, lat_accel, color='purple', s=40)
        
        # Fit polynomial regression (quadratic for weight distribution)
        if len(front_weight_pct) >= 3:
            coeffs = np.polyfit(front_weight_pct, lat_accel, 2)
            poly = np.poly1d(coeffs)
            
            # Find optimal weight distribution
            if coeffs[0] < 0:  # Check that it's a maximum, not minimum (we want max lateral accel)
                optimal_distribution = -coeffs[1] / (2 * coeffs[0])
                
                # Only display if it's within a reasonable range
                if 0.3 <= optimal_distribution <= 0.7:
                    # Plot polynomial curve
                    x_line = np.linspace(min(front_weight_pct), max(front_weight_pct), 100)
                    y_line = poly(x_line)
                    ax3.plot(x_line, y_line, 'r-', linewidth=DEFAULT_LINE_WIDTH)
                    
                    # Mark optimal point
                    optimal_accel = poly(optimal_distribution)
                    ax3.scatter([optimal_distribution], [optimal_accel], color='red', s=80, zorder=5)
                    
                    # Add annotation
                    ax3.annotate(f'Optimal: {optimal_distribution:.1%}\n({optimal_accel:.3f}g)', 
                               xy=(optimal_distribution, optimal_accel),
                               xytext=(optimal_distribution, optimal_accel - 0.05 * (max(lat_accel) - min(lat_accel))),
                               ha='center',
                               arrowprops=dict(arrowstyle='->', color='black'))
            else:
                # Just plot the curve without finding optimal (it's a minimum)
                x_line = np.linspace(min(front_weight_pct), max(front_weight_pct), 100)
                y_line = poly(x_line)
                ax3.plot(x_line, y_line, 'r-', linewidth=DEFAULT_LINE_WIDTH)
        
        ax3.set_xlabel('Front Weight Distribution (%)')
        ax3.set_ylabel('Lateral Acceleration (g)')
        ax3.set_title('Maximum Lateral Acceleration vs. Weight Distribution')
        ax3.grid(True, alpha=DEFAULT_GRID_ALPHA)
        
        # Convert x-axis to percentage
        ax3.set_xticklabels([f'{x:.0%}' for x in ax3.get_xticks()])
    
    # Plot combined event performance
    ax4 = fig.add_subplot(gs[1, 1])
    
    if lap_times and accel_times and len(front_weight_pct) > 0:
        # Normalize lap times and acceleration times for combined plot
        norm_lap_times = (lap_times - min(lap_times)) / (max(lap_times) - min(lap_times))
        norm_accel_times = (accel_times - min(accel_times)) / (max(accel_times) - min(accel_times))
        
        # Combined performance score (lower is better)
        combined_score = 0.6 * norm_lap_times + 0.4 * norm_accel_times
        
        # Plot data points
        ax4.scatter(front_weight_pct, combined_score, color='orange', s=40)
        
        # Fit polynomial regression
        if len(front_weight_pct) >= 3:
            coeffs = np.polyfit(front_weight_pct, combined_score, 2)
            poly = np.poly1d(coeffs)
            
            # Find optimal weight distribution
            if coeffs[0] > 0:  # Check that it's a minimum, not maximum
                optimal_distribution = -coeffs[1] / (2 * coeffs[0])
                
                # Only display if it's within a reasonable range
                if 0.3 <= optimal_distribution <= 0.7:
                    # Plot polynomial curve
                    x_line = np.linspace(min(front_weight_pct), max(front_weight_pct), 100)
                    y_line = poly(x_line)
                    ax4.plot(x_line, y_line, 'r-', linewidth=DEFAULT_LINE_WIDTH)
                    
                    # Mark optimal point
                    optimal_score = poly(optimal_distribution)
                    ax4.scatter([optimal_distribution], [optimal_score], color='red', s=80, zorder=5)
                    
                    # Add annotation
                    ax4.annotate(f'Optimal: {optimal_distribution:.1%}', 
                               xy=(optimal_distribution, optimal_score),
                               xytext=(optimal_distribution, optimal_score + 0.05),
                               ha='center',
                               arrowprops=dict(arrowstyle='->', color='black'))
            else:
                # Just plot the curve without finding optimal (it's a maximum)
                x_line = np.linspace(min(front_weight_pct), max(front_weight_pct), 100)
                y_line = poly(x_line)
                ax4.plot(x_line, y_line, 'r-', linewidth=DEFAULT_LINE_WIDTH)
        
        ax4.set_xlabel('Front Weight Distribution (%)')
        ax4.set_ylabel('Combined Score (lower is better)')
        ax4.set_title('Combined Performance (60% Lap, 40% Accel)')
        ax4.grid(True, alpha=DEFAULT_GRID_ALPHA)
        
        # Convert x-axis to percentage
        ax4.set_xticklabels([f'{x:.0%}' for x in ax4.get_xticks()])
    
    # Set overall title
    if title:
        fig.suptitle(title, fontsize=DEFAULT_TITLE_SIZE+2)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # Save if requested
    if save_path:
        save_plot(fig, save_path)
    
    return fig


#------------------------------------------------------------------------------
# Endurance plotting functions
#------------------------------------------------------------------------------

def plot_endurance_results(endurance_data: Dict, title: Optional[str] = None,
                         unit_system: str = 'metric',
                         save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot endurance event simulation results.
    
    Args:
        endurance_data: Dictionary with endurance simulation results
        title: Plot title
        unit_system: Unit system ('metric' or 'imperial')
        save_path: Path to save plot (if None, not saved)
        
    Returns:
        Matplotlib figure
    """
    # Extract endurance data
    lap_times = endurance_data.get('lap_times', [])
    lap_numbers = np.arange(1, len(lap_times) + 1)
    
    fuel_consumption = endurance_data.get('fuel_consumption', [])
    reliability_events = endurance_data.get('reliability_events', [])
    
    engine_temps = endurance_data.get('engine_temps', [])
    coolant_temps = endurance_data.get('coolant_temps', [])
    oil_temps = endurance_data.get('oil_temps', [])
    
    component_wear = endurance_data.get('component_wear', {})
    
    # Create figure
    fig = plt.figure(figsize=(15, 12))
    gs = gridspec.GridSpec(3, 2, height_ratios=[2, 1, 1])
    
    # Unit conversions
    if unit_system.lower() == 'imperial':
        fuel_factor = LITERS_TO_GAL  # Convert liters to gallons
        fuel_unit = "gal"
        temp_convert = lambda t: t * 9/5 + 32 if t is not None else None
        temp_unit = "°F"
    else:
        fuel_factor = 1.0
        fuel_unit = "L"
        temp_convert = lambda t: t  # No conversion needed
        temp_unit = "°C"
    
    # Plot lap times
    ax1 = fig.add_subplot(gs[0, 0])
    
    if lap_times:
        ax1.plot(lap_numbers, lap_times, 'b-o', linewidth=DEFAULT_LINE_WIDTH, markersize=5)
        
        # Calculate statistics
        avg_lap_time = np.mean(lap_times)
        min_lap_time = np.min(lap_times)
        
        # Add reference lines
        ax1.axhline(y=avg_lap_time, color='r', linestyle='--', alpha=0.7, 
                  label=f'Avg: {avg_lap_time:.2f}s')
        ax1.axhline(y=min_lap_time, color='g', linestyle='--', alpha=0.7, 
                  label=f'Best: {min_lap_time:.2f}s')
        
        # Mark reliability events
        if reliability_events:
            for event in reliability_events:
                lap = event.get('lap', 0)
                event_type = event.get('type', 'unknown')
                
                if 0 < lap <= len(lap_times):
                    ax1.scatter([lap], [lap_times[lap-1]], color='red', marker='x', s=100, zorder=5)
                    ax1.annotate(event_type, 
                               xy=(lap, lap_times[lap-1]),
                               xytext=(lap, lap_times[lap-1] + 0.5),
                               ha='center',
                               rotation=45,
                               size=8)
        
        ax1.set_xlabel('Lap Number')
        ax1.set_ylabel('Lap Time (s)')
        ax1.set_title('Lap Times')
        ax1.grid(True, alpha=DEFAULT_GRID_ALPHA)
        ax1.legend(loc='best')
    
    ax2.set_xlabel('Engine Speed (RPM)')
    ax2.set_ylabel(f'Power ({power_unit})')
    ax2.grid(True, alpha=DEFAULT_GRID_ALPHA)
    ax2.legend(loc='best')
    
    # Set title
    if title:
        fig.suptitle(title, fontsize=DEFAULT_TITLE_SIZE+2)
    else:
        fig.suptitle('Engine Performance Comparison', fontsize=DEFAULT_TITLE_SIZE+2)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    # Save if requested
    if save_path:
        save_plot(fig, save_path)
    
    return fig


#------------------------------------------------------------------------------
# Acceleration plotting functions
#------------------------------------------------------------------------------

def plot_acceleration_results(acceleration_data: Dict, title: Optional[str] = None,
                            unit_system: str = 'metric',
                            save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot acceleration simulation results.
    
    Args:
        acceleration_data: Dictionary with acceleration simulation results
        title: Plot title
        unit_system: Unit system ('metric' or 'imperial')
        save_path: Path to save plot (if None, not saved)
        
    Returns:
        Matplotlib figure
    """
    # Extract data
    time = acceleration_data.get('time', [])
    speed = acceleration_data.get('speed', [])
    acceleration = acceleration_data.get('acceleration', [])
    distance = acceleration_data.get('position', [])
    rpm = acceleration_data.get('engine_rpm', [])
    gear = acceleration_data.get('gear', [])
    
    # Validate data
    if not time or len(time) != len(speed):
        logger.error("Invalid acceleration data format")
        return None
    
    # Create figure with subplots
    fig = plt.figure(figsize=(12, 12))
    gs = gridspec.GridSpec(3, 2, height_ratios=[2, 1, 1])
    
    # Unit conversions
    if unit_system.lower() == 'imperial':
        speed_factor = MS_TO_MPH  # Convert m/s to mph
        speed_unit = "mph"
        accel_factor = 1.0/9.81  # Convert m/s² to g
        accel_unit = "g"
        distance_factor = 3.28084  # Convert meters to feet
        distance_unit = "ft"
    else:
        speed_factor = MS_TO_KMH  # Convert m/s to km/h
        speed_unit = "km/h"
        accel_factor = 1.0
        accel_unit = "m/s²"
        distance_factor = 1.0
        distance_unit = "m"
    
    # Plot speed vs time
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(time, np.array(speed) * speed_factor, 'b-', linewidth=DEFAULT_LINE_WIDTH)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel(f'Speed ({speed_unit})')
    ax1.set_title('Speed vs Time')
    ax1.grid(True, alpha=DEFAULT_GRID_ALPHA)
    
    # Highlight key points
    if 'time_to_60mph' in acceleration_data and acceleration_data['time_to_60mph'] is not None:
        t60 = acceleration_data['time_to_60mph']
        ax1.axvline(x=t60, color='r', linestyle='--', alpha=0.7)
        ax1.text(t60 + 0.1, 10, f"0-60 mph: {t60:.2f}s", color='r')
    
    if 'time_to_100kph' in acceleration_data and acceleration_data['time_to_100kph'] is not None:
        t100 = acceleration_data['time_to_100kph']
        ax1.axvline(x=t100, color='g', linestyle='--', alpha=0.7)
        ax1.text(t100 + 0.1, 20, f"0-100 km/h: {t100:.2f}s", color='g')
    
    # Plot distance vs time
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(time, np.array(distance) * distance_factor, 'g-', linewidth=DEFAULT_LINE_WIDTH)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel(f'Distance ({distance_unit})')
    ax2.set_title('Distance vs Time')
    ax2.grid(True, alpha=DEFAULT_GRID_ALPHA)
    
    # Highlight finish distance
    if 'finish_time' in acceleration_data and acceleration_data['finish_time'] is not None:
        finish_time = acceleration_data['finish_time']
        finish_dist = acceleration_data['distance']
        ax2.axvline(x=finish_time, color='r', linestyle='--', alpha=0.7)
        ax2.axhline(y=finish_dist * distance_factor, color='r', linestyle='--', alpha=0.7)
        ax2.text(finish_time / 2, finish_dist * distance_factor * 0.8, 
               f"{finish_dist}m in {finish_time:.2f}s", ha='center')
    
    # Plot acceleration
    ax3 = fig.add_subplot(gs[1, 0])
    if acceleration:
        ax3.plot(time, np.array(acceleration) * accel_factor, 'r-', linewidth=DEFAULT_LINE_WIDTH)
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel(f'Acceleration ({accel_unit})')
        ax3.set_title('Acceleration vs Time')
        ax3.grid(True, alpha=DEFAULT_GRID_ALPHA)
    
    # Plot RPM and gear
    ax4 = fig.add_subplot(gs[1, 1])
    if rpm:
        ax4.plot(time, rpm, 'b-', linewidth=DEFAULT_LINE_WIDTH)
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Engine RPM')
        ax4.set_title('Engine RPM vs Time')
        ax4.grid(True, alpha=DEFAULT_GRID_ALPHA)
    
    # Add gear information as a step plot on secondary axis
    if gear:
        ax4_twin = ax4.twinx()
        ax4_twin.step(time, gear, 'r-', linewidth=1.5, where='post', alpha=0.7)
        ax4_twin.set_ylabel('Gear', color='r')
        ax4_twin.tick_params(axis='y', colors='r')
        ax4_twin.set_yticks(range(1, max(gear) + 1))
    
    # Plot speed vs distance
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.plot(np.array(distance) * distance_factor, np.array(speed) * speed_factor, 'm-', 
           linewidth=DEFAULT_LINE_WIDTH)
    ax5.set_xlabel(f'Distance ({distance_unit})')
    ax5.set_ylabel(f'Speed ({speed_unit})')
    ax5.set_title('Speed vs Distance')
    ax5.grid(True, alpha=DEFAULT_GRID_ALPHA)
    
    # Plot gear vs distance
    ax6 = fig.add_subplot(gs[2, 1])
    if gear and distance:
        ax6.step(np.array(distance) * distance_factor, gear, 'g-', 
               linewidth=DEFAULT_LINE_WIDTH, where='post')
        ax6.set_xlabel(f'Distance ({distance_unit})')
        ax6.set_ylabel('Gear')
        ax6.set_title('Gear vs Distance')
        ax6.grid(True, alpha=DEFAULT_GRID_ALPHA)
        ax6.set_yticks(range(1, max(gear) + 1))
        ax6.yaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Set overall title
    if title:
        fig.suptitle(title, fontsize=DEFAULT_TITLE_SIZE+2)
    else:
        fig.suptitle('Acceleration Performance Analysis', fontsize=DEFAULT_TITLE_SIZE+2)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    # Add summary text
    summary_text = ""
    if 'time_to_60mph' in acceleration_data and acceleration_data['time_to_60mph'] is not None:
        summary_text += f"0-60 mph: {acceleration_data['time_to_60mph']:.2f}s | "
    if 'time_to_100kph' in acceleration_data and acceleration_data['time_to_100kph'] is not None:
        summary_text += f"0-100 km/h: {acceleration_data['time_to_100kph']:.2f}s | "
    if 'finish_time' in acceleration_data and acceleration_data['finish_time'] is not None:
        finish_time = acceleration_data['finish_time']
        finish_dist = acceleration_data.get('distance', 75.0)
        summary_text += f"{finish_dist}m time: {finish_time:.2f}s | "
    if 'finish_speed' in acceleration_data and acceleration_data['finish_speed'] is not None:
        finish_speed = acceleration_data['finish_speed'] * speed_factor
        summary_text += f"Final speed: {finish_speed:.1f} {speed_unit}"
    
    if summary_text:
        plt.figtext(0.5, 0.01, summary_text, ha='center', fontsize=DEFAULT_LABEL_SIZE, 
                   bbox=dict(facecolor='white', alpha=0.8, edgecolor='lightgray'))
        plt.subplots_adjust(bottom=0.08)
    
    # Save if requested
    if save_path:
        save_plot(fig, save_path)
    
    return fig


def plot_acceleration_comparison(comparison_data: List[Dict], title: Optional[str] = None,
                               unit_system: str = 'metric', metrics: Optional[List[str]] = None,
                               save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot comparison of multiple acceleration test results.
    
    Args:
        comparison_data: List of dictionaries with acceleration data and labels
        title: Plot title
        unit_system: Unit system ('metric' or 'imperial')
        metrics: List of metrics to compare ('time_to_60mph', 'time_to_100kph', 'finish_time')
        save_path: Path to save plot (if None, not saved)
        
    Returns:
        Matplotlib figure
    """
    if not comparison_data:
        logger.error("No comparison data provided")
        return None
    
    # Default metrics to compare
    if metrics is None:
        metrics = ['time_to_60mph', 'time_to_100kph', 'finish_time']
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Unit conversions
    if unit_system.lower() == 'imperial':
        speed_factor = MS_TO_MPH  # Convert m/s to mph
        speed_unit = "mph"
        distance_factor = 3.28084  # Convert meters to feet
        distance_unit = "ft"
    else:
        speed_factor = MS_TO_KMH  # Convert m/s to km/h
        speed_unit = "km/h"
        distance_factor = 1.0
        distance_unit = "m"
    
    # Colors for different configurations
    colors = COLOR_SCHEMES['formula_student']
    
    # Extract labels and metrics for bar chart
    labels = []
    metric_values = {metric: [] for metric in metrics}
    
    # Plot speed vs time for each configuration
    for i, data in enumerate(comparison_data):
        label = data.get('label', f'Config {i+1}')
        labels.append(label)
        
        time = data.get('time', [])
        speed = data.get('speed', [])
        
        if not time or not speed:
            logger.warning(f"Skipping config {i+1} due to missing time/speed data")
            continue
        
        # Plot speed vs time
        color = data.get('color', colors[i % len(colors)])
        ax1.plot(time, np.array(speed) * speed_factor, '-', 
               color=color, linewidth=DEFAULT_LINE_WIDTH, label=label)
        
        # Collect metrics for bar chart
        for metric in metrics:
            if metric in data and data[metric] is not None:
                metric_values[metric].append(data[metric])
            else:
                metric_values[metric].append(0)
    
    # Set axes labels for speed vs time plot
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel(f'Speed ({speed_unit})')
    ax1.set_title('Speed vs Time Comparison')
    ax1.grid(True, alpha=DEFAULT_GRID_ALPHA)
    ax1.legend(loc='best')
    
    # Create bar chart for metrics
    x = np.arange(len(labels))
    bar_width = 0.8 / len(metrics)
    
    for i, metric in enumerate(metrics):
        offset = (i - len(metrics)/2 + 0.5) * bar_width
        bars = ax2.bar(x + offset, metric_values[metric], bar_width, 
                    label=_format_metric_name(metric))
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                       f'{height:.2f}s', ha='center', va='bottom', fontsize=8)
    
    # Set axes labels for metrics bar chart
    ax2.set_xlabel('Configuration')
    ax2.set_ylabel('Time (s)')
    ax2.set_title('Performance Metrics Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha='right')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=DEFAULT_GRID_ALPHA, axis='y')
    
    # Set overall title
    if title:
        fig.suptitle(title, fontsize=DEFAULT_TITLE_SIZE+2)
    else:
        fig.suptitle('Acceleration Performance Comparison', fontsize=DEFAULT_TITLE_SIZE+2)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # Save if requested
    if save_path:
        save_plot(fig, save_path)
    
    return fig


#------------------------------------------------------------------------------
# Lap time plotting functions
#------------------------------------------------------------------------------

def plot_lap_time_results(lap_data: Dict, title: Optional[str] = None,
                        unit_system: str = 'metric',
                        save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot lap time simulation results.
    
    Args:
        lap_data: Dictionary with lap time simulation results
        title: Plot title
        unit_system: Unit system ('metric' or 'imperial')
        save_path: Path to save plot (if None, not saved)
        
    Returns:
        Matplotlib figure
    """
    # Extract data
    distance = lap_data.get('distance', [])
    speed = lap_data.get('speed', [])
    time = lap_data.get('time', [])
    lateral_g = lap_data.get('lateral_g', [])
    gear = lap_data.get('gear', [])
    
    # Validate data
    if not distance or len(distance) != len(speed):
        logger.error("Invalid lap data format")
        return None
    
    # Create figure with subplots
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(3, 2, height_ratios=[2, 1, 1])
    
    # Unit conversions
    if unit_system.lower() == 'imperial':
        speed_factor = MS_TO_MPH  # Convert m/s to mph
        speed_unit = "mph"
        distance_factor = 3.28084 / 5280  # Convert meters to miles
        distance_unit = "miles"
    else:
        speed_factor = MS_TO_KMH  # Convert m/s to km/h
        speed_unit = "km/h"
        distance_factor = 0.001  # Convert meters to kilometers
        distance_unit = "km"
    
    # Plot speed profile
    ax1 = fig.add_subplot(gs[0, :])
    
    # Color the speed profile by lateral g-force if available
    if lateral_g and len(lateral_g) == len(distance):
        points = np.array([distance, np.array(speed) * speed_factor]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        norm = plt.Normalize(0, np.percentile(lateral_g, 95))  # Cap at 95th percentile for better color scale
        lc = plt.matplotlib.collections.LineCollection(segments, cmap=ACCELERATION_CMAP, norm=norm)
        lc.set_array(np.array(lateral_g))
        lc.set_linewidth(DEFAULT_LINE_WIDTH)
        ax1.add_collection(lc)
        
        # Add colorbar
        cbar = plt.colorbar(lc, ax=ax1)
        cbar.set_label('Lateral G')
    else:
        # Simple line plot if no lateral g data
        ax1.plot(np.array(distance) * distance_factor, np.array(speed) * speed_factor, 
               'b-', linewidth=DEFAULT_LINE_WIDTH)
    
    # Set axis limits
    ax1.set_xlim(min(distance) * distance_factor, max(distance) * distance_factor)
    ax1.set_ylim(0, max(np.array(speed) * speed_factor) * 1.1)
    
    # Set labels and title
    ax1.set_xlabel(f'Distance ({distance_unit})')
    ax1.set_ylabel(f'Speed ({speed_unit})')
    ax1.set_title('Speed Profile')
    ax1.grid(True, alpha=DEFAULT_GRID_ALPHA)
    
    # Plot track layout if available
    if 'track_points' in lap_data:
        ax2 = fig.add_subplot(gs[1, 0])
        track_points = lap_data['track_points']
        
        # Plot track centerline
        ax2.plot(track_points[:, 0], track_points[:, 1], 'k-', alpha=0.5, linewidth=1, label='Track')
        
        # Plot racing line if available
        if 'racing_line' in lap_data:
            racing_line = lap_data['racing_line']
            ax2.plot(racing_line[:, 0], racing_line[:, 1], 'r-', linewidth=2, label='Racing Line')
        
        # Set equal aspect ratio
        ax2.set_aspect('equal')
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.set_title('Track Layout')
        ax2.legend(loc='best')
    else:
        # Plot lateral g-force
        ax2 = fig.add_subplot(gs[1, 0])
        if lateral_g:
            ax2.plot(np.array(distance) * distance_factor, lateral_g, 
                   'g-', linewidth=DEFAULT_LINE_WIDTH)
            ax2.set_xlabel(f'Distance ({distance_unit})')
            ax2.set_ylabel('Lateral G')
            ax2.set_title('Lateral G-Force')
            ax2.grid(True, alpha=DEFAULT_GRID_ALPHA)
    
    # Plot gear profile
    ax3 = fig.add_subplot(gs[1, 1])
    if gear:
        ax3.step(np.array(distance) * distance_factor, gear, 'r-', 
               where='post', linewidth=DEFAULT_LINE_WIDTH)
        ax3.set_xlabel(f'Distance ({distance_unit})')
        ax3.set_ylabel('Gear')
        ax3.set_title('Gear Profile')
        ax3.grid(True, alpha=DEFAULT_GRID_ALPHA)
        ax3.set_yticks(range(1, max(gear) + 1))
        ax3.yaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Plot sector times if available
    if 'sector_times' in lap_data:
        ax4 = fig.add_subplot(gs[2, 0])
        sector_times = lap_data['sector_times']
        
        # Extract sector data
        sector_numbers = [s.get('sector', i+1) for i, s in enumerate(sector_times)]
        sector_time_values = [s.get('time', 0) for s in sector_times]
        
        # Create bar chart
        bars = ax4.bar(sector_numbers, sector_time_values, color='teal')
        
        # Add time labels
        for bar, time_val in zip(bars, sector_time_values):
            ax4.text(bar.get_x() + bar.get_width()/2., time_val + 0.1,
                   f'{time_val:.1f}s', ha='center', va='bottom', fontsize=8)
        
        ax4.set_xlabel('Sector')
        ax4.set_ylabel('Time (s)')
        ax4.set_title('Sector Times')
        ax4.grid(True, alpha=DEFAULT_GRID_ALPHA, axis='y')
        ax4.set_xticks(sector_numbers)
    
    # Plot speed distribution or engine RPM if available
    ax5 = fig.add_subplot(gs[2, 1])
    if 'engine_rpm' in lap_data:
        rpm = lap_data['engine_rpm']
        if rpm:
            ax5.plot(np.array(distance) * distance_factor, rpm, 
                   'b-', linewidth=DEFAULT_LINE_WIDTH)
            ax5.set_xlabel(f'Distance ({distance_unit})')
            ax5.set_ylabel('Engine RPM')
            ax5.set_title('Engine RPM')
            ax5.grid(True, alpha=DEFAULT_GRID_ALPHA)
    else:
        # Create a speed histogram
        ax5.hist(np.array(speed) * speed_factor, bins=20, color='purple', alpha=0.7)
        ax5.set_xlabel(f'Speed ({speed_unit})')
        ax5.set_ylabel('Frequency')
        ax5.set_title('Speed Distribution')
        ax5.grid(True, alpha=DEFAULT_GRID_ALPHA, axis='y')
    
    # Set overall title
    if title:
        fig.suptitle(title, fontsize=DEFAULT_TITLE_SIZE+2)
    else:
        if 'lap_time' in lap_data:
            lap_time = lap_data['lap_time']
            fig.suptitle(f'Lap Time Analysis: {lap_time:.3f}s', fontsize=DEFAULT_TITLE_SIZE+2)
        else:
            fig.suptitle('Lap Time Analysis', fontsize=DEFAULT_TITLE_SIZE+2)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    # Add summary text
    summary_text = ""
    if 'lap_time' in lap_data:
        summary_text += f"Lap Time: {lap_data['lap_time']:.3f}s | "
    if 'avg_speed' in lap_data:
        avg_speed = lap_data['avg_speed'] * speed_factor
        summary_text += f"Avg Speed: {avg_speed:.1f} {speed_unit} | "
    if 'max_speed' in lap_data:
        max_speed = lap_data['max_speed'] * speed_factor
        summary_text += f"Max Speed: {max_speed:.1f} {speed_unit} | "
    if 'max_lateral_g' in lap_data:
        summary_text += f"Max Lateral G: {lap_data['max_lateral_g']:.2f}g"
    
    if summary_text:
        plt.figtext(0.5, 0.01, summary_text, ha='center', fontsize=DEFAULT_LABEL_SIZE, 
                   bbox=dict(facecolor='white', alpha=0.8, edgecolor='lightgray'))
        plt.subplots_adjust(bottom=0.08)
    
    # Save if requested
    if save_path:
        save_plot(fig, save_path)
    
    return fig


def plot_lap_time_comparison(comparison_data: List[Dict], title: Optional[str] = None,
                           unit_system: str = 'metric',
                           save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot comparison of multiple lap time simulation results.
    
    Args:
        comparison_data: List of dictionaries with lap data and labels
        title: Plot title
        unit_system: Unit system ('metric' or 'imperial')
        save_path: Path to save plot (if None, not saved)
        
    Returns:
        Matplotlib figure
    """
    if not comparison_data:
        logger.error("No comparison data provided")
        return None
    
    # Create figure
    fig = plt.figure(figsize=(15, 12))
    gs = gridspec.GridSpec(3, 2, height_ratios=[2, 1, 1])
    
    # Unit conversions
    if unit_system.lower() == 'imperial':
        speed_factor = MS_TO_MPH  # Convert m/s to mph
        speed_unit = "mph"
        distance_factor = 3.28084 / 5280  # Convert meters to miles
        distance_unit = "miles"
    else:
        speed_factor = MS_TO_KMH  # Convert m/s to km/h
        speed_unit = "km/h"
        distance_factor = 0.001  # Convert meters to kilometers
        distance_unit = "km"
    
    # Colors for different configurations
    colors = COLOR_SCHEMES['formula_student']
    
    # Extract lap times and config names for bar chart
    config_names = []
    lap_times = []
    avg_speeds = []
    max_speeds = []
    
    # Plot speed profiles
    ax1 = fig.add_subplot(gs[0, :])
    
    for i, data in enumerate(comparison_data):
        config_name = data.get('label', f'Config {i+1}')
        config_names.append(config_name)
        
        distance = data.get('distance', [])
        speed = data.get('speed', [])
        
        if not distance or not speed:
            logger.warning(f"Skipping config {i+1} due to missing distance/speed data")
            continue
        
        # Plot speed profile
        color = data.get('color', colors[i % len(colors)])
        ax1.plot(np.array(distance) * distance_factor, np.array(speed) * speed_factor, 
               '-', color=color, linewidth=DEFAULT_LINE_WIDTH, label=config_name)
        
        # Collect metrics for bar charts
        if 'lap_time' in data:
            lap_times.append(data['lap_time'])
        else:
            lap_times.append(0)
        
        if 'avg_speed' in data:
            avg_speeds.append(data['avg_speed'] * speed_factor)
        else:
            avg_speeds.append(0)
        
        if 'max_speed' in data:
            max_speeds.append(data['max_speed'] * speed_factor)
        else:
            max_speeds.append(0)
    
    # Set axis labels and title for speed profile plot
    ax1.set_xlabel(f'Distance ({distance_unit})')
    ax1.set_ylabel(f'Speed ({speed_unit})')
    ax1.set_title('Speed Profile Comparison')
    ax1.grid(True, alpha=DEFAULT_GRID_ALPHA)
    ax1.legend(loc='best')