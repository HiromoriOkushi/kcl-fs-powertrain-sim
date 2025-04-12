# kcl_fs_powertrain/utils/plotting.py
"""
Plotting utilities for Formula Student powertrain simulation.

This module provides a comprehensive set of plotting functions for visualizing
simulation results, vehicle performance metrics, and component behavior.
It aims to create consistent and informative plots for analysis.

Functions are designed to accept data primarily through dictionaries containing
NumPy arrays or basic Python types to minimize dependencies on other simulation modules.
Calling modules are responsible for extracting and formatting data correctly.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LinearSegmentedColormap, Normalize
import pandas as pd # Keep pandas for potential DataFrame input/output if needed by callers, but avoid internal dependency
from typing import Dict, List, Tuple, Optional, Union, Any, Callable # Added Callable
import os
import logging
import json # For potential export helpers later if needed

# Import only necessary constants from the sibling module
from .constants import (
    MS_TO_KMH, MS_TO_MPH, KG_TO_LBS, KW_TO_HP, HP_TO_KW, LITERS_TO_GAL,
    NM_TO_LBFT, LBFT_TO_NM, M_TO_INCH, M_TO_MM, M_TO_KM, GRAVITY,
    DEFAULT_AMBIENT_TEMP, DEFAULT_ENGINE_OPERATING_TEMP,
    DEFAULT_COOLANT_OPERATING_TEMP, DEFAULT_OIL_OPERATING_TEMP
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Plotting")

# Default style settings for plots
DEFAULT_FIG_SIZE = (12, 8)
DEFAULT_DPI = 120
DEFAULT_LINE_WIDTH = 1.8
DEFAULT_MARKER_SIZE = 5
DEFAULT_FONT_SIZE = 9
DEFAULT_TITLE_SIZE = 12
DEFAULT_LABEL_SIZE = 10
DEFAULT_LEGEND_SIZE = 9
DEFAULT_GRID_ALPHA = 0.4
DEFAULT_SAVE_FORMAT = 'png'

# Color schemes
COLOR_SCHEMES = {
    'default': plt.cm.tab10.colors,
    'formula_student': ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf'],
    'thermal': plt.cm.coolwarm,
    'speed': plt.cm.viridis,
    'acceleration': plt.cm.plasma
}
THERMAL_CMAP = COLOR_SCHEMES['thermal']
SPEED_CMAP = COLOR_SCHEMES['speed']


#------------------------------------------------------------------------------
# Utility functions (Internal to plotting)
#------------------------------------------------------------------------------

def set_plot_style(style: str = 'default') -> None:
    """
    Set global matplotlib style for consistent plots.

    Args:
        style: Style name ('default', 'clean', 'presentation', 'publication', 'seaborn')
    """
    styles = {
        'default': 'default',
        'clean': 'seaborn-v0_8-whitegrid',
        'presentation': 'seaborn-v0_8-talk',
        'publication': 'seaborn-v0_8-paper',
        'seaborn': 'seaborn-v0_8-darkgrid'
    }
    if style in plt.style.available:
        plt.style.use(style)
    elif style in styles:
        plt.style.use(styles[style])
    else:
        logger.warning(f"Unknown style: {style}. Using default.")
        plt.style.use(styles['default'])

    plt.rcParams.update({
        'font.size': DEFAULT_FONT_SIZE,
        'axes.titlesize': DEFAULT_TITLE_SIZE,
        'axes.labelsize': DEFAULT_LABEL_SIZE,
        'xtick.labelsize': DEFAULT_FONT_SIZE,
        'ytick.labelsize': DEFAULT_FONT_SIZE,
        'legend.fontsize': DEFAULT_LEGEND_SIZE,
        'figure.figsize': DEFAULT_FIG_SIZE,
        'figure.dpi': DEFAULT_DPI,
        'lines.linewidth': DEFAULT_LINE_WIDTH,
        'lines.markersize': DEFAULT_MARKER_SIZE,
        'grid.alpha': DEFAULT_GRID_ALPHA,
        'grid.linestyle': '--'
    })
    logger.info(f"Plot style set to '{style}'")


def save_plot(fig: plt.Figure, filename: str, directory: Optional[str] = None,
             plot_format: str = DEFAULT_SAVE_FORMAT, dpi: int = DEFAULT_DPI) -> Optional[str]:
    """
    Save a plot to file with proper directory handling.

    Args:
        fig: Matplotlib figure to save.
        filename: Base filename (without extension).
        directory: Directory to save in (created if doesn't exist).
        plot_format: File format ('png', 'pdf', 'svg', etc.).
        dpi: Resolution for raster formats.

    Returns:
        Full path to saved file, or None if saving failed.
    """
    if fig is None:
        logger.error("Cannot save plot: Figure object is None.")
        return None

    format_lower = plot_format.lower()
    if '.' in filename:
        base, ext = os.path.splitext(filename)
        if ext and ext[1:].lower() != format_lower:
            logger.warning(f"Filename extension ({ext}) doesn't match format ({format_lower}). Using {format_lower}.")
            filename = base
    filepath = f"{filename}.{format_lower}"

    if directory:
        try:
            os.makedirs(directory, exist_ok=True)
            filepath = os.path.join(directory, filepath)
        except OSError as e:
            logger.error(f"Could not create directory {directory}: {e}")
            return None

    try:
        fig.savefig(filepath, format=format_lower, dpi=dpi, bbox_inches='tight')
        logger.info(f"Plot saved to {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"Failed to save plot to {filepath}: {e}")
        return None


def _format_metric_name(metric: str) -> str:
    """Format metric name for display in plots."""
    name_map = {
        'time_to_60mph': '0-60 mph Time', 'time_to_100kph': '0-100 km/h Time',
        'finish_time': '75m Time', 'acceleration_time': '75m Time',
        'peak_acceleration_g': 'Peak Accel (g)', 'avg_speed_kph': 'Avg Speed (km/h)',
        'max_speed_kph': 'Max Speed (km/h)', 'max_lateral_g': 'Max Lateral (g)',
        'lap_time': 'Lap Time', 'lap_time_autocross': 'Autocross Lap Time',
        'engine_temp': 'Engine Temp', 'coolant_temp': 'Coolant Temp', 'oil_temp': 'Oil Temp',
        'max_engine_temp': 'Max Engine Temp', 'max_coolant_temp': 'Max Coolant Temp', 'max_oil_temp': 'Max Oil Temp',
        'power_to_weight': 'Power/Weight (kW/kg)', 'specific_power': 'Specific Power (HP/L)',
        'vehicle_mass': 'Vehicle Mass (kg)', 'weight_distribution': 'Front Weight Dist (%)',
        'skidpad_time': 'Skidpad Time', 'avg_speed_autocross': 'Autocross Avg Speed (km/h)',
        'fuel_efficiency_endurance': 'Endurance Fuel Eff (L/lap)', 'max_speed': 'Top Speed (km/h)',
        'average_lap_time': 'Avg Lap Time',
        'speed_at_end_of_accel': 'Accel End Speed',
        'average_rate_gs': 'Avg Fuel Rate (g/s)',
        'skidpad_consistency': 'Skidpad Consistency Error',
        'max_wear_percent': 'Max Component Wear (%)',
        'reliability_events_count': 'Reliability Issues (#)',
        'front_weight_pct': 'Front Weight (%)',
        'max_longitudinal_accel_g': 'Max Lon Accel (g)',
        'max_braking_decel_g': 'Max Braking (g)',
        'thermal_factor': 'Thermal Perf. Factor',
        'engine_block_temp': 'Engine Block Temp',
        'heat_rejection_W': 'Heat Rejection (W)',
        'heat_rejected_kw': 'Heat Rejected (kW)',
        'airflow_m3s': 'Airflow (m³/s)',
        'supplementary_airflow_m3s': 'Suppl. Airflow (m³/s)',
        'power_W': 'Power (W)',
        'coolant_flow_lpm': 'Coolant Flow (LPM)',
        'drag_N': 'Drag (N)',
        'downforce_N': 'Downforce (N)',
        'total_drag_N': 'Total Drag (N)',
        'total_downforce_N': 'Total Downforce (N)',
        'wheel_slip': 'Wheel Slip',
        'throttle_effective': 'Effective Throttle'
    }
    return name_map.get(metric, metric.replace('_', ' ').title())


def _apply_common_ax_settings(ax: plt.Axes, xlabel: str = "", ylabel: str = "", title: str = "", legend: bool = True):
    """Apply common settings to a matplotlib Axes object."""
    if xlabel: ax.set_xlabel(xlabel, fontsize=DEFAULT_LABEL_SIZE)
    if ylabel: ax.set_ylabel(ylabel, fontsize=DEFAULT_LABEL_SIZE)
    if title: ax.set_title(title, fontsize=DEFAULT_TITLE_SIZE)
    ax.grid(True, alpha=DEFAULT_GRID_ALPHA, linestyle='--')
    ax.tick_params(axis='both', which='major', labelsize=DEFAULT_FONT_SIZE)
    if legend:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc='best', fontsize=DEFAULT_LEGEND_SIZE)


def _plot_safety_lines(ax: plt.Axes, limits: Dict, temp_convert_func: Callable):
    """Helper to plot warning/critical lines on a temperature axis."""
    # Use .get() with default=None to safely access limits
    engine_warning = limits.get('engine_warning')
    engine_critical = limits.get('engine_critical')
    coolant_warning = limits.get('coolant_warning')
    coolant_critical = limits.get('coolant_critical')
    oil_warning = limits.get('oil_warning')
    oil_critical = limits.get('oil_critical')

    # Plot only if the limit value is not None
    if engine_warning is not None: ax.axhline(temp_convert_func(engine_warning), color='orange', linestyle='--', alpha=0.7, label='Eng Warn')
    if engine_critical is not None: ax.axhline(temp_convert_func(engine_critical), color='red', linestyle='-', alpha=0.7, label='Eng Crit')
    if coolant_warning is not None: ax.axhline(temp_convert_func(coolant_warning), color='cyan', linestyle='--', alpha=0.7, label='Cool Warn')
    if coolant_critical is not None: ax.axhline(temp_convert_func(coolant_critical), color='blue', linestyle='-', alpha=0.7, label='Cool Crit')
    if oil_warning is not None: ax.axhline(temp_convert_func(oil_warning), color='darkorange', linestyle='--', alpha=0.7, label='Oil Warn')
    if oil_critical is not None: ax.axhline(temp_convert_func(oil_critical), color='darkred', linestyle='-', alpha=0.7, label='Oil Crit')


#------------------------------------------------------------------------------
# Engine plotting functions
#------------------------------------------------------------------------------

def plot_engine_performance(engine_data: Dict, title: Optional[str] = None,
                          unit_system: str = 'metric', show_efficiency: bool = False,
                          save_path: Optional[str] = None) -> Optional[plt.Figure]:
    """
    Plot engine performance curves (torque, power, efficiency).

    Args:
        engine_data: Dictionary with 'rpm', 'torque', 'power' arrays (power in kW).
                     Optional: 'efficiency' (0-1), 'max_torque_rpm', 'max_torque' (Nm),
                               'max_power_rpm', 'max_power' (kW), 'idle_rpm', 'redline_rpm'.
        title: Plot title.
        unit_system: Unit system ('metric' or 'imperial').
        show_efficiency: Whether to show efficiency curves if available.
        save_path: Path to save plot (if None, not saved).

    Returns:
        Matplotlib figure or None if error.
    """
    try:
        rpm = np.array(engine_data['rpm'])
        torque = np.array(engine_data['torque'])
        power = np.array(engine_data['power']) # Assume power is in kW
        efficiency = np.array(engine_data.get('efficiency', [])) # Optional

        if len(rpm) < 2 or len(rpm) != len(torque) or len(rpm) != len(power):
            raise ValueError("Data length mismatch or insufficient points.")
    except KeyError as e:
        logger.error(f"Missing required key in engine_data for plotting: {e}")
        return None
    except ValueError as e:
        logger.error(f"Invalid data for engine plot: {e}")
        return None

    fig, ax1 = plt.subplots(figsize=DEFAULT_FIG_SIZE)

    # Unit conversions
    if unit_system.lower() == 'imperial':
        torque_factor, torque_unit = NM_TO_LBFT, "lb-ft"
        power_factor, power_unit = KW_TO_HP, "HP"
    else:
        torque_factor, torque_unit = 1.0, "Nm"
        power_factor, power_unit = 1.0, "kW"

    torque_line, = ax1.plot(rpm, torque * torque_factor, color=COLOR_SCHEMES['default'][0], label=f"Torque ({torque_unit})")
    _apply_common_ax_settings(ax1, xlabel='Engine Speed (RPM)', ylabel=f'Torque ({torque_unit})', legend=False)
    ax1.tick_params(axis='y', labelcolor=COLOR_SCHEMES['default'][0])

    ax2 = ax1.twinx()
    power_line, = ax2.plot(rpm, power * power_factor, color=COLOR_SCHEMES['default'][1], label=f"Power ({power_unit})")
    ax2.set_ylabel(f'Power ({power_unit})', color=COLOR_SCHEMES['default'][1], fontsize=DEFAULT_LABEL_SIZE)
    ax2.tick_params(axis='y', labelcolor=COLOR_SCHEMES['default'][1])

    lines = [torque_line, power_line]; labels = [l.get_label() for l in lines]

    if show_efficiency and len(efficiency) == len(rpm):
        ax3 = ax1.twinx()
        ax3.spines['right'].set_position(('outward', 60))
        eff_line, = ax3.plot(rpm, efficiency * 100, color=COLOR_SCHEMES['default'][2], linestyle='--', label="Efficiency (%)")
        ax3.set_ylabel('Efficiency (%)', color=COLOR_SCHEMES['default'][2], fontsize=DEFAULT_LABEL_SIZE)
        ax3.tick_params(axis='y', labelcolor=COLOR_SCHEMES['default'][2])
        ax3.set_ylim(0, max(50, np.nanmax(efficiency*100)*1.1) if np.any(np.isfinite(efficiency)) else 50)
        lines.append(eff_line); labels.append(eff_line.get_label())

    ax1.legend(lines, labels, loc='best')
    plot_title = title if title else 'Engine Performance Curves'
    ax1.set_title(plot_title, fontsize=DEFAULT_TITLE_SIZE)

    # Highlight peaks
    if 'max_torque_rpm' in engine_data and 'max_torque' in engine_data:
        rpm_tq, val_tq = engine_data['max_torque_rpm'], engine_data['max_torque'] * torque_factor
        ax1.plot(rpm_tq, val_tq, 'o', color=COLOR_SCHEMES['default'][0], markersize=DEFAULT_MARKER_SIZE+2)
        ax1.text(rpm_tq, val_tq*1.02, f"{val_tq:.1f} {torque_unit}\n@{rpm_tq:.0f} RPM", ha='center', va='bottom', fontsize=DEFAULT_FONT_SIZE-1)
    if 'max_power_rpm' in engine_data and 'max_power' in engine_data:
        rpm_pw, val_pw_kw = engine_data['max_power_rpm'], engine_data['max_power']
        val_pw_display = val_pw_kw * power_factor
        ax2.plot(rpm_pw, val_pw_display, 'o', color=COLOR_SCHEMES['default'][1], markersize=DEFAULT_MARKER_SIZE+2)
        ax2.text(rpm_pw, val_pw_display*0.98, f"{val_pw_display:.1f} {power_unit}\n@{rpm_pw:.0f} RPM", ha='center', va='top', fontsize=DEFAULT_FONT_SIZE-1)

    # Set x-axis limits
    if 'idle_rpm' in engine_data and 'redline_rpm' in engine_data: ax1.set_xlim(engine_data['idle_rpm'], engine_data['redline_rpm'])
    elif len(rpm) > 0: ax1.set_xlim(rpm[0], rpm[-1])

    plt.tight_layout()
    if save_path: save_plot(fig, save_path)
    return fig


def plot_torque_curves_comparison(curves_data: List[Dict], title: Optional[str] = None,
                                unit_system: str = 'metric',
                                save_path: Optional[str] = None) -> Optional[plt.Figure]:
    """
    Plot comparison of multiple torque curves.

    Args:
        curves_data: List of dictionaries, each with 'rpm', 'torque', 'power' (kW), 'label'.
        title: Plot title.
        unit_system: Unit system ('metric' or 'imperial').
        save_path: Path to save plot (if None, not saved).

    Returns:
        Matplotlib figure or None if error.
    """
    if not curves_data: logger.error("No curve data provided for comparison."); return None

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    if unit_system.lower() == 'imperial': torque_factor, torque_unit = NM_TO_LBFT, "lb-ft"; power_factor, power_unit = KW_TO_HP, "HP"
    else: torque_factor, torque_unit = 1.0, "Nm"; power_factor, power_unit = 1.0, "kW"

    colors = COLOR_SCHEMES['default']
    min_rpm_all, max_rpm_all = float('inf'), float('-inf')

    for i, curve in enumerate(curves_data):
        try:
            rpm = np.array(curve['rpm']); torque = np.array(curve['torque']); power = np.array(curve['power'])
            label = curve.get('label', f'Curve {i+1}'); color = curve.get('color', colors[i % len(colors)])
            if len(rpm) < 2 or len(rpm) != len(torque) or len(rpm) != len(power): raise ValueError("Data mismatch")
            min_rpm_all = min(min_rpm_all, rpm[0]); max_rpm_all = max(max_rpm_all, rpm[-1])
            ax1.plot(rpm, torque * torque_factor, '-', color=color, label=label)
            ax2.plot(rpm, power * power_factor, '-', color=color, label=label)
        except (KeyError, ValueError) as e:
            logger.warning(f"Skipping curve '{curve.get('label', i+1)}' due to invalid data: {e}")

    if min_rpm_all == float('inf'): logger.error("No valid curves plotted."); return None

    _apply_common_ax_settings(ax1, ylabel=f'Torque ({torque_unit})', title='Torque Comparison')
    _apply_common_ax_settings(ax2, xlabel='Engine Speed (RPM)', ylabel=f'Power ({power_unit})', title='Power Comparison')
    ax1.set_xlim(min_rpm_all, max_rpm_all)

    plot_title = title if title else 'Torque Curve Comparison'
    fig.suptitle(plot_title, fontsize=DEFAULT_TITLE_SIZE+2)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_path: save_plot(fig, save_path)
    return fig


#------------------------------------------------------------------------------
# Track plotting functions
#------------------------------------------------------------------------------
def plot_track_layout(track_data: Dict, show_racing_line: bool = True,
                    show_segments: bool = True, show_elevation: bool = False,
                    title: Optional[str] = None, save_path: Optional[str] = None) -> Optional[plt.Figure]:
    """
    Plot track layout with optional racing line and segments.

    Args:
        track_data: Dictionary with track data. Expected keys:
                    'points': Nx2 array of centerline points.
                    Optional: 'width' (scalar or Nx1 array), 'racing_line' (Mx2 array),
                              'segments' (list of dicts {'type', 'start_idx', 'end_idx'}),
                              'elevation' (Nx1 array), 'distance' (Nx1 array),
                              'start_position' (1x2 array), 'start_direction' (scalar, radians),
                              'name' (str), 'length' (float).
        show_racing_line: Whether to show racing line if available.
        show_segments: Whether to show track segments if available.
        show_elevation: Whether to add an elevation profile subplot.
        title: Plot title.
        save_path: Path to save plot (if None, not saved).

    Returns:
        Matplotlib figure or None if error.
    """
    try:
        track_points = np.array(track_data['points'])
        if len(track_points) < 2: raise ValueError("Requires at least 2 track points.")
    except (KeyError, ValueError) as e:
        logger.error(f"Invalid track data for plotting: {e}")
        return None

    # Extract optional data safely
    track_width_data = track_data.get('width')
    racing_line = np.array(track_data.get('racing_line')) if track_data.get('racing_line') is not None else None
    segments = track_data.get('segments', [])
    elevation = np.array(track_data.get('elevation')) if track_data.get('elevation') is not None else None
    distances = np.array(track_data.get('distance')) if track_data.get('distance') is not None else None

    # Setup figure
    if show_elevation and elevation is not None and len(elevation) == len(track_points):
        fig = plt.figure(figsize=(15, 12)); gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
        ax1 = fig.add_subplot(gs[0]); ax2 = fig.add_subplot(gs[1])
    else:
        fig = plt.figure(figsize=(12, 10)); ax1 = plt.gca(); ax2 = None

    # Calculate distances if needed for elevation plot
    if ax2 is not None and (distances is None or len(distances) != len(track_points)):
        distances = np.zeros(len(track_points))
        for i in range(1, len(track_points)):
            distances[i] = distances[i-1] + np.linalg.norm(track_points[i] - track_points[i-1])

    # Plot Centerline
    ax1.plot(track_points[:, 0], track_points[:, 1], 'k-', alpha=0.7, linewidth=1.5, label='Track Centerline')

    # Plot Boundaries
    track_width = None
    if isinstance(track_width_data, (int, float)): track_width = np.full(len(track_points), float(track_width_data))
    elif isinstance(track_width_data, (list, np.ndarray)) and len(track_width_data) == len(track_points): track_width = np.array(track_width_data)

    if track_width is not None:
        # Robust normals calculation
        normals = np.zeros_like(track_points)
        tangents = np.gradient(track_points, axis=0); norms = np.linalg.norm(tangents, axis=1)
        valid = norms > 1e-6
        normals[valid, 0] = -tangents[valid, 1] / norms[valid]; normals[valid, 1] = tangents[valid, 0] / norms[valid]
        if np.linalg.norm(track_points[0] - track_points[-1]) < 1e-3: # Closed loop
            normals[0] = normals[-1] = (normals[1] + normals[-2]) / 2.0; norm_val = np.linalg.norm(normals[0])
            if norm_val > 1e-6: normals[0] /= norm_val; normals[-1] = normals[0]
        left_boundary = track_points + normals * (track_width / 2.0)[:, np.newaxis]
        right_boundary = track_points - normals * (track_width / 2.0)[:, np.newaxis]
        ax1.plot(left_boundary[:, 0], left_boundary[:, 1], 'k-', alpha=0.3, linewidth=1)
        ax1.plot(right_boundary[:, 0], right_boundary[:, 1], 'k-', alpha=0.3, linewidth=1)

    # Plot Racing Line
    if show_racing_line and racing_line is not None and len(racing_line) > 1:
        ax1.plot(racing_line[:, 0], racing_line[:, 1], 'r-', label='Racing Line')

    # Plot Segments
    if show_segments and segments:
        segment_colors = {'STRAIGHT': 'green', 'CORNER_LEFT': 'blue', 'CORNER_RIGHT': 'red', 'HAIRPIN_LEFT': 'cyan', 'HAIRPIN_RIGHT': 'magenta', 'UNKNOWN': 'gray'}
        plotted_labels = set()
        for segment in segments:
            seg_type = segment.get('type', 'UNKNOWN').upper(); color = segment_colors.get(seg_type, 'gray'); label = seg_type.replace('_', ' ').title()
            start, end = segment.get('start_idx', 0), segment.get('end_idx', len(track_points) - 1)
            if 0 <= start <= end < len(track_points):
                pts = track_points[start : end + 1]
                current_label = label if label not in plotted_labels else ""
                ax1.plot(pts[:, 0], pts[:, 1], '-', color=color, linewidth=4, alpha=0.6, label=current_label)
                plotted_labels.add(label)

    # Plot Start/Finish
    start_pos = track_data.get('start_position'); start_dir = track_data.get('start_direction')
    if start_pos is not None:
        ax1.scatter(start_pos[0], start_pos[1], color='lime', marker='o', s=100, label='Start/Finish', zorder=5, edgecolors='k')
        if start_dir is not None:
            dir_vec = np.array([np.cos(start_dir), np.sin(start_dir)]) * 5
            ax1.arrow(start_pos[0], start_pos[1], dir_vec[0], dir_vec[1], head_width=1.5, head_length=2.0, fc='lime', ec='k', zorder=5)

    ax1.set_aspect('equal', adjustable='box');
    track_name = track_data.get('name', 'Track'); track_len = track_data.get('length', 0)
    ax1_title = f"{track_name} (Length: {track_len:.1f}m)" if track_len else track_name
    _apply_common_ax_settings(ax1, xlabel='X (m)', ylabel='Y (m)', title=ax1_title)

    # Plot Elevation
    if ax2 is not None and distances is not None and elevation is not None and len(distances) == len(elevation):
        ax2.plot(distances, elevation, 'g-')
        gain = np.sum(np.diff(elevation)[np.diff(elevation)>0]); loss = np.sum(np.diff(elevation)[np.diff(elevation)<0])
        info = f"Gain: {gain:.1f}m | Loss: {abs(loss):.1f}m"
        _apply_common_ax_settings(ax2, xlabel='Distance (m)', ylabel='Elevation (m)', title=f'Elevation Profile ({info})', legend=False)
    elif ax2 is not None:
        ax2.text(0.5, 0.5, "Elevation/Distance data unavailable", ha='center', va='center')
        _apply_common_ax_settings(ax2, title='Elevation Profile', legend=False)


    plot_title = title if title else "Track Layout Analysis"
    fig.suptitle(plot_title, fontsize=DEFAULT_TITLE_SIZE+2)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save_path: save_plot(fig, save_path)
    return fig


def plot_racing_line_analysis(racing_line_data: Dict, title: Optional[str] = None,
                            unit_system: str = 'metric',
                            save_path: Optional[str] = None) -> Optional[plt.Figure]:
    """
    Plot racing line analysis including curvature and speed profiles.

    Args:
        racing_line_data: Dictionary with racing line data. Expected keys:
                          'line' (Nx2 array), 'distances' (N array),
                          'curvature' (N array), 'speed_profile' (N array).
                          Optional: 'track_points' (Mx2), 'time_profile' (N array), 'lap_time' (float).
        title: Plot title.
        unit_system: Unit system ('metric' or 'imperial').
        save_path: Path to save plot (if None, not saved).

    Returns:
        Matplotlib figure or None if error.
    """
    try:
        line = np.array(racing_line_data['line']); distances = np.array(racing_line_data['distances'])
        curvature = np.array(racing_line_data['curvature']); speed_profile = np.array(racing_line_data['speed_profile'])
        if len(line) < 2 or not all(len(arr) == len(line) for arr in [distances, curvature, speed_profile]): raise ValueError("Data mismatch")
    except (KeyError, ValueError) as e:
        logger.error(f"Invalid racing line data for plotting: {e}"); return None

    time_profile = np.array(racing_line_data.get('time_profile', []))
    track_points = np.array(racing_line_data.get('track_points', []))
    lap_time = racing_line_data.get('lap_time')

    fig = plt.figure(figsize=(15, 12)); gs = gridspec.GridSpec(3, 2, height_ratios=[2, 1, 1])

    if unit_system.lower() == 'imperial': speed_factor, speed_unit = MS_TO_MPH, "mph"; dist_factor, dist_unit = M_TO_KM * 0.621371, "miles"
    else: speed_factor, speed_unit = MS_TO_KMH, "km/h"; dist_factor, dist_unit = M_TO_KM, "km"

    # Plot track layout with racing line colored by speed
    ax1 = fig.add_subplot(gs[0, 0])
    if len(track_points) > 1: ax1.plot(track_points[:, 0], track_points[:, 1], 'k--', alpha=0.4, linewidth=1, label='Track Centerline')
    points = line.reshape(-1, 1, 2); segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = Normalize(vmin=np.min(speed_profile * speed_factor), vmax=np.max(speed_profile * speed_factor))
    lc = plt.matplotlib.collections.LineCollection(segments, cmap=SPEED_CMAP, norm=norm); lc.set_array(speed_profile * speed_factor); lc.set_linewidth(DEFAULT_LINE_WIDTH)
    lc_handle = ax1.add_collection(lc); cbar = plt.colorbar(lc_handle, ax=ax1); cbar.set_label(f'Speed ({speed_unit})')
    ax1.set_aspect('equal', adjustable='box'); _apply_common_ax_settings(ax1, xlabel='X (m)', ylabel='Y (m)', title='Racing Line colored by Speed')

    # Plot speed profile vs distance
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(distances * dist_factor, speed_profile * speed_factor, color=COLOR_SCHEMES['default'][0])
    _apply_common_ax_settings(ax2, xlabel=f'Distance ({dist_unit})', ylabel=f'Speed ({speed_unit})', title='Speed Profile', legend=False)

    # Plot curvature vs distance
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(distances * dist_factor, curvature, color=COLOR_SCHEMES['default'][2])
    ax3.axhline(0, color='k', linestyle=':', alpha=0.5)
    _apply_common_ax_settings(ax3, xlabel=f'Distance ({dist_unit})', ylabel='Curvature (1/m)', title='Racing Line Curvature', legend=False)

    # Plot time profile vs distance
    ax4 = fig.add_subplot(gs[1, 1])
    if len(time_profile) == len(distances):
        ax4.plot(distances * dist_factor, time_profile, color=COLOR_SCHEMES['default'][4])
        _apply_common_ax_settings(ax4, xlabel=f'Distance ({distance_unit})', ylabel='Time (s)', title='Time Profile', legend=False)
        if lap_time is None: lap_time = time_profile[-1] if len(time_profile) > 0 else 0
        ax4.text(0.95, 0.95, f"Lap Time: {lap_time:.3f}s", transform=ax4.transAxes, ha='right', va='top', bbox=dict(facecolor='white', alpha=0.8))
    else: ax4.text(0.5, 0.5, "Time data unavailable", ha='center', va='center'); _apply_common_ax_settings(ax4, title='Time Profile', legend=False)

    # Plot Radius vs Distance
    ax5 = fig.add_subplot(gs[2, 0])
    radius = np.full_like(curvature, float('inf')); non_zero = np.abs(curvature) > 1e-6; radius[non_zero] = 1.0 / np.abs(curvature[non_zero])
    radius_plot = np.clip(radius, 1e-1, 1000) # Clip for log plot
    ax5.plot(distances[non_zero] * dist_factor, radius_plot[non_zero], '.', color=COLOR_SCHEMES['default'][5], markersize=DEFAULT_MARKER_SIZE-2)
    ax5.set_yscale('log'); ax5.set_ylim(bottom=1)
    _apply_common_ax_settings(ax5, xlabel=f'Distance ({distance_unit})', ylabel='Corner Radius (m) [log]', title='Corner Radius', legend=False)

    # Plot Acceleration vs Distance
    ax6 = fig.add_subplot(gs[2, 1])
    if len(time_profile) == len(distances) and np.all(np.diff(time_profile) > 1e-9): # Check time is valid for gradient
         long_accel = np.gradient(speed_profile, time_profile, edge_order=2)
    else: # Estimate from speed/distance if time is missing/invalid
        dv = np.gradient(speed_profile); ds = np.gradient(distances); ds[ds < 1e-6] = 1e-6; long_accel = speed_profile * dv / ds
    lateral_accel = speed_profile**2 * np.abs(curvature)
    ax6.plot(distances * dist_factor, long_accel / GRAVITY, color=COLOR_SCHEMES['default'][3], label='Longitudinal G')
    ax6.plot(distances * dist_factor, lateral_accel / GRAVITY, color=COLOR_SCHEMES['default'][6], label='Lateral G')
    ax6.axhline(0, color='k', linestyle=':', alpha=0.5)
    _apply_common_ax_settings(ax6, xlabel=f'Distance ({distance_unit})', ylabel='Acceleration (g)', title='Vehicle Acceleration')

    plot_title = title if title else "Racing Line Analysis"
    fig.suptitle(plot_title, fontsize=DEFAULT_TITLE_SIZE+2)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save_path: save_plot(fig, save_path)
    return fig


#------------------------------------------------------------------------------
# Thermal system plotting functions
#------------------------------------------------------------------------------
def plot_thermal_performance(thermal_data: Dict, title: Optional[str] = None,
                           unit_system: str = 'metric',
                           save_path: Optional[str] = None) -> Optional[plt.Figure]:
    """
    Plot thermal system performance over time or distance.

    Args:
        thermal_data: Dictionary with thermal data. Expected keys:
                      'time' or 'distance' (N array), and at least one of:
                      'engine_temp', 'coolant_temp', 'oil_temp', 'engine_block_temp' (N array).
                      Optional: 'ambient_temp' (scalar or N array), 'thermal_limits' (dict).
        title: Plot title.
        unit_system: Unit system ('metric' or 'imperial').
        save_path: Path to save plot (if None, not saved).

    Returns:
        Matplotlib figure or None if error.
    """
    # Determine x-axis
    time = np.array(thermal_data.get('time', []))
    distance = np.array(thermal_data.get('distance', []))
    if len(time) > 1: x_data, x_label = time, 'Time (s)'
    elif len(distance) > 1: x_data, x_label = distance, 'Distance (m)'
    else: logger.error("Thermal data requires 'time' or 'distance' array."); return None

    # Get temperature data
    temps = {}
    labels = {}
    colors = COLOR_SCHEMES['default']
    temp_keys = {'engine_temp': ('Engine', colors[3]),
                 'coolant_temp': ('Coolant', colors[0]),
                 'oil_temp': ('Oil', colors[2]),
                 'engine_block_temp': ('Eng Block', colors[5])}

    found_data = False
    for key, (label, color) in temp_keys.items():
         temp_values = thermal_data.get(key)
         if temp_values is not None:
              temp_array = np.array(temp_values)
              if len(temp_array) == len(x_data):
                  temps[key] = temp_array
                  labels[key] = label
                  colors[key] = color
                  found_data = True

    if not found_data: logger.error("No valid temperature data found in thermal_data."); return None

    fig, ax1 = plt.subplots(figsize=DEFAULT_FIG_SIZE)

    if unit_system.lower() == 'imperial': temp_convert = lambda t: t*9/5+32 if t is not None else None; temp_unit = "°F"
    else: temp_convert = lambda t: t; temp_unit = "°C"

    # Plot temperatures
    plotted_lines = []
    for key in temps:
        line, = ax1.plot(x_data, temp_convert(temps[key]), color=colors[key], label=labels[key])
        plotted_lines.append(line)

    # Plot ambient temp
    ambient_temp = thermal_data.get('ambient_temp')
    if ambient_temp is not None:
        if isinstance(ambient_temp, (int, float)): amb_line_data = np.full_like(x_data, temp_convert(ambient_temp))
        elif len(ambient_temp) == len(x_data): amb_line_data = temp_convert(np.array(ambient_temp))
        else: amb_line_data = None
        if amb_line_data is not None:
            line, = ax1.plot(x_data, amb_line_data, 'k--', alpha=0.6, linewidth=1, label='Ambient')
            plotted_lines.append(line)

    # Plot thermal limits
    _plot_safety_lines(ax1, thermal_data.get('thermal_limits', {}), temp_convert)

    _apply_common_ax_settings(ax1, xlabel=x_label, ylabel=f'Temperature ({temp_unit})')
    plot_title = title if title else 'Thermal Performance'
    ax1.set_title(plot_title)

    plt.tight_layout()
    if save_path: save_plot(fig, save_path)
    return fig


def plot_thermal_comparison(comparison_data: List[Dict], title: Optional[str] = None,
                          unit_system: str = 'metric',
                          save_path: Optional[str] = None) -> Optional[plt.Figure]:
    """
    Plot comparison of multiple thermal system configurations.

    Args:
        comparison_data: List of dictionaries, each with thermal data and a 'label'.
                         Expected keys: 'time', 'engine_temp', 'coolant_temp'. Optional: 'oil_temp', 'heat_rejection'.
        title: Plot title.
        unit_system: Unit system ('metric' or 'imperial').
        save_path: Path to save plot (if None, not saved).

    Returns:
        Matplotlib figure or None if error.
    """
    if not comparison_data: logger.error("No comparison data provided."); return None

    fig = plt.figure(figsize=(15, 10)); gs = gridspec.GridSpec(2, 2, height_ratios=[2, 1])

    if unit_system.lower() == 'imperial': temp_convert = lambda t: t*9/5+32 if t is not None else None; temp_unit = "°F"
    else: temp_convert = lambda t: t; temp_unit = "°C"

    colors = COLOR_SCHEMES['default']
    labels = [d.get('label', f'Config {i+1}') for i, d in enumerate(comparison_data)]
    x = np.arange(len(labels))

    # Plot Engine Temperatures
    ax1 = fig.add_subplot(gs[0, 0])
    max_temps_engine = []
    for i, data in enumerate(comparison_data):
        time = data.get('time'); temp = data.get('engine_temp')
        if time is not None and temp is not None and len(time) == len(temp):
            ax1.plot(time, temp_convert(np.array(temp)), '-', color=colors[i%len(colors)], label=labels[i])
            max_temps_engine.append(temp_convert(np.max(temp)))
        else: max_temps_engine.append(np.nan)
    _apply_common_ax_settings(ax1, xlabel='Time (s)', ylabel=f'Engine Temp ({temp_unit})', title='Engine Temperature Comparison')

    # Plot Coolant Temperatures
    ax2 = fig.add_subplot(gs[0, 1])
    max_temps_coolant = []
    for i, data in enumerate(comparison_data):
        time = data.get('time'); temp = data.get('coolant_temp')
        if time is not None and temp is not None and len(time) == len(temp):
            ax2.plot(time, temp_convert(np.array(temp)), '-', color=colors[i%len(colors)], label=labels[i])
            max_temps_coolant.append(temp_convert(np.max(temp)))
        else: max_temps_coolant.append(np.nan)
    _apply_common_ax_settings(ax2, xlabel='Time (s)', ylabel=f'Coolant Temp ({temp_unit})', title='Coolant Temperature Comparison')

    # Plot Max Temperatures Bar Chart
    ax3 = fig.add_subplot(gs[1, 0])
    width = 0.35
    valid_eng = ~np.isnan(max_temps_engine); valid_cool = ~np.isnan(max_temps_coolant)
    if np.any(valid_eng): bars1 = ax3.bar(x[valid_eng] - width/2, np.array(max_temps_engine)[valid_eng], width, label='Max Engine', color=colors[3])
    if np.any(valid_cool): bars2 = ax3.bar(x[valid_cool] + width/2, np.array(max_temps_coolant)[valid_cool], width, label='Max Coolant', color=colors[0])
    _apply_common_ax_settings(ax3, xlabel='Configuration', ylabel=f'Max Temp ({temp_unit})', title='Maximum Temperature Comparison')
    ax3.set_xticks(x); ax3.set_xticklabels(labels, rotation=30, ha='right')

    # Plot Heat Rejection Bar Chart
    ax4 = fig.add_subplot(gs[1, 1])
    avg_heat_rejections = []
    has_heat = False
    for data in comparison_data:
        heat_rej = data.get('heat_rejection'); time = data.get('time') # Expect Watts
        if heat_rej is not None and time is not None and len(heat_rej) == len(time) and len(time) > 1:
            avg_heat = np.trapz(heat_rej, time) / (time[-1] - time[0]) if time[-1] > time[0] else np.mean(heat_rej)
            avg_heat_rejections.append(avg_heat / 1000.0) # kW
            has_heat = True
        else: avg_heat_rejections.append(np.nan)
    if has_heat:
        valid_heat = ~np.isnan(avg_heat_rejections)
        if np.any(valid_heat):
            bars = ax4.bar(x[valid_heat], np.array(avg_heat_rejections)[valid_heat], color=colors[5])
            for bar, hr in zip(bars, np.array(avg_heat_rejections)[valid_heat]): ax4.text(bar.get_x()+bar.get_width()/2., hr + 0.1, f'{hr:.1f}', ha='center', va='bottom', fontsize=8)
            _apply_common_ax_settings(ax4, xlabel='Configuration', ylabel='Avg Heat Rejection (kW)', title='Average Heat Rejection Comparison', legend=False)
            ax4.set_xticks(x); ax4.set_xticklabels(labels, rotation=30, ha='right')
    else: ax4.text(0.5, 0.5, "Heat rejection data unavailable", ha='center', va='center'); ax4.set_title('Avg Heat Rejection Comparison')

    plot_title = title if title else 'Thermal System Configuration Comparison'
    fig.suptitle(plot_title, fontsize=DEFAULT_TITLE_SIZE+2)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save_path: save_plot(fig, save_path)
    return fig


def plot_cooling_system_map(cooling_data: Dict, title: Optional[str] = None,
                          unit_system: str = 'metric',
                          save_path: Optional[str] = None) -> Optional[plt.Figure]:
    """
    Plot cooling system performance map (e.g., temperature vs. speed and load/heat).

    Args:
        cooling_data: Dictionary with map data. Expected keys:
                      'speeds' or 'ambient_temps_C' (X-axis, array),
                      'engine_loads' or 'engine_heats_W' (Y-axis, array),
                      'temperature_map' (2D array, Temps in C).
                      Optional: 'coolant_warning_temp', 'coolant_critical_temp'.
        title: Plot title.
        unit_system: Unit system ('metric' or 'imperial').
        save_path: Path to save plot (if None, not saved).

    Returns:
        Matplotlib figure or None if error.
    """
    # Determine X and Y axes data
    if 'speeds' in cooling_data: x_data = np.array(cooling_data['speeds']); x_label_base = 'Vehicle Speed'
    elif 'ambient_temps_C' in cooling_data: x_data = np.array(cooling_data['ambient_temps_C']); x_label_base = 'Ambient Temp'
    else: logger.error("Missing X-axis data ('speeds' or 'ambient_temps_C')."); return None

    if 'engine_loads' in cooling_data: y_data = np.array(cooling_data['engine_loads']) * 100; y_label = 'Engine Load (%)'
    elif 'engine_heats_W' in cooling_data: y_data = np.array(cooling_data['engine_heats_W']) / 1000.0; y_label = 'Engine Heat Input (kW)'
    else: logger.error("Missing Y-axis data ('engine_loads' or 'engine_heats_W')."); return None

    try: temp_map = np.array(cooling_data['temperature_map']) # Expecting Celsius
    except KeyError: logger.error("Missing 'temperature_map' data."); return None

    if temp_map.shape != (len(y_data), len(x_data)): logger.error("Temperature map dimensions do not match axes data."); return None

    fig, ax = plt.subplots(figsize=(12, 8))

    if unit_system.lower() == 'imperial': temp_convert = lambda t: t * 9/5 + 32 if t is not None else None; temp_unit = "°F"
    else: temp_convert = lambda t: t; temp_unit = "°C"
    if x_label_base == 'Vehicle Speed': x_unit = "mph" if unit_system == 'imperial' else "km/h"; x_data *= (MS_TO_MPH if unit_system == 'imperial' else MS_TO_KMH)
    else: x_unit = "°F" if unit_system == 'imperial' else "°C"; x_data = temp_convert(x_data)

    display_temp_map = temp_convert(temp_map)
    X, Y = np.meshgrid(x_data, y_data)

    # Plot contour map
    contour = ax.contourf(X, Y, display_temp_map, 20, cmap=THERMAL_CMAP)
    cbar = plt.colorbar(contour, ax=ax); cbar.set_label(f'Coolant Temperature ({temp_unit})')
    contour_lines = ax.contour(X, Y, display_temp_map, 10, colors='black', alpha=0.6, linewidths=0.5)
    ax.clabel(contour_lines, inline=True, fontsize=8, fmt='%.0f')

    # Plot warning/critical lines
    limits = {'coolant_warning': cooling_data.get('coolant_warning_temp'), 'coolant_critical': cooling_data.get('coolant_critical_temp')}
    _plot_safety_lines(ax, limits, temp_convert)

    _apply_common_ax_settings(ax, xlabel=f'{x_label_base} ({x_unit})', ylabel=y_label, legend=False)
    plot_title = title if title else 'Cooling System Performance Map'
    ax.set_title(plot_title)

    plt.tight_layout()
    if save_path: save_plot(fig, save_path)
    return fig


#------------------------------------------------------------------------------
# Weight sensitivity plotting functions
#------------------------------------------------------------------------------
def plot_weight_sensitivity(sensitivity_data: Dict, title: Optional[str] = None,
                          unit_system: str = 'metric',
                          save_path: Optional[str] = None) -> Optional[plt.Figure]:
    # ... (extract data) ...
    try:
        weights = np.array(sensitivity_data['weights'])
        if len(weights) < 2: raise ValueError("Insufficient weight data points.")
    except (KeyError, ValueError) as e:
        logger.error(f"Invalid weight data for sensitivity plot: {e}"); return None

    lap_times = np.array(sensitivity_data.get('lap_times', []))
    accel_75m_times = np.array(sensitivity_data.get('acceleration_times', [])) # Assume this is 75m
    time_to_60mph = np.array(sensitivity_data.get('zero_to_sixty', [])) # Match key used in plotting example

    if len(lap_times) == 0 and len(accel_75m_times) == 0 and len(time_to_60mph) == 0:
        logger.error("No performance data provided for weight sensitivity plot."); return None

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=False) # Separate Y scales might be needed
    colors = COLOR_SCHEMES['default']
    
    if unit_system.lower() == 'imperial': weight_factor, weight_unit = KG_TO_LBS, "lbs"
    else: weight_factor, weight_unit = 1.0, "kg"
    display_weights = weights * weight_factor

    # Plot Lap Time Sensitivity
    ax1 = axes[0]
    if len(lap_times) == len(weights):
        valid = ~np.isnan(lap_times)
        if np.any(valid):
            ax1.plot(display_weights[valid], lap_times[valid], 'o-', color=colors[0], label='Lap Time')
            if len(weights[valid]) > 1: # Check for enough points for fit
                coeffs = np.polyfit(weights[valid], lap_times[valid], 1); poly = np.poly1d(coeffs)
                line_x = np.linspace(min(weights), max(weights), 100)
                ax1.plot(line_x * weight_factor, poly(line_x), '--', color=colors[0], alpha=0.7)
                ax1.text(0.05, 0.9, f"Sens: {coeffs[0]:.4f} s/kg", transform=ax1.transAxes, color=colors[0])
    _apply_common_ax_settings(ax1, xlabel=f'Vehicle Weight ({weight_unit})', ylabel='Lap Time (s)', title='Lap Time vs. Weight')

    # Plot Acceleration Sensitivity
    ax2 = axes[1]
    lines_accel = []
    labels_accel = []
    if len(accel_75m_times) == len(weights):
        valid = ~np.isnan(accel_75m_times)
        if np.any(valid):
            line, = ax2.plot(display_weights[valid], accel_75m_times[valid], 'o-', color=colors[1], label='75m Time')
            lines_accel.append(line)
            if len(weights[valid]) > 1:
                coeffs = np.polyfit(weights[valid], accel_75m_times[valid], 1); poly = np.poly1d(coeffs)
                line_x = np.linspace(min(weights), max(weights), 100)
                ax2.plot(line_x * weight_factor, poly(line_x), '--', color=colors[1], alpha=0.7)
                ax2.text(0.05, 0.8, f"75m Sens: {coeffs[0]:.4f} s/kg", transform=ax2.transAxes, color=colors[1])

    if len(time_to_60mph) == len(weights):
        valid = ~np.isnan(time_to_60mph)
        if np.any(valid):
            line, = ax2.plot(display_weights[valid], time_to_60mph[valid], 'o-', color=colors[2], label='0-60 mph Time')
            lines_accel.append(line)
            if len(weights[valid]) > 1:
                coeffs = np.polyfit(weights[valid], time_to_60mph[valid], 1); poly = np.poly1d(coeffs)
                line_x = np.linspace(min(weights), max(weights), 100)
                ax2.plot(line_x * weight_factor, poly(line_x), '--', color=colors[2], alpha=0.7)
                ax2.text(0.05, 0.7, f"0-60 Sens: {coeffs[0]:.4f} s/kg", transform=ax2.transAxes, color=colors[2])

    if not lines_accel: ax2.text(0.5, 0.5, "No acceleration data", ha='center', va='center')
    _apply_common_ax_settings(ax2, xlabel=f'Vehicle Weight ({weight_unit})', ylabel='Time (s)', title='Acceleration vs. Weight')

    plot_title = title if title else 'Weight Sensitivity Analysis'
    fig.suptitle(plot_title, fontsize=DEFAULT_TITLE_SIZE+2)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save_path: save_plot(fig, save_path)
    return fig

def plot_weight_distribution_sensitivity(sensitivity_data: Dict, title: Optional[str] = None,
                                       save_path: Optional[str] = None) -> Optional[plt.Figure]:
    """
    Plot weight distribution sensitivity analysis results.

    Args:
        sensitivity_data: Dict with 'front_weight_pct' (list/array, 0-1) and performance metric lists
                          (e.g., 'lap_times', 'lateral_acceleration' in g).
        title: Plot title.
        save_path: Path to save plot (if None, not saved).

    Returns:
        Matplotlib figure or None if error.
    """
    try:
        front_pct = np.array(sensitivity_data['front_weight_pct']) * 100 # Convert to %
        if len(front_pct) < 2: raise ValueError("Insufficient distribution data points.")
    except (KeyError, ValueError) as e:
        logger.error(f"Invalid weight distribution data for sensitivity plot: {e}"); return None

    lap_times = np.array(sensitivity_data.get('lap_times', []))
    accel_times = np.array(sensitivity_data.get('acceleration_times', [])) # Likely 75m time
    lat_accel = np.array(sensitivity_data.get('lateral_acceleration', [])) # Expecting g's

    if len(lap_times) == 0 and len(accel_times) == 0 and len(lat_accel) == 0:
        logger.error("No performance data for weight distribution sensitivity plot."); return None

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True)
    colors = COLOR_SCHEMES['default']

    # Plot Lap Time
    ax1 = axes[0]
    if len(lap_times) == len(front_pct):
        valid = ~np.isnan(lap_times)
        if np.any(valid):
            ax1.plot(front_pct[valid], lap_times[valid], 'o-', color=colors[0])
            if len(front_pct[valid]) >= 3: # Need 3+ for quadratic fit
                coeffs = np.polyfit(front_pct[valid]/100.0, lap_times[valid], 2); poly = np.poly1d(coeffs)
                x_line = np.linspace(min(front_pct), max(front_pct), 100)
                ax1.plot(x_line, poly(x_line/100.0), '--', color=colors[0], alpha=0.7)
                if abs(coeffs[0]) > 1e-6 and coeffs[0] > 0: # Min exists (a>0)
                    opt_pct = -coeffs[1] / (2 * coeffs[0]) * 100
                    if min(front_pct) <= opt_pct <= max(front_pct): ax1.scatter([opt_pct], [poly(opt_pct/100.0)], c='r', s=80, zorder=5, label=f'Opt: {opt_pct:.1f}%')
    _apply_common_ax_settings(ax1, xlabel='Front Weight Distribution (%)', ylabel='Lap Time (s)', title='Lap Time vs. Weight Dist.')

    # Plot Acceleration Time
    ax2 = axes[1]
    if len(accel_times) == len(front_pct):
        valid = ~np.isnan(accel_times)
        if np.any(valid):
            ax2.plot(front_pct[valid], accel_times[valid], 'o-', color=colors[1])
            if len(front_pct[valid]) >= 3:
                coeffs = np.polyfit(front_pct[valid]/100.0, accel_times[valid], 2); poly = np.poly1d(coeffs)
                x_line = np.linspace(min(front_pct), max(front_pct), 100)
                ax2.plot(x_line, poly(x_line/100.0), '--', color=colors[1], alpha=0.7)
                if abs(coeffs[0]) > 1e-6 and coeffs[0] > 0: # Min exists (a>0)
                    opt_pct = -coeffs[1] / (2 * coeffs[0]) * 100
                    if min(front_pct) <= opt_pct <= max(front_pct): ax2.scatter([opt_pct], [poly(opt_pct/100.0)], c='r', s=80, zorder=5, label=f'Opt: {opt_pct:.1f}%')
    _apply_common_ax_settings(ax2, xlabel='Front Weight Distribution (%)', ylabel='75m Time (s)', title='Accel vs. Weight Dist.')

    # Plot Lateral Accel
    ax3 = axes[2]
    if len(lat_accel) == len(front_pct):
        valid = ~np.isnan(lat_accel)
        if np.any(valid):
            ax3.plot(front_pct[valid], lat_accel[valid], 'o-', color=colors[2])
            if len(front_pct[valid]) >= 3:
                coeffs = np.polyfit(front_pct[valid]/100.0, lat_accel[valid], 2); poly = np.poly1d(coeffs)
                x_line = np.linspace(min(front_pct), max(front_pct), 100)
                ax3.plot(x_line, poly(x_line/100.0), '--', color=colors[2], alpha=0.7)
                if abs(coeffs[0]) > 1e-6 and coeffs[0] < 0: # Max exists (a<0)
                    opt_pct = -coeffs[1] / (2 * coeffs[0]) * 100
                    if min(front_pct) <= opt_pct <= max(front_pct): ax3.scatter([opt_pct], [poly(opt_pct/100.0)], c='r', s=80, zorder=5, label=f'Opt: {opt_pct:.1f}%')
    _apply_common_ax_settings(ax3, xlabel='Front Weight Distribution (%)', ylabel='Max Lateral Accel (g)', title='Cornering vs. Weight Dist.')

    plot_title = title if title else 'Weight Distribution Sensitivity Analysis'
    fig.suptitle(plot_title, fontsize=DEFAULT_TITLE_SIZE+2)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save_path: save_plot(fig, save_path)
    return fig


#------------------------------------------------------------------------------
# Endurance plotting functions
#------------------------------------------------------------------------------
def plot_endurance_results(endurance_data: Dict, title: Optional[str] = None,
                         unit_system: str = 'metric',
                         save_path: Optional[str] = None,
                         plot_type: str = 'summary') -> Optional[plt.Figure]:
    """
    Plot endurance event simulation results. Can plot specific aspects.

    Args:
        endurance_data: Dict containing endurance results. Expected keys depend on plot_type.
                        Should include 'lap_times'. Optionally 'fuel_consumption_laps', 'thermal_states',
                        'component_wear', 'reliability_events' (as list of strings), 'score',
                        'total_time_s', 'dnf_reason', 'fuel_capacity_L'.
        title: Plot title.
        unit_system: Unit system ('metric' or 'imperial').
        save_path: Path to save plot (if None, not saved).
        plot_type: Type of plot ('summary', 'lap_times', 'thermal', 'fuel', 'wear').

    Returns:
        Matplotlib figure or None if error.
    """
    lap_times = np.array(endurance_data.get('lap_times', []))
    lap_numbers = np.arange(1, len(lap_times) + 1) # lap_numbers is empty if lap_times is empty
    has_laps = len(lap_numbers) > 0

    if not has_laps and plot_type != 'wear':
        logger.warning("No lap times found in endurance data for plotting.")
        # Allow wear plot even without laps

    # Unit conversions
    if unit_system.lower() == 'imperial':
        fuel_factor, fuel_unit = LITERS_TO_GAL, "gal"
        temp_convert = lambda t: t * 9/5 + 32 if t is not None else None
        temp_unit = "°F"
    else:
        fuel_factor, fuel_unit = 1.0, "L"
        temp_convert = lambda t: t
        temp_unit = "°C"

    colors = COLOR_SCHEMES['default']

    # Determine plot layout
    if plot_type == 'summary':
        fig = plt.figure(figsize=(15, 14)); gs = gridspec.GridSpec(3, 2, height_ratios=[2, 1, 1])
        ax_lap = fig.add_subplot(gs[0, :]); ax_fuel = fig.add_subplot(gs[1, 0])
        ax_thermal = fig.add_subplot(gs[1, 1]); ax_wear = fig.add_subplot(gs[2, 0])
        ax_score = fig.add_subplot(gs[2, 1])
    elif plot_type == 'lap_times': fig, ax_lap = plt.subplots(figsize=DEFAULT_FIG_SIZE); ax_fuel, ax_thermal, ax_wear, ax_score = None, None, None, None
    elif plot_type == 'fuel': fig, ax_fuel = plt.subplots(figsize=DEFAULT_FIG_SIZE); ax_lap, ax_thermal, ax_wear, ax_score = None, None, None, None
    elif plot_type == 'thermal': fig, ax_thermal = plt.subplots(figsize=DEFAULT_FIG_SIZE); ax_lap, ax_fuel, ax_wear, ax_score = None, None, None, None
    elif plot_type == 'wear': fig, ax_wear = plt.subplots(figsize=(8, 6)); ax_lap, ax_fuel, ax_thermal, ax_score = None, None, None, None
    else: logger.error(f"Unknown plot_type for plot_endurance_results: {plot_type}"); return None

    # Plot Lap Times
    if ax_lap is not None:
        if has_laps:
            ax_lap.plot(lap_numbers, lap_times, 'b-o', label='Lap Time')
            avg_lap_time = np.mean(lap_times); best_lap_time = np.min(lap_times)
            ax_lap.axhline(y=avg_lap_time, color='r', linestyle='--', alpha=0.7, label=f'Avg: {avg_lap_time:.2f}s')
            ax_lap.axhline(y=best_lap_time, color='g', linestyle='--', alpha=0.7, label=f'Best: {best_lap_time:.2f}s')
            reliability_events = endurance_data.get('reliability_events', [])
            if len(reliability_events) == len(lap_numbers):
                for i, event_name in enumerate(reliability_events):
                    if event_name and isinstance(event_name, str) and event_name.upper() != 'NONE':
                        try:
                            ax_lap.scatter([lap_numbers[i]], [lap_times[i]], color='red', marker='x', s=100, zorder=5)
                            ax_lap.annotate(event_name.replace('_', ' ').title(), xy=(lap_numbers[i], lap_times[i]), xytext=(0, 5), textcoords='offset points', ha='center', rotation=30, size=8, color='red')
                        except IndexError: logger.warning(f"Index error plotting reliability event at lap {i+1}.")
            _apply_common_ax_settings(ax_lap, xlabel='Lap Number', ylabel='Lap Time (s)', title='Lap Times')
            ax_lap.xaxis.set_major_locator(MaxNLocator(integer=True))
        else: ax_lap.text(0.5, 0.5, "Lap time data unavailable", ha='center', va='center'); _apply_common_ax_settings(ax_lap, title='Lap Times', legend=False)

    # Plot Fuel Consumption
    if ax_fuel is not None:
        fuel_per_lap = np.array(endurance_data.get('fuel_consumption_laps', []))
        fuel_capacity = endurance_data.get('fuel_capacity_L')
        if has_laps and len(fuel_per_lap) == len(lap_numbers):
            cumulative_fuel = np.cumsum(fuel_per_lap) * fuel_factor
            ln_fuel = ax_fuel.plot(lap_numbers, cumulative_fuel, 'g-o', label=f'Cumulative ({fuel_unit})')
            _apply_common_ax_settings(ax_fuel, xlabel='Lap Number', ylabel=f'Cumulative Fuel ({fuel_unit})', title='Fuel Consumption', legend=False)
            if fuel_capacity: ax_fuel.axhline(fuel_capacity * fuel_factor, color='k', linestyle=':', alpha=0.5, label=f'Capacity ({fuel_capacity * fuel_factor:.1f} {fuel_unit})')
            ax_fuel_b = ax_fuel.twinx()
            bars_fuel = ax_fuel_b.bar(lap_numbers, fuel_per_lap * fuel_factor, alpha=0.3, color='green', label=f'Per Lap ({fuel_unit})')
            _apply_common_ax_settings(ax_fuel_b, ylabel=f'Per Lap ({fuel_unit})', legend=False)
            ax_fuel_b.tick_params(axis='y', colors='green')
            ax_fuel.legend(loc='upper left'); ax_fuel_b.legend(loc='upper right')
            ax_fuel.xaxis.set_major_locator(MaxNLocator(integer=True))
        else: ax_fuel.text(0.5, 0.5, "Fuel data unavailable", ha='center', va='center'); _apply_common_ax_settings(ax_fuel, title='Fuel Consumption', legend=False)

    # Plot Thermal Profile
    if ax_thermal is not None:
        thermal_states = endurance_data.get('thermal_states', [])
        if has_laps and len(thermal_states) == len(lap_numbers):
            temps = {'engine': [], 'coolant': [], 'oil': []}
            for state in thermal_states:
                temps['engine'].append(temp_convert(state.get('engine_temp')))
                temps['coolant'].append(temp_convert(state.get('coolant_temp')))
                temps['oil'].append(temp_convert(state.get('oil_temp')))
            lines = []
            if any(t is not None for t in temps['engine']): lines.append(ax_thermal.plot(lap_numbers, temps['engine'], 'r-o', label='Engine')[0])
            if any(t is not None for t in temps['coolant']): lines.append(ax_thermal.plot(lap_numbers, temps['coolant'], 'b-o', label='Coolant')[0])
            if any(t is not None for t in temps['oil']): lines.append(ax_thermal.plot(lap_numbers, temps['oil'], 'y-o', label='Oil')[0])
            _plot_safety_lines(ax_thermal, endurance_data.get('thermal_limits', {}), temp_convert)
            _apply_common_ax_settings(ax_thermal, xlabel='Lap Number', ylabel=f'End-of-Lap Temp ({temp_unit})', title='Thermal Profile')
            ax_thermal.xaxis.set_major_locator(MaxNLocator(integer=True))
        else: ax_thermal.text(0.5, 0.5, "Thermal data unavailable", ha='center', va='center'); _apply_common_ax_settings(ax_thermal, title='Thermal Profile', legend=False)

    # Plot Component Wear
    if ax_wear is not None:
        wear_data = endurance_data.get('component_wear', endurance_data.get('final_component_wear'))
        final_wear = {}
        if has_laps and isinstance(wear_data, list) and len(wear_data) == len(lap_numbers): final_wear = wear_data[-1] if wear_data else {}
        elif isinstance(wear_data, dict): final_wear = wear_data
        if final_wear:
            components = list(final_wear.keys()); wear_values = [w * 100 for w in final_wear.values()]
            colors_wear = [plt.cm.OrRd(min(1.0, w / 100.0)) for w in wear_values]
            bars = ax_wear.barh(components, wear_values, color=colors_wear)
            for bar, wear in zip(bars, wear_values): ax_wear.text(wear + 1, bar.get_y() + bar.get_height()/2., f'{wear:.1f}%', va='center', fontsize=8)
            _apply_common_ax_settings(ax_wear, xlabel='Final Wear (%)', ylabel='Component', title='Component Wear', legend=False)
            ax_wear.set_xlim(0, 105)
        else: ax_wear.text(0.5, 0.5, "Wear data unavailable", ha='center', va='center'); _apply_common_ax_settings(ax_wear, title='Component Wear', legend=False)

    # Plot Score Summary (only if summary plot)
    if ax_score is not None and plot_type == 'summary':
        scores = endurance_data.get('score', {})
        if scores:
            labels_score = ['Endurance', 'Efficiency']; values_score = [scores.get('endurance_score', 0), scores.get('efficiency_score', 0)]
            max_values_score = [scores.get('max_endurance_score', 275), scores.get('max_efficiency_score', 100)]
            percentages = [v / max_v * 100 if max_v > 0 else 0 for v, max_v in zip(values_score, max_values_score)]
            bars = ax_score.bar(labels_score, percentages, color=[colors[0], colors[2]])
            for bar, score, max_score in zip(bars, values_score, max_values_score): ax_score.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1, f'{score:.1f}/{max_score}', ha='center', va='bottom', fontsize=8)
            _apply_common_ax_settings(ax_score, ylabel='Score (% of Max)', title='Event Scores', legend=False)
            ax_score.set_ylim(0, 110); ax_score.axhline(100, color='k', linestyle='--', alpha=0.3)
        else: ax_score.text(0.5, 0.5, "Score data unavailable", ha='center', va='center'); ax_score.set_title('Event Scores')

    # Status Text (only for summary plot)
    if plot_type == 'summary':
        status = endurance_data.get('status', 'Finished' if endurance_data.get('completed') else 'DNF')
        reason = endurance_data.get('dnf_reason', '')
        time_val = endurance_data.get('total_time_s'); fuel_val = endurance_data.get('total_fuel_L')
        score_val = endurance_data.get('score', {}).get('total_score')
        status_text = f"Status: {status}" + (f" ({reason})" if reason else "")
        if time_val is not None: status_text += f" | Time: {time_val:.1f}s"
        if fuel_val is not None: status_text += f" | Fuel: {fuel_val * fuel_factor:.2f}{fuel_unit}"
        if score_val is not None: status_text += f" | Score: {score_val:.1f}"
        plt.figtext(0.5, 0.01, status_text, ha='center', fontsize=DEFAULT_LABEL_SIZE, bbox=dict(facecolor='white', alpha=0.8, edgecolor='lightgray'))

    plot_title = title if title else f'Endurance Results ({plot_type.replace("_"," ").title()})'
    fig.suptitle(plot_title, fontsize=DEFAULT_TITLE_SIZE+2)
    plt.tight_layout(rect=[0, 0.05 if plot_type=='summary' else 0.03, 1, 0.95])

    if save_path: save_plot(fig, save_path)
    return fig


def plot_endurance_comparison(comparison_data: List[Dict], title: Optional[str] = None,
                            unit_system: str = 'metric',
                            save_path: Optional[str] = None) -> Optional[plt.Figure]:
    """
    Plot comparison of multiple endurance configurations.

    Args:
        comparison_data: List of dictionaries, each with 'results', 'score', and 'label'.
                         'results' dict should contain 'total_time_s', 'total_fuel_L', 'lap_times',
                         'reliability_events', 'final_component_wear'.
                         'score' dict should contain 'total_score', 'endurance_score', 'efficiency_score'.
        title: Plot title.
        unit_system: Unit system ('metric' or 'imperial').
        save_path: Path to save plot (if None, not saved).

    Returns:
        Matplotlib figure or None if error.
    """
    if not comparison_data: logger.error("No data for endurance comparison plot."); return None

    fig = plt.figure(figsize=(16, 12)); gs = gridspec.GridSpec(3, 2)

    if unit_system.lower() == 'imperial': fuel_factor, fuel_unit = LITERS_TO_GAL, "gal"
    else: fuel_factor, fuel_unit = 1.0, "L"

    colors = COLOR_SCHEMES['default']
    labels = [d.get('label', f'Config {i+1}') for i, d in enumerate(comparison_data)]
    num_configs = len(labels)
    x = np.arange(num_configs)

    # Plot Total Score
    ax1 = fig.add_subplot(gs[0, 0])
    scores = [d.get('score', {}).get('total_score', 0) for d in comparison_data]
    max_score_ref = comparison_data[0].get('score', {}).get('max_endurance_score', 275) + comparison_data[0].get('score', {}).get('max_efficiency_score', 100)
    bars = ax1.bar(labels, scores, color=[colors[i % len(colors)] for i in range(num_configs)])
    ax1.axhline(max_score_ref, color='k', linestyle='--', alpha=0.5, label=f'Max Possible ({max_score_ref})')
    for bar, s in zip(bars, scores): ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 2, f'{s:.1f}', ha='center', va='bottom', fontsize=8)
    _apply_common_ax_settings(ax1, ylabel='Total Score', title='Total Endurance Score Comparison')
    ax1.tick_params(axis='x', rotation=30, ha='right')

    # Plot Endurance vs Efficiency Score
    ax2 = fig.add_subplot(gs[0, 1])
    end_scores = [d.get('score', {}).get('endurance_score', 0) for d in comparison_data]
    eff_scores = [d.get('score', {}).get('efficiency_score', 0) for d in comparison_data]
    width = 0.35
    bars1 = ax2.bar(x - width/2, end_scores, width, label='Endurance', color=colors[0])
    bars2 = ax2.bar(x + width/2, eff_scores, width, label='Efficiency', color=colors[2])
    _apply_common_ax_settings(ax2, ylabel='Score Points', title='Endurance vs. Efficiency Score')
    ax2.set_xticks(x); ax2.set_xticklabels(labels, rotation=30, ha='right')

    # Plot Total Time
    ax3 = fig.add_subplot(gs[1, 0])
    times = [d.get('results', {}).get('total_time_s') for d in comparison_data]
    plot_times = [t if t is not None else 0 for t in times]; plot_labels = [f'{t:.1f}s' if t is not None else 'DNF' for t in times]
    bars = ax3.bar(labels, plot_times, color='purple')
    for bar, lbl in zip(bars, plot_labels): ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 5, lbl, ha='center', va='bottom', fontsize=8)
    _apply_common_ax_settings(ax3, ylabel='Total Time (s)', title='Total Event Time (Lower is Better)', legend=False)
    ax3.tick_params(axis='x', rotation=30, ha='right')

    # Plot Total Fuel
    ax4 = fig.add_subplot(gs[1, 1])
    fuels = [d.get('results', {}).get('total_fuel_L', 0) * fuel_factor for d in comparison_data]
    bars = ax4.bar(labels, fuels, color='green')
    for bar, f in zip(bars, fuels): ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05 * max(1.0, max(fuels)), f'{f:.2f}', ha='center', va='bottom', fontsize=8)
    _apply_common_ax_settings(ax4, ylabel=f'Total Fuel ({fuel_unit})', title='Total Fuel Consumption', legend=False)
    ax4.tick_params(axis='x', rotation=30, ha='right')

    # Plot Reliability Issues Count
    ax5 = fig.add_subplot(gs[2, 0])
    issues = [len([e for e in d.get('results', {}).get('reliability_events', []) if str(e).upper() != 'NONE']) for d in comparison_data]
    bars = ax5.bar(labels, issues, color='red')
    for bar, count in zip(bars, issues): ax5.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1, f'{count}', ha='center', va='bottom', fontsize=9)
    _apply_common_ax_settings(ax5, ylabel='Number of Issues', title='Reliability Issues Count', legend=False)
    ax5.tick_params(axis='x', rotation=30, ha='right')
    ax5.yaxis.set_major_locator(MaxNLocator(integer=True))

    # Plot Max Component Wear
    ax6 = fig.add_subplot(gs[2, 1])
    max_wear = []
    for d in comparison_data: wear = d.get('results', {}).get('final_component_wear', {}); max_wear.append(max(wear.values())*100 if wear else 0)
    bars = ax6.bar(labels, max_wear, color='orange')
    for bar, w in zip(bars, max_wear): ax6.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1, f'{w:.1f}%', ha='center', va='bottom', fontsize=8)
    _apply_common_ax_settings(ax6, ylabel='Max Wear (%)', title='Maximum Component Wear', legend=False)
    ax6.tick_params(axis='x', rotation=30, ha='right')
    ax6.set_ylim(0, max(10, max(max_wear)*1.1 if max_wear else 10))

    plot_title = title if title else 'Endurance Configuration Comparison'
    fig.suptitle(plot_title, fontsize=DEFAULT_TITLE_SIZE+2)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save_path: save_plot(fig, save_path)
    return fig

#------------------------------------------------------------------------------
# Acceleration plotting functions
#------------------------------------------------------------------------------
def plot_acceleration_results(results: Dict, title: Optional[str] = None,
                            unit_system: str = 'metric', plot_wheel_slip: bool = False,
                            save_path: Optional[str] = None) -> Optional[plt.Figure]:
    """Plot results from a single acceleration simulation."""
    try:
        time = np.array(results['time']); speed = np.array(results['speed'])
        position = np.array(results['position']); acceleration = np.array(results['acceleration'])
        rpm = np.array(results['engine_rpm']); gear = np.array(results['gear'])
        slip = np.array(results.get('wheel_slip', [])); throttle = np.array(results.get('throttle_effective', []))
        if not all(len(arr) == len(time) for arr in [speed, position, acceleration, rpm, gear]) or len(time) < 2:
             raise ValueError("Input arrays missing or length mismatch.")
        if plot_wheel_slip and len(slip) != len(time): plot_wheel_slip = False; logger.warning("Wheel slip data missing or mismatched length, not plotting.")
        if len(throttle) != len(time): throttle = None # Throttle plot is optional
    except (KeyError, ValueError) as e:
        logger.error(f"Acceleration results dictionary missing required keys or invalid data: {e}")
        return None

    num_axes = 4 if plot_wheel_slip and throttle is not None else 3
    fig, axes = plt.subplots(num_axes, 1, figsize=(12, 10 if num_axes==4 else 8), sharex=True)
    colors = COLOR_SCHEMES['default']
    
    if unit_system.lower() == 'imperial': speed_factor, speed_unit = MS_TO_MPH, "mph"; pos_factor, pos_unit = M_TO_INCH / 12.0, "ft"; accel_factor, accel_unit = 1 / GRAVITY, "g"
    else: speed_factor, speed_unit = MS_TO_KMH, "km/h"; pos_factor, pos_unit = 1.0, "m"; accel_factor, accel_unit = 1 / GRAVITY, "g"

    # Plot Speed and Position
    ax1 = axes[0]
    ln1 = ax1.plot(time, speed * speed_factor, color=colors[0], label=f'Speed ({speed_unit})')
    _apply_common_ax_settings(ax1, ylabel=f'Speed ({speed_unit})', legend=False)
    ax1b = ax1.twinx(); ln2 = ax1b.plot(time, position * pos_factor, color=colors[1], label=f'Position ({pos_unit})')
    ax1b.set_ylabel(f'Position ({pos_unit})', color=colors[1]); ax1b.tick_params(axis='y', labelcolor=colors[1])
    ax1.legend(ln1 + ln2, [l.get_label() for l in ln1+ln2], loc='center left')

    # Plot Acceleration and Gear
    ax2 = axes[1]
    ln3 = ax2.plot(time, acceleration * accel_factor, color=colors[2], label=f'Acceleration ({accel_unit})')
    _apply_common_ax_settings(ax2, ylabel=f'Acceleration ({accel_unit})', legend=False)
    ax2b = ax2.twinx(); ln4 = ax2b.step(time, gear, where='post', color=colors[3], label='Gear')
    ax2b.set_ylabel('Gear', color=colors[3]); ax2b.tick_params(axis='y', labelcolor=colors[3])
    ax2b.yaxis.set_major_locator(MaxNLocator(integer=True)); ax2b.set_ylim(0.5, max(1, np.max(gear)*1.1) if np.any(gear) else 1.5)
    ax2.legend(ln3 + [ln4], [l.get_label() for l in ln3+[ln4]], loc='center left')

    # Plot RPM
    ax3 = axes[2]
    ax3.plot(time, rpm, color=colors[4], label='Engine RPM')
    _apply_common_ax_settings(ax3, xlabel='Time (s)' if num_axes==3 else '', ylabel='Engine RPM')

    # Plot Wheel Slip and Throttle (Optional)
    if num_axes == 4:
         ax4 = axes[3]
         ln6 = ax4.plot(time, slip * 100, color=colors[5], label='Wheel Slip (%)', linestyle='--')
         _apply_common_ax_settings(ax4, xlabel='Time (s)', ylabel='Wheel Slip (%)', legend=False)
         ax4.tick_params(axis='y', labelcolor=colors[5])
         ax4.set_ylim(bottom=0)
         ax4b = ax4.twinx()
         ln7 = ax4b.plot(time, throttle, color=colors[6], label='Effective Throttle')
         ax4b.set_ylabel('Effective Throttle (0-1)', color=colors[6]); ax4b.tick_params(axis='y', labelcolor=colors[6])
         ax4b.set_ylim(-0.05, 1.05)
         ax4.legend(ln6 + ln7, [l.get_label() for l in ln6+ln7], loc='center left')


    # Add finish time annotation
    finish_time = results.get('finish_time')
    if finish_time is not None:
        # Find the primary shared x-axis (usually the bottom-most one in this setup)
        primary_ax = axes[-1]
        primary_ax.axvline(finish_time, color='k', linestyle='--', alpha=0.8, label=f'Finish: {finish_time:.3f}s')
        # Update the legend on the axis where the line was added
        handles, labels_leg = primary_ax.get_legend_handles_labels()
        primary_ax.legend(handles=handles, labels=labels_leg, loc='best') 

    plot_title = title if title else f"Acceleration Run ({results.get('distance_m', 75):.0f}m)"
    fig.suptitle(plot_title, fontsize=DEFAULT_TITLE_SIZE + 2)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save_path: save_plot(fig, save_path)
    return fig


def plot_acceleration_comparison(comparison_data: List[Dict], title: Optional[str] = None,
                               unit_system: str = 'metric',
                               save_path: Optional[str] = None) -> Optional[plt.Figure]:
    """Plot comparison of multiple acceleration runs."""
    if not comparison_data: return None
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    colors = COLOR_SCHEMES['default']

    if unit_system.lower() == 'imperial': speed_factor, speed_unit = MS_TO_MPH, "mph"
    else: speed_factor, speed_unit = MS_TO_KMH, "km/h"

    max_time = 0
    for i, data in enumerate(comparison_data):
        label = data.get('label', f'Run {i+1}')
        time = data.get('time'); speed = data.get('speed'); acceleration = data.get('acceleration')
        if time is None or speed is None or acceleration is None: continue
        color = colors[i % len(colors)]
        max_time = max(max_time, time[-1])
        axes[0].plot(time, speed * speed_factor, color=color, label=label)
        axes[1].plot(time, acceleration / GRAVITY, color=color, label=label)

    if max_time == 0: logger.error("No valid data to plot for acceleration comparison."); return None

    _apply_common_ax_settings(axes[0], ylabel=f'Speed ({speed_unit})', title='Speed Comparison')
    _apply_common_ax_settings(axes[1], xlabel='Time (s)', ylabel='Acceleration (g)', title='Acceleration Comparison')
    axes[0].set_xlim(0, max_time)

    plot_title = title if title else 'Acceleration Run Comparison'
    fig.suptitle(plot_title, fontsize=DEFAULT_TITLE_SIZE+2)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_path: save_plot(fig, save_path)
    return fig


#------------------------------------------------------------------------------
# Lap Time plotting functions
#------------------------------------------------------------------------------
def plot_lap_time_results(results: Dict, title: Optional[str] = None,
                         unit_system: str = 'metric',
                         save_path: Optional[str] = None) -> Optional[plt.Figure]:
    # ... (try block to extract data) ...
    try:
        time = np.array(results['time']); distance = np.array(results['distance'])
        speed = np.array(results['speed']); acceleration = np.array(results['acceleration'])
        lateral_g = np.array(results.get('lateral_g', np.zeros_like(time))) # Default to zero if missing
        rpm = np.array(results['engine_rpm']); gear = np.array(results['gear'])
        if not all(len(arr) == len(time) for arr in [distance, speed, acceleration, lateral_g, rpm, gear]) or len(time) < 2:
            raise ValueError("Input arrays have mismatched lengths or are too short.")
    except (KeyError, ValueError) as e:
        logger.error(f"Invalid lap time results dictionary for plotting: {e}"); return None

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    colors = COLOR_SCHEMES['default']
    
    if unit_system.lower() == 'imperial': speed_factor, speed_unit = MS_TO_MPH, "mph"; dist_factor, dist_unit = M_TO_KM * 0.621371, "miles"
    else: speed_factor, speed_unit = MS_TO_KMH, "km/h"; dist_factor, dist_unit = M_TO_KM, "km"

    # Plot Speed and Gear
    ax1 = axes[0]; ln1 = ax1.plot(distance * dist_factor, speed * speed_factor, color=colors[0], label=f'Speed ({speed_unit})')
    _apply_common_ax_settings(ax1, ylabel=f'Speed ({speed_unit})', title='Lap Performance vs Distance', legend=False)
    ax1b = ax1.twinx(); ln2 = ax1b.step(distance * dist_factor, gear, where='post', color=colors[1], label='Gear')
    ax1b.set_ylabel('Gear', color=colors[1]); ax1b.tick_params(axis='y', labelcolor=colors[1])
    ax1b.yaxis.set_major_locator(MaxNLocator(integer=True)); ax1b.set_ylim(0.5, max(1, np.max(gear)*1.1) if np.any(gear) else 1.5)
    ax1.legend(ln1 + [ln2], [l.get_label() for l in ln1 + [ln2]], loc='upper left')

    # Plot Acceleration
    ax2 = axes[1]; ln3 = ax2.plot(distance * dist_factor, acceleration / GRAVITY, color=colors[2], label='Longitudinal Accel (g)')
    ln4 = ax2.plot(distance * dist_factor, lateral_g, color=colors[3], label='Lateral Accel (g)')
    _apply_common_ax_settings(ax2, ylabel='Acceleration (g)')

    # Plot Engine RPM
    ax3 = axes[2]; ax3.plot(distance * dist_factor, rpm, color=colors[4], label='Engine RPM')
    _apply_common_ax_settings(ax3, xlabel=f'Distance ({dist_unit})', ylabel='Engine RPM')

    lap_time = results.get('lap_time')
    if lap_time is not None: fig.text(0.5, 0.96, f"Lap Time: {lap_time:.3f}s", ha='center', fontsize=DEFAULT_TITLE_SIZE, weight='bold')

    plot_title = title if title else 'Lap Simulation Results'
    fig.suptitle(plot_title, fontsize=DEFAULT_TITLE_SIZE+2)
    plt.tight_layout(rect=[0, 0.03, 1, 0.94])
    if save_path: save_plot(fig, save_path)
    return fig



def plot_lap_time_comparison(comparison_data: List[Dict], labels: List[str],
                            title: Optional[str] = None, unit_system: str = 'metric',
                            save_path: Optional[str] = None) -> Optional[plt.Figure]:
    # ... (check data) ...
    if not comparison_data: logger.error("No data for lap time comparison plot."); return None
    if len(comparison_data) != len(labels) and not all('label' in d for d in comparison_data):
        logger.warning("Mismatch between number of data entries and labels, using default labels.")

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    colors = COLOR_SCHEMES['default']

    if unit_system.lower() == 'imperial': speed_factor, speed_unit = MS_TO_MPH, "mph"; dist_unit = "m"
    else: speed_factor, speed_unit = MS_TO_KMH, "km/h"; dist_unit = "m"

    max_dist = 0
    plot_labels = []
    for i, data in enumerate(comparison_data):
        try:
            distance = np.array(data['distance']); speed = np.array(data['speed'])
            lateral_g = np.array(data.get('lateral_g', np.zeros_like(distance))) # Default to 0 if missing
            if not all(len(arr) == len(distance) for arr in [speed, lateral_g]) or len(distance) < 2: raise ValueError("Data mismatch")
            label = data.get('label', labels[i] if i < len(labels) else f'Run {i+1}')
            plot_labels.append(label) # Store the label used
            color = colors[i % len(colors)]
            max_dist = max(max_dist, distance[-1])
            axes[0].plot(distance, speed * speed_factor, color=color, label=label)
            axes[1].plot(distance, lateral_g, color=color, label=label)
        except (KeyError, ValueError) as e:
            logger.warning(f"Skipping run '{labels[i] if i < len(labels) else i+1}' due to invalid data: {e}")

    if max_dist == 0: logger.error("No valid data plotted for lap time comparison."); return None

    _apply_common_ax_settings(axes[0], ylabel=f'Speed ({speed_unit})', title='Speed Profile Comparison')
    _apply_common_ax_settings(axes[1], xlabel=f'Distance ({dist_unit})', ylabel='Lateral Acceleration (g)', title='Lateral G Comparison')
    axes[0].set_xlim(0, max_dist)

    plot_title = title if title else 'Lap Time Comparison'
    fig.suptitle(plot_title, fontsize=DEFAULT_TITLE_SIZE+2)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save_path: save_plot(fig, save_path)
    return fig
