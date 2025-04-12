"""
Plotting utilities for Formula Student powertrain simulation.

This module provides a comprehensive set of plotting functions for visualizing
simulation results, vehicle performance metrics, and component behavior.
It aims to create consistent and informative plots for analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LinearSegmentedColormap, Normalize
from mpl_toolkits.mplot3d import Axes3D # Keep for potential 3D plots
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import os
import logging

# Import constants for unit conversions
from .constants import (
    MS_TO_KMH, MS_TO_MPH, KG_TO_LBS, KW_TO_HP, LITERS_TO_GAL,
    NM_TO_LBFT, M_TO_INCH, M_TO_MM
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("Plotting")

# Default style settings for plots
DEFAULT_FIG_SIZE = (12, 8)
DEFAULT_DPI = 150 # Adjusted for potentially large plots
DEFAULT_LINE_WIDTH = 2.0
DEFAULT_MARKER_SIZE = 5
DEFAULT_FONT_SIZE = 10
DEFAULT_TITLE_SIZE = 14
DEFAULT_LABEL_SIZE = 12
DEFAULT_LEGEND_SIZE = 10
DEFAULT_GRID_ALPHA = 0.4
DEFAULT_SAVE_FORMAT = 'png'

# Color schemes
COLOR_SCHEMES = {
    'default': plt.cm.tab10.colors, # Use matplotlib's default color cycle
    'formula_student': ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf'],
    'thermal': plt.cm.coolwarm, # Red-Blue diverging colormap
    'speed': plt.cm.viridis, # Perceptually uniform colormap
    'acceleration': plt.cm.plasma # Another perceptually uniform colormap
}

#------------------------------------------------------------------------------
# Utility functions
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
        'seaborn': 'seaborn-v0_8-darkgrid' # Added seaborn option
    }

    if style in styles:
        plt.style.use(styles[style])
    else:
        logger.warning(f"Unknown style: {style}. Using default.")
        plt.style.use(styles['default'])

    # Set common parameters more robustly
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

    # Use plot_format consistently
    format_lower = plot_format.lower()

    # Process filename
    if '.' in filename:
        base, ext = os.path.splitext(filename)
        if ext[1:].lower() != format_lower:
            logger.warning(f"Filename extension ({ext}) doesn't match format ({format_lower}). Using {format_lower}.")
            filename = base

    filepath = f"{filename}.{format_lower}"

    # Ensure directory exists
    if directory:
        try:
            os.makedirs(directory, exist_ok=True)
            filepath = os.path.join(directory, filepath)
        except OSError as e:
            logger.error(f"Could not create directory {directory}: {e}")
            return None

    # Save the figure
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
        'time_to_60mph': '0-60 mph Time',
        'time_to_100kph': '0-100 km/h Time',
        'finish_time': '75m Time',
        'peak_acceleration_g': 'Peak Accel (g)',
        'avg_speed_kph': 'Avg Speed (km/h)',
        'max_speed_kph': 'Max Speed (km/h)',
        'max_lateral_g': 'Max Lateral (g)',
        'lap_time': 'Lap Time',
        'engine_temp': 'Engine Temp',
        'coolant_temp': 'Coolant Temp',
        'oil_temp': 'Oil Temp'
    }
    return name_map.get(metric, metric.replace('_', ' ').title())


def _apply_common_ax_settings(ax: plt.Axes, xlabel: str = "", ylabel: str = "", title: str = ""):
    """Apply common settings to a matplotlib Axes object."""
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=DEFAULT_GRID_ALPHA)
    ax.tick_params(axis='both', which='major', labelsize=DEFAULT_FONT_SIZE)


#------------------------------------------------------------------------------
# Engine plotting functions
#------------------------------------------------------------------------------

def plot_engine_performance(engine_data: Dict, title: Optional[str] = None,
                          unit_system: str = 'metric', show_efficiency: bool = False,
                          save_path: Optional[str] = None) -> Optional[plt.Figure]:
    """
    Plot engine performance curves (torque, power, efficiency).

    Args:
        engine_data: Dictionary with 'rpm', 'torque', 'power' arrays. Optional 'efficiency'.
        title: Plot title.
        unit_system: Unit system ('metric' or 'imperial').
        show_efficiency: Whether to show efficiency curves.
        save_path: Path to save plot (if None, not saved).

    Returns:
        Matplotlib figure or None if error.
    """
    rpm = np.array(engine_data.get('rpm', []))
    torque = np.array(engine_data.get('torque', []))
    power = np.array(engine_data.get('power', [])) # Assume power is in kW
    efficiency = np.array(engine_data.get('efficiency', [])) # Efficiency should be 0-1

    if len(rpm) == 0 or len(rpm) != len(torque) or len(rpm) != len(power):
        logger.error("Invalid engine data format or missing required keys ('rpm', 'torque', 'power').")
        return None

    fig, ax1 = plt.subplots(figsize=DEFAULT_FIG_SIZE)

    # Unit conversions
    if unit_system.lower() == 'imperial':
        torque_factor, torque_unit = NM_TO_LBFT, "lb-ft"
        power_factor, power_unit = KW_TO_HP, "HP"
    else:
        torque_factor, torque_unit = 1.0, "Nm"
        power_factor, power_unit = 1.0, "kW"

    # Plot torque curve
    torque_line, = ax1.plot(rpm, torque * torque_factor, color=COLOR_SCHEMES['default'][0],
                         linewidth=DEFAULT_LINE_WIDTH, label=f"Torque ({torque_unit})")
    _apply_common_ax_settings(ax1, xlabel='Engine Speed (RPM)', ylabel=f'Torque ({torque_unit})')
    ax1.tick_params(axis='y', labelcolor=COLOR_SCHEMES['default'][0])

    # Twin axis for power
    ax2 = ax1.twinx()
    power_line, = ax2.plot(rpm, power * power_factor, color=COLOR_SCHEMES['default'][1],
                        linewidth=DEFAULT_LINE_WIDTH, label=f"Power ({power_unit})")
    ax2.set_ylabel(f'Power ({power_unit})', color=COLOR_SCHEMES['default'][1])
    ax2.tick_params(axis='y', labelcolor=COLOR_SCHEMES['default'][1])

    # Plot efficiency if requested
    lines = [torque_line, power_line]
    labels = [torque_line.get_label(), power_line.get_label()]

    if show_efficiency and len(efficiency) == len(rpm):
        ax3 = ax1.twinx()
        ax3.spines['right'].set_position(('outward', 60)) # Offset axis
        eff_line, = ax3.plot(rpm, efficiency * 100, color=COLOR_SCHEMES['default'][2],
                           linewidth=DEFAULT_LINE_WIDTH, label="Efficiency (%)", linestyle='--')
        ax3.set_ylabel('Efficiency (%)', color=COLOR_SCHEMES['default'][2])
        ax3.tick_params(axis='y', labelcolor=COLOR_SCHEMES['default'][2])
        ax3.set_ylim(0, 100)
        lines.append(eff_line)
        labels.append(eff_line.get_label())

    ax1.legend(lines, labels, loc='best')

    # Set title
    plot_title = title if title else 'Engine Performance Curves'
    ax1.set_title(plot_title) # Set title on the primary axis

    # Highlight peaks if available in data
    if 'max_torque_rpm' in engine_data and 'max_torque' in engine_data:
        rpm_tq, val_tq = engine_data['max_torque_rpm'], engine_data['max_torque'] * torque_factor
        ax1.plot(rpm_tq, val_tq, 'o', color=COLOR_SCHEMES['default'][0], markersize=DEFAULT_MARKER_SIZE+2)
        ax1.text(rpm_tq, val_tq*1.02, f"{val_tq:.1f} {torque_unit}\n@{rpm_tq:.0f} RPM", ha='center', va='bottom', fontsize=DEFAULT_FONT_SIZE-1)

    if 'max_power_rpm' in engine_data and 'max_power' in engine_data:
        rpm_pw, val_pw = engine_data['max_power_rpm'], engine_data['max_power'] * power_factor
        ax2.plot(rpm_pw, val_pw, 'o', color=COLOR_SCHEMES['default'][1], markersize=DEFAULT_MARKER_SIZE+2)
        ax2.text(rpm_pw, val_pw*0.98, f"{val_pw:.1f} {power_unit}\n@{rpm_pw:.0f} RPM", ha='center', va='top', fontsize=DEFAULT_FONT_SIZE-1)

    # Set x-axis limits
    if 'idle_rpm' in engine_data and 'redline_rpm' in engine_data:
        ax1.set_xlim(engine_data['idle_rpm'], engine_data['redline_rpm'])

    plt.tight_layout()

    # Save if requested
    if save_path:
        save_plot(fig, save_path)

    return fig


def plot_torque_curves_comparison(curves_data: List[Dict], title: Optional[str] = None,
                                unit_system: str = 'metric',
                                save_path: Optional[str] = None) -> Optional[plt.Figure]:
    """
    Plot comparison of multiple torque curves.

    Args:
        curves_data: List of dictionaries, each with 'rpm', 'torque', 'power', 'label'.
        title: Plot title.
        unit_system: Unit system ('metric' or 'imperial').
        save_path: Path to save plot (if None, not saved).

    Returns:
        Matplotlib figure or None if error.
    """
    if not curves_data:
        logger.error("No curve data provided for comparison.")
        return None

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Unit conversions
    if unit_system.lower() == 'imperial':
        torque_factor, torque_unit = NM_TO_LBFT, "lb-ft"
        power_factor, power_unit = KW_TO_HP, "HP"
    else:
        torque_factor, torque_unit = 1.0, "Nm"
        power_factor, power_unit = 1.0, "kW"

    # Colors for different curves
    colors = COLOR_SCHEMES['default']

    # Plot each curve
    min_rpm, max_rpm = float('inf'), float('-inf')
    for i, curve in enumerate(curves_data):
        rpm = np.array(curve.get('rpm', []))
        torque = np.array(curve.get('torque', []))
        power = np.array(curve.get('power', []))
        label = curve.get('label', f'Curve {i+1}')
        color = curve.get('color', colors[i % len(colors)])

        if len(rpm) == 0 or len(rpm) != len(torque) or len(rpm) != len(power):
            logger.warning(f"Skipping curve '{label}' due to data length mismatch or missing data.")
            continue

        min_rpm = min(min_rpm, rpm[0])
        max_rpm = max(max_rpm, rpm[-1])

        # Plot torque
        ax1.plot(rpm, torque * torque_factor, '-', color=color,
               linewidth=DEFAULT_LINE_WIDTH, label=label)

        # Plot power
        ax2.plot(rpm, power * power_factor, '-', color=color,
               linewidth=DEFAULT_LINE_WIDTH, label=label)

    # Set labels and grid
    _apply_common_ax_settings(ax1, ylabel=f'Torque ({torque_unit})', title='Torque Comparison')
    ax1.legend(loc='best')

    _apply_common_ax_settings(ax2, xlabel='Engine Speed (RPM)', ylabel=f'Power ({power_unit})', title='Power Comparison')
    ax2.legend(loc='best')

    # Set common x-axis limits if data was plotted
    if min_rpm < max_rpm:
        ax1.set_xlim(min_rpm, max_rpm)

    # Set overall title
    plot_title = title if title else 'Torque Curve Comparison'
    fig.suptitle(plot_title, fontsize=DEFAULT_TITLE_SIZE+2)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust for suptitle

    # Save if requested
    if save_path:
        save_plot(fig, save_path)

    return fig

#------------------------------------------------------------------------------
# Track plotting functions
#------------------------------------------------------------------------------
# plot_track_layout and plot_racing_line_analysis are implemented here as requested
# (These were previously defined in the thought block but should be implemented in plotting.py)

def plot_track_layout(track_data: Dict, show_racing_line: bool = True,
                    show_segments: bool = True, show_elevation: bool = False,
                    title: Optional[str] = None, save_path: Optional[str] = None) -> Optional[plt.Figure]:
    """
    Plot track layout with optional racing line and segments.

    Args:
        track_data: Dictionary with track data (needs 'points' key, optionally 'width', 'racing_line', 'segments', 'elevation', 'start_position', 'start_direction', 'name', 'length').
        show_racing_line: Whether to show racing line if available.
        show_segments: Whether to show track segments if available.
        show_elevation: Whether to add an elevation profile subplot.
        title: Plot title.
        save_path: Path to save plot (if None, not saved).

    Returns:
        Matplotlib figure or None if error.
    """
    # Extract track data
    track_points = np.array(track_data.get('points', []))
    track_width = track_data.get('width', [])
    racing_line = np.array(track_data.get('racing_line', [])) if 'racing_line' in track_data else None
    segments = track_data.get('segments', [])
    elevation = np.array(track_data.get('elevation', [])) if 'elevation' in track_data else None
    distances = np.array(track_data.get('distance', [])) if 'distance' in track_data else None

    # Validate data
    if len(track_points) < 2:
        logger.error("Invalid track data format: requires at least 2 points.")
        return None

    # Create figure
    if show_elevation and elevation is not None and len(elevation) == len(track_points):
        fig = plt.figure(figsize=(15, 12))
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
    else:
        fig = plt.figure(figsize=(12, 10))
        ax1 = plt.gca()
        ax2 = None # No elevation plot

    # Plot track centerline
    ax1.plot(track_points[:, 0], track_points[:, 1], 'k-', alpha=0.7, linewidth=1.5, label='Track Centerline')

    # Plot track boundaries if width is available
    if len(track_width) == len(track_points) and np.all(track_width > 0):
        # Calculate normal vectors
        normals = np.zeros_like(track_points)
        tangents = np.gradient(track_points, axis=0)
        norms = np.linalg.norm(tangents, axis=1)
        valid_norms = norms > 1e-6
        normals[valid_norms, 0] = -tangents[valid_norms, 1] / norms[valid_norms]
        normals[valid_norms, 1] = tangents[valid_norms, 0] / norms[valid_norms]
        # Handle start/end for closed loop
        if np.linalg.norm(track_points[0] - track_points[-1]) < 1e-3:
             normals[0] = normals[-1] = (normals[0] + normals[-1]) / 2 # Average normals at seam

        # Calculate left and right boundaries
        half_width = np.array(track_width) / 2.0
        left_boundary = track_points + normals * half_width[:, np.newaxis]
        right_boundary = track_points - normals * half_width[:, np.newaxis]

        # Plot boundaries
        ax1.plot(left_boundary[:, 0], left_boundary[:, 1], 'k-', alpha=0.3, linewidth=1)
        ax1.plot(right_boundary[:, 0], right_boundary[:, 1], 'k-', alpha=0.3, linewidth=1)

    # Plot racing line if requested
    if show_racing_line and racing_line is not None and len(racing_line) > 1:
        ax1.plot(racing_line[:, 0], racing_line[:, 1], 'r-',
               linewidth=DEFAULT_LINE_WIDTH, label='Racing Line')

    # Plot segments if requested
    if show_segments and segments:
        segment_colors = {
            'STRAIGHT': 'green', # Match TrackSegmentType if used
            'CORNER_LEFT': 'blue',
            'CORNER_RIGHT': 'red',
            'CHICANE': 'purple',
            'HAIRPIN': 'orange'
        }
        segment_patches = {}
        for segment in segments:
            segment_type = segment.get('type', 'STRAIGHT') # Use 'type' or 'segment_type'
            color = segment_colors.get(segment_type.upper(), 'gray')
            label = segment_type.replace('_', ' ').title()

            start_idx = segment.get('start_idx', 0)
            end_idx = segment.get('end_idx', len(track_points) - 1) # Use dict.get with default

            # Ensure indices are valid and in order
            if 0 <= start_idx <= end_idx < len(track_points):
                segment_points = track_points[start_idx : end_idx + 1]
                line = ax1.plot(segment_points[:, 0], segment_points[:, 1], '-',
                              color=color, linewidth=4, alpha=0.6, label=label if label not in segment_patches else "")
                if line and label not in segment_patches:
                    segment_patches[label] = line[0] # Store first line of this type for legend

    # Add start/finish marker
    if 'start_position' in track_data:
        start_pos = track_data['start_position']
        ax1.scatter(start_pos[0], start_pos[1], color='lime', marker='o', s=100, label='Start/Finish', zorder=5)

        if 'start_direction' in track_data:
            start_dir = track_data['start_direction']
            dir_vec = np.array([np.cos(start_dir), np.sin(start_dir)])
            ax1.arrow(start_pos[0], start_pos[1], dir_vec[0] * 5, dir_vec[1] * 5,
                    head_width=1.5, head_length=2.0, fc='lime', ec='lime', zorder=5)

    # Set equal aspect ratio and grid for track plot
    ax1.set_aspect('equal', adjustable='box')
    ax1.grid(True, alpha=DEFAULT_GRID_ALPHA)

    # Set title and labels for track plot
    track_name = track_data.get('name', 'Track')
    track_length = track_data.get('length', np.sum(np.linalg.norm(np.diff(track_points, axis=0), axis=1)) if len(track_points) > 1 else 0)
    ax1_title = f"{track_name} (Length: {track_length:.1f}m)"
    _apply_common_ax_settings(ax1, xlabel='X (m)', ylabel='Y (m)', title=ax1_title)
    ax1.legend(loc='best')

    # Plot elevation profile if requested
    if ax2 is not None:
        # Calculate distance along track if not provided
        if distances is None or len(distances) != len(track_points):
            distances = np.zeros(len(track_points))
            for i in range(1, len(track_points)):
                distances[i] = distances[i-1] + np.linalg.norm(track_points[i] - track_points[i-1])

        ax2.plot(distances, elevation, 'g-', linewidth=DEFAULT_LINE_WIDTH)

        # Calculate elevation gain/loss
        elevation_diff = np.diff(elevation)
        elevation_gain = np.sum(elevation_diff[elevation_diff > 0])
        elevation_loss = np.sum(elevation_diff[elevation_diff < 0])
        info_text = f"Elevation Gain: {elevation_gain:.1f}m | Loss: {abs(elevation_loss):.1f}m"
        _apply_common_ax_settings(ax2, xlabel='Distance (m)', ylabel='Elevation (m)', title=f'Elevation Profile ({info_text})')


    # Set overall title
    plot_title = title if title else "Track Layout Analysis"
    fig.suptitle(plot_title, fontsize=DEFAULT_TITLE_SIZE+2)

    plt.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust for suptitle

    # Save if requested
    if save_path:
        save_plot(fig, save_path)

    return fig

def plot_racing_line_analysis(racing_line_data: Dict, title: Optional[str] = None,
                            unit_system: str = 'metric',
                            save_path: Optional[str] = None) -> Optional[plt.Figure]:
    """
    Plot racing line analysis including curvature and speed profiles.

    Args:
        racing_line_data: Dictionary with racing line data (needs 'line', 'distances', 'curvature', 'speed_profile', optionally 'track_points', 'time_profile').
        title: Plot title.
        unit_system: Unit system ('metric' or 'imperial').
        save_path: Path to save plot (if None, not saved).

    Returns:
        Matplotlib figure or None if error.
    """
    # Extract data
    racing_line = np.array(racing_line_data.get('line', []))
    distances = np.array(racing_line_data.get('distances', []))
    curvature = np.array(racing_line_data.get('curvature', []))
    speed_profile = np.array(racing_line_data.get('speed_profile', []))
    time_profile = np.array(racing_line_data.get('time_profile', []))
    track_points = np.array(racing_line_data.get('track_points', [])) # Optional track centerline

    # Validate data
    if len(racing_line) < 2 or len(racing_line) != len(distances) or len(racing_line) != len(curvature) or len(racing_line) != len(speed_profile):
        logger.error("Invalid racing line data format or missing required keys.")
        return None

    # Create figure
    fig = plt.figure(figsize=(15, 12))
    gs = gridspec.GridSpec(3, 2, height_ratios=[2, 1, 1])

    # Unit conversions
    if unit_system.lower() == 'imperial':
        speed_factor, speed_unit = MS_TO_MPH, "mph"
        distance_factor, distance_unit = 1 / 1609.34, "miles" # Meters to miles
    else:
        speed_factor, speed_unit = MS_TO_KMH, "km/h"
        distance_factor, distance_unit = 0.001, "km" # Meters to km

    # Plot track layout with racing line colored by speed
    ax1 = fig.add_subplot(gs[0, 0])
    if len(track_points) > 1:
        ax1.plot(track_points[:, 0], track_points[:, 1], 'k--', alpha=0.4, linewidth=1, label='Track Centerline')

    points = racing_line.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = Normalize(vmin=np.min(speed_profile * speed_factor), vmax=np.max(speed_profile * speed_factor))
    lc = plt.matplotlib.collections.LineCollection(segments, cmap=SPEED_CMAP, norm=norm)
    lc.set_array(speed_profile * speed_factor)
    lc.set_linewidth(DEFAULT_LINE_WIDTH)
    line = ax1.add_collection(lc)
    cbar = plt.colorbar(line, ax=ax1)
    cbar.set_label(f'Speed ({speed_unit})')

    ax1.set_aspect('equal', adjustable='box')
    _apply_common_ax_settings(ax1, xlabel='X (m)', ylabel='Y (m)', title='Racing Line colored by Speed')
    if len(track_points) > 1: ax1.legend(loc='best')

    # Plot speed profile vs distance
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(distances * distance_factor, speed_profile * speed_factor, color=COLOR_SCHEMES['default'][0], linewidth=DEFAULT_LINE_WIDTH)
    _apply_common_ax_settings(ax2, xlabel=f'Distance ({distance_unit})', ylabel=f'Speed ({speed_unit})', title='Speed Profile')

    # Plot curvature vs distance
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(distances * distance_factor, curvature, color=COLOR_SCHEMES['default'][2], linewidth=DEFAULT_LINE_WIDTH)
    ax3.axhline(0, color='k', linestyle=':', alpha=0.5) # Zero curvature line
    _apply_common_ax_settings(ax3, xlabel=f'Distance ({distance_unit})', ylabel='Curvature (1/m)', title='Racing Line Curvature')

    # Plot time profile vs distance
    ax4 = fig.add_subplot(gs[1, 1])
    if len(time_profile) == len(distances):
        ax4.plot(distances * distance_factor, time_profile, color=COLOR_SCHEMES['default'][4], linewidth=DEFAULT_LINE_WIDTH)
        _apply_common_ax_settings(ax4, xlabel=f'Distance ({distance_unit})', ylabel='Time (s)', title='Time Profile')
        lap_time = time_profile[-1] if len(time_profile) > 0 else 0
        ax4.text(0.5, 0.9, f"Lap Time: {lap_time:.3f}s", transform=ax4.transAxes, ha='center',
                   bbox=dict(facecolor='white', alpha=0.8, edgecolor='lightgray'))

    # Plot Radius vs Distance
    ax5 = fig.add_subplot(gs[2, 0])
    # Calculate radius, handle near-zero curvature
    radius = np.full_like(curvature, float('inf'))
    non_zero_mask = np.abs(curvature) > 1e-6
    radius[non_zero_mask] = 1.0 / np.abs(curvature[non_zero_mask])
    radius = np.clip(radius, 0, 1000) # Clip large radii for plotting
    ax5.plot(distances[non_zero_mask] * distance_factor, radius[non_zero_mask], '.', color=COLOR_SCHEMES['default'][5], markersize=DEFAULT_MARKER_SIZE-2)
    ax5.set_yscale('log')
    _apply_common_ax_settings(ax5, xlabel=f'Distance ({distance_unit})', ylabel='Corner Radius (m) [log]', title='Corner Radius')

    # Plot Acceleration vs Distance
    ax6 = fig.add_subplot(gs[2, 1])
    # Calculate longitudinal acceleration: a = v * dv/ds
    dv = np.gradient(speed_profile)
    ds = np.gradient(distances)
    ds[ds < 1e-6] = 1e-6 # Avoid division by zero
    long_accel = speed_profile * dv / ds
    lateral_accel = speed_profile**2 * np.abs(curvature)

    ax6.plot(distances * distance_factor, long_accel / GRAVITY, color=COLOR_SCHEMES['default'][3], linewidth=DEFAULT_LINE_WIDTH, label='Longitudinal G')
    ax6.plot(distances * distance_factor, lateral_accel / GRAVITY, color=COLOR_SCHEMES['default'][6], linewidth=DEFAULT_LINE_WIDTH, label='Lateral G')
    ax6.axhline(0, color='k', linestyle=':', alpha=0.5)
    _apply_common_ax_settings(ax6, xlabel=f'Distance ({distance_unit})', ylabel='Acceleration (g)', title='Vehicle Acceleration')
    ax6.legend(loc='best')

    # Overall Title
    plot_title = title if title else "Racing Line Analysis"
    fig.suptitle(plot_title, fontsize=DEFAULT_TITLE_SIZE+2)

    plt.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust for suptitle

    # Save if requested
    if save_path:
        save_plot(fig, save_path)

    return fig


#------------------------------------------------------------------------------
# Thermal system plotting functions
#------------------------------------------------------------------------------
# plot_thermal_performance, plot_thermal_comparison, plot_cooling_system_map
# implemented here as requested.

def plot_thermal_performance(thermal_data: Dict, title: Optional[str] = None,
                           unit_system: str = 'metric',
                           save_path: Optional[str] = None) -> Optional[plt.Figure]:
    """
    Plot thermal system performance over time or distance.

    Args:
        thermal_data: Dictionary with thermal data (needs 'time' or 'distance', and temps like 'engine_temp', 'coolant_temp', 'oil_temp').
        title: Plot title.
        unit_system: Unit system ('metric' or 'imperial').
        save_path: Path to save plot (if None, not saved).

    Returns:
        Matplotlib figure or None if error.
    """
    time = np.array(thermal_data.get('time', []))
    distance = np.array(thermal_data.get('distance', []))
    engine_temp = np.array(thermal_data.get('engine_temp', []))
    coolant_temp = np.array(thermal_data.get('coolant_temp', []))
    oil_temp = np.array(thermal_data.get('oil_temp', []))

    # Determine x-axis (time or distance)
    if len(time) == len(engine_temp) and len(time) > 1:
        x_data = time
        x_label = 'Time (s)'
    elif len(distance) == len(engine_temp) and len(distance) > 1:
        x_data = distance
        x_label = 'Distance (m)'
    else:
        logger.error("Thermal data requires 'time' or 'distance' array matching temperature arrays.")
        return None

    # Validate temperature data exists
    if len(engine_temp) == 0 and len(coolant_temp) == 0 and len(oil_temp) == 0:
        logger.error("No temperature data found in thermal_data.")
        return None

    fig, ax1 = plt.subplots(figsize=DEFAULT_FIG_SIZE)

    # Unit conversions
    if unit_system.lower() == 'imperial':
        temp_convert = lambda t: t * 9/5 + 32 if t is not None else None
        temp_unit = "°F"
    else:
        temp_convert = lambda t: t # No conversion needed
        temp_unit = "°C"

    # Plot temperatures
    lines = []
    labels = []
    if len(engine_temp) == len(x_data):
        line, = ax1.plot(x_data, temp_convert(engine_temp), color=COLOR_SCHEMES['default'][3], linewidth=DEFAULT_LINE_WIDTH, label='Engine')
        lines.append(line)
        labels.append('Engine')
    if len(coolant_temp) == len(x_data):
        line, = ax1.plot(x_data, temp_convert(coolant_temp), color=COLOR_SCHEMES['default'][0], linewidth=DEFAULT_LINE_WIDTH, label='Coolant')
        lines.append(line)
        labels.append('Coolant')
    if len(oil_temp) == len(x_data):
        line, = ax1.plot(x_data, temp_convert(oil_temp), color=COLOR_SCHEMES['default'][2], linewidth=DEFAULT_LINE_WIDTH, label='Oil')
        lines.append(line)
        labels.append('Oil')

    # Add ambient temperature if available
    ambient_temp = thermal_data.get('ambient_temp')
    if ambient_temp is not None:
        if isinstance(ambient_temp, (int, float)):
            ambient_temp_line = np.full_like(x_data, temp_convert(ambient_temp))
        elif len(ambient_temp) == len(x_data):
             ambient_temp_line = temp_convert(np.array(ambient_temp))
        else:
            ambient_temp_line = None

        if ambient_temp_line is not None:
            line, = ax1.plot(x_data, ambient_temp_line, 'k--', alpha=0.6, linewidth=1, label='Ambient')
            lines.append(line)
            labels.append('Ambient')

    # Add thermal limits if available
    thermal_limits = thermal_data.get('thermal_limits', {})
    engine_warning = thermal_limits.get('engine_warning')
    engine_critical = thermal_limits.get('engine_critical')
    coolant_warning = thermal_limits.get('coolant_warning')
    coolant_critical = thermal_limits.get('coolant_critical')

    if engine_warning: ax1.axhline(temp_convert(engine_warning), color='orange', linestyle='--', label='Eng Warn')
    if engine_critical: ax1.axhline(temp_convert(engine_critical), color='red', linestyle='-', label='Eng Crit')
    if coolant_warning: ax1.axhline(temp_convert(coolant_warning), color='cyan', linestyle='--', label='Cool Warn')
    if coolant_critical: ax1.axhline(temp_convert(coolant_critical), color='blue', linestyle='-', label='Cool Crit')

    _apply_common_ax_settings(ax1, xlabel=x_label, ylabel=f'Temperature ({temp_unit})')
    ax1.legend(loc='best')

    plot_title = title if title else 'Thermal Performance'
    ax1.set_title(plot_title)

    plt.tight_layout()

    if save_path:
        save_plot(fig, save_path)

    return fig


def plot_thermal_comparison(comparison_data: List[Dict], title: Optional[str] = None,
                          unit_system: str = 'metric',
                          save_path: Optional[str] = None) -> Optional[plt.Figure]:
    """
    Plot comparison of multiple thermal system configurations.

    Args:
        comparison_data: List of dictionaries, each with thermal data and a 'label'.
        title: Plot title.
        unit_system: Unit system ('metric' or 'imperial').
        save_path: Path to save plot (if None, not saved).

    Returns:
        Matplotlib figure or None if error.
    """
    if not comparison_data:
        logger.error("No comparison data provided for thermal comparison plot.")
        return None

    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 2, height_ratios=[2, 1]) # Adjusted layout

    # Temperature conversion
    if unit_system.lower() == 'imperial':
        temp_convert = lambda t: t * 9/5 + 32 if t is not None else None
        temp_unit = "°F"
    else:
        temp_convert = lambda t: t
        temp_unit = "°C"

    colors = COLOR_SCHEMES['default']

    # Plot Engine Temperatures Comparison
    ax1 = fig.add_subplot(gs[0, 0])
    max_temps_engine = []
    config_labels = []

    for i, data in enumerate(comparison_data):
        label = data.get('label', f'Config {i+1}')
        config_labels.append(label)
        time = data.get('time')
        engine_temp = data.get('engine_temp')
        color = data.get('color', colors[i % len(colors)])

        if time is not None and engine_temp is not None and len(time) == len(engine_temp):
            ax1.plot(time, temp_convert(np.array(engine_temp)), '-', color=color, linewidth=DEFAULT_LINE_WIDTH, label=label)
            max_temps_engine.append(temp_convert(np.max(engine_temp)))
        else:
            max_temps_engine.append(np.nan) # Use NaN for missing data

    _apply_common_ax_settings(ax1, xlabel='Time (s)', ylabel=f'Engine Temperature ({temp_unit})', title='Engine Temperature Comparison')
    ax1.legend(loc='best')

    # Plot Coolant Temperatures Comparison
    ax2 = fig.add_subplot(gs[0, 1])
    max_temps_coolant = []

    for i, data in enumerate(comparison_data):
        time = data.get('time')
        coolant_temp = data.get('coolant_temp')
        color = data.get('color', colors[i % len(colors)])

        if time is not None and coolant_temp is not None and len(time) == len(coolant_temp):
            ax2.plot(time, temp_convert(np.array(coolant_temp)), '-', color=color, linewidth=DEFAULT_LINE_WIDTH, label=data.get('label', f'Config {i+1}'))
            max_temps_coolant.append(temp_convert(np.max(coolant_temp)))
        else:
            max_temps_coolant.append(np.nan)

    _apply_common_ax_settings(ax2, xlabel='Time (s)', ylabel=f'Coolant Temperature ({temp_unit})', title='Coolant Temperature Comparison')
    ax2.legend(loc='best')

    # Plot Maximum Temperature Bar Chart
    ax3 = fig.add_subplot(gs[1, 0])
    x = np.arange(len(config_labels))
    width = 0.35

    # Filter out NaN values before plotting bars
    valid_engine_idx = ~np.isnan(max_temps_engine)
    valid_coolant_idx = ~np.isnan(max_temps_coolant)

    if np.any(valid_engine_idx):
        bars1 = ax3.bar(x[valid_engine_idx] - width/2, np.array(max_temps_engine)[valid_engine_idx], width, label='Max Engine Temp', color=COLOR_SCHEMES['default'][3])
        for bar, temp in zip(bars1, np.array(max_temps_engine)[valid_engine_idx]):
            ax3.text(bar.get_x() + bar.get_width()/2., temp + 1, f'{temp:.1f}', ha='center', va='bottom', fontsize=8)

    if np.any(valid_coolant_idx):
        bars2 = ax3.bar(x[valid_coolant_idx] + width/2, np.array(max_temps_coolant)[valid_coolant_idx], width, label='Max Coolant Temp', color=COLOR_SCHEMES['default'][0])
        for bar, temp in zip(bars2, np.array(max_temps_coolant)[valid_coolant_idx]):
            ax3.text(bar.get_x() + bar.get_width()/2., temp + 1, f'{temp:.1f}', ha='center', va='bottom', fontsize=8)

    _apply_common_ax_settings(ax3, xlabel='Configuration', ylabel=f'Temperature ({temp_unit})', title='Maximum Temperature Comparison')
    ax3.set_xticks(x)
    ax3.set_xticklabels(config_labels, rotation=45, ha='right')
    ax3.legend(loc='best')

    # Plot Heat Rejection Comparison (if available)
    ax4 = fig.add_subplot(gs[1, 1])
    avg_heat_rejections = []
    heat_available = False

    for data in comparison_data:
        heat_rej = data.get('heat_rejection')
        if heat_rej is not None and len(heat_rej) > 0:
            avg_heat_rejections.append(np.mean(heat_rej) / 1000) # Convert to kW
            heat_available = True
        else:
            avg_heat_rejections.append(np.nan)

    if heat_available:
        valid_heat_idx = ~np.isnan(avg_heat_rejections)
        if np.any(valid_heat_idx):
            bars = ax4.bar(x[valid_heat_idx], np.array(avg_heat_rejections)[valid_heat_idx], color=COLOR_SCHEMES['default'][5])
            for bar, hr in zip(bars, np.array(avg_heat_rejections)[valid_heat_idx]):
                ax4.text(bar.get_x() + bar.get_width()/2., hr + 0.1, f'{hr:.1f}', ha='center', va='bottom', fontsize=8)

            _apply_common_ax_settings(ax4, xlabel='Configuration', ylabel='Avg Heat Rejection (kW)', title='Average Heat Rejection Comparison')
            ax4.set_xticks(x)
            ax4.set_xticklabels(config_labels, rotation=45, ha='right')
    else:
        ax4.text(0.5, 0.5, "Heat rejection data not available\nfor all configurations.", ha='center', va='center')
        ax4.set_xticks([])
        ax4.set_yticks([])
        ax4.set_title('Average Heat Rejection Comparison')


    plot_title = title if title else 'Thermal System Configuration Comparison'
    fig.suptitle(plot_title, fontsize=DEFAULT_TITLE_SIZE+2)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust for suptitle

    if save_path:
        save_plot(fig, save_path)

    return fig

def plot_cooling_system_map(cooling_data: Dict, title: Optional[str] = None,
                          unit_system: str = 'metric',
                          save_path: Optional[str] = None) -> Optional[plt.Figure]:
    """
    Plot cooling system performance map (e.g., temperature vs. speed and load).

    Args:
        cooling_data: Dictionary with map data (needs 'speeds', 'loads', 'temperature_map').
        title: Plot title.
        unit_system: Unit system ('metric' or 'imperial').
        save_path: Path to save plot (if None, not saved).

    Returns:
        Matplotlib figure or None if error.
    """
    speeds = np.array(cooling_data.get('speeds', [])) # Expecting m/s
    loads = np.array(cooling_data.get('engine_loads', [])) # Expecting 0-1
    temperature_map = np.array(cooling_data.get('temperature_map', [])) # Expecting Celsius

    if len(speeds) == 0 or len(loads) == 0 or temperature_map.shape != (len(loads), len(speeds)):
        logger.error("Invalid cooling map data format. Requires 'speeds', 'engine_loads', 'temperature_map' with matching dimensions.")
        return None

    fig, ax = plt.subplots(figsize=(12, 8))

    # Unit conversions
    if unit_system.lower() == 'imperial':
        temp_convert = lambda t: t * 9/5 + 32 if t is not None else None
        temp_unit = "°F"
        speed_factor = MS_TO_MPH
        speed_unit = "mph"
    else:
        temp_convert = lambda t: t
        temp_unit = "°C"
        speed_factor = MS_TO_KMH
        speed_unit = "km/h"

    # Convert data for display
    display_speeds = speeds * speed_factor
    display_loads = loads * 100 # Convert load to percentage
    display_temp_map = temp_convert(temperature_map)

    # Create meshgrid
    X, Y = np.meshgrid(display_speeds, display_loads)

    # Create contour plot
    contour = ax.contourf(X, Y, display_temp_map, 20, cmap=THERMAL_CMAP)
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label(f'Coolant Temperature ({temp_unit})')

    # Add contour lines
    contour_lines = ax.contour(X, Y, display_temp_map, 10, colors='black', alpha=0.6, linewidths=0.5)
    ax.clabel(contour_lines, inline=True, fontsize=8, fmt='%.0f')

    # Add warning/critical thresholds if available
    warning_temp = cooling_data.get('coolant_warning_temp')
    critical_temp = cooling_data.get('coolant_critical_temp')

    if warning_temp:
        contour_warn = ax.contour(X, Y, display_temp_map, [temp_convert(warning_temp)], colors='orange', linestyles='--', linewidths=1.5)
        plt.clabel(contour_warn, inline=True, fontsize=9, fmt='Warn: %.0f')

    if critical_temp:
        contour_crit = ax.contour(X, Y, display_temp_map, [temp_convert(critical_temp)], colors='red', linestyles='-', linewidths=2)
        plt.clabel(contour_crit, inline=True, fontsize=9, fmt='Crit: %.0f')


    _apply_common_ax_settings(ax, xlabel=f'Vehicle Speed ({speed_unit})', ylabel='Engine Load (%)')

    plot_title = title if title else 'Cooling System Performance Map'
    ax.set_title(plot_title)

    plt.tight_layout()

    if save_path:
        save_plot(fig, save_path)

    return fig

#------------------------------------------------------------------------------
# Weight sensitivity plotting functions
#------------------------------------------------------------------------------
# plot_weight_sensitivity and plot_weight_distribution_sensitivity implemented here.

def plot_weight_sensitivity(sensitivity_data: Dict, title: Optional[str] = None,
                          unit_system: str = 'metric',
                          save_path: Optional[str] = None) -> Optional[plt.Figure]:
    """
    Plot weight sensitivity analysis results.

    Args:
        sensitivity_data: Dict with keys like 'weights', 'lap_times', 'acceleration_times'.
        title: Plot title.
        unit_system: Unit system ('metric' or 'imperial').
        save_path: Path to save plot (if None, not saved).

    Returns:
        Matplotlib figure or None if error.
    """
    weights = np.array(sensitivity_data.get('weights', []))
    lap_times = np.array(sensitivity_data.get('lap_times', []))
    accel_75m_times = np.array(sensitivity_data.get('time_75m', [])) # Using 75m time
    time_to_60mph = np.array(sensitivity_data.get('time_to_60mph', []))

    if len(weights) < 2:
        logger.error("Insufficient weight data points for sensitivity plot.")
        return None
    if len(lap_times) == 0 and len(accel_75m_times) == 0 and len(time_to_60mph) == 0:
        logger.error("No performance data provided for weight sensitivity plot.")
        return None

    fig, axes = plt.subplots(1, 2, figsize=(16, 6)) # Two main plots

    # Unit conversions
    if unit_system.lower() == 'imperial':
        weight_factor, weight_unit = KG_TO_LBS, "lbs"
    else:
        weight_factor, weight_unit = 1.0, "kg"

    display_weights = weights * weight_factor

    # Plot Lap Time Sensitivity
    ax1 = axes[0]
    if len(lap_times) == len(weights):
        valid_idx = ~np.isnan(lap_times)
        if np.any(valid_idx):
            ax1.plot(display_weights[valid_idx], lap_times[valid_idx], 'o-', color=COLOR_SCHEMES['default'][0], label='Lap Time')

            # Fit linear regression
            coeffs = np.polyfit(weights[valid_idx], lap_times[valid_idx], 1)
            poly = np.poly1d(coeffs)
            x_line = np.linspace(min(weights), max(weights), 100)
            ax1.plot(x_line * weight_factor, poly(x_line), '--', color=COLOR_SCHEMES['default'][0], alpha=0.7)
            sens = coeffs[0]
            ax1.text(0.05, 0.9, f"Sensitivity: {sens:.4f} s/kg", transform=ax1.transAxes, color=COLOR_SCHEMES['default'][0])

    _apply_common_ax_settings(ax1, xlabel=f'Vehicle Weight ({weight_unit})', ylabel='Lap Time (s)', title='Lap Time vs. Weight')
    ax1.legend(loc='best')

    # Plot Acceleration Sensitivity
    ax2 = axes[1]
    plotted_accel = False
    if len(accel_75m_times) == len(weights):
        valid_idx = ~np.isnan(accel_75m_times)
        if np.any(valid_idx):
            ax2.plot(display_weights[valid_idx], accel_75m_times[valid_idx], 'o-', color=COLOR_SCHEMES['default'][1], label='75m Time')
            coeffs = np.polyfit(weights[valid_idx], accel_75m_times[valid_idx], 1)
            poly = np.poly1d(coeffs)
            x_line = np.linspace(min(weights), max(weights), 100)
            ax2.plot(x_line * weight_factor, poly(x_line), '--', color=COLOR_SCHEMES['default'][1], alpha=0.7)
            sens = coeffs[0]
            ax2.text(0.05, 0.8, f"75m Sens: {sens:.4f} s/kg", transform=ax2.transAxes, color=COLOR_SCHEMES['default'][1])
            plotted_accel = True

    if len(time_to_60mph) == len(weights):
        valid_idx = ~np.isnan(time_to_60mph)
        if np.any(valid_idx):
            # Use secondary y-axis if scales are very different
            if plotted_accel and (np.max(accel_75m_times[valid_idx]) / np.max(time_to_60mph[valid_idx]) > 2 or np.max(time_to_60mph[valid_idx]) / np.max(accel_75m_times[valid_idx]) > 2):
                ax2b = ax2.twinx()
                ax2b.plot(display_weights[valid_idx], time_to_60mph[valid_idx], 'o-', color=COLOR_SCHEMES['default'][2], label='0-60 mph Time')
                ax2b.set_ylabel('0-60 mph Time (s)', color=COLOR_SCHEMES['default'][2])
                ax2b.tick_params(axis='y', labelcolor=COLOR_SCHEMES['default'][2])
                ax2b.legend(loc='upper right')
            else:
                ax2.plot(display_weights[valid_idx], time_to_60mph[valid_idx], 'o-', color=COLOR_SCHEMES['default'][2], label='0-60 mph Time')
                ax2.legend(loc='best')

            coeffs = np.polyfit(weights[valid_idx], time_to_60mph[valid_idx], 1)
            poly = np.poly1d(coeffs)
            x_line = np.linspace(min(weights), max(weights), 100)
            ax2.plot(x_line * weight_factor, poly(x_line), '--', color=COLOR_SCHEMES['default'][2], alpha=0.7)
            sens = coeffs[0]
            ax2.text(0.05, 0.7, f"0-60 Sens: {sens:.4f} s/kg", transform=ax2.transAxes, color=COLOR_SCHEMES['default'][2])
            plotted_accel = True


    if plotted_accel:
         _apply_common_ax_settings(ax2, xlabel=f'Vehicle Weight ({weight_unit})', ylabel='Time (s)', title='Acceleration vs. Weight')
         ax2.legend(loc='best') # Add legend if not already added by twinx
    else:
         ax2.text(0.5, 0.5, "No acceleration data available", ha='center', va='center')
         ax2.set_title('Acceleration vs. Weight')

    plot_title = title if title else 'Weight Sensitivity Analysis'
    fig.suptitle(plot_title, fontsize=DEFAULT_TITLE_SIZE+2)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust for suptitle

    if save_path:
        save_plot(fig, save_path)

    return fig

def plot_weight_distribution_sensitivity(sensitivity_data: Dict, title: Optional[str] = None,
                                       unit_system: str = 'metric',
                                       save_path: Optional[str] = None) -> Optional[plt.Figure]:
    """
    Plot weight distribution sensitivity analysis results.

    Args:
        sensitivity_data: Dict with 'front_weight_pct', 'lap_times', 'acceleration_times', 'lateral_acceleration'.
        title: Plot title.
        unit_system: Unit system ('metric' or 'imperial').
        save_path: Path to save plot (if None, not saved).

    Returns:
        Matplotlib figure or None if error.
    """
    front_weight_pct = np.array(sensitivity_data.get('front_weight_pct', []))
    lap_times = np.array(sensitivity_data.get('lap_times', []))
    accel_times = np.array(sensitivity_data.get('acceleration_times', []))
    lat_accel = np.array(sensitivity_data.get('lateral_acceleration', [])) # Expecting g's

    if len(front_weight_pct) < 2:
        logger.error("Insufficient weight distribution data points for sensitivity plot.")
        return None
    if len(lap_times) == 0 and len(accel_times) == 0 and len(lat_accel) == 0:
        logger.error("No performance data provided for weight distribution sensitivity plot.")
        return None

    fig, axes = plt.subplots(1, 3, figsize=(18, 6)) # Three main plots

    # Plot Lap Time Sensitivity
    ax1 = axes[0]
    if len(lap_times) == len(front_weight_pct):
        valid_idx = ~np.isnan(lap_times)
        if np.any(valid_idx):
            ax1.plot(front_weight_pct[valid_idx] * 100, lap_times[valid_idx], 'o-', color=COLOR_SCHEMES['default'][0])
            # Fit quadratic
            if len(front_weight_pct[valid_idx]) >= 3:
                coeffs = np.polyfit(front_weight_pct[valid_idx], lap_times[valid_idx], 2)
                poly = np.poly1d(coeffs)
                x_line = np.linspace(min(front_weight_pct), max(front_weight_pct), 100)
                ax1.plot(x_line * 100, poly(x_line), '--', color=COLOR_SCHEMES['default'][0], alpha=0.7)
                if coeffs[0] > 0: # Minimum exists
                    optimal_pct = -coeffs[1] / (2 * coeffs[0])
                    if min(front_weight_pct) <= optimal_pct <= max(front_weight_pct):
                         ax1.scatter([optimal_pct * 100], [poly(optimal_pct)], color='red', s=80, zorder=5, label=f'Optimal: {optimal_pct:.1%}')

    _apply_common_ax_settings(ax1, xlabel='Front Weight Distribution (%)', ylabel='Lap Time (s)', title='Lap Time vs. Weight Dist.')
    if np.any(valid_idx): ax1.legend(loc='best')

    # Plot Acceleration Sensitivity
    ax2 = axes[1]
    if len(accel_times) == len(front_weight_pct):
        valid_idx = ~np.isnan(accel_times)
        if np.any(valid_idx):
            ax2.plot(front_weight_pct[valid_idx] * 100, accel_times[valid_idx], 'o-', color=COLOR_SCHEMES['default'][1])
             # Fit quadratic
            if len(front_weight_pct[valid_idx]) >= 3:
                coeffs = np.polyfit(front_weight_pct[valid_idx], accel_times[valid_idx], 2)
                poly = np.poly1d(coeffs)
                x_line = np.linspace(min(front_weight_pct), max(front_weight_pct), 100)
                ax2.plot(x_line * 100, poly(x_line), '--', color=COLOR_SCHEMES['default'][1], alpha=0.7)
                if coeffs[0] > 0: # Minimum exists
                    optimal_pct = -coeffs[1] / (2 * coeffs[0])
                    if min(front_weight_pct) <= optimal_pct <= max(front_weight_pct):
                         ax2.scatter([optimal_pct * 100], [poly(optimal_pct)], color='red', s=80, zorder=5, label=f'Optimal: {optimal_pct:.1%}')

    _apply_common_ax_settings(ax2, xlabel='Front Weight Distribution (%)', ylabel='75m Time (s)', title='Acceleration vs. Weight Dist.')
    if np.any(valid_idx): ax2.legend(loc='best')

    # Plot Lateral Acceleration Sensitivity
    ax3 = axes[2]
    if len(lat_accel) == len(front_weight_pct):
        valid_idx = ~np.isnan(lat_accel)
        if np.any(valid_idx):
            ax3.plot(front_weight_pct[valid_idx] * 100, lat_accel[valid_idx], 'o-', color=COLOR_SCHEMES['default'][2])
             # Fit quadratic
            if len(front_weight_pct[valid_idx]) >= 3:
                coeffs = np.polyfit(front_weight_pct[valid_idx], lat_accel[valid_idx], 2)
                poly = np.poly1d(coeffs)
                x_line = np.linspace(min(front_weight_pct), max(front_weight_pct), 100)
                ax3.plot(x_line * 100, poly(x_line), '--', color=COLOR_SCHEMES['default'][2], alpha=0.7)
                if coeffs[0] < 0: # Maximum exists (we want max lateral G)
                    optimal_pct = -coeffs[1] / (2 * coeffs[0])
                    if min(front_weight_pct) <= optimal_pct <= max(front_weight_pct):
                         ax3.scatter([optimal_pct * 100], [poly(optimal_pct)], color='red', s=80, zorder=5, label=f'Optimal: {optimal_pct:.1%}')

    _apply_common_ax_settings(ax3, xlabel='Front Weight Distribution (%)', ylabel='Max Lateral Accel (g)', title='Cornering vs. Weight Dist.')
    if np.any(valid_idx): ax3.legend(loc='best')

    plot_title = title if title else 'Weight Distribution Sensitivity Analysis'
    fig.suptitle(plot_title, fontsize=DEFAULT_TITLE_SIZE+2)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust for suptitle

    if save_path:
        save_plot(fig, save_path)

    return fig

#------------------------------------------------------------------------------
# Endurance plotting functions
#------------------------------------------------------------------------------
# plot_endurance_results and plot_endurance_comparison implemented here.

def plot_endurance_results(endurance_data: Dict, title: Optional[str] = None,
                         unit_system: str = 'metric',
                         save_path: Optional[str] = None) -> Optional[plt.Figure]:
    """
    Plot endurance event simulation results.

    Args:
        endurance_data: Dict from EnduranceSimulator.simulate_endurance or similar.
        title: Plot title.
        unit_system: Unit system ('metric' or 'imperial').
        save_path: Path to save plot (if None, not saved).

    Returns:
        Matplotlib figure or None if error.
    """
    # Extract primary results
    lap_times = endurance_data.get('lap_times', [])
    lap_numbers = np.arange(1, len(lap_times) + 1)
    total_time = endurance_data.get('total_time', 0)
    completed = endurance_data.get('completed', False)
    dnf_reason = endurance_data.get('dnf_reason')

    # Extract detailed results if available
    detailed = endurance_data.get('detailed_results', {})
    fuel_consumption = detailed.get('fuel_consumption', []) # Per lap
    thermal_states = detailed.get('thermal_states', []) # Per lap
    component_wear = endurance_data.get('component_wear', {})
    reliability_events = endurance_data.get('reliability_events', []) # List of ReliabilityEvent enums or names

    if not lap_times:
        logger.warning("No lap times found in endurance data.")
        # Decide if you want to plot partial results or return None
        # For now, let's try to plot what we can

    # Create figure
    fig = plt.figure(figsize=(15, 14)) # Taller figure for more plots
    gs = gridspec.GridSpec(4, 2) # 4 rows, 2 columns

    # Unit conversions
    if unit_system.lower() == 'imperial':
        fuel_factor, fuel_unit = LITERS_TO_GAL, "gal"
        temp_convert = lambda t: t * 9/5 + 32 if t is not None else None
        temp_unit = "°F"
    else:
        fuel_factor, fuel_unit = 1.0, "L"
        temp_convert = lambda t: t
        temp_unit = "°C"

    # --- Plot Lap Times ---
    ax1 = fig.add_subplot(gs[0, :]) # Span both columns
    if len(lap_times) > 0:
        ax1.plot(lap_numbers, lap_times, 'b-o', linewidth=DEFAULT_LINE_WIDTH, markersize=DEFAULT_MARKER_SIZE, label='Lap Time')
        avg_lap_time = np.mean(lap_times)
        best_lap_time = np.min(lap_times)
        ax1.axhline(y=avg_lap_time, color='r', linestyle='--', alpha=0.7, label=f'Avg: {avg_lap_time:.2f}s')
        ax1.axhline(y=best_lap_time, color='g', linestyle='--', alpha=0.7, label=f'Best: {best_lap_time:.2f}s')

        # Mark reliability events
        for i, event_enum in enumerate(reliability_events):
            if event_enum is not None and event_enum.name != 'NONE':
                 if i < len(lap_times): # Check index bounds
                    ax1.scatter([lap_numbers[i]], [lap_times[i]], color='red', marker='x', s=100, zorder=5)
                    ax1.annotate(event_enum.name.replace('_', ' ').title(),
                               xy=(lap_numbers[i], lap_times[i]),
                               xytext=(lap_numbers[i], lap_times[i] + 0.05 * (np.max(lap_times) - np.min(lap_times))),
                               ha='center', rotation=30, size=8, color='red')

    _apply_common_ax_settings(ax1, xlabel='Lap Number', ylabel='Lap Time (s)', title='Lap Times')
    ax1.legend(loc='best')
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    # --- Plot Fuel Consumption ---
    ax2 = fig.add_subplot(gs[1, 0])
    if len(fuel_consumption) == len(lap_numbers):
        cumulative_fuel = np.cumsum(fuel_consumption) * fuel_factor
        ax2.plot(lap_numbers, cumulative_fuel, 'g-o', linewidth=DEFAULT_LINE_WIDTH, markersize=DEFAULT_MARKER_SIZE, label='Cumulative Fuel')

        # Add per-lap consumption on secondary axis
        ax2b = ax2.twinx()
        per_lap_fuel = np.array(fuel_consumption) * fuel_factor
        ax2b.bar(lap_numbers, per_lap_fuel, alpha=0.3, color='green', label=f'Per Lap ({fuel_unit})')
        ax2b.set_ylabel(f'Per Lap Consumption ({fuel_unit})', color='green')
        ax2b.tick_params(axis='y', colors='green')
        ax2b.legend(loc='upper right')

    _apply_common_ax_settings(ax2, xlabel='Lap Number', ylabel=f'Cumulative Fuel ({fuel_unit})', title='Fuel Consumption')
    ax2.legend(loc='upper left')
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))

    # --- Plot Thermal Profile ---
    ax3 = fig.add_subplot(gs[1, 1])
    if len(thermal_states) == len(lap_numbers):
        engine_temps = [temp_convert(s.get('engine_temp')) for s in thermal_states]
        coolant_temps = [temp_convert(s.get('coolant_temp')) for s in thermal_states]
        oil_temps = [temp_convert(s.get('oil_temp')) for s in thermal_states]

        if any(engine_temps): ax3.plot(lap_numbers, engine_temps, 'r-o', label='Engine')
        if any(coolant_temps): ax3.plot(lap_numbers, coolant_temps, 'b-o', label='Coolant')
        if any(oil_temps): ax3.plot(lap_numbers, oil_temps, 'y-o', label='Oil') # Yellow for oil

        # Add critical temperature lines if available
        limits = endurance_data.get('thermal_limits', {})
        crit_eng = limits.get('engine_critical_temp')
        crit_cool = limits.get('coolant_critical_temp')
        if crit_eng: ax3.axhline(temp_convert(crit_eng), color='red', linestyle='--', alpha=0.6, label='Eng Crit')
        if crit_cool: ax3.axhline(temp_convert(crit_cool), color='blue', linestyle='--', alpha=0.6, label='Cool Crit')

        _apply_common_ax_settings(ax3, xlabel='Lap Number', ylabel=f'End-of-Lap Temp ({temp_unit})', title='Thermal Profile')
        ax3.legend(loc='best')
        ax3.xaxis.set_major_locator(MaxNLocator(integer=True))

    # --- Plot Component Wear ---
    ax4 = fig.add_subplot(gs[2, 0])
    if component_wear:
        components = list(component_wear.keys())
        wear_values = [w * 100 for w in component_wear.values()] # Percentage
        colors = [plt.cm.OrRd(w / 100.0) for w in wear_values] # Color based on wear

        bars = ax4.barh(components, wear_values, color=colors)
        for bar, wear in zip(bars, wear_values):
            ax4.text(wear + 1, bar.get_y() + bar.get_height()/2, f'{wear:.1f}%', va='center')

        _apply_common_ax_settings(ax4, xlabel='Wear (%)', ylabel='Component', title='Component Wear')
        ax4.set_xlim(0, 105)

    # --- Plot Score Summary ---
    ax5 = fig.add_subplot(gs[2, 1])
    scores = endurance_data.get('score', {})
    if scores:
        labels = ['Endurance', 'Efficiency']
        values = [scores.get('endurance_score', 0), scores.get('efficiency_score', 0)]
        max_values = [scores.get('max_endurance_score', FS_MAX_ENDURANCE_POINTS), scores.get('max_efficiency_score', FS_MAX_EFFICIENCY_POINTS)]

        # Plot scores relative to max possible
        percentages = [v / max_v * 100 if max_v > 0 else 0 for v, max_v in zip(values, max_values)]

        bars = ax5.bar(labels, percentages, color=['blue', 'green'])
        for bar, score, max_score in zip(bars, values, max_values):
            ax5.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1, f'{score:.1f}/{max_score}', ha='center', va='bottom')

        _apply_common_ax_settings(ax5, ylabel='Score (% of Max)', title='Event Scores')
        ax5.set_ylim(0, 110)
        ax5.axhline(100, color='k', linestyle='--', alpha=0.3)

    # --- Status Text ---
    status_text = f"Status: {'Completed' if completed else 'DNF'}"
    if dnf_reason: status_text += f" (Reason: {dnf_reason})"
    if total_time: status_text += f" | Total Time: {total_time:.1f}s"
    if fuel_consumption: status_text += f" | Fuel Used: {np.sum(fuel_consumption)*fuel_factor:.2f} {fuel_unit}"
    if scores: status_text += f" | Total Score: {scores.get('total_score', 0):.1f}"

    plt.figtext(0.5, 0.01, status_text, ha='center', fontsize=DEFAULT_LABEL_SIZE,
               bbox=dict(facecolor='white', alpha=0.8, edgecolor='lightgray'))

    plot_title = title if title else 'Endurance Event Results'
    fig.suptitle(plot_title, fontsize=DEFAULT_TITLE_SIZE+2)

    plt.tight_layout(rect=[0, 0.05, 1, 0.95]) # Adjust for suptitle and bottom text

    if save_path:
        save_plot(fig, save_path)

    return fig


def plot_endurance_comparison(comparison_data: List[Dict], title: Optional[str] = None,
                            unit_system: str = 'metric',
                            save_path: Optional[str] = None) -> Optional[plt.Figure]:
    """
    Plot comparison of multiple endurance configurations.

    Args:
        comparison_data: List of dictionaries, each with endurance data and 'label'.
        title: Plot title.
        unit_system: Unit system ('metric' or 'imperial').
        save_path: Path to save plot (if None, not saved).

    Returns:
        Matplotlib figure or None if error.
    """
    if not comparison_data:
        logger.error("No comparison data provided for endurance comparison plot.")
        return None

    fig = plt.figure(figsize=(16, 12)) # Larger figure for comparison
    gs = gridspec.GridSpec(3, 2) # 3 rows, 2 columns

    # Unit conversions
    if unit_system.lower() == 'imperial':
        fuel_factor, fuel_unit = LITERS_TO_GAL, "gal"
    else:
        fuel_factor, fuel_unit = 1.0, "L"

    colors = COLOR_SCHEMES['default']
    config_labels = [d.get('label', f'Config {i+1}') for i, d in enumerate(comparison_data)]
    num_configs = len(config_labels)
    x = np.arange(num_configs)

    # --- Plot Total Score ---
    ax1 = fig.add_subplot(gs[0, 0])
    total_scores = [d.get('score', {}).get('total_score', 0) for d in comparison_data]
    max_total_score = comparison_data[0].get('score', {}).get('max_endurance_score', FS_MAX_ENDURANCE_POINTS) + \
                      comparison_data[0].get('score', {}).get('max_efficiency_score', FS_MAX_EFFICIENCY_POINTS)

    bars = ax1.bar(config_labels, total_scores, color=[colors[i % len(colors)] for i in range(num_configs)])
    ax1.axhline(max_total_score, color='k', linestyle='--', alpha=0.5, label=f'Max Possible ({max_total_score})')
    for bar, score in zip(bars, total_scores):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 2, f'{score:.1f}', ha='center', va='bottom', fontsize=8)
    _apply_common_ax_settings(ax1, ylabel='Total Score', title='Total Endurance Score Comparison')
    ax1.tick_params(axis='x', rotation=45, ha='right')
    ax1.legend(loc='best')

    # --- Plot Endurance vs Efficiency Score ---
    ax2 = fig.add_subplot(gs[0, 1])
    endurance_scores = [d.get('score', {}).get('endurance_score', 0) for d in comparison_data]
    efficiency_scores = [d.get('score', {}).get('efficiency_score', 0) for d in comparison_data]
    width = 0.35
    bars1 = ax2.bar(x - width/2, endurance_scores, width, label='Endurance', color=colors[0])
    bars2 = ax2.bar(x + width/2, efficiency_scores, width, label='Efficiency', color=colors[1])
    _apply_common_ax_settings(ax2, ylabel='Score Points', title='Endurance vs. Efficiency Score')
    ax2.set_xticks(x)
    ax2.set_xticklabels(config_labels, rotation=45, ha='right')
    ax2.legend(loc='best')

    # --- Plot Total Time ---
    ax3 = fig.add_subplot(gs[1, 0])
    total_times = [d.get('results', {}).get('total_time') for d in comparison_data]
    # Handle DNF cases for plotting
    plot_times = [t if t is not None and t < float('inf') else 0 for t in total_times]
    plot_labels = [f'{t:.1f}s' if t is not None and t < float('inf') else 'DNF' for t in total_times]
    valid_idx = [i for i, t in enumerate(total_times) if t is not None and t < float('inf')]

    if valid_idx:
        bars = ax3.bar([config_labels[i] for i in valid_idx], [plot_times[i] for i in valid_idx], color='purple')
        for bar, label in zip(bars, [plot_labels[i] for i in valid_idx]):
             ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 5, label, ha='center', va='bottom', fontsize=8)
    _apply_common_ax_settings(ax3, ylabel='Total Time (s)', title='Total Event Time (Lower is Better)')
    ax3.tick_params(axis='x', rotation=45, ha='right')

    # --- Plot Total Fuel ---
    ax4 = fig.add_subplot(gs[1, 1])
    total_fuels = [d.get('results', {}).get('total_fuel', 0) * fuel_factor for d in comparison_data]
    bars = ax4.bar(config_labels, total_fuels, color='green')
    for bar, fuel in zip(bars, total_fuels):
         ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05, f'{fuel:.2f}', ha='center', va='bottom', fontsize=8)
    _apply_common_ax_settings(ax4, ylabel=f'Total Fuel ({fuel_unit})', title='Total Fuel Consumption')
    ax4.tick_params(axis='x', rotation=45, ha='right')

    # --- Plot Reliability Comparison ---
    ax5 = fig.add_subplot(gs[2, 0])
    reliability_counts = [len([e for e in d.get('reliability_events', []) if e != 'NONE']) for d in comparison_data]
    bars = ax5.bar(config_labels, reliability_counts, color='red')
    for bar, count in zip(bars, reliability_counts):
         ax5.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1, f'{count}', ha='center', va='bottom', fontsize=9)
    _apply_common_ax_settings(ax5, ylabel='Number of Issues', title='Reliability Issues Count')
    ax5.tick_params(axis='x', rotation=45, ha='right')
    ax5.yaxis.set_major_locator(MaxNLocator(integer=True))

    # --- Plot Wear Comparison (Example: Max Wear) ---
    ax6 = fig.add_subplot(gs[2, 1])
    max_wear = []
    for data in comparison_data:
        wear = data.get('results', {}).get('component_wear', {})
        max_wear.append(max(wear.values()) * 100 if wear else 0) # Max wear %

    bars = ax6.bar(config_labels, max_wear, color='orange')
    for bar, wear in zip(bars, max_wear):
         ax6.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1, f'{wear:.1f}%', ha='center', va='bottom', fontsize=8)
    _apply_common_ax_settings(ax6, ylabel='Max Wear (%)', title='Maximum Component Wear')
    ax6.tick_params(axis='x', rotation=45, ha='right')
    ax6.set_ylim(0, max(max_wear)*1.1 if max_wear else 10)


    plot_title = title if title else 'Endurance Configuration Comparison'
    fig.suptitle(plot_title, fontsize=DEFAULT_TITLE_SIZE+2)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust for suptitle

    if save_path:
        save_plot(fig, save_path)

    return fig