"""
Track representation and analysis module for Formula Student simulations.

Defines classes for representing track geometry, segments, and racing lines,
along with methods for loading, processing, analyzing, and visualizing tracks.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Any
from enum import Enum, auto
import logging
import yaml
import os
import csv
from scipy.interpolate import interp1d, CubicSpline
from scipy.signal import savgol_filter # For smoothing
from shapely.geometry import Point, LineString, Polygon
from shapely.ops import transform

# Import local dependencies (enums, utils)
try:
    from ..track_generator.enums import SimType
    from ..utils.track_utils import preprocess_track_points, ensure_unique_values
    from ..utils.plotting import save_plot, _apply_common_ax_settings, COLOR_SCHEMES
    from ..utils.constants import GRAVITY, DEG_TO_RAD
except ImportError:
    # Fallbacks for standalone execution
    class SimType(Enum): FSSIM='yaml'; FSDS='csv'; GPX='gpx'
    def preprocess_track_points(d): return d
    def ensure_unique_values(x): return x
    def save_plot(fig, path, **kwargs): pass
    def _apply_common_ax_settings(ax, **kwargs): pass
    COLOR_SCHEMES = {'default': plt.cm.tab10.colors}
    GRAVITY = 9.81; DEG_TO_RAD = np.pi / 180.0
    logger = logging.getLogger("Track_Fallback")
    logger.warning("Could not import all necessary modules. Using fallbacks.")


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Track")

# --- Enums and Helper Classes ---

class TrackSegmentType(Enum):
    """Type of track segment based on curvature."""
    STRAIGHT = auto()
    CORNER_LEFT = auto() # Positive curvature
    CORNER_RIGHT = auto() # Negative curvature
    HAIRPIN_LEFT = auto() # Sharp left
    HAIRPIN_RIGHT = auto() # Sharp right
    UNKNOWN = auto()

class TrackSegment:
    """Represents a logical segment of the track."""
    def __init__(self, segment_type: TrackSegmentType, start_idx: int, end_idx: int):
        self.segment_type = segment_type
        self.start_idx = start_idx
        self.end_idx = end_idx
        # Calculated properties
        self.length_m: Optional[float] = None
        self.avg_curvature: Optional[float] = None
        self.min_radius_m: Optional[float] = None # Minimum radius within segment
        self.entry_speed_mps: Optional[float] = None # Placeholder
        self.exit_speed_mps: Optional[float] = None # Placeholder

    def calculate_properties(self, distances: np.ndarray, curvature: np.ndarray):
        """Calculate length and curvature properties."""
        if self.end_idx >= len(distances) or self.start_idx < 0: return # Index check
        self.length_m = distances[self.end_idx] - distances[self.start_idx]
        if self.start_idx <= self.end_idx:
             segment_curvature = curvature[self.start_idx : self.end_idx + 1]
             self.avg_curvature = np.mean(segment_curvature) if len(segment_curvature) > 0 else 0.0
             abs_curve = np.abs(segment_curvature)
             max_curve = np.max(abs_curve) if len(abs_curve) > 0 else 0.0
             self.min_radius_m = 1.0 / max_curve if max_curve > 1e-6 else float('inf')

    def get_type_name(self) -> str:
        """Return human-readable type name."""
        return self.segment_type.name.replace('_', ' ').title()

    def __str__(self) -> str:
        details = f"{self.get_type_name()} ({self.start_idx}-{self.end_idx})"
        if self.length_m is not None: details += f", Len={self.length_m:.1f}m"
        if self.min_radius_m is not None and self.min_radius_m < 1000: details += f", Min R={self.min_radius_m:.1f}m"
        return details

class RacingLine:
    """Represents and calculates an optimized racing line for a Track."""
    def __init__(self, track: 'Track'): # Forward reference Track
        self.track = track
        self.line_points: Optional[np.ndarray] = None # Optimized (x, y) points
        self.distances_m: Optional[np.ndarray] = None # Distance along racing line
        self.curvature: Optional[np.ndarray] = None # Curvature along racing line
        self.speed_profile_mps: Optional[np.ndarray] = None # Speed profile
        self.lap_time_s: Optional[float] = None
        logger.debug("RacingLine initialized.")

    def _calculate_geometry(self):
        """Calculate distance and curvature for the current line_points."""
        if self.line_points is None or len(self.line_points) < 2: return

        # Calculate distances
        segment_lengths = np.sqrt(np.sum(np.diff(self.line_points, axis=0)**2, axis=1))
        self.distances_m = np.concatenate(([0], np.cumsum(segment_lengths)))

        # Calculate curvature
        if len(self.line_points) >= 3:
            dx = np.gradient(self.line_points[:, 0])
            dy = np.gradient(self.line_points[:, 1])
            d2x = np.gradient(dx)
            d2y = np.gradient(dy)
            denominator = np.maximum((dx**2 + dy**2)**1.5, 1e-9) # Avoid zero division
            self.curvature = (dx * d2y - dy * d2x) / denominator
            # Optionally smooth curvature
            # self.curvature = savgol_filter(self.curvature, window_length=11, polyorder=3)
        else:
            self.curvature = np.zeros(len(self.line_points))

    def optimize_geometric(self, smoothness: float = 0.1, inside_bias: float = 0.8) -> bool:
        """Optimize racing line using geometric approach (inside of corners)."""
        logger.info("Optimizing racing line using geometric method...")
        if not self.track.has_geometry():
             logger.error("Cannot optimize geometrically: Track geometry missing.")
             return False

        track_points = self.track.points
        track_width = self.track.width
        track_curvature = self.track.curvature # Use track's centerline curvature
        n_points = len(track_points)

        # 1. Calculate initial track position offset based on curvature
        # Move towards inside (-1 for right turn, +1 for left turn), scaled by curvature magnitude
        # Use tanh to smoothly approach the limit, scaled by inside_bias
        max_abs_curve = np.max(np.abs(track_curvature)) if np.any(track_curvature) else 1.0
        normalized_curve = track_curvature / max(max_abs_curve, 1e-6)
        track_positions = -np.tanh(normalized_curve * 5.0) * inside_bias # Sharpness=5, Bias=0.8

        # 2. Smooth the desired track positions
        # Savitzky-Golay filter can preserve features better than moving average
        window_size = max(5, n_points // 15) # Window size relative to track points
        if window_size % 2 == 0: window_size += 1 # Must be odd
        if window_size >= 3:
             track_positions = savgol_filter(track_positions, window_size, polyorder=3, mode='wrap') # Wrap for closed track

        # Clamp to limits
        track_positions = np.clip(track_positions, -self.track.position_limit, self.track.position_limit)

        # 3. Generate racing line points from smoothed positions
        self.line_points = self.track.get_points_at_position(track_positions)

        # 4. Recalculate geometry for the new line
        self._calculate_geometry()
        logger.info("Geometric racing line optimization complete.")
        return True

    def optimize_minimum_curvature(self, smoothness: float = 1.0, max_iter: int = 10) -> bool:
         """Optimize racing line to minimize maximum curvature using iterative smoothing."""
         logger.info("Optimizing racing line to minimize curvature...")
         if not self.track.has_geometry(): return False

         # Start with geometric line or centerline
         if self.line_points is None:
             self.optimize_geometric()
             if self.line_points is None: # If geometric also failed
                  self.line_points = self.track.points.copy() # Use centerline
                  self._calculate_geometry()

         initial_line = self.line_points.copy()
         current_line = initial_line
         normals = self.track.get_normals() # Normals of centerline
         track_width = self.track.width
         limit = self.track.position_limit

         for iteration in range(max_iter):
             last_line = current_line.copy()
             # Smooth the current line (Laplacian smoothing)
             smoothed_line = (np.roll(current_line, 1, axis=0) + np.roll(current_line, -1, axis=0)) / 2.0

             # Project smoothed points back towards centerline normal within limits
             for i in range(len(current_line)):
                 center_p = self.track.points[i]
                 normal_vec = normals[i]
                 width = track_width[i]

                 # Vector from centerline point to smoothed point
                 vec_to_smoothed = smoothed_line[i] - center_p

                 # Project onto normal vector to find distance from centerline
                 dist_from_center = np.dot(vec_to_smoothed, normal_vec)

                 # Clamp distance based on track width and limit
                 max_dist = limit * width / 2.0
                 clamped_dist = np.clip(dist_from_center, -max_dist, max_dist)

                 # New point is centerline + clamped_dist along normal
                 current_line[i] = center_p + normal_vec * clamped_dist

             # Check for convergence (optional)
             change = np.max(np.linalg.norm(current_line - last_line, axis=1))
             logger.debug(f" Min Curvature Iter {iteration+1}, Max Change: {change:.4f}")
             if change < 0.01: break # Converged

         self.line_points = current_line
         self._calculate_geometry()
         logger.info(f"Minimum curvature optimization complete after {iteration+1} iterations.")
         return True

    # optimize_lap_time method is too complex for base track class, belongs in optimal_lap_time.py

    def calculate_speed_profile(self, vehicle: 'Vehicle') -> Optional[np.ndarray]: # Forward reference Vehicle
        """Calculate speed profile (m/s) along this racing line."""
        if self.line_points is None or self.distances_m is None or self.curvature is None:
             logger.error("Racing line geometry not calculated.")
             return None
        if vehicle is None:
             logger.error("Vehicle object required for speed profile calculation.")
             return None

        n_points = len(self.line_points)
        speeds = np.zeros(n_points)
        distances = self.distances_m
        curvature = self.curvature

        # Get vehicle limits
        cornering_calc = CorneringPerformance(vehicle) # Use helper class
        max_vehicle_speed = vehicle.calculate_max_speed() if hasattr(vehicle, 'calculate_max_speed') else 50.0 # m/s fallback

        # --- Pass 1: Cornering Speed Limit ---
        for i in range(n_points):
            if abs(curvature[i]) > 1e-6:
                radius = 1.0 / abs(curvature[i])
                corner_speed_limit = cornering_calc.calculate_max_cornering_speed(radius)
                speeds[i] = min(corner_speed_limit, max_vehicle_speed)
            else:
                speeds[i] = max_vehicle_speed

        # --- Pass 2: Braking Limit (Backward Pass) ---
        max_braking_accel = vehicle.calculate_max_deceleration() if hasattr(vehicle, 'calculate_max_deceleration') else -1.8*GRAVITY # m/s^2

        for i in range(n_points - 2, -1, -1):
            ds = distances[i+1] - distances[i]
            if ds < 1e-6: continue
            v_next = speeds[i+1]
            # v_curr^2 <= v_next^2 - 2*a*ds (a is negative for braking)
            speed_limit_sq = v_next**2 - 2 * max_braking_accel * ds
            if speed_limit_sq < 0: speed_limit_sq = 0
            speeds[i] = min(speeds[i], np.sqrt(speed_limit_sq))

        # --- Pass 3: Acceleration Limit (Forward Pass) ---
        # Max accel depends on speed and gear - simplified here
        max_accel = 1.2 * GRAVITY # m/s^2 fallback

        for i in range(n_points - 1):
            ds = distances[i+1] - distances[i]
            if ds < 1e-6:
                 speeds[i+1] = min(speeds[i+1], speeds[i])
                 continue

            v_current = speeds[i]
            # Estimate max accel at this speed (more accurate would use gear/rpm)
            if hasattr(vehicle, 'calculate_max_acceleration'):
                 # Estimate gear based on speed - very approximate!
                 est_gear = int(np.clip(v_current // 15, 1, vehicle.drivetrain.num_gears)) if vehicle.drivetrain else 1
                 current_max_accel = vehicle.calculate_max_acceleration(v_current, est_gear)
            else:
                 current_max_accel = max_accel * (1 - v_current / max_vehicle_speed) # Simple reduction with speed

            # v_next^2 <= v_curr^2 + 2*a*ds
            speed_limit_sq = v_current**2 + 2 * current_max_accel * ds
            speeds[i+1] = min(speeds[i+1], np.sqrt(speed_limit_sq))

        self.speed_profile_mps = speeds
        self._calculate_lap_time() # Calculate time based on speed profile
        logger.info(f"Racing line speed profile calculated. Lap Time: {self.lap_time_s:.3f}s")
        return self.speed_profile_mps

    def _calculate_lap_time(self):
         """Calculate lap time from distances and speed profile."""
         if self.speed_profile_mps is None or self.distances_m is None or len(self.speed_profile_mps) < 2:
             self.lap_time_s = None
             return

         dt = np.zeros(len(self.distances_m) - 1)
         ds = np.diff(self.distances_m)
         avg_speed = (self.speed_profile_mps[:-1] + self.speed_profile_mps[1:]) / 2.0

         # Avoid division by zero for stationary segments
         valid_mask = avg_speed > 1e-3
         dt[valid_mask] = ds[valid_mask] / avg_speed[valid_mask]
         # Assign large time penalty for zero-speed segments if ds > 0
         dt[~valid_mask & (ds > 1e-6)] = 10.0 # 10s penalty

         self.lap_time_s = np.sum(dt)

    def get_stats(self) -> Dict:
         """Return statistics about the racing line."""
         stats = {'length_m': self.distances_m[-1] if self.distances_m is not None else None}
         if self.curvature is not None:
             abs_curve = np.abs(self.curvature)
             stats['max_curvature'] = np.max(abs_curve) if len(abs_curve)>0 else 0
             min_radius = 1.0 / stats['max_curvature'] if stats['max_curvature'] > 1e-6 else float('inf')
             stats['min_radius_m'] = min_radius
         if self.speed_profile_mps is not None:
             stats['max_speed_mps'] = np.max(self.speed_profile_mps)
             stats['avg_speed_mps'] = np.mean(self.speed_profile_mps)
         stats['lap_time_s'] = self.lap_time_s
         return stats

    def plot(self, plot_track=True, color='r', label='Racing Line', show_speed=True, **kwargs):
         """Plot the racing line, optionally with the track."""
         if self.line_points is None:
             logger.warning("Cannot plot: Racing line not calculated.")
             return

         if plot_track:
             # Plot track boundaries for context
             if self.track.left_boundary is not None and self.track.right_boundary is not None:
                  plt.plot(self.track.left_boundary[:, 0], self.track.left_boundary[:, 1], 'k--', alpha=0.3, linewidth=0.5)
                  plt.plot(self.track.right_boundary[:, 0], self.track.right_boundary[:, 1], 'k--', alpha=0.3, linewidth=0.5)

         if show_speed and self.speed_profile_mps is not None:
             # Color line by speed
             points = self.line_points.reshape(-1, 1, 2)
             segments = np.concatenate([points[:-1], points[1:]], axis=1)
             norm = plt.Normalize(vmin=np.min(self.speed_profile_mps), vmax=np.max(self.speed_profile_mps))
             lc = plt.matplotlib.collections.LineCollection(segments, cmap=plt.cm.viridis, norm=norm)
             lc.set_array(self.speed_profile_mps)
             lc.set_linewidth(kwargs.get('linewidth', 2))
             plt.gca().add_collection(lc)
             cbar = plt.colorbar(lc)
             cbar.set_label('Speed (m/s)')
             # Add label manually since LineCollection doesn't handle it well
             plt.plot([], [], color=color, label=label, linewidth=kwargs.get('linewidth', 2)) # Dummy plot for legend
         else:
             plt.plot(self.line_points[:, 0], self.line_points[:, 1], color=color, label=label, **kwargs)

         plt.gca().set_aspect('equal', adjustable='box')
         plt.xlabel("X (m)")
         plt.ylabel("Y (m)")
         plt.title("Track Racing Line")
         plt.legend()
         plt.grid(True, alpha=0.3)


# --- Main Track Class ---

class Track:
    """Represents a complete track for vehicle simulation."""
    def __init__(self, name: Optional[str] = "Unnamed Track"):
        self.name = name
        self.source_file: Optional[str] = None
        self.is_closed_circuit: bool = True # Assume closed unless specified otherwise

        # Core Geometry (Centerline)
        self.points: Optional[np.ndarray] = None # Nx2 or Nx3 array (x, y, [z])
        self.distances: Optional[np.ndarray] = None # Cumulative distance along centerline
        self.curvature: Optional[np.ndarray] = None # Curvature at each point
        self.width: Optional[np.ndarray] = None # Track width at each point (can be scalar)
        self.total_length: float = 0.0

        # Optional Detailed Geometry
        self.left_boundary: Optional[np.ndarray] = None
        self.right_boundary: Optional[np.ndarray] = None
        self.cones_left: Optional[np.ndarray] = None
        self.cones_right: Optional[np.ndarray] = None
        self.elevation: Optional[np.ndarray] = None
        self.banking: Optional[np.ndarray] = None # Banking angle (radians)

        # Analysis Results
        self.segments: List[TrackSegment] = []
        self.racing_line: Optional[RacingLine] = None

        # Start/Finish Info
        self.start_position: np.ndarray = np.array([0.0, 0.0]) # Default start
        self.start_heading: float = 0.0 # Radians, East

        # Configuration
        self.position_limit = 0.95 # Max deviation for racing line relative to half-width

        logger.info(f"Track '{self.name}' initialized.")

    def has_geometry(self) -> bool:
        """Check if essential geometry (points, distances, curvature, width) is loaded."""
        return (self.points is not None and len(self.points) > 2 and
                self.distances is not None and len(self.distances) == len(self.points) and
                self.curvature is not None and len(self.curvature) == len(self.points) and
                self.width is not None and (np.isscalar(self.width) or len(self.width) == len(self.points)))


    def load_from_file(self, filepath: str) -> bool:
        """Load track data from various file formats."""
        logger.info(f"Attempting to load track from: {filepath}")
        if not os.path.exists(filepath):
            logger.error(f"Track file not found: {filepath}")
            return False

        self.source_file = filepath
        _, ext = os.path.splitext(filepath)
        ext = ext.lower()
        success = False

        try:
            if ext in ['.yaml', '.yml']: success = self._load_yaml(filepath)
            elif ext == '.csv': success = self._load_csv(filepath)
            elif ext == '.gpx': success = self._load_gpx(filepath)
            else: logger.error(f"Unsupported track file format: {ext}")

            if success:
                self._post_load_processing()
                logger.info(f"Successfully loaded and processed track '{self.name}' from {filepath}")
            else:
                 logger.error(f"Failed to load track data from {filepath}")

        except Exception as e:
            logger.error(f"Error loading track from {filepath}: {e}", exc_info=True)
            success = False

        return success

    # --- Loading Methods (_load_yaml, _load_csv, _load_fssim, _load_fsds, _load_gpx) ---
    # These would parse specific file formats and populate self.points, self.width, etc.
    # Implementations are similar to the generator's export methods but reversed.
    # Example for _load_yaml (simplified):
    def _load_yaml(self, filepath: str) -> bool:
         with open(filepath, 'r') as f: track_data = yaml.safe_load(f)
         # Check FSSIM format first
         if 'cones_left' in track_data and 'cones_right' in track_data:
             return self._load_fssim_format(track_data)
         # Check generic format
         elif 'track' in track_data and isinstance(track_data['track'], list):
             points = []
             widths = []
             elevations = []
             for p in track_data['track']:
                 points.append([p.get('x', 0.0), p.get('y', 0.0)])
                 widths.append(p.get('width', 3.0))
                 elevations.append(p.get('z', 0.0)) # Add elevation if present
             self.points = np.array(points)
             self.width = np.array(widths)
             self.elevation = np.array(elevations) if np.any(elevations) else None
             meta = track_data.get('metadata', {})
             self.name = meta.get('name', os.path.basename(filepath))
             self.is_closed_circuit = meta.get('closed_circuit', True)
             start = track_data.get('start', {})
             self.start_position = np.array([start.get('x', self.points[0,0]), start.get('y', self.points[0,1])])
             self.start_heading = start.get('direction', 0.0) # Radians
             return True
         else:
             logger.error("Invalid YAML track format.")
             return False

    # Placeholder for other loading methods
    def _load_csv(self, filepath: str) -> bool:
         logger.warning("_load_csv not fully implemented.")
         # Add logic to detect FSDS or generic CSV and parse accordingly
         # Example: Read into pandas, check columns, populate self.points etc.
         # For FSDS, call self._load_fsds_format
         try:
             # Sniff to guess dialect/header
             with open(filepath, 'r') as f:
                  header = f.readline().lower()
                  sniffer = csv.Sniffer()
                  dialect = sniffer.sniff(f.read(1024))
                  f.seek(0) # Reset read position

             # Check for FSDS format (cone colors)
             if 'blue' in header or 'yellow' in header or 'color' in header:
                 return self._load_fsds_format(filepath)
             else: # Assume generic x,y,[z],[width]
                 df = pd.read_csv(filepath, dialect=dialect)
                 if 'x' not in df.columns or 'y' not in df.columns:
                      logger.error("Generic CSV must contain 'x' and 'y' columns.")
                      return False
                 self.points = df[['x', 'y']].values
                 self.width = df['width'].values if 'width' in df.columns else np.full(len(self.points), 3.0)
                 self.elevation = df['z'].values if 'z' in df.columns else None
                 self.name = os.path.basename(filepath)
                 return True
         except Exception as e:
              logger.error(f"Error reading CSV {filepath}: {e}")
              return False

    def _load_fssim_format(self, track_data: Dict) -> bool:
        logger.info("Loading track from FSSIM (cone-based) format...")
        cones_left = np.array(track_data.get('cones_left', []))
        cones_right = np.array(track_data.get('cones_right', []))
        if len(cones_left) < 3 or len(cones_right) < 3: return False
        self.cones_left = cones_left
        self.cones_right = cones_right
        self._create_centerline_from_cones() # Generates self.points and self.width
        meta = track_data.get('metadata', {})
        self.name = meta.get('name', self.name)
        start = track_data.get('starting_pose_cg', [0,0,0])
        self.start_position = np.array(start[:2])
        self.start_heading = start[2] if len(start)>2 else 0.0
        return True # Assumes _create_centerline worked

    def _load_fsds_format(self, filepath: str) -> bool:
         logger.info("Loading track from FSDS (cone-based) format...")
         # Implementation similar to generator's _export_fsds_csv reversed
         cones_left = []
         cones_right = []
         with open(filepath, 'r') as f:
             reader = csv.reader(f)
             for row in reader:
                 if len(row) >= 3:
                     color = row[0].lower()
                     try:
                          x, y = float(row[1]), float(row[2])
                          if 'blue' in color: cones_left.append([x, y])
                          elif 'yellow' in color: cones_right.append([x, y])
                          # Could also parse orange cones for start/finish
                     except ValueError: continue # Skip invalid rows
         if len(cones_left) < 3 or len(cones_right) < 3: return False
         self.cones_left = np.array(cones_left)
         self.cones_right = np.array(cones_right)
         self._create_centerline_from_cones()
         self.name = os.path.basename(filepath)
         # Need logic to find start pos from orange cones if present
         return True

    def _load_gpx(self, filepath: str) -> bool:
        logger.warning("_load_gpx not fully implemented.")
        # Use gpxpy to parse file, convert lat/lon to local x/y
        # Needs a reference point for conversion
        return False


    def _create_centerline_from_cones(self):
        """Generate centerline and width estimates from cone data."""
        if self.cones_left is None or self.cones_right is None or \
           len(self.cones_left) < 2 or len(self.cones_right) < 2:
            logger.error("Insufficient cone data to create centerline.")
            return

        logger.debug("Creating centerline from cones...")
        # More robust centerline generation: Find pairs and average
        # This is complex. Simplified approach: Average nearest points
        # Use a KDTree for efficient nearest neighbor search
        tree_left = spatial.KDTree(self.cones_left)
        tree_right = spatial.KDTree(self.cones_right)

        centerline = []
        widths = []

        # Iterate through left cones, find nearest right, calculate midpoint & width
        for i, p_left in enumerate(self.cones_left):
             dist, idx_right = tree_right.query(p_left)
             p_right = self.cones_right[idx_right]
             centerline.append((p_left + p_right) / 2.0)
             widths.append(dist)

        # Iterate through right cones, find nearest left, calculate midpoint & width
        for i, p_right in enumerate(self.cones_right):
             dist, idx_left = tree_left.query(p_right)
             p_left = self.cones_left[idx_left]
             centerline.append((p_left + p_right) / 2.0)
             widths.append(dist) # Distance is the width at this pairing

        centerline = np.array(centerline)
        widths = np.array(widths)

        # Sort the centerline points (essential step)
        # Start near the point with min x+y coordinate as heuristic
        start_idx = np.argmin(np.sum(centerline, axis=1))
        sorted_indices = [start_idx]
        remaining_indices = set(range(len(centerline)))
        remaining_indices.remove(start_idx)
        tree_center = spatial.KDTree(centerline)

        while remaining_indices:
            last_idx = sorted_indices[-1]
            # Find nearest neighbors among remaining points
            k_to_check = min(10, len(remaining_indices))
            distances, indices = tree_center.query(centerline[last_idx], k=k_to_check + 1) # Query more than needed
            # Find the closest point *that is still remaining*
            found_next = False
            for idx in indices[1:]: # Skip self (dist=0)
                if idx in remaining_indices:
                    sorted_indices.append(idx)
                    remaining_indices.remove(idx)
                    found_next = True
                    break
            if not found_next:
                 # If no close neighbor found among remaining, might be fragmented. Take closest overall remaining.
                 if not remaining_indices: break
                 closest_remaining_idx = min(remaining_indices, key=lambda idx: np.linalg.norm(centerline[idx] - centerline[last_idx]))
                 sorted_indices.append(closest_remaining_idx)
                 remaining_indices.remove(closest_remaining_idx)


        # Reorder points and widths
        self.points = centerline[sorted_indices]
        self.width = widths[sorted_indices] # Width corresponding to the centerline point

        # Optional: Smooth centerline and width
        if len(self.points) > 10:
            window = max(5, len(self.points)//20) | 1 # Odd window size
            self.points[:, 0] = savgol_filter(self.points[:, 0], window, 3, mode='wrap')
            self.points[:, 1] = savgol_filter(self.points[:, 1], window, 3, mode='wrap')
            self.width = savgol_filter(self.width, window, 3, mode='wrap')
            self.width = np.clip(self.width, 1.5, 10.0) # Clip to reasonable FS widths

        logger.info(f"Centerline created from cones: {len(self.points)} points.")


    def _post_load_processing(self):
        """Calculate distances, curvature, segments, boundaries after loading points."""
        if self.points is None or len(self.points) < 3:
             logger.error("Cannot process track: Insufficient points.")
             return

        # 0. Preprocess (remove duplicates)
        track_dict = {'points': self.points, 'width': self.width}
        if self.elevation is not None: track_dict['elevation'] = self.elevation
        processed_data = preprocess_track_points(track_dict)
        self.points = processed_data['points']
        self.width = processed_data['width']
        self.elevation = processed_data.get('elevation')
        # Distances are recalculated below

        # 1. Calculate Distances
        self._calculate_distances()

        # 2. Calculate Curvature
        self._calculate_curvature()

        # 3. Calculate Boundaries (if width available)
        if self.width is not None:
             self._calculate_boundaries()
        else:
             logger.warning("Track width not available, cannot calculate boundaries.")

        # 4. Segment Track
        self._segment_track()

        # 5. Refine start/finish if not set from data
        if np.allclose(self.start_position, [0,0]):
            if not self._find_start_position(self.points[:,0], self.points[:,1], self.curvature):
                 logger.warning("Could not automatically determine start/finish line. Using first point.")
                 self.start_position = self.points[0]
                 self.start_heading = np.arctan2(self.points[1,1]-self.points[0,1], self.points[1,0]-self.points[0,0])


    def _calculate_distances(self):
        """Calculate cumulative distances along the centerline."""
        segment_lengths = np.sqrt(np.sum(np.diff(self.points, axis=0)**2, axis=1))
        self.distances = np.concatenate(([0], np.cumsum(segment_lengths)))
        self.total_length = self.distances[-1]
        logger.debug(f"Distances calculated. Total length: {self.total_length:.2f} m")

    def _calculate_curvature(self):
        """Calculate curvature and radius along the centerline."""
        if len(self.points) < 3:
            self.curvature = np.zeros(len(self.points))
            logger.warning("Cannot calculate curvature, less than 3 points.")
            return

        dx = np.gradient(self.points[:, 0])
        dy = np.gradient(self.points[:, 1])
        d2x = np.gradient(dx)
        d2y = np.gradient(dy)
        denominator = np.maximum((dx**2 + dy**2)**1.5, 1e-9)
        self.curvature = (dx * d2y - dy * d2x) / denominator

        # Smooth curvature
        window = max(5, len(self.points)//25) | 1 # Odd window
        if len(self.curvature) > window:
             self.curvature = savgol_filter(self.curvature, window, 3, mode='wrap')

        logger.debug("Curvature calculated.")

    def _calculate_boundaries(self):
        """Calculate left and right track boundaries."""
        n_points = len(self.points)
        width = self.width if not np.isscalar(self.width) else np.full(n_points, self.width)
        normals = self.get_normals()

        self.left_boundary = self.points + normals * (width / 2.0)[:, np.newaxis]
        self.right_boundary = self.points - normals * (width / 2.0)[:, np.newaxis]
        logger.debug("Track boundaries calculated.")

    def get_normals(self) -> np.ndarray:
        """Calculate normal vectors at each centerline point."""
        tangents = np.gradient(self.points, axis=0)
        norms = np.linalg.norm(tangents, axis=1)
        valid = norms > 1e-6
        tangents[valid] /= norms[valid, np.newaxis]
        # Handle potential issues at start/end for closed loop
        if self.is_closed_circuit and np.linalg.norm(self.points[0] - self.points[-1]) < 1e-3:
            tangents[0] = tangents[-1] = (tangents[1] + tangents[-2]) / 2.0 # Average neighbors
            norm0 = np.linalg.norm(tangents[0])
            if norm0 > 1e-6: tangents[0] /= norm0
            tangents[-1] = tangents[0]

        normals = np.zeros_like(tangents)
        normals[:, 0] = -tangents[:, 1]
        normals[:, 1] = tangents[:, 0]
        return normals

    def _segment_track(self, straight_thresh: float = 0.02, hairpin_thresh: float = 0.2):
        """Segment track into straights, corners, hairpins."""
        if self.curvature is None: return
        self.segments = []
        n_points = len(self.points)
        current_type = TrackSegmentType.UNKNOWN
        start_idx = 0

        for i in range(n_points):
            curve_abs = abs(self.curvature[i])
            segment_type = TrackSegmentType.UNKNOWN

            if curve_abs < straight_thresh: segment_type = TrackSegmentType.STRAIGHT
            elif curve_abs >= hairpin_thresh: # Hairpin
                 segment_type = TrackSegmentType.HAIRPIN_LEFT if self.curvature[i] > 0 else TrackSegmentType.HAIRPIN_RIGHT
            elif curve_abs >= straight_thresh: # Normal Corner
                 segment_type = TrackSegmentType.CORNER_LEFT if self.curvature[i] > 0 else TrackSegmentType.CORNER_RIGHT

            if i == 0: # First point
                current_type = segment_type
                start_idx = 0
            elif segment_type != current_type: # Type changed
                # Finalize previous segment
                segment = TrackSegment(current_type, start_idx, i - 1)
                segment.calculate_properties(self.distances, self.curvature)
                self.segments.append(segment)
                # Start new segment
                current_type = segment_type
                start_idx = i

            # Handle last point
            if i == n_points - 1:
                segment = TrackSegment(current_type, start_idx, i)
                segment.calculate_properties(self.distances, self.curvature)
                self.segments.append(segment)

        # Optional: Merge short segments (e.g., < 5m)
        self._merge_short_segments()
        logger.info(f"Track segmented into {len(self.segments)} segments.")

    def _merge_short_segments(self, min_length: float = 5.0):
         """Merge segments shorter than min_length with neighbors."""
         if len(self.segments) <= 1: return
         merged_segments = []
         i = 0
         while i < len(self.segments):
             current_seg = self.segments[i]
             # Check if current segment is short
             if current_seg.length_m is not None and current_seg.length_m < min_length and len(merged_segments) > 0:
                 # Try merging with the *previous* merged segment
                 prev_merged_seg = merged_segments[-1]
                 # Logic to decide the new type (e.g., keep type of longer segment)
                 new_type = prev_merged_seg.segment_type if prev_merged_seg.length_m >= current_seg.length_m else current_seg.segment_type
                 # Create a new segment spanning both
                 merged = TrackSegment(new_type, prev_merged_seg.start_idx, current_seg.end_idx)
                 merged.calculate_properties(self.distances, self.curvature)
                 merged_segments[-1] = merged # Replace previous with merged
                 logger.debug(f"Merged short segment {i} (type {current_seg.segment_type.name}) into previous.")
             else:
                 # Keep the current segment as is
                 merged_segments.append(current_seg)
             i += 1
         self.segments = merged_segments


    def get_point_at_distance(self, distance: float) -> Optional[np.ndarray]:
        """Interpolate (x, y, [z]) coordinates at a specific distance along the track."""
        if self.points is None or self.distances is None or not self.has_geometry(): return None
        # Wrap distance for closed track
        distance = distance % self.total_length if self.is_closed_circuit else distance
        # Clamp distance to track length
        distance = np.clip(distance, 0, self.total_length)

        interp_x = interp1d(self.distances, self.points[:, 0], bounds_error=False, fill_value='extrapolate')
        interp_y = interp1d(self.distances, self.points[:, 1], bounds_error=False, fill_value='extrapolate')
        point = np.array([interp_x(distance), interp_y(distance)])

        if self.elevation is not None and len(self.elevation) == len(self.points):
             interp_z = interp1d(self.distances, self.elevation, bounds_error=False, fill_value='extrapolate')
             point = np.append(point, interp_z(distance))

        return point

    def get_properties_at_distance(self, distance: float) -> Dict:
        """Get interpolated track properties (curvature, width, etc.) at a distance."""
        props = {}
        if not self.has_geometry(): return props

        distance = distance % self.total_length if self.is_closed_circuit else distance
        distance = np.clip(distance, 0, self.total_length)

        props['distance'] = distance
        props['curvature'] = float(np.interp(distance, self.distances, self.curvature))
        if np.isscalar(self.width):
            props['width'] = float(self.width)
        else:
            props['width'] = float(np.interp(distance, self.distances, self.width))
        if self.elevation is not None:
             props['elevation'] = float(np.interp(distance, self.distances, self.elevation))
        if self.banking is not None:
             props['banking'] = float(np.interp(distance, self.distances, self.banking))

        return props

    def get_points_at_position(self, track_positions: np.ndarray) -> Optional[np.ndarray]:
        """Calculate world coordinates for points offset from centerline."""
        if not self.has_geometry() or len(track_positions) != len(self.points):
             logger.error("Cannot calculate offset points: Geometry missing or length mismatch.")
             return None

        width = self.width if not np.isscalar(self.width) else np.full(len(self.points), self.width)
        normals = self.get_normals()
        offsets = track_positions * width / 2.0
        offset_points = self.points + normals * offsets[:, np.newaxis]
        return offset_points


    def calculate_racing_line(self, method: str = 'geometric', vehicle=None) -> Optional[RacingLine]:
        """Calculate and return a RacingLine object."""
        self.racing_line = RacingLine(self)
        success = False
        if method == 'geometric':
            success = self.racing_line.optimize_geometric()
        elif method == 'minimum_curvature':
            success = self.racing_line.optimize_minimum_curvature()
        # Add more methods like 'lap_time' which would call a more complex optimizer
        else:
            logger.warning(f"Unsupported racing line method: {method}. Using geometric.")
            success = self.racing_line.optimize_geometric()

        if success:
            # Optionally calculate speed profile immediately
            if vehicle:
                self.racing_line.calculate_speed_profile(vehicle)
            return self.racing_line
        else:
            self.racing_line = None # Clear if optimization failed
            return None


    def get_track_stats(self) -> Dict:
        """Return dictionary of track statistics."""
        stats = {
            'name': self.name,
            'source_file': self.source_file,
            'is_closed': self.is_closed_circuit,
            'length_m': self.total_length,
            'num_points': len(self.points) if self.points is not None else 0,
        }
        if self.width is not None:
             stats['avg_width_m'] = np.mean(self.width) if not np.isscalar(self.width) else self.width
        if self.curvature is not None:
             abs_curve = np.abs(self.curvature)
             stats['max_abs_curvature'] = np.max(abs_curve) if len(abs_curve)>0 else 0
             min_radius = 1.0 / stats['max_abs_curvature'] if stats['max_abs_curvature'] > 1e-6 else float('inf')
             stats['min_radius_m'] = min_radius
        if self.segments:
             stats['num_segments'] = len(self.segments)
             stats['segment_types'] = {t.name: sum(1 for s in self.segments if s.segment_type == t) for t in TrackSegmentType}
             stats['total_straight_length_m'] = sum(s.length_m for s in self.segments if s.segment_type == TrackSegmentType.STRAIGHT and s.length_m)
             stats['total_corner_length_m'] = self.total_length - stats['total_straight_length_m']
             stats['straight_percentage'] = (stats['total_straight_length_m'] / self.total_length * 100) if self.total_length > 0 else 0
        if self.cones_left is not None: stats['num_cones_left'] = len(self.cones_left)
        if self.cones_right is not None: stats['num_cones_right'] = len(self.cones_right)
        return stats

    def visualize(self, show_racing_line: bool = True, show_segments: bool = True,
                show_elevation: bool = False, save_path: Optional[str] = None):
        """Visualize the track using the centralized plotting function."""
        from ..utils.plotting import plot_track_layout # Local import

        if not self.has_geometry():
             logger.error("Cannot visualize track: Geometry not loaded/calculated.")
             return

        plot_data = {
            'points': self.points,
            'width': self.width,
            'segments': [{'type': s.segment_type.name, 'start_idx': s.start_idx, 'end_idx': s.end_idx} for s in self.segments],
            'elevation': self.elevation,
            'distance': self.distances, # Pass distances for elevation plot
            'name': self.name,
            'length': self.total_length,
            'start_position': self.start_position,
            'start_direction': self.start_heading
        }
        if self.racing_line and self.racing_line.line_points is not None:
            plot_data['racing_line'] = self.racing_line.line_points

        fig = plot_track_layout(plot_data, show_racing_line, show_segments, show_elevation,
                                title=f"Track: {self.name}", save_path=save_path)
        if fig: plt.show() # Show the plot generated by the utility


# --- Factory/Helper Functions ---

def create_example_track(difficulty: str = 'medium') -> Track:
    """Creates a simple procedural example track."""
    track = Track(f"Example Procedural Track ({difficulty})")
    logger.info(f"Creating example procedural track (difficulty: {difficulty})...")

    # Parameters based on difficulty
    num_segments = {'easy': 8, 'medium': 12, 'hard': 16}.get(difficulty, 12)
    min_straight = {'easy': 50, 'medium': 40, 'hard': 30}.get(difficulty, 40)
    max_straight = {'easy': 150, 'medium': 120, 'hard': 100}.get(difficulty, 120)
    min_radius = {'easy': 15, 'medium': 10, 'hard': 6}.get(difficulty, 10)
    max_radius = {'easy': 50, 'medium': 40, 'hard': 30}.get(difficulty, 40)
    max_angle = {'easy': 100, 'medium': 135, 'hard': 160}.get(difficulty, 135) # Max turn angle in degrees

    points = [[0, 0]]
    current_heading = 0.0 # Radians, starting East

    for i in range(num_segments):
        last_point = points[-1]
        # Alternate straight and corner
        if i % 2 == 0: # Straight
            length = np.random.uniform(min_straight, max_straight)
            end_point = last_point + length * np.array([np.cos(current_heading), np.sin(current_heading)])
            # Add intermediate points for straights too
            num_inter = max(2, int(length / 10)) # Point every ~10m
            straight_pts = np.linspace(last_point, end_point, num_inter)[1:] # Exclude start point
            points.extend(straight_pts.tolist())
        else: # Corner
            radius = np.random.uniform(min_radius, max_radius)
            angle_deg = np.random.uniform(30, max_angle) # Turn angle
            direction = np.random.choice([-1, 1]) # Left or Right turn
            angle_rad = direction * np.radians(angle_deg)
            arc_length = radius * abs(angle_rad)
            num_arc_points = max(3, int(arc_length / 3)) # Point every ~3m

            # Calculate corner center
            normal_dir = current_heading + direction * np.pi / 2.0
            center = last_point + radius * np.array([np.cos(normal_dir), np.sin(normal_dir)])

            # Generate points along the arc
            start_angle = normal_dir + direction * np.pi # Angle from center to start point
            angles = np.linspace(0, angle_rad, num_arc_points) + start_angle

            arc_points = center + radius * np.column_stack((np.cos(angles), np.sin(angles)))
            points.extend(arc_points[1:].tolist()) # Exclude start point of arc

            # Update heading
            current_heading += angle_rad

    track.points = np.array(points)
    track.width = 3.0 # Constant width
    track.is_closed_circuit = False # Procedural track is likely open

    # Perform post-processing
    track._post_load_processing()
    logger.info(f"Example track created. Length: {track.total_length:.1f}m")
    return track

def load_track_from_file(filepath: str) -> Optional[Track]:
    """Utility function to load a track."""
    track = Track()
    if track.load_from_file(filepath):
        return track
    return None

# Example usage:
if __name__ == "__main__":
    # Create an example track
    example_track = create_example_track(difficulty='medium')
    print("\nExample Track Stats:")
    print(yaml.dump(example_track.get_track_stats(), default_flow_style=False))

    # Visualize the example track
    example_track.visualize(show_racing_line=False)

    # Example of loading a track (assuming a file exists)
    # loaded_track = load_track_from_file("path/to/your/track.yaml")
    # if loaded_track:
    #     loaded_track.visualize()