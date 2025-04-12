"""
Core track generation logic using Voronoi diagrams.
Based on the methodology by Bai Li (ETH Zurich ASL).
"""

import os
import numpy as np
from scipy import spatial, interpolate, signal
from shapely.geometry import Point, LineString, Polygon
from shapely.ops import transform # For scaling geometry
from datetime import datetime
import csv
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import gpxpy
import gpxpy.gpx
import math
import logging
import random
from typing import List, Dict, Tuple, Optional

# Import local enums
from .enums import TrackMode, SimType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("FSTrackGenerator")

class FSTrackGenerator:
    """Generates Formula Student compliant tracks using Voronoi diagrams."""

    def __init__(self,
                 base_dir: str, # Directory where metadata.csv is stored
                 output_dir_override: Optional[str] = None, # Explicit dir for tracks
                 visualize: bool = False,
                 track_width: float = 3.0,
                 min_length: float = 200.0,
                 max_length: float = 500.0, # Increased max length default
                 curvature_threshold: float = 1.0 / 3.75, # Max curvature (1/min_radius)
                 straight_threshold: float = 1.0 / 50.0, # Min curvature for straights
                 start_straight_length: float = 10.0, # Min length for start straight
                 n_points: int = 80, # Increased points for more complexity
                 n_regions: int = 25, # Increased regions
                 bounds: Tuple[float, float] = (0.0, 150.0), # Generation area bounds (m)
                 cone_spacing: float = 3.5): # Target cone spacing (m)
        """
        Initialize the track generator.

        Args:
            base_dir: Directory for metadata file.
            output_dir_override: Specific directory to save generated track files. If None, defaults to 'generated_tracks' inside base_dir.
            visualize: If True, show plots during generation.
            track_width: Standard width of the track (m).
            min_length: Minimum centerline length (m).
            max_length: Maximum centerline length (m).
            curvature_threshold: Maximum allowed curvature (1/minimum_radius).
            straight_threshold: Curvature below this is considered straight.
            start_straight_length: Minimum length for the start/finish straight (m).
            n_points: Number of initial random points for Voronoi.
            n_regions: Number of Voronoi regions to combine for track shape.
            bounds: Tuple (min_bound, max_bound) for the generation area.
            cone_spacing: Target distance between cones along boundaries (m).
        """
        self.TRACK_WIDTH = track_width
        self.MIN_LENGTH = min_length
        self.MAX_LENGTH = max_length
        self.CURVATURE_THRESHOLD = curvature_threshold
        self.STRAIGHT_THRESHOLD = straight_threshold
        self.START_STRAIGHT_LENGTH = start_straight_length

        self.N_POINTS = n_points
        self.N_REGIONS = n_regions
        self.MIN_BOUND, self.MAX_BOUND = bounds
        self.CONE_SPACING = cone_spacing

        self.visualize = visualize
        self.base_dir = base_dir # For metadata
        self.output_dir = output_dir_override if output_dir_override else os.path.join(base_dir, "generated_tracks")
        self.metadata_file = os.path.join(self.base_dir, "track_metadata.csv")

        self._ensure_directories()

        # Internal state, reset per generation
        self.track_centerline_x: Optional[np.ndarray] = None
        self.track_centerline_y: Optional[np.ndarray] = None
        self.track_curvature: Optional[np.ndarray] = None
        self.track_length: Optional[float] = None
        self.cones_left: Optional[np.ndarray] = None
        self.cones_right: Optional[np.ndarray] = None
        self.start_position: Optional[np.ndarray] = None # Center of start line
        self.start_heading: Optional[float] = None # Heading angle in radians
        self.start_cones: Optional[np.ndarray] = None # Orange cones for start/finish

    def _ensure_directories(self):
        """Create output directory and metadata file if they don't exist."""
        os.makedirs(self.output_dir, exist_ok=True)
        # Also ensure base_dir exists for metadata
        os.makedirs(self.base_dir, exist_ok=True)

        if not os.path.exists(self.metadata_file):
            headers = ['filename', 'filepath', 'track_length', 'num_cones',
                       'track_width', 'generation_mode', 'generation_time',
                       'min_radius', 'avg_radius'] # Added radius stats
            try:
                with open(self.metadata_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(headers)
            except IOError as e:
                 logger.error(f"Failed to create metadata file {self.metadata_file}: {e}")


    def _closest_node(self, node: np.ndarray, nodes: np.ndarray, k: int = 0) -> int:
        """Finds the index of the k-th closest node in 'nodes' to 'node'."""
        if k >= len(nodes): k = len(nodes) - 1 # Prevent index error
        deltas = nodes - node
        dist_sq = np.einsum('ij,ij->i', deltas, deltas) # More efficient squared distance
        # Use argpartition for efficiency (finds k-th smallest without full sort)
        return np.argpartition(dist_sq, k)[k]

    def _clockwise_sort(self, points: np.ndarray) -> np.ndarray:
        """Sorts 2D points clockwise around their centroid."""
        center = np.mean(points, axis=0)
        # Calculate angles relative to the center point
        angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
        # Sort points based on angle
        return points[np.argsort(angles)]

    def _calculate_curvature(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculates curvature and radius using finite differences."""
        # Use numpy gradient for derivatives
        dx_dt = np.gradient(x)
        dy_dt = np.gradient(y)
        d2x_dt2 = np.gradient(dx_dt)
        d2y_dt2 = np.gradient(dy_dt)

        # Curvature formula: k = (dx*d2y - dy*d2x) / (dx^2 + dy^2)^(3/2)
        numerator = dx_dt * d2y_dt2 - dy_dt * d2x_dt2
        denominator = (dx_dt**2 + dy_dt**2)**1.5

        # Avoid division by zero for stationary points
        curvature = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 1e-9)

        # Radius of curvature R = 1 / |k|
        radius = np.divide(1.0, np.abs(curvature), out=np.full_like(curvature, float('inf')), where=np.abs(curvature) > 1e-9)

        return curvature, radius

    def _bounded_voronoi(self, points: np.ndarray):
        """Creates a Voronoi diagram bounded by reflecting points."""
        # Mirror points across boundaries to enforce boundary conditions
        x_min, x_max = self.MIN_BOUND, self.MAX_BOUND
        y_min, y_max = self.MIN_BOUND, self.MAX_BOUND
        margin = (x_max - x_min) * 0.2 # Use a slightly larger margin

        mirrored_x_min = np.copy(points); mirrored_x_min[:, 0] = 2 * (x_min - margin) - points[:, 0]
        mirrored_x_max = np.copy(points); mirrored_x_max[:, 0] = 2 * (x_max + margin) - points[:, 0]
        mirrored_y_min = np.copy(points); mirrored_y_min[:, 1] = 2 * (y_min - margin) - points[:, 1]
        mirrored_y_max = np.copy(points); mirrored_y_max[:, 1] = 2 * (y_max + margin) - points[:, 1]

        # Combine original and mirrored points
        all_points = np.vstack([points, mirrored_x_min, mirrored_x_max, mirrored_y_min, mirrored_y_max])

        try:
            vor = spatial.Voronoi(all_points)
            # Filter regions to keep only those corresponding to original points
            # and that are finite (not open to infinity)
            vor.filtered_points = points # Store original points
            vor.filtered_regions = []
            for i, region_idx in enumerate(vor.point_region[:len(points)]): # Only check original points
                 region = vor.regions[region_idx]
                 if region and -1 not in region: # Check if region is bounded
                      vor.filtered_regions.append(region)
            return vor
        except spatial.qhull.QhullError as e:
            logger.error(f"QhullError during Voronoi generation: {e}. Points might be degenerate.")
            return None # Indicate failure

    def _get_track_polygon(self, vor, mode: TrackMode) -> Optional[Polygon]:
        """Selects Voronoi regions and creates the track outline polygon."""
        if not vor or not vor.filtered_regions:
             logger.warning("No valid filtered Voronoi regions available.")
             return None

        input_points = vor.filtered_points
        num_input_points = len(input_points)
        num_filtered_regions = len(vor.filtered_regions)

        if num_filtered_regions < self.N_REGIONS:
             logger.warning(f"Only {num_filtered_regions} finite regions found, less than requested {self.N_REGIONS}.")
             if num_filtered_regions < 3: return None # Need at least 3 regions
             num_to_select = num_filtered_regions
        else:
             num_to_select = self.N_REGIONS

        # --- Select Regions ---
        if mode == TrackMode.EXPAND:
            # Start from a random point and expand outwards
            start_idx = random.randrange(num_input_points)
            selected_indices = {start_idx}
            queue = [start_idx]
            while len(selected_indices) < num_to_select and queue:
                current_idx = queue.pop(0)
                # Find neighbors (this requires analyzing Voronoi connectivity, simplified here)
                # Simplified: find k-nearest neighbors
                for k in range(1, min(num_to_select + 5, num_input_points)): # Look at nearby points
                     neighbor_idx = self._closest_node(input_points[current_idx], input_points, k=k)
                     if neighbor_idx not in selected_indices:
                          selected_indices.add(neighbor_idx)
                          queue.append(neighbor_idx)
                          if len(selected_indices) >= num_to_select: break
            selected_point_indices = list(selected_indices)

        elif mode == TrackMode.EXTEND:
             # Select points close to a random line segment
             p1_idx, p2_idx = random.sample(range(num_input_points), 2)
             line = LineString([input_points[p1_idx], input_points[p2_idx]])
             distances = [Point(p).distance(line) for p in input_points]
             # Partition to find indices of N smallest distances
             selected_point_indices = np.argpartition(distances, num_to_select)[:num_to_select]

        else: # RANDOM
            selected_point_indices = random.sample(range(num_input_points), num_to_select)

        # --- Combine Regions ---
        selected_vertices = set()
        point_region_map = vor.point_region
        for idx in selected_point_indices:
             region_indices = vor.regions[point_region_map[idx]]
             if region_indices and -1 not in region_indices:
                 selected_vertices.update(region_indices)

        if not selected_vertices:
             logger.warning("No valid vertices selected for the track polygon.")
             return None

        # Get coordinates of selected vertices
        track_vertices = vor.vertices[list(selected_vertices)]

        # Sort vertices to form a polygon
        try:
            sorted_track_vertices = self._clockwise_sort(track_vertices)
            track_polygon = Polygon(sorted_track_vertices)
            # Simplify polygon slightly to remove potential self-intersections from Voronoi artifacts
            track_polygon = track_polygon.simplify(0.1, preserve_topology=True)
            # Buffer slightly inwards then outwards to smooth small irregularities
            track_polygon = track_polygon.buffer(-0.05, join_style=2).buffer(0.05, join_style=2)

            if track_polygon.is_empty or not track_polygon.is_valid:
                 logger.warning("Generated track polygon is invalid or empty after processing.")
                 return None

            # Ensure it's a single polygon (not MultiPolygon)
            if track_polygon.geom_type == 'MultiPolygon':
                 track_polygon = max(track_polygon.geoms, key=lambda p: p.area) # Take largest polygon
                 if track_polygon.geom_type != 'Polygon': return None # Still not valid

            return track_polygon
        except Exception as e:
             logger.error(f"Error creating track polygon from vertices: {e}")
             return None


    def _check_track_constraints(self, x: np.ndarray, y: np.ndarray, curvature: np.ndarray, radius: np.ndarray) -> bool:
        """Check if the generated track meets length and curvature constraints."""
        # 1. Calculate Length
        self.track_length = np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))
        if not (self.MIN_LENGTH <= self.track_length <= self.MAX_LENGTH):
            logger.debug(f"Constraint fail: Length {self.track_length:.1f}m not in range [{self.MIN_LENGTH}, {self.MAX_LENGTH}]m")
            return False

        # 2. Check Maximum Curvature (Minimum Radius)
        # Radius = 1 / Curvature
        min_radius_found = np.min(radius[radius > 0]) if np.any(radius > 0) else float('inf')
        min_allowed_radius = 1.0 / self.CURVATURE_THRESHOLD
        if min_radius_found < min_allowed_radius:
            logger.debug(f"Constraint fail: Min radius {min_radius_found:.2f}m < allowed {min_allowed_radius:.2f}m")
            return False

        # Optional: Check for excessively long straight sections (can lead to boring tracks)
        # straight_mask = np.abs(curvature) < self.STRAIGHT_THRESHOLD
        # Add logic to find contiguous straight sections and check length

        return True

    def _find_start_position(self, x: np.ndarray, y: np.ndarray, curvature: np.ndarray) -> bool:
        """Find a suitable start/finish straight."""
        if len(x) < 3: return False

        # Calculate segment lengths
        segment_lengths = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
        cumulative_length = np.concatenate(([0], np.cumsum(segment_lengths)))

        # Find potential straight sections
        straight_mask = np.abs(curvature) < self.STRAIGHT_THRESHOLD
        potential_straights = []
        current_start_idx = None

        for i in range(len(straight_mask)):
            if straight_mask[i]:
                if current_start_idx is None:
                    current_start_idx = i
            else:
                if current_start_idx is not None:
                    # End of a potential straight
                    end_idx = i - 1
                    if end_idx > current_start_idx: # Need at least 2 points
                         length = cumulative_length[end_idx+1] - cumulative_length[current_start_idx]
                         if length >= self.START_STRAIGHT_LENGTH:
                              potential_straights.append({'start': current_start_idx, 'end': end_idx, 'length': length})
                    current_start_idx = None
            # Handle straight at the end of the track
            if i == len(straight_mask) - 1 and current_start_idx is not None:
                 end_idx = i
                 if end_idx > current_start_idx:
                    length = cumulative_length[end_idx] - cumulative_length[current_start_idx] # Use end index for length
                    if length >= self.START_STRAIGHT_LENGTH:
                        potential_straights.append({'start': current_start_idx, 'end': end_idx, 'length': length})


        if not potential_straights:
            logger.warning(f"Could not find a straight section >= {self.START_STRAIGHT_LENGTH}m for start/finish.")
            return False

        # Select the longest straight section
        longest_straight = max(potential_straights, key=lambda s: s['length'])
        start_idx = longest_straight['start']
        end_idx = longest_straight['end']

        # Position start line near the beginning of the longest straight
        # Place start position 1m into the straight
        target_dist_into_straight = 1.0
        start_line_idx = start_idx
        dist_covered = 0.0
        while start_line_idx < end_idx:
            dist_covered += segment_lengths[start_line_idx]
            if dist_covered >= target_dist_into_straight:
                break
            start_line_idx += 1
        start_line_idx = min(start_line_idx, end_idx) # Ensure it doesn't go past the end

        # Start position is at start_line_idx
        self.start_position = np.array([x[start_line_idx], y[start_line_idx]])

        # Calculate heading using the next point on the straight
        next_idx = min(start_line_idx + 1, len(x) - 1)
        dx = x[next_idx] - x[start_line_idx]
        dy = y[next_idx] - y[start_line_idx]
        self.start_heading = np.arctan2(dy, dx)

        # Calculate orange cone positions based on start position and heading
        perp_vec = np.array([-np.sin(self.start_heading), np.cos(self.start_heading)])
        start_box_width = self.TRACK_WIDTH + 1.0 # Slightly wider start box

        # Cones defining the start line itself (at start_position)
        start_left = self.start_position + perp_vec * start_box_width / 2.0
        start_right = self.start_position - perp_vec * start_box_width / 2.0

        # Cones defining the finish line (e.g., 1m ahead)
        finish_center = self.start_position + np.array([np.cos(self.start_heading), np.sin(self.start_heading)]) * 1.0
        finish_left = finish_center + perp_vec * start_box_width / 2.0
        finish_right = finish_center - perp_vec * start_box_width / 2.0

        self.start_cones = np.array([start_left, start_right, finish_left, finish_right])

        return True

    def _place_cones(self, x: np.ndarray, y: np.ndarray):
        """Place cones along the track boundaries."""
        if len(x) < 2: return

        # Calculate tangent and normal vectors
        tangents = np.gradient(np.column_stack((x, y)), axis=0)
        norms = np.linalg.norm(tangents, axis=1)
        valid = norms > 1e-6
        tangents[valid] /= norms[valid, np.newaxis]
        # Handle potential issues at start/end for closed loop
        if np.allclose(x[0], x[-1]) and np.allclose(y[0], y[-1]):
             tangents[0] = tangents[-1] = (tangents[1] + tangents[-2]) / 2.0
             tangents[0] /= np.linalg.norm(tangents[0])
             tangents[-1] = tangents[0]

        normals = np.zeros_like(tangents)
        normals[:, 0] = -tangents[:, 1]
        normals[:, 1] = tangents[:, 0]

        # Create boundary lines
        half_width = self.TRACK_WIDTH / 2.0
        left_boundary_pts = np.column_stack((x, y)) + normals * half_width[:, np.newaxis]
        right_boundary_pts = np.column_stack((x, y)) - normals * half_width[:, np.newaxis]

        # Interpolate points along boundaries for even cone spacing
        left_line = LineString(left_boundary_pts)
        right_line = LineString(right_boundary_pts)

        num_cones_left = max(3, int(np.ceil(left_line.length / self.CONE_SPACING)))
        num_cones_right = max(3, int(np.ceil(right_line.length / self.CONE_SPACING)))

        left_distances = np.linspace(0, left_line.length, num_cones_left)
        right_distances = np.linspace(0, right_line.length, num_cones_right)

        self.cones_left = np.array([list(left_line.interpolate(d).coords)[0] for d in left_distances])
        self.cones_right = np.array([list(right_line.interpolate(d).coords)[0] for d in right_distances])

    def generate_track(self, mode: TrackMode = TrackMode.EXTEND, max_retries: int = 20) -> Optional[Dict]:
        """
        Main function to generate a valid Formula Student track.

        Args:
            mode: Track generation mode.
            max_retries: Maximum attempts to generate a valid track.

        Returns:
            Dictionary with track metadata if successful, None otherwise.
        """
        for retry in range(max_retries):
            logger.info(f"--- Track Generation Attempt {retry + 1}/{max_retries} (Mode: {mode.name}) ---")
            try:
                # 1. Generate initial points
                input_points = np.random.uniform(self.MIN_BOUND, self.MAX_BOUND, (self.N_POINTS, 2))

                # 2. Create Bounded Voronoi Diagram
                vor = self._bounded_voronoi(input_points)
                if vor is None: continue # Retry if Voronoi failed

                # 3. Select Regions and Create Track Polygon
                track_polygon = self._get_track_polygon(vor, mode)
                if track_polygon is None: continue # Retry if polygon creation failed

                # 4. Get Centerline from Polygon Exterior
                x_poly, y_poly = track_polygon.exterior.coords.xy
                x_poly, y_poly = np.array(x_poly), np.array(y_poly)
                # Ensure closed loop if it's not already
                if not np.allclose(x_poly[0], x_poly[-1]) or not np.allclose(y_poly[0], y_poly[-1]):
                     x_poly = np.append(x_poly, x_poly[0])
                     y_poly = np.append(y_poly, y_poly[0])

                # 5. Interpolate and Smooth Centerline (more points for smoother curvature)
                num_centerline_points = max(200, int(track_polygon.length * 2)) # More points based on length
                try:
                    # Use periodic spline for closed tracks
                    tck, u = interpolate.splprep([x_poly, y_poly], s=1.0, per=1) # Allow some smoothing (s=1.0)
                    t_interp = np.linspace(0, 1, num_centerline_points)
                    x_center, y_center = interpolate.splev(t_interp, tck, der=0)
                except Exception as e:
                     logger.warning(f"Spline interpolation failed: {e}. Using polygon vertices directly.")
                     x_center, y_center = x_poly, y_poly # Fallback

                # 6. Calculate Curvature and Radius
                curvature, radius = self._calculate_curvature(x_center, y_center)

                # 7. Check Constraints (Length and Curvature)
                if not self._check_track_constraints(x_center, y_center, curvature, radius):
                    logger.debug(f"Attempt {retry+1} failed constraints check.")
                    continue # Retry

                # 8. Find Start/Finish Position
                if not self._find_start_position(x_center, y_center, curvature):
                     logger.debug(f"Attempt {retry+1} failed to find start position.")
                     continue # Retry

                # 9. Place Cones along Boundaries
                self._place_cones(x_center, y_center)

                # Store final centerline data
                self.track_centerline_x = x_center
                self.track_centerline_y = y_center
                self.track_curvature = curvature

                # 10. Prepare Metadata
                # Calculate average radius (excluding straights)
                corner_radii = radius[radius < (1.0 / self.STRAIGHT_THRESHOLD)]
                avg_radius = np.mean(corner_radii) if len(corner_radii) > 0 else float('inf')
                min_radius = np.min(radius[radius > 0]) if np.any(radius > 0) else float('inf')

                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                mode_str = mode.name.lower()
                base_filename = f"fs_track_{mode_str}_{timestamp}"
                csv_filename = f"{base_filename}.csv" # Default save format
                filepath = os.path.join(self.output_dir, csv_filename)

                metadata = {
                    'filename': csv_filename,
                    'filepath': filepath,
                    'track_length': self.track_length,
                    'num_cones': len(self.cones_left) + len(self.cones_right) + len(self.start_cones),
                    'track_width': self.TRACK_WIDTH,
                    'generation_mode': mode.name,
                    'generation_time': datetime.now().isoformat(),
                    'min_radius': min_radius,
                    'avg_radius': avg_radius
                }

                # 11. Save default FSDS CSV track file
                if not self._export_fsds_csv(filepath):
                     logger.error(f"Failed to save track data to {filepath}")
                     continue # Retry

                # 12. Append metadata
                try:
                    pd.DataFrame([metadata]).to_csv(self.metadata_file, mode='a', header=False, index=False)
                except IOError as e:
                     logger.error(f"Failed to append metadata to {self.metadata_file}: {e}")
                except Exception as e:
                     logger.error(f"Unexpected error writing metadata: {e}")


                logger.info(f"Successfully generated track '{csv_filename}' after {retry + 1} attempts.")
                return metadata # Success

            except Exception as e:
                logger.error(f"Error during track generation attempt {retry + 1}: {e}", exc_info=True)
                # Continue to next retry

        logger.error(f"Failed to generate a valid track after {max_retries} attempts.")
        return None # Failed after all retries

    def export_track(self, output_path: str, sim_type: SimType = SimType.FSDS) -> bool:
        """Export the *last generated* track in the specified format."""
        if self.cones_left is None or self.cones_right is None or self.start_cones is None:
            logger.error("No track data generated yet to export.")
            return False

        logger.info(f"Exporting track to {sim_type.name} format: {output_path}")
        try:
            if sim_type == SimType.FSSIM:
                return self._export_fssim_yaml(output_path)
            elif sim_type == SimType.FSDS:
                return self._export_fsds_csv(output_path)
            elif sim_type == SimType.GPX:
                return self._export_gpx(output_path)
            else:
                logger.error(f"Unsupported export format: {sim_type}")
                return False
        except Exception as e:
            logger.error(f"Error during export to {sim_type.name}: {e}", exc_info=True)
            return False

    def _export_fssim_yaml(self, output_path: str) -> bool:
        """Exports track in FSSIM YAML format."""
        data = {
            'cones_left': self.cones_left.tolist(),
            'cones_right': self.cones_right.tolist(),
            'cones_orange': [], # Usually empty unless specific orange cones placed
            'cones_orange_big': self.start_cones.tolist(),
            'starting_pose_cg': [ # Center of gravity starting pose
                float(self.start_position[0]),
                float(self.start_position[1]),
                float(self.start_heading) # Yaw angle in radians
            ]
            # Optional: tk_device (timing gates) can be added if calculated
        }
        try:
            with open(output_path, 'w') as f:
                yaml.dump(data, f, default_flow_style=None, sort_keys=False)
            return True
        except IOError as e:
            logger.error(f"Failed to write FSSIM YAML file {output_path}: {e}")
            return False

    def _export_fsds_csv(self, output_path: str) -> bool:
        """Exports track in FSDS CSV format."""
        try:
            with open(output_path, 'w', newline='') as f:
                writer = csv.writer(f)
                # Writer header (optional but good practice)
                # writer.writerow(['color', 'x', 'y', 'z', 'dx', 'dy', 'dz']) # Example header

                # Write left cones (blue)
                for cone in self.cones_left:
                    writer.writerow(['blue', f"{cone[0]:.4f}", f"{cone[1]:.4f}", 0, 0.01, 0.01, 0]) # z, dx, dy, dz are often unused placeholders

                # Write right cones (yellow)
                for cone in self.cones_right:
                    writer.writerow(['yellow', f"{cone[0]:.4f}", f"{cone[1]:.4f}", 0, 0.01, 0.01, 0])

                # Write start/finish cones (big_orange)
                for cone in self.start_cones:
                    writer.writerow(['big_orange', f"{cone[0]:.4f}", f"{cone[1]:.4f}", 0, 0.01, 0.01, 0])
            return True
        except IOError as e:
             logger.error(f"Failed to write FSDS CSV file {output_path}: {e}")
             return False

    def _export_gpx(self, output_path: str, lat_offset=51.197682, lon_offset=5.323411):
        """Exports track centerline in GPX format."""
        if self.track_centerline_x is None or self.track_centerline_y is None:
             logger.error("Cannot export GPX: Track centerline not generated.")
             return False

        gpx = gpxpy.gpx.GPX()
        gpx_track = gpxpy.gpx.GPXTrack(name="Formula Student Generated Track")
        gpx.tracks.append(gpx_track)
        gpx_segment = gpxpy.gpx.GPXTrackSegment()
        gpx_track.segments.append(gpx_segment)

        # Add centerline points as track points
        # Convert local X, Y to pseudo Lat, Lon using offset and simple scaling
        earth_radius = 6371000.0 # meters
        center_x = np.mean(self.track_centerline_x)
        center_y = np.mean(self.track_centerline_y)

        for x, y in zip(self.track_centerline_x, self.track_centerline_y):
            # Simple planar conversion - not geodesically accurate but fine for local tracks
            lat = lat_offset + math.degrees((y - center_y) / earth_radius)
            lon = lon_offset + math.degrees((x - center_x) / (earth_radius * math.cos(math.radians(lat_offset))))
            gpx_segment.points.append(gpxpy.gpx.GPXTrackPoint(latitude=lat, longitude=lon))

        try:
            with open(output_path, 'w') as f:
                f.write(gpx.to_xml(version='1.1')) # Specify version
            return True
        except IOError as e:
            logger.error(f"Failed to write GPX file {output_path}: {e}")
            return False

    def plot_track(self):
        """Plots the generated track with cones and start/finish line."""
        if self.cones_left is None or self.cones_right is None or self.start_cones is None or self.start_position is None:
            logger.warning("Cannot plot track: Generation not complete or failed.")
            return

        plt.figure(figsize=(12, 10))
        ax = plt.gca()

        # Plot cones
        ax.scatter(self.cones_left[:, 0], self.cones_left[:, 1], color='blue', s=25, label='Left Cones (Blue)')
        ax.scatter(self.cones_right[:, 0], self.cones_right[:, 1], color='yellow', s=25, label='Right Cones (Yellow)', edgecolors='k', linewidths=0.5)
        ax.scatter(self.start_cones[:, 0], self.start_cones[:, 1], color='orange', s=50, label='Start/Finish Cones', edgecolors='k')

        # Plot start position and heading arrow
        ax.plot(self.start_position[0], self.start_position[1], 'go', markersize=10, label='Start Line Center')
        dir_vec = np.array([np.cos(self.start_heading), np.sin(self.start_heading)])
        ax.arrow(self.start_position[0], self.start_position[1], dir_vec[0]*5, dir_vec[1]*5,
                 head_width=1.5, head_length=2.0, fc='green', ec='green', linewidth=1.5, zorder=5)

        # Plot centerline for reference
        if self.track_centerline_x is not None:
             ax.plot(self.track_centerline_x, self.track_centerline_y, 'k--', alpha=0.4, linewidth=1, label='Centerline (Approx)')

        ax.set_aspect('equal', adjustable='box')
        _apply_common_ax_settings(ax, xlabel='X (m)', ylabel='Y (m)', title='Generated Formula Student Track')
        ax.legend(loc='best')

        plt.tight_layout()
        if self.visualize:
            plt.show()
        else:
            # If not visualizing interactively, ensure the plot is closed
            plt.close()