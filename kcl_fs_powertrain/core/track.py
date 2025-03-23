"""
Track module for Formula Student powertrain simulation.

This module provides classes for representing, analyzing, and
visualizing tracks for vehicle performance simulation.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Any
from enum import Enum
import logging
import yaml
import os
import csv
from scipy.interpolate import interp1d, CubicSpline
from scipy.optimize import minimize
from scipy.spatial import distance

from ..track_generator.enums import TrackMode, SimType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("Track")


class TrackSegmentType(Enum):
    """Type of track segment."""
    STRAIGHT = 0
    CORNER_LEFT = 1
    CORNER_RIGHT = 2
    CHICANE = 3
    HAIRPIN = 4


class TrackSegment:
    """Represents a segment of track with consistent characteristics."""
    
    def __init__(self, segment_type: TrackSegmentType, start_idx: int, end_idx: int, radius: Optional[float] = None):
        """
        Initialize segment with indices into track points.
        
        Args:
            segment_type: Type of segment (straight, corner, etc.)
            start_idx: Starting index in track points array
            end_idx: Ending index in track points array
            radius: Radius of curvature for corners (None for straights)
        """
        self.segment_type = segment_type
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.radius = radius
        
        # Additional properties to be calculated
        self.length = 0.0
        self.entry_speed = 0.0
        self.exit_speed = 0.0
        self.min_speed = 0.0
        self.max_speed = 0.0
        self.banking = 0.0  # Average banking angle in degrees
        self.elevation_change = 0.0  # Net elevation change in meters
        self.surface_friction = 1.0  # Coefficient of friction (1.0 = standard)
    
    def calculate_properties(self, track_points: np.ndarray, distances: np.ndarray):
        """
        Calculate segment properties based on track points.
        
        Args:
            track_points: Array of track points (x, y, [z])
            distances: Cumulative distances along track
        """
        # Calculate length
        self.length = distances[self.end_idx] - distances[self.start_idx]
        
        # Additional calculations can be added for banking, elevation, etc.
        # depending on the dimensionality of track_points
        
        # For 3D tracks, calculate elevation change
        if track_points.shape[1] > 2:
            self.elevation_change = track_points[self.end_idx, 2] - track_points[self.start_idx, 2]
    
    def get_curvature(self) -> float:
        """
        Get the curvature of this segment (1/radius).
        
        Returns:
            Curvature value (0 for straights)
        """
        if self.radius is None or self.radius == 0:
            return 0.0
        return 1.0 / abs(self.radius)
    
    def get_segment_type_name(self) -> str:
        """
        Get the name of the segment type.
        
        Returns:
            String representation of segment type
        """
        if self.segment_type == TrackSegmentType.STRAIGHT:
            return "Straight"
        elif self.segment_type == TrackSegmentType.CORNER_LEFT:
            return "Left Corner"
        elif self.segment_type == TrackSegmentType.CORNER_RIGHT:
            return "Right Corner"
        elif self.segment_type == TrackSegmentType.CHICANE:
            return "Chicane"
        elif self.segment_type == TrackSegmentType.HAIRPIN:
            return "Hairpin"
        return "Unknown"
    
    def __str__(self) -> str:
        """String representation of segment."""
        segment_str = f"{self.get_segment_type_name()}: {self.length:.1f}m"
        if self.radius is not None and self.radius > 0:
            segment_str += f", R={self.radius:.1f}m"
        return segment_str


class Track:
    """Represents a complete track for vehicle simulation."""
    
    def __init__(self, name: Optional[str] = None):
        """
        Initialize an empty track.
        
        Args:
            name: Optional name for the track
        """
        self.name = name if name else "Unnamed Track"
        
        # Track geometry
        self.points = np.array([])  # Track centerline points (x, y, [z])
        self.width = np.array([])  # Track width at each point
        self.left_boundary = np.array([])  # Left track boundary
        self.right_boundary = np.array([])  # Right track boundary
        self.cones_left = np.array([])  # Left cones (for Formula Student)
        self.cones_right = np.array([])  # Right cones (for Formula Student)
        
        # Track analysis
        self.distances = np.array([])  # Cumulative distances along track
        self.curvature = np.array([])  # Curvature at each point
        self.segments = []  # List of track segments
        self.total_length = 0.0  # Total track length
        
        # Additional track properties
        self.banking = np.array([])  # Banking angle at each point
        self.elevation = np.array([])  # Elevation at each point
        self.surface_friction = np.array([])  # Surface friction at each point
        
        # Racing line
        self.racing_line = None  # Current racing line object
        
        # Start/finish info
        self.start_position = np.array([0.0, 0.0])
        self.start_direction = 0.0  # radians
        
        # Metadata
        self.source_file = None
        self.is_closed_circuit = True
        
        logger.info("Track object initialized")
    
    def load_from_file(self, filepath: str) -> bool:
        """
        Load track from file (YAML, CSV, etc.).
        
        Args:
            filepath: Path to track file
            
        Returns:
            True if successful, False otherwise
        """
        if not os.path.exists(filepath):
            logger.error(f"Track file not found: {filepath}")
            return False
        
        try:
            # Get file extension
            _, ext = os.path.splitext(filepath)
            ext = ext.lower()
            
            # Load based on file type
            if ext == '.yaml' or ext == '.yml':
                success = self._load_yaml_track(filepath)
            elif ext == '.csv':
                success = self._load_csv_track(filepath)
            else:
                logger.error(f"Unsupported track file format: {ext}")
                return False
            
            if success:
                self.source_file = filepath
                self._post_load_processing()
                logger.info(f"Successfully loaded track from {filepath}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error loading track from {filepath}: {str(e)}")
            return False
    
    def _load_yaml_track(self, filepath: str) -> bool:
        """
        Load track from YAML file.
        
        Args:
            filepath: Path to YAML file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(filepath, 'r') as f:
                track_data = yaml.safe_load(f)
            
            # Check if this is a FSSIM track format
            if 'cones_left' in track_data and 'cones_right' in track_data:
                # FSSIM format
                return self._load_fssim_format(track_data)
            
            # Generic YAML format
            if 'track' in track_data:
                points = []
                width_values = []
                
                for point in track_data['track']:
                    points.append([point['x'], point['y']])
                    width_values.append(point.get('width', 3.0))
                
                self.points = np.array(points)
                self.width = np.array(width_values)
                
                # Load metadata if available
                if 'metadata' in track_data:
                    metadata = track_data['metadata']
                    self.name = metadata.get('name', self.name)
                    self.is_closed_circuit = metadata.get('closed_circuit', True)
                
                # Load start position if available
                if 'start' in track_data:
                    start = track_data['start']
                    self.start_position = np.array([start.get('x', 0.0), start.get('y', 0.0)])
                    self.start_direction = start.get('direction', 0.0)
                
                return True
            
            logger.error(f"Invalid YAML track format in {filepath}")
            return False
            
        except Exception as e:
            logger.error(f"Error parsing YAML track file {filepath}: {str(e)}")
            return False
    
    def _load_csv_track(self, filepath: str) -> bool:
        """
        Load track from CSV file.
        
        Args:
            filepath: Path to CSV file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Try to determine CSV format
            with open(filepath, 'r') as f:
                first_line = f.readline().strip()
            
            # Check if it's an FSDS format (color,x,y,z)
            if 'color' in first_line or 'blue' in first_line or 'yellow' in first_line:
                return self._load_fsds_format(filepath)
            
            # Assume generic CSV with x,y[,width] columns
            points = []
            width_values = []
            
            with open(filepath, 'r') as f:
                reader = csv.reader(f)
                header = next(reader)  # Skip header
                
                has_width = len(header) > 2 and 'width' in header[2].lower()
                
                for row in reader:
                    if len(row) >= 2:
                        x = float(row[0])
                        y = float(row[1])
                        points.append([x, y])
                        
                        if has_width and len(row) > 2:
                            width_values.append(float(row[2]))
                        else:
                            width_values.append(3.0)  # Default width
            
            if len(points) < 3:
                logger.error(f"CSV track file {filepath} has too few points")
                return False
            
            self.points = np.array(points)
            self.width = np.array(width_values)
            
            return True
            
        except Exception as e:
            logger.error(f"Error parsing CSV track file {filepath}: {str(e)}")
            return False
    
    def _load_fsds_format(self, filepath: str) -> bool:
        """
        Load track from FSDS CSV format.
        
        Args:
            filepath: Path to FSDS CSV file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            cones_left = []
            cones_right = []
            cones_orange = []
            
            with open(filepath, 'r') as f:
                reader = csv.reader(f)
                
                for row in reader:
                    if len(row) >= 3:  # Need at least color, x, y
                        cone_type = row[0].strip().lower()
                        x = float(row[1])
                        y = float(row[2])
                        
                        if cone_type == 'blue':
                            cones_left.append([x, y])
                        elif cone_type == 'yellow':
                            cones_right.append([x, y])
                        elif 'orange' in cone_type:
                            cones_orange.append([x, y])
            
            if len(cones_left) < 3 or len(cones_right) < 3:
                logger.error(f"FSDS track file {filepath} has too few cones")
                return False
            
            # Store cone positions
            self.cones_left = np.array(cones_left)
            self.cones_right = np.array(cones_right)
            
            # Create centerline from cones
            self._create_centerline_from_cones()
            
            # Try to determine start position from orange cones
            if len(cones_orange) >= 2:
                # Assume the first two orange cones define the start line
                start_line = np.array([cones_orange[0], cones_orange[1]])
                self.start_position = np.mean(start_line, axis=0)
                
                # Calculate start direction (perpendicular to start line)
                dx = start_line[1, 0] - start_line[0, 0]
                dy = start_line[1, 1] - start_line[0, 1]
                self.start_direction = np.arctan2(dx, -dy)  # Perpendicular to start line
            
            return True
            
        except Exception as e:
            logger.error(f"Error parsing FSDS track file {filepath}: {str(e)}")
            return False
    
    def _load_fssim_format(self, track_data: Dict) -> bool:
        """
        Load track from FSSIM YAML dictionary.
        
        Args:
            track_data: Track data dictionary in FSSIM format
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if 'cones_left' in track_data and 'cones_right' in track_data:
                cones_left = np.array(track_data['cones_left'])
                cones_right = np.array(track_data['cones_right'])
                
                if len(cones_left) < 3 or len(cones_right) < 3:
                    logger.error("FSSIM track has too few cones")
                    return False
                
                # Store cone positions
                self.cones_left = cones_left
                self.cones_right = cones_right
                
                # Create centerline from cones
                self._create_centerline_from_cones()
                
                # Get start position if available
                if 'starting_pose_cg' in track_data:
                    start_pose = track_data['starting_pose_cg']
                    if len(start_pose) >= 3:
                        self.start_position = np.array([start_pose[0], start_pose[1]])
                        self.start_direction = start_pose[2]
                
                return True
            
            logger.error("Invalid FSSIM track format")
            return False
            
        except Exception as e:
            logger.error(f"Error parsing FSSIM track data: {str(e)}")
            return False
    
    def _create_centerline_from_cones(self):
        """Create centerline points from left and right cones."""
        try:
            # Simple approach: for each left cone, find closest right cone and vice versa
            centerline_points = []
            
            # Process left cones
            for left_cone in self.cones_left:
                distances = np.sum((self.cones_right - left_cone)**2, axis=1)
                closest_idx = np.argmin(distances)
                closest_right = self.cones_right[closest_idx]
                
                # Midpoint between cones
                midpoint = (left_cone + closest_right) / 2
                centerline_points.append(midpoint)
            
            # Process right cones
            for right_cone in self.cones_right:
                distances = np.sum((self.cones_left - right_cone)**2, axis=1)
                closest_idx = np.argmin(distances)
                closest_left = self.cones_left[closest_idx]
                
                # Midpoint between cones
                midpoint = (right_cone + closest_left) / 2
                centerline_points.append(midpoint)
            
            # Remove duplicates (approximately)
            unique_points = []
            for point in centerline_points:
                is_duplicate = False
                for existing_point in unique_points:
                    if np.linalg.norm(point - existing_point) < 0.5:  # Within 0.5m
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    unique_points.append(point)
            
            # Sort points to form a continuous path
            sorted_points = self._sort_centerline_points(unique_points)
            
            # Store results
            self.points = np.array(sorted_points)
            
            # Create width array based on distance between cone pairs
            self.width = np.zeros(len(self.points))
            for i, point in enumerate(self.points):
                # Find closest left and right cones
                left_dists = np.sum((self.cones_left - point)**2, axis=1)
                right_dists = np.sum((self.cones_right - point)**2, axis=1)
                
                closest_left_idx = np.argmin(left_dists)
                closest_right_idx = np.argmin(right_dists)
                
                closest_left = self.cones_left[closest_left_idx]
                closest_right = self.cones_right[closest_right_idx]
                
                # Calculate width as distance between cones
                self.width[i] = np.linalg.norm(closest_left - closest_right)
            
            logger.info(f"Created centerline with {len(self.points)} points from cones")
            
        except Exception as e:
            logger.error(f"Error creating centerline from cones: {str(e)}")
            # Fallback to simple approach
            if len(self.cones_left) > 0 and len(self.cones_right) > 0:
                # Just use average of all cones to create a crude centerline
                avg_left = np.mean(self.cones_left, axis=0)
                avg_right = np.mean(self.cones_right, axis=0)
                self.points = np.array([avg_left, avg_right])
                self.width = np.array([3.0, 3.0])
    
    def _sort_centerline_points(self, points: List[np.ndarray]) -> List[np.ndarray]:
        """
        Sort a list of points to form a continuous path.
        
        Args:
            points: List of points to sort
            
        Returns:
            Sorted list of points
        """
        if len(points) <= 2:
            return points
        
        # Convert to numpy array for easier manipulation
        points_array = np.array(points)
        
        # Start with the leftmost point (minimum x-coordinate)
        start_idx = np.argmin(points_array[:, 0])
        sorted_indices = [start_idx]
        remaining_indices = set(range(len(points_array)))
        remaining_indices.remove(start_idx)
        
        # Iteratively find the closest point
        while remaining_indices:
            last_idx = sorted_indices[-1]
            last_point = points_array[last_idx]
            
            min_dist = float('inf')
            min_idx = -1
            
            for idx in remaining_indices:
                dist = np.linalg.norm(points_array[idx] - last_point)
                if dist < min_dist:
                    min_dist = dist
                    min_idx = idx
            
            if min_idx >= 0:
                sorted_indices.append(min_idx)
                remaining_indices.remove(min_idx)
            else:
                # Should never happen, but just in case
                break
        
        # Return sorted points
        return [points_array[idx] for idx in sorted_indices]
    
    def load_from_generator(self, generator_output: Dict) -> bool:
        """
        Load track from track generator output.
        
        Args:
            generator_output: Output dictionary from FSTrackGenerator
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if 'cones_left' in generator_output and 'cones_right' in generator_output:
                self.cones_left = np.array(generator_output['cones_left'])
                self.cones_right = np.array(generator_output['cones_right'])
                
                # Create centerline from cones
                self._create_centerline_from_cones()
                
                # Get track metadata
                if 'metadata' in generator_output:
                    metadata = generator_output['metadata']
                    self.name = metadata.get('name', self.name)
                    self.total_length = metadata.get('track_length', 0.0)
                
                # Get start position
                if 'start_position' in generator_output:
                    self.start_position = np.array(generator_output['start_position'])
                
                if 'start_heading' in generator_output:
                    self.start_direction = generator_output['start_heading']
                
                self._post_load_processing()
                logger.info("Successfully loaded track from generator output")
                return True
            
            logger.error("Invalid generator output format")
            return False
            
        except Exception as e:
            logger.error(f"Error loading track from generator output: {str(e)}")
            return False
    
    def _post_load_processing(self):
        """Perform post-load processing such as calculating distances and curvature."""
        # Calculate cumulative distances
        self._calculate_distances()
        
        # Calculate curvature
        self.calculate_curvature()
        
        # Segment the track
        self.segment_track()
        
        # Create track boundaries
        self._create_track_boundaries()
    
    def _calculate_distances(self):
        """Calculate cumulative distances along the track centerline."""
        if len(self.points) < 2:
            logger.warning("Not enough points to calculate distances")
            self.distances = np.array([0.0])
            self.total_length = 0.0
            return
        
        # Calculate segment lengths
        segments = np.diff(self.points, axis=0)
        segment_lengths = np.sqrt(np.sum(segments**2, axis=1))
        
        # Cumulative distances
        self.distances = np.zeros(len(self.points))
        self.distances[1:] = np.cumsum(segment_lengths)
        
        # Total length
        self.total_length = self.distances[-1]
        
        logger.info(f"Track length: {self.total_length:.1f}m")
    
    def _create_track_boundaries(self):
        """Create left and right track boundaries based on centerline and width."""
        if len(self.points) < 2 or len(self.width) != len(self.points):
            logger.warning("Cannot create track boundaries with current data")
            return
        
        # Initialize boundary arrays
        self.left_boundary = np.zeros_like(self.points)
        self.right_boundary = np.zeros_like(self.points)
        
        # Calculate normal vectors at each point
        normals = self._calculate_normals()
        
        # Create boundaries
        for i in range(len(self.points)):
            half_width = self.width[i] / 2.0
            self.left_boundary[i] = self.points[i] + normals[i] * half_width
            self.right_boundary[i] = self.points[i] - normals[i] * half_width
    
    def _calculate_normals(self) -> np.ndarray:
        """
        Calculate normal vectors at each point.
        
        Returns:
            Array of normal vectors
        """
        n_points = len(self.points)
        normals = np.zeros_like(self.points)
        
        for i in range(n_points):
            # Get adjacent points (with wraparound for closed circuits)
            prev_idx = (i - 1) % n_points
            next_idx = (i + 1) % n_points
            
            # Calculate tangent vector
            tangent = self.points[next_idx] - self.points[prev_idx]
            
            # Normalize
            if np.linalg.norm(tangent) > 1e-6:
                tangent = tangent / np.linalg.norm(tangent)
            
            # Calculate normal (90 degree rotation)
            normals[i] = np.array([-tangent[1], tangent[0]])
        
        return normals
    
    def calculate_curvature(self):
        """Calculate track curvature at each point."""
        if len(self.points) < 3:
            logger.warning("Not enough points to calculate curvature")
            self.curvature = np.zeros(len(self.points))
            return
        
        n_points = len(self.points)
        self.curvature = np.zeros(n_points)
        
        for i in range(n_points):
            # Get adjacent points (with wraparound for closed circuits)
            prev_idx = (i - 1) % n_points
            next_idx = (i + 1) % n_points
            
            # Get positions
            p_prev = self.points[prev_idx]
            p_curr = self.points[i]
            p_next = self.points[next_idx]
            
            # Calculate vectors
            v1 = p_prev - p_curr
            v2 = p_next - p_curr
            
            # Normalize vectors
            if np.linalg.norm(v1) > 1e-6 and np.linalg.norm(v2) > 1e-6:
                v1 = v1 / np.linalg.norm(v1)
                v2 = v2 / np.linalg.norm(v2)
                
                # Calculate angle between vectors
                dot_product = np.clip(np.dot(v1, v2), -1.0, 1.0)
                angle = np.arccos(dot_product)
                
                # Calculate direction of curve
                cross_product = np.cross(v1, v2)
                sign = 1.0 if cross_product > 0 else -1.0
                
                # Calculate curvature (inverse radius)
                if self.distances is not None and len(self.distances) > 0:
                    # Use distance along track for better estimation
                    l1 = self.distances[i] - self.distances[prev_idx] if i > prev_idx else self.total_length - self.distances[prev_idx] + self.distances[i]
                    l2 = self.distances[next_idx] - self.distances[i] if next_idx > i else self.total_length - self.distances[i] + self.distances[next_idx]
                    
                    # Curvature estimate
                    self.curvature[i] = sign * angle / ((l1 + l2) / 2.0)
                else:
                    # Simplified calculation
                    d1 = np.linalg.norm(p_prev - p_curr)
                    d2 = np.linalg.norm(p_next - p_curr)
                    self.curvature[i] = sign * angle / ((d1 + d2) / 2.0)
            else:
                self.curvature[i] = 0.0
        
        # Apply smoothing to curvature
        self.curvature = self._smooth_array(self.curvature, window_size=5)
        
        logger.info("Curvature calculated for track")
    
    def _smooth_array(self, array: np.ndarray, window_size: int = 3) -> np.ndarray:
        """
        Apply smoothing to an array.
        
        Args:
            array: Array to smooth
            window_size: Size of smoothing window
            
        Returns:
            Smoothed array
        """
        if window_size < 2:
            return array
        
        result = np.copy(array)
        n = len(array)
        half_window = window_size // 2
        
        for i in range(n):
            # Get window indices with wraparound
            window_indices = [(i + j - half_window) % n for j in range(window_size)]
            
            # Calculate mean for window
            result[i] = np.mean(array[window_indices])
        
        return result
    
    def segment_track(self):
        """Divide track into logical segments (straights, corners)."""
        if len(self.points) < 3 or len(self.curvature) != len(self.points):
            logger.warning("Not enough data to segment track")
            return
        
        # Thresholds for segmentation
        straight_threshold = 0.01  # Max curvature for straights
        corner_threshold = 0.05    # Min curvature for corners
        
        # Initialize segments
        self.segments = []
        
        # Identify segments
        current_type = None
        start_idx = 0
        
        for i in range(len(self.curvature)):
            curve = self.curvature[i]
            
            # Determine segment type
            if abs(curve) < straight_threshold:
                segment_type = TrackSegmentType.STRAIGHT
            elif curve > corner_threshold:
                segment_type = TrackSegmentType.CORNER_LEFT
            elif curve < -corner_threshold:
                segment_type = TrackSegmentType.CORNER_RIGHT
            else:
                # Transition zone, continue current segment
                continue
            
            # Check if segment type has changed
            if current_type is not None and segment_type != current_type:
                # End current segment
                radius = None
                if current_type != TrackSegmentType.STRAIGHT:
                    # Calculate average radius for corner
                    avg_curvature = np.mean(np.abs(self.curvature[start_idx:i]))
                    if avg_curvature > 1e-6:
                        radius = 1.0 / avg_curvature
                
                # Create segment
                segment = TrackSegment(current_type, start_idx, i, radius)
                segment.calculate_properties(self.points, self.distances)
                self.segments.append(segment)
                
                # Start new segment
                start_idx = i
            
            current_type = segment_type
        
        # Add final segment
        if current_type is not None:
            radius = None
            if current_type != TrackSegmentType.STRAIGHT:
                # Calculate average radius for corner
                avg_curvature = np.mean(np.abs(self.curvature[start_idx:]))
                if avg_curvature > 1e-6:
                    radius = 1.0 / avg_curvature
            
            # Create segment
            segment = TrackSegment(current_type, start_idx, len(self.curvature) - 1, radius)
            segment.calculate_properties(self.points, self.distances)
            self.segments.append(segment)
        
        # Merge very short segments with adjacent ones
        minimum_length = 5.0  # Minimum segment length in meters
        self._merge_short_segments(minimum_length)
        
        # Log segment info
        segment_counts = {
            'straight': sum(1 for s in self.segments if s.segment_type == TrackSegmentType.STRAIGHT),
            'left_corner': sum(1 for s in self.segments if s.segment_type == TrackSegmentType.CORNER_LEFT),
            'right_corner': sum(1 for s in self.segments if s.segment_type == TrackSegmentType.CORNER_RIGHT)
        }
        
        logger.info(f"Track segmented into {len(self.segments)} segments: " +
                   f"{segment_counts['straight']} straights, " +
                   f"{segment_counts['left_corner']} left corners, " +
                   f"{segment_counts['right_corner']} right corners")
    
    def _merge_short_segments(self, min_length: float):
        """
        Merge very short segments with adjacent ones.
        
        Args:
            min_length: Minimum segment length in meters
        """
        if len(self.segments) <= 1:
            return
        
        i = 0
        while i < len(self.segments) - 1:
            segment = self.segments[i]
            
            if segment.length < min_length:
                # Merge with the next segment
                next_segment = self.segments[i + 1]
                
                # Choose segment type based on length
                if segment.length > next_segment.length:
                    merged_type = segment.segment_type
                else:
                    merged_type = next_segment.segment_type
                
                # Calculate radius for corners
                radius = None
                if merged_type != TrackSegmentType.STRAIGHT:
                    # Use weighted average of radii
                    if segment.radius is not None and next_segment.radius is not None:
                        radius = (segment.radius * segment.length + next_segment.radius * next_segment.length) / (segment.length + next_segment.length)
                    elif segment.radius is not None:
                        radius = segment.radius
                    else:
                        radius = next_segment.radius
                
                # Create merged segment
                merged = TrackSegment(merged_type, segment.start_idx, next_segment.end_idx, radius)
                merged.calculate_properties(self.points, self.distances)
                
                # Replace segments
                self.segments[i] = merged
                del self.segments[i + 1]
            else:
                i += 1
    
    def calculate_racing_line(self, vehicle=None):
        """
        Calculate an optimized racing line.
        
        Args:
            vehicle: Optional vehicle model for constraints
            
        Returns:
            RacingLine object
        """
        # Create racing line object
        self.racing_line = RacingLine(self)
        
        # Optimize racing line
        self.racing_line.optimize(vehicle)
        
        return self.racing_line
    
    def calculate_elevation_profile(self):
        """
        Calculate elevation changes along the track.
        
        Returns:
            Array of elevation values
        """
        # For tracks with 3D points
        if self.points.shape[1] > 2 and len(self.points) > 0:
            self.elevation = self.points[:, 2]
            logger.info("Elevation profile calculated from 3D track data")
            return self.elevation
        
        # For 2D tracks, create a synthetic elevation profile
        if len(self.distances) > 0:
            # Generate a random but smooth elevation profile
            n_points = len(self.points)
            
            # Create a few random control points
            n_control = max(3, n_points // 10)
            control_distances = np.linspace(0, self.total_length, n_control)
            control_elevations = np.random.uniform(-10, 10, n_control)
            control_elevations[0] = 0.0  # Start at zero elevation
            control_elevations[-1] = 0.0  # End at zero elevation
            
            # Create a smooth interpolation
            elevation_interpolator = interp1d(control_distances, control_elevations, kind='cubic')
            
            # Interpolate for all track points
            self.elevation = elevation_interpolator(self.distances)
            
            logger.info("Synthetic elevation profile created for 2D track")
            return self.elevation
        
        logger.warning("Cannot calculate elevation profile without track distances")
        return np.zeros(len(self.points))
    
    def calculate_theoretical_speed_profile(self, vehicle: Any) -> np.ndarray:
        """
        Calculate theoretical speed profile based on vehicle limits.
        
        Args:
            vehicle: Vehicle model object
            
        Returns:
            Array of speeds for each track point
        """
        if len(self.points) < 2 or vehicle is None:
            return np.zeros(len(self.points))
        
        try:
            # Initialize speed profile
            speed_profile = np.zeros(len(self.points))
            
            # First pass: calculate maximum speed based on curvature
            for i, curve in enumerate(self.curvature):
                if abs(curve) > 1e-6:
                    # Calculate corner radius
                    radius = 1.0 / abs(curve)
                    
                    # Calculate maximum cornering speed
                    max_lateral_accel = 20.0  # Default if vehicle doesn't provide calculation
                    
                    if hasattr(vehicle, 'cornering'):
                        max_lateral_accel = vehicle.cornering.calculate_max_lateral_acceleration()
                    
                    # v^2 = a * r for constant radius cornering
                    max_corner_speed = np.sqrt(max_lateral_accel * radius)
                else:
                    # Straight section - use maximum vehicle speed
                    max_speed = 100.0  # Default if vehicle doesn't provide calculation
                    
                    if hasattr(vehicle, 'calculate_max_speed'):
                        max_speed = vehicle.calculate_max_speed()
                    
                    max_corner_speed = max_speed
                
                speed_profile[i] = max_corner_speed
            
            # Second pass: backward pass to ensure speed doesn't exceed braking limits
            max_decel = -20.0  # Default deceleration limit
            
            if hasattr(vehicle, 'calculate_max_deceleration'):
                max_decel = vehicle.calculate_max_deceleration(20.0)  # Typical speed argument
            
            for i in range(len(speed_profile) - 2, -1, -1):
                next_speed = speed_profile[i + 1]
                
                # Calculate distance to next point
                next_idx = (i + 1) % len(self.points)
                segment_length = self.distances[next_idx] - self.distances[i] if next_idx > i else self.total_length - self.distances[i] + self.distances[next_idx]
                
                # Calculate maximum entry speed based on braking distance
                # v^2 = u^2 + 2ad
                max_entry_speed = np.sqrt(next_speed**2 + 2.0 * abs(max_decel) * segment_length)
                
                # Take the minimum of cornering limit and braking limit
                speed_profile[i] = min(speed_profile[i], max_entry_speed)
            
            # Third pass: forward pass to ensure speed doesn't exceed acceleration limits
            max_accel = 10.0  # Default acceleration limit
            
            if hasattr(vehicle, 'calculate_max_acceleration'):
                max_accel = vehicle.calculate_max_acceleration(20.0, 3)  # Typical speed and gear arguments
            
            for i in range(1, len(speed_profile)):
                prev_speed = speed_profile[i - 1]
                
                # Calculate distance from previous point
                prev_idx = (i - 1) % len(self.points)
                segment_length = self.distances[i] - self.distances[prev_idx] if i > prev_idx else self.total_length - self.distances[prev_idx] + self.distances[i]
                
                # Calculate maximum exit speed based on acceleration
                # v^2 = u^2 + 2ad
                max_exit_speed = np.sqrt(prev_speed**2 + 2.0 * max_accel * segment_length)
                
                # Take the minimum of cornering limit and acceleration limit
                speed_profile[i] = min(speed_profile[i], max_exit_speed)
            
            logger.info("Theoretical speed profile calculated based on vehicle limits")
            return speed_profile
            
        except Exception as e:
            logger.error(f"Error calculating theoretical speed profile: {str(e)}")
            return np.zeros(len(self.points))
    
    def visualize(self, show_segments: bool = True, show_racing_line: bool = False):
        """
        Visualize the track.
        
        Args:
            show_segments: Whether to show track segments
            show_racing_line: Whether to show racing line
        """
        if len(self.points) < 2:
            logger.warning("Not enough points to visualize track")
            return
        
        plt.figure(figsize=(10, 8))
        
        # Plot track centerline
        plt.plot(self.points[:, 0], self.points[:, 1], 'k-', alpha=0.7, label='Track Centerline')
        
        # Plot track boundaries if available
        if len(self.left_boundary) > 0 and len(self.right_boundary) > 0:
            plt.plot(self.left_boundary[:, 0], self.left_boundary[:, 1], 'b-', alpha=0.5)
            plt.plot(self.right_boundary[:, 0], self.right_boundary[:, 1], 'b-', alpha=0.5)
        
        # Plot cones if available
        if len(self.cones_left) > 0:
            plt.scatter(self.cones_left[:, 0], self.cones_left[:, 1], color='blue', marker='^', s=20, label='Left Cones')
        
        if len(self.cones_right) > 0:
            plt.scatter(self.cones_right[:, 0], self.cones_right[:, 1], color='yellow', marker='^', s=20, label='Right Cones')
        
        # Plot start position
        if np.all(self.start_position != np.array([0.0, 0.0])):
            plt.scatter(self.start_position[0], self.start_position[1], color='green', marker='o', s=100, label='Start')
            
            # Plot start direction
            direction_vector = np.array([np.cos(self.start_direction), np.sin(self.start_direction)])
            arrow_end = self.start_position + direction_vector * 5.0
            plt.arrow(self.start_position[0], self.start_position[1], 
                     arrow_end[0] - self.start_position[0], arrow_end[1] - self.start_position[1], 
                     head_width=1.0, head_length=1.5, fc='g', ec='g')
        
        # Plot segments if requested
        if show_segments and self.segments:
            segment_colors = {
                TrackSegmentType.STRAIGHT: 'green',
                TrackSegmentType.CORNER_LEFT: 'red',
                TrackSegmentType.CORNER_RIGHT: 'blue',
                TrackSegmentType.CHICANE: 'purple',
                TrackSegmentType.HAIRPIN: 'orange'
            }
            
            for segment in self.segments:
                points = self.points[segment.start_idx:segment.end_idx + 1]
                plt.plot(points[:, 0], points[:, 1], color=segment_colors.get(segment.segment_type, 'gray'), 
                        linewidth=3, alpha=0.5)
        
        # Plot racing line if requested
        if show_racing_line and self.racing_line is not None and self.racing_line.line is not None:
            plt.plot(self.racing_line.line[:, 0], self.racing_line.line[:, 1], 'r-', linewidth=2, label='Racing Line')
        
        # Set equal aspect ratio and grid
        plt.axis('equal')
        plt.grid(True)
        plt.title(f'Track: {self.name} (Length: {self.total_length:.1f}m)')
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    def export_to_format(self, output_format: SimType, output_path: str) -> bool:
        """
        Export track to specific format.
        
        Args:
            output_format: Simulator format to export to
            output_path: Output file path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if output_format == SimType.FSSIM:
                return self._export_to_fssim(output_path)
            elif output_format == SimType.FSDS:
                return self._export_to_fsds(output_path)
            elif output_format == SimType.GPX:
                return self._export_to_gpx(output_path)
            else:
                logger.error(f"Unsupported export format: {output_format}")
                return False
        except Exception as e:
            logger.error(f"Error exporting track to {output_format}: {str(e)}")
            return False
    
    def _export_to_fssim(self, output_path: str) -> bool:
        """
        Export track to FSSIM YAML format.
        
        Args:
            output_path: Output file path
            
        Returns:
            True if successful, False otherwise
        """
        # Determine cones
        if len(self.cones_left) == 0 or len(self.cones_right) == 0:
            # Generate cones from boundaries if available
            if len(self.left_boundary) > 0 and len(self.right_boundary) > 0:
                # Subsample boundaries to create cones
                n_cones = max(20, int(self.total_length / 3.0))
                indices = np.linspace(0, len(self.left_boundary) - 1, n_cones, dtype=int)
                
                cones_left = self.left_boundary[indices].tolist()
                cones_right = self.right_boundary[indices].tolist()
            else:
                logger.error("Cannot export to FSSIM without cones or track boundaries")
                return False
        else:
            cones_left = self.cones_left.tolist()
            cones_right = self.cones_right.tolist()
        
        # Create FSSIM data structure
        fssim_data = {
            'cones_left': cones_left,
            'cones_right': cones_right,
            'cones_orange': [],
            'cones_orange_big': []
        }
        
        # Add start position if available
        if np.all(self.start_position != np.array([0.0, 0.0])):
            fssim_data['starting_pose_cg'] = [
                float(self.start_position[0]),
                float(self.start_position[1]),
                float(self.start_direction)
            ]
            
            # Add orange cones for start/finish
            # Create start line perpendicular to start direction
            perp_direction = np.array([-np.sin(self.start_direction), np.cos(self.start_direction)])
            start_left = self.start_position + perp_direction * 2.0
            start_right = self.start_position - perp_direction * 2.0
            
            fssim_data['cones_orange_big'] = [
                start_left.tolist(),
                start_right.tolist()
            ]
        
        # Write to file
        try:
            with open(output_path, 'w') as f:
                yaml.dump(fssim_data, f, default_flow_style=None)
            
            logger.info(f"Track exported to FSSIM format: {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error writing FSSIM file: {str(e)}")
            return False
    
    def _export_to_fsds(self, output_path: str) -> bool:
        """
        Export track to FSDS CSV format.
        
        Args:
            output_path: Output file path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(output_path, 'w', newline='') as f:
                writer = csv.writer(f)
                
                # Determine cones
                if len(self.cones_left) == 0 or len(self.cones_right) == 0:
                    # Generate cones from boundaries if available
                    if len(self.left_boundary) > 0 and len(self.right_boundary) > 0:
                        # Subsample boundaries to create cones
                        n_cones = max(20, int(self.total_length / 3.0))
                        indices = np.linspace(0, len(self.left_boundary) - 1, n_cones, dtype=int)
                        
                        # Write left cones
                        for idx in indices:
                            x, y = self.left_boundary[idx]
                            writer.writerow(['blue', x, y, 0, 0.01, 0.01, 0])
                        
                        # Write right cones
                        for idx in indices:
                            x, y = self.right_boundary[idx]
                            writer.writerow(['yellow', x, y, 0, 0.01, 0.01, 0])
                    else:
                        logger.error("Cannot export to FSDS without cones or track boundaries")
                        return False
                else:
                    # Write left cones
                    for cone in self.cones_left:
                        writer.writerow(['blue', cone[0], cone[1], 0, 0.01, 0.01, 0])
                    
                    # Write right cones
                    for cone in self.cones_right:
                        writer.writerow(['yellow', cone[0], cone[1], 0, 0.01, 0.01, 0])
                
                # Add start/finish cones if start position is available
                if np.all(self.start_position != np.array([0.0, 0.0])):
                    # Create start line perpendicular to start direction
                    perp_direction = np.array([-np.sin(self.start_direction), np.cos(self.start_direction)])
                    start_left = self.start_position + perp_direction * 2.0
                    start_right = self.start_position - perp_direction * 2.0
                    
                    writer.writerow(['big_orange', start_left[0], start_left[1], 0, 0.01, 0.01, 0])
                    writer.writerow(['big_orange', start_right[0], start_right[1], 0, 0.01, 0.01, 0])
            
            logger.info(f"Track exported to FSDS format: {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error writing FSDS file: {str(e)}")
            return False
    
    def _export_to_gpx(self, output_path: str) -> bool:
        """
        Export track to GPX format.
        
        Args:
            output_path: Output file path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if gpxpy is available
            import gpxpy
            import gpxpy.gpx
        except ImportError:
            logger.error("gpxpy package not available. Install with: pip install gpxpy")
            return False
        
        try:
            # Create GPX object
            gpx = gpxpy.gpx.GPX()
            
            # Create track
            gpx_track = gpxpy.gpx.GPXTrack(name=self.name)
            gpx.tracks.append(gpx_track)
            
            # Create segment
            gpx_segment = gpxpy.gpx.GPXTrackSegment()
            gpx_track.segments.append(gpx_segment)
            
            # Add track points
            # Use a reference location for GPS coords (approx. center of track)
            if len(self.points) > 0:
                center = np.mean(self.points, axis=0)
                lat_ref = 51.5  # Default latitude (London)
                lon_ref = -0.1  # Default longitude (London)
                
                earth_radius = 6378137.0  # Earth radius in meters
                
                for point in self.points:
                    # Convert local coordinates to GPS
                    # This is a simplified conversion that works for small areas
                    lat = lat_ref + (point[1] - center[1]) / earth_radius * (180.0 / np.pi)
                    lon = lon_ref + (point[0] - center[0]) / (earth_radius * np.cos(lat_ref * np.pi / 180.0)) * (180.0 / np.pi)
                    
                    # Add elevation if available
                    ele = None
                    if self.elevation is not None and len(self.elevation) == len(self.points):
                        idx = np.where((self.points == point).all(axis=1))[0][0]
                        ele = self.elevation[idx]
                    
                    gpx_segment.points.append(gpxpy.gpx.GPXTrackPoint(lat, lon, elevation=ele))
            
            # Write GPX file
            with open(output_path, 'w') as f:
                f.write(gpx.to_xml())
            
            logger.info(f"Track exported to GPX format: {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error writing GPX file: {str(e)}")
            return False
    
    def get_track_stats(self) -> Dict:
        """
        Get track statistics (length, corners, etc.).
        
        Returns:
            Dictionary with track statistics
        """
        stats = {
            'name': self.name,
            'length': self.total_length,
            'num_points': len(self.points)
        }
        
        # Count segment types
        if self.segments:
            segment_counts = {
                'straight': sum(1 for s in self.segments if s.segment_type == TrackSegmentType.STRAIGHT),
                'left_corner': sum(1 for s in self.segments if s.segment_type == TrackSegmentType.CORNER_LEFT),
                'right_corner': sum(1 for s in self.segments if s.segment_type == TrackSegmentType.CORNER_RIGHT),
                'chicane': sum(1 for s in self.segments if s.segment_type == TrackSegmentType.CHICANE),
                'hairpin': sum(1 for s in self.segments if s.segment_type == TrackSegmentType.HAIRPIN)
            }
            
            stats['segments'] = segment_counts
            stats['num_segments'] = len(self.segments)
            
            # Calculate total length for each segment type
            segment_lengths = {
                'straight': sum(s.length for s in self.segments if s.segment_type == TrackSegmentType.STRAIGHT),
                'left_corner': sum(s.length for s in self.segments if s.segment_type == TrackSegmentType.CORNER_LEFT),
                'right_corner': sum(s.length for s in self.segments if s.segment_type == TrackSegmentType.CORNER_RIGHT),
                'chicane': sum(s.length for s in self.segments if s.segment_type == TrackSegmentType.CHICANE),
                'hairpin': sum(s.length for s in self.segments if s.segment_type == TrackSegmentType.HAIRPIN)
            }
            
            stats['segment_lengths'] = segment_lengths
            
            # Calculate percentage of track type
            if self.total_length > 0:
                stats['straight_percent'] = segment_lengths['straight'] / self.total_length * 100.0
                stats['corner_percent'] = (segment_lengths['left_corner'] + segment_lengths['right_corner'] +
                                          segment_lengths['chicane'] + segment_lengths['hairpin']) / self.total_length * 100.0
        
        # Add cone counts if available
        if len(self.cones_left) > 0:
            stats['num_left_cones'] = len(self.cones_left)
        
        if len(self.cones_right) > 0:
            stats['num_right_cones'] = len(self.cones_right)
        
        # Add curvature statistics if available
        if len(self.curvature) > 0:
            stats['max_curvature'] = np.max(np.abs(self.curvature))
            if stats['max_curvature'] > 0:
                stats['min_radius'] = 1.0 / stats['max_curvature']
        
        return stats
    
    def __str__(self) -> str:
        """String representation of track."""
        stats = self.get_track_stats()
        
        track_str = f"Track: {stats['name']}\n"
        track_str += f"Length: {stats['length']:.1f}m\n"
        
        if 'num_segments' in stats:
            track_str += f"Segments: {stats['num_segments']} total\n"
            if 'segments' in stats:
                segments = stats['segments']
                track_str += f"  - {segments.get('straight', 0)} straights\n"
                track_str += f"  - {segments.get('left_corner', 0)} left corners\n"
                track_str += f"  - {segments.get('right_corner', 0)} right corners\n"
        
        if 'straight_percent' in stats:
            track_str += f"Composition: {stats['straight_percent']:.1f}% straight, {stats['corner_percent']:.1f}% corners\n"
        
        if 'min_radius' in stats:
            track_str += f"Minimum radius: {stats['min_radius']:.1f}m\n"
        
        return track_str.strip()


class RacingLine:
    """Represents an optimized racing line for a specific track."""
    
    def __init__(self, track: Track):
        """
        Initialize racing line for track.
        
        Args:
            track: Track to optimize racing line for
        """
        self.track = track
        self.line = None  # Array of points representing racing line
        self.track_positions = None  # Array of track positions (-1 to 1)
        self.distances = None  # Cumulative distances along racing line
        self.curvature = None  # Curvature at each point of racing line
        self.speed_profile = None  # Speed profile along racing line
        self.time_profile = None  # Time profile along racing line
        self.total_time = None  # Total lap time
        
        logger.info("Racing line object initialized")
    
    def optimize(self, vehicle=None, method: str = 'geometric') -> bool:
        """
        Optimize racing line using different methods.
        
        Args:
            vehicle: Optional vehicle model for constraints
            method: Optimization method ('geometric', 'minimum_curvature', 'lap_time')
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if method == 'geometric':
                return self._optimize_geometric()
            elif method == 'minimum_curvature':
                return self._optimize_minimum_curvature()
            elif method == 'lap_time':
                if vehicle is None:
                    logger.warning("Vehicle model required for lap time optimization")
                    return self._optimize_geometric()
                return self._optimize_lap_time(vehicle)
            else:
                logger.warning(f"Unknown optimization method: {method}. Using geometric method.")
                return self._optimize_geometric()
        except Exception as e:
            logger.error(f"Error optimizing racing line: {str(e)}")
            return False
    
    def _optimize_geometric(self) -> bool:
        """
        Optimize racing line using simple geometric method.
        
        Returns:
            True if successful, False otherwise
        """
        if len(self.track.points) < 3:
            logger.error("Not enough track points to optimize racing line")
            return False
        
        # Initialize track positions (0 = centerline)
        n_points = len(self.track.points)
        self.track_positions = np.zeros(n_points)
        
        # First pass: move to inside of corners
        for i in range(n_points):
            curve = self.track.curvature[i]
            
            # Move toward inside of corners
            if abs(curve) > 0.01:  # Significant corner
                # Positive curvature = left turn, negative curvature = right turn
                # Position of -1 = right edge, +1 = left edge
                position = -np.sign(curve) * 0.7  # Move 70% toward inside edge
                self.track_positions[i] = position
        
        # Apply smoothing to track positions
        self.track_positions = self._smooth_array(self.track_positions, window_size=max(3, n_points // 20))
        
        # Create racing line points
        self.line = np.zeros_like(self.track.points)
        
        # Calculate normal vectors
        normals = self._calculate_normals()
        
        # Generate racing line
        for i in range(n_points):
            # Calculate offset from centerline
            offset = self.track_positions[i] * self.track.width[i] / 2.0
            
            # Apply offset to centerline
            self.line[i] = self.track.points[i] + normals[i] * offset
        
        # Calculate distances along racing line
        self._calculate_distances()
        
        # Calculate curvature of racing line
        self._calculate_curvature()
        
        logger.info("Racing line optimized using geometric method")
        return True
    
    def _optimize_minimum_curvature(self) -> bool:
        """
        Optimize racing line to minimize maximum curvature.
        
        Returns:
            True if successful, False otherwise
        """
        if len(self.track.points) < 3:
            logger.error("Not enough track points to optimize racing line")
            return False
        
        # Initialize with geometric optimization
        if not self._optimize_geometric():
            return False
        
        try:
            # Set up optimization parameters
            n_points = len(self.track.points)
            n_control = max(10, n_points // 10)  # Number of control points
            
            # Indices of control points
            control_indices = np.linspace(0, n_points - 1, n_control, dtype=int)
            
            # Initial positions from geometric optimization
            initial_positions = self.track_positions[control_indices]
            
            # Define objective function to minimize maximum curvature
            def objective(positions):
                # Interpolate positions for all points
                interpolator = interp1d(control_indices, positions, kind='cubic', fill_value='extrapolate')
                all_positions = interpolator(np.arange(n_points))
                
                # Constrain to valid range
                all_positions = np.clip(all_positions, -0.9, 0.9)
                
                # Calculate racing line points
                line_points = np.zeros_like(self.track.points)
                normals = self._calculate_normals()
                
                for i in range(n_points):
                    offset = all_positions[i] * self.track.width[i] / 2.0
                    line_points[i] = self.track.points[i] + normals[i] * offset
                
                # Calculate curvature
                curvature = np.zeros(n_points)
                
                for i in range(n_points):
                    # Get adjacent points (with wraparound for closed circuits)
                    prev_idx = (i - 1) % n_points
                    next_idx = (i + 1) % n_points
                    
                    # Get positions
                    p_prev = line_points[prev_idx]
                    p_curr = line_points[i]
                    p_next = line_points[next_idx]
                    
                    # Calculate vectors
                    v1 = p_prev - p_curr
                    v2 = p_next - p_curr
                    
                    # Normalize vectors
                    if np.linalg.norm(v1) > 1e-6 and np.linalg.norm(v2) > 1e-6:
                        v1 = v1 / np.linalg.norm(v1)
                        v2 = v2 / np.linalg.norm(v2)
                        
                        # Calculate angle between vectors
                        dot_product = np.clip(np.dot(v1, v2), -1.0, 1.0)
                        angle = np.arccos(dot_product)
                        
                        # Calculate direction of curve
                        cross_product = np.cross(v1, v2)
                        sign = 1.0 if cross_product > 0 else -1.0
                        
                        # Calculate distances between points
                        d1 = np.linalg.norm(p_prev - p_curr)
                        d2 = np.linalg.norm(p_next - p_curr)
                        
                        # Curvature estimate
                        curvature[i] = sign * angle / ((d1 + d2) / 2.0)
                    else:
                        curvature[i] = 0.0
                
                # Return maximum absolute curvature
                return np.max(np.abs(curvature))
            
            # Set up bounds for control positions
            bounds = [(-0.9, 0.9) for _ in range(n_control)]
            
            # Run optimization
            result = minimize(
                objective,
                initial_positions,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 50}
            )
            
            if result.success:
                # Apply optimized control positions
                optimized_positions = result.x
                
                # Interpolate for all points
                interpolator = interp1d(control_indices, optimized_positions, kind='cubic', fill_value='extrapolate')
                self.track_positions = interpolator(np.arange(n_points))
                
                # Constrain to valid range
                self.track_positions = np.clip(self.track_positions, -0.9, 0.9)
                
                # Regenerate racing line points
                normals = self._calculate_normals()
                
                for i in range(n_points):
                    offset = self.track_positions[i] * self.track.width[i] / 2.0
                    self.line[i] = self.track.points[i] + normals[i] * offset
                
                # Recalculate distances and curvature
                self._calculate_distances()
                self._calculate_curvature()
                
                logger.info(f"Racing line optimized to minimize curvature: max curvature = {result.fun:.6f}")
                return True
            else:
                logger.warning(f"Curvature optimization did not converge: {result.message}")
                return True  # Still return True as we have the initial geometric line
            
        except Exception as e:
            logger.error(f"Error in minimum curvature optimization: {str(e)}")
            return True  # Still return True as we have the initial geometric line
    
    def _optimize_lap_time(self, vehicle) -> bool:
        """
        Optimize racing line for minimum lap time.
        
        Args:
            vehicle: Vehicle model for performance calculations
            
        Returns:
            True if successful, False otherwise
        """
        # Start with minimum curvature optimization
        if not self._optimize_minimum_curvature():
            return False
        
        # Calculate speed profile with current racing line
        self.calculate_speed_profile(vehicle)
        
        # Further optimization could be implemented here
        # This would require more sophisticated vehicle dynamics
        
        logger.info("Racing line optimized for lap time")
        return True
    
    def _calculate_normals(self) -> np.ndarray:
        """
        Calculate normal vectors for track points.
        
        Returns:
            Array of normal vectors
        """
        n_points = len(self.track.points)
        normals = np.zeros_like(self.track.points)
        
        for i in range(n_points):
            # Get adjacent points (with wraparound for closed circuits)
            prev_idx = (i - 1) % n_points
            next_idx = (i + 1) % n_points
            
            # Calculate tangent vector
            tangent = self.track.points[next_idx] - self.track.points[prev_idx]
            
            # Normalize
            if np.linalg.norm(tangent) > 1e-6:
                tangent = tangent / np.linalg.norm(tangent)
            
            # Calculate normal (90 degree rotation)
            normals[i] = np.array([-tangent[1], tangent[0]])
        
        return normals
    
    def _calculate_distances(self):
        """Calculate cumulative distances along the racing line."""
        if self.line is None or len(self.line) < 2:
            logger.warning("Not enough points to calculate distances")
            self.distances = np.array([0.0])
            return
        
        # Calculate segment lengths
        segments = np.diff(self.line, axis=0)
        segment_lengths = np.sqrt(np.sum(segments**2, axis=1))
        
        # Cumulative distances
        self.distances = np.zeros(len(self.line))
        self.distances[1:] = np.cumsum(segment_lengths)
    
    def _calculate_curvature(self):
        """Calculate curvature along the racing line."""
        if self.line is None or len(self.line) < 3:
            logger.warning("Not enough points to calculate curvature")
            self.curvature = np.zeros(len(self.line)) if self.line is not None else np.array([])
            return
        
        n_points = len(self.line)
        self.curvature = np.zeros(n_points)
        
        for i in range(n_points):
            # Get adjacent points (with wraparound for closed circuits)
            prev_idx = (i - 1) % n_points
            next_idx = (i + 1) % n_points
            
            # Get positions
            p_prev = self.line[prev_idx]
            p_curr = self.line[i]
            p_next = self.line[next_idx]
            
            # Calculate vectors
            v1 = p_prev - p_curr
            v2 = p_next - p_curr
            
            # Normalize vectors
            if np.linalg.norm(v1) > 1e-6 and np.linalg.norm(v2) > 1e-6:
                v1 = v1 / np.linalg.norm(v1)
                v2 = v2 / np.linalg.norm(v2)
                
                # Calculate angle between vectors
                dot_product = np.clip(np.dot(v1, v2), -1.0, 1.0)
                angle = np.arccos(dot_product)
                
                # Calculate direction of curve
                cross_product = np.cross(v1, v2)
                sign = 1.0 if cross_product > 0 else -1.0
                
                # Calculate distances between points
                d1 = np.linalg.norm(p_prev - p_curr)
                d2 = np.linalg.norm(p_next - p_curr)
                
                # Curvature estimate
                self.curvature[i] = sign * angle / ((d1 + d2) / 2.0)
            else:
                self.curvature[i] = 0.0
        
        # Apply smoothing to curvature
        self.curvature = self._smooth_array(self.curvature, window_size=5)
    
    def _smooth_array(self, array: np.ndarray, window_size: int = 3) -> np.ndarray:
        """
        Apply smoothing to an array.
        
        Args:
            array: Array to smooth
            window_size: Size of smoothing window
            
        Returns:
            Smoothed array
        """
        if window_size < 2:
            return array
        
        result = np.copy(array)
        n = len(array)
        half_window = window_size // 2
        
        for i in range(n):
            # Get window indices with wraparound
            window_indices = [(i + j - half_window) % n for j in range(window_size)]
            
            # Calculate mean for window
            result[i] = np.mean(array[window_indices])
        
        return result
    
    def calculate_speed_profile(self, vehicle) -> np.ndarray:
        """
        Calculate achievable speed profile along the line.
        
        Args:
            vehicle: Vehicle model for performance constraints
            
        Returns:
            Array of speeds at each point
        """
        if self.line is None or len(self.line) < 2:
            logger.warning("No racing line to calculate speed profile")
            return np.array([])
        
        try:
            # Initialize speed profile
            n_points = len(self.line)
            self.speed_profile = np.zeros(n_points)
            
            # First pass: calculate maximum speed based on curvature
            for i, curve in enumerate(self.curvature):
                if abs(curve) > 1e-6:
                    # Calculate corner radius
                    radius = 1.0 / abs(curve)
                    
                    # Calculate maximum cornering speed
                    max_lateral_accel = 20.0  # Default if vehicle doesn't provide calculation
                    
                    if hasattr(vehicle, 'cornering'):
                        max_lateral_accel = vehicle.cornering.calculate_max_lateral_acceleration()
                    
                    # v^2 = a * r for constant radius cornering
                    max_corner_speed = np.sqrt(max_lateral_accel * radius)
                else:
                    # Straight section - use maximum vehicle speed
                    max_speed = 100.0  # Default if vehicle doesn't provide calculation
                    
                    if hasattr(vehicle, 'calculate_max_speed'):
                        max_speed = vehicle.calculate_max_speed()
                    
                    max_corner_speed = max_speed
                
                self.speed_profile[i] = max_corner_speed
            
            # Second pass: backward pass to ensure speed doesn't exceed braking limits
            max_decel = -20.0  # Default deceleration limit
            
            if hasattr(vehicle, 'calculate_max_deceleration'):
                max_decel = vehicle.calculate_max_deceleration(20.0)  # Typical speed argument
            
            for i in range(n_points - 2, -1, -1):
                next_speed = self.speed_profile[(i + 1) % n_points]
                
                # Calculate distance to next point
                next_idx = (i + 1) % n_points
                segment_length = self.distances[next_idx] - self.distances[i] if next_idx > i else self.distances[-1] - self.distances[i] + self.distances[0]
                
                # Calculate maximum entry speed based on braking distance
                # v^2 = u^2 + 2ad
                max_entry_speed = np.sqrt(next_speed**2 + 2.0 * abs(max_decel) * segment_length)
                
                # Take the minimum of cornering limit and braking limit
                self.speed_profile[i] = min(self.speed_profile[i], max_entry_speed)
            
            # Third pass: forward pass to ensure speed doesn't exceed acceleration limits
            max_accel = 10.0  # Default acceleration limit
            
            if hasattr(vehicle, 'calculate_max_acceleration'):
                max_accel = vehicle.calculate_max_acceleration(20.0, 3)  # Typical speed and gear arguments
            
            for i in range(1, n_points):
                prev_speed = self.speed_profile[i - 1]
                
                # Calculate distance from previous point
                prev_idx = (i - 1) % n_points
                segment_length = self.distances[i] - self.distances[prev_idx] if i > prev_idx else self.distances[i] + self.distances[-1] - self.distances[prev_idx]
                
                # Calculate maximum exit speed based on acceleration
                # v^2 = u^2 + 2ad
                max_exit_speed = np.sqrt(prev_speed**2 + 2.0 * max_accel * segment_length)
                
                # Take the minimum of cornering limit and acceleration limit
                self.speed_profile[i] = min(self.speed_profile[i], max_exit_speed)
            
            # Calculate time profile
            self._calculate_time_profile()
            
            logger.info(f"Speed profile calculated: Max speed = {np.max(self.speed_profile):.1f} m/s, " +
                       f"Lap time = {self.total_time:.2f} s")
            
            return self.speed_profile
            
        except Exception as e:
            logger.error(f"Error calculating speed profile: {str(e)}")
            return np.zeros(len(self.line)) if self.line is not None else np.array([])
    
    def _calculate_time_profile(self):
        """Calculate time profile based on speed profile."""
        if self.speed_profile is None or len(self.speed_profile) < 2 or self.distances is None:
            logger.warning("Cannot calculate time profile without speed profile and distances")
            return
        
        n_points = len(self.speed_profile)
        self.time_profile = np.zeros(n_points)
        
        # Calculate time to travel each segment
        for i in range(1, n_points):
            # Distance of segment
            prev_idx = i - 1
            segment_length = self.distances[i] - self.distances[prev_idx]
            
            # Average speed for segment
            avg_speed = (self.speed_profile[prev_idx] + self.speed_profile[i]) / 2.0
            
            # Skip if speed is too low to avoid division by zero
            if avg_speed < 0.1:
                avg_speed = 0.1
            
            # Time to travel segment
            segment_time = segment_length / avg_speed
            
            # Accumulate time
            self.time_profile[i] = self.time_profile[prev_idx] + segment_time
        
        # Total lap time
        self.total_time = self.time_profile[-1]
    
    def visualize(self, with_speed: bool = False):
        """
        Visualize the racing line.
        
        Args:
            with_speed: Whether to color code by speed
        """
        if self.line is None or len(self.line) < 2:
            logger.warning("No racing line to visualize")
            return
        
        plt.figure(figsize=(10, 8))
        
        # Plot track centerline
        plt.plot(self.track.points[:, 0], self.track.points[:, 1], 'k-', alpha=0.3, label='Track Centerline')
        
        # Plot track boundaries if available
        if len(self.track.left_boundary) > 0 and len(self.track.right_boundary) > 0:
            plt.plot(self.track.left_boundary[:, 0], self.track.left_boundary[:, 1], 'k-', alpha=0.2)
            plt.plot(self.track.right_boundary[:, 0], self.track.right_boundary[:, 1], 'k-', alpha=0.2)
        
        # Plot racing line
        if with_speed and self.speed_profile is not None:
            # Color code by speed
            speed_norm = plt.Normalize(vmin=np.min(self.speed_profile), vmax=np.max(self.speed_profile))
            cmap = plt.cm.jet
            
            for i in range(len(self.line) - 1):
                plt.plot(self.line[i:i+2, 0], self.line[i:i+2, 1], '-', 
                        color=cmap(speed_norm(self.speed_profile[i])), linewidth=2)
            
            # Add colorbar
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=speed_norm)
            sm.set_array([])
            cbar = plt.colorbar(sm)
            cbar.set_label('Speed (m/s)')
        else:
            plt.plot(self.line[:, 0], self.line[:, 1], 'r-', linewidth=2, label='Racing Line')
        
        # Plot direction arrows on racing line
        arrow_indices = np.linspace(0, len(self.line) - 1, 20, dtype=int)
        for i in arrow_indices:
            next_idx = (i + 1) % len(self.line)
            dx = self.line[next_idx, 0] - self.line[i, 0]
            dy = self.line[next_idx, 1] - self.line[i, 1]
            
            norm = np.sqrt(dx**2 + dy**2)
            if norm > 1e-6:
                dx /= norm
                dy /= norm
            
            arrow_length = 2.0
            plt.arrow(self.line[i, 0], self.line[i, 1], dx * arrow_length, dy * arrow_length, 
                     head_width=0.8, head_length=1.2, fc='r', ec='r', alpha=0.7)
        
        # Set equal aspect ratio and grid
        plt.axis('equal')
        plt.grid(True)
        title = f'Racing Line: {self.track.name}'
        if self.total_time is not None:
            title += f' (Lap time: {self.total_time:.2f}s)'
        plt.title(title)
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    def get_stats(self) -> Dict:
        """
        Get racing line statistics.
        
        Returns:
            Dictionary with racing line statistics
        """
        stats = {}
        
        if self.line is not None:
            stats['num_points'] = len(self.line)
        
        if self.distances is not None and len(self.distances) > 0:
            stats['length'] = self.distances[-1]
        
        if self.curvature is not None and len(self.curvature) > 0:
            stats['max_curvature'] = np.max(np.abs(self.curvature))
            if stats['max_curvature'] > 0:
                stats['min_radius'] = 1.0 / stats['max_curvature']
        
        if self.speed_profile is not None and len(self.speed_profile) > 0:
            stats['max_speed'] = np.max(self.speed_profile)
            stats['min_speed'] = np.min(self.speed_profile)
            stats['avg_speed'] = np.mean(self.speed_profile)
        
        if self.total_time is not None:
            stats['lap_time'] = self.total_time
        
        return stats


def create_example_track() -> Track:
    """
    Create an example track for testing.
    
    Returns:
        Track object
    """
    track = Track("Example Track")
    
    # Create oval track
    t = np.linspace(0, 2*np.pi, 100)
    a = 50.0  # semi-major axis
    b = 30.0  # semi-minor axis
    
    # Generate points
    x = a * np.cos(t)
    y = b * np.sin(t)
    
    # Store track points
    track.points = np.column_stack((x, y))
    
    # Set constant width
    track.width = np.full(len(track.points), 10.0)
    
    # Set start position
    track.start_position = np.array([0.0, -b])
    track.start_direction = 0.0
    
    # Perform post-load processing
    track._post_load_processing()
    
    logger.info("Example track created")
    return track


def load_track_from_config(config_path: str) -> Optional[Track]:
    """
    Load track from configuration file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Track object, or None if loading failed
    """
    try:
        track = Track()
        if track.load_from_file(config_path):
            return track
        return None
    except Exception as e:
        logger.error(f"Error loading track from config {config_path}: {str(e)}")
        return None