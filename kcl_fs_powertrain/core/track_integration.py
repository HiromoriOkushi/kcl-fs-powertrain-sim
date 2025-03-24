"""Integration between track generator and powertrain simulation."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
from typing import Dict, List, Tuple, Optional
from ..track_generator.enums import SimType
from ..track_generator.generator import FSTrackGenerator
from ..utils.track_utils import preprocess_track_points

class TrackProfile:
    """Track profile for powertrain simulation."""
    
    def __init__(self, track_file: str):
        """
        Initialize track profile from file.
        
        Args:
            track_file: Path to track file (CSV or YAML)
        """
        self.track_file = track_file
        self.track_data = None
        self.track_length = 0.0
        self.sections = []
        self.load_track()
    
    def load_track(self):
        """Load track data from file."""
        _, ext = os.path.splitext(self.track_file)
        
        if ext.lower() == '.csv':
            self._load_fsds_format()
        elif ext.lower() == '.yaml':
            self._load_fssim_format()
        else:
            raise ValueError(f"Unsupported track file format: {ext}")
    
    def _load_fsds_format(self):
        """Load track from FSDS CSV format."""
        try:
            # FSDS format typically has columns for:
            # x, y, z, width, direction_x, direction_y 
            df = pd.read_csv(self.track_file)
            
            # Check required columns
            required_columns = ['x', 'y', 'width']
            if not all(col in df.columns for col in required_columns):
                missing = [col for col in required_columns if col not in df.columns]
                raise ValueError(f"FSDS CSV missing required columns: {missing}")
            
            # Extract track points
            track_points = df[['x', 'y']].values
            
            # Calculate track width at each point
            track_width = df['width'].values if 'width' in df.columns else np.full(len(track_points), 3.0)
            
            # Extract z-coordinates if available
            track_elevation = df['z'].values if 'z' in df.columns else np.zeros(len(track_points))
            
            # Store track data
            self.track_data = {
                'points': track_points,
                'width': track_width,
                'elevation': track_elevation
            }
            
            # Calculate track length
            self._calculate_track_length()
            
            # Identify track sections (straights, corners)
            self._identify_sections()
            
            print(f"Loaded FSDS track with {len(track_points)} points, length: {self.track_length:.1f} m")
            
        except Exception as e:
            raise ValueError(f"Error loading FSDS track file: {str(e)}")
    
    def _load_fssim_format(self):
        """Load track from FSSIM YAML format."""
        try:
            # FSSIM format uses YAML with an array of track point dictionaries
            with open(self.track_file, 'r') as f:
                track_data = yaml.safe_load(f)
            
            # FSSIM format should have 'track' key with list of points
            if 'track' not in track_data:
                raise ValueError("FSSIM YAML missing 'track' key")
            
            track_points = []
            track_width = []
            track_elevation = []
            
            # Extract track points
            for point in track_data['track']:
                if 'x' not in point or 'y' not in point:
                    raise ValueError("Track point missing 'x' or 'y' coordinate")
                
                track_points.append([point['x'], point['y']])
                
                # Extract width if available
                width = point.get('width', 3.0)  # Default width
                track_width.append(width)
                
                # Extract elevation if available
                elevation = point.get('z', 0.0)  # Default elevation
                track_elevation.append(elevation)
            
            # Convert to numpy arrays
            track_points = np.array(track_points)
            track_width = np.array(track_width)
            track_elevation = np.array(track_elevation)
            
            # Store track data
            self.track_data = {
                'points': track_points,
                'width': track_width,
                'elevation': track_elevation
            }
            
            # Extract additional metadata if available
            if 'metadata' in track_data:
                self.track_data['metadata'] = track_data['metadata']
            
            # Calculate track length
            self._calculate_track_length()
            
            # Identify track sections
            self._identify_sections()
            
            print(f"Loaded FSSIM track with {len(track_points)} points, length: {self.track_length:.1f} m")
            
        except Exception as e:
            raise ValueError(f"Error loading FSSIM track file: {str(e)}")
    
    def _calculate_track_length(self):
        """Calculate total track length and segment lengths."""
        if self.track_data is None or 'points' not in self.track_data:
            return
        
        points = self.track_data['points']
        
        # Calculate distances between consecutive points
        diffs = np.diff(points, axis=0)
        segment_lengths = np.sqrt(np.sum(diffs**2, axis=1))
        
        # Store segment lengths
        self.track_data['segment_lengths'] = segment_lengths
        
        # Calculate cumulative distance along track
        cum_distance = np.zeros(len(points))
        cum_distance[1:] = np.cumsum(segment_lengths)
        self.track_data['distance'] = cum_distance
        
        # Store total track length
        self.track_length = cum_distance[-1]
    
    def _identify_sections(self):
        """Identify track sections (straights, corners) and calculate curvature."""
        if self.track_data is None or 'points' not in self.track_data:
            return
        
        points = self.track_data['points']
        
        # Calculate track direction vectors
        diffs = np.diff(points, axis=0)
        directions = np.zeros_like(points)
        directions[:-1] = diffs
        directions[-1] = directions[-2]  # Use previous direction for last point
        
        # Normalize direction vectors
        norms = np.sqrt(np.sum(directions**2, axis=1)).reshape(-1, 1)
        norms[norms == 0] = 1  # Prevent division by zero
        directions = directions / norms
        
        # Calculate curvature using dot product of adjacent direction vectors
        curvature = np.zeros(len(points))
        segment_lengths = self.track_data['segment_lengths']
        
        for i in range(1, len(points) - 1):
            # Vector for previous segment
            prev_dir = directions[i-1]
            
            # Vector for next segment
            next_dir = directions[i]
            
            # Dot product gives cosine of angle
            cos_angle = np.dot(prev_dir, next_dir)
            cos_angle = max(-1.0, min(1.0, cos_angle))  # Ensure within valid range
            
            # Angle between segments
            angle = np.arccos(cos_angle)
            
            # Determine sign of curvature (left/right turn)
            # Cross product z-component sign tells us direction
            cross_z = prev_dir[0] * next_dir[1] - prev_dir[1] * next_dir[0]
            sign = 1 if cross_z >= 0 else -1
            
            # Curvature is angle divided by arc length
            # Use average of adjacent segment lengths
            avg_length = (segment_lengths[i-1] + segment_lengths[i-1]) / 2
            if avg_length > 0:
                curvature[i] = sign * angle / avg_length
        
        # Start and end points use nearest calculated curvature
        curvature[0] = curvature[1]
        curvature[-1] = curvature[-2]
        
        # Store curvature
        self.track_data['curvature'] = curvature
        self.track_data['directions'] = directions
        
        # Identify track sections
        self._segment_track_sections(curvature)
    
    def _segment_track_sections(self, curvature: np.ndarray):
        """
        Segment track into sections based on curvature.
        
        Args:
            curvature: Array of curvature values at each track point
        """
        # Parameters for section identification
        curvature_threshold = 0.01  # Threshold for distinguishing corners from straights
        min_section_length = 5.0   # Minimum section length in meters
        
        # Initialize sections
        sections = []
        current_section = {
            'type': 'straight',
            'start_idx': 0,
            'curvature': 0.0
        }
        
        # Get distances along track
        distances = self.track_data['distance']
        
        # Iterate through track points
        for i in range(1, len(curvature)):
            if abs(curvature[i]) > curvature_threshold:
                # This is a corner
                section_type = 'left_turn' if curvature[i] > 0 else 'right_turn'
                
                if current_section['type'] != section_type:
                    # End previous section
                    current_section['end_idx'] = i - 1
                    current_section['length'] = distances[i-1] - distances[current_section['start_idx']]
                    
                    # Only add if above minimum length
                    if current_section['length'] >= min_section_length:
                        sections.append(current_section)
                    
                    # Start new section
                    current_section = {
                        'type': section_type,
                        'start_idx': i,
                        'curvature': curvature[i]
                    }
                else:
                    # Update average curvature for existing corner
                    n = i - current_section['start_idx']
                    avg_curvature = (current_section['curvature'] * n + curvature[i]) / (n + 1)
                    current_section['curvature'] = avg_curvature
                
            else:
                # This is a straight
                if current_section['type'] != 'straight':
                    # End previous section
                    current_section['end_idx'] = i - 1
                    current_section['length'] = distances[i-1] - distances[current_section['start_idx']]
                    
                    # Only add if above minimum length
                    if current_section['length'] >= min_section_length:
                        sections.append(current_section)
                    
                    # Start new straight section
                    current_section = {
                        'type': 'straight',
                        'start_idx': i,
                        'curvature': 0.0
                    }
        
        # Add final section
        current_section['end_idx'] = len(curvature) - 1
        current_section['length'] = distances[-1] - distances[current_section['start_idx']]
        if current_section['length'] >= min_section_length:
            sections.append(current_section)
        
        # Store sections
        self.sections = sections
        
        # Print sections summary
        n_straights = sum(1 for s in sections if s['type'] == 'straight')
        n_left = sum(1 for s in sections if s['type'] == 'left_turn')
        n_right = sum(1 for s in sections if s['type'] == 'right_turn')
        
        print(f"Track sections: {len(sections)} total "
              f"({n_straights} straights, {n_left} left turns, {n_right} right turns)")
    
    def calculate_speed_profile(self, vehicle_params: Dict) -> np.ndarray:
        """
        Calculate speed profile along track based on vehicle parameters.
        
        Args:
            vehicle_params: Dictionary of vehicle parameters
                Required keys:
                - 'mass': Vehicle mass in kg
                - 'power': Maximum power in kW
                - 'max_lat_accel': Maximum lateral acceleration in m/s²
                - 'max_long_accel': Maximum longitudinal acceleration in m/s²
                - 'max_long_decel': Maximum longitudinal deceleration in m/s²
                - 'drag_coefficient': Aerodynamic drag coefficient
                - 'frontal_area': Frontal area in m²
                
        Returns:
            Array of speeds at each track point
        """
        if self.track_data is None:
            raise ValueError("Track data not loaded")
        
        # Ensure all required parameters are provided
        required_params = ['mass', 'power', 'max_lat_accel', 'max_long_accel', 
                           'max_long_decel', 'drag_coefficient', 'frontal_area']
        
        for param in required_params:
            if param not in vehicle_params:
                raise ValueError(f"Missing required vehicle parameter: {param}")
        
        # Extract track data
        points = self.track_data['points']
        curvature = self.track_data['curvature']
        distances = self.track_data['distance']
        
        # Initialize speed profile
        n_points = len(points)
        speed_profile = np.zeros(n_points)
        
        # Extract vehicle parameters
        mass = vehicle_params['mass']  # kg
        power = vehicle_params['power'] * 1000  # Convert kW to W
        max_lat_accel = vehicle_params['max_lat_accel']  # m/s²
        max_accel = vehicle_params['max_long_accel']  # m/s²
        max_decel = vehicle_params['max_long_decel']  # m/s²
        drag_coef = vehicle_params['drag_coefficient']
        frontal_area = vehicle_params['frontal_area']  # m²
        
        # Air density for drag calculations
        air_density = 1.225  # kg/m³
        
        # First pass: Calculate maximum speed based on curvature
        for i in range(n_points):
            # Maximum speed based on lateral acceleration (v² = a_lat * r = a_lat / curvature)
            if abs(curvature[i]) > 1e-6:  # Avoid division by zero
                max_corner_speed = np.sqrt(max_lat_accel / abs(curvature[i]))
            else:
                max_corner_speed = float('inf')  # No curvature limitation
            
            # Maximum speed based on powertrain (simplified)
            # Solve: P = F * v = (0.5 * rho * Cd * A * v²) * v
            # v³ = 2 * P / (rho * Cd * A)
            drag_factor = 0.5 * air_density * drag_coef * frontal_area
            if drag_factor > 0:
                max_power_speed = (2 * power / drag_factor) ** (1/3)
            else:
                max_power_speed = float('inf')
            
            # Take minimum of cornering and power limits
            speed_profile[i] = min(max_corner_speed, max_power_speed)
        
        # Apply maximum speed limit if provided
        if 'max_speed' in vehicle_params:
            speed_profile = np.minimum(speed_profile, vehicle_params['max_speed'])
        
        # Second pass: Forward simulation considering acceleration limits
        speed = 0.0  # Start from standstill
        for i in range(n_points - 1):
            # Current speed after previous point
            speed_profile[i] = speed
            
            # Maximum speed at next point based on curvature
            next_speed_limit = speed_profile[i + 1]
            
            # Distance to next point
            distance = distances[i + 1] - distances[i]
            
            # Maximum achievable speed considering acceleration
            # v² = u² + 2*a*s
            max_accel_speed = np.sqrt(speed**2 + 2 * max_accel * distance)
            
            # Minimum of acceleration limit and cornering limit
            speed = min(max_accel_speed, next_speed_limit)
        
        # Last point
        speed_profile[-1] = speed
        
        # Third pass: Backward simulation considering deceleration limits
        for i in range(n_points - 2, -1, -1):
            # Maximum speed at current point based on deceleration to next point
            next_speed = speed_profile[i + 1]
            distance = distances[i + 1] - distances[i]
            
            # Maximum allowable speed considering deceleration to next point
            # v² = u² - 2*a*s
            max_decel_speed = np.sqrt(next_speed**2 + 2 * max_decel * distance)
            
            # Take minimum of forward and backward simulations
            speed_profile[i] = min(speed_profile[i], max_decel_speed)
        
        return speed_profile
    
    def visualize(self, speed_profile: Optional[np.ndarray] = None, save_path: Optional[str] = None):
        """
        Visualize the track layout and optionally the speed profile.
        
        Args:
            speed_profile: Optional array of speeds at each track point
            save_path: Optional path to save the visualization
        """
        if self.track_data is None:
            raise ValueError("Track data not loaded")
        
        # Extract track data
        points = self.track_data['points']
        curvature = self.track_data.get('curvature')
        width = self.track_data.get('width')
        
        # Create figure
        fig = plt.figure(figsize=(12, 10))
        
        # Create layout based on what we want to plot
        if speed_profile is not None:
            gs = plt.GridSpec(2, 2, height_ratios=[3, 1])
            ax_track = plt.subplot(gs[0, :])  # Top row full width
            ax_speed = plt.subplot(gs[1, 0])  # Bottom left
            ax_curv = plt.subplot(gs[1, 1])  # Bottom right
        else:
            gs = plt.GridSpec(2, 1, height_ratios=[3, 1])
            ax_track = plt.subplot(gs[0])  # Top row
            ax_curv = plt.subplot(gs[1])  # Bottom row
            ax_speed = None
        
        # Plot track layout
        ax_track.plot(points[:, 0], points[:, 1], 'k-', linewidth=2)
        
        # If track width data is available, plot track boundaries
        if width is not None:
            # Calculate normal vectors to track direction
            directions = self.track_data['directions']
            normals = np.zeros_like(directions)
            normals[:, 0] = -directions[:, 1]  # Normal = (-dy, dx)
            normals[:, 1] = directions[:, 0]
            
            # Calculate inner and outer track boundaries
            inner_boundary = points - (width.reshape(-1, 1) / 2) * normals
            outer_boundary = points + (width.reshape(-1, 1) / 2) * normals
            
            # Plot boundaries
            ax_track.plot(inner_boundary[:, 0], inner_boundary[:, 1], 'b--', linewidth=1)
            ax_track.plot(outer_boundary[:, 0], outer_boundary[:, 1], 'r--', linewidth=1)
        
        # Plot start/finish line
        ax_track.plot(points[0, 0], points[0, 1], 'go', markersize=10, label='Start/Finish')
        
        # Plot section types with different colors
        for section in self.sections:
            start_idx = section['start_idx']
            end_idx = section['end_idx']
            section_points = points[start_idx:end_idx+1]
            
            if section['type'] == 'straight':
                color = 'green'
                label = 'Straight' if start_idx == 0 else None  # Only label once
            elif section['type'] == 'left_turn':
                color = 'blue'
                label = 'Left Turn' if label != 'Left Turn' else None
            else:  # right_turn
                color = 'red'
                label = 'Right Turn' if label != 'Right Turn' else None
                
            ax_track.plot(section_points[:, 0], section_points[:, 1], color=color, linewidth=4, alpha=0.5, label=label)
        
        # Add legend
        ax_track.legend()
        
        # Equal aspect ratio
        ax_track.set_aspect('equal')
        ax_track.grid(True)
        ax_track.set_title('Track Layout')
        ax_track.set_xlabel('X (m)')
        ax_track.set_ylabel('Y (m)')
        
        # Plot curvature
        if curvature is not None:
            distances = self.track_data['distance']
            ax_curv.plot(distances, curvature)
            ax_curv.grid(True)
            ax_curv.set_title('Track Curvature')
            ax_curv.set_xlabel('Distance (m)')
            ax_curv.set_ylabel('Curvature (1/m)')
            
            # Add horizontal line at y=0
            ax_curv.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        
        # Plot speed profile if provided
        if speed_profile is not None and ax_speed is not None:
            distances = self.track_data['distance']
            ax_speed.plot(distances, speed_profile * 3.6)  # Convert m/s to km/h
            ax_speed.grid(True)
            ax_speed.set_title('Speed Profile')
            ax_speed.set_xlabel('Distance (m)')
            ax_speed.set_ylabel('Speed (km/h)')
        
        plt.tight_layout()
        
        # Save figure if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Track visualization saved to {save_path}")
        
        plt.show()
    
    def export_to_simulator(self, output_file: str, sim_type: SimType = SimType.FSSIM):
        """
        Export track data to simulator format.
        
        Args:
            output_file: Path to output file
            sim_type: Simulator type (FSSIM, FSDS, GPX)
        """
        if self.track_data is None:
            raise ValueError("Track data not loaded")
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Export based on simulator type
        if sim_type == SimType.FSSIM:
            self._export_to_fssim(output_file)
        elif sim_type == SimType.FSDS:
            self._export_to_fsds(output_file)
        elif sim_type == SimType.GPX:
            self._export_to_gpx(output_file)
        else:
            raise ValueError(f"Unsupported simulator type: {sim_type}")
    
    def _export_to_fssim(self, output_file: str):
        """
        Export track data to FSSIM YAML format.
        
        Args:
            output_file: Path to output YAML file
        """
        points = self.track_data['points']
        width = self.track_data.get('width')
        
        # Create track data structure
        track_data = {
            'track': []
        }
        
        # Add each track point
        for i in range(len(points)):
            point_data = {
                'x': float(points[i, 0]),
                'y': float(points[i, 1])
            }
            
            # Add width if available
            if width is not None:
                point_data['width'] = float(width[i])
            
            track_data['track'].append(point_data)
        
        # Add metadata
        track_data['metadata'] = {
            'length': self.track_length,
            'num_points': len(points)
        }
        
        # Save to YAML file
        with open(output_file, 'w') as f:
            yaml.dump(track_data, f, default_flow_style=False)
        
        print(f"Track exported to FSSIM format: {output_file}")
    
    def _export_to_fsds(self, output_file: str):
        """
        Export track data to FSDS CSV format.
        
        Args:
            output_file: Path to output CSV file
        """
        points = self.track_data['points']
        width = self.track_data.get('width')
        directions = self.track_data.get('directions')
        
        # Create DataFrame
        data = {
            'x': points[:, 0],
            'y': points[:, 1],
            'z': self.track_data.get('elevation', np.zeros(len(points)))
        }
        
        # Add width if available
        if width is not None:
            data['width'] = width
        
        # Add directions if available
        if directions is not None:
            data['direction_x'] = directions[:, 0]
            data['direction_y'] = directions[:, 1]
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Save to CSV
        df.to_csv(output_file, index=False)
        
        print(f"Track exported to FSDS format: {output_file}")
    
    def _export_to_gpx(self, output_file: str):
        """
        Export track data to GPX format.
        
        Args:
            output_file: Path to output GPX file
        """
        points = self.track_data['points']
        elevation = self.track_data.get('elevation', np.zeros(len(points)))
        
        # Create GPX header
        gpx_content = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<gpx version="1.1" creator="KCL Formula Student Powertrain Sim">',
            '<trk>',
            '<name>Formula Student Track</name>',
            '<trkseg>'
        ]
        
        # Add track points
        for i in range(len(points)):
            x, y = points[i]
            z = elevation[i]
            gpx_content.append(f'<trkpt lat="{y}" lon="{x}"><ele>{z}</ele></trkpt>')
        
        # Add GPX footer
        gpx_content.extend(['</trkseg>', '</trk>', '</gpx>'])
        
        # Save to file
        with open(output_file, 'w') as f:
            f.write('\n'.join(gpx_content))
        
        print(f"Track exported to GPX format: {output_file}")

    def get_track_data(self):
        """
        Get processed track data.
        
        Returns:
            dict: Dictionary with track data
        """
        # Preprocess track data if it hasn't been done already
        if self.track_data and not getattr(self, '_track_data_preprocessed', False):
            self.track_data = preprocess_track_points(self.track_data)
            self._track_data_preprocessed = True
        
        return self.track_data

# Helper functions
def generate_and_load_track(generation_params: Dict = None, output_path: str = None) -> TrackProfile:
    """
    Generate a track and load it as a TrackProfile.
    
    Args:
        generation_params: Optional parameters for track generation
        output_path: Optional path to save generated track
        
    Returns:
        TrackProfile instance
    """
    # Use default parameters if none provided
    if generation_params is None:
        generation_params = {
            'track_width': 3.0,
            'min_length': 200,
            'max_length': 300,
            'curvature_threshold': 0.267
        }
    
    # Create track generator
    generator = FSTrackGenerator(
        track_width=generation_params.get('track_width', 3.0),
        min_length=generation_params.get('min_length', 200),
        max_length=generation_params.get('max_length', 300),
        curvature_threshold=generation_params.get('curvature_threshold', 0.267)
    )
    
    # Generate track
    track_data = generator.generate_track()
    
    # Export track if output path provided
    if output_path:
        generator.export_track(output_path, SimType.FSSIM)
        
        # Create TrackProfile from exported file
        track_profile = TrackProfile(output_path)
    else:
        # Create temporary file to store track
        temp_file = 'temp_track.yaml'
        generator.export_track(temp_file, SimType.FSSIM)
        
        # Load track profile
        track_profile = TrackProfile(temp_file)
        
        # Clean up temporary file
        os.remove(temp_file)
    
    return track_profile

def calculate_optimal_racing_line(track_profile: TrackProfile) -> np.ndarray:
    """
    Calculate the optimal racing line for the track.
    
    This is a simplified implementation that doesn't consider all racing dynamics.
    For a complete implementation, more advanced vehicle dynamics would be needed.
    
    Args:
        track_profile: TrackProfile instance
        
    Returns:
        Numpy array of racing line points
    """
    if track_profile.track_data is None:
        raise ValueError("Track data not loaded")
    
    # Extract track data
    track_points = track_profile.track_data['points']
    track_width = track_profile.track_data.get('width', np.full(len(track_points), 3.0))
    track_curvature = track_profile.track_data['curvature']
    
    # Initialize racing line points
    racing_line = np.zeros_like(track_points)
    
    # Calculate direction vectors
    directions = track_profile.track_data['directions']
    
    # Calculate normal vectors
    normals = np.zeros_like(directions)
    normals[:, 0] = -directions[:, 1]  # Normal = (-dy, dx)
    normals[:, 1] = directions[:, 0]
    
    # Calculate racing line offset based on curvature
    # Positive curvature (left turn) -> negative offset (inside)
    # Negative curvature (right turn) -> positive offset (inside)
    max_offset = track_width / 2 * 0.8  # Use 80% of max width for safety
    
    # Apply smoothing to curvature for racing line calculation
    smoothed_curvature = np.zeros_like(track_curvature)
    window_size = min(11, len(track_curvature))
    
    for i in range(len(track_curvature)):
        # Get indices for window
        half_window = window_size // 2
        start_idx = max(0, i - half_window)
        end_idx = min(len(track_curvature), i + half_window + 1)
        
        # Calculate weighted average
        weights = np.exp(-0.5 * np.arange(-half_window, half_window + 1)**2)
        weights = weights[-(end_idx - start_idx):]  # Adjust for edge cases
        
        window = track_curvature[start_idx:end_idx]
        if len(window) > 0:
            smoothed_curvature[i] = np.sum(window * weights) / np.sum(weights)
    
    # Apply racing line strategy:
    # - Go to outside before turn
    # - Apex in the middle of the turn
    # - Track out after the turn
    for i in range(len(track_points)):
        # Detect sections (turns)
        is_in_turn = abs(smoothed_curvature[i]) > 0.01
        
        if is_in_turn:
            # In a turn - drive on the inside
            offset = -np.sign(smoothed_curvature[i]) * max_offset[i]
        else:
            # On a straight - look ahead for upcoming turns
            look_ahead = min(20, len(track_points) - i - 1)
            upcoming_curvature = 0.0
            
            for j in range(1, look_ahead + 1):
                idx = (i + j) % len(track_points)
                if abs(smoothed_curvature[idx]) > abs(upcoming_curvature):
                    upcoming_curvature = smoothed_curvature[idx]
            
            # If approaching a turn, move to outside
            if abs(upcoming_curvature) > 0.01:
                offset = np.sign(upcoming_curvature) * max_offset[i]
            else:
                # No turn nearby - stay in the center
                offset = 0.0
        
        # Apply offset to track centerline
        racing_line[i] = track_points[i] + offset * normals[i]
    
    return racing_line