import os
import numpy as np
from scipy import spatial, interpolate, signal
from shapely.geometry import Point, LineString, Polygon
from datetime import datetime
from .enums import TrackMode, SimType
import csv
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import gpxpy
import gpxpy.gpx
import math
from typing import List, Dict, Tuple, Optional

from enum import Enum

class TrackMode(Enum):
    """Possible modes for how Voronoi regions are selected"""
    EXPAND = 1   # Results in roundish track shapes
    EXTEND = 2   # Results in elongated track shapes
    RANDOM = 3   # Select regions randomly

class SimType(Enum):
    """Selection between output format for different simulators"""
    FSSIM = 1     # FSSIM compatible .yaml file
    FSDS = 2       # FSDS compatible .csv file
    GPX = 3         # GPX track format

class FSTrackGenerator:
    def __init__(self, base_dir: str, visualize: bool = False, track_width: float = 3.0, min_length: float = 200, max_length: float = 300, curvature_threshold: float = (1.0 / 3.75), straight_threshold: float = 1.0 / 20.0, length_start_area: float = 6.0, n_points: int = 60, n_regions: int = 20, max_bound: float = 200, min_bound: float = 0, cone_spacing: float = 4.0):
        # Scale factor for generation (will be scaled down in output)
        self.SCALE_FACTOR = 2.0  # Generate at 2x size, then scale down
        
        # Formula Student Track parameters
        self.TRACK_WIDTH = track_width       # meters - standard FSG track width
        self.MIN_LENGTH = min_length         # meters
        self.MAX_LENGTH = max_length        # meters 
        self.CURVATURE_THRESHOLD = curvature_threshold  # Original curvature threshold
        self.STRAIGHT_THRESHOLD = straight_threshold
        self.LENGTH_START_AREA = length_start_area # meters

        # Generation parameters (scaled up)
        self.N_POINTS = 60          # Keep high number of points for detail
        self.N_REGIONS = 20          # Keep regions for complexity
        self.MAX_BOUND = 200         # Original bound
        self.MIN_BOUND = 0   
        
        # Modified spacing parameters
        self.CONE_SPACING = 4.0      # Original spacing
        
        self.visualize = visualize
        self.base_dir = base_dir
        self.output_dir = os.path.join(base_dir, "generated_tracks")
        self.metadata_file = os.path.join(base_dir, "track_metadata.csv")
        self._ensure_directories()
        
        self.track_points = None
        self.cones_left = None
        self.cones_right = None
        self.track_polygon = None
        self.start_line = None
        self.start_position = None
        self.start_heading = None
    
    def _scale_track(self, track_data: np.ndarray) -> np.ndarray:
        """Scale track coordinates to match Formula Student specifications"""
        return track_data / self.SCALE_FACTOR


    def _ensure_directories(self):
        """Create necessary directories and metadata file"""
        os.makedirs(self.output_dir, exist_ok=True)
        
        if not os.path.exists(self.metadata_file):
            headers = [
                'filename', 'filepath', 'track_length', 'num_cones',
                'track_width', 'generation_mode', 'generation_time'
            ]
            with open(self.metadata_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)

    def _closest_node(self, node: np.ndarray, nodes: np.ndarray, k: int) -> int:
        """Returns the index of the k-th closest node"""
        deltas = nodes - node
        dist_2 = np.einsum('ij,ij->i', deltas, deltas)
        return np.argpartition(dist_2, k)[k]

    def _clockwise_sort(self, points: np.ndarray) -> np.ndarray:
        """Sorts nodes in clockwise order"""
        center = np.mean(points, axis=0)
        angles = np.arctan2(points[:,0] - center[0], points[:,1] - center[1])
        return points[np.argsort(angles)]

    def _calculate_curvature(self, dx_dt, d2x_dt2, dy_dt, d2y_dt2):
        """Calculates the curvature along a line"""
        return (dx_dt**2 + dy_dt**2)**-1.5 * (dx_dt * d2y_dt2 - dy_dt * d2x_dt2)

    def _arc_length(self, x: np.ndarray, y: np.ndarray, R: np.ndarray) -> np.ndarray:
        """Calculate arc length between points based on radius of curvature"""
        x0, x1 = x[:-1], x[1:]
        y0, y1 = y[:-1], y[1:]
        R = R[:-1]
        
        distance = np.sqrt((x1 - x0)**2 + (y1 - y0)**2)
        theta = 2 * np.arcsin(0.5 * distance / R)
        return R * theta

    def _bounded_voronoi(self, points: np.ndarray):
        """Creates a bounded Voronoi diagram"""
        def mirror_points(boundary: float, axis: int) -> np.ndarray:
            mirrored = np.copy(points_center)
            mirrored[:, axis] = 2 * boundary - mirrored[:, axis]
            return mirrored
            
        points_center = points
        x_min, x_max = self.MIN_BOUND, self.MAX_BOUND
        y_min, y_max = self.MIN_BOUND, self.MAX_BOUND
        
        # Mirror points around boundaries with additional margin
        margin = (x_max - x_min) * 0.1  # 10% margin
        all_points = np.concatenate([
            points_center,
            mirror_points(x_min - margin, 0),
            mirror_points(x_max + margin, 0),
            mirror_points(y_min - margin, 1),
            mirror_points(y_max + margin, 1)
        ])
        
        vor = spatial.Voronoi(all_points)
        vor.filtered_points = points_center
        vor.filtered_regions = [
            region for region in vor.regions 
            if -1 not in region and len(region) > 2
        ]
        return vor

    def _validate_track(self, track: Polygon) -> bool:
        """Enhanced track validation"""
        if not track.is_valid:
            return False
            
        # Ensure track is a single polygon
        if track.geom_type != 'Polygon':
            return False
            
        # Check length is within FS limits
        length = track.length
        if length < self.MIN_LENGTH or length > self.MAX_LENGTH:
            return False
            
        # Check width/height ratio
        bounds = track.bounds
        width = bounds[2] - bounds[0]
        height = bounds[3] - bounds[1]
        if width > self.MAX_BOUND * 1.2 or height > self.MAX_BOUND * 1.2:
            return False
            
        # Check aspect ratio
        aspect_ratio = max(width, height) / min(width, height)
        if aspect_ratio > 3.0 or aspect_ratio < 1.0:
            return False
            
        return True


    def _find_start_position(self, x: np.ndarray, y: np.ndarray, curvature: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """Find suitable start position with straight section that's guaranteed to be on the track"""
        if self.track_polygon is None:
            raise ValueError("Track polygon must be generated before finding start position")
            
        straight_sections = np.abs(curvature) <= self.STRAIGHT_THRESHOLD
        if not any(straight_sections):
            raise ValueError("No suitable straight section found for start position")
            
        # Calculate cumulative distances
        dx = np.diff(x)
        dy = np.diff(y)
        segment_lengths = np.sqrt(dx**2 + dy**2)
        cumulative_length = np.cumsum(segment_lengths)
        
        # Find all potential straight sections
        straight_candidates = []
        current_length = 0
        current_start_idx = None
        
        for i in range(len(straight_sections)-1):
            if straight_sections[i]:
                if current_start_idx is None:
                    current_start_idx = i
                current_length += segment_lengths[i]
            else:
                if current_length >= self.LENGTH_START_AREA:
                    straight_candidates.append({
                        'start_idx': current_start_idx,
                        'length': current_length,
                        'end_idx': i
                    })
                current_length = 0
                current_start_idx = None
                
        # Add final section if it's straight
        if current_length >= self.LENGTH_START_AREA:
            straight_candidates.append({
                'start_idx': current_start_idx,
                'length': current_length,
                'end_idx': len(straight_sections)-1
            })
            
        if not straight_candidates:
            raise ValueError(f"No straight section longer than {self.LENGTH_START_AREA}m found")
            
        # Sort candidates by length (longest first)
        straight_candidates.sort(key=lambda x: x['length'], reverse=True)
        
        # Try each candidate until we find one that's on the track
        for candidate in straight_candidates:
            start_idx = candidate['start_idx']
            # Take a point slightly ahead for the start line
            ahead_idx = min(start_idx + int(self.LENGTH_START_AREA / segment_lengths[start_idx]), 
                        candidate['end_idx'])
            
            # Create points to test
            start_point = Point(x[start_idx], y[start_idx])
            start_line_point = Point(x[ahead_idx], y[ahead_idx])
            
            # Check if both points are within or on the track
            if self.track_polygon.contains(start_point) or self.track_polygon.touches(start_point):
                if self.track_polygon.contains(start_line_point) or self.track_polygon.touches(start_line_point):
                    # Calculate heading
                    start_heading = float(np.arctan2(
                        y[ahead_idx] - y[start_idx],
                        x[ahead_idx] - x[start_idx]
                    ))
                    
                    return (np.array([x[ahead_idx], y[ahead_idx]]), 
                            np.array([x[start_idx], y[start_idx]]), 
                            start_heading)
        
        raise ValueError("No valid straight section found within track boundaries")

    def visualize_voronoi(self, vor, sorted_vertices, random_point_indices, input_points, x, y):
        """Visualizes the Voronoi diagram and resulting track"""
        plt.figure(figsize=(12, 8))
        
        # Plot initial points
        plt.plot(vor.filtered_points[:, 0], vor.filtered_points[:, 1], 'b.')
        
        # Plot vertices points
        for region in vor.filtered_regions:
            vertices = vor.vertices[region, :]
            plt.plot(vertices[:, 0], vertices[:, 1], 'go')
            
        # Plot edges
        for region in vor.filtered_regions:
            vertices = vor.vertices[region + [region[0]], :]
            plt.plot(vertices[:, 0], vertices[:, 1], 'k-')
            
        # Plot selected vertices
        plt.scatter(sorted_vertices[:,0], sorted_vertices[:,1], 
                   color='y', s=200, label='Selected vertices')
        
        # Plot selected points
        plt.scatter(*input_points[random_point_indices].T, 
                   s=100, marker='x', color='b', label='Selected points')
        
        # Plot track
        plt.scatter(x, y, color='r', s=1, label='Track centerline')
        
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        plt.axis('equal')
        plt.legend()
        plt.grid(True)
        plt.title('Voronoi Diagram and Track Generation')
        plt.show()
        
    def plot_track(self):
        """Plots the generated track with cones and dynamic start position"""
        if self.cones_left is None or self.cones_right is None or self.start_position is None:
            raise ValueError("No track has been generated yet")
                
        plt.figure(figsize=(12, 8))
        
        # Plot cones
        plt.scatter(self.cones_left[:, 0], 
                self.cones_left[:, 1], 
                color='b', s=30, label='Blue Cones')
        plt.scatter(self.cones_right[:, 0], 
                self.cones_right[:, 1], 
                color='y', s=30, label='Yellow Cones')
        
        # Calculate start/finish box positions based on start_position and heading
        start_direction = np.array([np.cos(self.start_heading), np.sin(self.start_heading)])
        perpendicular = np.array([-np.sin(self.start_heading), np.cos(self.start_heading)])
        
        # Start box cones (2.5m to each side, at start position)
        start_left = self.start_position + perpendicular * 2.5
        start_right = self.start_position - perpendicular * 2.5
        
        # Finish line cones (2.5m to each side, 2.6m ahead of start)
        finish_center = self.start_position + start_direction * 2.6
        finish_left = finish_center + perpendicular * 2.5
        finish_right = finish_center - perpendicular * 2.5
        
        # Store start/finish positions for export
        self.start_cones = np.array([
            start_left,
            start_right,
            finish_left,
            finish_right
        ])
        
        # Plot start/finish cones
        plt.scatter(self.start_cones[:, 0], self.start_cones[:, 1], 
                color='orange', s=50, label='Start/Finish')
        
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        plt.axis('equal')
        plt.grid(True)
        plt.legend()
        plt.title('Formula Student Track Layout')
        plt.show()

    def export_track(self, output_path: str, sim_type: SimType = SimType.FSDS):
        """Export track in specified simulator format"""
        if sim_type == SimType.FSSIM:
            self._export_fssim_yaml(output_path)
        elif sim_type == SimType.FSDS:
            self._export_fsds_csv(output_path)
        elif sim_type == SimType.GPX:
            self._export_gpx(output_path)

    def _export_fssim_yaml(self, output_path: str):
        """Exports track in FSSIM YAML format with dynamic start position"""
        if self.start_cones is None:
            raise ValueError("Track must be plotted before export to generate start/finish positions")
            
        data = {
            'cones_left': self.cones_left.tolist(),
            'cones_right': self.cones_right.tolist(),
            'cones_orange': [],
            'cones_orange_big': self.start_cones.tolist(),
            'starting_pose_cg': [
                float(self.start_position[0]),
                float(self.start_position[1]),
                float(self.start_heading)
            ],
            'tk_device': [
                [self.start_position[0] + 1.3, self.start_position[1] + 3.0],
                [self.start_position[0] + 1.3, self.start_position[1] - 3.0]
            ]
        }
        
        with open(output_path, 'w') as f:
            yaml.dump(data, f)
            
    def _export_fsds_csv(self, output_path: str):
        """Exports track in FSDS CSV format with dynamic start position"""
        if self.start_cones is None:
            raise ValueError("Track must be plotted before export to generate start/finish positions")
            
        with open(output_path, 'w') as f:
            # Write left cones
            for cone in self.cones_left:
                f.write(f"blue,{cone[0]},{cone[1]},0,0.01,0.01,0\n")
                
            # Write right cones
            for cone in self.cones_right:
                f.write(f"yellow,{cone[0]},{cone[1]},0,0.01,0.01,0\n")
                
            # Write start/finish cones using calculated positions
            for cone in self.start_cones:
                f.write(f"big_orange,{cone[0]},{cone[1]},0,0.01,0.01,0\n")

    def _export_gpx(self, output_path: str, lat_offset=51.197682, lon_offset=5.323411):
        """Exports track in GPX format"""
        gpx = gpxpy.gpx.GPX()
        gpx_track = gpxpy.gpx.GPXTrack()
        gpx.tracks.append(gpx_track)
        
        # Add cone waypoints
        for cone in self.cones_left:
            lat = lat_offset + (cone[1] / 6378100) * (180 / math.pi)
            lon = lon_offset + (cone[0] / 6378100) * (180 / math.pi) / math.cos(lat_offset * math.pi/180)
            gpx.waypoints.append(gpxpy.gpx.GPXWaypoint(latitude=lat, longitude=lon, elevation=0))
            
        with open(output_path, 'w') as f:
            f.write(gpx.to_xml())

    def generate_track(self, mode: TrackMode = TrackMode.EXTEND, max_retries: int = 20) -> dict:
        """Generate track with Formula Student constraints"""
        for retry in range(max_retries):
            try:
                # Create initial points
                input_points = np.random.uniform(
                    self.MIN_BOUND,
                    self.MAX_BOUND,
                    (self.N_POINTS, 2)
                )
                
                # Generate Voronoi diagram
                vor = self._bounded_voronoi(input_points)
                
                # Select regions based on mode
                try:
                    if mode == TrackMode.EXTEND:
                        random_index = np.random.randint(0, len(input_points))
                        random_heading = np.random.uniform(0, np.pi/2)
                        random_point = input_points[random_index]
                        
                        start = (
                            random_point[0] - 0.5 * self.MAX_BOUND * np.cos(random_heading),
                            random_point[1] - 0.5 * self.MAX_BOUND * np.sin(random_heading)
                        )
                        end = (
                            random_point[0] + 0.5 * self.MAX_BOUND * np.cos(random_heading),
                            random_point[1] + 0.5 * self.MAX_BOUND * np.sin(random_heading)
                        )
                        line = LineString([start, end])
                        distances = [Point(p).distance(line) for p in input_points]
                        random_point_indices = np.argpartition(distances, self.N_REGIONS)[:self.N_REGIONS]
                    
                    elif mode == TrackMode.EXPAND:
                        random_index = np.random.randint(0, self.N_POINTS)
                        random_point_indices = [random_index]
                        random_point = input_points[random_index]
                        
                        for i in range(self.N_REGIONS - 1):
                            closest_point_index = self._closest_node(random_point, input_points, k=i+1)
                            random_point_indices.append(closest_point_index)
                    
                    else:  # RANDOM
                        random_point_indices = np.random.choice(
                            len(input_points),
                            size=self.N_REGIONS,
                            replace=False
                        )
                    
                    # Get regions belonging to selected points
                    regions = np.array([np.array(region) for region in vor.regions], dtype=object)
                    random_region_indices = vor.point_region[random_point_indices]
                    random_regions = np.concatenate(regions[random_region_indices])
                    
                    # Get vertices of random regions
                    random_vertices = np.unique(vor.vertices[random_regions], axis=0)
                    
                    # Sort vertices clockwise and close loop
                    sorted_vertices = self._clockwise_sort(random_vertices)
                    sorted_vertices = np.vstack([sorted_vertices, sorted_vertices[0]])
                    
                    while True:
                        # Interpolate with no smoothing
                        tck, _ = interpolate.splprep([sorted_vertices[:,0], sorted_vertices[:,1]], s=0, per=True)
                        t = np.linspace(0, 1, 1000)
                        x, y = interpolate.splev(t, tck, der=0)
                        dx_dt, dy_dt = interpolate.splev(t, tck, der=1)
                        d2x_dt2, d2y_dt2 = interpolate.splev(t, tck, der=2)
                        
                        # Calculate curvature
                        k = self._calculate_curvature(dx_dt, d2x_dt2, dy_dt, d2y_dt2)
                        abs_curvature = np.abs(k)
                        
                        # Check curvature peaks
                        peaks, _ = signal.find_peaks(abs_curvature)
                        exceeded_peaks = abs_curvature[peaks] > self.CURVATURE_THRESHOLD
                        
                        if any(exceeded_peaks):
                            # Remove vertex at highest curvature
                            max_peak_index = abs_curvature[peaks].argmax()
                            max_peak = peaks[max_peak_index]
                            peak_coordinate = (x[max_peak], y[max_peak])
                            vertice = self._closest_node(peak_coordinate, sorted_vertices, k=0)
                            sorted_vertices = np.delete(sorted_vertices, vertice, axis=0)
                            
                            if len(sorted_vertices) < 4:
                                print(f"Retry {retry + 1}: Too few vertices remain after curvature reduction")
                                break
                                
                            # Ensure loop is closed
                            if not np.array_equal(sorted_vertices[0], sorted_vertices[-1]):
                                sorted_vertices = np.vstack([sorted_vertices, sorted_vertices[0]])
                        else:
                            break
                    
                    # Create track boundaries
                    track = Polygon(zip(x, y))
                    if not track.is_valid or track.geom_type != 'Polygon':
                        print(f"Retry {retry + 1}: Invalid track geometry")
                        continue
                        
                    track_left = track.buffer(self.TRACK_WIDTH / 2)
                    track_right = track.buffer(-self.TRACK_WIDTH / 2)
                    
                    if not (track_left.is_valid and track_right.is_valid and 
                        track_left.geom_type == 'Polygon' and 
                        track_right.geom_type == 'Polygon'):
                        print(f"Retry {retry + 1}: Invalid track boundaries")
                        continue
                    
                    # Place cones with even spacing
                    cone_spacing_left = np.linspace(
                        0, 
                        track_left.length, 
                        np.ceil(track_left.length / self.TRACK_WIDTH).astype(int) + 1
                    )[:-1]
                    
                    cone_spacing_right = np.linspace(
                        0, 
                        track_right.length, 
                        np.ceil(track_right.length / self.TRACK_WIDTH).astype(int) + 1
                    )[:-1]
                    
                    self.cones_left = np.array([
                        track_left.exterior.interpolate(d).coords[0] 
                        for d in cone_spacing_left
                    ])
                    
                    self.cones_right = np.array([
                        track_right.exterior.interpolate(d).coords[0] 
                        for d in cone_spacing_right
                    ])
                    
                    # Scale down all track components
                    self.cones_left = self._scale_track(self.cones_left)
                    self.cones_right = self._scale_track(self.cones_right)
                    
                    # Scale track points
                    x = self._scale_track(x)
                    y = self._scale_track(y)
                    
                    # Scale and store track polygon BEFORE finding start position
                    scaled_coords = zip(self._scale_track(np.array([p[0] for p in track.exterior.coords])),
                                    self._scale_track(np.array([p[1] for p in track.exterior.coords])))
                    track = Polygon(list(scaled_coords))
                    self.track_polygon = track  # Store track polygon before finding start position
                    
                    # Store track points
                    self.track_points = (x, y)
                    
                    # Now find start position
                    try:
                        self.start_line, self.start_position, self.start_heading = (
                            self._find_start_position(x, y, abs_curvature)
                        )
                    except ValueError as e:
                        print(f"Retry {retry + 1}: {str(e)}")
                        continue
                    
                    # Generate visualization if enabled
                    if self.visualize:
                        try:
                            self.visualize_voronoi(vor, sorted_vertices, random_point_indices, input_points, x, y)
                            self.plot_track()
                        except Exception as e:
                            print(f"Visualization error (non-critical): {e}")
                    
                    # Save track data and return metadata
                    filename = f"track_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                    filepath = os.path.join(self.output_dir, filename)
                    
                    metadata = {
                        'filename': filename,
                        'filepath': filepath,
                        'track_length': float(track.length),
                        'num_cones': len(self.cones_left) + len(self.cones_right),
                        'track_width': self.TRACK_WIDTH,
                        'generation_mode': mode.value,
                        'generation_time': datetime.now().isoformat()
                    }
                    
                    pd.DataFrame([metadata]).to_csv(
                        self.metadata_file, 
                        mode='a', 
                        header=False, 
                        index=False
                    )
                    
                    cone_data = []
                    for x, y in self.cones_left:
                        cone_data.append(['blue', x, y, 0])
                    for x, y in self.cones_right:
                        cone_data.append(['yellow', x, y, 0])
                        
                    pd.DataFrame(
                        cone_data, 
                        columns=['color', 'x', 'y', 'z']
                    ).to_csv(filepath, index=False)
                    
                    print(f"Successfully generated track after {retry + 1} attempts")
                    return metadata
                    
                except Exception as e:
                    print(f"Error during region processing: {str(e)}")
                    continue
                    
            except Exception as e:
                print(f"Retry {retry + 1} failed: {str(e)}")
                if retry == max_retries - 1:
                    raise ValueError(f"Failed to generate valid track after {max_retries} attempts")
                continue
        
        raise ValueError("Could not generate valid track after maximum attempts")
def generate_multiple_tracks(
    num_tracks: int = 5,
    base_dir: str = "./tracks",
    mode: TrackMode = TrackMode.EXTEND,
    visualize: bool = False,
    export_formats: List[SimType] = [SimType.FSDS],
    max_retries: int = 20
    ) -> List[Dict]:
    """Generate multiple tracks and return their metadata"""
    # Convert base_dir to absolute path and print directory info
    base_dir = os.path.abspath(base_dir)
    print(f"\nOutput Directory Information:")
    print(f"Base directory: {base_dir}")
    print(f"Generated tracks will be saved in: {os.path.join(base_dir, 'generated_tracks')}\n")
    
    generator = FSTrackGenerator(base_dir, visualize=visualize)
    tracks = []
    
    for i in range(num_tracks):
        try:
            metadata = generator.generate_track(mode, max_retries=max_retries)
            tracks.append(metadata)
            print(f"\nGenerated track {i+1}/{num_tracks}:")
            print(f"  Base filename: {metadata['filename']}")
            print(f"  Saving to: {metadata['filepath']}")
            print(f"  Track length: {metadata['track_length']:.1f}m")
            print(f"  Number of cones: {metadata['num_cones']}")
            
            # Export in requested formats
            basename = os.path.splitext(metadata['filename'])[0]
            for fmt in export_formats:
                output_path = os.path.join(
                    generator.output_dir,
                    f"{basename}.{fmt.value}"
                )
                generator.export_track(output_path, fmt)
                print(f"  Also exported as: {output_path}")
            
        except Exception as e:
            print(f"Failed to generate track {i+1}: {str(e)}")
            continue
    
    # Print summary statistics with file locations
    if tracks:
        print(f"\nMetadata saved to: {generator.metadata_file}")
        df = pd.read_csv(generator.metadata_file)
        print("\nGeneration Summary:")
        print(f"Successfully generated: {len(tracks)}/{num_tracks} tracks")
        print(f"Average track length: {df['track_length'].mean():.1f}m")
        print(f"Average number of cones: {df['num_cones'].mean():.1f}")
        print(f"Generation modes used: {df['generation_mode'].value_counts().to_dict()}")
    
    return tracks

