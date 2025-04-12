"""
Integration between track representation and powertrain simulation.

Provides the TrackProfile class to load, process, and provide track data
in a format suitable for simulation loops. Includes racing line calculation.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import logging
from scipy.interpolate import interp1d, CubicSpline
from typing import Dict, List, Tuple, Optional

# Import necessary components from the package
try:
    from .track import Track # Import the main Track class
    from ..track_generator.enums import SimType # For format checking
    from ..utils.track_utils import preprocess_track_points, ensure_unique_values
    from ..utils.plotting import plot_track_layout, save_plot
except ImportError:
    # Fallbacks
    class Track: pass
    class SimType: FSSIM='yaml'; FSDS='csv'; GPX='gpx'
    def preprocess_track_points(d): return d
    def ensure_unique_values(x): return x
    def plot_track_layout(*args, **kwargs): plt.figure(); plt.plot([0,1]); plt.title("Fallback Plot"); plt.show(); plt.close(); return plt.gcf()
    def save_plot(fig, path, **kwargs): pass
    logger = logging.getLogger("TrackIntegration_Fallback")
    logger.warning("Could not import all necessary modules. Using fallbacks.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("TrackIntegration")

class TrackProfile:
    """
    Represents a processed track profile ready for simulation.

    Loads track data from various formats, preprocesses it, calculates
    essential properties like curvature, and provides methods to query
    track information at specific distances.
    """
    def __init__(self, track_file: Optional[str] = None, track_object: Optional[Track] = None):
        """
        Initialize TrackProfile from a file path or an existing Track object.

        Args:
            track_file: Path to the track file (YAML or CSV).
            track_object: An existing Track object instance.
        """
        self.track_file: Optional[str] = track_file
        self.track_data: Optional[Dict] = None # Processed data dictionary
        self.track_length: float = 0.0
        self.sections: List[Dict] = [] # List of identified track sections
        self.name: str = "Unnamed Track Profile"

        # Interpolation functions
        self._x_interp: Optional[Callable] = None
        self._y_interp: Optional[Callable] = None
        self._width_interp: Optional[Callable] = None
        self._curvature_interp: Optional[Callable] = None
        self._elevation_interp: Optional[Callable] = None

        if track_object:
            if not isinstance(track_object, Track):
                 raise TypeError("track_object must be an instance of Track.")
            logger.info(f"Initializing TrackProfile from Track object: {track_object.name}")
            self._initialize_from_track_object(track_object)
            self.track_file = track_object.source_file # Inherit source file path
        elif track_file:
            logger.info(f"Initializing TrackProfile from file: {track_file}")
            if not os.path.exists(track_file):
                raise FileNotFoundError(f"Track file not found: {track_file}")
            self._load_and_process_file(track_file)
        else:
            raise ValueError("Either track_file or track_object must be provided.")

    def _initialize_from_track_object(self, track: Track):
        """Initialize using data from an existing Track object."""
        if not track.has_geometry():
            logger.warning(f"Track object '{track.name}' lacks complete geometry. Attempting processing.")
            track._post_load_processing() # Ensure calculations are done
            if not track.has_geometry():
                 raise ValueError("Provided Track object could not be processed to get necessary geometry.")

        # Copy relevant data
        self.track_data = {
            'points': track.points.copy(),
            'distance': track.distances.copy(),
            'curvature': track.curvature.copy(),
            'width': track.width.copy() if not np.isscalar(track.width) else np.full(len(track.points), track.width),
            'name': track.name,
            'length': track.total_length,
            # Copy optional data if it exists
            'elevation': track.elevation.copy() if track.elevation is not None else None,
            'banking': track.banking.copy() if track.banking is not None else None,
            'sections': [{'type': s.segment_type.name, 'start_idx': s.start_idx, 'end_idx': s.end_idx, 'length': s.length_m} for s in track.segments] if track.segments else []
        }
        self.track_length = track.total_length
        self.name = track.name
        self.sections = self.track_data['sections']

        # Create interpolation functions
        self._create_interpolation_functions()


    def _load_and_process_file(self, track_file: str):
        """Load data from file and perform processing."""
        # Use the Track class loading mechanisms
        temp_track = Track()
        if not temp_track.load_from_file(track_file):
             raise ValueError(f"Failed to load track data from file: {track_file}")

        # Initialize from the loaded Track object
        self._initialize_from_track_object(temp_track)


    def _create_interpolation_functions(self):
        """Create interpolation functions for track properties."""
        if not self.track_data or 'distance' not in self.track_data or 'points' not in self.track_data:
             logger.error("Cannot create interpolators: Essential track data missing.")
             return

        dist = self.track_data['distance']
        points = self.track_data['points']
        width = self.track_data['width']
        curvature = self.track_data['curvature']
        elevation = self.track_data.get('elevation')

        # Ensure distance is strictly monotonic for interpolation
        dist_unique = ensure_unique_values(dist)
        if len(dist_unique) != len(dist):
             logger.warning("Distances were not strictly monotonic, adjustments made for interpolation.")
             # Re-interpolate points/width/curvature onto unique distances if needed
             if len(points) == len(dist): points = interp1d(dist, points, axis=0, kind='linear')(dist_unique)
             if len(width) == len(dist): width = interp1d(dist, width, kind='linear')(dist_unique)
             if len(curvature) == len(dist): curvature = interp1d(dist, curvature, kind='linear')(dist_unique)
             if elevation is not None and len(elevation) == len(dist): elevation = interp1d(dist, elevation, kind='linear')(dist_unique)
             dist = dist_unique # Use the unique distances

        # Use cubic interpolation for smoother path, linear for width/curvature
        interp_kind_path = 'cubic' if len(dist) >= 4 else 'linear'
        interp_kind_scalar = 'linear'

        try:
            self._x_interp = interp1d(dist, points[:, 0], kind=interp_kind_path, bounds_error=False, fill_value='extrapolate')
            self._y_interp = interp1d(dist, points[:, 1], kind=interp_kind_path, bounds_error=False, fill_value='extrapolate')
            self._width_interp = interp1d(dist, width, kind=interp_kind_scalar, bounds_error=False, fill_value=(width[0], width[-1]))
            self._curvature_interp = interp1d(dist, curvature, kind=interp_kind_scalar, bounds_error=False, fill_value=(curvature[0], curvature[-1]))
            if elevation is not None:
                self._elevation_interp = interp1d(dist, elevation, kind=interp_kind_scalar, bounds_error=False, fill_value=(elevation[0], elevation[-1]))
            else:
                 self._elevation_interp = None

            logger.debug("Track interpolation functions created.")
        except ValueError as e:
             logger.error(f"Failed to create interpolation functions: {e}. Check track data consistency.")
             # Reset interpolators
             self._x_interp = self._y_interp = self._width_interp = self._curvature_interp = self._elevation_interp = None


    def get_properties_at_distance(self, distance_m: float) -> Dict:
        """
        Get interpolated track properties at a specific distance.

        Args:
            distance_m: Distance along the track centerline (m).

        Returns:
            Dictionary with properties ('x', 'y', 'width', 'curvature', 'elevation').
            Returns empty dict if interpolators not ready.
        """
        if not all([self._x_interp, self._y_interp, self._width_interp, self._curvature_interp]):
            logger.warning("Interpolators not initialized. Cannot get properties.")
            return {}

        # Handle distance wrapping for closed circuits (assuming closed by default)
        distance_m = distance_m % self.track_length if self.track_length > 0 else distance_m

        props = {
            'distance': distance_m,
            'x': float(self._x_interp(distance_m)),
            'y': float(self._y_interp(distance_m)),
            'width': float(self._width_interp(distance_m)),
            'curvature': float(self._curvature_interp(distance_m)),
            'elevation': float(self._elevation_interp(distance_m)) if self._elevation_interp else 0.0
        }
        return props

    def get_track_data(self) -> Optional[Dict]:
        """Return the processed track data dictionary."""
        return self.track_data

    def visualize(self, save_path: Optional[str] = None):
        """Visualize the track profile using the centralized plotting function."""
        if not self.track_data:
            logger.error("Cannot visualize: Track data not loaded.")
            return

        plot_data = {
            'points': self.track_data.get('points'),
            'width': self.track_data.get('width'),
            'name': self.name,
            'length': self.track_length,
            'segments': self.sections # Pass processed sections
            # Add more data if needed by the plotting function
        }
        fig = plot_track_layout(plot_data, title=f"Track Profile: {self.name}", save_path=save_path)
        if fig: plt.show() # Show plot if not saved or if interactive mode
        # if fig: plt.close(fig) # Close after showing/saving

# --- Helper Function for Racing Line (can be called externally) ---
def calculate_optimal_racing_line(track_profile: TrackProfile,
                                 method: str = 'geometric',
                                 **kwargs) -> Optional[np.ndarray]:
    """
    Calculate an optimal racing line for the given track profile.

    Args:
        track_profile: A loaded and processed TrackProfile object.
        method: Optimization method ('geometric', 'minimum_curvature', etc.).
        **kwargs: Additional arguments for the specific optimization method.

    Returns:
        NumPy array of racing line points (x, y), or None if calculation fails.
    """
    if not track_profile or not track_profile.track_data:
        logger.error("Invalid TrackProfile provided for racing line calculation.")
        return None

    # Create a temporary Track object to use its RacingLine calculation methods
    # This feels slightly redundant, maybe RacingLine should operate on TrackProfile directly?
    # For now, follow the pattern of creating a Track object.
    temp_track = Track(name=track_profile.name)
    temp_track.points = track_profile.track_data.get('points')
    temp_track.width = track_profile.track_data.get('width')
    temp_track.distances = track_profile.track_data.get('distance')
    temp_track.curvature = track_profile.track_data.get('curvature')
    temp_track.total_length = track_profile.track_length
    # Transfer other relevant data if needed by RacingLine optimizer

    if not temp_track.has_geometry():
         logger.error("TrackProfile data is insufficient for racing line calculation.")
         return None

    # Instantiate RacingLine and run optimization
    racing_line_obj = RacingLine(temp_track)
    success = False
    if method == 'geometric':
        success = racing_line_obj.optimize_geometric(**kwargs)
    elif method == 'minimum_curvature':
        success = racing_line_obj.optimize_minimum_curvature(**kwargs)
    # Add other methods here...
    else:
        logger.warning(f"Unsupported racing line method: {method}. Using geometric.")
        success = racing_line_obj.optimize_geometric(**kwargs)

    if success and racing_line_obj.line_points is not None:
        return racing_line_obj.line_points
    else:
        logger.warning(f"Failed to calculate '{method}' racing line. Returning centerline.")
        return track_profile.track_data.get('points') # Fallback to centerline

# Example Usage
if __name__ == "__main__":
    logging.basicConfig(level=INFO)
    # Example: Load a track file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    # Use a track generated by the generator if available
    example_track_file = os.path.join(project_root, "data", "output", "tracks", "fs_track_extend_example.yaml") # Adjust filename

    if not os.path.exists(example_track_file):
         print(f"Example track file not found at {example_track_file}. Please generate one first.")
         # Create a dummy yaml for testing
         dummy_data = {'track': [{'x':0,'y':0,'width':3},{'x':50,'y':0,'width':3},{'x':50,'y':20,'width':3},{'x':0,'y':20,'width':3},{'x':0,'y':0,'width':3}]}
         with open(example_track_file, 'w') as f: yaml.dump(dummy_data, f)
         print(f"Created dummy track file: {example_track_file}")

    try:
        print(f"\nLoading track profile from: {example_track_file}")
        track_prof = TrackProfile(track_file=example_track_file)

        print(f"Track Name: {track_prof.name}")
        print(f"Track Length: {track_prof.track_length:.2f} m")
        print(f"Number of Points: {len(track_prof.track_data['points'])}")
        print(f"Number of Sections: {len(track_prof.sections)}")

        # Get properties at a specific distance
        dist = track_prof.track_length / 4.0
        props = track_prof.get_properties_at_distance(dist)
        print(f"\nProperties at {dist:.1f} m:")
        print(f"  Coordinates: ({props['x']:.2f}, {props['y']:.2f})")
        print(f"  Width: {props['width']:.2f} m")
        print(f"  Curvature: {props['curvature']:.4f} 1/m")
        print(f"  Elevation: {props['elevation']:.2f} m")

        # Calculate and visualize racing line
        print("\nCalculating geometric racing line...")
        racing_line_points = calculate_optimal_racing_line(track_prof, method='geometric')

        # Visualize the track and racing line
        print("\nVisualizing track...")
        plot_data = track_prof.get_track_data()
        if racing_line_points is not None:
             plot_data['racing_line'] = racing_line_points

        plot_dir = os.path.join(project_root, "plots", "track")
        os.makedirs(plot_dir, exist_ok=True)
        save_path = os.path.join(plot_dir, f"{os.path.splitext(os.path.basename(example_track_file))[0]}_profile_vis.png")
        plot_track_layout(plot_data, title=f"Track Profile: {track_prof.name}", save_path=save_path)
        print(f"Track visualization saved to {save_path}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except ValueError as e:
        print(f"Error processing track: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()