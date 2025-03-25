"""
Utility functions for track processing and analysis.
"""

import numpy as np
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("Track_Utils")

def preprocess_track_points(track_data):
    """
    Preprocess track data to eliminate duplicate or very close points that
    cause issues with interpolation functions.
    
    Args:
        track_data: Track data dictionary containing 'points', 'distance', etc.
        
    Returns:
        Processed track_data with duplicates removed
    """
    # Make a copy of the track data to avoid modifying the original
    processed_data = track_data.copy()
    
    if 'points' not in processed_data or len(processed_data['points']) < 2:
        return processed_data
    
    points = processed_data['points']
    distances = processed_data.get('distance', None)
    
    # Check if we have distances array, if not, create it
    if distances is None or len(distances) != len(points):
        # Calculate cumulative distances
        distances = np.zeros(len(points))
        for i in range(1, len(points)):
            distances[i] = distances[i-1] + np.linalg.norm(points[i] - points[i-1])
        processed_data['distance'] = distances
    else:
        distances = np.array(distances)
    
    # Find indices where distances are too close (potential duplicates)
    # or where x or y coordinates are duplicates
    min_distance_threshold = 1e-6
    duplicate_indices = []
    
    for i in range(1, len(distances)):
        # Check if distance is too small
        if distances[i] - distances[i-1] < min_distance_threshold:
            duplicate_indices.append(i)
            continue
        
        # Check if x or y coordinates are too close or duplicated
        x_diff = abs(points[i, 0] - points[i-1, 0])
        y_diff = abs(points[i, 1] - points[i-1, 1])
        
        if x_diff < min_distance_threshold or y_diff < min_distance_threshold:
            duplicate_indices.append(i)
    
    if not duplicate_indices:
        return processed_data  # No duplicates found
    
    # Remove duplicate points
    mask = np.ones(len(points), dtype=bool)
    mask[duplicate_indices] = False
    
    processed_data['points'] = points[mask]
    processed_data['distance'] = distances[mask]
    
    # Update other arrays in track_data that match the length of points
    for key, value in processed_data.items():
        if isinstance(value, np.ndarray) and len(value) == len(points):
            processed_data[key] = value[mask]
    
    # Recalculate distances to ensure monotonicity
    if len(processed_data['points']) > 1:
        distances = np.zeros(len(processed_data['points']))
        for i in range(1, len(processed_data['points'])):
            distances[i] = distances[i-1] + np.linalg.norm(processed_data['points'][i] - processed_data['points'][i-1])
        processed_data['distance'] = distances
    
    logger.info(f"Track preprocessing: removed {len(duplicate_indices)} duplicate/close points")
    
    return processed_data

def ensure_unique_values(x, y=None, min_sep=1e-10):
    """
    Ensure x values are unique by adding small increments to duplicates.
    
    Args:
        x: Array of x values
        y: Optional array of y values (will be adjusted if x is changed)
        min_sep: Minimum separation between consecutive values
        
    Returns:
        Tuple of (x_unique, y_adjusted) or just x_unique if y is None
    """
    if len(x) <= 1:
        return x if y is None else (x, y)
    
    x = np.array(x)
    if y is not None:
        y = np.array(y)
    
    # Find indices where x values are too close or identical
    too_close = np.where(np.diff(x) < min_sep)[0]
    
    if len(too_close) == 0:
        return x if y is None else (x, y)
    
    x_unique = x.copy()
    
    # Add small increments to duplicates
    for i in too_close:
        x_unique[i+1] = x_unique[i] + min_sep
    
    # Adjust any subsequent points that might now be too close
    for i in range(len(too_close), len(x)-1):
        if x_unique[i+1] <= x_unique[i]:
            x_unique[i+1] = x_unique[i] + min_sep
    
    if y is None:
        return x_unique
    else:
        return x_unique, y