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
    min_distance_threshold = 1e-6
    duplicate_indices = []
    
    for i in range(1, len(distances)):
        if distances[i] - distances[i-1] < min_distance_threshold:
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