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

def preprocess_track_points(track_data: dict) -> dict:
    """
    Preprocess track data to eliminate duplicate or very close points that
    cause issues with interpolation functions. Also ensures distance is monotonic.

    Args:
        track_data: Track data dictionary containing 'points', optionally 'distance', 'width', etc.

    Returns:
        Processed track_data dictionary with duplicates removed and distances corrected.
    """
    processed_data = track_data.copy()

    if 'points' not in processed_data or len(processed_data['points']) < 2:
        logger.warning("Track data has insufficient points for preprocessing.")
        return processed_data

    points = np.array(processed_data['points'])
    initial_point_count = len(points)

    # --- 1. Calculate distances if missing or potentially incorrect ---
    recalculate_distances = True
    if 'distance' in processed_data and len(processed_data['distance']) == len(points):
        distances = np.array(processed_data['distance'])
        # Check if distances are monotonically increasing
        if np.all(np.diff(distances) >= 0):
            recalculate_distances = False
    else:
        distances = None

    if recalculate_distances:
        distances = np.zeros(len(points))
        for i in range(1, len(points)):
            dist_increment = np.linalg.norm(points[i] - points[i-1])
            # Prevent zero distance increments which cause issues
            if dist_increment < 1e-9:
                 dist_increment = 1e-9 # Add tiny distance
            distances[i] = distances[i-1] + dist_increment
        processed_data['distance'] = distances
        logger.debug("Recalculated track distances.")

    # --- 2. Identify and remove duplicate/close points ---
    min_distance_threshold = 1e-6 # Threshold for points being "too close"
    indices_to_keep = [0] # Always keep the first point

    for i in range(1, initial_point_count):
        # Check distance from the *last kept* point, not just the previous one
        last_kept_idx = indices_to_keep[-1]
        dist_from_last_kept = np.linalg.norm(points[i] - points[last_kept_idx])

        if dist_from_last_kept >= min_distance_threshold:
            indices_to_keep.append(i)

    num_removed = initial_point_count - len(indices_to_keep)

    if num_removed > 0:
        logger.info(f"Track preprocessing: identified {num_removed} duplicate/close points to remove.")

        # Apply the mask to all relevant arrays
        for key, value in processed_data.items():
            if isinstance(value, (np.ndarray, list)) and len(value) == initial_point_count:
                 # Ensure we convert lists to numpy arrays for boolean indexing
                value_arr = np.array(value)
                processed_data[key] = value_arr[indices_to_keep]

        # --- 3. Recalculate distances again after removal to ensure accuracy ---
        points = processed_data['points']
        distances = np.zeros(len(points))
        for i in range(1, len(points)):
             dist_increment = np.linalg.norm(points[i] - points[i-1])
             if dist_increment < 1e-9: dist_increment = 1e-9
             distances[i] = distances[i-1] + dist_increment
        processed_data['distance'] = distances
        logger.debug("Recalculated distances after point removal.")

    else:
        logger.debug("No duplicate/close points found during preprocessing.")


    # --- 4. Ensure distances are strictly monotonic (needed for interp1d) ---
    processed_data['distance'] = ensure_unique_values(processed_data['distance'])

    return processed_data


def ensure_unique_values(x: np.ndarray, min_sep: float = 1e-9) -> np.ndarray:
    """
    Ensure values in an array are strictly increasing and unique by adding small increments.

    Args:
        x: NumPy array of values (should be mostly sorted).
        min_sep: Minimum separation to enforce.

    Returns:
        NumPy array with unique and strictly increasing values.
    """
    if len(x) <= 1:
        return x

    x_unique = x.copy()
    modified = False

    for i in range(len(x_unique) - 1):
        if x_unique[i+1] <= x_unique[i]:
            # If duplicate or non-monotonic, add the minimum separation
            x_unique[i+1] = x_unique[i] + min_sep
            modified = True
        elif x_unique[i+1] < x_unique[i] + min_sep:
             # If too close but monotonic, enforce minimum separation
             x_unique[i+1] = x_unique[i] + min_sep
             modified = True # Technically modified, even if slightly

    # If we modified any value, we might need a second pass to ensure monotonicity holds everywhere
    # This is because increasing x[i+1] might make it >= x[i+2]
    if modified:
        for i in range(len(x_unique) - 1):
            if x_unique[i+1] <= x_unique[i]:
                 x_unique[i+1] = x_unique[i] + min_sep

    if modified:
        logger.debug(f"Adjusted array to ensure unique/monotonic values with min_sep={min_sep}.")

    return x_unique