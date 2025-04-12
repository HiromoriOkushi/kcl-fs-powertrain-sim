"""Utility functions specifically for track generation tasks."""

import os
import pandas as pd
from typing import List, Dict, Optional
import logging

# Import necessary components from the package
from .enums import TrackMode, SimType
from .generator import FSTrackGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("TrackGeneratorUtils")

def generate_multiple_tracks(
    num_tracks: int = 5,
    base_dir: str = "./data",
    mode: TrackMode = TrackMode.EXTEND,
    visualize: bool = False,
    export_formats: Optional[List[SimType]] = None,
    max_retries: int = 20,
    **kwargs # Pass additional kwargs to FSTrackGenerator constructor
    ) -> List[Dict]:
    """
    Generates multiple Formula Student tracks using FSTrackGenerator.

    Args:
        num_tracks: The number of tracks to generate.
        base_dir: The base directory where 'output/tracks' will be created.
        mode: The track generation mode (EXPAND, EXTEND, RANDOM).
        visualize: Whether to show intermediate/final plots during generation.
        export_formats: A list of SimType enums for desired output formats. Defaults to [SimType.FSDS].
        max_retries: Maximum attempts per track generation.
        **kwargs: Additional keyword arguments passed to FSTrackGenerator constructor
                  (e.g., track_width, min_length, n_points).

    Returns:
        A list of dictionaries, each containing metadata of a successfully generated track.
    """
    if export_formats is None:
        export_formats = [SimType.FSDS] # Default to FSDS CSV

    # Define output directory within the base data directory
    output_dir = os.path.abspath(os.path.join(base_dir, "output", "tracks"))
    metadata_dir = os.path.abspath(os.path.join(base_dir, "output")) # Store metadata one level up

    logger.info(f"Attempting to generate {num_tracks} tracks in '{output_dir}'...")
    logger.info(f"Metadata will be stored in '{metadata_dir}'")

    # Initialize the generator (pass kwargs)
    generator = FSTrackGenerator(
        base_dir=metadata_dir, # Generator manages metadata file location
        output_dir_override=output_dir, # Explicitly set track output dir
        visualize=visualize,
        **kwargs # Pass other params like track_width, min_length etc.
        )
    tracks_metadata = []
    successful_generations = 0

    for i in range(num_tracks):
        logger.info(f"--- Generating Track {i+1}/{num_tracks} ---")
        try:
            # Generate the track, this saves the primary CSV and returns metadata
            metadata = generator.generate_track(mode=mode, max_retries=max_retries)

            if metadata:
                tracks_metadata.append(metadata)
                successful_generations += 1
                logger.info(f"Successfully generated track: {metadata['filename']}")
                logger.info(f"  Length: {metadata['track_length']:.1f}m, Cones: {metadata['num_cones']}")

                # Export in additional requested formats
                base_filename_no_ext = os.path.splitext(metadata['filename'])[0]
                for fmt in export_formats:
                     # Skip if format is the default (already saved) - Assuming default is CSV for generate_track
                     if fmt == SimType.FSDS and metadata['filepath'].endswith('.csv'):
                         continue # Already saved by generate_track

                     output_path = os.path.join(generator.output_dir, f"{base_filename_no_ext}.{fmt.value}")
                     try:
                         if generator.export_track(output_path, fmt):
                             logger.info(f"  Exported as {fmt.name}: {os.path.basename(output_path)}")
                         else:
                             logger.warning(f"  Failed to export as {fmt.name}")
                     except Exception as export_err:
                         logger.error(f"  Error exporting track {metadata['filename']} to {fmt.name}: {export_err}")

            else:
                 logger.warning(f"Track generation {i+1} failed after retries, skipping.")

        except Exception as e:
            logger.error(f"Critical error during generation of track {i+1}: {e}", exc_info=True)
            continue # Move to the next track generation attempt

    # Print summary statistics
    logger.info(f"\n--- Generation Summary ---")
    logger.info(f"Successfully generated: {successful_generations}/{num_tracks} tracks.")
    if successful_generations > 0:
        try:
            if os.path.exists(generator.metadata_file):
                 df = pd.read_csv(generator.metadata_file)
                 # Filter for only the tracks generated in this run if possible (e.g., based on timestamp or filenames)
                 # This part is tricky without unique run IDs. Let's assume the last N entries correspond to this run.
                 relevant_df = df.tail(successful_generations)
                 if not relevant_df.empty:
                     avg_len = relevant_df['track_length'].mean()
                     avg_cones = relevant_df['num_cones'].mean()
                     logger.info(f"Average track length: {avg_len:.1f}m")
                     logger.info(f"Average number of cones: {avg_cones:.0f}")
                 else:
                      logger.warning("Could not extract stats for this run from metadata.")
            else:
                 logger.warning("Metadata file not found for summary stats.")

        except Exception as read_err:
            logger.error(f"Could not read or parse metadata file for summary: {read_err}")

    logger.info(f"Track generation process finished. Tracks saved in: {output_dir}")
    return tracks_metadata

# Example usage (if run directly)
if __name__ == "__main__":
    print("Running example: Generate 3 tracks...")
    # Define parameters for generation
    gen_params = {
        'track_width': 3.5,
        'min_length': 300,
        'max_length': 500,
        'n_points': 70,
        'n_regions': 25
    }
    # Specify desired formats
    formats_to_export = [SimType.FSDS, SimType.FSSIM, SimType.GPX]

    # Generate tracks in './data' directory relative to this script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    data_dir = os.path.join(project_root, 'data')

    generated_metadata = generate_multiple_tracks(
        num_tracks=3,
        base_dir=data_dir, # Pass the base data directory
        mode=TrackMode.EXTEND,
        visualize=False, # Set to True to see plots during generation
        export_formats=formats_to_export,
        max_retries=15,
        **gen_params # Pass the generation parameters
    )

    print("\nMetadata of generated tracks:")
    for meta in generated_metadata:
        print(f"- {meta['filename']} (Length: {meta['track_length']:.1f}m)")