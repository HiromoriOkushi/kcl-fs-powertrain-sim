"""Utility functions for track generation."""

import os
import pandas as pd
from typing import List, Dict
from .enums import TrackMode, SimType
from .generator import FSTrackGenerator

def generate_multiple_tracks(
    num_tracks: int = 5,
    base_dir: str = "./data",
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
    print(f"Generated tracks will be saved in: {os.path.join(base_dir, 'output/tracks')}\n")
    
    # Modify to use our project structure
    output_dir = os.path.join(base_dir, "output/tracks")
    generator = FSTrackGenerator(output_dir, visualize=visualize)
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
