"""Example for generating a basic Formula Student track."""

import os
import sys
import matplotlib.pyplot as plt

# Add the parent directory to the path to find the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from kcl_fs_powertrain.track_generator.enums import TrackMode, SimType
from kcl_fs_powertrain.track_generator.generator import FSTrackGenerator

def main():
    # Create output directory
    output_dir = os.path.join("data", "output", "tracks")
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize track generator
    generator = FSTrackGenerator(output_dir, visualize=True)
    
    # Generate track
    print("Generating Formula Student track...")
    metadata = generator.generate_track(TrackMode.EXTEND, max_retries=20)
    
    # Print track information
    print("\nTrack Generated:")
    print(f"  Filename: {metadata['filename']}")
    print(f"  Track length: {metadata['track_length']:.1f}m")
    print(f"  Number of cones: {metadata['num_cones']}")
    
    # Export in different formats
    base_filename = os.path.splitext(metadata['filename'])[0]
    
    for sim_type in [SimType.FSDS, SimType.FSSIM]:
        output_path = os.path.join(output_dir, f"{base_filename}.{sim_type.value}")
        generator.export_track(output_path, sim_type)
        print(f"  Exported as: {output_path}")
    
    # Plot the track
    generator.plot_track()
    plt.savefig(os.path.join(output_dir, f"{base_filename}_visualization.png"))
    plt.show()

if __name__ == "__main__":
    main()
