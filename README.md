# Formula Student Track Generator

A Python tool for generating randomized Formula Student tracks that comply with competition regulations. This tool allows for rapid prototyping of track layouts for testing vehicle simulations and driver training.

## Features

- Generates randomized Formula Student-compliant track layouts
- Multiple track generation modes:
  - `EXPAND`: Results in roundish track shapes
  - `EXTEND`: Results in elongated track shapes
  - `RANDOM`: Select regions randomly
- Exports tracks in multiple formats:
  - `FSSIM`: FSSIM-compatible YAML format
  - `FSDS`: Formula Student Driverless Simulator CSV format
  - `GPX`: GPX track format for GPS simulations
- Visualizes track designs with cone placement
- Automatic start/finish line placement on appropriate straight sections
- Validates tracks against Formula Student regulations

## Requirements

- Python 3.7+
- Dependencies:
  - numpy
  - scipy
  - shapely
  - matplotlib
  - pyyaml
  - gpxpy
  - pandas

## Installation

```bash
# Clone the repository
git clone https://github.com/KCL-Racing/FS-Track-Generator.git
cd FS-Track-Generator

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from track_generator import FSTrackGenerator, TrackMode, SimType

# Initialize the generator
generator = FSTrackGenerator(base_dir="./outputs", visualize=True)

# Generate a single track
metadata = generator.generate_track(mode=TrackMode.EXTEND)

# Export the track in FSDS format
generator.export_track("./outputs/my_track.csv", sim_type=SimType.FSDS)

# Also export in FSSIM format
generator.export_track("./outputs/my_track.yaml", sim_type=SimType.FSSIM)
```

### Generating Multiple Tracks

```python
from track_generator import generate_multiple_tracks, TrackMode, SimType

# Generate 5 tracks with visualization and export as FSDS format
tracks = generate_multiple_tracks(
    num_tracks=5,
    base_dir="./outputs",
    mode=TrackMode.EXTEND,
    visualize=True,
    export_formats=[SimType.FSDS, SimType.FSSIM]
)
```

## Configuration

The track generator can be configured through the `config.yaml` file:

```yaml
# Configuration for track generator
output_dir: './outputs'
visualization: true
track_mode: 'extend'  # Options: extend, expand, random
export_formats: ['fsds', 'fssim', 'gpx']
```

## Track Parameters

The generated tracks follow these Formula Student guidelines:

- Track width: 3.0 meters (standard FSG track width)
- Track length: between 200-300 meters
- Appropriate curvature for Formula Student vehicles
- Proper cone spacing according to regulations

## Integration with Vehicle Simulation

The generated tracks can be directly used with simulation environments like FSDS (Formula Student Driverless Simulator) or FSSIM to test vehicle performance and control algorithms.


## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
