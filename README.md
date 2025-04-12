================================================
FILE: README.md
================================================
# KCL Formula Student Powertrain Simulation

This repository contains a comprehensive digital twin simulation environment for the King's College London (KCL) Formula Student team's powertrain. It allows for detailed analysis of engine performance, thermal management, transmission behavior, and overall vehicle dynamics on various track layouts.

## Features

*   **Modular Powertrain Components:** Detailed models for engine (Honda CBR600F4i), transmission (including CAS), fuel system, and thermal management systems (radiators, side pods, fans, electric compressor).
*   **Event Simulation:** Simulates key Formula Student dynamic events:
    *   Acceleration (75m)
    *   Skidpad
    *   Autocross (Lap Time)
    *   Endurance (multi-lap with thermal and reliability effects)
*   **Track Generation:** Includes a Voronoi-based track generator capable of creating realistic Formula Student track layouts.
*   **Performance Analysis:** Tools for:
    *   Lap time optimization (geometric and advanced numerical methods)
    *   Weight sensitivity analysis
    *   Cooling system performance comparison
    *   Shift strategy optimization
*   **Configuration Driven:** Utilizes YAML configuration files for easy parameterization of vehicle components and simulation settings.
*   **Visualization:** Centralized plotting system for visualizing simulation results, performance metrics, and component behavior.
*   **Validation Framework:** Utilities for validating simulation results against expected ranges and theoretical models.

## Project Structure
hiromoriokushi-kcl-fs-powertrain-sim/
├── configs/ # Configuration files (YAML)
├── data/ # Input/Output data (tracks, results)
├── examples/ # Example scripts demonstrating usage
├── kcl_fs_powertrain/ # Main simulation package source code
│ ├── core/ # Core simulation engine, vehicle, track models
│ ├── engine/ # Engine physics, thermal, fuel system models
│ ├── performance/ # Event simulation and analysis tools
│ ├── thermal/ # Detailed cooling system component models
│ ├── track_generator/ # Track generation tools
│ ├── transmission/ # Gearing, CAS, shift strategy models
│ └── utils/ # Constants, plotting, validation utilities
├── plots/ # Default directory for saved plots
├── scripts/ # Helper scripts (e.g., run scripts)
├── tests/ # Unit and integration tests (currently minimal)
├── main.py # Main entry point for running simulations
├── README.md # This file
└── requirements.txt # Python package dependencies



## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd hiromoriokushi-kcl-fs-powertrain-sim
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **(Optional) Install the package locally:**
    For easier imports and potential distribution:
    ```bash
    pip install -e .
    ```

## Usage

The main entry point for running comprehensive simulations is `main.py`.

```bash
python main.py [options]

Refer to the main.py script and its argument parser for available options to control which simulations and analyses are run.
Example scripts in the examples/ directory demonstrate specific functionalities:
generate_basic_track.py: Generates a new track layout.
run_engine_demo.py: Demonstrates basic engine modeling.
shift_visual.py: Visualizes transmission and shifting behavior.
Thermal System Visualization.py: Focuses on thermal system performance.

Configuration
Vehicle parameters, simulation settings, and target metrics are defined in YAML files within the configs/ directory. Modify these files to tailor the simulation to specific vehicle setups or analysis goals.
