"""
Unified interface for lap time optimization methods.

Provides functions to run either basic or advanced lap time simulations
and to compare the results of different methods.
"""

import os
import numpy as np
import yaml
import logging
from typing import Dict, Optional, Tuple, List, Literal
import time

# Import necessary components from the package
try:
    from ..core.vehicle import Vehicle
    from ..core.track_integration import TrackProfile
    from .lap_time import run_fs_lap_simulation # Basic simulation runner
    from .optimal_lap_time import run_advanced_lap_optimization # Advanced runner
    from ..utils.plotting import plot_lap_time_comparison as plot_lap_comp_unified # Unified comparison plotter
    from ..utils.plotting import save_plot, set_plot_style
except ImportError:
    # Fallbacks
    class Vehicle: pass
    class TrackProfile: pass
    def run_fs_lap_simulation(*args, **kwargs): logger.error("Basic sim failed: Module not found."); return {'lap_time': 999.9, 'error': 'Module not found'}
    def run_advanced_lap_optimization(*args, **kwargs): logger.error("Advanced sim failed: Module not found."); return {'lap_time': 999.9, 'error': 'Module not found'}
    def plot_lap_comp_unified(*args, **kwargs): plt.figure(); plt.plot([0,1]); plt.title("Fallback Comparison Plot"); plt.show(); plt.close(); return plt.gcf()
    def save_plot(fig, path, **kwargs): pass
    def set_plot_style(style): pass
    logger = logging.getLogger("LapTimeOpt_Fallback")
    logger.warning("Could not import all necessary modules. Using fallbacks.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("LapTimeOptimization")


def run_lap_optimization(
    vehicle: Vehicle,
    track_file: str,
    method: Literal['basic', 'advanced'] = 'basic',
    config_file: Optional[str] = None,
    include_thermal: bool = True,
    save_dir: Optional[str] = None
) -> Dict:
    """
    Run lap time calculation/optimization using the specified method.

    Args:
        vehicle: Vehicle model instance.
        track_file: Path to the track file (YAML or CSV).
        method: 'basic' for GG-based simulation, 'advanced' for numerical optimization.
        config_file: Optional path to YAML config file (used primarily by 'advanced').
        include_thermal: Whether to include thermal effects in the simulation.
        save_dir: Optional directory to save results and plots.

    Returns:
        Dictionary containing simulation/optimization results, including 'lap_time'.
    """
    start_time = time.time()
    logger.info(f"--- Running Lap Time Calculation ({method.capitalize()}) ---")
    logger.info(f"Track: {os.path.basename(track_file)}")
    logger.info(f"Include Thermal: {include_thermal}")

    # Create specific save directory for this method if main save_dir is provided
    method_save_dir = os.path.join(save_dir, method) if save_dir else None
    if method_save_dir: os.makedirs(method_save_dir, exist_ok=True)

    if method.lower() == 'advanced':
        results = run_advanced_lap_optimization(
            vehicle=vehicle,
            track_file=track_file,
            config_file=config_file,
            # Note: include_thermal is handled within the advanced optimizer based on its config
            save_dir=method_save_dir
        )
        # Add method info to results
        results['method'] = 'advanced'
    elif method.lower() == 'basic':
        results = run_fs_lap_simulation(
            vehicle=vehicle,
            track_file=track_file,
            include_thermal=include_thermal,
            save_dir=method_save_dir
        )
        # Add method info to results
        results['method'] = 'basic'
    else:
        logger.error(f"Invalid optimization method specified: '{method}'. Choose 'basic' or 'advanced'.")
        return {'error': f"Invalid method '{method}'", 'lap_time': None}

    elapsed_time = time.time() - start_time
    lap_time_result = results.get('lap_time')
    if lap_time_result is not None and 'error' not in results:
        logger.info(f"--- {method.capitalize()} Lap Time Calculation Finished ---")
        logger.info(f" Lap Time: {lap_time_result:.3f} s")
        logger.info(f" Calculation Time: {elapsed_time:.2f} s")
    else:
        logger.error(f"--- {method.capitalize()} Lap Time Calculation Failed ---")
        logger.error(f" Error: {results.get('error', 'Unknown error')}")
        logger.info(f" Calculation Time: {elapsed_time:.2f} s")


    return results


def compare_optimization_methods(
    vehicle: Vehicle,
    track_file: str,
    config_file: Optional[str] = None,
    include_thermal: bool = True,
    save_dir: Optional[str] = None
) -> Dict:
    """
    Run both basic and advanced lap time methods and compare results.

    Args:
        vehicle: Vehicle model instance.
        track_file: Path to the track file.
        config_file: Optional path to YAML config file (passed to advanced method).
        include_thermal: Whether to include thermal effects.
        save_dir: Optional directory to save comparison results and plots.

    Returns:
        Dictionary containing results from both methods and a comparison summary.
    """
    logger.info("--- Comparing Lap Time Optimization Methods ---")
    if save_dir: os.makedirs(save_dir, exist_ok=True)

    # Run Basic Simulation
    basic_results = run_lap_optimization(
        vehicle=vehicle,
        track_file=track_file,
        method='basic',
        include_thermal=include_thermal,
        save_dir=os.path.join(save_dir, 'basic') if save_dir else None
    )

    # Run Advanced Optimization
    advanced_results = run_lap_optimization(
        vehicle=vehicle,
        track_file=track_file,
        method='advanced',
        config_file=config_file,
        include_thermal=include_thermal, # Advanced handles this via its config now
        save_dir=os.path.join(save_dir, 'advanced') if save_dir else None
    )

    # --- Comparison Logic ---
    comparison = {'basic': basic_results, 'advanced': advanced_results, 'difference': {}}
    basic_time = basic_results.get('lap_time')
    advanced_time = advanced_results.get('lap_time')

    if basic_time is not None and advanced_time is not None and basic_time > 0:
         time_diff = basic_time - advanced_time
         time_diff_pct = (time_diff / basic_time) * 100.0
         comparison['difference'] = {
             'lap_time_diff_s': time_diff,
             'lap_time_improvement_pct': time_diff_pct
         }
         logger.info("\n--- Comparison Summary ---")
         logger.info(f" Basic Lap Time:    {basic_time:.3f} s")
         logger.info(f" Advanced Lap Time: {advanced_time:.3f} s")
         logger.info(f" Difference:        {time_diff:.3f} s ({time_diff_pct:+.2f}%)")
    else:
         logger.warning("Could not calculate comparison difference due to missing or invalid lap times.")

    # --- Generate Comparison Plot ---
    if save_dir:
        # Prepare data for the unified comparison plotter
        plot_comparison_data = []
        if basic_results and 'error' not in basic_results:
            plot_comparison_data.append({**basic_results.get('results',{}), 'label': 'Basic Sim', 'lap_time': basic_time})
        if advanced_results and 'error' not in advanced_results:
             # Extract necessary plotting data from advanced results
             adv_plot_data = {
                  'label': 'Advanced Opt',
                  'lap_time': advanced_time,
                  'distance': np.array([s['distance'] for s in advanced_results.get('vehicle_states', [])]),
                  'speed': np.array([s['speed'] for s in advanced_results.get('vehicle_states', [])]),
                  'racing_line': advanced_results.get('racing_line'),
                  'track_points': advanced_results.get('track_data',{}).get('points') # Pass track points if possible
             }
             plot_comparison_data.append(adv_plot_data)


        if len(plot_comparison_data) > 1:
             plot_lap_comp_unified(
                 plot_comparison_data,
                 labels=[d['label'] for d in plot_comparison_data], # Pass labels explicitly
                 save_path=os.path.join(save_dir, 'lap_method_comparison.png')
             )
        else:
             logger.warning("Could not generate comparison plot: Insufficient valid results.")

    return comparison

# Example Usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        from ..core.vehicle import create_formula_student_vehicle
        from .lap_time import create_example_track # Use lap_time's version
        import tempfile

        print("Lap Time Optimization Comparison Demo")
        print("-" * 35)

        vehicle = create_formula_student_vehicle()
        output_dir = tempfile.mkdtemp()
        track_file = os.path.join(output_dir, "compare_track.yaml")
        config_file = os.path.join(output_dir, "compare_optim_config.yaml")

        print(f"Output directory: {output_dir}")
        create_example_track(track_file, difficulty='medium') # Medium track

        # Create a config for advanced optimization (if needed)
        optim_config = {
             'optimization': {
                  'max_iterations': 20, # Reduced iterations
                  'num_control_points': 35,
                  'dt': 0.025
             }
        }
        with open(config_file, 'w') as f: yaml.dump(optim_config, f)

        # Run comparison
        comparison_results = compare_optimization_methods(
            vehicle,
            track_file,
            config_file=config_file,
            include_thermal=True,
            save_dir=output_dir
        )

        # Print results
        if 'difference' in comparison_results and 'lap_time_diff_s' in comparison_results['difference']:
            print("\n--- Comparison Results ---")
            print(f" Basic Lap Time:    {comparison_results['basic']['lap_time']:.3f} s")
            print(f" Advanced Lap Time: {comparison_results['advanced']['lap_time']:.3f} s")
            print(f" Difference:        {comparison_results['difference']['lap_time_diff_s']:.3f} s ({comparison_results['difference']['lap_time_improvement_pct']:.2f} %)")
        else:
            print("\nComparison failed or produced invalid results.")
            if 'error' in comparison_results.get('basic', {}): print(f" Basic Error: {comparison_results['basic']['error']}")
            if 'error' in comparison_results.get('advanced', {}): print(f" Advanced Error: {comparison_results['advanced']['error']}")


    except ImportError as e:
        print(f"\nError: Could not import necessary modules ({e}).")
    except FileNotFoundError as e:
         print(f"\nError: Configuration file not found. {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()