"""
Integration module for lap time optimization.

This module provides a unified interface for running lap time simulations
with different optimization methods, from basic simulation to advanced
numerical optimization with Runge-Kutta integration.
"""

import os
import numpy as np
import yaml
import logging
from typing import Dict, Optional, Tuple, List, Literal

from ..core.vehicle import Vehicle
from ..core.track_integration import TrackProfile
from .lap_time import create_example_track
from .lap_time import LapTimeSimulator, create_lap_time_simulator, run_fs_lap_simulation
from .optimal_lap_time import OptimalLapTimeOptimizer, run_advanced_lap_optimization

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("Lap_Time_Optimization")


def run_lap_optimization(
    vehicle: Vehicle,
    track_file: str,
    method: Literal['basic', 'advanced'] = 'basic',
    config_file: Optional[str] = None,
    include_thermal: bool = True,
    save_dir: Optional[str] = None
) -> Dict:
    """
    Run lap time optimization with the specified method.
    
    Args:
        vehicle: Vehicle model
        track_file: Path to track file
        method: Optimization method ('basic' or 'advanced')
        config_file: Optional path to configuration file
        include_thermal: Whether to include thermal effects
        save_dir: Optional directory to save results
        
    Returns:
        Dictionary with optimization results
    """
    # Load configuration if provided
    config = {}
    if config_file and os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_file}")
    
    # Load track
    track_profile = TrackProfile(track_file)
    
    # Run appropriate optimization method
    if method.lower() == 'advanced':
        # Advanced optimization with Runge-Kutta integration
        logger.info("Running advanced lap time optimization...")
        
        # Create optimizer with configuration
        optimizer = OptimalLapTimeOptimizer(vehicle, track_profile)
        
        # Apply configuration if provided
        if 'optimization' in config:
            opt_config = config['optimization']
            optimizer.dt = opt_config.get('dt', optimizer.dt)
            optimizer.max_time = opt_config.get('max_time', optimizer.max_time)
            optimizer.include_thermal = opt_config.get('include_thermal', include_thermal)
            optimizer.max_iterations = opt_config.get('max_iterations', optimizer.max_iterations)
            optimizer.tolerance = opt_config.get('tolerance', optimizer.tolerance)
            optimizer.optimization_method = opt_config.get('method', optimizer.optimization_method)
            optimizer.num_control_points = opt_config.get('num_control_points', optimizer.num_control_points)
        
        # Run optimization
        results = optimizer.optimize_lap_time()
        
        # Visualize and save results if requested
        if save_dir and results['racing_line'] is not None:
            os.makedirs(save_dir, exist_ok=True)
            
            # Save visualization
            optimizer.visualize_optimization_results(
                results,
                save_path=os.path.join(save_dir, "optimal_lap.png")
            )
            
            # Save racing line to CSV
            np.savetxt(
                os.path.join(save_dir, "optimal_racing_line.csv"),
                results['racing_line'],
                delimiter=',',
                header='x,y'
            )
        
        return results
    else:
        # Basic lap time simulation
        logger.info("Running basic lap time simulation...")
        return run_fs_lap_simulation(
            vehicle,
            track_file,
            include_thermal=include_thermal,
            save_dir=save_dir
        )


def compare_optimization_methods(
    vehicle: Vehicle,
    track_file: str,
    config_file: Optional[str] = None,
    include_thermal: bool = True,
    save_dir: Optional[str] = None
) -> Dict:
    """
    Compare basic and advanced lap time optimization methods.
    
    Args:
        vehicle: Vehicle model
        track_file: Path to track file
        config_file: Optional path to configuration file
        include_thermal: Whether to include thermal effects
        save_dir: Optional directory to save results
        
    Returns:
        Dictionary with comparison results
    """
    logger.info("Comparing lap time optimization methods...")
    
    # Run basic optimization
    basic_results = run_lap_optimization(
        vehicle,
        track_file,
        method='basic',
        config_file=config_file,
        include_thermal=include_thermal,
        save_dir=os.path.join(save_dir, 'basic') if save_dir else None
    )
    
    # Run advanced optimization
    advanced_results = run_lap_optimization(
        vehicle,
        track_file,
        method='advanced',
        config_file=config_file,
        include_thermal=include_thermal,
        save_dir=os.path.join(save_dir, 'advanced') if save_dir else None
    )
    
    # Create comparison visualization
    if save_dir:
        import matplotlib.pyplot as plt
        
        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Create comparison figure
        plt.figure(figsize=(12, 8))
        
        # Basic metrics
        basic_lap_time = basic_results['lap_time']
        basic_avg_speed = basic_results['metrics']['avg_speed_kph']
        
        # Advanced metrics
        advanced_lap_time = advanced_results['lap_time']
        advanced_racing_line = advanced_results.get('racing_line')
        
        # Create bar chart for lap times
        plt.subplot(211)
        methods = ['Basic Simulation', 'Advanced Optimization']
        lap_times = [basic_lap_time, advanced_lap_time]
        
        bars = plt.bar(methods, lap_times)
        plt.ylabel('Lap Time (s)')
        plt.title('Lap Time Comparison')
        
        # Add lap time labels
        for bar, time in zip(bars, lap_times):
            plt.text(
                bar.get_x() + bar.get_width()/2.,
                time + 0.1,
                f'{time:.3f}s',
                ha='center',
                va='bottom'
            )
        
        # Plot racing lines
        plt.subplot(212)
        
        # Load track profile for visualization
        track_profile = TrackProfile(track_file)
        track_data = track_profile.get_track_data()
        track_points = track_data['points']
        
        # Plot track centerline
        plt.plot(
            track_points[:, 0],
            track_points[:, 1],
            'k--',
            alpha=0.5,
            label='Track Centerline'
        )
        
        # Plot racing lines if available
        if 'results' in basic_results and 'racing_line' in basic_results:
            basic_line = basic_results['results'].get('racing_line')
            if basic_line is not None:
                plt.plot(
                    basic_line[:, 0],
                    basic_line[:, 1],
                    'b-',
                    linewidth=2,
                    label='Basic Racing Line'
                )
        
        if advanced_racing_line is not None:
            plt.plot(
                advanced_racing_line[:, 0],
                advanced_racing_line[:, 1],
                'r-',
                linewidth=2,
                label='Optimized Racing Line'
            )
        
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.title('Racing Line Comparison')
        plt.legend()
        plt.axis('equal')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'method_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Create comparison dictionary
    comparison = {
        'basic': {
            'lap_time': basic_results['lap_time'],
            'metrics': basic_results.get('metrics', {}),
            'full_results': basic_results
        },
        'advanced': {
            'lap_time': advanced_results['lap_time'],
            'optimization_success': advanced_results.get('optimization_success', False),
            'optimization_time': advanced_results.get('optimization_time', 0),
            'full_results': advanced_results
        },
        'difference': {
            'lap_time_diff': basic_results['lap_time'] - advanced_results['lap_time'],
            'lap_time_percent': ((basic_results['lap_time'] - advanced_results['lap_time']) / 
                              basic_results['lap_time'] * 100)
        }
    }
    
    return comparison


# Example usage
if __name__ == "__main__":
    from ..core.vehicle import create_formula_student_vehicle
    import tempfile
    
    print("Lap Time Optimization Comparison")
    print("--------------------------------")
    
    # Create a Formula Student vehicle
    vehicle = create_formula_student_vehicle()
    
    # Create a temporary directory for outputs
    output_dir = tempfile.mkdtemp()
    print(f"Creating output directory: {output_dir}")
    
    # Create an example track
    track_file = os.path.join(output_dir, "example_track.yaml")
    print("Generating example track...")
    create_example_track(track_file, difficulty='easy')  # Use easy track for faster optimization
    
    # Create a simple configuration
    config = {
        'optimization': {
            'max_iterations': 20,     # Limit iterations for faster demonstration
            'num_control_points': 30  # Fewer control points for faster optimization
        }
    }
    
    # Write config to file
    config_file = os.path.join(output_dir, "lap_optimization.yaml")
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Run comparison
    print("\nComparing optimization methods (this may take a while)...")
    comparison = compare_optimization_methods(
        vehicle,
        track_file,
        config_file=config_file,
        include_thermal=True,
        save_dir=output_dir
    )
    
    # Print comparison results
    print("\nLap Time Comparison:")
    print(f"  Basic Simulation: {comparison['basic']['lap_time']:.3f}s")
    print(f"  Advanced Optimization: {comparison['advanced']['lap_time']:.3f}s")
    print(f"  Difference: {comparison['difference']['lap_time_diff']:.3f}s " +
          f"({comparison['difference']['lap_time_percent']:.2f}%)")
    
    # Print optimization details
    if comparison['advanced']['optimization_success']:
        print("\nOptimization Details:")
        print(f"  Success: {comparison['advanced']['optimization_success']}")
        print(f"  Optimization Time: {comparison['advanced']['optimization_time']:.1f}s")
    
    print(f"\nComparison results saved to: {output_dir}")