"""
Weight sensitivity analysis module for Formula Student powertrain simulation.

Analyzes the impact of vehicle mass changes on key performance metrics like
acceleration times and lap times.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Callable
import logging
import time
import copy # To avoid modifying the original vehicle

# Import core components
try:
    from ..core.vehicle import Vehicle
    from .acceleration import AccelerationSimulator, create_acceleration_simulator
    from .lap_time import LapTimeSimulator, create_lap_time_simulator
    from ..utils.plotting import plot_weight_sensitivity as plot_weight_sens_unified
    from ..utils.plotting import plot_weight_distribution_sensitivity as plot_dist_sens_unified
    from ..utils.plotting import save_plot, _apply_common_ax_settings
except ImportError:
    # Fallbacks
    class Vehicle: pass
    class AccelerationSimulator: pass
    class LapTimeSimulator: pass
    def create_acceleration_simulator(*args, **kwargs): return None
    def create_lap_time_simulator(*args, **kwargs): return None
    def plot_weight_sens_unified(*args, **kwargs): plt.figure(); plt.plot([0,1]); plt.title("Fallback Plot"); plt.show(); plt.close(); return plt.gcf()
    def plot_dist_sens_unified(*args, **kwargs): plt.figure(); plt.plot([0,1]); plt.title("Fallback Plot"); plt.show(); plt.close(); return plt.gcf()
    def save_plot(fig, path, **kwargs): pass
    def _apply_common_ax_settings(ax, **kwargs): pass
    logger = logging.getLogger("WeightSensitivity_Fallback")
    logger.warning("Could not import all necessary modules. Using fallbacks.")


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("WeightSensitivity")

class WeightSensitivityAnalyzer:
    """Analyzes sensitivity of vehicle performance to mass changes."""

    def __init__(self, base_vehicle: Vehicle):
        """
        Args:
            base_vehicle: The baseline Vehicle object to analyze.
        """
        if not isinstance(base_vehicle, Vehicle):
            if "MockVehicle" not in str(type(base_vehicle)):
                 raise TypeError("base_vehicle must be an instance of the Vehicle class.")
        self.base_vehicle = copy.deepcopy(base_vehicle) # Work on a copy
        self.base_weight_kg = self.base_vehicle.mass

        # Results storage
        self.analysis_results: Dict[str, Dict] = {} # Store results keyed by analysis type

        logger.info(f"WeightSensitivityAnalyzer initialized with base weight: {self.base_weight_kg:.1f} kg")

    def _run_simulation_at_weight(self, weight_kg: float, analysis_type: str, **kwargs) -> Optional[Dict]:
        """Helper function to run a specific simulation type at a given weight."""
        temp_vehicle = copy.deepcopy(self.base_vehicle)
        temp_vehicle.mass = weight_kg
        logger.debug(f" Running {analysis_type} simulation at {weight_kg:.1f} kg...")

        try:
            if analysis_type == 'acceleration':
                # Use pre-configured simulator if available, else create one
                if 'accel_simulator' not in self.analysis_results:
                     self.analysis_results['accel_simulator'] = create_acceleration_simulator(temp_vehicle)
                else:
                    # Update vehicle reference in existing simulator - IMPORTANT
                    self.analysis_results['accel_simulator'].vehicle = temp_vehicle

                simulator = self.analysis_results['accel_simulator']
                # Pass relevant kwargs
                sim_results = simulator.simulate_acceleration(
                    use_launch_control=kwargs.get('use_launch_control', True)
                    # Note: uses optimized shifts setting from simulator instance
                )
                metrics = simulator.analyze_performance_metrics(sim_results)
                return metrics # Return key metrics

            elif analysis_type == 'lap_time':
                track_file = kwargs.get('track_file')
                if not track_file: raise ValueError("track_file needed for lap time analysis")
                # Use pre-configured simulator if available, else create one
                if 'lap_simulator' not in self.analysis_results or \
                   self.analysis_results['lap_simulator'].track_profile.track_file != track_file:
                     self.analysis_results['lap_simulator'] = create_lap_time_simulator(temp_vehicle, track_file)
                else:
                     # Update vehicle reference
                     self.analysis_results['lap_simulator'].vehicle = temp_vehicle
                     # Re-init cornering calculator
                     self.analysis_results['lap_simulator'].cornering = CorneringPerformance(temp_vehicle)
                     # Reset profiles
                     self.analysis_results['lap_simulator'].speed_profile_mps = None


                simulator = self.analysis_results['lap_simulator']
                # Pass relevant kwargs
                lap_results = simulator.simulate_lap(
                     include_thermal=kwargs.get('include_thermal', True)
                )
                metrics = simulator.analyze_lap_performance(lap_results)
                return metrics # Return key metrics

            else:
                logger.error(f"Unsupported analysis_type: {analysis_type}")
                return None

        except Exception as e:
            logger.error(f"Error running {analysis_type} simulation at {weight_kg:.1f} kg: {e}", exc_info=False)
            return None # Return None on failure

    def analyze_acceleration_sensitivity(self,
                                  weight_range_kg: Tuple[float, float],
                                  num_points: int = 7, # More points for better curve
                                  use_launch_control: bool = True) -> Optional[Dict]:
        """
        Analyze acceleration sensitivity to weight.

        Args:
            weight_range_kg: Tuple (min_weight, max_weight).
            num_points: Number of weight points to test.
            use_launch_control: Use launch control during simulations.

        Returns:
            Dictionary with sensitivity results, or None if analysis fails.
        """
        logger.info(f"Analyzing Acceleration Sensitivity: Weight Range {weight_range_kg}, Points {num_points}")
        weight_points = np.linspace(weight_range_kg[0], weight_range_kg[1], num_points)
        results_list = []

        for weight in weight_points:
             metrics = self._run_simulation_at_weight(
                 weight, 'acceleration', use_launch_control=use_launch_control
             )
             if metrics:
                 results_list.append({'weight': weight, **metrics})
             else:
                 # If one sim fails, maybe skip the rest or record failure
                 logger.warning(f"Acceleration sim failed for weight {weight:.1f}kg. Skipping this point.")
                 results_list.append({'weight': weight}) # Add weight but no metrics

        if not any(r.get('finish_time') is not None for r in results_list):
             logger.error("Acceleration sensitivity analysis failed: No successful simulations.")
             return None

        # Store raw results
        self.analysis_results['acceleration'] = results_list

        # Calculate sensitivity coefficients
        sensitivities = self._calculate_sensitivities(results_list, ['finish_time', 'time_to_60mph', 'time_to_100kph'])
        self.analysis_results['acceleration_sensitivity'] = sensitivities

        logger.info("Acceleration sensitivity analysis complete.")
        for metric, data in sensitivities.items():
             logger.info(f"  {_format_metric_name(metric)} Sensitivity: {data['slope_per_kg']:.4f} s/kg ({data['slope_per_10kg']:.4f} s/10kg)")

        return self.analysis_results['acceleration_sensitivity'] # Return calculated sensitivities

    def analyze_lap_time_sensitivity(self,
                                  track_file: str,
                                  weight_range_kg: Tuple[float, float],
                                  num_points: int = 7,
                                  include_thermal: bool = True) -> Optional[Dict]:
        """
        Analyze lap time sensitivity to weight.

        Args:
            track_file: Path to track file.
            weight_range_kg: Tuple (min_weight, max_weight).
            num_points: Number of weight points to test.
            include_thermal: Include thermal effects in lap simulation.

        Returns:
            Dictionary with sensitivity results, or None if analysis fails.
        """
        logger.info(f"Analyzing Lap Time Sensitivity: Weight Range {weight_range_kg}, Points {num_points}")
        weight_points = np.linspace(weight_range_kg[0], weight_range_kg[1], num_points)
        results_list = []

        for weight in weight_points:
            metrics = self._run_simulation_at_weight(
                 weight, 'lap_time', track_file=track_file, include_thermal=include_thermal
             )
            if metrics and metrics.get('lap_time') is not None:
                 results_list.append({'weight': weight, **metrics})
            else:
                 logger.warning(f"Lap time sim failed for weight {weight:.1f}kg. Skipping this point.")
                 results_list.append({'weight': weight}) # Add weight but no metrics

        if not any(r.get('lap_time') is not None for r in results_list):
             logger.error("Lap time sensitivity analysis failed: No successful simulations.")
             return None

        # Store raw results
        self.analysis_results['lap_time'] = results_list

        # Calculate sensitivity coefficients
        sensitivities = self._calculate_sensitivities(results_list, ['lap_time', 'avg_speed_kph'])
        self.analysis_results['lap_time_sensitivity'] = sensitivities

        logger.info("Lap time sensitivity analysis complete.")
        for metric, data in sensitivities.items():
             unit = "s/kg" if "time" in metric else "kph/kg"
             unit10 = "s/10kg" if "time" in metric else "kph/10kg"
             logger.info(f"  {_format_metric_name(metric)} Sensitivity: {data['slope_per_kg']:.4f} {unit} ({data['slope_per_10kg']:.4f} {unit10})")

        return self.analysis_results['lap_time_sensitivity']


    def analyze_weight_distribution_sensitivity(self,
                                            track_file: str,
                                            distribution_range: Tuple[float, float] = (0.42, 0.52), # Front % range
                                            num_points: int = 5,
                                            include_thermal: bool = True) -> Optional[Dict]:
         """Analyze sensitivity to front weight distribution changes (keeping total mass constant)."""
         logger.info(f"Analyzing Weight Distribution Sensitivity: Front Range {distribution_range[0]:.1%} to {distribution_range[1]:.1%}")
         dist_points = np.linspace(distribution_range[0], distribution_range[1], num_points)
         results_list = []

         # Store original distribution
         original_dist = self.base_vehicle.weight_distribution_front

         for dist in dist_points:
             # --- Modify Vehicle for this Distribution ---
             temp_vehicle = copy.deepcopy(self.base_vehicle)
             temp_vehicle.weight_distribution_front = dist
             # Recalculate cornering limits if necessary (CorneringPerformance uses vehicle attrs)
             # This assumes the CorneringPerformance object is created fresh or updated
             temp_cornering = CorneringPerformance(temp_vehicle)

             # Recreate LapTimeSimulator with modified vehicle
             # This ensures the cornering object inside is updated
             temp_lap_simulator = create_lap_time_simulator(temp_vehicle, track_file)

             # Run lap simulation
             try:
                 lap_results = temp_lap_simulator.simulate_lap(include_thermal=include_thermal)
                 metrics = temp_lap_simulator.analyze_lap_performance(lap_results)
                 # Add lateral G from cornering analysis (e.g., max achieved during lap)
                 max_lat_g = np.max(np.abs(lap_results.get('lateral_g', [0])))
                 metrics['max_lateral_g'] = max_lat_g
                 results_list.append({'front_weight_pct': dist, **metrics})
                 logger.info(f" Front Weight: {dist:.1%}, Lap Time: {metrics['lap_time']:.3f}s, Max Lat G: {max_lat_g:.3f}")
             except Exception as e:
                 logger.warning(f"Lap sim failed for front weight {dist:.1%}: {e}. Skipping.")
                 results_list.append({'front_weight_pct': dist})

         # Restore original distribution on base vehicle
         self.base_vehicle.weight_distribution_front = original_dist

         if not any(r.get('lap_time') is not None for r in results_list):
              logger.error("Weight distribution sensitivity analysis failed: No successful simulations.")
              return None

         # Store raw results
         self.analysis_results['distribution'] = results_list

         # Calculate sensitivities (note: sensitivity here is performance / % distribution change)
         # Quadratic fit might be more appropriate than linear for distribution effects
         sensitivities = self._calculate_sensitivities(results_list, ['lap_time', 'max_lateral_g'], x_key='front_weight_pct', fit_degree=2)
         self.analysis_results['distribution_sensitivity'] = sensitivities

         logger.info("Weight distribution sensitivity analysis complete.")
         # Optimal values are more relevant than linear slopes here
         if 'lap_time' in sensitivities and 'optimal_x' in sensitivities['lap_time']:
              logger.info(f" Optimal Front Weight % for Lap Time: {sensitivities['lap_time']['optimal_x']*100:.1f}%")
         if 'max_lateral_g' in sensitivities and 'optimal_x' in sensitivities['max_lateral_g']:
              logger.info(f" Optimal Front Weight % for Lateral G: {sensitivities['max_lateral_g']['optimal_x']*100:.1f}%")

         return self.analysis_results['distribution_sensitivity']


    def _calculate_sensitivities(self, results_list: List[Dict], metrics: List[str], x_key: str = 'weight', fit_degree: int = 1) -> Dict:
        """Helper to calculate sensitivity coefficients using polyfit."""
        sensitivities = {}
        if not results_list: return sensitivities

        x_values = np.array([r.get(x_key) for r in results_list])

        for metric in metrics:
            y_values = np.array([r.get(metric) for r in results_list])

            # Filter out None/NaN values for fitting
            valid_mask = np.isfinite(x_values) & np.isfinite(y_values)
            if np.sum(valid_mask) < fit_degree + 1:
                logger.warning(f"Insufficient valid data points ({np.sum(valid_mask)}) to calculate sensitivity for '{metric}' with degree {fit_degree}.")
                sensitivities[metric] = {'coeffs': None, 'poly': None}
                continue

            x_valid = x_values[valid_mask]
            y_valid = y_values[valid_mask]

            try:
                coeffs = np.polyfit(x_valid, y_valid, fit_degree)
                poly = np.poly1d(coeffs)
                sensitivities[metric] = {'coeffs': coeffs.tolist(), 'poly': poly}

                # For linear fit, extract slope and per-10kg value
                if fit_degree == 1:
                     slope = coeffs[0]
                     sensitivities[metric]['slope_per_kg'] = slope
                     sensitivities[metric]['slope_per_10kg'] = slope * 10
                     # Calculate % improvement if baseline is valid
                     if y_valid[0] != 0:
                          sensitivities[metric]['percent_per_10kg'] = abs(slope * 10 / y_valid[0]) * 100 * (-1 if slope > 0 else 1) # Improvement implies negative slope for time

                # For quadratic fit, find optimum if applicable
                elif fit_degree == 2:
                     # Optimum at x = -b / 2a
                     if abs(coeffs[0]) > 1e-6: # Avoid division by zero
                          optimal_x = -coeffs[1] / (2 * coeffs[0])
                          sensitivities[metric]['optimal_x'] = optimal_x
                          sensitivities[metric]['optimal_y'] = poly(optimal_x)
                          # Determine if optimum is min or max
                          sensitivities[metric]['optimum_type'] = 'minimum' if coeffs[0] > 0 else 'maximum'


            except Exception as e:
                logger.error(f"Failed to calculate sensitivity for '{metric}': {e}")
                sensitivities[metric] = {'coeffs': None, 'poly': None}

        return sensitivities


    def calculate_weight_reduction_targets(self,
                                         performance_target: float,
                                         metric_name: str = 'finish_time') -> Optional[Dict]:
        """
        Estimate required weight reduction to meet a performance target, based on linear sensitivity.

        Args:
            performance_target: The desired performance value (e.g., 4.0 seconds for 75m).
            metric_name: The performance metric key (e.g., 'finish_time', 'lap_time').

        Returns:
            Dictionary with target analysis, or None if analysis not possible.
        """
        sensitivity_key = None
        analysis_data = None
        if metric_name in ['finish_time', 'time_to_60mph', 'time_to_100kph']:
            sensitivity_key = 'acceleration_sensitivity'
            analysis_data = self.analysis_results.get('acceleration')
        elif metric_name == 'lap_time':
            sensitivity_key = 'lap_time_sensitivity'
            analysis_data = self.analysis_results.get('lap_time')
        else:
            logger.error(f"Weight reduction targets not implemented for metric: {metric_name}")
            return None

        if sensitivity_key not in self.analysis_results or not analysis_data:
            logger.error(f"Sensitivity analysis for '{metric_name}' not performed yet.")
            return None

        sensitivity_coeffs = self.analysis_results[sensitivity_key]
        if metric_name not in sensitivity_coeffs or sensitivity_coeffs[metric_name]['coeffs'] is None or len(sensitivity_coeffs[metric_name]['coeffs']) != 2:
             logger.error(f"Linear sensitivity coefficient not available for '{metric_name}'. Cannot calculate targets.")
             return None

        # Use linear sensitivity (slope)
        sensitivity = sensitivity_coeffs[metric_name]['slope_per_kg']
        if abs(sensitivity) < 1e-6:
             logger.warning(f"Sensitivity for '{metric_name}' is near zero. Target may be unachievable via weight reduction alone.")
             return None

        # Find current performance at base weight
        current_performance = None
        for result in analysis_data:
            if abs(result['weight'] - self.base_weight_kg) < 1e-3:
                 current_performance = result.get(metric_name)
                 break
        if current_performance is None:
             # Interpolate if exact base weight wasn't simulated
             poly = sensitivity_coeffs[metric_name]['poly']
             if poly: current_performance = poly(self.base_weight_kg)
        if current_performance is None:
             logger.error(f"Could not determine current performance for '{metric_name}' at base weight.")
             return None

        # Calculate required change and weight reduction
        required_improvement = current_performance - performance_target
        # Weight reduction needed = improvement / (-sensitivity) because lower weight improves time (negative slope)
        required_reduction_kg = required_improvement / (-sensitivity)
        target_weight_kg = self.base_weight_kg - required_reduction_kg

        # Check feasibility (e.g., cannot be less than driver weight + minimum chassis)
        min_feasible_weight = 80 # Example minimum feasible weight
        is_achievable = target_weight_kg >= min_feasible_weight

        result = {
            'metric_name': _format_metric_name(metric_name),
            'current_performance': current_performance,
            'target_performance': performance_target,
            'required_improvement': required_improvement,
            'sensitivity_per_kg': sensitivity,
            'current_weight_kg': self.base_weight_kg,
            'required_reduction_kg': required_reduction_kg,
            'target_weight_kg': target_weight_kg,
            'is_achievable': is_achievable
        }

        logger.info(f"Target Calculation for {result['metric_name']}:")
        logger.info(f" Current: {current_performance:.3f} -> Target: {performance_target:.3f}")
        logger.info(f" Requires {required_reduction_kg:.2f} kg reduction.")
        logger.info(f" Target Weight: {target_weight_kg:.2f} kg ({'Achievable' if is_achievable else 'Unlikely'})")

        return result

    # --- Plotting Wrappers ---
    def plot_weight_sensitivity_curves(self, save_path: Optional[str] = None):
         """Plot sensitivity curves using the unified plotter."""
         plot_data = {'weights': [], 'lap_times': [], 'acceleration_times': [], 'zero_to_sixty': []}
         if 'acceleration' in self.analysis_results:
              plot_data['weights'] = [r['weight'] for r in self.analysis_results['acceleration']]
              plot_data['acceleration_times'] = [r.get('finish_time') for r in self.analysis_results['acceleration']] # Use 75m time
              plot_data['zero_to_sixty'] = [r.get('time_to_60mph') for r in self.analysis_results['acceleration']]
         if 'lap_time' in self.analysis_results:
              # Ensure weights match if both analyses run
              if not plot_data['weights']: plot_data['weights'] = [r['weight'] for r in self.analysis_results['lap_time']]
              plot_data['lap_times'] = [r.get('lap_time') for r in self.analysis_results['lap_time']]

         fig = plot_weight_sens_unified(plot_data, save_path=save_path)
         # if fig: plt.close(fig)

    def plot_weight_distribution_sensitivity(self, save_path: Optional[str] = None):
         """Plot distribution sensitivity using the unified plotter."""
         if 'distribution' not in self.analysis_results:
              logger.error("Weight distribution analysis not performed yet.")
              return
         plot_data = {'front_weight_pct': [], 'lap_times': [], 'acceleration_times': [], 'lateral_acceleration': []}
         for r in self.analysis_results['distribution']:
              plot_data['front_weight_pct'].append(r.get('front_weight_pct'))
              plot_data['lap_times'].append(r.get('lap_time'))
              # Note: Need to run accel analysis at each distribution for this data
              # plot_data['acceleration_times'].append(r.get('finish_time'))
              plot_data['lateral_acceleration'].append(r.get('max_lateral_g'))

         fig = plot_dist_sens_unified(plot_data, save_path=save_path)
         # if fig: plt.close(fig)

    def generate_weight_sensitivity_report(self, save_dir: Optional[str] = None) -> Dict:
        """Generate a report summarizing sensitivity findings."""
        if not self.analysis_results:
            logger.error("No analysis results available to generate report.")
            return {}

        if save_dir: os.makedirs(save_dir, exist_ok=True)

        report = {'base_vehicle': self.base_vehicle.get_vehicle_specs()}

        # Add sensitivity summaries
        if 'acceleration_sensitivity' in self.analysis_results:
             report['acceleration_sensitivity'] = {k: v for k, v in self.analysis_results['acceleration_sensitivity'].items() if k != 'poly'}
        if 'lap_time_sensitivity' in self.analysis_results:
             report['lap_time_sensitivity'] = {k: v for k, v in self.analysis_results['lap_time_sensitivity'].items() if k != 'poly'}
        if 'distribution_sensitivity' in self.analysis_results:
             report['distribution_sensitivity'] = {k: v for k, v in self.analysis_results['distribution_sensitivity'].items() if k != 'poly'}

        # Generate and save plots
        if save_dir:
             if 'acceleration' in self.analysis_results or 'lap_time' in self.analysis_results:
                  self.plot_weight_sensitivity_curves(save_path=os.path.join(save_dir, "weight_sensitivity_curves.png"))
             if 'distribution' in self.analysis_results:
                  self.plot_weight_distribution_sensitivity(save_path=os.path.join(save_dir, "weight_distribution_sensitivity.png"))

             # Save summary report to JSON
             report_path = os.path.join(save_dir, "weight_sensitivity_report.json")
             try:
                 with open(report_path, 'w') as f:
                     # Custom encoder for numpy types if needed
                     json.dump(report, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x))
                 logger.info(f"Weight sensitivity report saved to {report_path}")
             except Exception as e:
                 logger.error(f"Failed to save weight sensitivity report: {e}")

        return report


# --- Standalone Runner Function ---
def analyze_weight_sensitivity(vehicle: Vehicle, track_file: str,
                             weight_range_kg: Tuple[float, float] = (180, 280), # Wider default range
                             distribution_range: Tuple[float, float] = (0.43, 0.53),
                             num_points: int = 7,
                             save_dir: Optional[str] = None) -> Dict:
    """
    Convenience function to run full weight sensitivity analysis.

    Args:
        vehicle: Base vehicle model.
        track_file: Path to the track file for lap time sims.
        weight_range_kg: Min/max total mass range.
        distribution_range: Min/max front weight % range.
        num_points: Number of points for each analysis dimension.
        save_dir: Directory to save results and plots.

    Returns:
        Dictionary containing the full analysis report.
    """
    try:
        analyzer = WeightSensitivityAnalyzer(vehicle)

        # Run analyses
        analyzer.analyze_acceleration_sensitivity(weight_range_kg, num_points)
        analyzer.analyze_lap_time_sensitivity(track_file, weight_range_kg, num_points)
        analyzer.analyze_weight_distribution_sensitivity(track_file, distribution_range, num_points)

        # Generate report
        report = analyzer.generate_weight_sensitivity_report(save_dir)
        return report

    except Exception as e:
        logger.error(f"Error during weight sensitivity analysis: {e}", exc_info=True)
        return {'error': str(e)}


# Example Usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        from ..core.vehicle import create_formula_student_vehicle
        from .lap_time import create_example_track # Use lap_time's version
        import tempfile

        print("Weight Sensitivity Analysis Demo")
        print("-" * 30)

        vehicle = create_formula_student_vehicle()
        output_dir = tempfile.mkdtemp()
        track_file = os.path.join(output_dir, "sensitivity_track.yaml")
        create_example_track(track_file, difficulty='easy') # Easier track for faster sims

        print(f"Output directory: {output_dir}")
        print(f"Track file: {track_file}")

        # Define analysis ranges
        w_range = (vehicle.mass - 25, vehicle.mass + 25) # +/- 25kg
        dist_range = (vehicle.weight_distribution_front - 0.03, vehicle.weight_distribution_front + 0.03) # +/- 3%

        # Run full analysis
        report = analyze_weight_sensitivity(
            vehicle,
            track_file,
            weight_range_kg=w_range,
            distribution_range=dist_range,
            num_points=5, # Fewer points for faster demo
            save_dir=output_dir
        )

        if 'error' not in report:
            print("\n--- Analysis Summary ---")
            if 'acceleration_sensitivity' in report:
                print(f" Accel (75m) Sensitivity: {report['acceleration_sensitivity']['sensitivity_75m']:.4f} s/kg")
            if 'lap_time_sensitivity' in report:
                print(f" Lap Time Sensitivity:    {report['lap_time_sensitivity']['sensitivity_lap_time']:.4f} s/kg")
            if 'distribution_sensitivity' in report:
                 lap_opt = report['distribution_sensitivity'].get('lap_time',{}).get('optimal_x')
                 lat_opt = report['distribution_sensitivity'].get('max_lateral_g',{}).get('optimal_x')
                 print(f" Optimal Weight Dist (Lap): {lap_opt*100:.1f}%" if lap_opt else " Optimal Weight Dist (Lap): N/A")
                 print(f" Optimal Weight Dist (LatG): {lat_opt*100:.1f}%" if lat_opt else " Optimal Weight Dist (LatG): N/A")

            print(f"\nDetailed report and plots saved to: {output_dir}")
        else:
             print(f"\nAnalysis failed: {report['error']}")


    except ImportError as e:
        print(f"\nError: Could not import necessary modules ({e}).")
    except FileNotFoundError as e:
         print(f"\nError: Configuration file not found. {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()