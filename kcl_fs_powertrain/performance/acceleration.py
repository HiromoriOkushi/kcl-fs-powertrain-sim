"""
Acceleration performance simulation and analysis for Formula Student.

Simulates the 75m acceleration event, optimizes launch control,
and analyzes performance metrics.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Callable
import logging
import time
from scipy.optimize import minimize_scalar

# Import core components
try:
    from ..core.vehicle import Vehicle
    from ..transmission.cas_system import ShiftDirection # If needed for shift control
    from ..utils.constants import FS_ACCELERATION_LENGTH, MS_TO_MPH, MS_TO_KMH
    from ..utils.plotting import plot_acceleration_results as plot_accel_unified # Use unified plotter
    from ..utils.plotting import plot_acceleration_comparison as plot_accel_comp_unified
    from ..utils.plotting import save_plot, set_plot_style, _apply_common_ax_settings
except ImportError:
    # Fallbacks for standalone execution or testing
    FS_ACCELERATION_LENGTH = 75.0
    MS_TO_MPH = 2.23694
    MS_TO_KMH = 3.6
    class Vehicle: pass
    class ShiftDirection: UP = 1; DOWN = -1
    def plot_accel_unified(*args, **kwargs): plt.figure(); plt.plot([0,1],[0,1]); plt.title("Fallback Plot"); plt.show(); plt.close()
    def plot_accel_comp_unified(*args, **kwargs): plt.figure(); plt.plot([0,1],[0,1]); plt.title("Fallback Comparison Plot"); plt.show(); plt.close()
    def save_plot(fig, path, **kwargs): pass
    def set_plot_style(style): pass
    def _apply_common_ax_settings(ax, **kwargs): pass
    logger = logging.getLogger("AccelerationPerformance_Fallback")
    logger.warning("Could not import all necessary modules. Using fallbacks.")


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("AccelerationPerformance")


class AccelerationSimulator:
    """Simulates Formula Student acceleration events (e.g., 75m)."""

    def __init__(self, vehicle: Vehicle):
        """
        Args:
            vehicle: Vehicle model instance.
        """
        if not isinstance(vehicle, Vehicle):
             # Check if the placeholder was imported
             if "MockVehicle" not in str(type(vehicle)): # Allow mock/placeholder for testing
                 raise TypeError("vehicle must be an instance of the Vehicle class.")
        self.vehicle = vehicle

        # Default simulation parameters
        self.distance_m: float = FS_ACCELERATION_LENGTH
        self.time_step_s: float = 0.005 # Smaller time step for accuracy
        self.max_time_s: float = 10.0

        # Launch control parameters
        self.launch_rpm: float = 9000.0 # Default launch RPM
        self.launch_slip_target: float = 0.18 # Default target wheel slip
        self.launch_duration_s: float = 0.6 # Duration of launch control phase
        self.use_traction_control: bool = True # Simple TC during launch

        # Shift strategy parameters
        self.use_optimized_shifts: bool = True
        self.shift_rpm_offset: float = -200 # RPM relative to redline for shift trigger

        # Results cache
        self.results_cache: Dict[str, Dict] = {}

        logger.info("Acceleration simulator initialized.")

    def configure(self, distance_m: Optional[float] = None,
                time_step_s: Optional[float] = None,
                max_time_s: Optional[float] = None):
        """Configure basic simulation parameters."""
        if distance_m is not None: self.distance_m = distance_m
        if time_step_s is not None: self.time_step_s = time_step_s
        if max_time_s is not None: self.max_time_s = max_time_s
        logger.info(f"Simulator configured: Distance={self.distance_m}m, TimeStep={self.time_step_s}s, MaxTime={self.max_time_s}s")

    def configure_launch_control(self, launch_rpm: Optional[float] = None,
                              launch_slip_target: Optional[float] = None,
                              launch_duration_s: Optional[float] = None,
                              use_traction_control: Optional[bool] = None):
        """Configure launch control parameters."""
        if launch_rpm is not None: self.launch_rpm = launch_rpm
        if launch_slip_target is not None: self.launch_slip_target = np.clip(launch_slip_target, 0.05, 0.4) # Sensible range
        if launch_duration_s is not None: self.launch_duration_s = launch_duration_s
        if use_traction_control is not None: self.use_traction_control = use_traction_control
        logger.info(f"Launch control configured: RPM={self.launch_rpm:.0f}, Slip={self.launch_slip_target:.2f}, Duration={self.launch_duration_s:.1f}s, TC={'On' if self.use_traction_control else 'Off'}")

    def configure_shifting(self, use_optimized: bool, rpm_offset: Optional[float] = None):
         """Configure shifting strategy."""
         self.use_optimized_shifts = use_optimized
         if rpm_offset is not None: self.shift_rpm_offset = rpm_offset
         logger.info(f"Shifting configured: Optimized={self.use_optimized_shifts}, Offset={self.shift_rpm_offset} RPM")

    def _get_shift_rpm(self, gear: int) -> float:
        """Determine the RPM to upshift from the given gear."""
        if self.use_optimized_shifts:
            # Try to use vehicle's optimized points if available
            if hasattr(self.vehicle, 'get_optimal_shift_point') and callable(self.vehicle.get_optimal_shift_point):
                 optimal_rpm = self.vehicle.get_optimal_shift_point(gear)
                 if optimal_rpm: return optimal_rpm
            # Fallback if vehicle method doesn't exist or fails
            logger.debug(f"Using calculated optimal shift points for gear {gear}.")
            # This requires the DrivetrainSystem to have the optimization method
            if hasattr(self.vehicle.drivetrain, 'calculate_optimal_shift_points') and self.vehicle.engine:
                # Need engine torque curve representation for drivetrain method
                # For now, use simpler redline offset method
                 return self.vehicle.engine.redline_rpm + self.shift_rpm_offset # Use configured offset
            else:
                 return self.vehicle.engine.redline_rpm * 0.97 # Fallback if optimization unavailable
        else:
            # Fixed offset from redline
            return self.vehicle.engine.redline_rpm + self.shift_rpm_offset

    def simulate_acceleration(self, use_launch_control: bool = True) -> Dict:
        """
        Simulate the acceleration run.

        Args:
            use_launch_control: Override instance setting for this run.

        Returns:
            Dictionary with simulation results (time series and summary metrics).
        """
        # --- Prepare Cache Key ---
        # Key should include relevant varying parameters
        cache_key_parts = [
            f"dist{self.distance_m}",
            f"ts{self.time_step_s}",
            f"lc{use_launch_control}",
            f"lcRpm{self.launch_rpm if use_launch_control else 0}",
            f"lcSlip{self.launch_slip_target if use_launch_control else 0}",
            f"lcDur{self.launch_duration_s if use_launch_control else 0}",
            f"optSh{self.use_optimized_shifts}",
            f"shOff{self.shift_rpm_offset}",
            f"mass{self.vehicle.mass:.1f}" # Include mass
        ]
        cache_key = "_".join(cache_key_parts)

        if cache_key in self.results_cache:
            logger.info("Using cached acceleration results for key: %s", cache_key)
            return self.results_cache[cache_key]

        logger.info(f"Simulating acceleration: LC={'On' if use_launch_control else 'Off'}, Shifts={'Optimized' if self.use_optimized_shifts else 'Fixed Offset'}")

        # --- Reset Vehicle State ---
        self.vehicle.current_speed = 0.0
        self.vehicle.current_position = 0.0
        self.vehicle.current_acceleration = 0.0
        self.vehicle.change_gear(1) # Start in first gear
        self.vehicle.current_engine_rpm = self.vehicle.engine.idle_rpm # Start at idle

        # --- Simulation Loop ---
        time_s = 0.0
        dt = self.time_step_s
        in_launch_phase = use_launch_control

        # History lists
        history = {
            'time': [0.0], 'speed': [0.0], 'position': [0.0], 'acceleration': [0.0],
            'engine_rpm': [self.vehicle.current_engine_rpm], 'gear': [1], 'wheel_slip': [0.0],
            'throttle_effective': [0.0]
        }

        # Get number of gears
        num_gears = self.vehicle.drivetrain.num_gears

        while time_s < self.max_time_s and self.vehicle.current_position < self.distance_m:
            # --- Determine Controls ---
            throttle_request = 1.0 # Driver wants full throttle
            brake_request = 0.0
            effective_throttle = throttle_request # Actual throttle applied after LC/TC

            # Launch Control Logic
            if in_launch_phase and time_s < self.launch_duration_s:
                # Option 1: RPM Limit
                # if self.vehicle.current_engine_rpm > self.launch_rpm * 1.02:
                #     effective_throttle *= 0.8 # Reduce throttle slightly if over limit
                # elif self.vehicle.current_engine_rpm < self.launch_rpm * 0.98:
                #      effective_throttle = 1.0 # Allow full throttle if below target

                # Option 2: Slip Target (more realistic but complex)
                # Calculate current wheel slip
                wheel_angular_vel_rad_s = self.vehicle.current_speed / self.vehicle.wheel_radius_m if self.vehicle.wheel_radius_m > 0 else 0
                engine_angular_vel_rad_s = self.vehicle.current_engine_rpm * (2 * np.pi / 60)
                overall_ratio = self.vehicle.drivetrain.get_overall_ratio()
                driven_wheel_angular_vel_rad_s = engine_angular_vel_rad_s / overall_ratio if overall_ratio > 0 else 0

                # Slip Ratio = (WheelSpeed_FromEngine - WheelSpeed_Ground) / WheelSpeed_Ground (or Engine for low speed)
                denominator = max(driven_wheel_angular_vel_rad_s, wheel_angular_vel_rad_s, 0.1) # Avoid division by zero
                wheel_slip = max(0, (driven_wheel_angular_vel_rad_s - wheel_angular_vel_rad_s) / denominator)

                if self.use_traction_control:
                     # Simple P-controller for slip
                     slip_error = wheel_slip - self.launch_slip_target
                     throttle_reduction = np.clip(slip_error * 5.0, 0.0, 0.7) # Proportional reduction (gain=5), max 70% cut
                     effective_throttle = throttle_request * (1.0 - throttle_reduction)

            else:
                in_launch_phase = False
                wheel_slip = 0.0 # Simplified assumption after launch

            # --- Shifting Logic ---
            current_gear = self.vehicle.get_current_gear()
            if current_gear > 0 and current_gear < num_gears:
                 shift_trigger_rpm = self._get_shift_rpm(current_gear)
                 if self.vehicle.current_engine_rpm >= shift_trigger_rpm:
                      # Initiate Upshift
                      success = self.vehicle.change_gear(current_gear + 1)
                      if success:
                           logger.debug(f"Shift {current_gear}->{current_gear+1} triggered at {self.vehicle.current_engine_rpm:.0f} RPM")
                           # Optional: Add shift time penalty here if CASSystem not integrated
                           # time_s += SHIFT_PENALTY
                           # effective_throttle = 0 # During shift

            # --- Update Vehicle Physics ---
            # Calculate acceleration with effective throttle
            current_accel = self.vehicle.calculate_acceleration(
                throttle=effective_throttle,
                brake=brake_request
            )
            # Update vehicle state (speed, position, RPM) using simple Euler integration
            self.vehicle.current_speed += current_accel * dt
            self.vehicle.current_speed = max(0.0, self.vehicle.current_speed) # No negative speed
            self.vehicle.current_position += self.vehicle.current_speed * dt
            # Update RPM based on new speed and current gear
            if self.vehicle.get_current_gear() > 0:
                 self.vehicle.current_engine_rpm = self.vehicle.drivetrain.calculate_engine_speed_rpm(
                     self.vehicle.current_speed, self.vehicle.get_current_gear()
                 )
                 # Clamp RPM just in case
                 self.vehicle.current_engine_rpm = np.clip(self.vehicle.current_engine_rpm, self.vehicle.engine.idle_rpm, self.vehicle.engine.redline_rpm)
            else:
                 # Allow RPM to drop if clutch disengaged (neutral) - simplified
                 self.vehicle.current_engine_rpm = max(self.vehicle.engine.idle_rpm, self.vehicle.current_engine_rpm - 500*dt)


            # Update simulation time
            time_s += dt

            # --- Store History ---
            history['time'].append(time_s)
            history['speed'].append(self.vehicle.current_speed)
            history['position'].append(self.vehicle.current_position)
            history['acceleration'].append(current_accel)
            history['engine_rpm'].append(self.vehicle.current_engine_rpm)
            history['gear'].append(self.vehicle.get_current_gear())
            history['wheel_slip'].append(wheel_slip)
            history['throttle_effective'].append(effective_throttle)

        # --- Post-processing ---
        logger.info(f"Simulation loop finished at time {time_s:.3f}s, position {self.vehicle.current_position:.2f}m")
        results = {key: np.array(val) for key, val in history.items()} # Convert lists to numpy arrays

        # Interpolate finish time and speed
        finish_time = None
        finish_speed = None
        if results['position'][-1] >= self.distance_m:
            try:
                # Find index where distance first crosses the target
                finish_idx = np.argmax(results['position'] >= self.distance_m)
                if finish_idx > 0:
                     # Interpolate between finish_idx-1 and finish_idx
                     t_before, t_after = results['time'][finish_idx-1], results['time'][finish_idx]
                     p_before, p_after = results['position'][finish_idx-1], results['position'][finish_idx]
                     s_before, s_after = results['speed'][finish_idx-1], results['speed'][finish_idx]

                     if p_after > p_before: # Avoid division by zero
                         interp_factor = (self.distance_m - p_before) / (p_after - p_before)
                         finish_time = t_before + interp_factor * (t_after - t_before)
                         finish_speed = s_before + interp_factor * (s_after - s_before)
                     else: # Landed exactly on the point
                         finish_time = t_after
                         finish_speed = s_after

                else: # Crossed at the very first step (unlikely)
                    finish_time = results['time'][finish_idx]
                    finish_speed = results['speed'][finish_idx]
            except Exception as e:
                 logger.error(f"Error during finish time interpolation: {e}")

        results['finish_time'] = finish_time
        results['finish_speed'] = finish_speed

        # Calculate 0-60 mph and 0-100 kph times
        def time_to_speed(target_speed_mps):
            if results['speed'][-1] < target_speed_mps: return None # Didn't reach target speed
            try:
                target_idx = np.argmax(results['speed'] >= target_speed_mps)
                if target_idx == 0: return 0.0 # Reached instantly (or started above)
                t_before, t_after = results['time'][target_idx-1], results['time'][target_idx]
                s_before, s_after = results['speed'][target_idx-1], results['speed'][target_idx]
                if s_after > s_before: # Avoid division by zero
                    interp_factor = (target_speed_mps - s_before) / (s_after - s_before)
                    return t_before + interp_factor * (t_after - t_before)
                else:
                     return t_after # Landed exactly
            except Exception as e:
                 logger.error(f"Error interpolating time to speed {target_speed_mps:.1f} m/s: {e}")
                 return None

        results['time_to_60mph'] = time_to_speed(60.0 / MS_TO_MPH)
        results['time_to_100kph'] = time_to_speed(100.0 / MS_TO_KMH)

        # Add metadata to results
        results['distance_m'] = self.distance_m
        results['used_launch_control'] = use_launch_control
        results['used_optimized_shifts'] = self.use_optimized_shifts
        results['config'] = {
            'launch_rpm': self.launch_rpm,
            'launch_slip_target': self.launch_slip_target,
            'launch_duration_s': self.launch_duration_s,
            'shift_rpm_offset': self.shift_rpm_offset
        }

        # Cache and return
        self.results_cache[cache_key] = results
        logger.info(f"Accel sim complete. Time: {results.get('finish_time', -1):.3f}s, "
                   f"0-60mph: {results.get('time_to_60mph', -1):.3f}s, "
                   f"0-100kph: {results.get('time_to_100kph', -1):.3f}s")
        return results


    def optimize_launch_control(self,
                             rpm_bounds: Tuple[float, float] = (7000, 11000),
                             slip_bounds: Tuple[float, float] = (0.10, 0.30),
                             duration_bounds: Tuple[float, float] = (0.3, 1.0),
                             max_evals: int = 15) -> Dict:
        """
        Optimize launch control parameters (RPM, Slip Target, Duration) for minimum 75m time.
        Uses a simple scalar optimization approach for each parameter sequentially.

        Args:
            rpm_bounds: Min/max RPM to test.
            slip_bounds: Min/max slip target to test.
            duration_bounds: Min/max launch phase duration to test.
            max_evals: Maximum function evaluations per parameter optimization.

        Returns:
            Dictionary with optimal parameters and best time achieved.
        """
        logger.info("Optimizing Launch Control parameters...")

        # Store initial settings
        initial_rpm = self.launch_rpm
        initial_slip = self.launch_slip_target
        initial_duration = self.launch_duration_s

        # --- Objective Function ---
        def objective(value: float, param_to_optimize: str) -> float:
            # Set the parameter being optimized
            if param_to_optimize == 'rpm': self.configure_launch_control(launch_rpm=value)
            elif param_to_optimize == 'slip': self.configure_launch_control(launch_slip_target=value)
            elif param_to_optimize == 'duration': self.configure_launch_control(launch_duration_s=value)

            # Run simulation
            results = self.simulate_acceleration(use_launch_control=True)
            finish_time = results.get('finish_time')

            # Return finish time (or large penalty if failed/DNF)
            return finish_time if finish_time is not None else self.max_time_s * 2

        # --- Optimize Parameters Sequentially ---
        best_params = {'rpm': initial_rpm, 'slip': initial_slip, 'duration': initial_duration}
        optimization_options = {'maxiter': max_evals, 'xatol': 50 if 'rpm' else 0.01} # Tolerance based on param

        # Optimize RPM
        logger.debug(f"Optimizing Launch RPM (Bounds: {rpm_bounds})...")
        res_rpm = minimize_scalar(lambda r: objective(r, 'rpm'), bounds=rpm_bounds, method='bounded', options=optimization_options)
        if res_rpm.success: best_params['rpm'] = res_rpm.x
        self.configure_launch_control(launch_rpm=best_params['rpm']) # Update simulator

        # Optimize Slip Target
        logger.debug(f"Optimizing Slip Target (Bounds: {slip_bounds})...")
        res_slip = minimize_scalar(lambda s: objective(s, 'slip'), bounds=slip_bounds, method='bounded', options=optimization_options)
        if res_slip.success: best_params['slip'] = res_slip.x
        self.configure_launch_control(launch_slip_target=best_params['slip']) # Update simulator

        # Optimize Duration
        logger.debug(f"Optimizing Duration (Bounds: {duration_bounds})...")
        res_dur = minimize_scalar(lambda d: objective(d, 'duration'), bounds=duration_bounds, method='bounded', options=optimization_options)
        if res_dur.success: best_params['duration'] = res_dur.x
        self.configure_launch_control(launch_duration_s=best_params['duration']) # Update simulator

        # Run final simulation with best parameters
        final_results = self.simulate_acceleration(use_launch_control=True)
        best_time = final_results.get('finish_time')

        logger.info("Launch control optimization complete.")
        logger.info(f"  Optimal RPM: {best_params['rpm']:.0f}")
        logger.info(f"  Optimal Slip Target: {best_params['slip']:.3f}")
        logger.info(f"  Optimal Duration: {best_params['duration']:.2f} s")
        logger.info(f"  Best 75m Time Achieved: {best_time:.3f} s" if best_time else "  Best time not found.")

        # Restore initial settings? Or keep optimized? Keep optimized for now.
        # self.configure_launch_control(initial_rpm, initial_slip, initial_duration)

        return {
            'optimal_rpm': best_params['rpm'],
            'optimal_slip_target': best_params['slip'],
            'optimal_duration_s': best_params['duration'],
            'best_finish_time_s': best_time,
            'final_run_results': final_results
        }


    def analyze_performance_metrics(self, results: Dict) -> Dict:
        """Calculate and analyze standard acceleration performance metrics."""
        time = results.get('time', np.array([]))
        speed = results.get('speed', np.array([]))
        accel = results.get('acceleration', np.array([]))
        finish_time = results.get('finish_time')

        metrics = {
            'finish_time': finish_time,
            'finish_speed_mps': results.get('finish_speed'),
            'time_to_60mph': results.get('time_to_60mph'),
            'time_to_100kph': results.get('time_to_100kph'),
            'peak_acceleration_mpss': np.max(accel) if len(accel) > 0 else None,
            'avg_acceleration_mpss': None,
            'performance_grade': 'N/A'
        }

        if metrics['peak_acceleration_mpss'] is not None:
            metrics['peak_acceleration_g'] = metrics['peak_acceleration_mpss'] / 9.81

        if finish_time is not None and finish_time > 0 and metrics['finish_speed_mps'] is not None:
            metrics['avg_acceleration_mpss'] = metrics['finish_speed_mps'] / finish_time

        # Grade based on 75m time
        if finish_time is not None:
             if finish_time < 3.8: grade = 'A+'
             elif finish_time < 4.0: grade = 'A'
             elif finish_time < 4.2: grade = 'B+'
             elif finish_time < 4.4: grade = 'B'
             elif finish_time < 4.7: grade = 'C'
             elif finish_time < 5.0: grade = 'D'
             else: grade = 'F'
             metrics['performance_grade'] = grade

        return metrics


    def plot_acceleration_results(self, results: Dict, save_path: Optional[str] = None,
                                plot_wheel_slip: bool = False):
        """Plot acceleration results using the unified plotting function."""
        # Prepare data in the format expected by the unified plotting function
        plot_data = results.copy() # Start with all results
        # The unified function handles unit conversion if needed by other plots, but accel plots usually use specific units.
        # We can add specific metrics here if the unified plot needs them structured differently.
        # plot_data['title_override'] = f"Acceleration Run ({results['distance_m']}m)"

        fig = plot_accel_unified(plot_data, save_path=save_path, plot_wheel_slip=plot_wheel_slip)
        # Optional: Close figure after saving/showing
        # if fig: plt.close(fig)


    def plot_acceleration_comparison(self, results_list: List[Dict], labels: List[str],
                                  save_path: Optional[str] = None):
        """Plot comparison of multiple acceleration runs using the unified plotting function."""
        # Prepare data in the format expected by the unified plotting function
        comparison_data = []
        for res, lbl in zip(results_list, labels):
            comp_item = res.copy()
            comp_item['label'] = lbl
            comparison_data.append(comp_item)

        fig = plot_accel_comp_unified(comparison_data, save_path=save_path)
        # Optional: Close figure
        # if fig: plt.close(fig)


    def generate_acceleration_report(self, save_dir: Optional[str] = None) -> Dict:
        """
        Generate a report comparing different acceleration configurations.

        Args:
            save_dir: Directory to save plots and summary CSV.

        Returns:
            Dictionary containing the comparison results and metrics.
        """
        if save_dir: os.makedirs(save_dir, exist_ok=True)

        logger.info("Generating Acceleration Performance Report...")

        sim_configs = {
            "Baseline": {'use_launch_control': False, 'use_optimized_shifts': False},
            "Optimized Shifts": {'use_launch_control': False, 'use_optimized_shifts': True},
            "Launch Control": {'use_launch_control': True, 'use_optimized_shifts': False},
            "Full Optimization": {'use_launch_control': True, 'use_optimized_shifts': True},
        }

        results_list = []
        metrics_list = []
        labels = list(sim_configs.keys())

        # Store original settings
        orig_lc = self.launch_duration_s > 0 # Check if LC was originally configured
        orig_opt_shifts = self.use_optimized_shifts

        for label, config_params in sim_configs.items():
            logger.info(f" Running simulation for: {label}")
            # Apply config settings for this run
            self.configure_shifting(use_optimized=config_params['use_optimized_shifts'])
            # Note: simulate_acceleration internally checks use_launch_control flag

            results = self.simulate_acceleration(use_launch_control=config_params['use_launch_control'])
            metrics = self.analyze_performance_metrics(results)
            results['label'] = label # Add label for comparison plot
            results_list.append(results)
            metrics_list.append({**{'Configuration': label}, **metrics}) # Add label to metrics dict

        # Restore original settings
        self.configure_shifting(use_optimized=orig_opt_shifts)
        # We assume configure_launch_control was called before if needed for baseline

        # Generate comparison plot
        comp_plot_path = os.path.join(save_dir, "acceleration_comparison.png") if save_dir else None
        self.plot_acceleration_comparison(results_list, labels, save_path=comp_plot_path)

        # Generate individual plot for the best configuration
        best_config_idx = np.argmin([m['finish_time'] if m['finish_time'] is not None else float('inf') for m in metrics_list])
        best_results = results_list[best_config_idx]
        best_plot_path = os.path.join(save_dir, f"acceleration_{labels[best_config_idx].replace(' ','_')}.png") if save_dir else None
        self.plot_acceleration_results(best_results, save_path=best_plot_path, plot_wheel_slip=True)

        # Save metrics summary
        if save_dir:
            metrics_df = pd.DataFrame(metrics_list)
            # Reorder columns for clarity
            cols_order = ['Configuration', 'finish_time', 'time_to_60mph', 'time_to_100kph',
                          'finish_speed_mps', 'peak_acceleration_g', 'avg_acceleration_mpss',
                          'performance_grade']
            # Filter out missing columns before reordering
            cols_order = [col for col in cols_order if col in metrics_df.columns]
            metrics_df = metrics_df[cols_order]
            metrics_df.to_csv(os.path.join(save_dir, "acceleration_metrics_summary.csv"), index=False, float_format='%.3f')

        logger.info("Acceleration report generation complete.")
        return {'simulations': results_list, 'metrics': metrics_list}


# --- Standalone Runner Functions ---

def create_acceleration_simulator(vehicle: Vehicle) -> AccelerationSimulator:
    """Factory function to create a pre-configured AccelerationSimulator."""
    # Check if vehicle has required components
    if not hasattr(vehicle, 'engine') or not hasattr(vehicle, 'drivetrain'):
        raise ValueError("Vehicle object must have 'engine' and 'drivetrain' attributes.")

    simulator = AccelerationSimulator(vehicle)
    # Configure with standard FS defaults
    simulator.configure()
    # Configure LC with potentially better defaults based on vehicle
    launch_rpm_guess = vehicle.engine.max_torque_rpm * 1.05 if vehicle.engine else 9000
    simulator.configure_launch_control(launch_rpm=launch_rpm_guess)
    simulator.configure_shifting(use_optimized=True) # Default to optimized shifts
    return simulator

def run_fs_acceleration_simulation(vehicle: Vehicle, save_dir: Optional[str] = None) -> Dict:
    """
    High-level function to run a standard FS acceleration simulation and report.

    Args:
        vehicle: Vehicle model instance.
        save_dir: Optional directory to save results and plots.

    Returns:
        Dictionary containing the full report data.
    """
    try:
        simulator = create_acceleration_simulator(vehicle)
        report = simulator.generate_acceleration_report(save_dir=save_dir)

        # Add estimated points (simplified)
        best_metrics = report['metrics'][-1] # Assume last one is full optimization
        best_time = best_metrics.get('finish_time')
        points = 0.0
        if best_time:
            # Simplified scoring - real rules are more complex
            time_min_ref = 3.6 # Reference minimum time
            time_max_ref = time_min_ref * 1.5
            if best_time <= time_min_ref: points = 75.0
            elif best_time >= time_max_ref: points = 4.5
            else: points = 4.5 + 70.5 * ((time_max_ref / best_time) - 1) / ((time_max_ref / time_min_ref) - 1)
            points = round(max(0.0, points), 1)

        report['estimated_points'] = points
        logger.info(f"Estimated Acceleration Points: {points:.1f} / 75.0")

        return report

    except Exception as e:
        logger.error(f"Error running FS acceleration simulation: {e}", exc_info=True)
        return {'error': str(e)}


# Example Usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO) # Ensure logs are shown
    # Need to create a vehicle instance first
    try:
        from ..core.vehicle import create_formula_student_vehicle
        print("Creating Formula Student vehicle...")
        vehicle_instance = create_formula_student_vehicle()
        print("Vehicle created.")

        # Define output directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
        output_directory = os.path.join(project_root, "plots", "acceleration_report")
        print(f"Output will be saved to: {output_directory}")

        # Run the full simulation and report generation
        print("\nRunning full acceleration simulation and report...")
        full_report = run_fs_acceleration_simulation(vehicle_instance, save_dir=output_directory)

        if 'error' not in full_report:
            print("\n--- Best Configuration Metrics ---")
            best_metrics = full_report['metrics'][-1] # Last one is full optimization
            for key, val in best_metrics.items():
                if isinstance(val, float): print(f"  {key}: {val:.3f}")
                else: print(f"  {key}: {val}")
            print(f"  Estimated Points: {full_report.get('estimated_points', 'N/A')}")
        else:
            print(f"\nSimulation failed: {full_report['error']}")

    except ImportError:
        print("\nError: Cannot run example without core vehicle modules.")
    except FileNotFoundError as e:
         print(f"\nError: Configuration file not found. Make sure default configs exist.")
         print(e)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()