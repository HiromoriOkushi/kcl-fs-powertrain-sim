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
import copy # Needed for optimization part

# Import core components directly
from ..core.vehicle import Vehicle
from ..transmission.cas_system import ShiftDirection, ShiftState # Import ShiftState for checking
from ..utils.constants import FS_ACCELERATION_LENGTH, MS_TO_MPH, MS_TO_KMH, GRAVITY
from ..utils.plotting import plot_acceleration_results as plot_accel_unified
from ..utils.plotting import plot_acceleration_comparison as plot_accel_comp_unified
from ..utils.plotting import save_plot, set_plot_style, _apply_common_ax_settings, _format_metric_name
# Ensure CorneringPerformance is imported if create_acceleration_simulator uses it
try:
    from ..performance.lap_time import CorneringPerformance
except ImportError:
    CorneringPerformance = None

# Configure logging
logger = logging.getLogger(__name__)

class AccelerationSimulator:
    """Simulates Formula Student acceleration events (e.g., 75m)."""

    def __init__(self, vehicle: Vehicle):
        """
        Args:
            vehicle: Vehicle model instance.
        """
        if not isinstance(vehicle, Vehicle):
            raise TypeError("vehicle must be an instance of the Vehicle class.")
        self.vehicle = vehicle

        # Default simulation parameters
        self.distance_m: float = FS_ACCELERATION_LENGTH
        self.time_step_s: float = 0.005
        self.max_time_s: float = 10.0

        # Launch control parameters
        self.launch_rpm: float = 9000.0
        self.launch_slip_target: float = 0.18
        self.launch_duration_s: float = 0.6
        self.use_traction_control: bool = True

        # Shift strategy parameters
        self.use_optimized_shifts: bool = True
        self.shift_rpm_offset: float = -200

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
        if launch_slip_target is not None: self.launch_slip_target = np.clip(launch_slip_target, 0.05, 0.4)
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
        if not hasattr(self.vehicle, 'engine') or not self.vehicle.engine:
            logger.warning("Cannot determine shift RPM: Vehicle engine not initialized.")
            return 13000
        # Use simpler redline offset logic for now
        return self.vehicle.engine.redline_rpm + self.shift_rpm_offset

    def simulate_acceleration(self, use_launch_control: bool = True) -> Dict:
        # ... (cache check) ...
        logger.info(f"Simulating acceleration: LC={'On' if use_launch_control else 'Off'}, Shifts={'Optimized' if self.use_optimized_shifts else 'Fixed Offset'}")

        # --- Reset Vehicle State ---
        # ... (reset position, speed, etc.) ...
        if hasattr(self.vehicle, 'cas_system') and self.vehicle.cas_system:
            self.vehicle.cas_system.reset()

        # --- Engage 1st Gear ---
        logger.debug("Attempting to engage 1st gear...")
        success_gear1, _ = self.vehicle.change_gear(1)
        # Directly check and set gear if change_gear didn't (e.g., CAS busy at start?)
        if self.vehicle.current_gear() == 0: # Use getter method if available
            if success_gear1:
                logger.warning("change_gear(1) succeeded but vehicle gear is still 0. Forcing gear 1.")
                self.vehicle.current_gear = 1 # Force it if needed
                if self.vehicle.cas_system: self.vehicle.cas_system.current_gear = 1 # Sync CAS too
            else:
                logger.error("Failed to engage 1st gear at simulation start.")
                return {'error': 'Failed to engage 1st gear', 'finish_time': None} # Ensure finish_time is None
        logger.info(f"Starting simulation in Gear: {self.vehicle.current_gear()}")

        # --- Simulation Loop ---
        time_s = 0.0
        dt = self.time_step_s
        in_launch_phase = use_launch_control
        history = {'time': [0.0], 'speed': [0.0], 'position': [0.0], 'acceleration': [0.0], 'engine_rpm': [self.vehicle.current_engine_rpm], 'gear': [self.vehicle.current_gear], 'wheel_slip': [0.0], 'throttle_effective': [0.0]}
        num_gears = getattr(getattr(self.vehicle, 'drivetrain', None), 'num_gears', 0)

        while time_s < self.max_time_s and self.vehicle.current_position < self.distance_m:
            # --- Determine Controls for THIS step ---
            throttle_request = 1.0
            brake_request = 0.0
            effective_throttle = throttle_request
            wheel_slip = 0.0

            # Launch Control Logic
            if in_launch_phase and time_s < self.launch_duration_s:
                if self.use_traction_control and hasattr(self.vehicle, 'tire_radius_m') and self.vehicle.tire_radius_m > 0:
                    try:
                        wheel_speed_ground = self.vehicle.current_speed
                        engine_ang_vel = self.vehicle.current_engine_rpm * (2 * np.pi / 60)
                        overall_ratio = self.vehicle.drivetrain.get_overall_ratio(self.vehicle.current_gear)
                        wheel_speed_engine = (engine_ang_vel / overall_ratio) * self.vehicle.tire_radius_m if overall_ratio > 0 else 0
                        denominator = max(wheel_speed_ground, 0.1)
                        wheel_slip = max(0, (wheel_speed_engine - wheel_speed_ground) / denominator)
                        slip_error = wheel_slip - self.launch_slip_target
                        throttle_reduction = np.clip(slip_error * 5.0, 0.0, 0.8)
                        effective_throttle = throttle_request * (1.0 - throttle_reduction)
                    except AttributeError as e:
                        logger.warning(f"Could not calculate wheel slip for TC: {e}")
            else:
                in_launch_phase = False

            # --- Shifting Logic (Request Shift) ---
            current_gear = self.vehicle.current_gear
            # Check if ready to shift (not already shifting via CAS)
            cas_is_ready = not (self.vehicle.cas_system and self.vehicle.cas_system.system_state != ShiftState.IDLE)
            if cas_is_ready and 0 < current_gear < num_gears:
                 shift_trigger_rpm = self._get_shift_rpm(current_gear)
                 if self.vehicle.current_engine_rpm >= shift_trigger_rpm:
                      # Request upshift via Vehicle method (handles CAS interaction)
                      success_shift, shift_duration = self.vehicle.change_gear(current_gear + 1)
                      if success_shift:
                           logger.debug(f"Shift {current_gear}->{current_gear+1} requested at {self.vehicle.current_engine_rpm:.0f} RPM.")
                           # Note: The actual gear change happens when CAS completes,
                           # handled within vehicle.update_vehicle_state

            # --- Set Vehicle Inputs for THIS step ---
            self.vehicle.throttle_input = effective_throttle
            self.vehicle.brake_input = brake_request
            # Ensure vehicle knows its current RPM before update
            # (Vehicle might recalculate internally, but good practice)
            if hasattr(self.vehicle, 'current_engine_rpm'):
                 self.vehicle.current_engine_rpm = max(getattr(self.vehicle.engine, 'idle_rpm', 1300), self.vehicle.current_engine_rpm)

            # --- Update Vehicle State using its internal method ---
            # This single call handles physics integration, CAS state update, thermal update, etc.
            self.vehicle.update_vehicle_state(dt, ambient_temp_C=25.0) # Pass ambient temp

            # --- Advance Time ---
            time_s += dt

            # --- Store History (using state AFTER update) ---
            history['time'].append(time_s)
            history['speed'].append(self.vehicle.current_speed)
            history['position'].append(self.vehicle.current_position)
            history['acceleration'].append(self.vehicle.current_acceleration)
            history['engine_rpm'].append(self.vehicle.current_engine_rpm)
            history['gear'].append(self.vehicle.current_gear)
            history['wheel_slip'].append(wheel_slip) # Log slip calculated at step start
            history['throttle_effective'].append(effective_throttle) # Log throttle applied

        # --- Post-processing ---
        logger.info(f"Simulation loop finished at time {time_s:.3f}s, position {self.vehicle.current_position:.2f}m")
        results = {key: np.array(val) for key, val in history.items()}
        # Interpolate finish time and speed
        finish_time = None; finish_speed = None
        if results['position'][-1] >= self.distance_m:
            try:
                finish_idx = np.argmax(results['position'] >= self.distance_m)
                if finish_idx > 0:
                     t_before, t_after = results['time'][finish_idx-1], results['time'][finish_idx]
                     p_before, p_after = results['position'][finish_idx-1], results['position'][finish_idx]
                     s_before, s_after = results['speed'][finish_idx-1], results['speed'][finish_idx]
                     if p_after > p_before:
                         interp_factor = (self.distance_m - p_before) / (p_after - p_before)
                         finish_time = t_before + interp_factor * (t_after - t_before)
                         finish_speed = s_before + interp_factor * (s_after - s_before)
                     else: finish_time, finish_speed = t_after, s_after
                else: finish_time, finish_speed = results['time'][0], results['speed'][0]
            except Exception as e: logger.error(f"Error during finish time interpolation: {e}")
        results['finish_time'] = finish_time; results['finish_speed'] = finish_speed

        def time_to_speed(target_speed_mps):
            if len(results['speed']) == 0 or results['speed'][-1] < target_speed_mps: return None
            try:
                target_idx = np.argmax(results['speed'] >= target_speed_mps)
                if target_idx == 0: return 0.0
                t_b, t_a = results['time'][target_idx-1], results['time'][target_idx]
                s_b, s_a = results['speed'][target_idx-1], results['speed'][target_idx]
                return t_b + (target_speed_mps - s_b) / (s_a - s_b) * (t_a - t_b) if s_a > s_b else t_a
            except Exception as e: logger.error(f"Error interpolating time to speed {target_speed_mps:.1f} m/s: {e}"); return None
        results['time_to_60mph'] = time_to_speed(60.0 / MS_TO_MPH)
        results['time_to_100kph'] = time_to_speed(100.0 / MS_TO_KMH)
        logger.debug(f"Results before final log: finish_time={results.get('finish_time')}, 0-60={results.get('time_to_60mph')}, 0-100={results.get('time_to_100kph')}")

        # Add metadata
        results['distance_m'] = self.distance_m; results['used_launch_control'] = use_launch_control
        results['used_optimized_shifts'] = self.use_optimized_shifts
        results['config'] = {'launch_rpm': self.launch_rpm, 'launch_slip_target': self.launch_slip_target, 'launch_duration_s': self.launch_duration_s, 'shift_rpm_offset': self.shift_rpm_offset}

        self.results_cache[cache_key] = results
        ft = results.get('finish_time')
        t60 = results.get('time_to_60mph')
        t100 = results.get('time_to_100kph')
        log_msg = (
            f"Accel sim complete. " +
            (f"Time: {ft:.3f}s" if ft is not None else "Time: DNF") + ", " +
            (f"0-60mph: {t60:.3f}s" if t60 is not None else "0-60mph: N/A") + ", " +
            (f"0-100kph: {t100:.3f}s" if t100 is not None else "0-100kph: N/A")
        )
        logger.info(log_msg)
        return results

    def optimize_launch_control(self,
                             rpm_bounds: Tuple[float, float] = (7000, 11000),
                             slip_bounds: Tuple[float, float] = (0.10, 0.30),
                             duration_bounds: Tuple[float, float] = (0.3, 1.0),
                             max_evals: int = 15) -> Dict:
        """Optimize launch control parameters."""
        logger.info("Optimizing Launch Control parameters...")
        initial_rpm, initial_slip, initial_duration = self.launch_rpm, self.launch_slip_target, self.launch_duration_s

        def objective(value: float, param_to_optimize: str) -> float:
            # Create a temporary copy for optimization runs if needed, or ensure state is reset
            # vehicle_copy = copy.deepcopy(self.vehicle) # Could be slow
            # temp_sim = AccelerationSimulator(vehicle_copy) # Create temp simulator
            # temp_sim.configure(...) # Configure with current best params
            # Modify the one parameter being optimized
            if param_to_optimize == 'rpm': self.configure_launch_control(launch_rpm=value)
            elif param_to_optimize == 'slip': self.configure_launch_control(launch_slip_target=value)
            elif param_to_optimize == 'duration': self.configure_launch_control(launch_duration_s=value)
            results = self.simulate_acceleration(use_launch_control=True)
            finish_time = results.get('finish_time')
            # Restore params after run? No, let optimizer find best combo sequentially.
            return finish_time if finish_time is not None else self.max_time_s * 2

        best_params = {'rpm': initial_rpm, 'slip': initial_slip, 'duration': initial_duration}
        options = {'maxiter': max_evals, 'xatol': 1.0} # Loose tolerance

        try:
            logger.debug(f"Optimizing Launch RPM (Bounds: {rpm_bounds})...")
            options['xatol'] = 50 # RPM tolerance
            res_rpm = minimize_scalar(lambda r: objective(r, 'rpm'), bounds=rpm_bounds, method='bounded', options=options)
            if res_rpm.success: best_params['rpm'] = res_rpm.x
            self.configure_launch_control(launch_rpm=best_params['rpm'])

            logger.debug(f"Optimizing Slip Target (Bounds: {slip_bounds})...")
            options['xatol'] = 0.005 # Slip tolerance
            res_slip = minimize_scalar(lambda s: objective(s, 'slip'), bounds=slip_bounds, method='bounded', options=options)
            if res_slip.success: best_params['slip'] = res_slip.x
            self.configure_launch_control(launch_slip_target=best_params['slip'])

            logger.debug(f"Optimizing Duration (Bounds: {duration_bounds})...")
            options['xatol'] = 0.01 # Duration tolerance
            res_dur = minimize_scalar(lambda d: objective(d, 'duration'), bounds=duration_bounds, method='bounded', options=options)
            if res_dur.success: best_params['duration'] = res_dur.x
            self.configure_launch_control(launch_duration_s=best_params['duration'])
        except Exception as e:
            logger.error(f"Error during launch optimization: {e}. Using intermediate results.")
            self.configure_launch_control(initial_rpm, initial_slip, initial_duration) # Restore

        # Final run with optimized parameters
        final_results = self.simulate_acceleration(use_launch_control=True)
        best_time = final_results.get('finish_time')

        logger.info("Launch control optimization complete.")
        logger.info(f"  Optimal RPM: {best_params['rpm']:.0f}")
        logger.info(f"  Optimal Slip Target: {best_params['slip']:.3f}")
        logger.info(f"  Optimal Duration: {best_params['duration']:.2f} s")
        logger.info(f"  Best 75m Time Achieved: {best_time:.3f} s" if best_time else "  Best time not found.")

        return {'optimal_rpm': best_params['rpm'], 'optimal_slip_target': best_params['slip'], 'optimal_duration_s': best_params['duration'], 'best_finish_time_s': best_time, 'final_run_results': final_results}

    def analyze_performance_metrics(self, results: Dict) -> Dict:
        """Calculate and analyze standard acceleration performance metrics."""
        time = results.get('time', np.array([]))
        speed = results.get('speed', np.array([]))
        accel = results.get('acceleration', np.array([]))
        finish_time = results.get('finish_time')
        metrics = {'finish_time': finish_time, 'finish_speed_mps': results.get('finish_speed'), 'time_to_60mph': results.get('time_to_60mph'), 'time_to_100kph': results.get('time_to_100kph'), 'peak_acceleration_mpss': np.max(accel) if len(accel) > 0 else None, 'avg_acceleration_mpss': None, 'performance_grade': 'N/A'}
        if metrics['peak_acceleration_mpss'] is not None: metrics['peak_acceleration_g'] = metrics['peak_acceleration_mpss'] / GRAVITY
        if finish_time is not None and finish_time > 0 and metrics['finish_speed_mps'] is not None: metrics['avg_acceleration_mpss'] = metrics['finish_speed_mps'] / finish_time
        if finish_time is not None:
            if finish_time < 3.8: grade = 'A+'; 
            elif finish_time < 4.0: grade = 'A'; 
            elif finish_time < 4.2: grade = 'B+'; 
            elif finish_time < 4.4: grade = 'B'; 
            elif finish_time < 4.7: grade = 'C'; 
            elif finish_time < 5.0: grade = 'D'; 
            else: grade = 'F'
            metrics['performance_grade'] = grade
        return metrics

    def plot_acceleration_results(self, results: Dict, save_path: Optional[str] = None, plot_wheel_slip: bool = False):
        """Plot acceleration results using the unified plotting function."""
        plot_data = results.copy()
        fig = plot_accel_unified(plot_data, save_path=save_path, plot_wheel_slip=plot_wheel_slip)
        if fig: plt.close(fig)

    def plot_acceleration_comparison(self, results_list: List[Dict], labels: List[str], save_path: Optional[str] = None):
        """Plot comparison of multiple acceleration runs using the unified plotting function."""
        comparison_data = [{'label': lbl, **res} for res, lbl in zip(results_list, labels)]
        fig = plot_accel_comp_unified(comparison_data, save_path=save_path)
        if fig: plt.close(fig)

    def generate_acceleration_report(self, save_dir: Optional[str] = None) -> Dict:
        """Generate a report comparing different acceleration configurations."""
        if save_dir: os.makedirs(save_dir, exist_ok=True)
        logger.info("Generating Acceleration Performance Report...")
        sim_configs = {"Baseline": {'use_launch_control': False, 'use_optimized_shifts': False}, "Optimized Shifts": {'use_launch_control': False, 'use_optimized_shifts': True}, "Launch Control": {'use_launch_control': True, 'use_optimized_shifts': False}, "Full Optimization": {'use_launch_control': True, 'use_optimized_shifts': True}}
        results_list, metrics_list = [], []
        labels = list(sim_configs.keys())
        orig_opt_shifts = self.use_optimized_shifts

        for label, config_params in sim_configs.items():
            logger.info(f" Running simulation for: {label}")
            self.configure_shifting(use_optimized=config_params['use_optimized_shifts'])
            try:
                results = self.simulate_acceleration(use_launch_control=config_params['use_launch_control'])
                if 'error' in results: raise RuntimeError(results['error'])
                metrics = self.analyze_performance_metrics(results)
                results['label'] = label
                results_list.append(results)
                metrics_list.append({**{'Configuration': label}, **metrics})
            except Exception as e:
                logger.error(f" Simulation failed for '{label}': {e}")
                results_list.append({'label': label, 'error': str(e)})
                metrics_list.append({'Configuration': label, 'finish_time': None, 'error': str(e)})

        self.configure_shifting(use_optimized=orig_opt_shifts)
        valid_results_list = [r for r in results_list if 'error' not in r]
        valid_metrics_list = [m for m in metrics_list if 'error' not in m]

        if not valid_results_list:
            logger.error("No simulations completed successfully for the report.")
            return {'simulations': results_list, 'metrics': metrics_list, 'error': 'All simulations failed'}

        comp_plot_path = os.path.join(save_dir, "acceleration_comparison.png") if save_dir else None
        self.plot_acceleration_comparison(valid_results_list, [r['label'] for r in valid_results_list], save_path=comp_plot_path)

        if valid_metrics_list:
            best_config_idx = np.argmin([m['finish_time'] if m['finish_time'] is not None else float('inf') for m in valid_metrics_list])
            if best_config_idx < len(valid_results_list):
                best_results = valid_results_list[best_config_idx]
                best_plot_path = os.path.join(save_dir, f"acceleration_{best_results['label'].replace(' ','_')}.png") if save_dir else None
                self.plot_acceleration_results(best_results, save_path=best_plot_path, plot_wheel_slip=True)

        if save_dir and metrics_list:
            metrics_df = pd.DataFrame(metrics_list)
            cols_order = ['Configuration', 'finish_time', 'time_to_60mph', 'time_to_100kph', 'finish_speed_mps', 'peak_acceleration_g', 'avg_acceleration_mpss', 'performance_grade', 'error']
            cols_order = [col for col in cols_order if col in metrics_df.columns]
            metrics_df = metrics_df[cols_order]
            metrics_df.to_csv(os.path.join(save_dir, "acceleration_metrics_summary.csv"), index=False, float_format='%.3f')

        logger.info("Acceleration report generation complete.")
        return {'simulations': results_list, 'metrics': metrics_list}

# --- Standalone Runner Functions ---

def create_acceleration_simulator(vehicle: Vehicle) -> Optional[AccelerationSimulator]:
    """Factory function to create a pre-configured AccelerationSimulator."""
    if not hasattr(vehicle, 'engine') or not hasattr(vehicle, 'drivetrain'):
        logger.error("Vehicle object must have 'engine' and 'drivetrain' attributes for AccelerationSimulator.")
        return None
    simulator = AccelerationSimulator(vehicle)
    simulator.configure()
    launch_rpm_guess = getattr(getattr(vehicle, 'engine', None), 'max_torque_rpm', 9000) * 1.05
    simulator.configure_launch_control(launch_rpm=launch_rpm_guess)
    simulator.configure_shifting(use_optimized=True)
    return simulator

def run_fs_acceleration_simulation(vehicle: Vehicle, save_dir: Optional[str] = None) -> Dict:
    """
    High-level function to run a standard FS acceleration simulation and report.
    """
    try:
        simulator = create_acceleration_simulator(vehicle)
        if simulator is None: return {'error': "Failed to create AccelerationSimulator"}
        report = simulator.generate_acceleration_report(save_dir=save_dir)
        # Add estimated points if report successful
        if 'metrics' in report and report['metrics']:
            best_metrics = None
            # Find the first non-error metric dict, assuming the last valid one is best
            for m in reversed(report['metrics']):
                if 'error' not in m:
                    best_metrics = m
                    break
            if best_metrics:
                 best_time = best_metrics.get('finish_time'); points = 0.0
                 if best_time is not None:
                     time_min_ref = 3.6; time_max_ref = time_min_ref * 1.5
                     if best_time <= time_min_ref: points = 75.0
                     elif best_time >= time_max_ref: points = 4.5
                     else: points = 4.5 + 70.5 * ((time_max_ref / best_time) - 1) / ((time_max_ref / time_min_ref) - 1)
                     points = round(max(0.0, points), 1)
                 report['estimated_points'] = points
                 logger.info(f"Estimated Acceleration Points: {points:.1f} / 75.0")
            else:
                 report['estimated_points'] = 0.0
                 logger.warning("Could not determine best metrics for scoring.")
        return report
    except Exception as e:
        logger.error(f"Error running FS acceleration simulation: {e}", exc_info=True)
        return {'error': str(e)}

# Example Usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        from ..core.vehicle import create_formula_student_vehicle
        print("Creating Formula Student vehicle...")
        vehicle_instance = create_formula_student_vehicle()
        print("Vehicle created.")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
        output_directory = os.path.join(project_root, "plots", "acceleration_report")
        print(f"Output will be saved to: {output_directory}")
        print("\nRunning full acceleration simulation and report...")
        full_report = run_fs_acceleration_simulation(vehicle_instance, save_dir=output_directory)
        if 'error' not in full_report:
            print("\n--- Best Configuration Metrics ---")
            # Find best metrics reliably
            best_metrics = None
            for m in reversed(full_report.get('metrics', [])):
                if 'error' not in m and m.get('finish_time') is not None:
                    best_metrics = m
                    break
            if best_metrics:
                for key, val in best_metrics.items():
                    if isinstance(val, float): print(f"  {key}: {val:.3f}")
                    else: print(f"  {key}: {val}")
                print(f"  Estimated Points: {full_report.get('estimated_points', 'N/A')}")
            else:
                print("  Could not determine best metrics from the report.")
        else: print(f"\nSimulation failed: {full_report['error']}")
    except ImportError as e: print(f"\nError: Cannot run example without core vehicle modules ({e}).")
    except FileNotFoundError as e: print(f"\nError: Configuration file not found. {e}")
    except Exception as e: print(f"\nAn unexpected error occurred: {e}"); import traceback; traceback.print_exc()