"""
Torque curve module for Formula Student powertrain simulation.

Provides a class to represent, analyze, manipulate, and optimize engine torque curves,
complementing the core engine model.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Callable
from scipy.interpolate import interp1d, CubicSpline
from scipy.optimize import minimize, fminbound
import logging

# Import necessary components (adjust relative path if needed)
try:
    # Try importing MotorcycleEngine first - necessary for load_from_engine
    from .motorcycle_engine import MotorcycleEngine
except ImportError:
    # Define a placeholder if MotorcycleEngine isn't available during direct run
    # This allows the class definition to proceed but load_from_engine will fail
    class MotorcycleEngine: pass
    MotorcycleEngine = None # Indicate it's not the real class

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("TorqueCurve")

# Constants from utils (import directly or define here)
# It's better practice to import from a central constants module
try:
    from ..utils.constants import KW_TO_HP, HP_TO_KW
except ImportError:
    # Define fallbacks if utils not available (e.g., when running script directly)
    KW_TO_HP = 1.34102
    HP_TO_KW = 1 / KW_TO_HP


class TorqueCurve:
    """
    Represents and analyzes an engine's torque curve.

    Provides methods for loading, interpolating, analyzing (finding peaks),
    modifying (e.g., for different fuels or components), and visualizing torque
    and power curves.
    """

    def __init__(self, rpm_points: Optional[np.ndarray] = None, torque_values_nm: Optional[np.ndarray] = None):
        """
        Initialize a TorqueCurve object.

        Args:
            rpm_points: Array of RPM points.
            torque_values_nm: Array of torque values (in Nm) corresponding to rpm_points.
        """
        self.rpm_points: Optional[np.ndarray] = None
        self.torque_values_nm: Optional[np.ndarray] = None
        self.power_values_kw: Optional[np.ndarray] = None
        self.torque_function: Optional[Callable] = None
        self.power_function_kw: Optional[Callable] = None

        # Default RPM range if not set by data
        self.min_rpm: float = 1000.0
        self.max_rpm: float = 14000.0

        if rpm_points is not None and torque_values_nm is not None:
            self.set_curve(rpm_points, torque_values_nm)

    def set_curve(self, rpm_points: np.ndarray, torque_values_nm: np.ndarray):
        """
        Set the torque curve data and update derived properties.

        Args:
            rpm_points: Array of RPM points.
            torque_values_nm: Array of torque values (in Nm).
        """
        if len(rpm_points) != len(torque_values_nm):
            raise ValueError("RPM points and torque values must have the same length.")
        if len(rpm_points) < 2:
            raise ValueError("Torque curve requires at least 2 data points.")

        # Sort data by RPM
        idx = np.argsort(rpm_points)
        self.rpm_points = np.array(rpm_points[idx], dtype=float)
        self.torque_values_nm = np.array(torque_values_nm[idx], dtype=float)

        # Update RPM limits
        self.min_rpm = self.rpm_points[0]
        self.max_rpm = self.rpm_points[-1]

        # Calculate power values
        self._calculate_power()

        # Create interpolation functions
        self._create_interpolation_functions()
        logger.debug(f"Torque curve set with {len(self.rpm_points)} points from {self.min_rpm:.0f} to {self.max_rpm:.0f} RPM.")

    def _calculate_power(self):
        """Calculate power (kW) from torque (Nm) and RPM."""
        if self.rpm_points is None or self.torque_values_nm is None: return
        # Power (W) = Torque (Nm) * Angular velocity (rad/s)
        # Angular velocity = RPM * 2 * pi / 60
        power_watts = self.torque_values_nm * self.rpm_points * (2 * np.pi / 60)
        self.power_values_kw = power_watts / 1000.0

    def _create_interpolation_functions(self):
        """Create interpolation functions for smooth curve access."""
        if self.rpm_points is None or len(self.rpm_points) < 2:
            logger.warning("Insufficient data to create interpolation functions.")
            self.torque_function = None
            self.power_function_kw = None
            return

        # Ensure RPM points are strictly increasing for spline interpolation
        unique_rpm, unique_indices = np.unique(self.rpm_points, return_index=True)
        unique_torque = self.torque_values_nm[unique_indices]
        unique_power = self.power_values_kw[unique_indices] if self.power_values_kw is not None else None

        # Use cubic spline if enough points, otherwise linear
        interp_kind = 'cubic' if len(unique_rpm) >= 4 else 'linear'
        fill_value_torque = (unique_torque[0], unique_torque[-1])
        fill_value_power = (unique_power[0], unique_power[-1]) if unique_power is not None else (0,0)

        try:
            self.torque_function = interp1d(
                unique_rpm, unique_torque, kind=interp_kind,
                bounds_error=False, fill_value=fill_value_torque
            )
            if unique_power is not None:
                self.power_function_kw = interp1d(
                    unique_rpm, unique_power, kind=interp_kind,
                    bounds_error=False, fill_value=fill_value_power
                )
            else:
                self.power_function_kw = None # Recalculate if needed
        except ValueError as e:
            logger.error(f"Interpolation failed ({e}). Check data uniqueness/order.")
            self.torque_function = None
            self.power_function_kw = None

    def load_from_dyno_data(self, file_path: str, rpm_column: str = 'RPM',
                          torque_column: str = 'Torque_Nm') -> bool:
        """
        Load torque curve from dyno data file (CSV).

        Args:
            file_path: Path to dyno data CSV file.
            rpm_column: Column name for RPM values.
            torque_column: Column name for torque values (in Nm).

        Returns:
            True if successful, False otherwise.
        """
        if not os.path.exists(file_path):
            logger.error(f"Dyno data file not found: {file_path}")
            return False
        try:
            data = pd.read_csv(file_path)
            if rpm_column not in data.columns or torque_column not in data.columns:
                raise ValueError(f"Required columns '{rpm_column}' or '{torque_column}' not found.")

            # Basic cleaning: remove NaN/inf, convert to numeric
            data = data[[rpm_column, torque_column]].dropna()
            data[rpm_column] = pd.to_numeric(data[rpm_column], errors='coerce')
            data[torque_column] = pd.to_numeric(data[torque_column], errors='coerce')
            data = data.dropna()

            if data.empty:
                 raise ValueError("No valid numeric data found in specified columns.")

            self.set_curve(data[rpm_column].values, data[torque_column].values)
            logger.info(f"Torque curve loaded from dyno data: {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading dyno data from {file_path}: {e}")
            return False

    def load_from_engine(self, engine: 'MotorcycleEngine'):
        """
        Load torque curve from a MotorcycleEngine object.

        Args:
            engine: MotorcycleEngine object instance.
        """
        if not MotorcycleEngine:
             logger.error("Cannot load from engine: MotorcycleEngine class not available.")
             return
        if not isinstance(engine, MotorcycleEngine):
             raise TypeError("Input must be an instance of MotorcycleEngine")
        if engine.rpm_range is None or engine.torque_curve is None:
            raise ValueError("Engine object missing required torque curve attributes (rpm_range, torque_curve). Ensure engine curves are generated.")

        self.set_curve(engine.rpm_range, engine.torque_curve)
        self.min_rpm = engine.idle_rpm
        self.max_rpm = engine.redline_rpm
        logger.info(f"Torque curve loaded from {engine.make} {engine.model} engine object.")

    def get_torque(self, rpm: float) -> float:
        """Get interpolated torque at a specific RPM."""
        if self.torque_function is None:
            logger.warning("Torque function not initialized, returning 0.")
            return 0.0
        # Clip RPM to the defined range before interpolating
        rpm_clipped = np.clip(rpm, self.min_rpm, self.max_rpm)
        return float(self.torque_function(rpm_clipped))

    def get_power_kw(self, rpm: float) -> float:
        """Get interpolated power in kW at a specific RPM."""
        if self.power_function_kw:
            rpm_clipped = np.clip(rpm, self.min_rpm, self.max_rpm)
            return float(self.power_function_kw(rpm_clipped))
        elif self.torque_function:
            # Calculate from torque if power function isn't available
            torque = self.get_torque(rpm)
            power_watts = torque * rpm * (2 * np.pi / 60)
            return power_watts / 1000.0
        else:
            logger.warning("Power function not initialized, returning 0.")
            return 0.0

    def get_power_hp(self, rpm: float) -> float:
        """Get interpolated power in HP at a specific RPM."""
        return self.get_power_kw(rpm) * KW_TO_HP

    def find_peak_torque(self) -> Tuple[float, float]:
        """Find the RPM and value of maximum torque."""
        if self.torque_function is None: raise ValueError("Torque curve not initialized")
        # Use optimization on the interpolation function for better accuracy
        res = minimize(lambda r: -self.torque_function(r), # Minimize negative torque
                       x0=(self.min_rpm + self.max_rpm) / 2, # Initial guess
                       bounds=[(self.min_rpm, self.max_rpm)],
                       method='L-BFGS-B') # Bounded optimization
        if res.success:
            rpm_at_max = res.x[0]
            max_torque = -res.fun
            return float(rpm_at_max), float(max_torque)
        else:
            # Fallback to max of sampled points if optimization fails
            idx = np.argmax(self.torque_values_nm)
            return self.rpm_points[idx], self.torque_values_nm[idx]

    def find_peak_power(self) -> Tuple[float, float]:
        """Find the RPM and value (kW) of maximum power."""
        if self.power_function_kw is None and self.torque_function is None:
             raise ValueError("Torque/Power curve not initialized")
        if self.power_function_kw is None:
             self._create_interpolation_functions() # Ensure power function exists
             if self.power_function_kw is None: return 0.0, 0.0 # Still failed

        res = minimize(lambda r: -self.power_function_kw(r), # Minimize negative power
                       x0=(self.min_rpm + self.max_rpm) / 2,
                       bounds=[(self.min_rpm, self.max_rpm)],
                       method='L-BFGS-B')
        if res.success:
            rpm_at_max = res.x[0]
            max_power_kw = -res.fun
            return float(rpm_at_max), float(max_power_kw)
        else:
            # Fallback
            idx = np.argmax(self.power_values_kw)
            return self.rpm_points[idx], self.power_values_kw[idx]

    def get_optimal_shift_points(self, gear_ratios: List[float],
                              final_drive_ratio: float) -> List[Optional[float]]:
        """
        Calculate optimal shift points (RPM) for maximum tractive force.

        Args:
            gear_ratios: List of transmission gear ratios [g1, g2, ...].
            final_drive_ratio: Final drive ratio.

        Returns:
            List of optimal upshift RPMs for gears 1 to N-1. Returns None for a gear if no optimal point found.
        """
        if len(gear_ratios) < 2: return []
        if self.torque_function is None: raise ValueError("Torque curve not initialized")

        shift_points_rpm = []

        for i in range(len(gear_ratios) - 1):
            current_gear_idx = i
            next_gear_idx = i + 1
            current_trans_ratio = gear_ratios[current_gear_idx]
            next_trans_ratio = gear_ratios[next_gear_idx]

            # Overall ratio = transmission * final_drive
            overall_ratio_current = current_trans_ratio * final_drive_ratio
            overall_ratio_next = next_trans_ratio * final_drive_ratio

            # Define the function where tractive force difference is zero
            # Tractive Force = Engine_Torque * Overall_Ratio / Wheel_Radius
            # We want Tq(rpm) * OR_current = Tq(rpm_next) * OR_next
            # where rpm_next = rpm * (OR_next / OR_current) = rpm * (next_trans_ratio / current_trans_ratio)
            ratio_of_ratios = next_trans_ratio / current_trans_ratio

            def force_difference(rpm: float) -> float:
                rpm = np.clip(rpm, self.min_rpm, self.max_rpm)
                rpm_next = rpm * ratio_of_ratios
                rpm_next = np.clip(rpm_next, self.min_rpm, self.max_rpm)

                force_current = self.torque_function(rpm) * overall_ratio_current
                force_next = self.torque_function(rpm_next) * overall_ratio_next
                return force_current - force_next # Find where this is zero

            # Find the root (where forces are equal) using optimization or root finding
            # Start search slightly above peak torque RPM
            search_start_rpm = self.find_peak_torque()[0] * 1.05
            search_end_rpm = self.max_rpm

            try:
                # Use bounded scalar optimization to find the root (intersection)
                # fminbound finds the minimum, so we minimize the absolute difference
                opt_result = fminbound(lambda r: abs(force_difference(r)),
                                       search_start_rpm, search_end_rpm,
                                       xtol=10) # Tolerance in RPM
                optimal_rpm = opt_result
                # Verify the difference is close to zero
                if abs(force_difference(optimal_rpm)) > 10: # Allow some tolerance (10Nm equiv)
                     optimal_rpm = None # Didn't find a good intersection
            except:
                 optimal_rpm = None # Optimization failed

            # If no intersection found, use a default (e.g., 95% of redline)
            if optimal_rpm is None:
                optimal_rpm = self.max_rpm * 0.95
                logger.warning(f"Optimal shift point for {i+1}->{i+2} not found, using default {optimal_rpm:.0f} RPM")

            shift_points_rpm.append(optimal_rpm)

        return shift_points_rpm


    def apply_modification(self, modification_function: Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]):
        """Apply a modification function to the raw torque curve data."""
        if self.rpm_points is None or self.torque_values_nm is None:
            raise ValueError("Cannot modify uninitialized torque curve")
        rpm_modified, torque_modified = modification_function(self.rpm_points.copy(), self.torque_values_nm.copy())
        self.set_curve(rpm_modified, torque_modified) # Recalculates power and interpolators

    def modify_for_e85(self, power_increase_factor: float = 1.05, torque_shift_rpm: float = -500):
        """
        Apply typical modifications for E85 fuel (more power, torque shifts earlier).

        Args:
            power_increase_factor: Multiplicative factor for power increase (e.g., 1.05 for 5%).
            torque_shift_rpm: RPM shift for the torque peak (negative shifts left).
        """
        if self.rpm_points is None: raise ValueError("Curve not set")
        logger.info(f"Applying E85 modification: Power x{power_increase_factor}, Torque Peak Shift {torque_shift_rpm} RPM")

        # Power increase effectively scales torque appropriately
        new_torque = self.torque_values_nm * power_increase_factor

        # Shift the RPM points for the peak torque effect
        # Create a mapping from old RPM to new RPM (shifting peak)
        peak_rpm_orig, _ = self.find_peak_torque()
        peak_rpm_new = peak_rpm_orig + torque_shift_rpm

        def rpm_shift_map(rpm):
             # Simple linear shift around the peak
             if rpm <= peak_rpm_orig: # Below original peak
                 # Shift left more as we approach the original peak
                 factor = (rpm - self.min_rpm) / max(1.0, peak_rpm_orig - self.min_rpm)
                 return rpm + torque_shift_rpm * factor
             else: # Above original peak
                 # Shift left less as we move away from the peak
                 factor = (self.max_rpm - rpm) / max(1.0, self.max_rpm - peak_rpm_orig)
                 return rpm + torque_shift_rpm * factor

        # Apply RPM shift using interpolation (might require re-gridding)
        # For simplicity here, we'll just scale torque directly. A better approach
        # would involve re-evaluating the underlying efficiency models if available,
        # or using a more sophisticated warp function on the RPM axis.
        # This simplified approach just scales the torque values.
        self.set_curve(self.rpm_points, new_torque)

    def modify_for_exhaust(self, mid_range_boost: float = 1.04, top_end_boost: float = 1.02):
        """Apply typical modifications for a performance exhaust."""
        if self.rpm_points is None: raise ValueError("Curve not set")
        logger.info(f"Applying exhaust modification: Mid x{mid_range_boost}, Top x{top_end_boost}")

        # Define RPM ranges (relative to redline)
        mid_start = 0.4 * self.max_rpm
        mid_end = 0.7 * self.max_rpm
        top_start = 0.7 * self.max_rpm

        boost_factors = np.ones_like(self.torque_values_nm)

        # Mid-range boost (linear ramp up and down)
        mid_mask = (self.rpm_points >= mid_start) & (self.rpm_points <= mid_end)
        if np.any(mid_mask):
             mid_center_idx = np.where(mid_mask)[0][0] + np.sum(mid_mask)//2
             mid_ramp_up = np.linspace(1.0, mid_range_boost, mid_center_idx - np.where(mid_mask)[0][0] +1)
             mid_ramp_down = np.linspace(mid_range_boost, 1.0, np.where(mid_mask)[0][-1] - mid_center_idx + 1)
             boost_factors[mid_mask] = np.concatenate((mid_ramp_up[:-1], mid_ramp_down))

        # Top-end boost (linear ramp up)
        top_mask = self.rpm_points >= top_start
        if np.any(top_mask):
             top_ramp = np.linspace(1.0, top_end_boost, np.sum(top_mask))
             boost_factors[top_mask] = np.maximum(boost_factors[top_mask], top_ramp) # Take max boost

        self.set_curve(self.rpm_points, self.torque_values_nm * boost_factors)

    def modify_for_intake(self, overall_boost: float = 1.02, resonance_rpm: float = 8000, resonance_boost: float = 1.05, resonance_width: float = 500):
         """Apply typical modifications for a performance intake."""
         if self.rpm_points is None: raise ValueError("Curve not set")
         logger.info(f"Applying intake modification: Overall x{overall_boost}, Resonance +{resonance_boost-1:.1%} @{resonance_rpm} RPM")

         # Apply overall boost
         new_torque = self.torque_values_nm * overall_boost

         # Apply resonance effect (Gaussian shape)
         resonance_effect = (resonance_boost - 1.0) * np.exp(-(self.rpm_points - resonance_rpm)**2 / (2 * resonance_width**2))
         new_torque *= (1.0 + resonance_effect)

         self.set_curve(self.rpm_points, new_torque)

    def predict_wheel_torque(self, gear_ratio: float, final_drive_ratio: float,
                          transmission_efficiency: float = 0.96, # Slightly higher default
                          final_drive_efficiency: float = 0.97,
                          wheel_radius_m: float = 0.2286) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate wheel torque and corresponding vehicle speed for a specific gear.

        Args:
            gear_ratio: Transmission gear ratio.
            final_drive_ratio: Final drive ratio.
            transmission_efficiency: Efficiency of the transmission (0-1).
            final_drive_efficiency: Efficiency of the final drive (0-1).
            wheel_radius_m: Wheel radius in meters.

        Returns:
            Tuple of (vehicle_speed_kph, wheel_torque_nm) arrays.
        """
        if self.rpm_points is None or self.torque_function is None:
            raise ValueError("Torque curve not initialized")

        # Calculate overall ratio and efficiency
        overall_ratio = gear_ratio * final_drive_ratio
        overall_efficiency = transmission_efficiency * final_drive_efficiency

        # Calculate wheel torque across the RPM range
        wheel_torque_nm = self.torque_function(self.rpm_points) * overall_ratio * overall_efficiency

        # Calculate wheel speed in RPM
        wheel_rpm = self.rpm_points / overall_ratio

        # Convert wheel RPM to vehicle speed in km/h
        # Speed (m/s) = Wheel RPM * (2 * pi * Radius) / 60
        # Speed (km/h) = Speed (m/s) * 3.6
        vehicle_speed_kph = wheel_rpm * (2 * np.pi * wheel_radius_m / 60) * 3.6

        return vehicle_speed_kph, wheel_torque_nm

    def save_to_csv(self, file_path: str):
        """Save the current torque and power curve data to a CSV file."""
        if self.rpm_points is None or self.torque_values_nm is None or self.power_values_kw is None:
            raise ValueError("Cannot save uninitialized torque curve")

        df = pd.DataFrame({
            'RPM': self.rpm_points,
            'Torque_Nm': self.torque_values_nm,
            'Power_kW': self.power_values_kw,
            'Power_HP': self.power_values_kw * KW_TO_HP
        })

        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            df.to_csv(file_path, index=False, float_format='%.3f')
            logger.info(f"Torque curve data saved to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save torque curve data to {file_path}: {e}")

    def plot_curve(self, show_power: bool = True, title: str = 'Engine Performance Curve',
                 save_path: Optional[str] = None):
        """Plot the torque and power curves using the centralized plotting utility."""
        if self.rpm_points is None or self.torque_values_nm is None or self.power_values_kw is None:
            logger.error("Cannot plot curve: Data not initialized.")
            return

        from ..utils.plotting import plot_engine_performance, save_plot # Local import

        peak_tq_rpm, peak_tq = self.find_peak_torque()
        peak_pw_rpm, peak_pw = self.find_peak_power()

        engine_data = {
            'rpm': self.rpm_points,
            'torque': self.torque_values_nm,
            'power': self.power_values_kw,
            'max_torque_rpm': peak_tq_rpm,
            'max_torque': peak_tq,
            'max_power_rpm': peak_pw_rpm,
            'max_power': peak_pw # Pass power in kW for annotation
        }

        fig = plot_engine_performance(engine_data, title=title, show_efficiency=False) # Efficiency not part of this class

        if save_path and fig:
            save_plot(fig, save_path)
        elif fig:
            plt.show()
        # Close the figure if it wasn't saved or shown explicitly elsewhere
        if fig:
            plt.close(fig)

    def plot_comparison(self, other_curve: 'TorqueCurve', labels: Tuple[str, str] = ('Original', 'Modified'),
                      title: str = 'Torque Curve Comparison', save_path: Optional[str] = None):
        """Plot a comparison between this curve and another using the centralized utility."""
        if self.rpm_points is None or other_curve.rpm_points is None:
             logger.error("Cannot plot comparison: One or both curves are not initialized.")
             return

        from ..utils.plotting import plot_torque_curves_comparison, save_plot # Local import

        curves_data = [
            {'rpm': self.rpm_points, 'torque': self.torque_values_nm, 'power': self.power_values_kw, 'label': labels[0]},
            {'rpm': other_curve.rpm_points, 'torque': other_curve.torque_values_nm, 'power': other_curve.power_values_kw, 'label': labels[1]}
        ]

        fig = plot_torque_curves_comparison(curves_data, title=title)

        if save_path and fig:
            save_plot(fig, save_path)
        elif fig:
            plt.show()
        # Close the figure
        if fig:
            plt.close(fig)


# Example usage
if __name__ == "__main__":
    # Create a dummy torque curve
    rpm = np.linspace(2000, 14000, 100)
    # Realistic-ish shape peaking around 10500 RPM
    torque = 50 + 20 * np.exp(-((rpm - 10500) / 3000)**2) - 5 * ((rpm - 2000) / 12000)
    base_curve = TorqueCurve(rpm, torque)

    # --- Analysis ---
    peak_tq_rpm, peak_tq = base_curve.find_peak_torque()
    peak_pw_rpm, peak_pw_kw = base_curve.find_peak_power()
    print(f"Base Curve - Peak Torque: {peak_tq:.1f} Nm @ {peak_tq_rpm:.0f} RPM")
    print(f"Base Curve - Peak Power: {peak_pw_kw:.1f} kW ({peak_pw_kw * KW_TO_HP:.1f} HP) @ {peak_pw_rpm:.0f} RPM")

    # --- Modification ---
    modified_curve = TorqueCurve(rpm, torque) # Start with a copy
    modified_curve.modify_for_e85(power_increase_factor=1.06, torque_shift_rpm=-400)
    modified_curve.modify_for_exhaust(mid_range_boost=1.05, top_end_boost=1.03)
    modified_curve.modify_for_intake(overall_boost=1.02, resonance_rpm=9000, resonance_boost=1.04, resonance_width=400)

    peak_tq_rpm_mod, peak_tq_mod = modified_curve.find_peak_torque()
    peak_pw_rpm_mod, peak_pw_kw_mod = modified_curve.find_peak_power()
    print(f"\nModified Curve - Peak Torque: {peak_tq_mod:.1f} Nm @ {peak_tq_rpm_mod:.0f} RPM")
    print(f"Modified Curve - Peak Power: {peak_pw_kw_mod:.1f} kW ({peak_pw_kw_mod * KW_TO_HP:.1f} HP) @ {peak_pw_rpm_mod:.0f} RPM")

    # --- Plotting ---
    # Setup plots directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    plot_dir = os.path.join(project_root, "plots", "engine")
    os.makedirs(plot_dir, exist_ok=True)

    base_curve.plot_curve(title="Base Torque Curve", save_path=os.path.join(plot_dir, "base_curve.png"))
    modified_curve.plot_curve(title="Modified Torque Curve (E85+Exhaust+Intake)", save_path=os.path.join(plot_dir, "modified_curve.png"))
    base_curve.plot_comparison(modified_curve, labels=("Base", "Modified"), title="Curve Comparison", save_path=os.path.join(plot_dir, "curve_comparison.png"))

    # --- Shift Points ---
    gear_ratios = [2.750, 2.000, 1.667, 1.444, 1.304, 1.208]
    final_drive = 53 / 14.0
    shift_points = modified_curve.get_optimal_shift_points(gear_ratios, final_drive)
    print("\nOptimal Upshift Points (Modified Curve):")
    for i, rpm in enumerate(shift_points):
        if rpm:
             print(f"  Shift {i+1}->{i+2}: {rpm:.0f} RPM")
        else:
             print(f"  Shift {i+1}->{i+2}: Not found")

    # --- Wheel Torque ---
    fig_wt, ax_wt = plt.subplots(figsize=(10,6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(gear_ratios)))
    for i, gear_ratio in enumerate(gear_ratios):
        speed_kph, wheel_torque = modified_curve.predict_wheel_torque(gear_ratio, final_drive)
        ax_wt.plot(speed_kph, wheel_torque, label=f"Gear {i+1}", color=colors[i], linewidth=2)

    ax_wt.set_xlabel("Vehicle Speed (km/h)")
    ax_wt.set_ylabel("Wheel Torque (Nm)")
    ax_wt.set_title("Wheel Torque vs Speed (Modified Curve)")
    ax_wt.grid(True, alpha=0.5)
    ax_wt.legend()
    ax_wt.set_xlim(0, max(speed_kph)*1.05)
    ax_wt.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "wheel_torque.png"))
    plt.show()