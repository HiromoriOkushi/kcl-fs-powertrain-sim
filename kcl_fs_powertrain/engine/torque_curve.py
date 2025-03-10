"""
Torque curve module for Formula Student powertrain simulation.

This module extends the engine modeling capabilities with detailed torque curve
analysis, manipulation, and optimization tools. It works alongside the
MotorcycleEngine class to provide enhanced torque modeling features.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Callable
from scipy.interpolate import interp1d, CubicSpline
from scipy.optimize import minimize


class TorqueCurve:
    """
    Class for handling torque curves in Formula Student applications.
    
    Provides tools for manipulating, analyzing, and optimizing engine torque curves
    to improve vehicle performance characteristics.
    """
    
    def __init__(self, rpm_points: np.ndarray = None, torque_values: np.ndarray = None):
        """
        Initialize a TorqueCurve object.
        
        Args:
            rpm_points: Array of RPM points
            torque_values: Array of torque values (in Nm) corresponding to rpm_points
        """
        self.rpm_points = rpm_points
        self.torque_values = torque_values
        self.power_values = None
        self.torque_function = None
        self.power_function = None
        
        # Default RPM ranges for analysis
        self.idle_rpm = 1300
        self.redline_rpm = 14000
        
        # Initialize if data provided
        if rpm_points is not None and torque_values is not None:
            self.set_curve(rpm_points, torque_values)
    
    def set_curve(self, rpm_points: np.ndarray, torque_values: np.ndarray):
        """
        Set the torque curve based on RPM and torque value arrays.
        
        Args:
            rpm_points: Array of RPM points
            torque_values: Array of torque values (in Nm) corresponding to rpm_points
        """
        if len(rpm_points) != len(torque_values):
            raise ValueError("RPM points and torque values must have the same length")
        
        # Sort data by RPM (in case it's not already sorted)
        idx = np.argsort(rpm_points)
        self.rpm_points = rpm_points[idx]
        self.torque_values = torque_values[idx]
        
        # Update range values
        if len(self.rpm_points) > 0:
            self.idle_rpm = self.rpm_points[0]
            self.redline_rpm = self.rpm_points[-1]
        
        # Calculate power values
        self._calculate_power()
        
        # Create interpolation functions
        self._create_interpolation_functions()
    
    def _calculate_power(self):
        """Calculate power values from torque and RPM."""
        if self.rpm_points is None or self.torque_values is None:
            return
        
        # Power (kW) = Torque (Nm) * Angular velocity (rad/s) / 1000
        # Angular velocity = 2π * rpm / 60
        self.power_values = self.torque_values * self.rpm_points * 2 * np.pi / 60 / 1000
    
    def _create_interpolation_functions(self):
        """Create interpolation functions for torque and power curves."""
        if self.rpm_points is None or self.torque_values is None:
            return
        
        # Create cubic spline interpolation for smoother curves
        self.torque_function = CubicSpline(
            self.rpm_points,
            self.torque_values,
            bc_type='natural'  # Natural boundary conditions
        )
        
        # Create power interpolation function
        if self.power_values is not None:
            self.power_function = CubicSpline(
                self.rpm_points,
                self.power_values,
                bc_type='natural'
            )
    
    def load_from_dyno_data(self, file_path: str, rpm_column: str = 'RPM', 
                          torque_column: str = 'Torque_Nm'):
        """
        Load torque curve from dyno data file.
        
        Args:
            file_path: Path to dyno data file (CSV format)
            rpm_column: Column name for RPM values
            torque_column: Column name for torque values
        """
        try:
            data = pd.read_csv(file_path)
            
            # Check if required columns exist
            if rpm_column not in data.columns or torque_column not in data.columns:
                raise ValueError(f"Required columns '{rpm_column}' and/or '{torque_column}' not found")
            
            # Extract data
            rpm_points = data[rpm_column].values
            torque_values = data[torque_column].values
            
            # Set curve
            self.set_curve(rpm_points, torque_values)
            
            return True
        except Exception as e:
            print(f"Error loading dyno data: {str(e)}")
            return False
    
    def load_from_engine(self, engine):
        """
        Load torque curve from a MotorcycleEngine object.
        
        Args:
            engine: MotorcycleEngine object with torque curve data
        """
        if not hasattr(engine, 'torque_curve') or not hasattr(engine, 'rpm_range'):
            raise ValueError("Engine object missing required torque curve attributes")
        
        self.set_curve(engine.rpm_range, engine.torque_curve)
        self.idle_rpm = engine.idle_rpm
        self.redline_rpm = engine.redline
    
    def get_torque(self, rpm: float) -> float:
        """
        Get torque at specified RPM using interpolation.
        
        Args:
            rpm: Engine speed in RPM
            
        Returns:
            Torque in Nm
        """
        if self.torque_function is None:
            raise ValueError("Torque curve not initialized")
        
        # Ensure RPM is within bounds
        rpm = max(self.idle_rpm, min(rpm, self.redline_rpm))
        
        return float(self.torque_function(rpm))
    
    def get_power(self, rpm: float) -> float:
        """
        Get power at specified RPM using interpolation.
        
        Args:
            rpm: Engine speed in RPM
            
        Returns:
            Power in kW
        """
        if self.power_function is None:
            # Calculate power directly if function not available
            return self.get_torque(rpm) * rpm * 2 * np.pi / 60 / 1000
        
        # Ensure RPM is within bounds
        rpm = max(self.idle_rpm, min(rpm, self.redline_rpm))
        
        return float(self.power_function(rpm))
    
    def get_power_hp(self, rpm: float) -> float:
        """
        Get power at specified RPM in horsepower.
        
        Args:
            rpm: Engine speed in RPM
            
        Returns:
            Power in hp
        """
        power_kw = self.get_power(rpm)
        return power_kw * 1.34102  # Convert kW to hp
    
    def find_peak_torque(self) -> Tuple[float, float]:
        """
        Find RPM at which torque is maximum.
        
        Returns:
            Tuple of (RPM at max torque, max torque value)
        """
        if self.rpm_points is None or self.torque_values is None:
            raise ValueError("Torque curve not initialized")
        
        # Generate dense RPM grid for more precise peak finding
        dense_rpm = np.linspace(self.idle_rpm, self.redline_rpm, 1000)
        dense_torque = self.torque_function(dense_rpm)
        
        # Find maximum
        max_idx = np.argmax(dense_torque)
        return dense_rpm[max_idx], dense_torque[max_idx]
    
    def find_peak_power(self) -> Tuple[float, float]:
        """
        Find RPM at which power is maximum.
        
        Returns:
            Tuple of (RPM at max power, max power value in kW)
        """
        if self.rpm_points is None or self.torque_values is None:
            raise ValueError("Torque curve not initialized")
        
        # Generate dense RPM grid
        dense_rpm = np.linspace(self.idle_rpm, self.redline_rpm, 1000)
        dense_power = np.array([self.get_power(r) for r in dense_rpm])
        
        # Find maximum
        max_idx = np.argmax(dense_power)
        return dense_rpm[max_idx], dense_power[max_idx]
    
    def get_optimal_shift_points(self, gear_ratios: List[float], 
                              final_drive_ratio: float, 
                              wheel_radius: float) -> List[float]:
        """
        Calculate optimal shift points for given gear ratios.
        
        Finds the RPM at which shifting to the next gear provides maximum acceleration.
        
        Args:
            gear_ratios: List of gear ratios (ordered from first to last gear)
            final_drive_ratio: Final drive ratio
            wheel_radius: Wheel radius in meters
            
        Returns:
            List of RPM values for optimal shifts between gears
        """
        if len(gear_ratios) < 2:
            return []
        
        shift_points = []
        
        for i in range(len(gear_ratios) - 1):
            # Current gear ratio
            current_ratio = gear_ratios[i] * final_drive_ratio
            # Next gear ratio
            next_ratio = gear_ratios[i + 1] * final_drive_ratio
            
            # Function to minimize (negative of acceleration difference)
            def acceleration_diff(rpm):
                # Torque at the wheels in current gear
                current_torque = self.get_torque(rpm) * current_ratio
                # Wheel force = torque / radius
                current_force = current_torque / wheel_radius
                
                # RPM in next gear after shift
                next_rpm = rpm * next_ratio / current_ratio
                if next_rpm < self.idle_rpm:
                    return 1e10  # Heavy penalty if next gear RPM is below idle
                
                # Torque and force in next gear
                next_torque = self.get_torque(next_rpm) * next_ratio
                next_force = next_torque / wheel_radius
                
                # Return negative of force difference (for minimization)
                return -(current_force - next_force)
            
            # Find where current gear and next gear forces are equal
            result = minimize(
                acceleration_diff,
                x0=0.7 * self.redline_rpm,  # Initial guess at 70% of redline
                bounds=[(self.idle_rpm, self.redline_rpm)],
                method='SLSQP'
            )
            
            if result.success:
                shift_points.append(float(result.x))
            else:
                # Fallback: use 95% of redline if optimization fails
                shift_points.append(0.95 * self.redline_rpm)
        
        return shift_points
    
    def apply_modification(self, modification_function: Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]):
        """
        Apply a modification function to the torque curve.
        
        Args:
            modification_function: Function that takes (rpm_points, torque_values) and returns
                                  modified (rpm_points, torque_values)
        """
        if self.rpm_points is None or self.torque_values is None:
            raise ValueError("Cannot modify uninitialized torque curve")
        
        rpm_modified, torque_modified = modification_function(self.rpm_points, self.torque_values)
        self.set_curve(rpm_modified, torque_modified)
    
    def modify_for_e85(self, increase_factor: float = 1.10, low_end_boost: float = 1.15):
        """
        Modify torque curve for E85 fuel conversion.
        
        E85 typically provides increased torque, especially at lower RPMs,
        at the cost of increased fuel consumption.
        
        Args:
            increase_factor: Overall torque increase factor
            low_end_boost: Additional boost factor for low-end torque
        """
        def e85_modification(rpm, torque):
            # Calculate normalized RPM position (0 to 1)
            norm_position = (rpm - self.idle_rpm) / (self.redline_rpm - self.idle_rpm)
            
            # Low-end boost decreases linearly with RPM
            low_end_factor = 1.0 + (low_end_boost - 1.0) * (1.0 - norm_position)
            
            # Apply both factors
            modified_torque = torque * increase_factor * low_end_factor
            
            return rpm, modified_torque
        
        self.apply_modification(e85_modification)
    
    def modify_for_exhaust(self, mid_range_boost: float = 1.08, top_end_boost: float = 1.05,
                         mid_range_center: float = 0.5, top_range_center: float = 0.8):
        """
        Modify torque curve to simulate performance exhaust system effects.
        
        Performance exhaust typically improves mid and top-end torque.
        
        Args:
            mid_range_boost: Mid-range torque boost factor
            top_end_boost: Top-end torque boost factor
            mid_range_center: Center point for mid-range boost (0-1 normalized RPM)
            top_range_center: Center point for top-end boost (0-1 normalized RPM)
        """
        def exhaust_modification(rpm, torque):
            # Calculate normalized RPM position (0 to 1)
            norm_position = (rpm - self.idle_rpm) / (self.redline_rpm - self.idle_rpm)
            
            # Apply mid-range boost with Gaussian profile
            mid_boost = (mid_range_boost - 1.0) * np.exp(-15 * (norm_position - mid_range_center) ** 2)
            
            # Apply top-end boost with Gaussian profile
            top_boost = (top_end_boost - 1.0) * np.exp(-15 * (norm_position - top_range_center) ** 2)
            
            # Combine boosts
            total_boost = 1.0 + mid_boost + top_boost
            
            # Apply boost
            modified_torque = torque * total_boost
            
            return rpm, modified_torque
        
        self.apply_modification(exhaust_modification)
    
    def modify_for_intake(self, overall_boost: float = 1.05, resonance_rpm: float = None,
                        resonance_boost: float = 1.10, resonance_width: float = 1000):
        """
        Modify torque curve to simulate performance intake system effects.
        
        Performance intake typically provides an overall boost with specific
        resonance effects at certain RPM ranges.
        
        Args:
            overall_boost: Overall torque boost factor
            resonance_rpm: RPM at which intake resonance occurs
            resonance_boost: Additional boost at resonance RPM
            resonance_width: Width of resonance effect in RPM
        """
        def intake_modification(rpm, torque):
            # Default resonance at 75% of redline if not specified
            if resonance_rpm is None:
                res_rpm = 0.75 * self.redline_rpm
            else:
                res_rpm = resonance_rpm
            
            # Apply overall boost
            modified_torque = torque * overall_boost
            
            # Apply resonance effect
            if resonance_boost > 1.0:
                resonance_factor = (resonance_boost - 1.0) * np.exp(-(rpm - res_rpm) ** 2 / (2 * resonance_width ** 2))
                modified_torque = modified_torque * (1.0 + resonance_factor)
            
            return rpm, modified_torque
        
        self.apply_modification(intake_modification)
    
    def predict_transmission_limited_curve(self, transmission_efficiency: float = 0.92,
                                        drivetrain_inertia: float = 0.05):
        """
        Generate a transmission-limited torque curve accounting for losses.
        
        Args:
            transmission_efficiency: Efficiency of the transmission (0-1)
            drivetrain_inertia: Inertia factor for transient effects
            
        Returns:
            New TorqueCurve object with transmission effects
        """
        if self.rpm_points is None or self.torque_values is None:
            raise ValueError("Torque curve not initialized")
        
        # Apply efficiency losses
        modified_torque = self.torque_values * transmission_efficiency
        
        # Create new curve
        limited_curve = TorqueCurve(self.rpm_points, modified_torque)
        
        return limited_curve
    
    def predict_wheel_torque(self, gear_ratio: float, final_drive_ratio: float,
                          transmission_efficiency: float = 0.92) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate wheel torque for a specific gear ratio.
        
        Args:
            gear_ratio: Gear ratio
            final_drive_ratio: Final drive ratio
            transmission_efficiency: Efficiency of the transmission
            
        Returns:
            Tuple of (wheel_speed_kph, wheel_torque_nm) arrays
        """
        if self.rpm_points is None or self.torque_values is None:
            raise ValueError("Torque curve not initialized")
        
        # Calculate total gear ratio
        total_ratio = gear_ratio * final_drive_ratio
        
        # Apply efficiency and gear ratio to torque
        wheel_torque = self.torque_values * total_ratio * transmission_efficiency
        
        # Convert engine RPM to wheel RPM
        wheel_rpm = self.rpm_points / total_ratio
        
        # Convert wheel RPM to vehicle speed (assuming wheel diameter of 0.52m)
        # Speed (kph) = wheel_rpm * π * wheel_diameter * 60 / 1000
        wheel_diameter = 0.52  # meters (typical for Formula Student)
        wheel_speed_kph = wheel_rpm * np.pi * wheel_diameter * 60 / 1000
        
        return wheel_speed_kph, wheel_torque
    
    def save_to_csv(self, file_path: str):
        """
        Save torque curve to CSV file.
        
        Args:
            file_path: Path to output CSV file
        """
        if self.rpm_points is None or self.torque_values is None:
            raise ValueError("Cannot save uninitialized torque curve")
        
        # Create DataFrame with RPM, Torque, and Power
        power_hp = self.power_values * 1.34102 if self.power_values is not None else None
        
        data = {
            'RPM': self.rpm_points,
            'Torque_Nm': self.torque_values,
            'Power_kW': self.power_values
        }
        
        if power_hp is not None:
            data['Power_hp'] = power_hp
        
        df = pd.DataFrame(data)
        
        # Save to CSV
        df.to_csv(file_path, index=False)
    
    def plot_curve(self, show_power: bool = True, title: str = 'Engine Torque Curve',
                 save_path: Optional[str] = None):
        """
        Plot the torque curve.
        
        Args:
            show_power: Whether to include power curve on secondary y-axis
            title: Plot title
            save_path: If provided, save plot to this path
        """
        if self.rpm_points is None or self.torque_values is None:
            raise ValueError("Cannot plot uninitialized torque curve")
        
        plt.figure(figsize=(10, 6))
        
        # Plot torque
        ax1 = plt.gca()
        ax1.plot(self.rpm_points, self.torque_values, 'b-', linewidth=2, label='Torque (Nm)')
        ax1.set_xlabel('Engine Speed (RPM)')
        ax1.set_ylabel('Torque (Nm)', color='b')
        ax1.tick_params(axis='y', colors='b')
        
        # Plot power on secondary y-axis if requested
        if show_power and self.power_values is not None:
            ax2 = ax1.twinx()
            power_hp = self.power_values * 1.34102  # Convert to hp
            ax2.plot(self.rpm_points, power_hp, 'r-', linewidth=2, label='Power (hp)')
            ax2.set_ylabel('Power (hp)', color='r')
            ax2.tick_params(axis='y', colors='r')
            
            # Add legend
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        else:
            ax1.legend(loc='upper left')
        
        # Find and mark peak torque
        peak_torque_rpm, peak_torque = self.find_peak_torque()
        ax1.plot(peak_torque_rpm, peak_torque, 'bo', markersize=8)
        ax1.text(peak_torque_rpm, peak_torque * 1.05, 
                f'{peak_torque:.1f} Nm @ {peak_torque_rpm:.0f} RPM', 
                color='b', fontweight='bold', ha='center')
        
        # Find and mark peak power if showing power
        if show_power and self.power_values is not None:
            peak_power_rpm, peak_power = self.find_peak_power()
            peak_power_hp = peak_power * 1.34102
            ax2.plot(peak_power_rpm, peak_power_hp, 'ro', markersize=8)
            ax2.text(peak_power_rpm, peak_power_hp * 1.05, 
                    f'{peak_power_hp:.1f} hp @ {peak_power_rpm:.0f} RPM', 
                    color='r', fontweight='bold', ha='center')
        
        # Add title and grid
        plt.title(title)
        plt.grid(True, alpha=0.3)
        
        # Set x-axis limits
        plt.xlim(self.idle_rpm, self.redline_rpm)
        
        plt.tight_layout()
        
        # Save plot if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        plt.show()
    
    def plot_comparison(self, other_curve, labels: Tuple[str, str] = ('Original', 'Modified'),
                      title: str = 'Torque Curve Comparison', save_path: Optional[str] = None):
        """
        Plot comparison between this curve and another.
        
        Args:
            other_curve: Another TorqueCurve object for comparison
            labels: Tuple of (this_curve_label, other_curve_label)
            title: Plot title
            save_path: If provided, save plot to this path
        """
        if (self.rpm_points is None or self.torque_values is None or 
            other_curve.rpm_points is None or other_curve.torque_values is None):
            raise ValueError("Cannot compare uninitialized torque curves")
        
        plt.figure(figsize=(12, 8))
        
        # Plot torque curves
        plt.subplot(2, 1, 1)
        plt.plot(self.rpm_points, self.torque_values, 'b-', linewidth=2, label=f'{labels[0]} Torque')
        plt.plot(other_curve.rpm_points, other_curve.torque_values, 'r-', linewidth=2, label=f'{labels[1]} Torque')
        plt.ylabel('Torque (Nm)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Plot power curves
        plt.subplot(2, 1, 2)
        self_power_hp = self.power_values * 1.34102 if self.power_values is not None else None
        other_power_hp = other_curve.power_values * 1.34102 if other_curve.power_values is not None else None
        
        if self_power_hp is not None:
            plt.plot(self.rpm_points, self_power_hp, 'b-', linewidth=2, label=f'{labels[0]} Power')
        
        if other_power_hp is not None:
            plt.plot(other_curve.rpm_points, other_power_hp, 'r-', linewidth=2, label=f'{labels[1]} Power')
        
        plt.xlabel('Engine Speed (RPM)')
        plt.ylabel('Power (hp)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Set common x-axis limits
        min_rpm = min(self.idle_rpm, other_curve.idle_rpm)
        max_rpm = max(self.redline_rpm, other_curve.redline_rpm)
        plt.xlim(min_rpm, max_rpm)
        
        # Add title
        plt.suptitle(title)
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for suptitle
        
        # Save plot if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        plt.show()


# Example usage
if __name__ == "__main__":
    import os
    import sys
    sys.path.append('..')  # Add parent directory to path
    
    try:
        from engine.motorcycle_engine import MotorcycleEngine
        
        # Create engine
        config_path = os.path.join("configs", "engine", "cbr600f4i.yaml")
        engine = MotorcycleEngine(config_path=config_path)
        
        # Create torque curve from engine
        torque_curve = TorqueCurve()
        torque_curve.load_from_engine(engine)
        
        # Find peak values
        peak_torque_rpm, peak_torque = torque_curve.find_peak_torque()
        peak_power_rpm, peak_power = torque_curve.find_peak_power()
        
        print(f"Peak Torque: {peak_torque:.1f} Nm @ {peak_torque_rpm:.0f} RPM")
        print(f"Peak Power: {peak_power * 1.34102:.1f} hp @ {peak_power_rpm:.0f} RPM")
        
        # Create modified curve for comparison
        modified_curve = TorqueCurve(torque_curve.rpm_points.copy(), torque_curve.torque_values.copy())
        modified_curve.modify_for_e85()
        modified_curve.modify_for_exhaust()
        
        # Plot comparison
        torque_curve.plot_comparison(
            modified_curve, 
            labels=('Stock', 'E85 + Performance Exhaust'),
            title='CBR600F4i Torque Curve Modifications',
            save_path=os.path.join("data", "output", "engine", "torque_comparison.png")
        )
        
        # Calculate optimal shift points
        gear_ratios = [2.750, 2.000, 1.667, 1.444, 1.304, 1.208]  # CBR600F4i gear ratios
        final_drive = 2.688  # example final drive ratio
        wheel_radius = 0.26  # meters (for 13-inch wheels with tires)
        
        shift_points = modified_curve.get_optimal_shift_points(gear_ratios, final_drive, wheel_radius)
        
        print("\nOptimal Shift Points:")
        for i, rpm in enumerate(shift_points):
            print(f"Shift {i+1}→{i+2}: {rpm:.0f} RPM")
        
        # Calculate wheel torque in different gears
        print("\nWheel torque at 50 km/h in different gears:")
        for i, ratio in enumerate(gear_ratios):
            speed, torque = modified_curve.predict_wheel_torque(ratio, final_drive)
            
            # Find torque at 50 km/h
            idx = np.argmin(np.abs(speed - 50))
            if idx < len(torque):
                print(f"Gear {i+1}: {torque[idx]:.1f} Nm")
    
    except ImportError:
        print("MotorcycleEngine class not available, using mock data")
        
        # Create mock torque curve
        rpm_points = np.arange(1000, 14001, 500)
        torque_values = 60 * np.sin((rpm_points - 1000) * np.pi / 15000) + 5
        
        # Create torque curve
        torque_curve = TorqueCurve(rpm_points, torque_values)
        
        # Plot curve
        torque_curve.plot_curve(title="Mock Engine Torque Curve")