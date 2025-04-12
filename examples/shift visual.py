"""
Transmission System Visualization Tool for KCL Formula Student Powertrain

This tool provides interactive visualizations of the transmission system, including:
- Gear ratio visualizations
- Speed profiles across different gears
- Shift strategy comparisons
- Performance impact of different transmission configurations

The tool integrates data from the engine, transmission, and shift strategy modules
to provide a comprehensive view of the powertrain system.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons, CheckButtons
import yaml
from typing import Dict, List, Tuple, Optional
import pandas as pd

# --- Add project root to Python path ---
# This allows running this script directly from the examples directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# -----------------------------------------

# Import modules from the kcl_fs_powertrain package (Use absolute paths)
try:
    from kcl_fs_powertrain.engine.motorcycle_engine import MotorcycleEngine
    from kcl_fs_powertrain.engine.torque_curve import TorqueCurve
    from kcl_fs_powertrain.transmission.gearing import Transmission, FinalDrive, Differential, DrivetrainSystem
    from kcl_fs_powertrain.transmission.shift_strategy import (
        ShiftStrategy, MaxAccelerationStrategy, MaxEfficiencyStrategy,
        EnduranceStrategy, AccelerationEventStrategy, StrategyManager,
        create_formula_student_strategies, ShiftCondition # Removed ShiftCondition from import as it's defined inline
    )
    from kcl_fs_powertrain.transmission.cas_system import CASSystem, ShiftDirection, ShiftState
    from kcl_fs_powertrain.utils.plotting import save_plot, _apply_common_ax_settings # Import needed plot utils
    from kcl_fs_powertrain.utils.constants import KW_TO_HP, HP_TO_KW # Import needed constants
    ENGINE_MODULES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Unable to import KCL Formula Student modules: {e}")
    print("Creating visualization with placeholder data instead.")
    ENGINE_MODULES_AVAILABLE = False
    # Define placeholders for missing classes/enums if needed for the script to run in fallback mode
    class ShiftState: pass
    class ShiftDirection: UP=1; DOWN=-1; NEUTRAL=0
    class ShiftCondition: RPM_THRESHOLD=0 # Dummy value


class TransmissionVisualizer:
    """
    Visualization tool for the Formula Student transmission system.
    """
    
    def __init__(self, config_dir="configs"):
        """
        Initialize the visualizer with configuration files.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = config_dir
        
        # Load configurations
        self.engine_config = self._load_config(os.path.join(config_dir, "engine", "cbr600f4i.yaml"))
        self.gearing_config = self._load_config(os.path.join(config_dir, "transmission", "gearing.yaml"))
        self.shift_strategy_config = self._load_config(os.path.join(config_dir, "transmission", "shift_strategy.yaml"))
        
        # Set up components if modules are available
        if ENGINE_MODULES_AVAILABLE:
            self._setup_components()
        else:
            self._setup_placeholder_data()
    
    def _load_config(self, config_path):
        """
        Load a YAML configuration file.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            Dictionary with configuration data
        """
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Warning: Config file not found: {config_path}")
            return {}
    
    def _setup_components(self):
        """Set up powertrain components using the actual modules."""
        # Create engine
        self.engine = MotorcycleEngine(engine_params=self.engine_config)
        
        # Create transmission components
        gear_ratios = self.gearing_config.get('gear_ratios', [2.750, 2.000, 1.667, 1.444, 1.304, 1.208])
        self.transmission = Transmission(gear_ratios)
        
        # Create final drive
        drive_sprocket = self.gearing_config.get('final_drive', {}).get('drive_sprocket_teeth', 14)
        driven_sprocket = self.gearing_config.get('final_drive', {}).get('driven_sprocket_teeth', 53)
        self.final_drive = FinalDrive(drive_sprocket, driven_sprocket)
        
        # Create differential
        self.differential = Differential(locked=True)  # Most FS cars use a solid axle
        
        # Create drivetrain system
        wheel_radius = self.gearing_config.get('vehicle', {}).get('wheel_radius', 0.2286)
        self.drivetrain = DrivetrainSystem(self.transmission, self.final_drive, self.differential, wheel_radius)
        
        # Create shift strategies
        engine_max_rpm = self.engine.redline
        engine_peak_power_rpm = self.engine.max_power_rpm
        engine_peak_torque_rpm = self.engine.max_torque_rpm
        
        self.strategies = create_formula_student_strategies(
            engine_max_rpm, 
            engine_peak_power_rpm,
            engine_peak_torque_rpm,
            gear_ratios,
            wheel_radius,
            230  # Typical FS car mass in kg
        )
        
        # Create torque curve from engine
        self.torque_curve = TorqueCurve()
        self.torque_curve.load_from_engine(self.engine)
    
    def _setup_placeholder_data(self):
        """Set up placeholder data when modules are not available."""
        # Engine data
        self.engine_rpm_range = np.arange(1000, 14001, 100)
        self.engine_torque = 60 * np.sin((self.engine_rpm_range - 1000) * np.pi / 15000) + 5
        self.engine_power = self.engine_torque * self.engine_rpm_range * 2 * np.pi / 60 / 1000
        
        # Gear ratios
        self.gear_ratios = [2.750, 2.000, 1.667, 1.444, 1.304, 1.208]
        self.num_gears = len(self.gear_ratios)
        
        # Final drive
        self.final_drive_ratio = 53/14
        
        # Overall ratios
        self.overall_ratios = [gr * self.final_drive_ratio for gr in self.gear_ratios]
        
        # Wheel radius
        self.wheel_radius = 0.2286  # meters
        
        # Vehicle parameters
        self.vehicle_mass = 230  # kg
    
    def plot_gear_ratios(self, save_path=None):
        """
        Plot transmission gear ratios and overall drivetrain ratios.
        
        Args:
            save_path: Optional path to save the figure
        """
        if ENGINE_MODULES_AVAILABLE:
            gear_ratios = self.transmission.gear_ratios
            overall_ratios = self.drivetrain.overall_ratios
            num_gears = self.transmission.num_gears
        else:
            gear_ratios = self.gear_ratios
            overall_ratios = self.overall_ratios
            num_gears = self.num_gears
        
        plt.figure(figsize=(12, 8))
        
        # Create bar chart for gear ratios
        ax1 = plt.subplot(2, 1, 1)
        bar_width = 0.35
        index = np.arange(num_gears)
        
        bars1 = ax1.bar(index, gear_ratios, bar_width, label='Transmission Ratio', color='#3498db')
        bars2 = ax1.bar(index + bar_width, overall_ratios, bar_width, label='Overall Ratio', color='#e74c3c')
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        for bar in bars2:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax1.set_xlabel('Gear', fontsize=12)
        ax1.set_ylabel('Ratio', fontsize=12)
        ax1.set_title('Transmission and Overall Gear Ratios', fontsize=14, fontweight='bold')
        ax1.set_xticks(index + bar_width / 2)
        ax1.set_xticklabels([f'Gear {i+1}' for i in range(num_gears)], fontsize=11)
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='y', labelsize=11)
        
        # Create line chart showing relationship between gears
        ax2 = plt.subplot(2, 1, 2)
        ax2.plot(range(1, num_gears + 1), gear_ratios, 'o-', linewidth=2, label='Transmission Ratio', color='#3498db')
        ax2.plot(range(1, num_gears + 1), overall_ratios, 's-', linewidth=2, label='Overall Ratio', color='#e74c3c')
        
        # Calculate percentage difference between gears
        gear_diffs = []
        for i in range(len(gear_ratios) - 1):
            diff = (gear_ratios[i] - gear_ratios[i+1]) / gear_ratios[i] * 100
            gear_diffs.append(diff)
            ax2.annotate(f"{diff:.1f}%", 
                        xy=((i+1 + i+2)/2, (gear_ratios[i] + gear_ratios[i+1])/2), 
                        xytext=(0, 15), 
                        textcoords='offset points',
                        ha='center', fontsize=10, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        ax2.set_xlabel('Gear', fontsize=12)
        ax2.set_ylabel('Ratio', fontsize=12)
        ax2.set_title('Gear Ratio Progression', fontsize=14, fontweight='bold')
        ax2.set_xticks(range(1, num_gears + 1))
        ax2.set_xticklabels([f'Gear {i+1}' for i in range(num_gears)], fontsize=11)
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(labelsize=11)
        
        # Add final drive ratio annotation
        if ENGINE_MODULES_AVAILABLE:
            fd_ratio = self.final_drive.get_ratio()
            drive_teeth = self.final_drive.drive_sprocket_teeth
            driven_teeth = self.final_drive.driven_sprocket_teeth
        else:
            fd_ratio = self.final_drive_ratio
            drive_teeth = 14  # Default values
            driven_teeth = 53
            
        fd_text = f"Final Drive: {drive_teeth}:{driven_teeth} = {fd_ratio:.2f}"
        plt.figtext(0.5, 0.01, fd_text, ha='center', fontsize=12, fontweight='bold',
                  bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5'))
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_speed_profiles(self, save_path=None):
        """
        Plot vehicle speed profiles for each gear across the engine RPM range.
        
        Args:
            save_path: Optional path to save the figure
        """
        plt.figure(figsize=(12, 8))
        
        if ENGINE_MODULES_AVAILABLE:
            rpm_range = np.arange(self.engine.idle_rpm, self.engine.redline + 1, 100)
            num_gears = self.transmission.num_gears
            
            # Plot speed profile for each gear
            for gear in range(1, num_gears + 1):
                speeds = [self.drivetrain.calculate_vehicle_speed(rpm, gear) * 3.6 for rpm in rpm_range]  # km/h
                plt.plot(rpm_range, speeds, linewidth=2, label=f"Gear {gear}")
            
            # Add vertical lines for key engine RPMs
            plt.axvline(x=self.engine.max_torque_rpm, color='b', linestyle='--', alpha=0.5, 
                      label=f'Max Torque: {self.engine.max_torque} Nm @ {self.engine.max_torque_rpm} RPM')
            plt.axvline(x=self.engine.max_power_rpm, color='r', linestyle='--', alpha=0.5,
                      label=f'Max Power: {self.engine.max_power} hp @ {self.engine.max_power_rpm} RPM')
        else:
            rpm_range = self.engine_rpm_range
            
            # Calculate speed for each gear using placeholder data
            for gear in range(1, self.num_gears + 1):
                gear_ratio = self.gear_ratios[gear-1]
                overall_ratio = self.overall_ratios[gear-1]
                
                # Calculate speeds: v = ω * r = (engine_rpm / overall_ratio) * 2π/60 * wheel_radius
                speeds = [(rpm / overall_ratio) * (2 * np.pi / 60) * self.wheel_radius * 3.6 for rpm in rpm_range]
                plt.plot(rpm_range, speeds, linewidth=2, label=f"Gear {gear}")
            
            # Add vertical lines for key engine RPMs (placeholders)
            plt.axvline(x=10500, color='b', linestyle='--', alpha=0.5, 
                      label=f'Max Torque @ 10500 RPM')
            plt.axvline(x=12500, color='r', linestyle='--', alpha=0.5,
                      label=f'Max Power @ 12500 RPM')
        
        plt.xlabel("Engine Speed (RPM)")
        plt.ylabel("Vehicle Speed (km/h)")
        plt.title("Vehicle Speed Profile by Gear")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc='upper left')
        
        # Add annotations for maximum speeds in each gear
        if ENGINE_MODULES_AVAILABLE:
            for gear in range(1, num_gears + 1):
                max_speed = self.drivetrain.calculate_vehicle_speed(self.engine.redline, gear) * 3.6
                plt.text(self.engine.redline + 50, max_speed, f"{max_speed:.1f} km/h", 
                       fontsize=8, ha='left', va='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_engine_with_shift_strategies(self, save_path=None):
        """
        Plot engine performance curves with shift points for different strategies.
        
        Args:
            save_path: Optional path to save the figure
        """
        plt.figure(figsize=(12, 10))
        
        if ENGINE_MODULES_AVAILABLE:
            rpm_range = self.engine.rpm_range
            torque_curve = self.engine.torque_curve
            power_curve = self.engine.power_curve * 1.34102  # Convert to hp
            
            # Plot torque
            ax1 = plt.subplot(2, 1, 1)
            ax1.plot(rpm_range, torque_curve, 'b-', linewidth=2, label='Torque (Nm)')
            ax1.set_ylabel('Torque (Nm)', color='b')
            ax1.tick_params(axis='y', colors='b')
            
            # Plot power on secondary y-axis
            ax1_twin = ax1.twinx()
            ax1_twin.plot(rpm_range, power_curve, 'r-', linewidth=2, label='Power (hp)')
            ax1_twin.set_ylabel('Power (hp)', color='r')
            ax1_twin.tick_params(axis='y', colors='r')
            
            # Add vertical lines for key points
            ax1.axvline(x=self.engine.max_torque_rpm, color='b', linestyle='--', alpha=0.5)
            ax1.axvline(x=self.engine.max_power_rpm, color='r', linestyle='--', alpha=0.5)
            
            # Get strategies
            strategy_names = ["Maximum Acceleration", "Maximum Efficiency", "Endurance"]
            
            # Create vehicle state for shift strategy evaluation
            vehicle_state = {
                'gear_ratios': self.transmission.gear_ratios,
                'wheel_radius': self.drivetrain.wheel_radius,
                'current_gear': 1,
                'final_drive_ratio': self.final_drive.get_ratio()
            }
            
            # Plot shift points for different strategies
            ax2 = plt.subplot(2, 1, 2, sharex=ax1)
            
            for i, strategy_name in enumerate(strategy_names):
                if strategy_name in self.strategies.strategies:
                    strategy = self.strategies.strategies[strategy_name]
                    
                    # Get upshift points
                    upshift_rpms = []
                    for gear in range(1, self.transmission.num_gears):
                        if gear in strategy.upshift_points:
                            for sp in strategy.upshift_points[gear]:
                                if sp.condition_type == ShiftCondition.RPM_THRESHOLD:
                                    upshift_rpms.append((gear, sp.target_gear, sp.threshold_value))
                    
                    # Plot shift points
                    for gear, target_gear, rpm in upshift_rpms:
                        # Calculate speed at this shift point
                        speed = self.drivetrain.calculate_vehicle_speed(rpm, gear) * 3.6  # km/h
                        
                        # Plot vertical line at shift point
                        line_style = ['--', '-.', ':'][i % 3]
                        ax2.axvline(x=rpm, color=f'C{i}', linestyle=line_style, alpha=0.5)
                        
                        # Add annotation
                        ax2.text(rpm, 10 + i*10, f"{strategy_name}\n{gear}→{target_gear}", 
                               color=f'C{i}', rotation=90, fontsize=8, ha='right')
            
            # Plot speed profiles in the second subplot
            for gear in range(1, self.transmission.num_gears + 1):
                speeds = [self.drivetrain.calculate_vehicle_speed(rpm, gear) * 3.6 for rpm in rpm_range]  # km/h
                ax2.plot(rpm_range, speeds, linewidth=2, label=f"Gear {gear}")
        
        else:
            # Use placeholder data
            rpm_range = self.engine_rpm_range
            torque_curve = self.engine_torque
            power_curve = self.engine_power * 1.34102  # Convert to hp
            
            # Plot torque
            ax1 = plt.subplot(2, 1, 1)
            ax1.plot(rpm_range, torque_curve, 'b-', linewidth=2, label='Torque (Nm)')
            ax1.set_ylabel('Torque (Nm)', color='b')
            ax1.tick_params(axis='y', colors='b')
            
            # Plot power on secondary y-axis
            ax1_twin = ax1.twinx()
            ax1_twin.plot(rpm_range, power_curve, 'r-', linewidth=2, label='Power (hp)')
            ax1_twin.set_ylabel('Power (hp)', color='r')
            ax1_twin.tick_params(axis='y', colors='r')
            
            # Add vertical lines for key points
            ax1.axvline(x=10500, color='b', linestyle='--', alpha=0.5, label='Max Torque')
            ax1.axvline(x=12500, color='r', linestyle='--', alpha=0.5, label='Max Power')
            
            # Plot speed profiles in the second subplot
            ax2 = plt.subplot(2, 1, 2, sharex=ax1)
            
            # Add placeholder shift strategies
            strategy_shifts = {
                "Maximum Acceleration": [
                    (1, 2, 13000),
                    (2, 3, 13000),
                    (3, 4, 13000),
                    (4, 5, 13000),
                    (5, 6, 13000)
                ],
                "Maximum Efficiency": [
                    (1, 2, 11000),
                    (2, 3, 11000),
                    (3, 4, 11000),
                    (4, 5, 11000),
                    (5, 6, 11000)
                ],
                "Endurance": [
                    (1, 2, 12000),
                    (2, 3, 12000),
                    (3, 4, 12000),
                    (4, 5, 12000),
                    (5, 6, 12000)
                ]
            }
            
            # Plot shift points for each strategy
            for i, (strategy_name, shifts) in enumerate(strategy_shifts.items()):
                for gear, target_gear, rpm in shifts:
                    # Plot vertical line at shift point
                    line_style = ['--', '-.', ':'][i % 3]
                    ax2.axvline(x=rpm, color=f'C{i}', linestyle=line_style, alpha=0.5)
                    
                    # Add annotation
                    ax2.text(rpm, 10 + i*10, f"{strategy_name}\n{gear}→{target_gear}", 
                           color=f'C{i}', rotation=90, fontsize=8, ha='right')
            
            # Calculate and plot speed profiles
            for gear in range(1, self.num_gears + 1):
                overall_ratio = self.overall_ratios[gear-1]
                speeds = [(rpm / overall_ratio) * (2 * np.pi / 60) * self.wheel_radius * 3.6 for rpm in rpm_range]
                ax2.plot(rpm_range, speeds, linewidth=2, label=f"Gear {gear}")
        
        # Set labels and title for second subplot
        ax2.set_xlabel("Engine Speed (RPM)")
        ax2.set_ylabel("Vehicle Speed (km/h)")
        ax2.set_title("Speed Profiles with Shift Strategies")
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend(loc='upper left')
        
        # Set title for first subplot
        ax1.set_title("Engine Performance Curves")
        
        # Add legend for first subplot
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_twin.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_wheel_torque_comparison(self, save_path=None):
        """
        Plot wheel torque comparison across gears.
        
        Args:
            save_path: Optional path to save the figure
        """
        plt.figure(figsize=(12, 8))
        
        if ENGINE_MODULES_AVAILABLE:
            rpm_range = np.arange(self.engine.idle_rpm, self.engine.redline + 1, 100)
            
            # Calculate engine torque at each RPM
            engine_torque = np.array([self.engine.get_torque(rpm) for rpm in rpm_range])
            
            # Calculate wheel torque for each gear
            wheel_torques = []
            wheel_speeds = []
            
            for gear in range(1, self.transmission.num_gears + 1):
                torques = []
                speeds = []
                
                for rpm, et in zip(rpm_range, engine_torque):
                    wt = self.drivetrain.calculate_wheel_torque(et, gear)
                    ws = self.drivetrain.calculate_vehicle_speed(rpm, gear) * 3.6  # km/h
                    
                    torques.append(wt)
                    speeds.append(ws)
                
                wheel_torques.append(torques)
                wheel_speeds.append(speeds)
        else:
            # Use placeholder data
            rpm_range = self.engine_rpm_range
            
            # Generate placeholder engine torque curve
            engine_torque = self.engine_torque
            
            # Calculate wheel torque for each gear
            wheel_torques = []
            wheel_speeds = []
            
            for gear in range(1, self.num_gears + 1):
                gear_ratio = self.gear_ratios[gear-1]
                overall_ratio = self.overall_ratios[gear-1]
                
                # Simple model: wheel_torque = engine_torque * overall_ratio * efficiency
                efficiency = 0.92
                torques = [et * overall_ratio * efficiency for et in engine_torque]
                
                # Calculate speeds: v = ω * r = (engine_rpm / overall_ratio) * 2π/60 * wheel_radius
                speeds = [(rpm / overall_ratio) * (2 * np.pi / 60) * self.wheel_radius * 3.6 for rpm in rpm_range]
                
                wheel_torques.append(torques)
                wheel_speeds.append(speeds)
        
        # First subplot: wheel torque vs engine RPM
        plt.subplot(2, 1, 1)
        for i, torques in enumerate(wheel_torques):
            plt.plot(rpm_range, torques, linewidth=2, label=f"Gear {i+1}")
        
        plt.xlabel("Engine Speed (RPM)")
        plt.ylabel("Wheel Torque (Nm)")
        plt.title("Wheel Torque vs Engine RPM")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc='upper right')
        
        # Second subplot: wheel torque vs vehicle speed
        plt.subplot(2, 1, 2)
        for i, (speeds, torques) in enumerate(zip(wheel_speeds, wheel_torques)):
            plt.plot(speeds, torques, linewidth=2, label=f"Gear {i+1}")
        
        plt.xlabel("Vehicle Speed (km/h)")
        plt.ylabel("Wheel Torque (Nm)")
        plt.title("Wheel Torque vs Vehicle Speed")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc='upper right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_interactive_comparison(self):
        """Create an interactive plot to compare different gearing configurations."""
        # Define the figure layout
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        plt.subplots_adjust(bottom=0.25)  # Make room for sliders
        
        # Initial values for interactive parameters
        if ENGINE_MODULES_AVAILABLE:
            init_drive_sprocket = self.final_drive.drive_sprocket_teeth
            init_driven_sprocket = self.final_drive.driven_sprocket_teeth
            gear_ratios = self.transmission.gear_ratios.copy()
            rpm_range = np.arange(self.engine.idle_rpm, self.engine.redline + 1, 100)
        else:
            init_drive_sprocket = 14
            init_driven_sprocket = 53
            gear_ratios = self.gear_ratios.copy()
            rpm_range = self.engine_rpm_range
        
        # Initialize plots
        self._update_interactive_plots(axs, init_drive_sprocket, init_driven_sprocket, gear_ratios, rpm_range)
        
        # Create sliders for parameters
        ax_drive = plt.axes([0.25, 0.15, 0.65, 0.03])
        ax_driven = plt.axes([0.25, 0.10, 0.65, 0.03])
        ax_first_gear = plt.axes([0.25, 0.05, 0.65, 0.03])
        
        s_drive = Slider(ax_drive, 'Drive Sprocket', 12, 20, valinit=init_drive_sprocket, valstep=1)
        s_driven = Slider(ax_driven, 'Driven Sprocket', 35, 65, valinit=init_driven_sprocket, valstep=1)
        s_first_gear = Slider(ax_first_gear, 'First Gear Ratio', 2.0, 3.5, valinit=gear_ratios[0], valfmt='%.2f')
        
        # Create reset button
        ax_reset = plt.axes([0.8, 0.01, 0.1, 0.03])
        button_reset = Button(ax_reset, 'Reset')
        
        # Function to update plot
        def update(val):
            drive_sprocket = int(s_drive.val)
            driven_sprocket = int(s_driven.val)
            new_first_gear = s_first_gear.val
            
            # Create modified gear ratios by scaling the original ones
            scale_factor = new_first_gear / gear_ratios[0]
            new_gear_ratios = [gr * scale_factor for gr in gear_ratios]
            
            self._update_interactive_plots(axs, drive_sprocket, driven_sprocket, new_gear_ratios, rpm_range)
            fig.canvas.draw_idle()
        
        # Function to reset to initial values
        def reset(event):
            s_drive.reset()
            s_driven.reset()
            s_first_gear.reset()
        
        # Connect callbacks
        s_drive.on_changed(update)
        s_driven.on_changed(update)
        s_first_gear.on_changed(update)
        button_reset.on_clicked(reset)
        
        plt.suptitle("Interactive Transmission System Comparison", fontsize=16)
        plt.show()
    
    def _update_interactive_plots(self, axs, drive_sprocket, driven_sprocket, gear_ratios, rpm_range):
        """
        Update the interactive plots with new parameters.
        
        Args:
            axs: Array of subplots
            drive_sprocket: Number of teeth on drive sprocket
            driven_sprocket: Number of teeth on driven sprocket
            gear_ratios: List of gear ratios
            rpm_range: Range of engine RPMs to plot
        """
        # Clear all axes
        for ax in axs.flat:
            ax.clear()
        
        # Calculate new ratios
        final_drive_ratio = driven_sprocket / drive_sprocket
        overall_ratios = [gr * final_drive_ratio for gr in gear_ratios]
        num_gears = len(gear_ratios)
        
        # Plot 1: Gear Ratios
        ax1 = axs[0, 0]
        bar_width = 0.35
        index = np.arange(num_gears)
        
        ax1.bar(index, gear_ratios, bar_width, label='Transmission Ratio')
        ax1.bar(index + bar_width, overall_ratios, bar_width, label='Overall Ratio')
        
        ax1.set_xlabel('Gear')
        ax1.set_ylabel('Ratio')
        ax1.set_title('Gear Ratios')
        ax1.set_xticks(index + bar_width / 2)
        ax1.set_xticklabels([f'{i+1}' for i in range(num_gears)])
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Speed Profiles
        ax2 = axs[0, 1]
        
        if ENGINE_MODULES_AVAILABLE:
            # Calculate speed for each gear using the components
            for gear in range(1, num_gears + 1):
                # Create temporary components with new ratios
                temp_transmission = Transmission(gear_ratios)
                temp_final_drive = FinalDrive(drive_sprocket, driven_sprocket)
                temp_drivetrain = DrivetrainSystem(temp_transmission, temp_final_drive, 
                                                self.differential, self.drivetrain.wheel_radius)
                
                speeds = [temp_drivetrain.calculate_vehicle_speed(rpm, gear) * 3.6 for rpm in rpm_range]  # km/h
                ax2.plot(rpm_range, speeds, linewidth=2, label=f"Gear {gear}")
        else:
            # Calculate speed using placeholder method
            wheel_radius = self.wheel_radius
            for gear in range(1, num_gears + 1):
                overall_ratio = overall_ratios[gear-1]
                speeds = [(rpm / overall_ratio) * (2 * np.pi / 60) * wheel_radius * 3.6 for rpm in rpm_range]
                ax2.plot(rpm_range, speeds, linewidth=2, label=f"Gear {gear}")
        
        ax2.set_xlabel("Engine Speed (RPM)")
        ax2.set_ylabel("Vehicle Speed (km/h)")
        ax2.set_title("Speed Profile by Gear")
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=8, loc='upper left')
        
        # Plot 3: Wheel Torque
        ax3 = axs[1, 0]
        
        # Get engine torque
        if ENGINE_MODULES_AVAILABLE:
            engine_torque = np.array([self.engine.get_torque(rpm) for rpm in rpm_range])
        else:
            engine_torque = self.engine_torque
        
        # Create temporary components for each gear
        for gear in range(1, num_gears + 1):
            if ENGINE_MODULES_AVAILABLE:
                # Create temporary components with new ratios
                temp_transmission = Transmission(gear_ratios)
                temp_final_drive = FinalDrive(drive_sprocket, driven_sprocket)
                temp_drivetrain = DrivetrainSystem(temp_transmission, temp_final_drive, 
                                                self.differential, self.drivetrain.wheel_radius)
                
                # Calculate wheel torque
                wheel_torque = [temp_drivetrain.calculate_wheel_torque(et, gear) for et in engine_torque]
            else:
                # Calculate using placeholder method
                overall_ratio = overall_ratios[gear-1]
                efficiency = 0.92
                wheel_torque = [et * overall_ratio * efficiency for et in engine_torque]
            
            ax3.plot(rpm_range, wheel_torque, linewidth=2, label=f"Gear {gear}")
        
        ax3.set_xlabel("Engine Speed (RPM)")
        ax3.set_ylabel("Wheel Torque (Nm)")
        ax3.set_title("Wheel Torque vs Engine RPM")
        ax3.grid(True, alpha=0.3)
        ax3.legend(fontsize=8, loc='upper right')
        
        # Plot 4: Tractive Force vs Speed
        ax4 = axs[1, 1]
        
        # Calculate tractive force and speed for each gear
        for gear in range(1, num_gears + 1):
            if ENGINE_MODULES_AVAILABLE:
                # Create temporary components with new ratios
                temp_transmission = Transmission(gear_ratios)
                temp_final_drive = FinalDrive(drive_sprocket, driven_sprocket)
                temp_drivetrain = DrivetrainSystem(temp_transmission, temp_final_drive, 
                                                self.differential, self.drivetrain.wheel_radius)
                
                # Calculate wheel torque and speed
                wheel_torque = [temp_drivetrain.calculate_wheel_torque(et, gear) for et in engine_torque]
                speeds = [temp_drivetrain.calculate_vehicle_speed(rpm, gear) * 3.6 for rpm in rpm_range]  # km/h
                
                # Calculate tractive force (F = T/r)
                tractive_force = [wt / temp_drivetrain.wheel_radius for wt in wheel_torque]
            else:
                # Calculate using placeholder method
                overall_ratio = overall_ratios[gear-1]
                efficiency = 0.92
                wheel_radius = self.wheel_radius
                
                # Wheel torque
                wheel_torque = [et * overall_ratio * efficiency for et in engine_torque]
                
                # Vehicle speed
                speeds = [(rpm / overall_ratio) * (2 * np.pi / 60) * wheel_radius * 3.6 for rpm in rpm_range]
                
                # Tractive force
                tractive_force = [wt / wheel_radius for wt in wheel_torque]
            
            ax4.plot(speeds, tractive_force, linewidth=2, label=f"Gear {gear}")
        
        ax4.set_xlabel("Vehicle Speed (km/h)")
        ax4.set_ylabel("Tractive Force (N)")
        ax4.set_title("Tractive Force vs Vehicle Speed")
        ax4.grid(True, alpha=0.3)
        ax4.legend(fontsize=8, loc='upper right')
        
        # Add some stats in the upper corner
        fd_text = f"Final Drive: {drive_sprocket}:{driven_sprocket} = {final_drive_ratio:.2f}"
        gear_text = f"Gear Ratios: " + ", ".join([f"{gr:.2f}" for gr in gear_ratios])
        overall_text = f"Overall Ratios: " + ", ".join([f"{gr:.2f}" for gr in overall_ratios])
        
        ax4.text(0.5, 0.02, f"{fd_text}\n{gear_text}\n{overall_text}", 
               transform=ax4.transAxes, fontsize=9, ha='center',
               bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))


    def plot_shifter_comparison(self, save_path=None):
        """
        Plot comparison of different shifting systems, including the CAS system.
        
        Args:
            save_path: Optional path to save the figure
        """
        plt.figure(figsize=(12, 10))
        
        # Define different shifter types for comparison
        shifter_types = [
            "Manual Shifter",
            "Pneumatic Shifter",
            "Clutchless Automatic Shifter (CAS)",
            "Paddle Shifter with Auto-Blip"
        ]
        
        # Define approximate shift times for each system (in milliseconds)
        shift_times = {
            "Manual Shifter": 300,  # 300ms average for manual shifts
            "Pneumatic Shifter": 100,  # 100ms for basic pneumatic system
            "Clutchless Automatic Shifter (CAS)": 45,  # Based on CAS system specs
            "Paddle Shifter with Auto-Blip": 80  # Electronic paddle shifter with auto-blip
        }
        
        # Define key characteristics for radar chart
        characteristics = [
            "Shift Speed",
            "Weight",
            "Complexity",
            "Reliability",
            "Cost",
            "Driver Workload"
        ]
        
        # Scores for each characteristic (1-5 scale, 5 is best)
        # These are approximations for visualization purposes
        scores = {
            "Manual Shifter": [1, 5, 5, 5, 5, 1],
            "Pneumatic Shifter": [3, 3, 3, 3, 3, 4],
            "Clutchless Automatic Shifter (CAS)": [5, 2, 1, 3, 2, 5],
            "Paddle Shifter with Auto-Blip": [4, 3, 2, 4, 3, 4]
        }
        
        # Plot 1: Shift Time Comparison
        ax1 = plt.subplot(2, 2, 1)
        colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
        bars = ax1.bar(shifter_types, [shift_times[s] for s in shifter_types], color=colors)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 5,
                   f'{height:.0f} ms', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax1.set_ylabel('Shift Time (ms)', fontsize=12)
        ax1.set_title('Shift Time Comparison', fontsize=14, fontweight='bold')
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right', fontsize=10)
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
        ax1.tick_params(axis='y', labelsize=10)
        
        # Add explanatory text
        ax1.text(0.5, 0.05, "Shorter shift times = faster acceleration",
               transform=ax1.transAxes, ha='center', fontsize=10, fontweight='bold',
               bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))
        
        # Plot 2: CAS System State Diagram
        ax2 = plt.subplot(2, 2, 2)
        
        # Create a state transition diagram for CAS system
        from matplotlib.patches import FancyArrowPatch
        
        # Define state positions
        states = {
            "IDLE": (0.5, 0.8),
            "PREPARING": (0.2, 0.6),
            "IGNITION_CUT": (0.2, 0.3),
            "SHIFTING": (0.5, 0.1),
            "RECOVERING": (0.8, 0.3),
            "COOLING": (0.8, 0.6),
            "ERROR": (0.5, 0.5)
        }
        
        # Define transitions
        transitions = [
            ("IDLE", "PREPARING", "Request Shift"),
            ("PREPARING", "IGNITION_CUT", "Cut Ignition"),
            ("IGNITION_CUT", "SHIFTING", "Actuate Shift"),
            ("SHIFTING", "RECOVERING", "Shift Complete"),
            ("RECOVERING", "IDLE", "Normal Operation"),
            ("RECOVERING", "COOLING", "High Frequency"),
            ("COOLING", "IDLE", "Cooldown Complete"),
            ("PREPARING", "ERROR", "Error"),
            ("IGNITION_CUT", "ERROR", "Error"),
            ("SHIFTING", "ERROR", "Error"),
            ("ERROR", "IDLE", "Reset")
        ]
        
        # Draw states
        for state, (x, y) in states.items():
            if state == "ERROR":
                color = 'red'
            elif state == "IDLE":
                color = 'green'
            else:
                color = 'skyblue'
            
            circle = plt.Circle((x, y), 0.08, color=color, alpha=0.7)
            ax2.add_patch(circle)
            ax2.text(x, y, state, ha='center', va='center', fontsize=9, fontweight='bold')
        
        # Draw transitions
        for src, dst, label in transitions:
            src_x, src_y = states[src]
            dst_x, dst_y = states[dst]
            
            # Calculate angle for offset to avoid overlapping arrows
            angle = np.arctan2(dst_y - src_y, dst_x - src_x)
            
            # Create curved arrow
            arrow = FancyArrowPatch(
                (src_x, src_y), (dst_x, dst_y),
                connectionstyle=f"arc3,rad=0.2",
                arrowstyle="->", color="gray", lw=1, alpha=0.7
            )
            ax2.add_patch(arrow)
            
            # Add transition label at midpoint
            mid_x = (src_x + dst_x) / 2 + 0.05 * np.sin(angle)
            mid_y = (src_y + dst_y) / 2 - 0.05 * np.cos(angle)
            ax2.text(mid_x, mid_y, label, fontsize=7, ha='center', va='center',
                   bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
        
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.axis('off')
        ax2.set_title('CAS System State Transitions', fontsize=14, fontweight='bold')
        
        # Add diagram legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Normal State'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='skyblue', markersize=10, label='Shifting State'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Error State')
        ]
        ax2.legend(handles=legend_elements, loc='upper right', fontsize=9)
        
        # Plot 3: Radar chart of shifter characteristics
        ax3 = plt.subplot(2, 2, (3, 4), polar=True)
        
        # Number of characteristics
        N = len(characteristics)
        
        # Compute angle for each characteristic
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop
        
        # Set the labels
        ax3.set_xticks(angles[:-1])
        ax3.set_xticklabels(characteristics, fontsize=12, fontweight='bold')
        
        # Set y-axis limits
        ax3.set_ylim(0, 5)
        ax3.set_yticks(np.arange(1, 6))
        ax3.set_yticklabels(['1', '2', '3', '4', '5'], fontsize=10)
        
        # Add grid lines
        ax3.grid(True)
        
        # Plot data for each shifter type
        colors = plt.cm.tab10.colors
        for i, shifter in enumerate(shifter_types):
            values = scores[shifter]
            values += values[:1]  # Close the loop
            
            ax3.plot(angles, values, linewidth=2, linestyle='solid', color=colors[i], label=shifter)
            ax3.fill(angles, values, alpha=0.25, color=colors[i])
        
        ax3.legend(loc='upper right', bbox_to_anchor=(1.3, 1), fontsize=10)
        ax3.set_title('Shifter System Characteristics', fontsize=14, fontweight='bold', y=1.1)
        
        # Add legend explanation
        plt.figtext(0.5, 0.01, "Higher scores are better (5 = excellent, 1 = poor)\nCAS system offers the fastest shifts and lowest driver workload",
                  ha='center', fontsize=11, fontweight='bold', 
                  bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray', boxstyle='round,pad=0.5'))
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.suptitle("Comparison of Formula Student Shifting Systems", fontsize=16, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_cas_performance(self, save_path=None):
        """
        Plot performance characteristics of the CAS system.
        
        Args:
            save_path: Optional path to save the figure
        """
        plt.figure(figsize=(12, 10))
        
        # Plot 1: Shift time distribution by gear
        ax1 = plt.subplot(2, 2, 1)
        
        # CAS system configuration from the configuration file or defaults
        if ENGINE_MODULES_AVAILABLE and hasattr(self, 'shift_strategy_config'):
            cas_config = self.shift_strategy_config.get('cas', {})
            ignition_cut_time = cas_config.get('ignition_cut_time', 20)  # ms
            shift_actuation_time = cas_config.get('shift_actuation_time', 15)  # ms
            throttle_blip_time = cas_config.get('throttle_blip_time', 25)  # ms
            recovery_time = cas_config.get('recovery_time', 10)  # ms
        else:
            # Use default values
            ignition_cut_time = 20
            shift_actuation_time = 15
            throttle_blip_time = 25
            recovery_time = 10
        
        # Create synthetic shift time data
        gears = range(1, 6)  # Gears 1-5 (for upshifts from 1-5)
        
        # Base shift times (increases slightly in higher gears)
        base_times = np.array([ignition_cut_time + shift_actuation_time + recovery_time] * 5)
        
        # Add a small progressive increase for higher gears
        gear_factor = np.array([1.0, 1.05, 1.1, 1.2, 1.3])
        upshift_times = base_times * gear_factor
        
        # Downshift times include throttle blip
        downshift_times = base_times * gear_factor + throttle_blip_time
        
        # Plot data
        width = 0.35
        bars1 = ax1.bar(gears, upshift_times, width, label='Upshift Time', color='#3498db')
        bars2 = ax1.bar([g + width for g in gears], downshift_times, width, label='Downshift Time', color='#e74c3c')
        
        # Add value labels
        for i, v in enumerate(upshift_times):
            ax1.text(i + 1, v + 1, f'{v:.1f}ms', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        for i, v in enumerate(downshift_times):
            ax1.text(i + 1 + width, v + 1, f'{v:.1f}ms', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Add a clear header showing component timing
        header_text = f"CAS System Timing Components:\n"
        header_text += f"• Ignition Cut: {ignition_cut_time}ms\n"
        header_text += f"• Shift Actuation: {shift_actuation_time}ms\n" 
        header_text += f"• Throttle Blip (downshift only): {throttle_blip_time}ms\n"
        header_text += f"• Recovery: {recovery_time}ms"
        
        ax1.text(0.5, 0.95, header_text, transform=ax1.transAxes, ha='center', va='top',
               fontsize=10, bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5'))
        
        ax1.set_xlabel('Gear Change', fontsize=12)
        ax1.set_ylabel('Shift Time (ms)', fontsize=12)
        ax1.set_title('CAS Shift Time by Gear', fontsize=14, fontweight='bold')
        ax1.set_xticks([g + width/2 for g in gears])
        ax1.set_xticklabels([f'Gear {g}→{g+1}' for g in gears], fontsize=10)
        ax1.legend(fontsize=10)
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
        ax1.tick_params(axis='y', labelsize=10)
        
        # Plot 2: Throttle and ignition timing during shift
        ax2 = plt.subplot(2, 2, 2)
        
        # Create time points for shift sequence
        shift_time = np.arange(0, 150)  # 0-150ms
        
        # Create signals for throttle, ignition, and shift actuation
        throttle = np.ones(150)
        ignition = np.ones(150)
        shift_actuator = np.zeros(150)
        
        # Add shift sequence timing
        # 1. Prepare for shift - reduce throttle
        throttle_cut_percentage = 0.8
        prepare_start = 10
        prepare_duration = 10
        throttle[prepare_start:prepare_start+prepare_duration] = np.linspace(1, 1-throttle_cut_percentage, prepare_duration)
        throttle[prepare_start+prepare_duration:prepare_start+prepare_duration+ignition_cut_time] = 1-throttle_cut_percentage
        
        # 2. Cut ignition
        ignition_start = prepare_start + prepare_duration
        ignition[ignition_start:ignition_start+ignition_cut_time] = 0
        
        # 3. Shift actuation
        shift_start = ignition_start + ignition_cut_time
        shift_actuator[shift_start:shift_start+shift_actuation_time] = 1
        
        # 4. Recovery
        recovery_start = shift_start + shift_actuation_time
        throttle[recovery_start:recovery_start+recovery_time] = np.linspace(1-throttle_cut_percentage, 1, recovery_time)
        
        # Plot signals
        ax2.plot(shift_time, throttle, 'g-', label='Throttle', linewidth=2)
        ax2.plot(shift_time, ignition, 'r-', label='Ignition', linewidth=2)
        ax2.plot(shift_time, shift_actuator, 'b-', label='Shift Actuator', linewidth=2)
        
        # Add phase annotations with clearer labels
        ax2.axvspan(prepare_start, ignition_start, alpha=0.2, color='gray')
        ax2.text((prepare_start + ignition_start)/2, 1.1, 'PREPARE', ha='center', fontsize=10, fontweight='bold')
        
        ax2.axvspan(ignition_start, shift_start, alpha=0.2, color='red')
        ax2.text((ignition_start + shift_start)/2, 1.1, 'IGNITION CUT', ha='center', fontsize=10, fontweight='bold')
        
        ax2.axvspan(shift_start, recovery_start, alpha=0.2, color='blue')
        ax2.text((shift_start + recovery_start)/2, 1.1, 'SHIFT', ha='center', fontsize=10, fontweight='bold')
        
        ax2.axvspan(recovery_start, recovery_start+recovery_time, alpha=0.2, color='green')
        ax2.text((recovery_start + recovery_start+recovery_time)/2, 1.1, 'RECOVERY', ha='center', fontsize=10, fontweight='bold')
        
        # Add timing markers
        ax2.text(prepare_start, -0.1, f"t=0", ha='center', fontsize=9)
        ax2.text(ignition_start, -0.1, f"t={ignition_start-prepare_start}ms", ha='center', fontsize=9)
        ax2.text(shift_start, -0.1, f"t={shift_start-prepare_start}ms", ha='center', fontsize=9)
        ax2.text(recovery_start, -0.1, f"t={recovery_start-prepare_start}ms", ha='center', fontsize=9)
        ax2.text(recovery_start+recovery_time, -0.1, f"t={recovery_start+recovery_time-prepare_start}ms", ha='center', fontsize=9)
        
        ax2.set_ylim(-0.2, 1.2)
        ax2.set_xlabel('Time (ms)', fontsize=12)
        ax2.set_ylabel('Signal Level', fontsize=12)
        ax2.set_title('CAS System Shift Sequence Timing', fontsize=14, fontweight='bold')
        ax2.legend(loc='center right', fontsize=10)
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.tick_params(labelsize=10)
        
        # Plot 3: Performance impact comparison
        ax3 = plt.subplot(2, 1, 2)
        
        # Create acceleration data for vehicles with different shift systems
        time_points = np.linspace(0, 5, 500)  # 0-5 seconds
        
        # Base acceleration curve
        def acceleration_curve(t):
            return 15 * (1 - np.exp(-1.5 * t))
        
        # Add speed curves with different shifting systems
        speeds = {
            'Manual Shifter (300ms)': np.zeros_like(time_points),
            'Paddle Shifter (80ms)': np.zeros_like(time_points),
            'CAS System (45ms)': np.zeros_like(time_points)
        }
        
        # Define shift points (approximations)
        shift_points = {
            'Manual Shifter (300ms)': [1.2, 2.4, 3.6],  # seconds
            'Paddle Shifter (80ms)': [1.1, 2.2, 3.3],
            'CAS System (45ms)': [1.0, 2.0, 3.0]
        }
        
        # Define shift durations
        shift_durations = {
            'Manual Shifter (300ms)': 0.3,  # seconds
            'Paddle Shifter (80ms)': 0.08,
            'CAS System (45ms)': 0.045
        }
        
        # Calculate cumulative speed for each system
        for system, speed in speeds.items():
            # Initialize acceleration and current speed
            current_speed = 0
            
            for i, t in enumerate(time_points):
                # Check if we're in a shift
                shifting = False
                for shift_t in shift_points[system]:
                    if shift_t <= t < shift_t + shift_durations[system]:
                        shifting = True
                        break
                
                # Apply acceleration or shifting effect
                if shifting:
                    # During shift, minimal acceleration (coasting)
                    accel = 0.5  # small acceleration during shift
                else:
                    # Normal acceleration based on curve
                    accel = acceleration_curve(t)
                
                # Update speed
                if i > 0:
                    dt = time_points[i] - time_points[i-1]
                    current_speed += accel * dt
                
                # Store speed
                speed[i] = current_speed
        
        # Plot speed curves
        colors = ['#ff9999', '#66b3ff', '#99ff99']
        for i, (system, speed) in enumerate(speeds.items()):
            ax3.plot(time_points, speed, color=colors[i], linewidth=3, label=system)
            
            # Add markers for shift points with improved positioning
            for j, shift_t in enumerate(shift_points[system]):
                shift_index = np.argmin(np.abs(time_points - shift_t))
                shift_speed = speed[shift_index]
                ax3.plot(shift_t, shift_speed, 'o', color=colors[i], markersize=8)
                
                # Add shift duration visualization
                shift_end_index = np.argmin(np.abs(time_points - (shift_t + shift_durations[system])))
                shift_end_speed = speed[shift_end_index]
                ax3.plot([shift_t, shift_t + shift_durations[system]], 
                       [shift_speed, shift_end_speed], '-', color=colors[i], alpha=0.8, linewidth=2)
                
                # Stagger shift labels with vertical offsets to avoid overlap
                vert_offset = -6 - (i * 4)  # Different offset for each system
                # Additionally alternate between top and bottom for consecutive shifts
                if j % 2 == 1:
                    vert_offset = 6 + (i * 4)
                    va_setting = 'bottom'
                else:
                    va_setting = 'top'
                
                # Compact label showing just the shift time
                ax3.text(shift_t + shift_durations[system]/2, shift_speed + vert_offset, 
                       f"{shift_durations[system]*1000:.0f}ms", 
                       ha='center', va=va_setting, fontsize=8, color=colors[i],
                       bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
        
        # Add 75m finish line with staggered labels to prevent overlap
        # Assuming a 75m finish is around 4-4.5 seconds in a good FS car
        finish_line_times = []
        for system, speed in speeds.items():
            # Find time at which distance reaches 75m
            distance = np.cumsum(speed * np.gradient(time_points))
            finish_index = np.argmin(np.abs(distance - 75))
            finish_time = time_points[finish_index]
            finish_speed = speed[finish_index]
            finish_line_times.append((system, finish_time, finish_speed))
        
        # Sort systems by finish time for better label placement
        finish_line_times.sort(key=lambda x: x[1])
        
        # Add finish markers with staggered vertical positioning
        for i, (system, finish_time, finish_speed) in enumerate(finish_line_times):
            sys_idx = list(speeds.keys()).index(system)
            
            # Add vertical finish line
            ax3.axvline(x=finish_time, color=colors[sys_idx], 
                      linestyle='--', alpha=0.7, linewidth=2)
            
            # Place labels with staggered vertical positions and consistent spacing
            # Use different offsets for each system to avoid overlap
            # Place the fastest system at the top, second fastest in the middle, slowest at bottom
            if i == 0:  # Fastest (likely CAS)
                vert_position = finish_speed + 8
                va_setting = 'bottom'
            elif i == 1:  # Middle
                vert_position = finish_speed - 8
                va_setting = 'top'
            else:  # Slowest
                vert_position = finish_speed - 16
                va_setting = 'top'
            
            # Format finish time with distinct background box per system
            ax3.text(finish_time + 0.1, vert_position, 
                   f"{system}: {finish_time:.2f}s", 
                   color=colors[sys_idx], fontsize=10, fontweight='bold',
                   ha='left', va=va_setting,
                   bbox=dict(facecolor='white', alpha=0.8, edgecolor=colors[sys_idx], 
                           boxstyle='round,pad=0.3', linewidth=1.5))
        
        # Add annotation explaining the importance of shift time
        # Calculate time difference between fastest (CAS) and slowest (Manual) system
        # Get the systems in finish time order
        sorted_systems = [x[0] for x in sorted(finish_line_times, key=lambda x: x[1])]
        fastest_system = sorted_systems[0]
        slowest_system = sorted_systems[-1]
        fastest_time = finish_line_times[0][1]  # We sorted by time, so first is fastest
        slowest_time = finish_line_times[-1][1]  # Last is slowest
        time_diff = slowest_time - fastest_time
        
        # Add annotation box in an empty area of the plot
        ax3.text(0.5, 0.05, f"{fastest_system} is {time_diff:.2f}s faster over 75m\ncompared to {slowest_system}",
               transform=ax3.transAxes, ha='center', va='center', fontsize=12, fontweight='bold',
               bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray', boxstyle='round,pad=0.5', linewidth=1.5))
        
        ax3.set_xlabel('Time (s)', fontsize=12)
        ax3.set_ylabel('Speed (km/h)', fontsize=12)
        ax3.set_title('Impact of Shifting System on Acceleration Performance', fontsize=14, fontweight='bold')
        ax3.legend(fontsize=11, loc='upper left')
        ax3.grid(True, linestyle='--', alpha=0.7)
        ax3.tick_params(labelsize=10)
        
        plt.tight_layout(rect=[0, 0.0, 1, 0.97])
        plt.suptitle("CAS System Performance Analysis", fontsize=16, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
    def plot_interactive_cas(self):
        """Create an interactive plot to explore CAS system parameters."""
        # Define the figure layout
        fig, axs = plt.subplots(2, 1, figsize=(12, 10))
        plt.subplots_adjust(bottom=0.35)  # Make room for sliders
        
        # Initial values for interactive parameters
        init_ignition_cut_time = 20  # ms
        init_shift_actuation_time = 15  # ms
        init_throttle_blip_time = 25  # ms
        init_recovery_time = 10  # ms
        init_throttle_cut_pct = 80  # %
        init_rpm = 10000  # RPM
        
        # Plot initial state
        self._update_interactive_cas_plots(axs, 
                                          init_ignition_cut_time, 
                                          init_shift_actuation_time,
                                          init_throttle_blip_time,
                                          init_recovery_time,
                                          init_throttle_cut_pct,
                                          init_rpm)
        
        # Create sliders for parameters
        ax_ignition_cut = plt.axes([0.2, 0.25, 0.65, 0.03])
        ax_actuation = plt.axes([0.2, 0.20, 0.65, 0.03])
        ax_throttle_blip = plt.axes([0.2, 0.15, 0.65, 0.03])
        ax_recovery = plt.axes([0.2, 0.10, 0.65, 0.03])
        ax_throttle_cut = plt.axes([0.2, 0.05, 0.65, 0.03])
        ax_rpm = plt.axes([0.2, 0.30, 0.65, 0.03])
        
        s_ignition_cut = Slider(ax_ignition_cut, 'Ignition Cut (ms)', 5, 40, valinit=init_ignition_cut_time, valstep=1)
        s_actuation = Slider(ax_actuation, 'Shift Actuation (ms)', 5, 30, valinit=init_shift_actuation_time, valstep=1)
        s_throttle_blip = Slider(ax_throttle_blip, 'Throttle Blip (ms)', 10, 50, valinit=init_throttle_blip_time, valstep=1)
        s_recovery = Slider(ax_recovery, 'Recovery (ms)', 5, 30, valinit=init_recovery_time, valstep=1)
        s_throttle_cut = Slider(ax_throttle_cut, 'Throttle Cut (%)', 20, 100, valinit=init_throttle_cut_pct, valstep=5)
        s_rpm = Slider(ax_rpm, 'Engine RPM', 4000, 14000, valinit=init_rpm, valstep=500)
        
        # Create reset button
        ax_reset = plt.axes([0.8, 0.01, 0.1, 0.03])
        button_reset = Button(ax_reset, 'Reset')
        
        # Function to update plot
        def update(val):
            ignition_cut_time = s_ignition_cut.val
            shift_actuation_time = s_actuation.val
            throttle_blip_time = s_throttle_blip.val
            recovery_time = s_recovery.val
            throttle_cut_pct = s_throttle_cut.val
            rpm = s_rpm.val
            
            self._update_interactive_cas_plots(axs, 
                                               ignition_cut_time, 
                                               shift_actuation_time,
                                               throttle_blip_time,
                                               recovery_time,
                                               throttle_cut_pct,
                                               rpm)
            fig.canvas.draw_idle()
        
        # Function to reset to initial values
        def reset(event):
            s_ignition_cut.reset()
            s_actuation.reset()
            s_throttle_blip.reset()
            s_recovery.reset()
            s_throttle_cut.reset()
            s_rpm.reset()
        
        # Connect callbacks
        s_ignition_cut.on_changed(update)
        s_actuation.on_changed(update)
        s_throttle_blip.on_changed(update)
        s_recovery.on_changed(update)
        s_throttle_cut.on_changed(update)
        s_rpm.on_changed(update)
        button_reset.on_clicked(reset)
        
        plt.suptitle("Interactive CAS System Exploration", fontsize=16)
        plt.show()
    
    def _update_interactive_cas_plots(self, axs, 
                                   ignition_cut_time, 
                                   shift_actuation_time,
                                   throttle_blip_time,
                                   recovery_time, 
                                   throttle_cut_pct,
                                   rpm):
        """
        Update the interactive CAS system plots with new parameters.
        
        Args:
            axs: Array of subplots
            ignition_cut_time: Ignition cut duration in ms
            shift_actuation_time: Shift actuation duration in ms
            throttle_blip_time: Throttle blip duration in ms
            recovery_time: Recovery duration in ms
            throttle_cut_pct: Throttle cut percentage
            rpm: Engine RPM
        """
        # Clear all axes
        for ax in axs:
            ax.clear()
        
        # Total shift time
        total_shift_time = ignition_cut_time + shift_actuation_time + recovery_time
        
        # Plot 1: Shift sequence timing diagram
        ax1 = axs[0]
        
        # Create time points for shift sequence
        # Add some buffer at start and end
        buffer = 20
        shift_time = np.arange(0, total_shift_time + 2*buffer)  # ms
        
        # Create signals for throttle, ignition, and shift actuation
        throttle = np.ones(len(shift_time))
        ignition = np.ones(len(shift_time))
        shift_actuator = np.zeros(len(shift_time))
        
        # Add shift sequence timing
        # 1. Prepare for shift - reduce throttle
        throttle_cut = throttle_cut_pct / 100
        prepare_start = buffer
        throttle[prepare_start:] = 1 - throttle_cut
        
        # 2. Cut ignition
        ignition_start = prepare_start
        ignition[ignition_start:ignition_start+ignition_cut_time] = 0
        
        # 3. Shift actuation
        shift_start = ignition_start + ignition_cut_time
        shift_actuator[shift_start:shift_start+shift_actuation_time] = 1
        
        # 4. Recovery
        recovery_start = shift_start + shift_actuation_time
        recovery_end = recovery_start + recovery_time
        throttle[recovery_start:recovery_end] = np.linspace(1-throttle_cut, 1, recovery_time)
        throttle[recovery_end:] = 1
        
        # Plot signals
        ax1.plot(shift_time, throttle, 'g-', label='Throttle')
        ax1.plot(shift_time, ignition, 'r-', label='Ignition')
        ax1.plot(shift_time, shift_actuator, 'b-', label='Shift Actuator')
        
        # Add phase annotations
        ax1.axvspan(prepare_start, ignition_start + ignition_cut_time, alpha=0.2, color='red', label='Ignition Cut')
        ax1.axvspan(shift_start, recovery_start, alpha=0.2, color='blue', label='Shift Actuation')
        ax1.axvspan(recovery_start, recovery_end, alpha=0.2, color='green', label='Recovery')
        
        # Add total shift time annotation with improved visibility
        ax1.text(0.5, 0.95, f"Total Shift Time: {total_shift_time} ms", 
               transform=ax1.transAxes, fontsize=14, ha='center', fontweight='bold',
               bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5'))
        
        ax1.set_xlabel('Time (ms)', fontsize=12)
        ax1.set_ylabel('Signal Level', fontsize=12)
        ax1.set_title('CAS System Shift Sequence Timing', fontsize=14, fontweight='bold')
        ax1.legend(loc='center right', fontsize=11, framealpha=0.8)
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.tick_params(labelsize=11)
        
        # Plot 2: Engine RPM during shift
        ax2 = axs[1]
        
        # Create engine RPM profile during shift
        engine_rpm = np.ones(len(shift_time)) * rpm
        
        # Calculate RPM drop during ignition cut (simplified model)
        # RPM drops with rate dependent on engine inertia and speed
        rpm_drop_rate = rpm * 0.015  # RPM drop per ms (simplified approximation)
        
        # Apply RPM drop during ignition cut
        for i in range(ignition_cut_time):
            if ignition_start + i < len(engine_rpm):
                engine_rpm[ignition_start + i] = rpm - (i * rpm_drop_rate)
        
        # Set RPM at shift point
        shift_rpm = rpm - (ignition_cut_time * rpm_drop_rate)
        
        # Get next gear RPM (assuming 1.2:1 ratio between gears - simplified approximation)
        # For a typical motorcycle transmission
        gear_ratio = 1.2
        next_gear_rpm = shift_rpm / gear_ratio
        
        # Set RPM in next gear
        engine_rpm[shift_start:] = next_gear_rpm
        
        # Apply RPM recovery after shift
        rpm_recovery_rate = (next_gear_rpm * 0.1) / recovery_time  # RPM increase per ms
        for i in range(recovery_time):
            if recovery_start + i < len(engine_rpm):
                engine_rpm[recovery_start + i] = next_gear_rpm + (i * rpm_recovery_rate)
        
        # Set stable RPM after recovery
        stable_rpm = next_gear_rpm + (recovery_time * rpm_recovery_rate)
        engine_rpm[recovery_end:] = stable_rpm
        
        # Plot RPM profile
        ax2.plot(shift_time, engine_rpm, 'r-', linewidth=2)
        
        # Add phase annotations
        ax2.axvspan(prepare_start, ignition_start + ignition_cut_time, alpha=0.2, color='red')
        ax2.axvspan(shift_start, recovery_start, alpha=0.2, color='blue')
        ax2.axvspan(recovery_start, recovery_end, alpha=0.2, color='green')
        
        # Add RPM annotations
        ax2.text(prepare_start - 5, rpm, f"{rpm:.0f} RPM", ha='right', va='center')
        ax2.text(shift_start + 5, next_gear_rpm, f"{next_gear_rpm:.0f} RPM", ha='left', va='center')
        ax2.text(recovery_end + 5, stable_rpm, f"{stable_rpm:.0f} RPM", ha='left', va='center')
        
        # Add gear shift annotation
        current_gear = 3  # Arbitrary example gear
        ax2.text(0.5, 0.95, f"Shift: Gear {current_gear} → {current_gear+1}", 
               transform=ax2.transAxes, fontsize=12, ha='center',
               bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))
        
        ax2.set_xlabel('Time (ms)')
        ax2.set_ylabel('Engine RPM')
        ax2.set_title('Engine RPM During Shift')
        ax2.grid(True, linestyle='--', alpha=0.7)

def run_visualization():
    """Run the complete visualization suite."""
    visualizer = TransmissionVisualizer()
    
    print("\n┌─────────────────────────────────────────────────────────┐")
    print("│  KCL Formula Student Transmission System Visualization  │")
    print("└─────────────────────────────────────────────────────────┘")
    
    print("\n[1/8] Generating gear ratio visualization...")
    visualizer.plot_gear_ratios(save_path="gear_ratios.png")
    print("     ✓ Complete! (saved as gear_ratios.png)")
    
    print("\n[2/8] Generating speed profile visualization...")
    visualizer.plot_speed_profiles(save_path="speed_profiles.png")
    print("     ✓ Complete! (saved as speed_profiles.png)")
    
    print("\n[3/8] Generating engine and shift strategy visualization...")
    visualizer.plot_engine_with_shift_strategies(save_path="shift_strategies.png")
    print("     ✓ Complete! (saved as shift_strategies.png)")
    
    print("\n[4/8] Generating wheel torque comparison...")
    visualizer.plot_wheel_torque_comparison(save_path="wheel_torque.png")
    print("     ✓ Complete! (saved as wheel_torque.png)")
    
    print("\n[5/8] Generating shifter comparison...")
    visualizer.plot_shifter_comparison(save_path="shifter_comparison.png")
    print("     ✓ Complete! (saved as shifter_comparison.png)")
    
    print("\n[6/8] Generating CAS system performance visualization...")
    visualizer.plot_cas_performance(save_path="cas_performance.png")
    print("     ✓ Complete! (saved as cas_performance.png)")
    
    print("\n[7/8] Launching interactive transmission comparison tool...")
    print("     ► Close the window when finished to continue to the next visualization")
    visualizer.plot_interactive_comparison()
    print("     ✓ Interactive visualization complete!")
    
    print("\n[8/8] Launching interactive CAS system exploration tool...")
    print("     ► Close the window when finished to complete the visualization suite")
    visualizer.plot_interactive_cas()
    print("     ✓ Interactive visualization complete!")
    
    print("\n┌─────────────────────────────────────────────────────────┐")
    print("│  All visualizations complete! Summary of outputs:        │")
    print("│                                                          │")
    print("│  • gear_ratios.png - Gear ratio visualization            │")
    print("│  • speed_profiles.png - Vehicle speed in each gear       │")
    print("│  • shift_strategies.png - Engine performance with shifts │")
    print("│  • wheel_torque.png - Wheel torque comparison            │")
    print("│  • shifter_comparison.png - Different shifter systems    │")
    print("│  • cas_performance.png - CAS system performance analysis │")
    print("└─────────────────────────────────────────────────────────┘")


if __name__ == "__main__":
    run_visualization()