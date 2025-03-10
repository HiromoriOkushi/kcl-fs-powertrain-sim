"""
Gearing system module for Formula Student powertrain simulation.

This module provides classes and functions for modeling the gearing system of a
Formula Student car, including the transmission, final drive, and differential.
It calculates gear ratios, wheel torque, theoretical acceleration, and vehicle
speed based on engine and drivetrain parameters.

The system is designed to work with the Honda CBR600F4i motorcycle engine and
includes a chain-driven final drive system with the capability to optimize
gear ratios for Formula Student competitions.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Callable
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("Gearing_System")


class Transmission:
    """
    Transmission model for Formula Student car based on motorcycle gearbox.
    
    This class models the behavior of a sequential motorcycle transmission
    including gear ratios, efficiency, and shift characteristics.
    """
    
    def __init__(self, gear_ratios: List[float], efficiency: float = 0.94):
        """
        Initialize the transmission with gear ratios and efficiency.
        
        Args:
            gear_ratios: List of gear ratios (ordered from first to last gear)
            efficiency: Power transmission efficiency (0-1)
        """
        self.gear_ratios = gear_ratios
        self.num_gears = len(gear_ratios)
        self.efficiency = efficiency
        self.current_gear = 0  # 0 = neutral, 1-N = gear number
        
        # Default parameters
        self.shift_time = 0.040  # seconds (40ms - from CAS system)
        self.neutral_efficiency = 0.0  # No power transfer in neutral
        
        logger.info(f"Transmission initialized with {self.num_gears} gears: {gear_ratios}")
    
    def get_ratio(self, gear: Optional[int] = None) -> float:
        """
        Get the gear ratio for the specified gear.
        
        Args:
            gear: Gear number (1-N) or None for current gear
            
        Returns:
            Gear ratio or 0.0 for neutral
        """
        if gear is None:
            gear = self.current_gear
        
        if gear == 0:  # Neutral
            return 0.0
        elif 1 <= gear <= self.num_gears:
            return self.gear_ratios[gear - 1]
        else:
            logger.warning(f"Invalid gear requested: {gear}")
            return 0.0
    
    def get_efficiency(self, gear: Optional[int] = None) -> float:
        """
        Get the transmission efficiency for the specified gear.
        
        Args:
            gear: Gear number (1-N) or None for current gear
            
        Returns:
            Transmission efficiency for the specified gear
        """
        if gear is None:
            gear = self.current_gear
        
        if gear == 0:  # Neutral
            return self.neutral_efficiency
        elif 1 <= gear <= self.num_gears:
            return self.efficiency
        else:
            logger.warning(f"Invalid gear requested: {gear}")
            return 0.0
    
    def change_gear(self, gear: int) -> bool:
        """
        Change to the specified gear.
        
        Args:
            gear: Target gear number (0-N)
            
        Returns:
            True if gear change was successful, False otherwise
        """
        if 0 <= gear <= self.num_gears:
            self.current_gear = gear
            logger.info(f"Transmission gear changed to {gear}")
            return True
        else:
            logger.warning(f"Invalid gear requested: {gear}")
            return False
    
    def calculate_output_torque(self, input_torque: float, gear: Optional[int] = None) -> float:
        """
        Calculate the output torque based on the input torque and gear ratio.
        
        Args:
            input_torque: Input torque in Nm
            gear: Gear number (1-N) or None for current gear
            
        Returns:
            Output torque in Nm
        """
        ratio = self.get_ratio(gear)
        efficiency = self.get_efficiency(gear)
        
        # Torque multiplication: output_torque = input_torque * ratio * efficiency
        return input_torque * ratio * efficiency
    
    def calculate_output_speed(self, input_speed: float, gear: Optional[int] = None) -> float:
        """
        Calculate the output speed based on the input speed and gear ratio.
        
        Args:
            input_speed: Input speed in RPM
            gear: Gear number (1-N) or None for current gear
            
        Returns:
            Output speed in RPM
        """
        ratio = self.get_ratio(gear)
        
        if ratio == 0.0:  # Neutral
            return 0.0
        
        # Speed reduction: output_speed = input_speed / ratio
        return input_speed / ratio


class FinalDrive:
    """
    Final drive model for Formula Student car.
    
    This class models the behavior of a chain-driven final drive system,
    including sprocket ratios, efficiency, and chain characteristics.
    """
    
    def __init__(self, drive_sprocket_teeth: int, driven_sprocket_teeth: int, 
                 efficiency: float = 0.95, chain_pitch: float = 0.0125):  # 12.5mm standard pitch
        """
        Initialize the final drive with sprocket teeth counts and efficiency.
        
        Args:
            drive_sprocket_teeth: Number of teeth on the drive sprocket
            driven_sprocket_teeth: Number of teeth on the driven sprocket
            efficiency: Power transmission efficiency (0-1)
            chain_pitch: Chain pitch in meters
        """
        self.drive_sprocket_teeth = drive_sprocket_teeth
        self.driven_sprocket_teeth = driven_sprocket_teeth
        self.efficiency = efficiency
        self.chain_pitch = chain_pitch  # meters
        
        # Calculate ratio and chain length
        self.ratio = driven_sprocket_teeth / drive_sprocket_teeth
        
        logger.info(f"Final drive initialized with ratio {self.ratio:.2f} ({drive_sprocket_teeth}:{driven_sprocket_teeth})")
    
    def get_ratio(self) -> float:
        """
        Get the final drive ratio.
        
        Returns:
            Final drive ratio
        """
        return self.ratio
    
    def calculate_output_torque(self, input_torque: float) -> float:
        """
        Calculate the output torque based on the input torque and drive ratio.
        
        Args:
            input_torque: Input torque in Nm
            
        Returns:
            Output torque in Nm
        """
        # Torque multiplication: output_torque = input_torque * ratio * efficiency
        return input_torque * self.ratio * self.efficiency
    
    def calculate_output_speed(self, input_speed: float) -> float:
        """
        Calculate the output speed based on the input speed and drive ratio.
        
        Args:
            input_speed: Input speed in RPM
            
        Returns:
            Output speed in RPM
        """
        # Speed reduction: output_speed = input_speed / ratio
        return input_speed / self.ratio
    
    def calculate_chain_length(self, center_distance: float) -> float:
        """
        Calculate the required chain length.
        
        Args:
            center_distance: Distance between sprocket centers in meters
            
        Returns:
            Chain length in links
        """
        small_sprocket = min(self.drive_sprocket_teeth, self.driven_sprocket_teeth)
        large_sprocket = max(self.drive_sprocket_teeth, self.driven_sprocket_teeth)
        
        # Formula for chain length in links
        chain_length = (
            2 * (center_distance / self.chain_pitch) + 
            (small_sprocket + large_sprocket) / 2 + 
            ((large_sprocket - small_sprocket) ** 2) / (4 * np.pi ** 2 * (center_distance / self.chain_pitch))
        )
        
        return np.ceil(chain_length)  # Round up to the next whole link
    
    def optimize_sprockets(self, target_ratio: float, 
                          min_drive_teeth: int = 12, max_drive_teeth: int = 20,
                          min_driven_teeth: int = 35, max_driven_teeth: int = 65) -> List[Tuple[int, int, float]]:
        """
        Find optimal sprocket combinations for a target ratio.
        
        Args:
            target_ratio: Desired final drive ratio
            min_drive_teeth: Minimum teeth for drive sprocket
            max_drive_teeth: Maximum teeth for drive sprocket
            min_driven_teeth: Minimum teeth for driven sprocket
            max_driven_teeth: Maximum teeth for driven sprocket
            
        Returns:
            List of (drive_teeth, driven_teeth, actual_ratio) tuples, sorted by closeness to target
        """
        combinations = []
        
        for drive in range(min_drive_teeth, max_drive_teeth + 1):
            for driven in range(min_driven_teeth, max_driven_teeth + 1):
                ratio = driven / drive
                error = abs(ratio - target_ratio)
                combinations.append((drive, driven, ratio, error))
        
        # Sort by error (closeness to target ratio)
        combinations.sort(key=lambda x: x[3])
        
        # Return top 5 combinations without the error value
        return [(d, s, r) for d, s, r, _ in combinations[:5]]


class Differential:
    """
    Differential model for Formula Student car.
    
    This class models the behavior of a limited slip differential (LSD) in a
    Formula Student car, including torque distribution and efficiency.
    
    Note: Many Formula Student cars use a solid rear axle without a differential,
    in which case this class can be configured as a fixed (locked) differential.
    """
    
    def __init__(self, ratio: float = 1.0, locked: bool = True, efficiency: float = 0.97):
        """
        Initialize the differential with ratio and configuration.
        
        Args:
            ratio: Differential ratio (typically 1.0 for direct drive)
            locked: Whether the differential is locked (solid axle)
            efficiency: Power transmission efficiency (0-1)
        """
        self.ratio = ratio
        self.locked = locked
        self.efficiency = efficiency
        self.torque_bias_ratio = 1.0
        
        logger.info(f"Differential initialized: {'locked' if locked else 'open'}, ratio: {ratio}")
    
    def get_ratio(self) -> float:
        """
        Get the differential ratio.
        
        Returns:
            Differential ratio
        """
        return self.ratio
    
    def set_torque_bias(self, bias_ratio: float):
        """
        Set the torque bias ratio for limited slip operation.
        
        Args:
            bias_ratio: Torque bias ratio (1.0 = open diff, >1.0 = limited slip)
        """
        if bias_ratio < 1.0:
            logger.warning(f"Invalid torque bias ratio: {bias_ratio}. Must be >= 1.0")
            return
        
        self.torque_bias_ratio = bias_ratio
        logger.info(f"Differential torque bias set to {bias_ratio}")
    
    def calculate_wheel_torques(self, input_torque: float, left_speed: float, right_speed: float) -> Tuple[float, float]:
        """
        Calculate the torque delivered to each wheel.
        
        Args:
            input_torque: Input torque in Nm
            left_speed: Left wheel speed in RPM
            right_speed: Right wheel speed in RPM
            
        Returns:
            Tuple of (left_wheel_torque, right_wheel_torque) in Nm
        """
        # Apply differential ratio and efficiency
        total_torque = input_torque * self.ratio * self.efficiency
        
        if self.locked:
            # Equal torque to both wheels in a locked differential
            return total_torque / 2, total_torque / 2
        
        # For open or limited slip differential
        speed_difference = abs(left_speed - right_speed)
        
        if speed_difference < 0.1:  # Negligible difference
            return total_torque / 2, total_torque / 2
        
        # Determine which wheel is faster
        if left_speed > right_speed:
            faster_wheel = "left"
        else:
            faster_wheel = "right"
        
        # Calculate torque based on limited slip characteristics
        if self.torque_bias_ratio == 1.0:  # Open differential
            return total_torque / 2, total_torque / 2
        else:
            # Simplified limited slip model
            if faster_wheel == "left":
                right_torque = total_torque / (1 + (1 / self.torque_bias_ratio))
                left_torque = total_torque - right_torque
            else:
                left_torque = total_torque / (1 + (1 / self.torque_bias_ratio))
                right_torque = total_torque - left_torque
            
            return left_torque, right_torque


class DrivetrainSystem:
    """
    Complete drivetrain system for Formula Student car.
    
    This class integrates the transmission, final drive, and differential into a
    complete drivetrain system, allowing for calculations of wheel torque, vehicle
    speed, and acceleration based on engine output.
    """
    
    def __init__(self, transmission: Transmission, final_drive: FinalDrive, 
                differential: Optional[Differential] = None,
                wheel_radius: float = 0.2286):  # 9-inch wheel radius (typical for 13-inch wheels)
        """
        Initialize the drivetrain system with components.
        
        Args:
            transmission: Transmission object
            final_drive: FinalDrive object
            differential: Optional Differential object
            wheel_radius: Wheel radius in meters (default 9 inches = 0.2286m)
        """
        self.transmission = transmission
        self.final_drive = final_drive
        self.differential = differential
        self.wheel_radius = wheel_radius
        
        # If no differential provided, create a locked one
        if self.differential is None:
            self.differential = Differential(locked=True)
        
        # Calculate overall ratios for each gear
        self.overall_ratios = self._calculate_overall_ratios()
        
        # Drivetrain inertia parameters - can be adjusted for specific vehicle
        self.drivetrain_inertia = 0.25  # kg·m² (estimated rotational inertia)
        
        logger.info(f"Drivetrain system initialized with wheel radius: {wheel_radius:.3f}m")
        logger.info(f"Overall gear ratios: {[f'{ratio:.2f}' for ratio in self.overall_ratios]}")
    
    def _calculate_overall_ratios(self) -> List[float]:
        """
        Calculate the overall drive ratios for each gear.
        
        Returns:
            List of overall ratios for each gear
        """
        fd_ratio = self.final_drive.get_ratio()
        diff_ratio = self.differential.get_ratio()
        
        # Calculate combined ratio for each gear
        overall_ratios = []
        for i in range(1, self.transmission.num_gears + 1):
            gear_ratio = self.transmission.get_ratio(i)
            overall_ratio = gear_ratio * fd_ratio * diff_ratio
            overall_ratios.append(overall_ratio)
        
        return overall_ratios
    
    def get_overall_ratio(self, gear: Optional[int] = None) -> float:
        """
        Get the overall drive ratio for the specified gear.
        
        Args:
            gear: Gear number (1-N) or None for current gear
            
        Returns:
            Overall drive ratio
        """
        if gear is None:
            gear = self.transmission.current_gear
        
        if gear == 0:  # Neutral
            return 0.0
        elif 1 <= gear <= self.transmission.num_gears:
            return self.overall_ratios[gear - 1]
        else:
            logger.warning(f"Invalid gear requested: {gear}")
            return 0.0
    
    def change_gear(self, gear: int) -> bool:
        """
        Change to the specified gear.
        
        Args:
            gear: Target gear number (0-N)
            
        Returns:
            True if gear change was successful, False otherwise
        """
        return self.transmission.change_gear(gear)
    
    def get_current_gear(self) -> int:
        """
        Get the current transmission gear.
        
        Returns:
            Current gear number (0-N)
        """
        return self.transmission.current_gear
    
    def calculate_wheel_torque(self, engine_torque: float, gear: Optional[int] = None) -> float:
        """
        Calculate the torque at the wheels based on engine torque and current gear.
        
        Args:
            engine_torque: Engine torque in Nm
            gear: Gear number (1-N) or None for current gear
            
        Returns:
            Wheel torque in Nm
        """
        if gear is None:
            gear = self.transmission.current_gear
        
        # Calculate torque through drivetrain
        trans_out = self.transmission.calculate_output_torque(engine_torque, gear)
        final_drive_out = self.final_drive.calculate_output_torque(trans_out)
        
        # For a locked differential or straight calculation without wheel speeds
        if self.differential.locked:
            wheel_torque = final_drive_out
        else:
            # Simplified model when wheel speeds are unknown
            left_torque, right_torque = self.differential.calculate_wheel_torques(
                final_drive_out, 1.0, 1.0)  # Equal speeds
            wheel_torque = left_torque + right_torque
        
        return wheel_torque
    
    def calculate_engine_speed(self, vehicle_speed: float, gear: Optional[int] = None) -> float:
        """
        Calculate the engine speed based on vehicle speed and current gear.
        
        Args:
            vehicle_speed: Vehicle speed in m/s
            gear: Gear number (1-N) or None for current gear
            
        Returns:
            Engine speed in RPM
        """
        if gear is None:
            gear = self.transmission.current_gear
        
        if gear == 0:  # Neutral
            return 0.0
        
        # Calculate wheel rotation speed in RPM
        wheel_rpm = (vehicle_speed * 60) / (2 * np.pi * self.wheel_radius)
        
        # Calculate engine RPM using overall ratio
        overall_ratio = self.get_overall_ratio(gear)
        engine_rpm = wheel_rpm * overall_ratio
        
        return engine_rpm
    
    def calculate_vehicle_speed(self, engine_rpm: float, gear: Optional[int] = None) -> float:
        """
        Calculate the vehicle speed based on engine RPM and current gear.
        
        Args:
            engine_rpm: Engine speed in RPM
            gear: Gear number (1-N) or None for current gear
            
        Returns:
            Vehicle speed in m/s
        """
        if gear is None:
            gear = self.transmission.current_gear
        
        if gear == 0 or engine_rpm == 0:  # Neutral or engine stopped
            return 0.0
        
        # Calculate using overall ratio
        overall_ratio = self.get_overall_ratio(gear)
        wheel_rpm = engine_rpm / overall_ratio
        
        # Convert wheel RPM to vehicle speed
        vehicle_speed = (wheel_rpm * 2 * np.pi * self.wheel_radius) / 60
        
        return vehicle_speed
    
    def calculate_acceleration(self, engine_torque: float, vehicle_mass: float,
                              gear: Optional[int] = None, rolling_resistance: float = 0.015,
                              drag_coefficient: float = 0.8, frontal_area: float = 1.0,
                              air_density: float = 1.225) -> float:
        """
        Calculate vehicle acceleration based on engine torque, vehicle parameters, and current gear.
        
        Args:
            engine_torque: Engine torque in Nm
            vehicle_mass: Vehicle mass in kg
            gear: Gear number (1-N) or None for current gear
            rolling_resistance: Rolling resistance coefficient
            drag_coefficient: Aerodynamic drag coefficient
            frontal_area: Vehicle frontal area in m²
            air_density: Air density in kg/m³
            
        Returns:
            Vehicle acceleration in m/s²
        """
        if gear is None:
            gear = self.transmission.current_gear
        
        if gear == 0:  # Neutral
            return 0.0
        
        # Calculate wheel torque
        wheel_torque = self.calculate_wheel_torque(engine_torque, gear)
        
        # Calculate tractive force at wheels
        tractive_force = wheel_torque / self.wheel_radius
        
        # Calculate current vehicle speed
        # For this we need engine RPM, which is not provided.
        # We'll assume a steady-state calculation at the current operating point.
        # In a full simulation, this would use the current vehicle speed.
        vehicle_speed = 10.0  # Placeholder speed in m/s
        
        # Calculate resistive forces
        rolling_resistance_force = vehicle_mass * 9.81 * rolling_resistance
        aerodynamic_drag = 0.5 * air_density * drag_coefficient * frontal_area * vehicle_speed**2
        total_resistance = rolling_resistance_force + aerodynamic_drag
        
        # Net force
        net_force = tractive_force - total_resistance
        
        # Convert to vehicle acceleration (F = ma)
        acceleration = net_force / vehicle_mass
        
        return acceleration
    
    def calculate_optimal_shift_points(self, engine_torque_curve: Dict[float, float],
                                     vehicle_mass: float, shift_strategy: str = "max_acceleration") -> Dict[int, Tuple[float, float]]:
        """
        Calculate optimal shift points based on engine torque curve and vehicle parameters.
        
        Args:
            engine_torque_curve: Dictionary mapping RPM to torque in Nm
            vehicle_mass: Vehicle mass in kg
            shift_strategy: Strategy for determining shift points ('max_acceleration', 'max_power', 'efficiency')
            
        Returns:
            Dictionary mapping gear numbers to (upshift_rpm, downshift_rpm) tuples
        """
        # Sort the torque curve by RPM
        rpm_points = sorted(engine_torque_curve.keys())
        torque_values = [engine_torque_curve[rpm] for rpm in rpm_points]
        
        # Calculate acceleration profile for each gear
        acceleration_profiles = {}
        for gear in range(1, self.transmission.num_gears + 1):
            accel_profile = []
            for rpm, torque in zip(rpm_points, torque_values):
                # Calculate vehicle speed at this engine RPM
                speed = self.calculate_vehicle_speed(rpm, gear)
                
                # Calculate wheel torque
                wheel_torque = self.calculate_wheel_torque(torque, gear)
                
                # Calculate tractive force
                tractive_force = wheel_torque / self.wheel_radius
                
                # Simplified acceleration (not accounting for drag)
                acceleration = tractive_force / vehicle_mass
                
                accel_profile.append((rpm, speed, acceleration))
            
            acceleration_profiles[gear] = accel_profile
        
        # Determine optimal shift points based on strategy
        shift_points = {}
        
        for gear in range(1, self.transmission.num_gears):
            if gear < self.transmission.num_gears:  # Not the highest gear
                next_gear = gear + 1
                
                # Find intersection point where next gear provides better acceleration
                intersection_rpm = None
                
                current_gear_profile = acceleration_profiles[gear]
                next_gear_profile = acceleration_profiles[next_gear]
                
                # Interpolate next gear profile to match current gear RPM points
                current_rpm_values = [point[0] for point in current_gear_profile]
                current_accel_values = [point[2] for point in current_gear_profile]
                
                next_gear_speeds = [point[1] for point in next_gear_profile]
                next_gear_rpms = [point[0] for point in next_gear_profile]
                next_gear_accels = [point[2] for point in next_gear_profile]
                
                # For each point in the current gear, find equivalent point in next gear
                for i, (rpm, speed, accel) in enumerate(current_gear_profile):
                    # Skip very low RPMs where comparison isn't meaningful
                    if rpm < 3000:
                        continue
                    
                    # Calculate what the RPM would be in the next gear at this speed
                    next_gear_rpm = rpm * (self.get_overall_ratio(gear) / self.get_overall_ratio(next_gear))
                    
                    # Check if this RPM is within the engine's operating range
                    if min(rpm_points) <= next_gear_rpm <= max(rpm_points):
                        # Interpolate acceleration in next gear at this speed
                        next_gear_accel = np.interp(next_gear_rpm, next_gear_rpms, next_gear_accels)
                        
                        # Check if next gear gives better acceleration
                        if shift_strategy == "max_acceleration" and next_gear_accel > accel:
                            intersection_rpm = rpm
                            break
                
                # If no intersection found, use 90% of redline
                if intersection_rpm is None:
                    intersection_rpm = 0.9 * max(rpm_points)
                
                # Determine downshift point (rpm in higher gear where downshifting is beneficial)
                # Typically at lower RPM where torque is falling off
                downshift_rpm = None
                max_torque_rpm = rpm_points[torque_values.index(max(torque_values))]
                downshift_rpm = max(min(rpm_points) + 1000, max_torque_rpm * 0.7)
                
                shift_points[gear] = (intersection_rpm, downshift_rpm)
            else:
                # For highest gear, no upshift point
                shift_points[gear] = (max(rpm_points), min(rpm_points) + 1000)
        
        return shift_points
    
    def plot_speed_profile(self, engine_rpm_range: List[float], save_path: Optional[str] = None):
        """
        Plot the vehicle speed profile for each gear across an engine RPM range.
        
        Args:
            engine_rpm_range: List of engine RPM points to plot
            save_path: Optional path to save the plot
        """
        plt.figure(figsize=(10, 6))
        
        # Plot speed profile for each gear
        for gear in range(1, self.transmission.num_gears + 1):
            speeds = [self.calculate_vehicle_speed(rpm, gear) * 3.6 for rpm in engine_rpm_range]  # Convert to km/h
            plt.plot(engine_rpm_range, speeds, label=f"Gear {gear}")
        
        plt.xlabel("Engine Speed (RPM)")
        plt.ylabel("Vehicle Speed (km/h)")
        plt.title("Vehicle Speed Profile by Gear")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_acceleration_profile(self, engine_torque_curve: Dict[float, float], 
                                vehicle_mass: float, save_path: Optional[str] = None):
        """
        Plot the vehicle acceleration profile for each gear based on the engine torque curve.
        
        Args:
            engine_torque_curve: Dictionary mapping RPM to torque in Nm
            vehicle_mass: Vehicle mass in kg
            save_path: Optional path to save the plot
        """
        plt.figure(figsize=(10, 8))
        
        # Sort the torque curve by RPM
        rpm_points = sorted(engine_torque_curve.keys())
        torque_values = [engine_torque_curve[rpm] for rpm in rpm_points]
        
        # Plot acceleration profile for each gear
        for gear in range(1, self.transmission.num_gears + 1):
            accel_values = []
            speed_values = []
            
            for rpm, torque in zip(rpm_points, torque_values):
                speed = self.calculate_vehicle_speed(rpm, gear) * 3.6  # Convert to km/h
                wheel_torque = self.calculate_wheel_torque(torque, gear)
                
                # Calculate tractive force
                tractive_force = wheel_torque / self.wheel_radius
                
                # Simplified acceleration calculation
                acceleration = tractive_force / vehicle_mass
                
                accel_values.append(acceleration)
                speed_values.append(speed)
            
            plt.plot(speed_values, accel_values, label=f"Gear {gear}")
        
        plt.xlabel("Vehicle Speed (km/h)")
        plt.ylabel("Acceleration (m/s²)")
        plt.title("Vehicle Acceleration Profile by Gear")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def get_drivetrain_specs(self) -> Dict:
        """
        Get specifications of the complete drivetrain.
        
        Returns:
            Dictionary with drivetrain specifications
        """
        specs = {
            "transmission_ratios": self.transmission.gear_ratios,
            "final_drive_ratio": self.final_drive.get_ratio(),
            "differential_ratio": self.differential.get_ratio(),
            "overall_ratios": self.overall_ratios,
            "wheel_radius": self.wheel_radius,
            "drivetrain_efficiency": {
                "transmission": self.transmission.efficiency,
                "final_drive": self.final_drive.efficiency,
                "differential": self.differential.efficiency
            }
        }
        
        return specs


# Example usage
if __name__ == "__main__":
    # Define Honda CBR600F4i gear ratios
    cbr600f4i_gear_ratios = [2.750, 2.000, 1.667, 1.444, 1.304, 1.208]
    
    # Define sprocket configuration (14:53 as mentioned in the Formula Student project)
    drive_sprocket = 14
    driven_sprocket = 53
    
    # Create drivetrain components
    transmission = Transmission(cbr600f4i_gear_ratios)
    final_drive = FinalDrive(drive_sprocket, driven_sprocket)
    differential = Differential(locked=True)  # Most FS cars use a solid axle
    
    # Create complete drivetrain system
    drivetrain = DrivetrainSystem(transmission, final_drive, differential, wheel_radius=0.2286)
    
    # Print drivetrain specifications
    specs = drivetrain.get_drivetrain_specs()
    print("Drivetrain Specifications:")
    print(f"  Transmission ratios: {specs['transmission_ratios']}")
    print(f"  Final drive ratio: {specs['final_drive_ratio']:.2f}")
    print(f"  Overall ratios: {[f'{ratio:.2f}' for ratio in specs['overall_ratios']]}")
    print(f"  Wheel radius: {specs['wheel_radius']:.3f} meters")
    
    # Calculate vehicle speeds at different RPMs in each gear
    print("\nVehicle speeds at 8000 RPM:")
    for gear in range(1, transmission.num_gears + 1):
        speed = drivetrain.calculate_vehicle_speed(8000, gear)
        print(f"  Gear {gear}: {speed:.1f} m/s ({speed*3.6:.1f} km/h)")
    
    # Calculate wheel torque with 60 Nm engine torque in each gear
    print("\nWheel torque with 60 Nm engine torque:")
    for gear in range(1, transmission.num_gears + 1):
        torque = drivetrain.calculate_wheel_torque(60, gear)
        print(f"  Gear {gear}: {torque:.1f} Nm")
    
    # Calculate optimal sprockets for different final drive ratios
    print("\nSprocket optimization for 3.5:1 ratio:")
    optimal_sprockets = final_drive.optimize_sprockets(3.5)
    for drive, driven, ratio in optimal_sprockets:
        print(f"  {drive}:{driven} = {ratio:.3f}")