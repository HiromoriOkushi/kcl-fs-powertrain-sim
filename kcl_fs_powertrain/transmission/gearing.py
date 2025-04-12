"""
Gearing system components: Transmission, Final Drive, Differential, DrivetrainSystem.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Callable
import logging
import itertools # For sprocket optimization
import os
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Gearing")

# Constants
try:
    from ..utils.constants import M_TO_MM
except ImportError:
    M_TO_MM = 1000.0

class Transmission:
    """Models a sequential motorcycle transmission."""
    def __init__(self, gear_ratios: List[float], efficiency: float = 0.96):
        """
        Args:
            gear_ratios: List of ratios [1st, 2nd, ...].
            efficiency: Power transmission efficiency (0-1).
        """
        if not gear_ratios: raise ValueError("Gear ratios list cannot be empty.")
        self.gear_ratios = np.array(gear_ratios)
        self.num_gears = len(gear_ratios)
        self.efficiency = np.clip(efficiency, 0.0, 1.0)
        self.current_gear: int = 0  # 0 = neutral, 1-N = gear number

        logger.info(f"Transmission initialized: {self.num_gears} gears, Efficiency={self.efficiency:.2f}")

    def get_ratio(self, gear: Optional[int] = None) -> float:
        """Get ratio for a specific gear (or current gear)."""
        gear_idx = gear if gear is not None else self.current_gear
        if 1 <= gear_idx <= self.num_gears:
            return self.gear_ratios[gear_idx - 1]
        elif gear_idx == 0:
            return 0.0 # Neutral has no effective ratio for power transmission
        else:
            logger.warning(f"Invalid gear requested: {gear_idx}. Returning 0.")
            return 0.0

    def get_efficiency(self, gear: Optional[int] = None) -> float:
        """Get efficiency for a specific gear."""
        gear_idx = gear if gear is not None else self.current_gear
        return self.efficiency if 1 <= gear_idx <= self.num_gears else 0.0

    def change_gear(self, gear: int) -> bool:
        """Set the current gear."""
        if 0 <= gear <= self.num_gears:
            if self.current_gear != gear:
                 logger.debug(f"Transmission shifting to gear {gear}")
                 self.current_gear = gear
            return True
        else:
            logger.error(f"Cannot change to invalid gear: {gear}")
            return False

    def calculate_output_torque(self, input_torque_nm: float, gear: Optional[int] = None) -> float:
        """Calculate output torque (Nm) after gearbox reduction and losses."""
        # --- Add type check for gear ---
        if not isinstance(gear, (int, np.integer)):
             logger.error(f"Invalid type for gear in calculate_output_torque: {type(gear)}")
             return 0.0
        # ---
        ratio = self.get_ratio(gear)
        efficiency = self.get_efficiency(gear)
        # Torque increases by ratio, decreases by efficiency
        # --- Add type check for ratio/efficiency ---
        if not isinstance(ratio, (float, np.floating)) or not isinstance(efficiency, (float, np.floating)):
            logger.error(f"Invalid types for ratio ({type(ratio)}) or efficiency ({type(efficiency)})")
            return 0.0
        # ---
        return input_torque_nm * ratio * efficiency if ratio != 0 else 0.0

    def calculate_output_speed_rpm(self, input_speed_rpm: float, gear: Optional[int] = None) -> float:
        """Calculate output shaft speed (RPM) after gearbox reduction."""
        ratio = self.get_ratio(gear)
        # Speed decreases by ratio
        return input_speed_rpm / ratio if ratio != 0 else 0.0


class FinalDrive:
    """Models a chain-driven final drive system."""
    def __init__(self, drive_sprocket_teeth: int, driven_sprocket_teeth: int,
                 efficiency: float = 0.97, chain_pitch_m: float = 0.0127): # 0.5 inch
        """
        Args:
            drive_sprocket_teeth: Number of teeth on the drive (front) sprocket.
            driven_sprocket_teeth: Number of teeth on the driven (rear) sprocket.
            efficiency: Chain drive efficiency (0-1).
            chain_pitch_m: Chain pitch in meters.
        """
        if drive_sprocket_teeth <= 0 or driven_sprocket_teeth <= 0:
             raise ValueError("Sprocket teeth numbers must be positive.")
        self.drive_sprocket_teeth = drive_sprocket_teeth
        self.driven_sprocket_teeth = driven_sprocket_teeth
        self.efficiency = np.clip(efficiency, 0.0, 1.0)
        self.chain_pitch_m = chain_pitch_m
        self.ratio = float(driven_sprocket_teeth) / drive_sprocket_teeth

        logger.info(f"Final Drive initialized: {drive_sprocket_teeth}T:{driven_sprocket_teeth}T, Ratio={self.ratio:.3f}, Efficiency={self.efficiency:.2f}")

    def get_ratio(self) -> float:
        """Get the final drive ratio."""
        return self.ratio

    def calculate_output_torque(self, input_torque_nm: float) -> float:
        """Calculate output torque (Nm) after final drive reduction and losses."""
        return input_torque_nm * self.ratio * self.efficiency

    def calculate_output_speed_rpm(self, input_speed_rpm: float) -> float:
        """Calculate output speed (RPM) after final drive reduction."""
        return input_speed_rpm / self.ratio if self.ratio != 0 else 0.0

    def calculate_chain_length_links(self, center_distance_m: float) -> int:
        """Calculate required chain length in links."""
        if center_distance_m <= 0: return 0
        N1 = self.drive_sprocket_teeth
        N2 = self.driven_sprocket_teeth
        C = center_distance_m / self.chain_pitch_m # Center distance in pitches

        # Approximate formula for chain length in links
        length_links = 2 * C + (N1 + N2) / 2.0 + ((N2 - N1)**2) / (4 * (np.pi**2) * C)

        # Chain length must be an even number of links for standard chains
        return int(np.ceil(length_links / 2.0) * 2)

    @staticmethod
    def optimize_sprockets(target_ratio: float,
                          min_drive: int = 12, max_drive: int = 18,
                          min_driven: int = 40, max_driven: int = 60,
                          max_results: int = 5) -> List[Tuple[int, int, float]]:
        """Find optimal (drive, driven) sprocket pairs near a target ratio."""
        combinations = []
        for drive in range(min_drive, max_drive + 1):
            # Calculate the ideal driven sprocket for the target ratio
            ideal_driven = target_ratio * drive
            # Check nearby integer values for the driven sprocket
            for driven in range(max(min_driven, int(ideal_driven)-2), min(max_driven, int(ideal_driven)+3)):
                 ratio = float(driven) / drive
                 error = abs(ratio - target_ratio)
                 combinations.append({'drive': drive, 'driven': driven, 'ratio': ratio, 'error': error})

        # Sort by error
        combinations.sort(key=lambda x: x['error'])

        # Format results
        return [(c['drive'], c['driven'], c['ratio']) for c in combinations[:max_results]]


class Differential:
    """Models a differential (open, locked, or limited slip)."""
    def __init__(self, ratio: float = 1.0, diff_type: str = "LOCKED", # Options: OPEN, LOCKED, LIMITED_SLIP
                 efficiency: float = 0.98, lsd_bias_ratio: float = 2.0, lsd_preload_nm: float = 10.0):
        """
        Args:
            ratio: Differential gear ratio (usually 1.0).
            diff_type: Type of differential ('OPEN', 'LOCKED', 'LIMITED_SLIP').
            efficiency: Power transmission efficiency (0-1).
            lsd_bias_ratio: Torque bias ratio for LSD (ratio of torque to high-traction vs low-traction wheel).
            lsd_preload_nm: Preload torque for clutch-based LSD (Nm).
        """
        self.ratio = ratio
        self.diff_type = diff_type.upper()
        self.efficiency = np.clip(efficiency, 0.0, 1.0)
        self.lsd_bias_ratio = max(1.0, lsd_bias_ratio) # Must be >= 1
        self.lsd_preload_nm = max(0.0, lsd_preload_nm)

        if self.diff_type not in ["OPEN", "LOCKED", "LIMITED_SLIP"]:
            logger.warning(f"Invalid diff_type '{self.diff_type}'. Defaulting to LOCKED.")
            self.diff_type = "LOCKED"

        logger.info(f"Differential initialized: Type={self.diff_type}, Ratio={self.ratio:.2f}, Efficiency={self.efficiency:.2f}")

    def get_ratio(self) -> float:
        """Get the differential gear ratio."""
        return self.ratio

    def calculate_wheel_torques(self, input_torque_nm: float,
                                wheel_slip_left: float = 0.0, wheel_slip_right: float = 0.0,
                                available_grip_left: float = 1.0, available_grip_right: float = 1.0
                                ) -> Tuple[float, float]:
        """
        Calculate torque distribution to left/right wheels based on type and conditions.

        Args:
            input_torque_nm: Torque input to the differential housing.
            wheel_slip_left/right: Slip ratio of each wheel (0-1).
            available_grip_left/right: Factor (0-1) representing available traction.

        Returns:
            Tuple of (left_wheel_torque_nm, right_wheel_torque_nm).
        """
        total_output_torque = input_torque_nm * self.ratio * self.efficiency

        if self.diff_type == "LOCKED":
            # Locked diff distributes torque equally regardless of slip/grip
            return total_output_torque / 2.0, total_output_torque / 2.0

        elif self.diff_type == "OPEN":
            # Open diff torque is limited by the wheel with less grip
            # Estimate max torque based on grip (simplified)
            # Assuming max torque per wheel is proportional to grip * total_output / 2
            max_torque_left = available_grip_left * total_output_torque # Simplistic grip limit
            max_torque_right = available_grip_right * total_output_torque
            limiting_torque = min(max_torque_left, max_torque_right)
            # In an open diff, both wheels get the *minimum* of the limiting torques
            # For simplicity, let's assume it distributes based on the wheel that would slip first
            # but is ultimately limited by the *lower* grip potential.
            # A better model would consider the torque that *causes* slip.
            # Simplified: torque is limited by the lower grip side
            torque_per_wheel = min(total_output_torque / 2.0, limiting_torque / 2.0) # Max half torque, limited by grip
            return torque_per_wheel, torque_per_wheel


        elif self.diff_type == "LIMITED_SLIP":
            # LSD attempts to send more torque to the wheel with more grip
            # Model based on bias ratio and preload

            # Start with equal distribution
            t_left = total_output_torque / 2.0
            t_right = total_output_torque / 2.0

            # Add preload effect (always tries to resist difference)
            preload_effect = self.lsd_preload_nm # Acts against speed difference

            # Calculate speed difference effect (higher slip = more locking)
            # This requires wheel speeds, which are not directly input here.
            # We use slip as a proxy for speed difference potential.
            slip_diff = abs(wheel_slip_left - wheel_slip_right)
            locking_torque = preload_effect + slip_diff * (self.lsd_bias_ratio - 1.0) * 50 # Scaled locking effect based on slip diff and bias

            # Distribute torque: send more to the lower slip (higher grip) wheel
            if wheel_slip_left < wheel_slip_right: # Left has more grip potential
                 transfer_torque = min(locking_torque, t_right) # Can't transfer more than available
                 t_left += transfer_torque
                 t_right -= transfer_torque
            elif wheel_slip_right < wheel_slip_left: # Right has more grip potential
                 transfer_torque = min(locking_torque, t_left)
                 t_right += transfer_torque
                 t_left -= transfer_torque
            # If slips are equal, torque remains ~equal (modified only by preload if model includes it resisting any diff)

            # Ensure torques are not negative and sum correctly (approx)
            t_left = max(0, t_left)
            t_right = max(0, t_right)
            # Rescale slightly if needed due to clamping
            current_total = t_left + t_right
            if current_total > 1e-3:
                scale = total_output_torque / current_total
                t_left *= scale
                t_right *= scale

            return t_left, t_right

        else: # Should not happen
            return total_output_torque / 2.0, total_output_torque / 2.0


class DrivetrainSystem:
    """Integrates Transmission, FinalDrive, and Differential."""
    def __init__(self, transmission: Transmission, final_drive: FinalDrive,
                differential: Optional[Differential] = None,
                wheel_radius_m: float = 0.2286,
                config_path: Optional[str] = None):
        # ... (initialization of components) ...

        self.overall_ratios = self._calculate_overall_ratios() # Calculation should return floats now
        self.num_gears = self.transmission.num_gears

        logger.info(f"Drivetrain System initialized. Wheel Radius: {self.wheel_radius_m*M_TO_MM:.1f} mm")
        # --- Fix Logging ---
        # Ensure the list comprehension formats each ratio as a float string
        ratios_str_list = [f'{r:.3f}' for r in self.overall_ratios]
        logger.info(f" Overall Ratios: {ratios_str_list}") # Log list of formatted strings

    def _calculate_overall_ratios(self) -> List[float]:
        """Calculate overall gear reduction ratio for each gear."""
        ratios = []
        # --- Get component ratios ONCE ---
        fd_ratio = self.final_drive.get_ratio()
        diff_ratio = self.differential.get_ratio()
        # ---
        for i in range(1, self.transmission.num_gears + 1):
             # --- Get transmission ratio ---
             trans_ratio = self.transmission.get_ratio(i)
             # ---
             # Check for potential issues before calculation
             if not isinstance(trans_ratio, (float, np.floating)) or \
                not isinstance(fd_ratio, (float, np.floating)) or \
                not isinstance(diff_ratio, (float, np.floating)):
                 logger.error(f"Non-float ratio encountered for gear {i}: T={trans_ratio}, FD={fd_ratio}, Diff={diff_ratio}")
                 overall_ratio = 0.0 # Assign safe value on error
             else:
                 overall_ratio = trans_ratio * fd_ratio * diff_ratio
             # --- Explicitly cast to float before appending ---
             ratios.append(float(overall_ratio))
             # ---
        return ratios

    def get_overall_ratio(self, gear: Optional[int] = None) -> float:
        """Get overall ratio for a specific gear."""
        gear_idx = gear if gear is not None else self.transmission.current_gear
        logger.debug(f"Getting overall ratio for gear_idx={gear_idx}") # DEBUG
        if 1 <= gear_idx <= self.num_gears:
            # --- Add Debugging ---
            logger.debug(f"  overall_ratios type: {type(self.overall_ratios)}")
            logger.debug(f"  Index: {gear_idx - 1}, Value type: {type(self.overall_ratios[gear_idx - 1])}")
            # ---
            return self.overall_ratios[gear_idx - 1]
        elif gear_idx == 0:
            logger.debug("  Returning 0 for Neutral gear.") # DEBUG
            return 0.0 # Neutral
        else:
             logger.warning(f"Invalid gear {gear_idx} requested for overall ratio.")
             return 0.0

    def change_gear(self, gear: int) -> bool:
        """Change the current gear in the transmission."""
        return self.transmission.change_gear(gear)

    def get_current_gear(self) -> int:
        """Get the currently selected gear."""
        return self.transmission.current_gear

    def calculate_total_wheel_torque(self, engine_torque_nm: float, gear: Optional[int] = None) -> float:
        """Calculate total torque (Nm) delivered to both driven wheels."""
        gear_idx = gear if gear is not None else self.transmission.current_gear
        logger.debug(f"Calculating wheel torque for gear_idx={gear_idx} (type={type(gear_idx)})") # DEBUG

        # --- Check type before calling transmission methods ---
        if not isinstance(gear_idx, (int, np.integer)):
             logger.error(f"Invalid type for gear_idx in calculate_total_wheel_torque: {type(gear_idx)}")
             return 0.0 # Return zero torque if gear is invalid type
        # ---

        trans_out_nm = self.transmission.calculate_output_torque(engine_torque_nm, gear_idx)
        logger.debug(f"  Transmission output torque: {trans_out_nm:.2f}") # DEBUG

        # --- Check type before calling final_drive methods ---
        if not isinstance(trans_out_nm, (float, np.floating)):
            logger.error(f"Invalid type for trans_out_nm: {type(trans_out_nm)}")
            return 0.0
        # ---

        final_drive_out_nm = self.final_drive.calculate_output_torque(trans_out_nm)
        logger.debug(f"  Final drive output torque: {final_drive_out_nm:.2f}") # DEBUG

        # Assuming locked diff or summing both wheels for total output
        # --- Check type before calling differential methods ---
        if not isinstance(final_drive_out_nm, (float, np.floating)):
             logger.error(f"Invalid type for final_drive_out_nm: {type(final_drive_out_nm)}")
             return 0.0
        if not callable(getattr(self.differential, 'get_ratio', None)):
             logger.error("Differential.get_ratio is not callable!")
             return 0.0
        if not isinstance(getattr(self.differential, 'efficiency', None), (float, np.floating)):
            logger.error(f"Differential.efficiency is not a float: {type(self.differential.efficiency)}")
            return 0.0
        # ---

        diff_ratio = self.differential.get_ratio()
        diff_efficiency = self.differential.efficiency
        logger.debug(f"  Differential ratio={diff_ratio}, eff={diff_efficiency}") # DEBUG

        total_wheel_torque = final_drive_out_nm * diff_ratio * diff_efficiency
        logger.debug(f"  Total wheel torque calculated: {total_wheel_torque:.2f}") # DEBUG
        return total_wheel_torque

    def calculate_engine_speed_rpm(self, vehicle_speed_mps: float, gear: Optional[int] = None) -> float:
        """Calculate engine speed (RPM) for a given vehicle speed and gear."""
        gear_idx = gear if gear is not None else self.transmission.current_gear
        logger.debug(f"Calculating engine RPM for speed={vehicle_speed_mps:.2f}, gear={gear_idx}")
        logger.debug(f"  Type of self.get_overall_ratio: {type(getattr(self, 'get_overall_ratio', None))}")

        overall_ratio = self.get_overall_ratio(gear_idx)
        logger.debug(f"  Overall ratio for gear {gear_idx}: {overall_ratio} (type={type(overall_ratio)})")

        if overall_ratio <= 0 or self.wheel_radius_m <= 0: return 0.0 # Avoid division by zero

        # Wheel speed (rad/s) = vehicle_speed / wheel_radius
        wheel_speed_rad_s = vehicle_speed_mps / self.wheel_radius_m
        # Wheel speed (RPM) = wheel_speed (rad/s) * 60 / (2 * pi)
        wheel_speed_rpm = wheel_speed_rad_s * 60.0 / (2.0 * np.pi)

        # Engine speed (RPM) = wheel_speed (RPM) * overall_ratio
        engine_rpm = wheel_speed_rpm * overall_ratio
        logger.debug(f"  Calculated engine RPM: {engine_rpm:.1f}") 
        return engine_rpm

    def calculate_vehicle_speed_mps(self, engine_rpm: float, gear: Optional[int] = None) -> float:
        """Calculate vehicle speed (m/s) for a given engine speed and gear."""
        gear_idx = gear if gear is not None else self.transmission.current_gear
        overall_ratio = self.get_overall_ratio(gear_idx)

        if overall_ratio <= 0 or engine_rpm <= 0: return 0.0

        # Wheel speed (RPM) = engine_speed / overall_ratio
        wheel_speed_rpm = engine_rpm / overall_ratio
        # Wheel speed (rad/s) = wheel_speed (RPM) * (2 * pi) / 60
        wheel_speed_rad_s = wheel_speed_rpm * (2.0 * np.pi) / 60.0

        # Vehicle speed (m/s) = wheel_speed (rad/s) * wheel_radius
        vehicle_speed_mps = wheel_speed_rad_s * self.wheel_radius_m
        return vehicle_speed_mps

    def calculate_tractive_force_N(self, engine_torque_nm: float, gear: Optional[int] = None) -> float:
        """Calculate total tractive force (N) at the driven wheels."""
        if self.wheel_radius_m <= 0: return 0.0
        total_wheel_torque = self.calculate_total_wheel_torque(engine_torque_nm, gear)
        # Force = Torque / Radius
        tractive_force = total_wheel_torque / self.wheel_radius_m
        return tractive_force

    def get_drivetrain_specs(self) -> Dict:
        """Get specifications of the drivetrain system."""
        return {
            "transmission_ratios": self.transmission.gear_ratios.tolist(),
            "num_gears": self.transmission.num_gears,
            "transmission_efficiency": self.transmission.efficiency,
            "final_drive_ratio": self.final_drive.get_ratio(),
            "final_drive_sprockets": f"{self.final_drive.drive_sprocket_teeth}:{self.final_drive.driven_sprocket_teeth}",
            "final_drive_efficiency": self.final_drive.efficiency,
            "differential_type": self.differential.diff_type,
            "differential_ratio": self.differential.get_ratio(),
            "differential_efficiency": self.differential.efficiency,
            "lsd_bias_ratio": self.differential.lsd_bias_ratio if self.differential.diff_type == "LIMITED_SLIP" else None,
            "overall_ratios": self.overall_ratios,
            "wheel_radius_m": self.wheel_radius_m,
            "drivetrain_inertia_kgm2": self.drivetrain_inertia_kgm2
        }

    # --- Plotting Wrappers ---
    def plot_speed_profile(self, engine_rpm_max: float, num_points: int = 100, save_path: Optional[str] = None):
         """Plot vehicle speed vs engine RPM for each gear."""
         from ..utils.plotting import save_plot # Local import
         engine_rpms = np.linspace(0, engine_rpm_max, num_points)
         fig, ax = plt.subplots(figsize=(10, 6))
         colors = plt.cm.viridis(np.linspace(0, 1, self.num_gears))
         for gear in range(1, self.num_gears + 1):
             speeds_mps = [self.calculate_vehicle_speed_mps(rpm, gear) for rpm in engine_rpms]
             speeds_kph = np.array(speeds_mps) * 3.6
             ax.plot(engine_rpms, speeds_kph, label=f"Gear {gear}", color=colors[gear-1], linewidth=2)

         _apply_common_ax_settings(ax, xlabel="Engine Speed (RPM)", ylabel="Vehicle Speed (km/h)", title="Vehicle Speed vs Engine RPM")
         ax.legend()
         ax.set_xlim(left=0)
         ax.set_ylim(bottom=0)
         plt.tight_layout()
         if save_path: save_plot(fig, save_path)
         plt.show()
         plt.close(fig)


    def plot_tractive_force_profile(self, engine_torque_curve: Callable[[float], float],
                                   engine_rpm_max: float, num_points: int = 100,
                                   save_path: Optional[str] = None):
         """Plot tractive force vs vehicle speed for each gear."""
         from ..utils.plotting import save_plot # Local import
         fig, ax = plt.subplots(figsize=(10, 6))
         colors = plt.cm.plasma(np.linspace(0, 1, self.num_gears))
         max_speed_overall = 0

         for gear in range(1, self.num_gears + 1):
             engine_rpms = np.linspace(0, engine_rpm_max, num_points)
             speeds_mps = np.array([self.calculate_vehicle_speed_mps(rpm, gear) for rpm in engine_rpms])
             torques_nm = np.array([engine_torque_curve(rpm) for rpm in engine_rpms])
             forces_N = np.array([self.calculate_tractive_force_N(tq, gear) for tq in torques_nm])

             speeds_kph = speeds_mps * 3.6
             ax.plot(speeds_kph, forces_N, label=f"Gear {gear}", color=colors[gear-1], linewidth=2)
             max_speed_overall = max(max_speed_overall, np.max(speeds_kph))

         _apply_common_ax_settings(ax, xlabel="Vehicle Speed (km/h)", ylabel="Tractive Force (N)", title="Tractive Force vs Vehicle Speed")
         ax.legend()
         ax.set_xlim(left=0, right=max_speed_overall * 1.05)
         ax.set_ylim(bottom=0)
         plt.tight_layout()
         if save_path: save_plot(fig, save_path)
         plt.show()
         plt.close(fig)


# Example Usage
if __name__ == "__main__":
    # Assume config files exist in a relative 'configs/transmission' directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_dir = os.path.abspath(os.path.join(script_dir, '..', '..', 'configs', 'transmission'))
    gearing_config_path = os.path.join(config_dir, 'gearing.yaml')

    if os.path.exists(gearing_config_path):
        print(f"Loading drivetrain config from: {gearing_config_path}")
        try:
            # Create components by loading nested configs if they exist
            # This requires the config file to have sections like 'transmission', 'final_drive'
            with open(gearing_config_path, 'r') as f:
                 full_config = yaml.safe_load(f)

            transmission = Transmission(**full_config.get('transmission', {'gear_ratios': [2.75, 2.0, 1.67, 1.44, 1.3, 1.2]})) # Provide default ratios
            final_drive = FinalDrive(**full_config.get('final_drive', {'drive_sprocket_teeth': 14, 'driven_sprocket_teeth': 53})) # Provide defaults
            differential = Differential(**full_config.get('differential', {'diff_type': 'LOCKED'})) # Provide default
            vehicle_config = full_config.get('vehicle', {})
            wheel_rad = float(vehicle_config.get('wheel_radius', 0.2286)) # Default radius

            drivetrain = DrivetrainSystem(transmission, final_drive, differential, wheel_rad)
            print("\nDrivetrain loaded from config:")

        except Exception as e:
             print(f"Error loading from config: {e}. Using manual setup.")
             # Fallback to manual setup
             transmission = Transmission([2.75, 2.0, 1.67, 1.44, 1.3, 1.2])
             final_drive = FinalDrive(14, 53)
             differential = Differential(locked=True)
             drivetrain = DrivetrainSystem(transmission, final_drive, differential)
    else:
        print("Config file not found. Using manual setup.")
        transmission = Transmission([2.75, 2.0, 1.67, 1.44, 1.3, 1.2])
        final_drive = FinalDrive(14, 53)
        differential = Differential(locked=True)
        drivetrain = DrivetrainSystem(transmission, final_drive, differential)

    print("\nDrivetrain Specifications:")
    print(yaml.dump(drivetrain.get_drivetrain_specs(), default_flow_style=False, sort_keys=False))

    print("\nVehicle speed at 10000 RPM in 3rd gear:",
          f"{drivetrain.calculate_vehicle_speed_mps(10000, 3) * 3.6:.1f} km/h")

    print("\nEngine RPM at 100 km/h in 6th gear:",
          f"{drivetrain.calculate_engine_speed_rpm(100 / 3.6, 6):.0f} RPM")

    print("\nTotal wheel torque with 65 Nm engine torque in 1st gear:",
           f"{drivetrain.calculate_total_wheel_torque(65, 1):.1f} Nm")

    # --- Plotting Examples ---
    plot_dir = os.path.abspath(os.path.join(script_dir, '..', '..', 'plots', 'transmission'))
    os.makedirs(plot_dir, exist_ok=True)

    # Plot speed profile
    drivetrain.plot_speed_profile(engine_rpm_max=14000, save_path=os.path.join(plot_dir, "speed_profile.png"))

    # Plot tractive force (requires a simple torque curve function)
    def example_torque_curve(rpm):
        # Simplified curve peaking around 10500
        peak_rpm = 10500
        max_torque = 65
        idle_rpm = 1300
        # Simple quadratic decay from peak
        torque = max_torque * (1 - ((rpm - peak_rpm) / (14000 - peak_rpm + 1000))**2)
        # Add rise from idle
        rise_factor = np.clip((rpm - idle_rpm) / (peak_rpm - idle_rpm), 0, 1)**0.5
        return np.clip(torque * rise_factor, 0, max_torque)

    drivetrain.plot_tractive_force_profile(example_torque_curve, engine_rpm_max=14000,
                                          save_path=os.path.join(plot_dir, "tractive_force_profile.png"))