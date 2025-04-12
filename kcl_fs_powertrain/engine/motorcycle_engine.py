"""
Motorcycle Engine module for Formula Student powertrain simulation.

This module models the behavior of a Honda CBR600F4i motorcycle engine modified
for Formula Student competition. It includes torque/power curves, thermal modeling,
and integration with the transmission system.
"""

import os
import numpy as np
import yaml
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union
from scipy.interpolate import interp1d, CubicSpline
import logging

# Import from local modules
try:
    from .engine_thermal import EngineHeatModel, ThermalConfig
except ImportError:
    class EngineHeatModel: pass
    class ThermalConfig: pass
    EngineHeatModel = None
    ThermalConfig = None
    
# Import constants and fuel properties
try:
    from ..utils.constants import HP_TO_KW, KW_TO_HP
    from .fuel_systems import FuelProperties, FuelType
except ImportError:
    HP_TO_KW = 0.7457
    KW_TO_HP = 1 / HP_TO_KW
    class FuelProperties: pass
    class FuelType: E85 = 0 
    FuelProperties = None 

logger = logging.getLogger("Motorcycle Engine")

class MotorcycleEngine:
    """
    Models a motorcycle engine, specifically tuned for Formula Student context.

    Simulates torque/power generation based on RPM, throttle, and temperature,
    along with basic thermal behavior and fuel consumption estimates.
    """

    def __init__(self, config_path: Optional[str] = None, engine_params: Optional[Dict] = None):
        """
        Initialize the MotorcycleEngine.

        Args:
            config_path: Path to YAML configuration file.
            engine_params: Dictionary of engine parameters (used if config_path is None).
        """
        # --- Default Engine Specifications (Honda CBR600F4i base) ---
        self.make: str = "Honda"
        self.model: str = "CBR600F4i"
        self.displacement_cc: float = 599.0
        self.cylinders: int = 4
        self.configuration: str = "Inline-4"
        self.compression_ratio: float = 12.0
        self.bore_mm: float = 67.0
        self.stroke_mm: float = 42.5
        self.valves_per_cylinder: int = 4
        self.valve_train_type: str = 'DOHC'
        self.max_power_hp: float = 110.0 # Stock value, adjust based on FS restrictor/tuning
        self.max_power_rpm: float = 12500.0
        self.max_torque_nm: float = 65.0 # Stock value
        self.max_torque_rpm: float = 10500.0
        self.redline_rpm: float = 14000.0
        self.idle_rpm: float = 1300.0
        self.weight_kg: float = 57.0 # Dry weight estimate
        self.fuel_type: str = 'E85' # Default for FS

        # --- Engine State Variables ---
        self.current_rpm: float = self.idle_rpm
        self.throttle_position: float = 0.0  # 0.0 to 1.0
        self.engine_temperature: float = 25.0  # °C, overall engine block temp
        self.oil_temperature: float = 25.0  # °C
        self.coolant_temperature: float = 25.0  # °C
        self.thermal_factor: float = 1.0 # Performance multiplier based on temperature (1.0 = optimal)

        # --- Performance Maps & Functions ---
        self.rpm_range: Optional[np.ndarray] = None
        self.torque_curve: Optional[np.ndarray] = None # Base torque curve at optimal conditions
        self.power_curve_kw: Optional[np.ndarray] = None # Base power curve in kW
        self.torque_function: Optional[Callable] = None # Interpolated function
        self.power_function_kw: Optional[Callable] = None # Interpolated function

        # --- Thermal Model ---
        # Use default config initially, can be overridden
        if ThermalConfig:
            self.thermal_config = ThermalConfig()
        else:
            self.thermal_config = None
            logger.warning("ThermalConfig not available, thermal model disabled.")
        
        if EngineHeatModel and self.thermal_config:
            self.heat_model = EngineHeatModel(self.thermal_config, self) # Pass self for potential access
        else:
            self.heat_model = None
            logger.warning("EngineHeatModel not available, thermal model disabled.")
        
        if FuelProperties and hasattr(self, 'fuel_type'):
            try:
                ft = FuelType[self.fuel_type.upper()]
                self.fuel_properties = FuelProperties(ft)
            except (KeyError, TypeError):
                logger.warning(f"Could not create FuelProperties for type '{self.fuel_type}'. Using defaults for heat calc.")
                self.fuel_properties = None
        else:
            self.fuel_properties = None
            
        # --- Load Configuration ---
        self.config = {} # Store loaded config
        if config_path:
            self.load_config(config_path)
        elif engine_params:
            self.set_parameters(engine_params)
        else:
            logger.warning("No configuration provided, using default engine parameters.")
            # Ensure curves are generated even with defaults
            self._initialize_rpm_range()
            self.generate_performance_curves()

    def load_config(self, config_path: str):
        """Load engine configuration from YAML file."""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Engine configuration file not found: {config_path}")

        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            self.set_parameters(self.config)
            logger.info(f"Engine configuration loaded from {config_path}")

            # Load associated thermal config if specified
            thermal_config_path = self.config.get('thermal_config_path')
            if thermal_config_path and os.path.exists(thermal_config_path):
                self.thermal_config.load_from_file(thermal_config_path)
                self.heat_model = EngineHeatModel(self.thermal_config, self) # Re-init with loaded config
                logger.info(f"Associated thermal configuration loaded from {thermal_config_path}")

        except Exception as e:
            logger.error(f"Error loading engine configuration from {config_path}: {e}")
            # Fallback to defaults if loading fails partially
            self._initialize_rpm_range()
            self.generate_performance_curves()


    def set_parameters(self, params: Dict):
        """Set engine parameters from dictionary."""
        self.make = params.get('make', self.make)
        self.model = params.get('model', self.model)
        self.displacement_cc = float(params.get('displacement_cc', self.displacement_cc))
        self.cylinders = int(params.get('cylinders', self.cylinders))
        self.configuration = params.get('configuration', self.configuration)
        self.compression_ratio = float(params.get('compression_ratio', self.compression_ratio))
        self.bore_mm = float(params.get('bore_mm', self.bore_mm))
        self.stroke_mm = float(params.get('stroke_mm', self.stroke_mm))
        self.valves_per_cylinder = int(params.get('valves_per_cylinder', self.valves_per_cylinder))
        self.valve_train_type = params.get('valve_train_type', self.valve_train_type)
        self.max_power_hp = float(params.get('max_power_hp', self.max_power_hp))
        self.max_power_rpm = float(params.get('max_power_rpm', self.max_power_rpm))
        self.max_torque_nm = float(params.get('max_torque_nm', self.max_torque_nm))
        self.max_torque_rpm = float(params.get('max_torque_rpm', self.max_torque_rpm))
        self.redline_rpm = float(params.get('redline_rpm', self.redline_rpm))
        self.idle_rpm = float(params.get('idle_rpm', self.idle_rpm))
        self.weight_kg = float(params.get('dry_weight_kg', self.weight_kg)) # Use dry_weight_kg from config
        self.fuel_type = params.get('fuel_type', self.fuel_type)

        # Apply FS modifications (e.g., weight reduction)
        fs_mods = params.get('fs_modifications', {})
        weight_reduction = float(fs_mods.get('weight_reduction_kg', 0.0))
        if weight_reduction > 0:
             self.weight_kg -= weight_reduction
             logger.info(f"Applied weight reduction of {weight_reduction} kg. New weight: {self.weight_kg:.1f} kg")

        # Update RPM range based on new redline/idle
        self._initialize_rpm_range()

        # Check for dyno data path
        dyno_path = params.get('dyno_data_file')
        if dyno_path and os.path.exists(dyno_path):
             self._load_curves_from_dyno(dyno_path)
        else:
             if dyno_path: logger.warning(f"Dyno data file not found: {dyno_path}. Generating curves.")
             self.generate_performance_curves() # Regenerate curves with new parameters

    def _initialize_rpm_range(self, num_points: int = 150):
        """Initialize the RPM range array."""
        self.rpm_range = np.linspace(self.idle_rpm, self.redline_rpm, num_points)

    def _load_curves_from_dyno(self, file_path: str):
        """Load torque and power curves from a CSV dyno file."""
        try:
            data = pd.read_csv(file_path)
            # Assuming columns 'RPM', 'Torque_Nm', 'Power_kW'
            rpm_col = next((col for col in data.columns if 'rpm' in col.lower()), None)
            torque_col = next((col for col in data.columns if 'torque' in col.lower() and 'nm' in col.lower()), None)
            power_col = next((col for col in data.columns if 'power' in col.lower() and 'kw' in col.lower()),
                             next((col for col in data.columns if 'power' in col.lower() and 'hp' in col.lower()), None)) # Allow HP too

            if not rpm_col or not torque_col:
                raise ValueError("Dyno file must contain RPM and Torque_Nm columns.")

            # Sort by RPM and remove duplicates
            data = data.sort_values(by=rpm_col).drop_duplicates(subset=rpm_col)

            self.rpm_range = data[rpm_col].values
            self.torque_curve = data[torque_col].values

            if power_col:
                self.power_curve_kw = data[power_col].values
                # Convert HP to kW if necessary
                if 'hp' in power_col.lower():
                    self.power_curve_kw *= HP_TO_KW
            else:
                 # Calculate power from torque if not provided
                 self._calculate_power_from_torque()

            # Update engine specs based on loaded data peaks
            idx_tq = np.argmax(self.torque_curve)
            self.max_torque_nm = self.torque_curve[idx_tq]
            self.max_torque_rpm = self.rpm_range[idx_tq]

            idx_pw = np.argmax(self.power_curve_kw)
            self.max_power_hp = self.power_curve_kw[idx_pw] * KW_TO_HP
            self.max_power_rpm = self.rpm_range[idx_pw]

            self.idle_rpm = self.rpm_range[0]
            self.redline_rpm = self.rpm_range[-1]

            self._create_interpolation_functions()
            logger.info(f"Loaded performance curves from dyno file: {file_path}")

        except Exception as e:
            logger.error(f"Error loading curves from dyno file {file_path}: {e}. Generating curves instead.")
            self.generate_performance_curves()

    def _calculate_power_from_torque(self):
        """Calculate power (kW) curve from torque (Nm) curve."""
        if self.torque_curve is None or self.rpm_range is None: return
        angular_velocity_rad_s = self.rpm_range * (2 * np.pi / 60)
        power_watts = self.torque_curve * angular_velocity_rad_s
        self.power_curve_kw = power_watts / 1000.0

    def _create_interpolation_functions(self):
        """Create interpolation functions for torque and power."""
        if self.rpm_range is None or self.torque_curve is None or self.power_curve_kw is None or len(self.rpm_range) < 2:
            logger.warning("Cannot create interpolation functions: Insufficient data points.")
            self.torque_function = None
            self.power_function_kw = None
            return

        # Use cubic interpolation for smoother results, handle potential issues
        try:
            # Ensure RPM points are strictly increasing for CubicSpline
            unique_rpm, unique_indices = np.unique(self.rpm_range, return_index=True)
            unique_torque = self.torque_curve[unique_indices]
            unique_power = self.power_curve_kw[unique_indices]

            if len(unique_rpm) < 4: # CubicSpline needs at least 4 points
                 kind = 'linear'
                 logger.warning("Less than 4 unique points, using linear interpolation.")
            else:
                 kind = 'cubic'

            self.torque_function = interp1d(
                unique_rpm, unique_torque, kind=kind,
                bounds_error=False, fill_value=(unique_torque[0], unique_torque[-1])
            )
            self.power_function_kw = interp1d(
                unique_rpm, unique_power, kind=kind,
                bounds_error=False, fill_value=(unique_power[0], unique_power[-1])
            )
        except ValueError as e:
             logger.error(f"Interpolation failed: {e}. Falling back to linear.")
             self.torque_function = interp1d(
                self.rpm_range, self.torque_curve, kind='linear',
                bounds_error=False, fill_value=(self.torque_curve[0], self.torque_curve[-1])
             )
             self.power_function_kw = interp1d(
                self.rpm_range, self.power_curve_kw, kind='linear',
                bounds_error=False, fill_value=(self.power_curve_kw[0], self.power_curve_kw[-1])
             )

    def generate_performance_curves(self):
        """
        Generate torque and power curves based on engine parameters using a simplified model.
        This is used if no dyno data is provided.
        """
        if self.rpm_range is None: self._initialize_rpm_range()

        # Model based on peak values and typical inline-4 shape
        rpm = self.rpm_range
        peak_tq_rpm = self.max_torque_rpm
        peak_pw_rpm = self.max_power_rpm
        max_tq = self.max_torque_nm
        max_pw_kw = self.max_power_hp * HP_TO_KW

        # --- Torque Curve Model ---
        # Use a skewed Gaussian-like function combined with a base level
        tq_shape = np.exp(-0.5 * ((rpm - peak_tq_rpm) / (peak_tq_rpm * 0.3))**2) # Gaussian part
        tq_skew = 1 + 0.1 * np.tanh((rpm - peak_tq_rpm) / (self.redline_rpm * 0.2)) # Skewness
        idle_torque = max_tq * 0.2 # Torque at idle
        base_torque = idle_torque + (max_tq - idle_torque) * tq_shape * tq_skew

        # --- Power Curve Model (for scaling/validation) ---
        pw_shape = np.exp(-0.5 * ((rpm - peak_pw_rpm) / (peak_pw_rpm * 0.2))**2)
        pw_skew = 1 - 0.1 * np.tanh((rpm - peak_pw_rpm) / (self.redline_rpm * 0.15))
        idle_power = (idle_torque * self.idle_rpm * 2 * np.pi / 60 / 1000) # kW at idle
        base_power_kw = idle_power + (max_pw_kw - idle_power) * pw_shape * pw_skew

        # --- Reconciliation and Scaling ---
        # Calculate power from the generated torque curve
        power_from_torque_kw = base_torque * rpm * 2 * np.pi / 60 / 1000

        # Scale the torque curve so that its derived power matches the max_power_hp at max_power_rpm
        power_at_peak_rpm = np.interp(peak_pw_rpm, rpm, power_from_torque_kw)
        if power_at_peak_rpm > 1e-3:
             scale_factor_p = max_pw_kw / power_at_peak_rpm
        else:
             scale_factor_p = 1.0
             logger.warning("Could not scale torque curve based on power peak, using base torque.")

        self.torque_curve = base_torque * scale_factor_p

        # Recalculate power and peaks from the final scaled torque curve
        self._calculate_power_from_torque()
        # Update peak values based on generated curve
        idx_tq = np.argmax(self.torque_curve)
        self.max_torque_nm = self.torque_curve[idx_tq]
        self.max_torque_rpm = self.rpm_range[idx_tq]
        idx_pw = np.argmax(self.power_curve_kw)
        self.max_power_hp = self.power_curve_kw[idx_pw] * KW_TO_HP
        self.max_power_rpm = self.rpm_range[idx_pw]

        self._create_interpolation_functions()
        logger.info("Generated performance curves based on engine parameters.")


    def get_torque(self, rpm: float, throttle: float = 1.0, engine_temp: Optional[float] = None) -> float:
        """
        Calculate engine torque at specified RPM, throttle, and temperature.

        Args:
            rpm: Engine speed in RPM.
            throttle: Throttle position (0.0 to 1.0).
            engine_temp: Engine temperature in °C (uses self.engine_temperature if None).

        Returns:
            Torque in Nm.
        """
        if self.torque_function is None:
            logger.warning("Torque function not initialized, returning 0.")
            return 0.0

        # Use current engine temperature if not provided
        temp = engine_temp if engine_temp is not None else self.engine_temperature

        # Ensure RPM is within operating range
        rpm = np.clip(rpm, self.idle_rpm, self.redline_rpm)
        throttle = np.clip(throttle, 0.0, 1.0)

        # Get base torque from the interpolated curve
        base_torque = float(self.torque_function(rpm))

        # Apply throttle position (non-linear relationship - throttle^gamma)
        # Gamma < 1 means torque increases faster at lower throttle openings
        throttle_gamma = 0.8
        throttle_factor = throttle ** throttle_gamma

        # Apply thermal performance factor
        temp_factor = self._get_thermal_performance_factor(temp)

        # Calculate final torque
        actual_torque = base_torque * throttle_factor * temp_factor

        return actual_torque

    def _get_thermal_performance_factor(self, temp_c: float) -> float:
        """Calculate performance factor based on engine temperature."""
        # Get optimal range from thermal config if available
        if self.thermal_config and hasattr(self.thermal_config, 'optimal_temp_engine'):
            optimal_low, optimal_high = self.thermal_config.optimal_temp_engine
            warning_temp = self.thermal_config.warning_temp_engine
            critical_temp = self.thermal_config.critical_temp_engine
        else: # Fallback if thermal_config is missing or lacks attributes
            optimal_low, optimal_high = (85.0, 100.0)
            warning_temp = 105.0
            critical_temp = 115.0
        # ---------------------------------------------------------------------

        if temp_c < optimal_low: # Too cold
            # Linear penalty from 0.7 at 20C to 1.0 at optimal_low
            cold_limit = 20.0
            factor = 0.7 + 0.3 * (temp_c - cold_limit) / max(1.0, optimal_low - cold_limit)
            return max(0.7, min(1.0, factor)) # Ensure factor doesn't exceed 1.0
        elif temp_c <= optimal_high: # Optimal range
            return 1.0
        elif temp_c <= warning_temp: # Slightly too hot
            # Linear penalty from 1.0 at optimal_high to 0.9 at warning_temp
            factor = 1.0 - 0.1 * (temp_c - optimal_high) / max(1.0, warning_temp - optimal_high)
            return max(0.9, factor)
        elif temp_c <= critical_temp: # Warning range
             # Linear penalty from 0.9 at warning_temp to 0.7 at critical_temp
            factor = 0.9 - 0.2 * (temp_c - warning_temp) / max(1.0, critical_temp - warning_temp)
            return max(0.7, factor)
        else: # Critical overheating
            # Rapidly drop performance
            factor = 0.7 - 0.4 * min(1.0, (temp_c - critical_temp) / 10.0) # Drop to ~30% within 10C over critical
            return max(0.3, factor)


    def get_power(self, rpm: float, throttle: float = 1.0, engine_temp: Optional[float] = None) -> float:
        """
        Calculate engine power at specified RPM, throttle, and temperature.

        Args:
            rpm: Engine speed in RPM.
            throttle: Throttle position (0.0 to 1.0).
            engine_temp: Engine temperature in °C (uses self.engine_temperature if None).

        Returns:
            Power in kW.
        """
        # Get torque considering temperature
        torque = self.get_torque(rpm, throttle, engine_temp)

        # Power (W) = Torque (Nm) * Angular velocity (rad/s)
        power_watts = torque * rpm * (2 * np.pi / 60)
        power_kw = power_watts / 1000.0

        return power_kw

    def get_fuel_consumption(self, rpm: float, throttle: float = 1.0) -> float:
        """
        Estimate fuel consumption rate based on simplified BSFC model.

        Args:
            rpm: Engine speed in RPM.
            throttle: Throttle position (0.0 to 1.0).

        Returns:
            Fuel consumption in g/s.
        """
        # Get power output (kW) at current conditions (using current engine temp)
        power_kw = self.get_power(rpm, throttle)

        # Base BSFC (g/kWh) - depends on fuel type and engine tuning
        # E85 requires more fuel mass for same energy -> higher BSFC number
        # Typical efficient points might be ~280-300 g/kWh for gasoline,
        # E85 might be ~30-40% higher => ~360-420 g/kWh.
        base_bsfc = 400.0 if 'E85' in self.fuel_type.upper() else 300.0

        # BSFC variation with load (throttle) and RPM
        # Efficiency is generally best near peak torque RPM and high load (~70-90% throttle)
        norm_rpm = np.clip((rpm - self.idle_rpm) / (self.redline_rpm - self.idle_rpm), 0, 1)
        peak_torque_norm_rpm = (self.max_torque_rpm - self.idle_rpm) / (self.redline_rpm - self.idle_rpm)

        # RPM factor: higher BSFC away from peak torque
        rpm_factor = 1.0 + 0.3 * abs(norm_rpm - peak_torque_norm_rpm)**1.5

        # Load factor: higher BSFC at very low and very high loads
        load_factor = 1.0 + 0.4 * (1.0 - throttle)**2 + 0.1 * throttle**3

        # Calculate actual BSFC
        actual_bsfc = base_bsfc * rpm_factor * load_factor

        # Convert g/kWh to g/s
        # g/s = (g/kWh * kW) / 3600
        fuel_consumption_g_s = (actual_bsfc * power_kw) / 3600.0

        # Add idle fuel consumption (small baseline rate)
        idle_consumption = 0.05 # g/s (example value)
        fuel_consumption_g_s = max(idle_consumption, fuel_consumption_g_s)

        # Store current rate
        self.fuel_consumption_rate = fuel_consumption_g_s

        return fuel_consumption_g_s

    def update_thermal_state(self, ambient_temp: float, cooling_effectiveness: float, dt: float):
        """
        Update engine thermal state using the internal heat model.

        Args:
            ambient_temp: Ambient temperature in °C.
            cooling_effectiveness: Factor representing cooling system performance (0-1).
            dt: Time step in seconds.

        Returns:
            Dictionary with updated temperatures ('engine_temp', 'coolant_temp', 'oil_temp').
        """
        if not self.heat_model:
            logger.warning("No heat model available for thermal update.")
            return {'engine_temp': self.engine_temperature, 'coolant_temp': self.coolant_temperature, 'oil_temp': self.oil_temperature}

        # --- 1. Calculate Heat Generation ---
        # Get current power and fuel flow
        power_kw = self.get_power(self.current_rpm, self.throttle_position, self.engine_temperature)
        fuel_flow_g_s = self.get_fuel_consumption(self.current_rpm, self.throttle_position)

        # Calculate fuel power (Need fuel properties)
        # Assume E85 for now if fuel type not properly set
        fuel_energy_density_MJ_kg = 29.2 # E85 approx
        if self.fuel_type == 'GASOLINE_98RON': fuel_energy_density_MJ_kg = 44.4
        fuel_power_kw = (fuel_flow_g_s / 1000.0) * fuel_energy_density_MJ_kg * 1000.0

        # Heat generated (kW)
        heat_generated_kw = max(0.0, fuel_power_kw - power_kw)
        heat_generated_watts = heat_generated_kw * 1000.0

        # --- 2. Calculate Heat Dissipation ---
        # Simplified: Cooling power proportional to temp diff and effectiveness
        # Cooling effectiveness already incorporates fan, speed etc. factors
        temp_diff_engine_ambient = self.engine_temperature - ambient_temp
        cooling_power_watts = cooling_effectiveness * temp_diff_engine_ambient * 50 # Base heat transfer rate factor W/K

        # --- 3. Update Temperatures ---
        net_heat_watts = heat_generated_watts - cooling_power_watts

        # Get thermal capacities
        capacities = self.thermal_config.get_thermal_capacities()

        # Update engine block temperature
        if capacities['engine_block'] > 0:
             # Distribute net heat (e.g., 70% affects block directly)
             block_heat = net_heat_watts * 0.7
             delta_t_engine = (block_heat * dt) / capacities['engine_block']
             self.engine_temperature += delta_t_engine
        else:
             logger.warning("Engine block thermal capacity is zero.")

        # Update coolant temperature (receives heat from block, loses heat via radiator)
        if capacities['coolant'] > 0:
             # Heat transfer from block to coolant
             htc_coolant = self.thermal_config.htc_coolant_to_block # W/(m^2*K) - needs area
             area_coolant = 0.5 # Estimated contact area m^2
             q_block_to_coolant = htc_coolant * area_coolant * (self.engine_temperature - self.coolant_temperature)

             # Net heat change in coolant = heat from engine - heat rejected by radiator
             net_coolant_heat = q_block_to_coolant - cooling_power_watts # Cooling power acts on coolant
             delta_t_coolant = (net_coolant_heat * dt) / capacities['coolant']
             self.coolant_temperature += delta_t_coolant
        else:
             logger.warning("Coolant thermal capacity is zero.")


        # Update oil temperature (receives heat from block, simplified cooling)
        if capacities['engine_oil'] > 0:
             # Heat transfer from block to oil
             htc_oil = self.thermal_config.htc_oil_to_block # W/(m^2*K) - needs area
             area_oil = 0.3 # Estimated contact area m^2
             q_block_to_oil = htc_oil * area_oil * (self.engine_temperature - self.oil_temperature)

             # Simplified oil cooling (e.g., convection to ambient)
             htc_oil_ambient = 15 # W/(m^2*K)
             area_oil_ambient = 0.2 # m^2
             q_oil_to_ambient = htc_oil_ambient * area_oil_ambient * (self.oil_temperature - ambient_temp)

             net_oil_heat = q_block_to_oil - q_oil_to_ambient
             delta_t_oil = (net_oil_heat * dt) / capacities['engine_oil']
             self.oil_temperature += delta_t_oil
        else:
             logger.warning("Oil thermal capacity is zero.")

        # Clamp temperatures to realistic bounds
        self.engine_temperature = np.clip(self.engine_temperature, ambient_temp - 10, 150)
        self.coolant_temperature = np.clip(self.coolant_temperature, ambient_temp - 10, 130)
        self.oil_temperature = np.clip(self.oil_temperature, ambient_temp - 10, 160)

        # Update thermal factor based on new engine temperature
        self.thermal_factor = self._get_thermal_performance_factor(self.engine_temperature)

        return {
            'engine_temp': self.engine_temperature,
            'coolant_temp': self.coolant_temperature,
            'oil_temp': self.oil_temperature
        }


    def plot_performance_curves(self, save_path: Optional[str] = None):
        """
        Plot generated torque and power curves using the centralized plotting function.

        Args:
            save_path: If provided, save the plot to this file path.
        """
        if self.rpm_range is None or self.torque_curve is None or self.power_curve_kw is None:
             logger.error("Cannot plot performance curves: Data not generated.")
             return

        from ..utils.plotting import plot_engine_performance # Import plotting function

        engine_data = {
            'rpm': self.rpm_range,
            'torque': self.torque_curve,
            'power': self.power_curve_kw, # Pass power in kW
            # Add peaks for plotting annotations
            'max_torque_rpm': self.max_torque_rpm,
            'max_torque': self.max_torque_nm,
            'max_power_rpm': self.max_power_rpm,
            'max_power': self.max_power_hp * HP_TO_KW # Provide max power in kW
        }

        plot_title = f'{self.make} {self.model} Engine Performance'
        fig = plot_engine_performance(engine_data, title=plot_title, save_path=save_path)
        # Optionally close the figure after saving/showing if needed
        # plt.close(fig)

    def get_engine_specs(self) -> Dict:
        """Get a dictionary of current engine specifications."""
        return {
            'make': self.make,
            'model': self.model,
            'displacement_cc': self.displacement_cc,
            'cylinders': self.cylinders,
            'configuration': self.configuration,
            'compression_ratio': self.compression_ratio,
            'bore_mm': self.bore_mm,
            'stroke_mm': self.stroke_mm,
            'valves_per_cylinder': self.valves_per_cylinder,
            'valve_train_type': self.valve_train_type,
            'max_power_hp': self.max_power_hp,
            'max_power_rpm': self.max_power_rpm,
            'max_torque_nm': self.max_torque_nm,
            'max_torque_rpm': self.max_torque_rpm,
            'redline_rpm': self.redline_rpm,
            'idle_rpm': self.idle_rpm,
            'weight_kg': self.weight_kg,
            'fuel_type': self.fuel_type
        }

# Example usage
if __name__ == "__main__":
    # Path to config file (relative to the project root)
    # Assuming the script is run from the project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..')) # Go up two levels
    config_path = os.path.join(project_root, "configs", "engine", "cbr600f4i.yaml")

    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        print("Please ensure the path is correct relative to where you run the script.")
    else:
        print(f"Loading engine configuration from: {config_path}")
        # Create engine
        engine = MotorcycleEngine(config_path=config_path)

        # Print engine specs
        print("\nEngine Specifications:")
        specs = engine.get_engine_specs()
        for key, value in specs.items():
            print(f"  {key}: {value}")

        # Plot performance curves
        print("\nPlotting performance curves...")
        plot_save_path = os.path.join(project_root, "plots", "cbr600f4i_performance.png")
        os.makedirs(os.path.dirname(plot_save_path), exist_ok=True)
        engine.plot_performance_curves(save_path=plot_save_path)
        print(f"Performance plot saved to: {plot_save_path}")

        # Calculate torque and power at various RPMs and throttle positions
        print("\nPerformance Examples:")
        test_rpms = [3000, 6000, 9000, 12000, 13500]
        for rpm in test_rpms:
            torque_full = engine.get_torque(rpm, throttle=1.0)
            power_kw_full = engine.get_power(rpm, throttle=1.0)
            torque_half = engine.get_torque(rpm, throttle=0.5)
            power_kw_half = engine.get_power(rpm, throttle=0.5)
            fuel_full = engine.get_fuel_consumption(rpm, throttle=1.0)
            fuel_half = engine.get_fuel_consumption(rpm, throttle=0.5)

            print(f"At {rpm:.0f} RPM:")
            print(f"  Full Throttle: Torque = {torque_full:.1f} Nm, Power = {power_kw_full:.1f} kW ({power_kw_full*KW_TO_HP:.1f} HP), Fuel = {fuel_full:.2f} g/s")
            print(f"  Half Throttle: Torque = {torque_half:.1f} Nm, Power = {power_kw_half:.1f} kW ({power_kw_half*KW_TO_HP:.1f} HP), Fuel = {fuel_half:.2f} g/s")

        # Example of thermal update
        print("\nSimulating thermal update:")
        engine.current_rpm = 10000
        engine.throttle_position = 0.8
        initial_temps = engine.update_thermal_state(ambient_temp=30.0, cooling_effectiveness=0.7, dt=1.0)
        print(f"  Initial Temps: Engine={initial_temps['engine_temp']:.1f}C, Coolant={initial_temps['coolant_temp']:.1f}C, Oil={initial_temps['oil_temp']:.1f}C")
        temps_after_10s = engine.update_thermal_state(ambient_temp=30.0, cooling_effectiveness=0.7, dt=10.0)
        print(f"  Temps after 10s: Engine={temps_after_10s['engine_temp']:.1f}C, Coolant={temps_after_10s['coolant_temp']:.1f}C, Oil={temps_after_10s['oil_temp']:.1f}C")
        print(f"  Current Thermal Factor: {engine.thermal_factor:.2f}")