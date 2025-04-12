"""
Electric compressor module for supplemental cooling airflow in Formula Student simulation.

Models an electric compressor/fan used to boost airflow through radiators,
especially at low vehicle speeds. Includes performance characteristics, power use,
and control strategies.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Callable
import logging
from enum import Enum, auto
import yaml
import os
from scipy.interpolate import interp1d

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ElectricCompressor")

class CompressorType(Enum):
    """Types of electric compressors/blowers."""
    CENTRIFUGAL = auto()
    AXIAL = auto()
    MIXED_FLOW = auto()
    CUSTOM = auto()

class CompressorControl(Enum):
    """Control methods for the electric compressor."""
    ON_OFF = auto()         # Simple threshold-based control
    VARIABLE_SPEED = auto() # Speed proportional to a signal (e.g., temp error)
    PID_CONTROL = auto()    # PID loop controlling based on temperature error
    ADAPTIVE = auto()       # Control based on multiple factors (temp, speed, load)
    CUSTOM = auto()

class ElectricCompressor:
    """Models an electric compressor/blower for cooling assist."""
    def __init__(self,
                 compressor_type: CompressorType = CompressorType.CENTRIFUGAL,
                 max_airflow_m3s: float = 0.25,    # Max flow at zero backpressure
                 max_pressure_pa: float = 200.0,   # Max static pressure at zero flow
                 max_power_W: float = 120.0,
                 voltage_V: float = 13.5,
                 efficiency: float = 0.65,         # Aerodynamic efficiency
                 weight_kg: float = 0.7,
                 response_time_s: float = 0.3,    # Time constant for speed changes
                 config_path: Optional[str] = None,
                 custom_params: Optional[Dict] = None):
        """
        Initialize the electric compressor model.

        Args:
            compressor_type: Type of compressor.
            max_airflow_m3s: Maximum airflow (m³/s).
            max_pressure_pa: Maximum static pressure rise (Pa).
            max_power_W: Maximum electrical power consumption (W).
            voltage_V: Nominal operating voltage (V).
            efficiency: Aerodynamic efficiency (0-1).
            weight_kg: Compressor weight (kg).
            response_time_s: Time constant for speed changes (s).
            config_path: Optional path to YAML config file.
            custom_params: Optional dictionary for CUSTOM type or overrides.
        """
        self.compressor_type = compressor_type
        self.max_airflow_m3s = max_airflow_m3s
        self.max_pressure_pa = max_pressure_pa
        self.max_power_W = max_power_W
        self.voltage_V = voltage_V
        self.efficiency = efficiency
        self.weight_kg = weight_kg
        self.response_time_s = max(1e-3, response_time_s) # Avoid zero response time

        # Load from config if path provided
        if config_path and os.path.exists(config_path):
            self._load_config(config_path)

        # Apply custom params or set defaults based on type
        params = custom_params or {}
        self._apply_type_defaults_and_custom(params)

        # State variables
        self.current_speed_ratio: float = 0.0 # Speed relative to max (0-1)
        self.target_speed_ratio: float = 0.0
        self.current_airflow_m3s: float = 0.0
        self.current_pressure_pa: float = 0.0 # Pressure generated at current speed/flow
        self.current_power_W: float = 0.0
        self.is_active: bool = False

        # Simplified performance curves (Pressure vs Flow at different speeds)
        # P = Pmax(speed) * (1 - (Q / Qmax(speed))^2)
        # Power = Pmax_elec * speed^3 (approx)
        self._create_performance_model()

        logger.info(f"Electric Compressor initialized: Type={self.compressor_type.name}, MaxAirflow={self.max_airflow_m3s:.2f}m³/s, MaxPressure={self.max_pressure_pa:.0f}Pa")

    def _load_config(self, config_path: str):
        """Load parameters from YAML config file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            comp_config = config.get('compressor', {})
            self.compressor_type = CompressorType[comp_config.get('type', self.compressor_type.name).upper()]
            self.max_airflow_m3s = float(comp_config.get('max_airflow', self.max_airflow_m3s))
            self.max_pressure_pa = float(comp_config.get('max_pressure', self.max_pressure_pa))
            self.max_power_W = float(comp_config.get('max_power', self.max_power_W))
            self.voltage_V = float(comp_config.get('voltage', self.voltage_V))
            self.efficiency = float(comp_config.get('efficiency', self.efficiency))
            self.weight_kg = float(comp_config.get('weight', self.weight_kg))
            self.response_time_s = float(comp_config.get('response_time', self.response_time_s))
            self._apply_type_defaults_and_custom(comp_config)
            logger.info(f"Compressor config loaded from {config_path}")
        except Exception as e:
            logger.error(f"Error loading compressor config from {config_path}: {e}. Using existing values.")


    def _apply_type_defaults_and_custom(self, params: Dict):
        """Set default properties based on type, overridden by params."""
        # These defaults are illustrative
        defaults = {}
        if self.compressor_type == CompressorType.CENTRIFUGAL:
             defaults = {'efficiency': 0.68, 'noise_db': 70}
        elif self.compressor_type == CompressorType.AXIAL:
             defaults = {'efficiency': 0.75, 'noise_db': 75}
        elif self.compressor_type == CompressorType.MIXED_FLOW:
             defaults = {'efficiency': 0.72, 'noise_db': 72}
        elif self.compressor_type == CompressorType.CUSTOM:
             defaults = {'efficiency': 0.70, 'noise_db': 70} # Generic custom

        self.efficiency = float(params.get('efficiency', defaults.get('efficiency', 0.65)))
        self.noise_db = float(params.get('noise_db', defaults.get('noise_db', 70)))


    def _create_performance_model(self):
        """Create simplified model for pressure-flow relationship."""
        # P = Pmax(speed) * (1 - (Q / Qmax(speed))^2)
        # Qmax(speed) = Qmax_rated * speed_ratio^3
        # Pmax(speed) = Pmax_rated * speed_ratio^2
        pass # No explicit curve stored, calculated on the fly in get_airflow

    def update_speed(self, target_speed_ratio: float, dt: float):
        """Update the compressor's rotational speed based on target and response time."""
        target_speed_ratio = np.clip(target_speed_ratio, 0.0, 1.0)
        self.target_speed_ratio = target_speed_ratio

        # Simple first-order lag for response time
        delta_speed = (target_speed_ratio - self.current_speed_ratio) * (1 - np.exp(-dt / self.response_time_s))
        self.current_speed_ratio += delta_speed
        self.current_speed_ratio = np.clip(self.current_speed_ratio, 0.0, 1.0)

        self.is_active = self.current_speed_ratio > 0.05 # Consider active if speed > 5%

        # Update power consumption based on new speed
        self._update_power_consumption()

    def _update_power_consumption(self):
        """Update electrical power consumption based on current speed."""
        # Power ~ Speed^3 (Fan Laws)
        self.current_power_W = self.max_power_W * (self.current_speed_ratio**3)
        self.current_power_W = max(0.0, self.current_power_W) # Ensure non-negative

    def get_airflow_m3s(self, system_pressure_drop_pa: float = 0.0) -> float:
        """
        Calculate the actual airflow delivered against system pressure drop.

        Args:
            system_pressure_drop_pa: Backpressure from the system (Pa).

        Returns:
            Actual airflow (m³/s).
        """
        if not self.is_active or self.current_speed_ratio <= 0:
            self.current_airflow_m3s = 0.0
            self.current_pressure_pa = 0.0 # No pressure if not running
            return 0.0

        # Calculate max flow and pressure at current speed using fan laws
        effective_q_max = self.max_airflow_m3s * self.current_speed_ratio**3
        effective_p_max = self.max_pressure_pa * self.current_speed_ratio**2

        # Use the simplified fan curve: Q = Qmax * sqrt(1 - P/Pmax)
        if effective_p_max <= 0 or system_pressure_drop_pa >= effective_p_max:
            flow_rate = 0.0
            pressure_generated = effective_p_max # Max pressure it can generate against blockage
        else:
            pressure_ratio = system_pressure_drop_pa / effective_p_max
            flow_rate = effective_q_max * np.sqrt(1.0 - pressure_ratio)
            pressure_generated = system_pressure_drop_pa # Pressure matches system drop if flow > 0

        self.current_airflow_m3s = max(0.0, flow_rate)
        self.current_pressure_pa = pressure_generated # Store pressure generated
        return self.current_airflow_m3s

    def get_power_consumption_W(self) -> float:
        """Return the current electrical power consumption."""
        # Power is updated in update_speed
        return self.current_power_W

    def get_compressor_state(self) -> Dict:
        """Get current operational state of the compressor."""
        return {
            'is_active': self.is_active,
            'speed_ratio': self.current_speed_ratio, # Renamed from control_signal
            'target_speed_ratio': self.target_speed_ratio,
            'airflow_m3s': self.current_airflow_m3s,
            'pressure_pa': self.current_pressure_pa, # Pressure generated
            'power_W': self.current_power_W
        }

    def get_compressor_specs(self) -> Dict:
        """Get compressor specifications."""
        return {
            'type': self.compressor_type.name,
            'max_airflow_m3s': self.max_airflow_m3s,
            'max_pressure_pa': self.max_pressure_pa,
            'max_power_W': self.max_power_W,
            'voltage_V': self.voltage_V,
            'efficiency': self.efficiency,
            'weight_kg': self.weight_kg,
            'response_time_s': self.response_time_s,
            'noise_db': self.noise_db
        }

    def plot_performance_curves(self, save_path: Optional[str] = None):
        """Plot simplified compressor performance curves (Flow, Pressure, Power vs Speed Ratio)."""
        from ..utils.plotting import save_plot # Local import

        fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        speed_ratios = np.linspace(0, 1, 101)

        # Calculate curves based on fan laws
        airflows = self.max_airflow_m3s * speed_ratios**3
        pressures = self.max_pressure_pa * speed_ratios**2
        powers = self.max_power_W * speed_ratios**3

        axes[0].plot(speed_ratios, airflows, color=COLOR_SCHEMES['default'][0])
        _apply_common_ax_settings(axes[0], ylabel='Max Airflow (m³/s)', title='Compressor Performance vs Speed Ratio')

        axes[1].plot(speed_ratios, pressures, color=COLOR_SCHEMES['default'][1])
        _apply_common_ax_settings(axes[1], ylabel='Max Static Pressure (Pa)')

        axes[2].plot(speed_ratios, powers, color=COLOR_SCHEMES['default'][2])
        _apply_common_ax_settings(axes[2], xlabel='Speed Ratio (0-1)', ylabel='Power Consumption (W)')

        plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust for suptitle if added
        fig.suptitle(f'{self.compressor_type.name} Compressor Performance', fontsize=14)

        if save_path: save_plot(fig, save_path)
        plt.show()
        plt.close(fig)

    def plot_compressor_map(self, num_speeds=5, save_path: Optional[str] = None):
        """Plot a simplified Pressure vs Flow map for different speed lines."""
        from ..utils.plotting import save_plot # Local import

        fig, ax = plt.subplots(figsize=(10, 8))
        speed_ratios = np.linspace(0.2, 1.0, num_speeds) # From 20% to 100% speed

        for i, speed_ratio in enumerate(speed_ratios):
            effective_q_max = self.max_airflow_m3s * speed_ratio**3
            effective_p_max = self.max_pressure_pa * speed_ratio**2

            flow_points = np.linspace(0, effective_q_max, 50)
            pressure_points = effective_p_max * (1.0 - (flow_points / effective_q_max)**2)

            color = plt.cm.viridis(speed_ratio) # Color by speed ratio
            ax.plot(flow_points, pressure_points, color=color, label=f'{speed_ratio*100:.0f}% Speed')

        _apply_common_ax_settings(ax, xlabel='Airflow (m³/s)', ylabel='Static Pressure (Pa)', title='Simplified Compressor Map')
        ax.legend(title='Speed Ratio')
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)

        plt.tight_layout()
        if save_path: save_plot(fig, save_path)
        plt.show()
        plt.close(fig)


class CompressorControlModule:
    """Manages the control logic for an ElectricCompressor."""
    def __init__(self, compressor: ElectricCompressor,
                 control_strategy: CompressorControl = CompressorControl.ADAPTIVE,
                 config_path: Optional[str] = None):
        """
        Initialize compressor control module.

        Args:
            compressor: ElectricCompressor instance to control.
            control_strategy: Control strategy enum member.
            config_path: Optional path to YAML config file for control params.
        """
        self.compressor = compressor
        self.control_strategy = control_strategy

        # Default parameters (can be overridden by config)
        self.target_temp_C: float = 90.0
        self.max_temp_C: float = 98.0 # Temp for 100% activation
        self.min_temp_C: float = 85.0 # Temp below which compressor is off

        # PID specific
        self.pid_kp: float = 0.15
        self.pid_ki: float = 0.02
        self.pid_kd: float = 0.05
        self._pid_integral: float = 0.0
        self._pid_prev_error: float = 0.0

        # Adaptive specific
        self.adaptive_temp_weight: float = 0.7
        self.adaptive_speed_weight: float = 0.2 # Weight for 1 - normalized speed
        self.adaptive_load_weight: float = 0.1
        self.adaptive_activation_threshold: float = 0.2 # Min weighted value to turn on

        # Load from config if path provided
        if config_path and os.path.exists(config_path):
            self._load_config(config_path)

        # Set the compressor's control method to match the module's
        # self.compressor.control_method = self.control_strategy # This might be redundant if compressor handles it

        self.current_control_signal: float = 0.0

        logger.info(f"Compressor Control Module initialized with strategy: {self.control_strategy.name}")

    def _load_config(self, config_path: str):
        """Load control parameters from YAML config file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            ctrl_config = config.get('control', {})
            self.control_strategy = CompressorControl[ctrl_config.get('strategy', self.control_strategy.name).upper()]
            self.target_temp_C = float(ctrl_config.get('target_temp', self.target_temp_C))
            self.max_temp_C = float(ctrl_config.get('max_temp', self.max_temp_C))
            self.min_temp_C = float(ctrl_config.get('min_temp', self.min_temp_C))

            if self.control_strategy == CompressorControl.PID_CONTROL:
                 pid_params = ctrl_config.get('pid', {})
                 self.pid_kp = float(pid_params.get('kp', self.pid_kp))
                 self.pid_ki = float(pid_params.get('ki', self.pid_ki))
                 self.pid_kd = float(pid_params.get('kd', self.pid_kd))
            elif self.control_strategy == CompressorControl.ADAPTIVE:
                 adapt_params = ctrl_config.get('adaptive', {})
                 self.adaptive_temp_weight = float(adapt_params.get('temp_weight', self.adaptive_temp_weight))
                 self.adaptive_speed_weight = float(adapt_params.get('speed_weight', self.adaptive_speed_weight))
                 self.adaptive_load_weight = float(adapt_params.get('load_weight', self.adaptive_load_weight))
                 self.adaptive_activation_threshold = float(adapt_params.get('activation_threshold', self.adaptive_activation_threshold))

            logger.info(f"Compressor control config loaded from {config_path}")
        except Exception as e:
            logger.error(f"Error loading compressor control config from {config_path}: {e}. Using existing values.")

    def update_control(self, coolant_temp_C: float, vehicle_speed_mps: float,
                     engine_load: float, dt: float):
        """Calculate and apply the control signal to the compressor."""
        signal = 0.0 # Default to off

        if self.control_strategy == CompressorControl.ON_OFF:
            if coolant_temp_C >= self.target_temp_C: signal = 1.0
            elif coolant_temp_C <= self.min_temp_C: signal = 0.0
            else: signal = self.compressor.current_speed_ratio # Hysteresis

        elif self.control_strategy == CompressorControl.VARIABLE_SPEED:
            if coolant_temp_C <= self.min_temp_C: signal = 0.0
            elif coolant_temp_C >= self.max_temp_C: signal = 1.0
            else:
                signal = (coolant_temp_C - self.min_temp_C) / max(1e-3, (self.max_temp_C - self.min_temp_C))
            # Speed factor reduction
            speed_factor = max(0.0, 1.0 - vehicle_speed_mps / 25.0) # Reduce effect above 25 m/s
            signal *= speed_factor

        elif self.control_strategy == CompressorControl.PID_CONTROL:
            error = coolant_temp_C - self.target_temp_C
            if coolant_temp_C < self.min_temp_C: # Turn off below min temp
                signal = 0.0
                self._pid_integral = 0.0 # Reset integral
                self._pid_prev_error = 0.0
            else:
                self._pid_integral += error * dt
                self._pid_integral = np.clip(self._pid_integral, -10.0, 10.0) # Anti-windup
                derivative = (error - self._pid_prev_error) / dt if dt > 0 else 0.0
                self._pid_prev_error = error
                output = self.pid_kp * error + self.pid_ki * self._pid_integral + self.pid_kd * derivative
                signal = np.clip(output, 0.0, 1.0)
                # Speed factor reduction
                speed_factor = max(0.0, 1.0 - vehicle_speed_mps / 25.0)
                signal *= speed_factor

        elif self.control_strategy == CompressorControl.ADAPTIVE:
            temp_factor = np.clip((coolant_temp_C - self.min_temp_C) / max(1e-3, (self.max_temp_C - self.min_temp_C)), 0.0, 1.0)
            speed_factor = np.clip(1.0 - vehicle_speed_mps / 25.0, 0.0, 1.0) # Max effect at 0 speed, zero effect at 25 m/s
            load_factor = np.clip(engine_load, 0.0, 1.0)
            weighted_sum = (self.adaptive_temp_weight * temp_factor +
                            self.adaptive_speed_weight * speed_factor +
                            self.adaptive_load_weight * load_factor)
            signal = np.clip(weighted_sum, 0.0, 1.0)
            if signal < self.adaptive_activation_threshold:
                 signal = 0.0 # Turn off if weighted sum is too low


        self.current_control_signal = signal
        self.compressor.update_speed(signal, dt) # Update compressor speed based on signal

    def override_control(self, control_signal: float, dt: float):
        """Manually set the compressor speed ratio."""
        signal = np.clip(control_signal, 0.0, 1.0)
        self.current_control_signal = signal # Store the override signal
        self.compressor.update_speed(signal, dt)

    def reset_controller(self):
        """Reset internal controller states (e.g., PID integral)."""
        self._pid_integral = 0.0
        self._pid_prev_error = 0.0
        self.current_control_signal = 0.0
        # Optionally reset compressor speed too
        # self.compressor.update_speed(0.0, 0.1)

    def get_control_state(self) -> Dict:
        """Get current state of the control module."""
        state = {'strategy': self.control_strategy.name, 'control_signal': self.current_control_signal}
        if self.control_strategy == CompressorControl.PID_CONTROL:
             state.update({'pid_integral': self._pid_integral, 'pid_prev_error': self._pid_prev_error})
        return state


class CoolingAssistSystem:
    """Integrates the electric compressor and control module."""
    def __init__(self,
                 compressor: ElectricCompressor,
                 control_module: CompressorControlModule,
                 duct_pressure_loss_pa_per_m3s: float = 50.0): # Simplified duct loss
        """
        Initialize the integrated cooling assist system.

        Args:
            compressor: ElectricCompressor instance.
            control_module: CompressorControlModule instance.
            duct_pressure_loss_pa_per_m3s: Pressure loss per unit airflow (Pa / (m³/s)).
        """
        self.compressor = compressor
        self.control_module = control_module
        self.duct_pressure_loss_pa_per_m3s = duct_pressure_loss_pa_per_m3s

        # State
        self.supplementary_airflow_m3s: float = 0.0
        self.total_power_W: float = 0.0

        logger.info("Cooling Assist System initialized.")

    def update_system(self, coolant_temp_C: float, vehicle_speed_mps: float,
                    engine_load: float, dt: float,
                    radiator_pressure_drop_pa: float = 0.0):
        """
        Update the state of the cooling assist system.

        Args:
            coolant_temp_C: Current coolant temperature (°C).
            vehicle_speed_mps: Current vehicle speed (m/s).
            engine_load: Current engine load (0-1).
            dt: Time step (s).
            radiator_pressure_drop_pa: Backpressure from the radiator (Pa).
        """
        # Update the controller to determine the target compressor speed
        self.control_module.update_control(coolant_temp_C, vehicle_speed_mps, engine_load, dt)

        # Calculate total system backpressure (radiator + ducting)
        # Duct pressure loss depends on the flow itself - iterative or simplified
        # Simplified: Assume duct loss is proportional to *target* flow based on speed ratio
        target_airflow = self.compressor.max_airflow_m3s * self.compressor.target_speed_ratio**3
        duct_pressure_loss = self.duct_pressure_loss_pa_per_m3s * target_airflow

        total_backpressure_pa = radiator_pressure_drop_pa + duct_pressure_loss

        # Calculate the actual airflow the compressor delivers against this backpressure
        self.supplementary_airflow_m3s = self.compressor.get_airflow_m3s(total_backpressure_pa)

        # Get the power consumption
        self.total_power_W = self.compressor.get_power_consumption_W()

    def get_supplementary_airflow_m3s(self) -> float:
        """Get the current supplementary airflow provided by the system."""
        return self.supplementary_airflow_m3s

    def get_total_power_W(self) -> float:
        """Get the current total power consumption of the system."""
        return self.total_power_W

    def get_system_state(self) -> Dict:
        """Get the current state of the integrated system."""
        return {
            'supplementary_airflow_m3s': self.supplementary_airflow_m3s,
            'total_power_W': self.total_power_W,
            'compressor_state': self.compressor.get_compressor_state(),
            'control_state': self.control_module.get_control_state()
        }

    def get_system_specs(self) -> Dict:
        """Get the specifications of the integrated system."""
        return {
            'compressor_specs': self.compressor.get_compressor_specs(),
            'control_strategy': self.control_module.control_strategy.name,
            'duct_pressure_loss_pa_per_m3s': self.duct_pressure_loss_pa_per_m3s
        }

    def plot_system_performance(self, vehicle_speed_range: np.ndarray,
                              coolant_temp: float = 90.0, engine_load: float = 0.5,
                              radiator_pressure_drop_func: Optional[Callable[[float], float]] = None,
                              save_path: Optional[str] = None):
        """Plot system performance (airflow, power) vs. vehicle speed."""
        from ..utils.plotting import save_plot # Local import

        airflows = []
        powers = []
        control_signals = []

        if radiator_pressure_drop_func is None:
            # Simple constant pressure drop if no function provided
            radiator_pressure_drop_func = lambda speed: 80.0 # Constant 80 Pa

        for speed in vehicle_speed_range:
            # Simulate a step to get the state at this speed
            self.update_system(coolant_temp, speed, engine_load, dt=0.1,
                              radiator_pressure_drop_pa=radiator_pressure_drop_func(speed))
            state = self.get_system_state()
            airflows.append(state['supplementary_airflow_m3s'])
            powers.append(state['total_power_W'])
            control_signals.append(state['control_state']['control_signal'])

        fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

        axes[0].plot(vehicle_speed_range, control_signals, color=COLOR_SCHEMES['default'][0])
        _apply_common_ax_settings(axes[0], ylabel='Control Signal (0-1)', title=f'Cooling Assist Performance (Coolant={coolant_temp}°C, Load={engine_load*100:.0f}%)')

        axes[1].plot(vehicle_speed_range, airflows, color=COLOR_SCHEMES['default'][1])
        _apply_common_ax_settings(axes[1], ylabel='Supplementary Airflow (m³/s)')

        axes[2].plot(vehicle_speed_range, powers, color=COLOR_SCHEMES['default'][2])
        _apply_common_ax_settings(axes[2], xlabel='Vehicle Speed (m/s)', ylabel='Power Consumption (W)')

        plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust for suptitle

        if save_path: save_plot(fig, save_path)
        plt.show()
        plt.close(fig)


# --- Factory Functions ---
def create_default_cooling_assist_system(config_dir: str = "configs/thermal") -> CoolingAssistSystem:
    """Create a default configuration CoolingAssistSystem."""
    try:
        comp_path = os.path.join(config_dir, "electric_compressor.yaml")
        compressor = ElectricCompressor(config_path=comp_path) if os.path.exists(comp_path) else ElectricCompressor()
        # Assume control params are also in electric_compressor.yaml
        control_module = CompressorControlModule(compressor, config_path=comp_path)
        # Get duct params from config if available
        duct_loss = 50.0 # Default
        if os.path.exists(comp_path):
             with open(comp_path, 'r') as f:
                 config = yaml.safe_load(f)
                 duct_loss = config.get('ducting', {}).get('pressure_loss_coeff', duct_loss)

        return CoolingAssistSystem(compressor, control_module, duct_pressure_loss_pa_per_m3s=duct_loss)
    except Exception as e:
        logger.error(f"Failed to create default cooling assist system from config: {e}. Returning basic default.")
        compressor = ElectricCompressor()
        control_module = CompressorControlModule(compressor)
        return CoolingAssistSystem(compressor, control_module)

# Add other factory functions (high_performance, lightweight) similarly, maybe defining
# specific parameters directly or loading different config files if they exist.

def create_high_performance_cooling_assist_system() -> CoolingAssistSystem:
    """Create a high-performance cooling assist system."""
    compressor = ElectricCompressor(max_airflow_m3s=0.35, max_pressure_pa=250, max_power_W=150, efficiency=0.72)
    control_module = CompressorControlModule(compressor, control_strategy=CompressorControl.PID_CONTROL, target_temp_C=88)
    return CoolingAssistSystem(compressor, control_module, duct_pressure_loss_pa_per_m3s=40)

def create_lightweight_cooling_assist_system() -> CoolingAssistSystem:
    """Create a lightweight cooling assist system."""
    compressor = ElectricCompressor(max_airflow_m3s=0.20, max_pressure_pa=150, max_power_W=80, weight_kg=0.5)
    control_module = CompressorControlModule(compressor, control_strategy=CompressorControl.ON_OFF, target_temp_C=92)
    return CoolingAssistSystem(compressor, control_module, duct_pressure_loss_pa_per_m3s=60)

# Example Usage
if __name__ == "__main__":
    # Create default system
    assist_system = create_default_cooling_assist_system()

    print("\n--- Default Cooling Assist Specs ---")
    specs = assist_system.get_system_specs()
    print(yaml.dump(specs, default_flow_style=False))

    # Test performance plot
    print("\n--- Plotting Performance ---")
    # Define a simple radiator pressure drop function: P = k * Q^2
    radiator_k = 100 / (0.2**2) # Assumes 100Pa drop at 0.2 m3/s
    def rad_pressure_drop(speed_mps):
         # Estimate airflow through radiator due to speed first
         ram_air_flow = speed_mps * 0.15 * 0.7 # Speed * Area * Efficiency
         return radiator_k * ram_air_flow**2

    assist_system.plot_system_performance(
         vehicle_speed_range=np.linspace(0, 25, 26),
         coolant_temp=95.0, # Test at higher temp
         engine_load=0.8,
         radiator_pressure_drop_func=rad_pressure_drop
         # save_path="plots/cooling_assist_perf.png" # Optional save
    )

    # Test compressor map plot
    assist_system.compressor.plot_compressor_map() # Optional save path