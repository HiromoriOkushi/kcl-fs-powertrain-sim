"""
Side pod module for Formula Student powertrain simulation.

Models aerodynamic and thermal aspects of side pods, including ducting,
radiator integration, and their impact on vehicle performance.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Callable
import logging
from enum import Enum, auto
import yaml

# Import base components
try:
    from .cooling_system import Radiator, RadiatorType, CoolingFan, FanType
except ImportError:
    # Placeholders if run directly
    class Radiator: pass
    class CoolingFan: pass
    class RadiatorType(Enum): SINGLE_CORE_ALUMINUM=auto(); DOUBLE_CORE_ALUMINUM=auto()
    class FanType(Enum): VARIABLE_SPEED=auto(); DUAL_FAN=auto(); SINGLE_SPEED=auto()
    logger.warning("Could not import base cooling system components. Using placeholders.")

# Constants (import or define fallback)
try:
    from ..utils.constants import AIR_DENSITY_SEA_LEVEL
    from ..utils.plotting import save_plot, _apply_common_ax_settings, COLOR_SCHEMES
except ImportError:
    AIR_DENSITY_SEA_LEVEL = 1.225
    # Fallback plotting utils
    def save_plot(fig, path, **kwargs): pass
    def _apply_common_ax_settings(ax, **kwargs): pass
    COLOR_SCHEMES = {'default': plt.cm.tab10.colors}
    logger.warning("Could not import utils.constants or utils.plotting. Using fallback values/functions.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SidePod")


class SidePodType(Enum):
    """Side pod design philosophies."""
    MINIMAL = auto()
    STANDARD = auto()
    UNDERCUT = auto() # Optimized for flow underneath
    HIGH_DOWNFORCE = auto()
    COOLING_FOCUSED = auto()
    CUSTOM = auto()

class RadiatorOrientation(Enum):
    """Radiator mounting orientation within the side pod."""
    VERTICAL = auto()
    ANGLED = auto() # Angled relative to vertical/horizontal
    HORIZONTAL = auto()
    SPLIT = auto() # Multiple smaller radiators
    CUSTOM = auto()

class SidePod:
    """Models the geometry and basic aero properties of a single side pod."""
    def __init__(self,
                 pod_type: SidePodType = SidePodType.STANDARD,
                 length: float = 0.8,          # m
                 max_width: float = 0.3,       # m
                 max_height: float = 0.35,     # m
                 inlet_area: float = 0.04,     # m²
                 outlet_area: float = 0.05,    # m²
                 radiator_orientation: RadiatorOrientation = RadiatorOrientation.VERTICAL,
                 config_path: Optional[str] = None,
                 custom_params: Optional[Dict] = None):
        """
        Initialize side pod model.

        Args:
            pod_type: Design type of the side pod.
            length: Overall length (m).
            max_width: Maximum width (m).
            max_height: Maximum height (m).
            inlet_area: Air inlet area (m²).
            outlet_area: Air outlet area (m²).
            radiator_orientation: Orientation of the radiator within the pod.
            config_path: Optional path to YAML config file.
            custom_params: Optional dictionary for CUSTOM type or overrides.
        """
        self.pod_type = pod_type
        self.length = length
        self.max_width = max_width
        self.max_height = max_height
        self.inlet_area_m2 = inlet_area
        self.outlet_area_m2 = outlet_area
        self.radiator_orientation = radiator_orientation

        # Aero and other properties (defaults, overridden by type/config)
        self.drag_coefficient_pod: float = 0.25 # Cd contribution of the pod itself
        self.lift_coefficient_pod: float = -0.1 # Cl contribution (negative=downforce)
        self.cooling_efficiency_duct: float = 0.85 # Efficiency of duct capturing/directing air
        self.weight_kg: float = 4.0
        self.radiator_volume_fraction: float = 0.6 # Fraction of internal vol usable for rad

        # Load from config if path provided
        if config_path and os.path.exists(config_path):
             self._load_config(config_path)

        # Apply custom params or set defaults based on type
        params = custom_params or {}
        self._apply_type_defaults_and_custom(params)

        # Derived geometric properties
        self.volume_m3 = self._calculate_volume()
        self.frontal_area_m2 = self._calculate_frontal_area()
        self.duct_loss_coefficient_k = self._calculate_duct_loss_k()

        logger.info(f"Side Pod initialized: Type={self.pod_type.name}, Inlet={self.inlet_area_m2:.3f}m², Outlet={self.outlet_area_m2:.3f}m²")

    def _load_config(self, config_path: str):
        """Load parameters from YAML config file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            pod_config = config.get('side_pod', {}) # Look for 'side_pod' section
            # Safely get enums from names
            pod_type_name = pod_config.get('type', self.pod_type.name).upper()
            self.pod_type = SidePodType[pod_type_name] if pod_type_name in SidePodType.__members__ else self.pod_type
            rad_orient_name = pod_config.get('radiator_orientation', self.radiator_orientation.name).upper()
            self.radiator_orientation = RadiatorOrientation[rad_orient_name] if rad_orient_name in RadiatorOrientation.__members__ else self.radiator_orientation

            self.length = float(pod_config.get('length', self.length))
            self.max_width = float(pod_config.get('max_width', self.max_width))
            self.max_height = float(pod_config.get('max_height', self.max_height))
            self.inlet_area_m2 = float(pod_config.get('inlet_area', self.inlet_area_m2))
            self.outlet_area_m2 = float(pod_config.get('outlet_area', self.outlet_area_m2))
            # Allow custom params in config to override defaults for the loaded type
            self._apply_type_defaults_and_custom(pod_config)
            logger.info(f"Side pod config loaded from {config_path}")
        except Exception as e:
            logger.error(f"Error loading side pod config from {config_path}: {e}. Using existing values.")

    def _apply_type_defaults_and_custom(self, params: Dict):
        """Set default properties based on type, overridden by params."""
        defaults = {}
        # These are illustrative defaults - need tuning based on CFD/testing
        if self.pod_type == SidePodType.MINIMAL:
            defaults = {'drag_coefficient_pod': 0.20, 'lift_coefficient_pod': 0.0, 'cooling_efficiency_duct': 0.75, 'weight_kg': 2.5, 'radiator_volume_fraction': 0.5}
        elif self.pod_type == SidePodType.STANDARD:
            defaults = {'drag_coefficient_pod': 0.25, 'lift_coefficient_pod': -0.1, 'cooling_efficiency_duct': 0.85, 'weight_kg': 4.0, 'radiator_volume_fraction': 0.6}
        elif self.pod_type == SidePodType.UNDERCUT:
            defaults = {'drag_coefficient_pod': 0.28, 'lift_coefficient_pod': -0.25, 'cooling_efficiency_duct': 0.80, 'weight_kg': 4.5, 'radiator_volume_fraction': 0.55}
        elif self.pod_type == SidePodType.HIGH_DOWNFORCE:
            defaults = {'drag_coefficient_pod': 0.32, 'lift_coefficient_pod': -0.40, 'cooling_efficiency_duct': 0.70, 'weight_kg': 5.5, 'radiator_volume_fraction': 0.5}
        elif self.pod_type == SidePodType.COOLING_FOCUSED:
            defaults = {'drag_coefficient_pod': 0.30, 'lift_coefficient_pod': -0.15, 'cooling_efficiency_duct': 0.92, 'weight_kg': 5.0, 'radiator_volume_fraction': 0.7}
        elif self.pod_type == SidePodType.CUSTOM:
             defaults = {'drag_coefficient_pod': 0.27, 'lift_coefficient_pod': -0.15, 'cooling_efficiency_duct': 0.85, 'weight_kg': 4.5, 'radiator_volume_fraction': 0.6}

        self.drag_coefficient_pod = float(params.get('drag_coefficient_pod', defaults.get('drag_coefficient_pod', 0.25)))
        self.lift_coefficient_pod = float(params.get('lift_coefficient_pod', defaults.get('lift_coefficient_pod', -0.1)))
        self.cooling_efficiency_duct = float(params.get('cooling_efficiency_duct', defaults.get('cooling_efficiency_duct', 0.85)))
        self.weight_kg = float(params.get('weight_kg', defaults.get('weight_kg', 4.0)))
        self.radiator_volume_fraction = float(params.get('radiator_volume_fraction', defaults.get('radiator_volume_fraction', 0.6)))

    def _calculate_volume(self) -> float:
        """Estimate side pod volume."""
        # Simple estimate: length * width * height * form_factor
        form_factor = 0.6 # Assume a general form factor
        return self.length * self.max_width * self.max_height * form_factor

    def _calculate_frontal_area(self) -> float:
        """Estimate side pod frontal area contributing to drag."""
        # Based on max width/height, adjusted by shape
        shape_factor = 0.7 # Typical aerodynamic shape factor
        return self.max_width * self.max_height * shape_factor

    def _calculate_duct_loss_k(self) -> float:
        """Estimate the duct pressure loss coefficient k."""
        # Simplified model based on area ratio and type
        area_ratio = self.outlet_area_m2 / self.inlet_area_m2 if self.inlet_area_m2 > 0 else 1.0
        # Base loss + expansion/contraction loss + bend loss (estimated)
        k = 0.5 + 0.5 * abs(1.0 - area_ratio)**1.5 + 0.3
        # Adjust based on type (lower for cooling focused)
        if self.pod_type == SidePodType.COOLING_FOCUSED: k *= 0.8
        if self.pod_type == SidePodType.MINIMAL: k *= 1.2
        return max(0.1, k) # Ensure minimum loss

    def calculate_max_radiator_dimensions(self) -> Tuple[float, float, float]:
        """Estimate max radiator dimensions fitting inside."""
        # This is highly dependent on internal shape, simplified here
        available_volume = self.volume_m3 * self.radiator_volume_fraction
        # Assume radiator roughly occupies a box within the max dimensions
        max_w = self.max_width * 0.85
        max_h = self.max_height * 0.85
        max_t = min(0.1, available_volume / (max_w * max_h) if max_w * max_h > 0 else 0.1) # Cap thickness
        return max_w, max_h, max_t

    def calculate_internal_airflow_m3s(self, vehicle_speed_mps: float) -> float:
        """Calculate airflow entering the side pod inlet."""
        # Q = InletArea * VehicleSpeed * DuctEfficiency
        airflow = self.inlet_area_m2 * vehicle_speed_mps * self.cooling_efficiency_duct
        return max(0.0, airflow)

    def calculate_pressure_drop_pa(self, airflow_m3s: float) -> float:
        """Calculate pressure drop through the side pod ducting."""
        if self.inlet_area_m2 <= 0: return float('inf')
        velocity = airflow_m3s / self.inlet_area_m2
        dynamic_pressure = 0.5 * AIR_DENSITY_SEA_LEVEL * velocity**2
        return self.duct_loss_coefficient_k * dynamic_pressure

    def calculate_aero_forces(self, vehicle_speed_mps: float) -> Tuple[float, float]:
        """Calculate drag and lift/downforce generated by the pod itself."""
        dynamic_pressure = 0.5 * AIR_DENSITY_SEA_LEVEL * vehicle_speed_mps**2
        drag_force = self.drag_coefficient_pod * self.frontal_area_m2 * dynamic_pressure
        lift_force = self.lift_coefficient_pod * self.frontal_area_m2 * dynamic_pressure # Use frontal area as ref
        downforce = -lift_force # Positive downforce convention
        return drag_force, downforce

    def get_side_pod_specs(self) -> Dict:
        """Get side pod specifications."""
        max_rad_dims = self.calculate_max_radiator_dimensions()
        return {
            'type': self.pod_type.name,
            'radiator_orientation': self.radiator_orientation.name,
            'length_m': self.length,
            'max_width_m': self.max_width,
            'max_height_m': self.max_height,
            'inlet_area_m2': self.inlet_area_m2,
            'outlet_area_m2': self.outlet_area_m2,
            'estimated_volume_m3': self.volume_m3,
            'estimated_frontal_area_m2': self.frontal_area_m2,
            'drag_coefficient_pod': self.drag_coefficient_pod,
            'lift_coefficient_pod': self.lift_coefficient_pod,
            'cooling_efficiency_duct': self.cooling_efficiency_duct,
            'weight_kg': self.weight_kg,
            'duct_loss_coefficient_k': self.duct_loss_coefficient_k,
            'max_radiator_width_m': max_rad_dims[0],
            'max_radiator_height_m': max_rad_dims[1],
            'max_radiator_thickness_m': max_rad_dims[2],
        }


class SidePodRadiator:
    """Models a radiator integrated within a side pod."""
    def __init__(self,
                 radiator: Radiator, # Base radiator object
                 side_pod: SidePod,  # SidePod object it's mounted in
                 orientation: Optional[RadiatorOrientation] = None, # If None, use side_pod's default
                 tilt_angle_deg: float = 0.0, # Angle relative to primary orientation plane
                 position_factor: float = 0.5, # 0=front, 1=rear
                 config_path: Optional[str] = None,
                 custom_params: Optional[Dict] = None):
        """
        Initialize a side pod radiator configuration.

        Args:
            radiator: Base Radiator object.
            side_pod: SidePod object housing the radiator.
            orientation: Mounting orientation (overrides side_pod default if given).
            tilt_angle_deg: Tilt angle (degrees) relative to orientation plane.
            position_factor: Position along side pod length (0=front, 1=rear).
            config_path: Optional path to YAML config file (for radiator section).
            custom_params: Optional dictionary for CUSTOM type or overrides.
        """
        if not isinstance(radiator, Radiator): raise TypeError("Radiator must be Radiator instance.")
        if not isinstance(side_pod, SidePod): raise TypeError("Side pod must be SidePod instance.")

        self.base_radiator = radiator
        self.side_pod = side_pod
        self.orientation = orientation if orientation is not None else side_pod.radiator_orientation
        self.tilt_angle_deg = tilt_angle_deg
        self.tilt_angle_rad = np.radians(tilt_angle_deg)
        self.position_factor = np.clip(position_factor, 0.0, 1.0)

        # Load from config if provided (can potentially override base radiator params too)
        if config_path and os.path.exists(config_path):
            self._load_config(config_path)

        # Validate dimensions
        self._validate_dimensions()

        # Calculate efficiencies based on placement
        self.orientation_efficiency = self._calculate_orientation_efficiency()
        self.position_efficiency = self._calculate_position_efficiency()

        logger.info(f"Side Pod Radiator initialized: Orientation={self.orientation.name}, Tilt={self.tilt_angle_deg:.1f}deg, Pos={self.position_factor:.2f}")

    def _load_config(self, config_path: str):
        """Load parameters from YAML, potentially overriding base radiator."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            rad_config = config.get('radiator', {}) # Look for 'radiator' section in side pod config
            # Override base radiator params if present in this config
            self.base_radiator._load_config(config_path) # Reload base with same file
            # Override sidepod specific params
            orient_name = rad_config.get('orientation', self.orientation.name).upper()
            self.orientation = RadiatorOrientation[orient_name] if orient_name in RadiatorOrientation.__members__ else self.orientation
            self.tilt_angle_deg = float(rad_config.get('tilt_angle', self.tilt_angle_deg))
            self.tilt_angle_rad = np.radians(self.tilt_angle_deg)
            self.position_factor = np.clip(float(rad_config.get('position_factor', self.position_factor)), 0.0, 1.0)
            logger.info(f"Side pod radiator config loaded from {config_path}")
        except Exception as e:
             logger.error(f"Error loading side pod radiator config from {config_path}: {e}.")

    def _validate_dimensions(self):
        """Check if radiator dimensions fit within side pod estimates."""
        max_w, max_h, max_t = self.side_pod.calculate_max_radiator_dimensions()
        rad_area = self.base_radiator.core_area
        rad_thick = self.base_radiator.core_thickness

        fits = True
        if rad_thick > max_t * 1.05: # Allow 5% tolerance
             logger.warning(f"Radiator thickness ({rad_thick*1000:.0f}mm) may exceed estimated pod capacity ({max_t*1000:.0f}mm).")
             fits = False
        # Area check is harder due to shape, perform a basic check
        if rad_area > max_w * max_h * 1.1: # Allow 10% tolerance
             logger.warning(f"Radiator area ({rad_area:.3f}m²) seems large for estimated pod capacity ({max_w*max_h:.3f}m²).")
             fits = False
        return fits


    def _calculate_orientation_efficiency(self) -> float:
        """Estimate airflow efficiency based on radiator orientation."""
        # Base efficiency values (tunable)
        eff_map = {
            RadiatorOrientation.VERTICAL: 0.95,
            RadiatorOrientation.ANGLED: 0.90, # Assume 45 deg base angle
            RadiatorOrientation.HORIZONTAL: 0.80,
            RadiatorOrientation.SPLIT: 0.88, # Slightly lower due to splitting flow
            RadiatorOrientation.CUSTOM: 0.85
        }
        base_eff = eff_map.get(self.orientation, 0.85)

        # Tilt adjustment (relative to the orientation's primary plane)
        # Small tilts might improve, large tilts reduce effective area/flow
        tilt_factor = 1.0 - 0.3 * abs(np.sin(self.tilt_angle_rad))**1.5 # Penalty for large tilt
        return base_eff * tilt_factor

    def _calculate_position_efficiency(self) -> float:
        """Estimate airflow efficiency based on position within the pod."""
        # Assumes airflow is strongest near the middle, weaker near front/back
        # Simple quadratic profile peaking at position_factor = 0.5
        efficiency = 1.0 - 0.3 * (abs(self.position_factor - 0.5) * 2)**2
        return max(0.7, efficiency) # Ensure minimum efficiency

    def calculate_effective_airflow_m3s(self, side_pod_airflow_m3s: float, fan_airflow_m3s: float = 0.0) -> float:
        """Calculate effective airflow through the radiator core."""
        # Airflow reaching the radiator face (from side pod flow)
        area_ratio = np.clip(self.base_radiator.core_area / self.side_pod.inlet_area_m2, 0.1, 1.5) # Ratio relative to inlet
        positional_efficiency = self.orientation_efficiency * self.position_efficiency
        ram_air_to_rad = side_pod_airflow_m3s * area_ratio * positional_efficiency

        # Combine ram air and fan air (assuming fan helps pull air through)
        # This interaction is complex; simplified addition here.
        total_airflow = ram_air_to_rad + fan_airflow_m3s * positional_efficiency # Fan also affected by position

        return max(0.0, total_airflow)

    def calculate_heat_rejection(self, coolant_temp_C: float, ambient_temp_C: float,
                               coolant_flow_lpm: float, vehicle_speed_mps: float,
                               fan_airflow_m3s: float = 0.0) -> float:
        """Calculate heat rejection (W) for the side pod radiator."""
        # 1. Calculate airflow entering the side pod
        side_pod_airflow = self.side_pod.calculate_internal_airflow_m3s(vehicle_speed_mps)

        # 2. Calculate effective airflow reaching the radiator
        effective_airflow = self.calculate_effective_airflow_m3s(side_pod_airflow, fan_airflow_m3s)

        # 3. Use base radiator's method with the effective airflow
        heat_rejection_W = self.base_radiator.calculate_heat_rejection(
            coolant_temp_C, ambient_temp_C, coolant_flow_lpm, effective_airflow
        )
        return heat_rejection_W

    def get_radiator_specs(self) -> Dict:
        """Get combined specifications."""
        base_specs = self.base_radiator.get_radiator_specs()
        sidepod_rad_specs = {
            'orientation': self.orientation.name,
            'tilt_angle_deg': self.tilt_angle_deg,
            'position_factor': self.position_factor,
            'orientation_efficiency': self.orientation_efficiency,
            'position_efficiency': self.position_efficiency,
        }
        return {**base_specs, **sidepod_rad_specs}


class SidePodSystem:
    """Integrates SidePod, SidePodRadiator, and optional CoolingFan."""
    def __init__(self,
                 side_pod: SidePod,
                 radiator: SidePodRadiator,
                 cooling_fan: Optional[CoolingFan] = None,
                 is_left_side: bool = True): # Track which side this is
        """Initialize the complete side pod system."""
        if not isinstance(side_pod, SidePod): raise TypeError("side_pod must be SidePod instance.")
        if not isinstance(radiator, SidePodRadiator): raise TypeError("radiator must be SidePodRadiator instance.")
        if cooling_fan and not isinstance(cooling_fan, CoolingFan): raise TypeError("cooling_fan must be CoolingFan instance.")

        self.side_pod = side_pod
        self.radiator = radiator
        self.cooling_fan = cooling_fan
        self.is_left_side = is_left_side

        # State
        self.current_airflow_m3s: float = 0.0 # Airflow through radiator
        self.heat_rejection_W: float = 0.0
        self.drag_force_N: float = 0.0
        self.downforce_N: float = 0.0

        logger.info(f"Side Pod System initialized for {'left' if is_left_side else 'right'} side.")

    def update_fan_control(self, control_signal: float):
        """Update fan speed based on control signal (0-1)."""
        if self.cooling_fan:
            self.cooling_fan.update_control(control_signal)
        # else: logger.warning("Attempted fan control, but no fan present.") # Reduce verbosity

    def calculate_system_airflow_m3s(self, vehicle_speed_mps: float) -> float:
        """Calculate effective airflow through the radiator."""
        side_pod_airflow = self.side_pod.calculate_internal_airflow_m3s(vehicle_speed_mps)
        fan_airflow = self.cooling_fan.current_airflow_m3s if self.cooling_fan else 0.0
        # Use the SidePodRadiator's method to get effective flow at the core
        self.current_airflow_m3s = self.radiator.calculate_effective_airflow_m3s(side_pod_airflow, fan_airflow)
        return self.current_airflow_m3s

    def calculate_heat_rejection(self, coolant_temp_C: float, ambient_temp_C: float,
                               coolant_flow_lpm: float, vehicle_speed_mps: float) -> float:
        """Calculate heat rejection (W) for this side pod system."""
        # Fan airflow is handled internally by calculate_system_airflow
        _ = self.calculate_system_airflow_m3s(vehicle_speed_mps) # Update internal airflow state
        # Use the effective airflow calculated above
        self.heat_rejection_W = self.radiator.base_radiator.calculate_heat_rejection(
             coolant_temp_C, ambient_temp_C, coolant_flow_lpm, self.current_airflow_m3s
        )
        return self.heat_rejection_W

    def calculate_aerodynamic_forces(self, vehicle_speed_mps: float) -> Tuple[float, float]:
        """Calculate drag (N) and downforce (N) for this side pod."""
        drag, downforce = self.side_pod.calculate_aero_forces(vehicle_speed_mps)
        # Add internal drag component (momentum loss)
        internal_drag = AIR_DENSITY_SEA_LEVEL * self.current_airflow_m3s * (vehicle_speed_mps * 0.3) # Approx exit velocity deficit
        self.drag_force_N = drag + internal_drag
        self.downforce_N = downforce
        return self.drag_force_N, self.downforce_N

    def simulate_step(self, coolant_temp_C: float, ambient_temp_C: float,
                    coolant_flow_lpm: float, vehicle_speed_mps: float,
                    auto_fan_control: bool = True, target_temp: float = 90.0) -> Dict:
        """Simulate one step, updating internal state."""
        if auto_fan_control and self.cooling_fan:
             # Simple automatic fan control
             temp_error = coolant_temp_C - target_temp
             control_signal = np.clip(temp_error / 10.0, 0.0, 1.0) # Ramp over 10C
             speed_factor = max(0.0, 1.0 - vehicle_speed_mps / 20.0) # Reduce fan at speed > 20 m/s
             self.update_fan_control(control_signal * speed_factor)
        elif not self.cooling_fan:
             self.update_fan_control(0.0)

        self.calculate_heat_rejection(coolant_temp_C, ambient_temp_C, coolant_flow_lpm, vehicle_speed_mps)
        self.calculate_aerodynamic_forces(vehicle_speed_mps)
        # Note: Coolant temp not updated here, just heat rejection calculated

        return self.get_system_state()

    def get_system_state(self) -> Dict:
        """Get current state."""
        state = {
            'airflow_m3s': self.current_airflow_m3s,
            'heat_rejection_W': self.heat_rejection_W,
            'drag_N': self.drag_force_N,
            'downforce_N': self.downforce_N,
            'side': 'left' if self.is_left_side else 'right'
        }
        if self.cooling_fan: state['fan'] = self.cooling_fan.get_fan_state()
        return state

    def get_system_specs(self) -> Dict:
        """Get component specifications."""
        specs = {
            'side_pod': self.side_pod.get_side_pod_specs(),
            'radiator': self.radiator.get_radiator_specs()
        }
        if self.cooling_fan: specs['fan'] = self.cooling_fan.get_fan_specs()
        return specs

    def analyze_performance(self, vehicle_speed_range: np.ndarray,
                          coolant_temp: float = 90.0, ambient_temp: float = 25.0,
                          coolant_flow_rate: float = 25.0) -> Dict: # Note: Flow rate is per side pod here
        """Analyze performance over a speed range."""
        results = {'vehicle_speeds_mps': vehicle_speed_range, 'airflows': [], 'heat_rejections': [], 'drags': [], 'downforces': []}
        self.update_fan_control(1.0) # Analyze with full fan
        for speed in vehicle_speed_range:
            state = self.simulate_step(coolant_temp, ambient_temp, coolant_flow_rate, speed, auto_fan_control=False)
            results['airflows'].append(state['airflow_m3s'])
            results['heat_rejections'].append(state['heat_rejection_W'])
            results['drags'].append(state['drag_N'])
            results['downforces'].append(state['downforce_N'])
        # Convert lists to arrays
        for key in ['airflows', 'heat_rejections', 'drags', 'downforces']:
             results[key] = np.array(results[key])
        results['conditions'] = {'coolant_temp':coolant_temp, 'ambient_temp':ambient_temp, 'flow_rate_per_pod':coolant_flow_rate}
        return results

    def plot_performance_curves(self, analysis_results: Dict, save_path: Optional[str] = None):
        """Plot performance curves."""
        from ..utils.plotting import save_plot, _apply_common_ax_settings, COLOR_SCHEMES # Local import

        speeds = analysis_results['vehicle_speeds_mps']
        airflows = analysis_results['airflows']
        heat_rejections = analysis_results['heat_rejections']
        drags = analysis_results['drags']
        downforces = analysis_results['downforces']
        conditions = analysis_results['conditions']

        fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        side = 'Left' if self.is_left_side else 'Right'

        # Plot Airflow & Heat Rejection
        ax1 = axes[0]
        ln1 = ax1.plot(speeds, airflows, color=COLOR_SCHEMES['default'][0], label='Airflow')
        _apply_common_ax_settings(ax1, ylabel='Airflow (m³/s)', title=f'{side} Side Pod Performance')
        ax1b = ax1.twinx()
        ln2 = ax1b.plot(speeds, heat_rejections / 1000.0, color=COLOR_SCHEMES['default'][1], label='Heat Rejection')
        ax1b.set_ylabel('Heat Rejection (kW)', color=COLOR_SCHEMES['default'][1])
        ax1b.tick_params(axis='y', labelcolor=COLOR_SCHEMES['default'][1])
        lns = ln1 + ln2
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, loc='upper left')

        # Plot Drag
        axes[1].plot(speeds, drags, color=COLOR_SCHEMES['default'][2])
        _apply_common_ax_settings(axes[1], ylabel='Drag (N)')

        # Plot Downforce
        axes[2].plot(speeds, downforces, color=COLOR_SCHEMES['default'][4])
        _apply_common_ax_settings(axes[2], xlabel='Vehicle Speed (m/s)', ylabel='Downforce (N)')

        fig.suptitle(f"Conditions: {conditions['coolant_temp']}°C Coolant, {conditions['ambient_temp']}°C Ambient, {conditions['flow_rate_per_pod']:.1f} LPM Flow")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        if save_path: save_plot(fig, save_path)
        plt.show()
        plt.close(fig)


class DualSidePodSystem:
    """Manages both left and right SidePodSystems."""
    def __init__(self, left_system: SidePodSystem, right_system: SidePodSystem):
        """Initialize with left and right side pod systems."""
        self.left_system = left_system
        self.right_system = right_system
        # Ensure they are marked correctly
        self.left_system.is_left_side = True
        self.right_system.is_left_side = False
        logger.info("Dual Side Pod System initialized.")

    def update_fan_control(self, left_control: float, right_control: float):
        """Update fan controls for both sides."""
        self.left_system.update_fan_control(left_control)
        self.right_system.update_fan_control(right_control)

    def calculate_total_airflow_m3s(self, vehicle_speed_mps: float) -> float:
        """Calculate total airflow through both pods."""
        return self.left_system.calculate_system_airflow_m3s(vehicle_speed_mps) + \
               self.right_system.calculate_system_airflow_m3s(vehicle_speed_mps)

    def calculate_total_heat_rejection(self, coolant_temp_C: float, ambient_temp_C: float,
                                     total_coolant_flow_lpm: float, vehicle_speed_mps: float) -> float:
        """Calculate total heat rejection, assuming flow splits evenly."""
        flow_per_pod = total_coolant_flow_lpm / 2.0
        left_rej = self.left_system.calculate_heat_rejection(coolant_temp_C, ambient_temp_C, flow_per_pod, vehicle_speed_mps)
        right_rej = self.right_system.calculate_heat_rejection(coolant_temp_C, ambient_temp_C, flow_per_pod, vehicle_speed_mps)
        return left_rej + right_rej

    def calculate_total_aerodynamic_forces(self, vehicle_speed_mps: float) -> Tuple[float, float]:
        """Calculate total drag and downforce from both pods."""
        left_drag, left_downforce = self.left_system.calculate_aerodynamic_forces(vehicle_speed_mps)
        right_drag, right_downforce = self.right_system.calculate_aerodynamic_forces(vehicle_speed_mps)
        return left_drag + right_drag, left_downforce + right_downforce

    def automatic_fan_control(self, coolant_temp: float, target_temp: float = 90.0,
                            hysteresis: float = 5.0, vehicle_speed: float = 0.0):
         """Apply automatic fan control to both sides."""
         self.left_system.automatic_fan_control(coolant_temp, target_temp, hysteresis, vehicle_speed)
         self.right_system.automatic_fan_control(coolant_temp, target_temp, hysteresis, vehicle_speed)

    def get_system_state(self) -> Dict:
        """Get combined state."""
        left = self.left_system.get_system_state()
        right = self.right_system.get_system_state()
        total_airflow = left['airflow_m3s'] + right['airflow_m3s']
        total_heat = left['heat_rejection_W'] + right['heat_rejection_W']
        total_drag = left['drag_N'] + right['drag_N']
        total_downforce = left['downforce_N'] + right['downforce_N']
        return {
            'left': left, 'right': right,
            'total': {'airflow_m3s': total_airflow, 'heat_rejection_W': total_heat,
                      'drag_N': total_drag, 'downforce_N': total_downforce}
        }

    def get_system_specs(self) -> Dict:
        """Get combined specs."""
        left_specs = self.left_system.get_system_specs()
        right_specs = self.right_system.get_system_specs()
        total_weight = left_specs['side_pod']['weight_kg'] + right_specs['side_pod']['weight_kg']
        if 'fan' in left_specs: total_weight += left_specs['fan']['weight_kg']
        if 'fan' in right_specs: total_weight += right_specs['fan']['weight_kg']
        return {'left': left_specs, 'right': right_specs, 'total_weight_kg': total_weight}

    def analyze_system_performance(self, vehicle_speed_range: np.ndarray,
                                 coolant_temp: float = 90.0, ambient_temp: float = 25.0,
                                 total_coolant_flow_rate: float = 50.0) -> Dict:
        """Analyze performance of the dual system."""
        flow_per_pod = total_coolant_flow_rate / 2.0
        left_analysis = self.left_system.analyze_performance(vehicle_speed_range, coolant_temp, ambient_temp, flow_per_pod)
        right_analysis = self.right_system.analyze_performance(vehicle_speed_range, coolant_temp, ambient_temp, flow_per_pod)

        total_airflows = left_analysis['airflows'] + right_analysis['airflows']
        total_heat_rejections = left_analysis['heat_rejections'] + right_analysis['heat_rejections']
        total_drags = left_analysis['drags'] + right_analysis['drags']
        total_downforces = left_analysis['downforces'] + right_analysis['downforces']

        # Aero efficiency L/D
        aero_efficiency = np.divide(total_downforces, total_drags, out=np.zeros_like(total_drags), where=total_drags!=0)
        # Cooling efficiency W/N
        cooling_efficiency = np.divide(total_heat_rejections, total_drags, out=np.zeros_like(total_drags), where=total_drags!=0)

        return {
            'vehicle_speeds_mps': vehicle_speed_range,
            'total_airflows': total_airflows,
            'total_heat_rejections': total_heat_rejections,
            'total_drags': total_drags,
            'total_downforces': total_downforces,
            'aero_efficiency_L_D': aero_efficiency,
            'cooling_efficiency_W_N': cooling_efficiency,
            'left_analysis': left_analysis, # Include individual results
            'right_analysis': right_analysis,
            'conditions': {'coolant_temp':coolant_temp, 'ambient_temp':ambient_temp, 'total_flow_rate':total_coolant_flow_rate}
        }

    def plot_combined_performance(self, analysis_results: Dict, save_path: Optional[str] = None):
        """Plot combined performance using the centralized plotting utility."""
        from ..utils.plotting import save_plot, _apply_common_ax_settings, COLOR_SCHEMES # Local import

        speeds = analysis_results['vehicle_speeds_mps']
        airflows = analysis_results['total_airflows']
        heat_rejections = analysis_results['total_heat_rejections']
        drags = analysis_results['total_drags']
        downforces = analysis_results['total_downforces']
        aero_eff = analysis_results['aero_efficiency_L_D']
        cool_eff = analysis_results['cooling_efficiency_W_N']
        conditions = analysis_results['conditions']

        fig, axes = plt.subplots(3, 1, figsize=(12, 14), sharex=True)

        # Plot Heat Rejection & Airflow
        ax1 = axes[0]
        ln1 = ax1.plot(speeds, heat_rejections / 1000.0, color=COLOR_SCHEMES['default'][1], label='Total Heat Rejection')
        _apply_common_ax_settings(ax1, ylabel='Heat Rejection (kW)', title='Dual Side Pod System Performance')
        ax1b = ax1.twinx()
        ln2 = ax1b.plot(speeds, airflows, color=COLOR_SCHEMES['default'][0], label='Total Airflow')
        ax1b.set_ylabel('Airflow (m³/s)', color=COLOR_SCHEMES['default'][0])
        ax1b.tick_params(axis='y', labelcolor=COLOR_SCHEMES['default'][0])
        lns = ln1 + ln2
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, loc='upper left')

        # Plot Drag & Downforce
        axes[1].plot(speeds, drags, color=COLOR_SCHEMES['default'][2], label='Total Drag')
        axes[1].plot(speeds, downforces, color=COLOR_SCHEMES['default'][4], label='Total Downforce')
        _apply_common_ax_settings(axes[1], ylabel='Aerodynamic Force (N)')
        axes[1].legend(loc='best')

        # Plot Efficiencies
        ax3 = axes[2]
        ln3 = ax3.plot(speeds, aero_eff, color=COLOR_SCHEMES['default'][5], label='Aero Efficiency (L/D)')
        _apply_common_ax_settings(ax3, xlabel='Vehicle Speed (m/s)', ylabel='Aero Efficiency (L/D)')
        ax3b = ax3.twinx()
        ln4 = ax3b.plot(speeds, cool_eff, color=COLOR_SCHEMES['default'][6], label='Cooling Efficiency (W/N)')
        ax3b.set_ylabel('Cooling Efficiency (W/N)', color=COLOR_SCHEMES['default'][6])
        ax3b.tick_params(axis='y', labelcolor=COLOR_SCHEMES['default'][6])
        lns = ln3 + ln4
        labs = [l.get_label() for l in lns]
        ax3.legend(lns, labs, loc='center right')

        fig.suptitle(f"Conditions: {conditions['coolant_temp']}°C Coolant, {conditions['ambient_temp']}°C Ambient, {conditions['total_flow_rate']} LPM Total Flow")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        if save_path: save_plot(fig, save_path)
        plt.show()
        plt.close(fig)

# --- Factory Functions ---
# (Keep existing factory functions, potentially update to use config files)
def create_standard_side_pod_system(config_dir: str = "configs/thermal") -> DualSidePodSystem:
    """Create a standard dual side pod system, potentially loading from config."""
    try:
        # Assume a 'standard_sidepod.yaml' or similar exists, or use defaults
        # For simplicity, we'll use the defaults defined in the classes if no file found
        pod_config_path = os.path.join(config_dir, "side_pod_standard.yaml") # Example path
        fan_config_path = os.path.join(config_dir, "fan_standard.yaml")
        rad_config_path = os.path.join(config_dir, "radiator_standard.yaml")

        # Create components, loading from specific files if they exist
        left_pod = SidePod(pod_type=SidePodType.STANDARD, config_path=pod_config_path)
        right_pod = SidePod(pod_type=SidePodType.STANDARD, config_path=pod_config_path) # Assume symmetry
        left_rad_base = Radiator(config_path=rad_config_path)
        right_rad_base = Radiator(config_path=rad_config_path)
        left_rad = SidePodRadiator(left_rad_base, left_pod)
        right_rad = SidePodRadiator(right_rad_base, right_pod)
        left_fan = CoolingFan(config_path=fan_config_path) if os.path.exists(fan_config_path) else CoolingFan(fan_type=FanType.VARIABLE_SPEED)
        right_fan = CoolingFan(config_path=fan_config_path) if os.path.exists(fan_config_path) else CoolingFan(fan_type=FanType.VARIABLE_SPEED)

        left_sys = SidePodSystem(left_pod, left_rad, left_fan, is_left_side=True)
        right_sys = SidePodSystem(right_pod, right_rad, right_fan, is_left_side=False)
        return DualSidePodSystem(left_sys, right_sys)
    except Exception as e:
        logger.error(f"Error creating standard side pod system: {e}. Returning basic default.")
        # Basic fallback
        lp = SidePod()
        rp = SidePod()
        lr = SidePodRadiator(Radiator(), lp)
        rr = SidePodRadiator(Radiator(), rp)
        return DualSidePodSystem(SidePodSystem(lp, lr), SidePodSystem(rp, rr))

# Implement other factory functions (aero_optimized, cooling_optimized, minimum_weight) similarly,
# potentially pointing to different config files or defining parameters directly.

def create_aero_optimized_side_pod_system() -> DualSidePodSystem:
    """Create an aero-optimized dual side pod system."""
    # Example direct parameter definition for aero focus
    left_pod = SidePod(pod_type=SidePodType.UNDERCUT, length=0.85, max_width=0.26, max_height=0.30, inlet_area=0.03)
    right_pod = SidePod(pod_type=SidePodType.UNDERCUT, length=0.85, max_width=0.26, max_height=0.30, inlet_area=0.03)
    left_rad_base = Radiator(core_area=0.13, core_thickness=0.04)
    right_rad_base = Radiator(core_area=0.13, core_thickness=0.04)
    left_rad = SidePodRadiator(left_rad_base, left_pod, orientation=RadiatorOrientation.ANGLED, tilt_angle_deg=15)
    right_rad = SidePodRadiator(right_rad_base, right_pod, orientation=RadiatorOrientation.ANGLED, tilt_angle_deg=15)
    left_fan = CoolingFan(max_airflow_m3s=0.15, diameter_m=0.22)
    right_fan = CoolingFan(max_airflow_m3s=0.15, diameter_m=0.22)
    return DualSidePodSystem(SidePodSystem(left_pod, left_rad, left_fan, True), SidePodSystem(right_pod, right_rad, right_fan, False))

def create_cooling_optimized_side_pod_system() -> DualSidePodSystem:
    """Create a cooling-optimized dual side pod system."""
    left_pod = SidePod(pod_type=SidePodType.COOLING_FOCUSED, inlet_area=0.045, outlet_area=0.055)
    right_pod = SidePod(pod_type=SidePodType.COOLING_FOCUSED, inlet_area=0.045, outlet_area=0.055)
    left_rad_base = Radiator(radiator_type=RadiatorType.DOUBLE_CORE_ALUMINUM, core_area=0.16, core_thickness=0.05)
    right_rad_base = Radiator(radiator_type=RadiatorType.DOUBLE_CORE_ALUMINUM, core_area=0.16, core_thickness=0.05)
    left_rad = SidePodRadiator(left_rad_base, left_pod)
    right_rad = SidePodRadiator(right_rad_base, right_pod)
    left_fan = CoolingFan(fan_type=FanType.DUAL_FAN, max_airflow_m3s=0.25, diameter_m=0.22) # Representing dual fan per pod
    right_fan = CoolingFan(fan_type=FanType.DUAL_FAN, max_airflow_m3s=0.25, diameter_m=0.22)
    return DualSidePodSystem(SidePodSystem(left_pod, left_rad, left_fan, True), SidePodSystem(right_pod, right_rad, right_fan, False))

def create_minimum_weight_side_pod_system() -> DualSidePodSystem:
    """Create a minimum weight dual side pod system."""
    left_pod = SidePod(pod_type=SidePodType.MINIMAL, length=0.7, max_width=0.24)
    right_pod = SidePod(pod_type=SidePodType.MINIMAL, length=0.7, max_width=0.24)
    left_rad_base = Radiator(core_area=0.12, core_thickness=0.03, tube_rows=1)
    right_rad_base = Radiator(core_area=0.12, core_thickness=0.03, tube_rows=1)
    left_rad = SidePodRadiator(left_rad_base, left_pod)
    right_rad = SidePodRadiator(right_rad_base, right_pod)
    left_fan = CoolingFan(fan_type=FanType.SINGLE_SPEED, max_airflow_m3s=0.15, weight_kg=0.4)
    right_fan = CoolingFan(fan_type=FanType.SINGLE_SPEED, max_airflow_m3s=0.15, weight_kg=0.4)
    return DualSidePodSystem(SidePodSystem(left_pod, left_rad, left_fan, True), SidePodSystem(right_pod, right_rad, right_fan, False))


# Example Usage
if __name__ == "__main__":
    cooling_system = create_cooling_optimized_side_pod_system()

    print("\n--- Cooling Optimized Side Pod System Specs ---")
    specs = cooling_system.get_system_specs()
    # Use yaml dump for cleaner printing of nested specs
    print(yaml.dump(specs, default_flow_style=False, sort_keys=False, indent=2))

    print("\n--- Analyzing Cooling Optimized System Performance ---")
    speeds = np.linspace(0, 30, 11) # 0 to 30 m/s
    perf_data = cooling_system.analyze_system_performance(speeds)

    print("\nPerformance Summary:")
    print("Speed | Airflow | Heat Rej | Drag | Downforce | Aero Eff | Cool Eff")
    print("(m/s) | (m³/s)  |   (kW)   | (N)  |    (N)    |   (L/D)  |  (W/N)")
    print("-" * 75)
    for i, speed in enumerate(perf_data['vehicle_speeds_mps']):
         print(f"{speed:5.1f} | {perf_data['total_airflows'][i]:7.3f} | {perf_data['total_heat_rejections'][i]/1000.0:8.2f} | {perf_data['total_drags'][i]:6.1f} | {perf_data['total_downforces'][i]:9.1f} | {perf_data['aero_efficiency_L_D'][i]:8.2f} | {perf_data['cooling_efficiency_W_N'][i]:8.1f}")

    # Plotting (requires utils.plotting)
    try:
         from ..utils.plotting import set_plot_style
         set_plot_style('clean')
         cooling_system.plot_combined_performance(perf_data)
         # cooling_system.plot_combined_performance(perf_data, save_path="plots/sidepod_cooling_optimized_perf.png")
    except ImportError:
         print("\nPlotting skipped: utils.plotting not found.")
    except Exception as e:
        print(f"\nPlotting error: {e}")