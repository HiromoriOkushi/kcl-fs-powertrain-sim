"""
Engine thermal management module for Formula Student powertrain simulation.

Models thermal behavior, including heat generation, transfer between components
(engine block, oil, coolant), and interaction with the cooling system.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Callable
import yaml
from scipy.interpolate import interp1d
import logging

# Assuming MotorcycleEngine might be needed for properties
try:
    from .motorcycle_engine import MotorcycleEngine
except ImportError:
    class MotorcycleEngine: pass # Placeholder
    MotorcycleEngine = None

# Assuming CoolingSystem from thermal package might be needed
try:
    from ..thermal.cooling_system import CoolingSystem as ExternalCoolingSystem
except ImportError:
    class ExternalCoolingSystem: pass # Placeholder
    ExternalCoolingSystem = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("EngineThermal")

# Constants (should ideally be imported from utils.constants)
WATER_SPECIFIC_HEAT = 4186.0  # J/(kg·K)
AIR_DENSITY_SEA_LEVEL = 1.225 # kg/m³
AIR_SPECIFIC_HEAT_CP = 1005.0 # J/(kg·K)

class ThermalConfig:
    """Configuration parameters for engine thermal model."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize with defaults or load from file."""
        # --- Default Thermal Properties ---
        # Specific Heats (J/kg·K)
        self.specific_heat_engine_block: float = 500.0  # Typical for Aluminum/Steel mix
        self.specific_heat_engine_oil: float = 2100.0   # Typical for engine oil
        self.specific_heat_coolant: float = 3800.0      # Typical for 50/50 Glycol/Water

        # Component Effective Thermal Masses (kg) - Represents the mass involved in heat storage
        self.mass_engine_block: float = 25.0 # Effective thermal mass of block/head
        self.mass_engine_oil: float = 2.8 # Includes oil in sump and passages
        self.mass_coolant_engine: float = 1.5 # Coolant within engine block/head jackets

        # Heat Transfer Coefficients (W/K) - Lumped coefficients between components
        # These are highly dependent on geometry, flow rates, etc. - Requires tuning/CFD
        self.htc_oil_to_block: float = 800.0   # Heat transfer between oil and block
        self.htc_coolant_to_block: float = 1500.0 # Heat transfer between coolant and block
        self.htc_block_to_ambient: float = 40.0  # Convection/radiation from block to air

        # Simplified Heat Distribution Factors (fraction of waste heat)
        self.heat_distribution = {
            'coolant': 0.55, # Fraction of waste heat going to coolant
            'oil': 0.20,     # Fraction of waste heat going to oil
            'exhaust': 0.15, # Fraction lost directly through exhaust gas
            'ambient': 0.10  # Fraction lost directly to ambient (radiation/convection)
        }

        # Temperature Limits (from thermal_limits.yaml usually, but defaults here)
        self.optimal_temp_engine: Tuple[float, float] = (85.0, 100.0)
        self.warning_temp_engine: float = 105.0
        self.critical_temp_engine: float = 115.0

        self.warning_temp_coolant: float = 98.0
        self.critical_temp_coolant: float = 108.0

        self.warning_temp_oil: float = 125.0
        self.critical_temp_oil: float = 135.0

        if config_path:
            self.load_from_file(config_path)

    def load_from_file(self, config_path: str):
        """Load thermal configuration from YAML file."""
        if not os.path.exists(config_path):
            logger.warning(f"Thermal configuration file not found: {config_path}. Using defaults.")
            return
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

            # Load specific heats
            sh = config.get('specific_heat', {})
            self.specific_heat_engine_block = float(sh.get('engine_block', self.specific_heat_engine_block))
            self.specific_heat_engine_oil = float(sh.get('engine_oil', self.specific_heat_engine_oil))
            self.specific_heat_coolant = float(sh.get('coolant', self.specific_heat_coolant))

            # Load masses
            mass = config.get('mass', {})
            self.mass_engine_block = float(mass.get('engine_block', self.mass_engine_block))
            self.mass_engine_oil = float(mass.get('engine_oil', self.mass_engine_oil))
            self.mass_coolant_engine = float(mass.get('coolant_engine', self.mass_coolant_engine)) # Coolant in engine

            # Load heat transfer coefficients
            htc = config.get('heat_transfer', {})
            self.htc_oil_to_block = float(htc.get('oil_to_block', self.htc_oil_to_block))
            self.htc_coolant_to_block = float(htc.get('coolant_to_block', self.htc_coolant_to_block))
            self.htc_block_to_ambient = float(htc.get('block_to_ambient', self.htc_block_to_ambient))

            # Load heat distribution factors
            dist = config.get('heat_distribution', {})
            self.heat_distribution['coolant'] = float(dist.get('coolant', self.heat_distribution['coolant']))
            self.heat_distribution['oil'] = float(dist.get('oil', self.heat_distribution['oil']))
            self.heat_distribution['exhaust'] = float(dist.get('exhaust', self.heat_distribution['exhaust']))
            self.heat_distribution['ambient'] = float(dist.get('ambient', self.heat_distribution['ambient']))
            # Normalize distribution factors if they don't sum to 1
            total_dist = sum(self.heat_distribution.values())
            if abs(total_dist - 1.0) > 1e-3:
                logger.warning(f"Heat distribution factors do not sum to 1 ({total_dist:.2f}). Normalizing.")
                for k in self.heat_distribution:
                    self.heat_distribution[k] /= total_dist

            # Load temperature limits (can also be loaded from thermal_limits.yaml)
            limits_eng = config.get('engine_limits', {})
            self.optimal_temp_engine = tuple(limits_eng.get('optimal_range', self.optimal_temp_engine))
            self.warning_temp_engine = float(limits_eng.get('warning', self.warning_temp_engine))
            self.critical_temp_engine = float(limits_eng.get('critical', self.critical_temp_engine))
            # Add similar loading for coolant and oil limits if defined in this file

            logger.info(f"Thermal configuration loaded from {config_path}")

        except Exception as e:
            logger.error(f"Error loading thermal config from {config_path}: {e}. Using defaults.")

    def get_thermal_capacities(self) -> Dict[str, float]:
        """Calculate thermal capacities (Mass * SpecificHeat) in J/K."""
        return {
            'engine_block': self.mass_engine_block * self.specific_heat_engine_block,
            'engine_oil': self.mass_engine_oil * self.specific_heat_engine_oil,
            'coolant_engine': self.mass_coolant_engine * self.specific_heat_coolant
        }

    def to_dict(self) -> Dict:
        """Convert configuration to dictionary."""
        return self.__dict__ # Return all attributes


class EngineHeatModel:
    """Calculates heat generation and transfer within the engine."""

    def __init__(self, config: ThermalConfig, engine: Optional[MotorcycleEngine] = None):
        """
        Initialize engine heat model.

        Args:
            config: ThermalConfig instance.
            engine: Optional MotorcycleEngine instance for power/fuel calculations.
        """
        self.config = config
        self.engine = engine

        # Fuel properties (needed for heat calculation) - Default to E85 if engine not provided
        # It's better if the FuelSystem provides this.
        self.fuel_energy_density_J_kg = 29.2e6 # Approx E85
        if engine and hasattr(engine, 'fuel_properties'):
             # Assuming engine has loaded fuel properties
             self.fuel_energy_density_J_kg = engine.fuel_properties.energy_density_J_per_kg

    def calculate_heat_generation(self, rpm: float, throttle: float) -> Dict[str, float]:
        """
        Calculate heat generation sources based on engine operating point.

        Args:
            rpm: Engine speed (RPM).
            throttle: Throttle position (0-1).

        Returns:
            Dictionary with heat generation rates (W) for 'total', 'coolant', 'oil', etc.
        """
        if self.engine is None:
            logger.warning("Engine model needed for accurate heat generation calculation. Using estimates.")
            # Estimate power and fuel flow if no engine model
            # Very rough estimation
            max_power_kw = 70.0 # Example
            power_kw = max_power_kw * throttle * (rpm / 14000.0)**0.8
            fuel_flow_g_s = (power_kw * 400.0) / 3600.0 # Estimate from BSFC
        else:
            # Use engine model methods
            power_kw = self.engine.get_power(rpm, throttle)
            fuel_flow_g_s = self.engine.get_fuel_consumption(rpm, throttle)

        fuel_power_watts = (fuel_flow_g_s / 1000.0) * self.fuel_energy_density_J_kg
        engine_power_watts = power_kw * 1000.0

        # Total waste heat
        total_waste_heat_watts = max(0.0, fuel_power_watts - engine_power_watts)

        # Distribute waste heat
        dist = self.config.heat_distribution
        heat_to_coolant = total_waste_heat_watts * dist['coolant']
        heat_to_oil = total_waste_heat_watts * dist['oil']
        heat_to_exhaust = total_waste_heat_watts * dist['exhaust']
        heat_to_ambient = total_waste_heat_watts * dist['ambient']

        return {
            'total_waste': total_waste_heat_watts,
            'to_coolant': heat_to_coolant,
            'to_oil': heat_to_oil,
            'to_exhaust': heat_to_exhaust, # Usually not tracked further
            'to_ambient': heat_to_ambient  # Direct loss from block
        }

    def calculate_internal_heat_transfer(self, temps: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate heat transfer rates between internal components (W).

        Args:
            temps: Dictionary with current temperatures {'engine', 'oil', 'coolant'}.

        Returns:
            Dictionary with heat transfer rates {'oil_to_block', 'coolant_to_block'}.
            Positive value means heat flows FROM the first component TO the second.
        """
        eng_temp = temps['engine']
        oil_temp = temps['oil']
        cool_temp = temps['coolant']

        # Heat transfer from oil TO engine block
        q_oil_to_block = self.config.htc_oil_to_block * (oil_temp - eng_temp)

        # Heat transfer from coolant TO engine block
        q_coolant_to_block = self.config.htc_coolant_to_block * (cool_temp - eng_temp)

        return {
            # Note the signs: positive Q means heat flows FROM first TO second
            'oil_to_block': -q_oil_to_block,  # Heat from block TO oil
            'coolant_to_block': -q_coolant_to_block # Heat from block TO coolant
        }

    def calculate_ambient_heat_loss(self, temps: Dict[str, float], ambient_temp: float, vehicle_speed: float) -> Dict[str, float]:
         """
         Calculate heat loss from components directly to the ambient air (W).

         Args:
             temps: Dictionary with current temperatures {'engine', 'oil', 'coolant'}.
             ambient_temp: Ambient air temperature (°C).
             vehicle_speed: Vehicle speed (m/s).

         Returns:
             Dictionary with heat loss rates {'block_to_ambient', 'oil_to_ambient'}.
             Positive value means heat flows FROM component TO ambient.
         """
         eng_temp = temps['engine']
         oil_temp = temps['oil']
         # Coolant loss primarily via radiator, handled by CoolingSystem

         # Heat loss from engine block to ambient
         # HTC increases with speed (convection)
         htc_ambient = self.config.htc_block_to_ambient * (1 + 0.05 * vehicle_speed) # Simple speed dependence
         q_block_to_ambient = htc_ambient * (eng_temp - ambient_temp)

         # Simplified oil sump cooling
         # Assume a small effective area and HTC for oil sump exposed to air
         htc_oil_ambient = 20.0 * (1 + 0.08 * vehicle_speed) # W/K total, includes area factor
         q_oil_to_ambient = htc_oil_ambient * (oil_temp - ambient_temp)

         return {
             'block_to_ambient': max(0, q_block_to_ambient), # Ensure non-negative loss
             'oil_to_ambient': max(0, q_oil_to_ambient)
         }


# --- External Cooling System (Placeholder/Interface) ---
# This defines the expected interface for the cooling system that interacts
# with the ThermalSimulation. The actual implementation comes from thermal.cooling_system.
class CoolingSystemInterface:
    def calculate_heat_rejection(self, coolant_temp: float, ambient_temp: float, coolant_flow_rate: float, vehicle_speed: float) -> float:
        """Calculates heat rejection by the radiator system."""
        raise NotImplementedError
    def get_total_coolant_mass(self) -> float:
        """Returns the total mass of coolant in the external system (radiator, hoses)."""
        raise NotImplementedError


class ThermalSimulation:
    """Simulates the engine's thermal dynamics over time."""

    def __init__(self, engine_model: EngineHeatModel, cooling_system: CoolingSystemInterface):
        """
        Initialize thermal simulation.

        Args:
            engine_model: EngineHeatModel instance.
            cooling_system: Instance conforming to CoolingSystemInterface.
        """
        self.engine_model = engine_model
        self.cooling_system = cooling_system
        self.config = engine_model.config # Use the config from the heat model

        # Get thermal capacities
        self.capacities = self.config.get_thermal_capacities() # J/K
        # Add capacity for coolant in the external system
        coolant_mass_external = self.cooling_system.get_total_coolant_mass() if hasattr(self.cooling_system, 'get_total_coolant_mass') else 1.5 # Default guess
        self.capacities['coolant_external'] = coolant_mass_external * self.config.specific_heat_coolant
        self.capacities['coolant_total'] = self.capacities['coolant_engine'] + self.capacities['coolant_external']

        if any(c <= 0 for c in self.capacities.values()):
            logger.warning(f"Zero or negative thermal capacity detected: {self.capacities}. Temperature changes might be unstable.")
            # Set very small positive values to avoid division by zero
            for k,v in self.capacities.items():
                if v <= 0: self.capacities[k] = 1e-3


        # Simulation state (temperatures in Celsius)
        self.temps = {
            'engine': 25.0,
            'oil': 25.0,
            'coolant': 25.0 # Represents average coolant temp in the whole system
        }

        # History storage
        self.history = {'time': [], 'temps': [], 'heat_flows': []}

    def reset(self, initial_temps: Dict[str, float] = None):
        """Reset simulation to initial state."""
        if initial_temps:
            self.temps['engine'] = initial_temps.get('engine', 25.0)
            self.temps['oil'] = initial_temps.get('oil', 25.0)
            self.temps['coolant'] = initial_temps.get('coolant', 25.0)
        else:
            self.temps = {'engine': 25.0, 'oil': 25.0, 'coolant': 25.0}
        self.history = {'time': [], 'temps': [], 'heat_flows': []}
        logger.info(f"Thermal simulation reset. Initial temps: {self.temps}")

    def run_step(self, rpm: float, throttle: float, ambient_temp: float,
               vehicle_speed: float, coolant_flow_rate: float, dt: float) -> Dict[str, float]:
        """
        Run a single simulation step using Euler integration.

        Args:
            rpm: Engine RPM.
            throttle: Throttle position (0-1).
            ambient_temp: Ambient temperature (°C).
            vehicle_speed: Vehicle speed (m/s).
            coolant_flow_rate: Coolant flow rate (L/min).
            dt: Time step (s).

        Returns:
            Dictionary with updated temperatures.
        """
        # --- 1. Calculate Heat Generation ---
        heat_gen = self.engine_model.calculate_heat_generation(rpm, throttle)

        # --- 2. Calculate Heat Transfer ---
        heat_transfer_internal = self.engine_model.calculate_internal_heat_transfer(self.temps)
        heat_loss_ambient = self.engine_model.calculate_ambient_heat_loss(self.temps, ambient_temp, vehicle_speed)
        heat_rejection_radiator = self.cooling_system.calculate_heat_rejection(
            self.temps['coolant'], ambient_temp, coolant_flow_rate, vehicle_speed
        )

        # --- 3. Calculate Net Heat Flow for Each Component ---
        # Engine Block: Gains from oil/coolant sources, loses to ambient, loses to internal transfer
        q_net_engine = (heat_gen['to_ambient'] # Direct heat gen to ambient (handled by loss)
                       - heat_transfer_internal['oil_to_block'] # From block to oil
                       - heat_transfer_internal['coolant_to_block'] # From block to coolant
                       - heat_loss_ambient['block_to_ambient']) # From block to ambient air

        # Oil: Gains from engine source, gains from block, loses to ambient
        q_net_oil = (heat_gen['to_oil']
                    + heat_transfer_internal['oil_to_block'] # From block to oil
                    - heat_loss_ambient['oil_to_ambient'])

        # Coolant: Gains from engine source, gains from block, loses via radiator
        q_net_coolant = (heat_gen['to_coolant']
                        + heat_transfer_internal['coolant_to_block'] # From block to coolant
                        - heat_rejection_radiator) # Rejected by external system

        # --- 4. Update Temperatures (dT = Q * dt / C) ---
        delta_t_engine = (q_net_engine * dt) / self.capacities['engine_block']
        delta_t_oil = (q_net_oil * dt) / self.capacities['engine_oil']
        # Use total coolant capacity for the average coolant temperature change
        delta_t_coolant = (q_net_coolant * dt) / self.capacities['coolant_total']

        self.temps['engine'] += delta_t_engine
        self.temps['oil'] += delta_t_oil
        self.temps['coolant'] += delta_t_coolant

        # Clamp temperatures (using config limits, ensure they are loaded)
        self.temps['engine'] = np.clip(self.temps['engine'], ambient_temp - 15, self.config.critical_temp_engine + 20)
        self.temps['coolant'] = np.clip(self.temps['coolant'], ambient_temp - 15, self.config.critical_temp_coolant + 15)
        self.temps['oil'] = np.clip(self.temps['oil'], ambient_temp - 15, self.config.critical_temp_oil + 20)


        # --- 5. Store History ---
        current_time = self.history['time'][-1] + dt if self.history['time'] else dt
        self.history['time'].append(current_time)
        self.history['temps'].append(self.temps.copy())
        self.history['heat_flows'].append({
            'gen_total': heat_gen['total_waste'],
            'gen_coolant': heat_gen['to_coolant'],
            'gen_oil': heat_gen['to_oil'],
            'tr_oil_block': heat_transfer_internal['oil_to_block'],
            'tr_coolant_block': heat_transfer_internal['coolant_to_block'],
            'loss_block_amb': heat_loss_ambient['block_to_ambient'],
            'loss_oil_amb': heat_loss_ambient['oil_to_ambient'],
            'rej_radiator': heat_rejection_radiator
        })

        return self.temps.copy()

    def run_profile(self, profile_data: pd.DataFrame, dt: float = 0.1) -> pd.DataFrame:
        """
        Run simulation over a time profile provided as a DataFrame.

        Args:
            profile_data: DataFrame with columns 'time', 'rpm', 'throttle', 'ambient_temp', 'vehicle_speed', 'coolant_flow'.
            dt: Simulation time step (s).

        Returns:
            DataFrame with simulation results including temperatures.
        """
        required_cols = ['time', 'rpm', 'throttle', 'ambient_temp', 'vehicle_speed', 'coolant_flow']
        if not all(col in profile_data.columns for col in required_cols):
            raise ValueError(f"Profile data missing required columns: {required_cols}")

        # Reset simulation, start at first profile temperature or 25C
        initial_temp = profile_data['ambient_temp'].iloc[0] if 'ambient_temp' in profile_data else 25.0
        self.reset(initial_temps={'engine': initial_temp+5, 'oil': initial_temp, 'coolant': initial_temp})

        profile_time = profile_data['time'].values
        sim_times = np.arange(profile_time[0], profile_time[-1] + dt, dt)

        # Create interpolation functions for inputs
        interp_rpm = interp1d(profile_time, profile_data['rpm'], bounds_error=False, fill_value='extrapolate')
        interp_throttle = interp1d(profile_time, profile_data['throttle'], bounds_error=False, fill_value='extrapolate')
        interp_ambient = interp1d(profile_time, profile_data['ambient_temp'], bounds_error=False, fill_value='extrapolate')
        interp_speed = interp1d(profile_time, profile_data['vehicle_speed'], bounds_error=False, fill_value='extrapolate')
        interp_flow = interp1d(profile_time, profile_data['coolant_flow'], bounds_error=False, fill_value='extrapolate')

        for t in sim_times[1:]: # Start from the second time step
            # Interpolate inputs at current time t
            current_rpm = float(interp_rpm(t))
            current_throttle = float(interp_throttle(t))
            current_ambient = float(interp_ambient(t))
            current_speed = float(interp_speed(t))
            current_flow = float(interp_flow(t))

            # Run simulation step
            self.run_step(current_rpm, current_throttle, current_ambient, current_speed, current_flow, dt)

        # Create results DataFrame
        results_df = pd.DataFrame({
            'time': self.history['time'],
            'engine_temp': [t['engine'] for t in self.history['temps']],
            'coolant_temp': [t['coolant'] for t in self.history['temps']],
            'oil_temp': [t['oil'] for t in self.history['temps']],
            'heat_rejected_kw': [h['rej_radiator']/1000.0 for h in self.history['heat_flows']]
        })
        logger.info(f"Profile simulation complete. Max temps: Eng={results_df['engine_temp'].max():.1f}C, Cool={results_df['coolant_temp'].max():.1f}C, Oil={results_df['oil_temp'].max():.1f}C")
        return results_df


    def run_steady_state(self, rpm: float, throttle: float, ambient_temp: float,
                       vehicle_speed: float, coolant_flow_rate: float,
                       max_time: float = 1200.0, tolerance: float = 0.01) -> Dict[str, float]:
        """
        Run simulation until steady state temperatures are reached.

        Args:
            rpm, throttle, ambient_temp, vehicle_speed, coolant_flow_rate: Operating conditions.
            max_time: Maximum simulation time (s).
            tolerance: Temperature change tolerance (°C/s) to define steady state.

        Returns:
            Dictionary with steady state temperatures and time to reach steady state.
        """
        self.reset(initial_temps={'engine': ambient_temp+5, 'oil': ambient_temp, 'coolant': ambient_temp})
        dt = 0.5 # Use a slightly larger time step for steady state convergence
        time = 0.0
        last_temps = self.temps.copy()

        while time < max_time:
            temps = self.run_step(rpm, throttle, ambient_temp, vehicle_speed, coolant_flow_rate, dt)
            time += dt

            # Check for convergence
            temp_change_rate = abs(temps['engine'] - last_temps['engine']) / dt
            if time > 30 and temp_change_rate < tolerance: # Check after initial warmup
                 logger.info(f"Steady state reached at t={time:.1f}s. Temps: {temps}")
                 return {**temps, 'time_to_steady': time}

            last_temps = temps.copy()

        logger.warning(f"Steady state not reached within max_time={max_time}s. Returning final temps.")
        return {**self.temps, 'time_to_steady': max_time}

    # --- Plotting Methods ---
    def plot_temperature_profile(self, save_path: Optional[str] = None):
        """Plot temperature profile over time using the centralized plotting function."""
        if not self.history['time']:
             logger.error("No simulation history to plot.")
             return

        from ..utils.plotting import plot_thermal_performance, save_plot

        # Prepare data dictionary for the plotting function
        plot_data = {
            'time': self.history['time'],
            'engine_temp': [t['engine'] for t in self.history['temps']],
            'coolant_temp': [t['coolant'] for t in self.history['temps']],
            'oil_temp': [t['oil'] for t in self.history['temps']],
            # Include limits if available in config
            'thermal_limits': {
                'engine_warning': self.config.warning_temp_engine,
                'engine_critical': self.config.critical_temp_engine,
                'coolant_warning': self.config.warning_temp_coolant,
                'coolant_critical': self.config.critical_temp_coolant
            }
            # Add ambient temp if needed for plot context (assuming constant for now)
            # 'ambient_temp': [ambient_temp_used] * len(self.history['time'])
        }

        fig = plot_thermal_performance(plot_data, title="Engine Thermal Simulation")
        if save_path and fig: save_plot(fig, save_path)
        elif fig: plt.show()
        if fig: plt.close(fig)

    def plot_heat_flow(self, save_path: Optional[str] = None):
        """Plot heat flow rates over time."""
        if not self.history['time']:
             logger.error("No simulation history to plot.")
             return

        from ..utils.plotting import save_plot # Local import

        time = self.history['time']
        fig, ax = plt.subplots(figsize=(12, 7))

        # Plot heat generation/rejection rates
        ax.plot(time, [h['gen_total']/1000 for h in self.history['heat_flows']], label='Total Waste Heat Gen (kW)', color='black', linestyle=':')
        ax.plot(time, [h['gen_coolant']/1000 for h in self.history['heat_flows']], label='Heat to Coolant (kW)', color='blue', alpha=0.7)
        ax.plot(time, [h['gen_oil']/1000 for h in self.history['heat_flows']], label='Heat to Oil (kW)', color='green', alpha=0.7)
        ax.plot(time, [h['rej_radiator']/1000 for h in self.history['heat_flows']], label='Heat Rejected by Radiator (kW)', color='red')
        ax.plot(time, [h['loss_block_amb']/1000 for h in self.history['heat_flows']], label='Block Ambient Loss (kW)', color='grey', linestyle='--')

        _apply_common_ax_settings(ax, xlabel='Time (s)', ylabel='Heat Flow (kW)', title='Engine Heat Flows')
        ax.legend(loc='best')

        plt.tight_layout()
        if save_path: save_plot(fig, save_path)
        plt.show()
        plt.close(fig)

# --- Cooling Performance Analysis Class ---
# (This class uses the ThermalSimulation to analyze performance)
class CoolingPerformance:
    """Analyzes cooling system performance based on thermal simulations."""

    def __init__(self, engine_model: EngineHeatModel, cooling_system: CoolingSystemInterface):
        """Initialize with engine and cooling system models."""
        self.engine_model = engine_model
        self.cooling_system = cooling_system
        self.simulation = ThermalSimulation(engine_model, cooling_system)
        self.results = {} # Store analysis results

    def generate_steady_state_map(self, rpm_range: List[float], load_range: List[float],
                                ambient_temp: float = 25.0, vehicle_speed: float = 15.0,
                                coolant_flow_rate: float = 50.0) -> Dict:
        """
        Generate steady state temperature map over engine RPM and load.

        Args:
            rpm_range: List or array of engine speeds (RPM).
            load_range: List or array of engine loads (0-1 throttle equivalent).
            ambient_temp: Ambient temperature (°C).
            vehicle_speed: Vehicle speed (m/s).
            coolant_flow_rate: Coolant flow rate (L/min).

        Returns:
            Dictionary containing the steady state map results.
        """
        n_rpm = len(rpm_range)
        n_load = len(load_range)
        temp_engine_map = np.zeros((n_load, n_rpm))
        temp_coolant_map = np.zeros((n_load, n_rpm))
        temp_oil_map = np.zeros((n_load, n_rpm))
        time_map = np.zeros((n_load, n_rpm))

        logger.info(f"Generating steady state map ({n_rpm} RPMs x {n_load} loads)...")

        for j, load in enumerate(load_range):
            for i, rpm in enumerate(rpm_range):
                # Throttle approx = load
                throttle = load
                # Torque approx from engine model (can be refined)
                torque = self.engine_model.engine.get_torque(rpm, throttle) if self.engine_model.engine else 50.0 * load

                steady_state = self.simulation.run_steady_state(
                    rpm, throttle, ambient_temp, vehicle_speed, coolant_flow_rate
                )
                temp_engine_map[j, i] = steady_state['engine']
                temp_coolant_map[j, i] = steady_state['coolant']
                temp_oil_map[j, i] = steady_state['oil']
                time_map[j, i] = steady_state['time_to_steady']
                logger.debug(f"  RPM={rpm:.0f}, Load={load:.1f} -> Eng={steady_state['engine']:.1f}C")

        self.results['steady_state_map'] = {
            'rpms': np.array(rpm_range),
            'loads': np.array(load_range),
            'engine_temps': temp_engine_map,
            'coolant_temps': temp_coolant_map,
            'oil_temps': temp_oil_map,
            'time_to_steady': time_map,
            'conditions': {'ambient': ambient_temp, 'speed': vehicle_speed, 'flow': coolant_flow_rate}
        }
        logger.info("Steady state map generation complete.")
        return self.results['steady_state_map']

    def analyze_transient_response(self, step_change: Dict, duration: float = 300.0, dt: float = 0.1) -> Dict:
        """
        Analyze transient thermal response to a step change in operating conditions.

        Args:
            step_change: Dict defining the step, e.g.,
                         {'initial': {'rpm': 3000, 'throttle': 0.2},
                          'final': {'rpm': 10000, 'throttle': 0.8},
                          'common': {'ambient_temp': 25, 'vehicle_speed': 15, 'coolant_flow': 50}}
            duration: Simulation duration after the step (s).
            dt: Simulation time step (s).

        Returns:
            Dictionary with transient response data (time, temps).
        """
        initial = step_change['initial']
        final = step_change['final']
        common = step_change['common']

        logger.info(f"Analyzing transient response from {initial} to {final}...")

        # 1. Run to initial steady state
        init_steady = self.simulation.run_steady_state(
            initial['rpm'], initial['throttle'], common['ambient_temp'],
            common['vehicle_speed'], common['coolant_flow']
        )

        # 2. Run profile simulation after the step change
        self.simulation.reset(initial_temps=init_steady) # Start from steady state
        profile_time = np.arange(0, duration + dt, dt)
        profile_data = pd.DataFrame({
            'time': profile_time,
            'rpm': np.full_like(profile_time, final['rpm']),
            'throttle': np.full_like(profile_time, final['throttle']),
            'ambient_temp': np.full_like(profile_time, common['ambient_temp']),
            'vehicle_speed': np.full_like(profile_time, common['vehicle_speed']),
            'coolant_flow': np.full_like(profile_time, common['coolant_flow'])
        })
        transient_results_df = self.simulation.run_profile(profile_data, dt=dt)

        step_key = f"rpm{initial['rpm']:.0f}t{initial['throttle']:.1f}_to_rpm{final['rpm']:.0f}t{final['throttle']:.1f}"
        self.results[f'transient_{step_key}'] = transient_results_df
        logger.info("Transient response simulation complete.")
        return transient_results_df # Return the DataFrame

    def analyze_cooling_system_sizing(self, op_point: Dict, radiator_sizes: List[float],
                                    ambient_temps: List[float]) -> Dict:
        """
        Analyze required radiator sizing for a given operating point across ambient temps.

        Args:
            op_point: Dict defining the operating point {'rpm', 'throttle'}.
            radiator_sizes: List of radiator core areas (m^2) to test.
            ambient_temps: List of ambient temperatures (°C) to test.

        Returns:
            Dictionary with max temperatures for each size and ambient temp.
        """
        if not ExternalCoolingSystem:
             logger.error("ExternalCoolingSystem implementation not available for sizing analysis.")
             return {}

        original_radiator_area = self.cooling_system.radiator.core_area # Assuming access to radiator

        results = {'ambient_temps': ambient_temps, 'radiator_sizes': radiator_sizes}
        max_temps = np.zeros((len(ambient_temps), len(radiator_sizes)))

        logger.info("Analyzing cooling system sizing...")
        for j, size in enumerate(radiator_sizes):
            # Modify the cooling system's radiator area for this test
            self.cooling_system.radiator.core_area = size
            logger.debug(f" Testing radiator size: {size:.3f} m^2")
            for i, ambient in enumerate(ambient_temps):
                steady_state = self.simulation.run_steady_state(
                    op_point['rpm'], op_point['throttle'], ambient,
                    vehicle_speed=15.0, coolant_flow_rate=50.0 # Use representative speed/flow
                )
                max_temps[i, j] = steady_state['engine'] # Store max engine temp

        self.cooling_system.radiator.core_area = original_radiator_area # Restore original size
        results['max_engine_temps'] = max_temps
        self.results['sizing_analysis'] = results
        logger.info("Cooling sizing analysis complete.")
        return results

    # --- Plotting Wrappers ---
    def plot_steady_state_map(self, map_type: str = 'engine', save_path: Optional[str] = None):
        """Plot the generated steady state map."""
        if 'steady_state_map' not in self.results:
             logger.error("Steady state map not generated yet.")
             return

        from ..utils.plotting import plot_cooling_system_map, save_plot # Local import

        map_data = self.results['steady_state_map']
        plot_data = {
             'speeds': map_data['rpms'], # Use RPM for x-axis
             'engine_loads': map_data['loads'], # Use Load for y-axis
             'temperature_map': map_data[f'{map_type}_temps'],
             'ambient_temperature': map_data['conditions']['ambient']
             # Add limits if available in config
        }

        fig = plot_cooling_system_map(plot_data, title=f'Steady State {map_type.title()} Temperature Map')
        if save_path and fig: save_plot(fig, save_path)
        elif fig: plt.show()
        if fig: plt.close(fig)


    def plot_transient_response(self, step_key: str, save_path: Optional[str] = None):
        """Plot a specific transient response."""
        result_key = f'transient_{step_key}'
        if result_key not in self.results:
            logger.error(f"Transient response '{step_key}' not found.")
            return

        from ..utils.plotting import plot_thermal_performance, save_plot # Local import

        transient_df = self.results[result_key]
        # Convert DataFrame back to dict format expected by plot_thermal_performance
        plot_data = transient_df.to_dict(orient='list')

        fig = plot_thermal_performance(plot_data, title=f'Transient Response: {step_key}')
        if save_path and fig: save_plot(fig, save_path)
        elif fig: plt.show()
        if fig: plt.close(fig)

    def plot_cooling_sizing(self, save_path: Optional[str] = None):
        """Plot the results of the cooling system sizing analysis."""
        if 'sizing_analysis' not in self.results:
            logger.error("Sizing analysis not performed yet.")
            return

        from ..utils.plotting import save_plot # Local import

        sizing_data = self.results['sizing_analysis']
        ambient_temps = sizing_data['ambient_temps']
        radiator_sizes = sizing_data['radiator_sizes']
        max_engine_temps = sizing_data['max_engine_temps']

        fig, ax = plt.subplots(figsize=(10, 8))
        X, Y = np.meshgrid(ambient_temps, radiator_sizes)
        contour = ax.contourf(X, Y, max_engine_temps.T, levels=15, cmap=THERMAL_CMAP)
        cbar = plt.colorbar(contour)
        cbar.set_label('Max Engine Temperature (°C)')

        # Add contour lines
        contour_lines = ax.contour(X, Y, max_engine_temps.T, levels=[105, 115, 125], colors='black', linestyles=['--', '-', ':'])
        ax.clabel(contour_lines, inline=True, fontsize=8, fmt='%.0f C')

        _apply_common_ax_settings(ax, xlabel='Ambient Temperature (°C)', ylabel='Radiator Core Area (m^2)', title='Cooling System Sizing Analysis')

        plt.tight_layout()
        if save_path: save_plot(fig, save_path)
        plt.show()
        plt.close(fig)