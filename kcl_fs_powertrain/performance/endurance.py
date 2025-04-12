"""
Endurance simulation module for Formula Student powertrain.

Simulates the multi-lap endurance event, including thermal effects,
fuel consumption, driver changes, and basic reliability modeling.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Callable
import logging
import time
import random
from enum import Enum, auto

# Import core components
try:
    from ..core.vehicle import Vehicle
    from ..core.track_integration import TrackProfile
    from .lap_time import LapTimeSimulator # Need lap sim for individual laps
    from ..engine.fuel_systems import FuelSystem # For fuel tracking
    from ..utils.constants import (
        FS_MAX_ENDURANCE_POINTS, FS_MAX_EFFICIENCY_POINTS,
        GRAVITY, MS_TO_KMH, LITERS_TO_M3, FuelPropertiesConstants
    )
    from ..utils.plotting import plot_endurance_results as plot_endurance_unified
    from ..utils.plotting import plot_endurance_comparison as plot_endurance_comp_unified
    from ..utils.plotting import save_plot, _apply_common_ax_settings, COLOR_SCHEMES
except ImportError:
    # Fallbacks
    FS_MAX_ENDURANCE_POINTS = 275.0
    FS_MAX_EFFICIENCY_POINTS = 100.0
    GRAVITY=9.81; MS_TO_KMH=3.6; LITERS_TO_M3=0.001
    class Vehicle: pass
    class TrackProfile: pass
    class LapTimeSimulator: pass
    class FuelSystem: pass
    class FuelPropertiesConstants: pass # Mock
    def plot_endurance_unified(*args, **kwargs): plt.figure(); plt.plot([0,1]); plt.title("Fallback Plot"); plt.show(); plt.close(); return plt.gcf()
    def plot_endurance_comp_unified(*args, **kwargs): plt.figure(); plt.plot([0,1]); plt.title("Fallback Comparison Plot"); plt.show(); plt.close(); return plt.gcf()
    def save_plot(fig, path, **kwargs): pass
    def _apply_common_ax_settings(ax, **kwargs): pass
    COLOR_SCHEMES = {'default': plt.cm.tab10.colors}
    logger = logging.getLogger("Endurance_Fallback")
    logger.warning("Could not import all necessary modules. Using fallbacks.")


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("EnduranceSimulation")


class ReliabilityEvent(Enum):
    """Possible reliability events during endurance."""
    NONE = auto()
    ENGINE_OVERHEAT = auto()
    COOLING_FAILURE = auto() # e.g., hose leak, pump failure
    TRANSMISSION_JAM = auto()
    FUEL_PRESSURE_LOSS = auto()
    ELECTRICAL_FAULT = auto() # ECU, sensors, wiring
    SUSPENSION_DAMAGE = auto()
    TIRE_PUNCTURE = auto()
    # Add more specific failures as needed


class EnduranceSimulator:
    """Simulates the Formula Student endurance event."""

    def __init__(self, vehicle: Vehicle, lap_simulator: LapTimeSimulator):
        """
        Args:
            vehicle: Vehicle model instance.
            lap_simulator: Pre-configured LapTimeSimulator instance for the event track.
        """
        if not isinstance(vehicle, Vehicle):
             if "MockVehicle" not in str(type(vehicle)):
                 raise TypeError("vehicle must be an instance of the Vehicle class.")
        if not isinstance(lap_simulator, LapTimeSimulator):
             raise TypeError("lap_simulator must be an instance of LapTimeSimulator.")
        if lap_simulator.track_profile is None or lap_simulator.track_data is None:
             raise ValueError("LapTimeSimulator must have a track loaded.")

        self.vehicle = vehicle
        self.lap_simulator = lap_simulator # Use the passed-in simulator

        # --- Endurance Event Configuration ---
        self.num_laps: int = 22
        self.driver_change_lap: int = 11 # Lap *after* which change occurs (0 for none)
        self.driver_change_time_s: float = 120.0 # Estimated time loss for driver swap

        # --- Vehicle Endurance State ---
        # Use FuelSystem if available on vehicle, otherwise estimate
        self.fuel_system = getattr(vehicle, 'fuel_system', None)
        self.fuel_capacity_L = self.fuel_system.tank_capacity_L if self.fuel_system else 7.0
        self.current_fuel_L = self.fuel_capacity_L # Start full

        # Thermal state tracking
        self.current_thermal_state = {'engine_temp': 25.0, 'coolant_temp': 25.0, 'oil_temp': 25.0} # Start cold
        self.optimal_engine_temp = (85.0, 100.0) # Range for optimal performance
        self.critical_engine_temp = 118.0 # Temp triggering severe issues/DNF
        self.thermal_degradation_factor = 0.015 # % performance loss per degree C over optimal_high
        self.inter_lap_cooling_time_s = 15.0 # Assumed time for some cooling between laps

        # Reliability modeling
        self.base_reliability_per_lap: float = 0.995 # Probability of completing a lap without issues
        self.reliability_stress_factor: float = 0.01 # How much hard driving increases failure chance
        self.reliability_temp_factor: float = 0.008 # How much overheating increases failure chance
        self.component_wear: Dict[str, float] = {'engine': 0, 'transmission': 0, 'cooling': 0} # 0-1 wear scale
        self.wear_rate_per_lap: Dict[str, float] = {'engine': 0.002, 'transmission': 0.0015, 'cooling': 0.001}

        # Driver modeling
        self.driver_factors = {'driver1': 1.0, 'driver2': 0.98} # Performance multiplier

        # Simulation Results
        self.results: Dict = {}

        logger.info("EnduranceSimulator initialized.")

    def configure_event(self, num_laps: int = 22, driver_change_lap: int = 11,
                      driver_change_time_s: float = 120.0):
        """Configure event parameters."""
        self.num_laps = num_laps
        self.driver_change_lap = driver_change_lap
        self.driver_change_time_s = driver_change_time_s
        logger.info(f"Event configured: {num_laps} laps, driver change after lap {driver_change_lap}, {driver_change_time_s}s change time.")

    def configure_vehicle_state(self, initial_fuel_L: Optional[float] = None,
                             initial_temps: Optional[Dict] = None):
        """Configure initial vehicle state."""
        self.current_fuel_L = initial_fuel_L if initial_fuel_L is not None else self.fuel_capacity_L
        if initial_temps:
             self.current_thermal_state = initial_temps
        logger.info(f"Initial vehicle state: Fuel={self.current_fuel_L:.2f}L, Temps={self.current_thermal_state}")

    def configure_reliability(self, base_reliability: float = 0.995,
                            stress_factor: float = 0.01, temp_factor: float = 0.008,
                            wear_rates: Optional[Dict] = None):
        """Configure reliability parameters."""
        self.base_reliability_per_lap = base_reliability
        self.reliability_stress_factor = stress_factor
        self.reliability_temp_factor = temp_factor
        if wear_rates: self.wear_rate_per_lap.update(wear_rates)
        logger.info(f"Reliability configured: Base={base_reliability:.4f}")


    def _get_current_driver_factor(self, lap_num: int) -> float:
        """Get the performance factor for the current driver."""
        if self.driver_change_lap > 0 and lap_num > self.driver_change_lap:
            return self.driver_factors['driver2']
        else:
            return self.driver_factors['driver1']

    def _apply_performance_factors(self, driver_factor: float, thermal_factor: float, wear_factor: float):
        """Apply performance factors to the vehicle model (temporarily)."""
        # Store original values
        self._original_mass = self.vehicle.mass
        self._original_drag = self.vehicle.drag_coefficient
        # Store original engine thermal factor if exists
        self._original_thermal_factor = getattr(self.vehicle.engine, 'thermal_factor', 1.0)

        # Modify vehicle parameters based on factors
        # Simplified approach: Adjust mass for driver/wear, adjust engine factor for thermal
        # Lower factor = better performance -> Lower mass equivalent
        mass_multiplier = 1.0 / (driver_factor * wear_factor)
        # Increase drag slightly with wear (loose bodywork etc.)
        drag_multiplier = 1.0 + (1.0 - wear_factor) * 0.1

        self.vehicle.mass = self._original_mass * mass_multiplier
        self.vehicle.drag_coefficient = self._original_drag * drag_multiplier
        if hasattr(self.vehicle.engine, 'thermal_factor'):
             self.vehicle.engine.thermal_factor = thermal_factor
        # logger.debug(f" Applied factors: Driver={driver_factor:.2f}, Thermal={thermal_factor:.2f}, Wear={wear_factor:.3f} -> Mass={self.vehicle.mass:.1f}, Drag={self.vehicle.drag_coefficient:.3f}")


    def _restore_performance_factors(self):
        """Restore vehicle parameters to original values."""
        if hasattr(self, '_original_mass'): self.vehicle.mass = self._original_mass
        if hasattr(self, '_original_drag'): self.vehicle.drag_coefficient = self._original_drag
        if hasattr(self, '_original_thermal_factor') and hasattr(self.vehicle.engine, 'thermal_factor'):
            self.vehicle.engine.thermal_factor = self._original_thermal_factor


    def _calculate_thermal_performance_factor(self) -> float:
        """Calculate performance factor (0-1) based on current engine temp."""
        temp = self.current_thermal_state['engine_temp']
        opt_low, opt_high = self.optimal_engine_temp

        if temp < opt_low:
            # Cold penalty
            return max(0.8, 1.0 - 0.01 * (opt_low - temp))
        elif temp <= opt_high:
            # Optimal range
            return 1.0
        else:
            # Overheat penalty
            penalty = self.thermal_degradation_factor * (temp - opt_high)
            return max(0.5, 1.0 - penalty) # Limit penalty to 50% loss

    def _calculate_wear_performance_factor(self) -> float:
         """Calculate performance factor (0-1) based on average component wear."""
         avg_wear = np.mean(list(self.component_wear.values()))
         # Simple linear degradation: 10% loss at 100% wear
         return max(0.9, 1.0 - 0.1 * avg_wear)

    def _update_thermal_state_between_laps(self):
        """Simulate partial cooling during the short break between laps."""
        ambient = 25.0 # Assume ambient temp
        # Simple exponential decay towards ambient
        cooling_rate = 0.05 # Rate per second (tune this)
        decay_factor = np.exp(-cooling_rate * self.inter_lap_cooling_time_s)

        for key in ['engine_temp', 'coolant_temp', 'oil_temp']:
            current_temp = self.current_thermal_state[key]
            cooled_temp = ambient + (current_temp - ambient) * decay_factor
            self.current_thermal_state[key] = cooled_temp

    def _update_component_wear(self, lap_time_s: float):
         """Update wear based on lap time and current wear state."""
         stress_factor = max(1.0, (90.0 / lap_time_s)**0.5) # Faster laps = more stress
         temp_factor = max(1.0, 1.0 + (self.current_thermal_state['engine_temp'] - self.optimal_engine_temp[1]) * 0.01) # Higher temp = more wear

         for component, base_rate in self.wear_rate_per_lap.items():
              # Wear increases with stress, temp, and existing wear (compounds)
              wear_increase = base_rate * stress_factor * temp_factor * (1 + self.component_wear[component])
              self.component_wear[component] = min(1.0, self.component_wear[component] + wear_increase) # Cap wear at 1.0


    def _check_reliability(self, lap_num: int) -> ReliabilityEvent:
        """Check for reliability issues on the current lap."""
        # Probability of failure increases with lap number, temperature, and wear
        base_fail_prob = 1.0 - self.base_reliability_per_lap

        # Lap factor (increases chance over time)
        lap_factor = 1.0 + (lap_num / self.num_laps) * 0.5

        # Temp factor (increases chance above optimal)
        temp_deviation = max(0, self.current_thermal_state['engine_temp'] - self.optimal_engine_temp[1])
        temp_factor = 1.0 + temp_deviation * self.reliability_temp_factor

        # Wear factor (average wear)
        avg_wear = np.mean(list(self.component_wear.values()))
        wear_factor = 1.0 + avg_wear * 2.0 # Wear significantly increases chance

        # Calculate final failure probability for this lap
        fail_prob = base_fail_prob * lap_factor * temp_factor * wear_factor
        fail_prob = min(fail_prob, 0.2) # Cap max failure probability per lap

        if random.random() < fail_prob:
            # Failure occurred! Determine type (simplified)
            if temp_deviation > (self.critical_engine_temp - self.optimal_engine_temp[1]) and random.random() < 0.5:
                 return ReliabilityEvent.ENGINE_OVERHEAT
            elif self.component_wear['cooling'] > 0.5 and random.random() < 0.3:
                 return ReliabilityEvent.COOLING_FAILURE
            elif self.component_wear['transmission'] > 0.6 and random.random() < 0.4:
                 return ReliabilityEvent.TRANSMISSION_JAM
            else:
                 # Random other failure
                 return random.choice([
                      ReliabilityEvent.FUEL_PRESSURE_LOSS,
                      ReliabilityEvent.ELECTRICAL_FAULT,
                      ReliabilityEvent.SUSPENSION_DAMAGE,
                      ReliabilityEvent.TIRE_PUNCTURE # Assume this is race-ending
                 ])
        else:
             return ReliabilityEvent.NONE


    def simulate_endurance(self, include_thermal: bool = True) -> Dict:
        """Simulate the full endurance event."""
        self.lap_simulator.include_thermal = include_thermal # Sync lap sim setting
        self.results = { # Reset results
            'lap_times': [], 'thermal_states': [], 'fuel_consumption_laps': [],
            'reliability_events': [], 'component_wear': [], 'driver_log': []
        }
        self.current_fuel_L = self.fuel_capacity_L # Start full
        self.current_thermal_state = {'engine_temp': 25.0, 'coolant_temp': 25.0, 'oil_temp': 25.0}
        self.component_wear = {comp: 0.0 for comp in self.wear_rate_per_lap}
        total_time_s = 0.0
        current_driver = 'driver1'
        completed_laps = 0
        dnf_reason = None

        logger.info(f"--- Starting Endurance: {self.num_laps} laps ---")

        for lap in range(1, self.num_laps + 1):
            # Check for driver change
            if self.driver_change_lap > 0 and lap == self.driver_change_lap + 1:
                logger.info(f"Lap {lap}: Driver change initiated ({self.driver_change_time_s:.0f}s added).")
                total_time_s += self.driver_change_time_s
                current_driver = 'driver2'
                self.results['driver_log'].append({'lap': lap, 'driver': current_driver, 'event': 'Change Start'})

            # Get performance factors for this lap
            driver_factor = self._get_current_driver_factor(lap)
            thermal_factor = self._calculate_thermal_performance_factor() if include_thermal else 1.0
            wear_factor = self._calculate_wear_performance_factor()
            self.results['driver_log'].append({'lap': lap, 'driver': current_driver, 'event': 'Lap Start',
                                               'thermal_factor': thermal_factor, 'wear_factor': wear_factor})

            # Apply factors temporarily
            self._apply_performance_factors(driver_factor, thermal_factor, wear_factor)

            # Simulate lap
            try:
                lap_result = self.lap_simulator.simulate_lap(include_thermal=include_thermal)
                lap_time = lap_result.get('lap_time')

                if lap_time is None or 'error' in lap_result:
                    raise ValueError(f"Lap simulation failed: {lap_result.get('error', 'Unknown error')}")

            except Exception as e:
                logger.error(f"Error simulating lap {lap}: {e}", exc_info=True)
                dnf_reason = f"Lap Simulation Error ({lap})"
                self._restore_performance_factors() # Ensure cleanup
                break # End endurance

            # Restore vehicle parameters
            self._restore_performance_factors()

            # --- Post-Lap Updates ---
            total_time_s += lap_time
            self.results['lap_times'].append(lap_time)

            # Estimate fuel used (requires fuel model or estimate)
            fuel_used_L = 0.0
            if self.fuel_system:
                 # More accurate: Get consumption rate from lap results if available
                 # Placeholder: Estimate based on average power during lap
                 avg_power_kw = 40.0 * driver_factor * thermal_factor # Rough estimate
                 cons_analyzer = FuelConsumption(self.fuel_system.fuel_properties, self.vehicle.engine)
                 cons_rate_gs = cons_analyzer.calculate_fuel_mass_flow_g_s(avg_power_kw)
                 fuel_used_g = cons_rate_gs * lap_time
                 fuel_used_L = fuel_used_g / (self.fuel_system.fuel_properties.density_kg_per_L * 1000)
                 self.current_fuel_L = self.fuel_system.update_fuel_level(cons_rate_gs, lap_time)
            else:
                 # Basic estimation
                 fuel_used_L = 0.4 * driver_factor * (1.0 / max(0.5, thermal_factor)) # Base L/lap, adjust
                 self.current_fuel_L -= fuel_used_L
            self.results['fuel_consumption_laps'].append(fuel_used_L)

            logger.info(f" Lap {lap}: Time={lap_time:.3f}s | Fuel Used={fuel_used_L:.3f}L | Remaining={self.current_fuel_L:.2f}L")

            # Check fuel level
            if self.current_fuel_L <= 0.0:
                logger.error(f"DNF on lap {lap}: Out of fuel.")
                dnf_reason = "Out of Fuel"
                break

            # Update thermal state (end-of-lap temps from simulation)
            if include_thermal and 'engine_temp' in lap_result:
                self.current_thermal_state['engine_temp'] = lap_result['engine_temp'][-1]
                self.current_thermal_state['coolant_temp'] = lap_result['coolant_temp'][-1]
                self.current_thermal_state['oil_temp'] = lap_result.get('oil_temp', lap_result['engine_temp'][-1]-10)[-1] # Use estimate if not present
            elif include_thermal: # Estimate temp increase if lap sim didn't provide it
                 self.current_thermal_state['engine_temp'] += 3.0 # Simple increase per lap
                 self.current_thermal_state['coolant_temp'] += 2.5
                 self.current_thermal_state['oil_temp'] += 2.0
            self.results['thermal_states'].append(self.current_thermal_state.copy())

            # Update wear
            self._update_component_wear(lap_time)
            self.results['component_wear'].append(self.component_wear.copy())

            # Check reliability
            event = self._check_reliability(lap)
            self.results['reliability_events'].append(event)
            if event != ReliabilityEvent.NONE:
                logger.warning(f"Reliability Event on lap {lap}: {event.name}")
                # Check if event is race-ending
                if event in [ReliabilityEvent.COOLING_FAILURE, ReliabilityEvent.TRANSMISSION_JAM,
                             ReliabilityEvent.ELECTRICAL_FAULT, ReliabilityEvent.SUSPENSION_DAMAGE,
                             ReliabilityEvent.TIRE_PUNCTURE]:
                    logger.error(f"DNF on lap {lap} due to {event.name}")
                    dnf_reason = event.name
                    break
                elif event == ReliabilityEvent.ENGINE_OVERHEAT:
                     logger.warning("Engine overheating detected, performance may degrade further.")
                     # Could add logic here to force lower performance factor next lap

            # Simulate inter-lap cooling
            self._update_thermal_state_between_laps()
            completed_laps = lap # Update completed laps count

        # --- Final Results ---
        self.results['completed'] = (dnf_reason is None) and (completed_laps == self.num_laps)
        self.results['dnf_reason'] = dnf_reason
        self.results['completed_laps'] = completed_laps
        self.results['total_time_s'] = total_time_s
        self.results['total_fuel_L'] = sum(self.results['fuel_consumption_laps'])
        self.results['average_lap_s'] = np.mean(self.results['lap_times']) if self.results['lap_times'] else None
        self.results['fastest_lap_s'] = np.min(self.results['lap_times']) if self.results['lap_times'] else None
        self.results['final_fuel_L'] = self.current_fuel_L
        self.results['final_thermal_state'] = self.current_thermal_state
        self.results['final_component_wear'] = self.component_wear

        logger.info(f"--- Endurance Simulation Finished ---")
        logger.info(f" Status: {'Completed' if self.results['completed'] else 'DNF'}")
        if dnf_reason: logger.info(f" DNF Reason: {dnf_reason}")
        logger.info(f" Laps Completed: {completed_laps}/{self.num_laps}")
        logger.info(f" Total Time: {total_time_s:.2f} s")
        logger.info(f" Total Fuel Used: {self.results['total_fuel_L']:.2f} L")
        if self.results['average_lap_s']: logger.info(f" Average Lap: {self.results['average_lap_s']:.3f} s")

        return self.results


    def calculate_score(self, results: Dict, t_min: Optional[float] = None,
                      eff_factor_min: Optional[float] = None, eff_factor_max: Optional[float] = None) -> Dict:
        """
        Calculate Endurance and Efficiency scores based on FS rules (simplified).

        Args:
            results: Dictionary from simulate_endurance.
            t_min: The fastest endurance time of the event (seconds). If None, calculates score relative to self.
            eff_factor_min: The best (lowest) efficiency factor of the event. If None, calculates relative score.
            eff_factor_max: The maximum allowed efficiency factor. If None, uses rulebook estimate.

        Returns:
            Dictionary with calculated scores and status.
        """
        score_results = {
            'endurance_score': 0.0, 'efficiency_score': 0.0, 'total_score': 0.0,
            'status': 'DNF' if not results.get('completed', False) else 'Finished',
            'reason': results.get('dnf_reason'),
            'max_endurance_score': FS_MAX_ENDURANCE_POINTS,
            'max_efficiency_score': FS_MAX_EFFICIENCY_POINTS
        }

        if not results.get('completed', False):
            return score_results # DNF = 0 points for both

        # --- Endurance Score ---
        t_your = results['total_time_s']
        if t_min is None: t_min = t_your # Score relative to own time if no minimum provided
        t_max = t_min * 1.45 # Max time allowed based on rules

        if t_your > t_max:
            score_endurance = 25.0 # Minimum points for finishing > Tmax
        else:
            # Formula: 250 * ((Tmax / Tyour) - 1) / ((Tmax / Tmin) - 1) + 25
            denominator = (t_max / t_min) - 1
            if denominator > 1e-6:
                 score_endurance = 250.0 * ((t_max / t_your) - 1) / denominator + 25.0
            else: # Avoid division by zero if Tmax somehow equals Tmin
                 score_endurance = FS_MAX_ENDURANCE_POINTS if abs(t_your - t_min) < 1e-3 else 25.0

        score_results['endurance_score'] = np.clip(score_endurance, 0, FS_MAX_ENDURANCE_POINTS)

        # --- Efficiency Score ---
        # Efficiency Factor (EF) = (Tmin / Tyour) * (Fuel_min / Fuel_your)^2 -- Simplified typical logic
        # Or use CO2 / Laptime logic if available
        fuel_your_L = results['total_fuel_L']
        fuel_min_L = 3.0 # Estimate minimum possible fuel usage (needs data)
        eff_factor_your = (fuel_your_L / fuel_min_L)**2 * (t_your / t_min) if t_min > 0 else float('inf')

        if eff_factor_min is None: eff_factor_min = eff_factor_your # Score relative to self
        if eff_factor_max is None: eff_factor_max = eff_factor_min * 2.0 # Estimate max factor

        if eff_factor_your > eff_factor_max:
             score_efficiency = 0.0
        else:
             # Formula: 100 * (EF_max - EF_your) / (EF_max - EF_min)
             denominator = eff_factor_max - eff_factor_min
             if denominator > 1e-6:
                  score_efficiency = FS_MAX_EFFICIENCY_POINTS * (eff_factor_max - eff_factor_your) / denominator
             else: # Avoid division by zero
                  score_efficiency = FS_MAX_EFFICIENCY_POINTS if abs(eff_factor_your - eff_factor_min) < 1e-3 else 0.0

        score_results['efficiency_score'] = np.clip(score_efficiency, 0, FS_MAX_EFFICIENCY_POINTS)
        score_results['efficiency_factor'] = eff_factor_your # Store calculated factor

        # --- Total Score ---
        score_results['total_score'] = score_results['endurance_score'] + score_results['efficiency_score']
        score_results['status'] = 'Finished'

        return score_results


    # --- Plotting Wrappers ---
    def plot_lap_times(self, results: Optional[Dict] = None, save_path: Optional[str] = None):
        """Plot lap times using the unified plotting function."""
        if results is None: results = self.results
        if not results or 'lap_times' not in results:
             logger.error("No endurance results available to plot lap times.")
             return
        fig = plot_endurance_unified(results, plot_type='lap_times', save_path=save_path)
        # if fig: plt.close(fig)

    def plot_thermal_profile(self, results: Optional[Dict] = None, save_path: Optional[str] = None):
        """Plot thermal profile using the unified plotting function."""
        if results is None: results = self.results
        if not results or 'detailed_results' not in results or 'thermal_states' not in results['detailed_results']:
             logger.error("No detailed thermal results available to plot.")
             return
        # Prepare data for unified plotter
        plot_data = {
            'lap_times': results.get('lap_times', []), # Need lap numbers
            'engine_temps': [s['engine_temp'] for s in results['detailed_results']['thermal_states']],
            'coolant_temps': [s['coolant_temp'] for s in results['detailed_results']['thermal_states']],
            'oil_temps': [s.get('oil_temp') for s in results['detailed_results']['thermal_states']], # Handle missing oil temp
            'thermal_limits': { # Pass limits for plotting thresholds
                 'engine_warning': self.critical_engine_temp - 10, # Estimate
                 'engine_critical': self.critical_engine_temp
            }
        }
        fig = plot_endurance_unified(plot_data, plot_type='thermal', save_path=save_path)
        # if fig: plt.close(fig)

    def plot_fuel_consumption(self, results: Optional[Dict] = None, save_path: Optional[str] = None):
        """Plot fuel consumption using the unified plotting function."""
        if results is None: results = self.results
        if not results or 'detailed_results' not in results or 'fuel_consumption_laps' not in results['detailed_results']:
             logger.error("No detailed fuel consumption results available to plot.")
             return
        plot_data = {
             'lap_times': results.get('lap_times', []), # Need lap numbers
             'fuel_consumption': results['detailed_results']['fuel_consumption_laps'],
             'fuel_capacity': self.fuel_capacity_L
        }
        fig = plot_endurance_unified(plot_data, plot_type='fuel', save_path=save_path)
        # if fig: plt.close(fig)

    def plot_component_wear(self, results: Optional[Dict] = None, save_path: Optional[str] = None):
        """Plot component wear using the unified plotting function."""
        if results is None: results = self.results
        if not results or 'final_component_wear' not in results:
             logger.error("No final component wear data available to plot.")
             return
        plot_data = {'component_wear': results['final_component_wear']}
        fig = plot_endurance_unified(plot_data, plot_type='wear', save_path=save_path)
        # if fig: plt.close(fig)

    def generate_endurance_report(self, results: Dict, score: Dict, save_dir: Optional[str] = None):
        """Generate a summary report with plots."""
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            logger.info(f"Generating endurance report in: {save_dir}")
            # Save plots
            self.plot_lap_times(results, save_path=os.path.join(save_dir, "endurance_lap_times.png"))
            self.plot_thermal_profile(results, save_path=os.path.join(save_dir, "endurance_thermal.png"))
            self.plot_fuel_consumption(results, save_path=os.path.join(save_dir, "endurance_fuel.png"))
            self.plot_component_wear(results, save_path=os.path.join(save_dir, "endurance_wear.png"))

            # Save summary JSON
            report_summary = {
                'status': score.get('status', 'Unknown'),
                'dnf_reason': score.get('reason'),
                'laps_completed': results.get('completed_laps', 0),
                'total_time_s': results.get('total_time_s'),
                'average_lap_s': results.get('average_lap_s'),
                'fastest_lap_s': results.get('fastest_lap_s'),
                'total_fuel_L': results.get('total_fuel_L'),
                'endurance_score': score.get('endurance_score'),
                'efficiency_score': score.get('efficiency_score'),
                'total_score': score.get('total_score'),
                'reliability_events_count': len([e for e in results.get('reliability_events',[]) if e != ReliabilityEvent.NONE]),
                'max_wear_component': max(results.get('final_component_wear',{}), key=results.get('final_component_wear',{}).get) if results.get('final_component_wear') else None,
                'max_wear_percent': max(results.get('final_component_wear',{}).values())*100 if results.get('final_component_wear') else None
            }
            summary_path = os.path.join(save_dir, "endurance_summary.json")
            try:
                 with open(summary_path, 'w') as f:
                      json.dump(report_summary, f, indent=2)
                 logger.info(f"Endurance summary saved to {summary_path}")
            except Exception as e:
                 logger.error(f"Failed to save endurance summary JSON: {e}")

        else:
            # Just show plots if no save directory
            self.plot_lap_times(results)
            self.plot_thermal_profile(results)
            self.plot_fuel_consumption(results)
            self.plot_component_wear(results)


class EnduranceAnalysis:
    """Tools for analyzing endurance simulation results and comparing configurations."""
    def __init__(self):
        self.comparison_data: List[Dict] = [] # Store {'label': str, 'results': Dict, 'score': Dict}
        logger.info("EnduranceAnalysis tool initialized.")

    def add_simulation_result(self, label: str, results: Dict, score: Dict):
        """Add a simulation result set for comparison."""
        if not results or not score:
             logger.warning(f"Cannot add result for '{label}': Missing results or score data.")
             return
        self.comparison_data.append({'label': label, 'results': results, 'score': score})
        logger.info(f"Added endurance result for configuration: '{label}'")

    def compare_configurations(self, save_dir: Optional[str] = None) -> Optional[pd.DataFrame]:
        """Compare added configurations based on score, time, fuel."""
        if not self.comparison_data:
            logger.warning("No simulation results added for comparison.")
            return None

        summary_list = []
        for data in self.comparison_data:
            res = data['results']
            scr = data['score']
            summary_list.append({
                'Configuration': data['label'],
                'Status': scr.get('status', 'Unknown'),
                'Total Score': scr.get('total_score'),
                'Endurance Score': scr.get('endurance_score'),
                'Efficiency Score': scr.get('efficiency_score'),
                'Total Time (s)': res.get('total_time_s'),
                'Avg Lap (s)': res.get('average_lap_s'),
                'Best Lap (s)': res.get('fastest_lap_s'),
                'Total Fuel (L)': res.get('total_fuel_L'),
                'Laps Completed': res.get('completed_laps'),
                'DNF Reason': res.get('dnf_reason')
            })

        summary_df = pd.DataFrame(summary_list)
        summary_df = summary_df.sort_values(by='Total Score', ascending=False).reset_index(drop=True)

        logger.info("\n--- Endurance Configuration Comparison ---")
        print(summary_df.to_string(index=False, float_format='%.2f'))

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            # Save summary table
            csv_path = os.path.join(save_dir, "endurance_comparison_summary.csv")
            summary_df.to_csv(csv_path, index=False, float_format='%.3f')
            logger.info(f"Comparison summary saved to {csv_path}")
            # Generate comparison plot
            plot_path = os.path.join(save_dir, "endurance_comparison_plot.png")
            self._plot_comparison(summary_df, save_path=plot_path) # Use helper

        return summary_df

    def _plot_comparison(self, summary_df: pd.DataFrame, save_path: Optional[str] = None):
         """Helper to plot the comparison summary DataFrame."""
         plot_data = []
         for _, row in summary_df.iterrows():
              # Reconstruct dict format for plotter if needed, or adapt plotter
              # For now, pass the dataframe directly if plotter can handle it
              plot_data.append({
                   'label': row['Configuration'],
                   'score': {'total_score': row['Total Score'], 'endurance_score': row['Endurance Score'], 'efficiency_score': row['Efficiency Score']},
                   'results': {'total_time_s': row['Total Time (s)'], 'total_fuel_L': row['Total Fuel (L)'], 'average_lap_s': row['Avg Lap (s)']},
                   # Add dummy lap_times/reliability if needed by plotter structure
                   'lap_times': [row['Avg Lap (s)']] * int(row.get('Laps Completed',1)) if row['Avg Lap (s)'] else [],
                   'reliability_events': []
              })

         fig = plot_endurance_comp_unified(plot_data, save_path=save_path) # Use unified plotter
         # if fig: plt.close(fig)

    # optimize_vehicle_setup and plot_optimization_results could be added here,
    # similar to LapTimeAnalysis, but would require running the EnduranceSimulator
    # repeatedly within the optimization loop.


# --- Factory and Runner Functions ---
def create_endurance_simulator(vehicle: Vehicle, track_file: str, num_laps: int = 22) -> EnduranceSimulator:
    """Factory to create and configure an EnduranceSimulator."""
    try:
        # We need a LapTimeSimulator instance for the endurance sim
        from .lap_time import LapTimeSimulator # Local import
        lap_sim = LapTimeSimulator(vehicle, track_file=track_file)
        # Pre-calculate racing line/speed profile for the lap simulator if desired
        # lap_sim.calculate_racing_line()
        # lap_sim.calculate_speed_profile() # Might be recalculated each lap anyway

        endurance_sim = EnduranceSimulator(vehicle, lap_sim)
        endurance_sim.configure_event(num_laps=num_laps)
        # Configure initial fuel from vehicle's fuel system if possible
        if hasattr(vehicle, 'fuel_system') and vehicle.fuel_system:
             endurance_sim.configure_vehicle_state(initial_fuel_L=vehicle.fuel_system.current_fuel_level_L)

        return endurance_sim
    except Exception as e:
        logger.error(f"Failed to create EnduranceSimulator: {e}", exc_info=True)
        raise # Re-raise the exception

def run_endurance_simulation(vehicle: Vehicle, track_file: str,
                           output_dir: Optional[str] = None,
                           include_thermal: bool = True,
                           num_laps: int = 22,
                           t_min_ref: Optional[float] = None # Optional reference time for scoring
                           ) -> Dict:
    """High-level function to run endurance simulation and generate report."""
    try:
        simulator = create_endurance_simulator(vehicle, track_file, num_laps=num_laps)
        results = simulator.simulate_endurance(include_thermal=include_thermal)
        # Pass reference time to scoring if provided
        score = simulator.calculate_score(results, fastest_time=t_min_ref)

        # Combine results and score
        full_results = {'results': results, 'score': score, 'simulator_instance': simulator}

        if output_dir:
            simulator.generate_endurance_report(results, score, save_dir=output_dir)

        logger.info(f"Endurance simulation finished. Status: {score.get('status', 'Unknown')}, Score: {score.get('total_score', 0):.1f}")
        return full_results

    except Exception as e:
        logger.error(f"Error running endurance simulation: {e}", exc_info=True)
        return {'error': str(e), 'results': None, 'score': None}

# --- Optimization/Comparison Wrappers ---
# (These would call methods on EnduranceAnalysis)

def optimize_endurance_setup(*args, **kwargs):
     # Placeholder - Requires implementing EnduranceAnalysis.optimize_vehicle_setup
     logger.warning("Endurance setup optimization not fully implemented yet.")
     return {}

def compare_endurance_configurations(vehicle_configs: Dict[str, Vehicle], track_file: str,
                                   output_dir: Optional[str] = None) -> Dict:
     """Compare multiple vehicle configurations for endurance."""
     analyzer = EnduranceAnalysis()
     all_results = {}
     for label, vehicle in vehicle_configs.items():
          logger.info(f"--- Simulating Endurance for Config: {label} ---")
          sim_output = run_endurance_simulation(vehicle, track_file, include_thermal=True) # Run sim
          if 'error' not in sim_output:
               analyzer.add_simulation_result(label, sim_output['results'], sim_output['score'])
               all_results[label] = sim_output
          else:
               logger.error(f"Simulation failed for config {label}: {sim_output['error']}")

     # Perform and optionally save comparison
     comparison_df = analyzer.compare_configurations(save_dir=output_dir)
     all_results['comparison_summary'] = comparison_df.to_dict('records') if comparison_df is not None else None
     return all_results


# Example Usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        from ..core.vehicle import create_formula_student_vehicle
        from .lap_time import create_example_track
        import tempfile

        print("Endurance Simulation Demo")
        print("-" * 27)

        vehicle = create_formula_student_vehicle()
        # Make sure fuel system exists for fuel tracking
        if not hasattr(vehicle, 'fuel_system') or vehicle.fuel_system is None:
             from ..engine.fuel_systems import FuelSystem, FuelProperties, FuelPump, FuelInjector, FuelType
             props = FuelProperties(FuelType.E85)
             pump = FuelPump()
             injectors = [FuelInjector() for _ in range(vehicle.engine.cylinders)]
             vehicle.fuel_system = FuelSystem(props, pump, injectors)
             print("Added default FuelSystem to vehicle.")


        output_dir = tempfile.mkdtemp()
        track_file = os.path.join(output_dir, "endurance_demo_track.yaml")
        create_example_track(track_file, difficulty='medium')

        print(f"Output directory: {output_dir}")
        print(f"Track file: {track_file}")

        # Run endurance simulation
        endurance_output = run_endurance_simulation(vehicle, track_file, output_dir=output_dir, num_laps=5) # Shorter for demo

        if 'error' not in endurance_output:
             print("\n--- Simulation Summary ---")
             print(f" Status: {endurance_output['score']['status']}")
             print(f" Score: {endurance_output['score']['total_score']:.1f}")
             print(f" Time: {endurance_output['results']['total_time_s']:.2f}s")
             print(f" Fuel: {endurance_output['results']['total_fuel_L']:.2f}L")
             print(f" Avg Lap: {endurance_output['results']['average_lap_s']:.3f}s")
        else:
             print(f"Endurance simulation failed: {endurance_output['error']}")

    except ImportError as e:
        print(f"\nError: Could not import necessary modules ({e}). Run from project root or ensure package is installed.")
    except FileNotFoundError as e:
         print(f"\nError: Configuration file not found. {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()