"""
Endurance simulation module for Formula Student powertrain.

This module provides classes and functions for simulating, analyzing, and optimizing
the endurance performance of a Formula Student vehicle. It includes comprehensive
modeling of thermal buildup over multiple laps, fuel consumption tracking, and
reliability considerations for the endurance event, which is typically the highest-scoring
dynamic event in Formula Student competitions.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Callable
import logging
import time
import random
import types
import time
from enum import Enum, auto

# Import from other modules
from ..core.vehicle import Vehicle
from ..core.track_integration import TrackProfile
from .lap_time import LapTimeSimulator, CorneringPerformance
from ..engine.fuel_systems import FuelConsumption, FuelSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("Endurance_Simulation")

class ReliabilityEvent(Enum):
    """Enumeration of possible reliability events during endurance."""
    NONE = auto()                    # No reliability issues
    ENGINE_OVERHEATING = auto()      # Engine temperature too high
    COOLING_SYSTEM_FAILURE = auto()  # Cooling system failure
    TRANSMISSION_ISSUE = auto()      # Transmission or shifting issue
    FUEL_SYSTEM_ISSUE = auto()       # Fuel delivery issue
    ELECTRONICS_FAILURE = auto()     # Electronics or control system failure
    MECHANICAL_FAILURE = auto()      # Mechanical component failure
    DRIVER_ERROR = auto()            # Driver-related issue


class EnduranceSimulator:
    """
    Simulator for Formula Student endurance event.
    
    This class simulates a complete endurance event, including multiple laps,
    thermal buildup, fuel consumption, and reliability considerations.
    """
    
    def __init__(self, vehicle: Vehicle, track_profile: Optional[TrackProfile] = None, 
                track_file: Optional[str] = None, lap_simulator: Optional[LapTimeSimulator] = None):
        """
        Initialize the endurance simulator.
        
        Args:
            vehicle: Vehicle model to use for simulation
            track_profile: Optional TrackProfile object
            track_file: Optional path to track file
            lap_simulator: Optional LapTimeSimulator instance
        """
        self.vehicle = vehicle
        self.track_profile = track_profile
        
        # Create lap simulator if not provided
        if lap_simulator is None:
            self.lap_simulator = LapTimeSimulator(vehicle, track_profile, track_file)
        else:
            self.lap_simulator = lap_simulator
            # Update simulator with provided track if any
            if track_profile is not None:
                self.lap_simulator.track_profile = track_profile
            elif track_file is not None:
                self.lap_simulator.load_track(track_file)
        
        # If track_file provided but no track_profile, load it
        if track_profile is None and track_file is not None and self.lap_simulator.track_profile is None:
            self.lap_simulator.load_track(track_file)
        
        # Endurance configuration parameters
        self.num_laps = 22           # Standard Formula Student endurance laps
        self.driver_change_lap = 11  # Lap for driver change
        self.driver_change_time = 180.0  # Time for driver change in seconds
        
        # Thermal management configuration
        self.thermal_recovery_time = 30.0  # Time in seconds between laps for thermal recovery
        self.thermal_degradation_factor = 0.02  # Performance degradation per degree above optimal
        self.optimal_engine_temp = 90.0  # Optimal engine temperature (°C)
        self.critical_engine_temp = 115.0  # Critical engine temperature (°C)
        
        # Fuel consumption configuration
        self.fuel_capacity = 7.0  # Typical Formula Student fuel tank capacity (L)
        self.current_fuel_level = self.fuel_capacity  # Start with full tank
        
        # Reliability configuration
        self.base_reliability = 0.99  # Base reliability per lap (0-1)
        self.reliability_thermal_factor = 0.005  # Reliability reduction per degree above optimal
        self.component_wear_rates = {
            'engine': 0.001,         # Wear rate per lap
            'transmission': 0.0008,  # Wear rate per lap
            'cooling': 0.0005,       # Wear rate per lap
            'electronics': 0.0003    # Wear rate per lap
        }
        
        # Driver performance
        self.driver_performance = {
            'driver1': 1.0,          # First driver performance factor (1.0 = nominal)
            'driver2': 0.98          # Second driver performance factor (typically slightly lower)
        }
        
        # Simulation results storage
        self.lap_results = []
        self.lap_times = []
        self.thermal_states = []
        self.fuel_consumption = []
        self.reliability_events = []
        self.component_wear = {component: 0.0 for component in self.component_wear_rates}
        
        logger.info("Endurance simulator initialized")
    
    def configure_event(self, num_laps: int = 22, driver_change_lap: int = 11, 
                      driver_change_time: float = 180.0):
        """
        Configure the endurance event parameters.
        
        Args:
            num_laps: Total number of laps in the endurance event
            driver_change_lap: Lap number for driver change (0 for no change)
            driver_change_time: Time in seconds for driver change
        """
        self.num_laps = num_laps
        self.driver_change_lap = driver_change_lap
        self.driver_change_time = driver_change_time
        
        logger.info(f"Endurance event configured: {num_laps} laps, driver change at lap {driver_change_lap}")
    
    def configure_vehicle(self, fuel_capacity: float = 7.0, initial_fuel: Optional[float] = None):
        """
        Configure vehicle-specific parameters for endurance.
        
        Args:
            fuel_capacity: Fuel tank capacity in liters
            initial_fuel: Initial fuel level (defaults to full tank if None)
        """
        self.fuel_capacity = fuel_capacity
        self.current_fuel_level = initial_fuel if initial_fuel is not None else fuel_capacity
        
        logger.info(f"Vehicle configured: {fuel_capacity}L fuel capacity, {self.current_fuel_level}L initial fuel")
    
    def configure_reliability(self, base_reliability: float = 0.99, 
                            thermal_factor: float = 0.005, 
                            component_wear_rates: Optional[Dict[str, float]] = None):
        """
        Configure reliability parameters for the simulation.
        
        Args:
            base_reliability: Base reliability per lap (0-1)
            thermal_factor: Reliability reduction per degree above optimal
            component_wear_rates: Dictionary of component wear rates per lap
        """
        self.base_reliability = base_reliability
        self.reliability_thermal_factor = thermal_factor
        
        if component_wear_rates:
            self.component_wear_rates.update(component_wear_rates)
            
        # Reset component wear
        self.component_wear = {component: 0.0 for component in self.component_wear_rates}
        
        logger.info(f"Reliability configured: {base_reliability:.4f} base reliability")
    
    def _calculate_reliability_probability(self, lap: int, engine_temp: float) -> float:
        """
        Calculate the probability of a reliability issue on this lap.
        
        Args:
            lap: Current lap number
            engine_temp: Current engine temperature
        
        Returns:
            Probability of a reliability issue (0-1)
        """
        # Base reliability decreases with each lap
        lap_factor = 1.0 - (lap / (2 * self.num_laps))  # Gentle degradation
        
        # Calculate thermal factor
        if engine_temp > self.optimal_engine_temp:
            thermal_factor = (engine_temp - self.optimal_engine_temp) * self.reliability_thermal_factor
        else:
            thermal_factor = 0.0
        
        # Calculate component wear factor
        wear_factor = sum(self.component_wear.values()) / len(self.component_wear)
        
        # Combined reliability
        reliability = self.base_reliability * lap_factor * (1.0 - thermal_factor) * (1.0 - wear_factor)
        
        # Cap between 0 and 1
        reliability = max(0.0, min(1.0, reliability))
        
        # Return probability of an issue
        return 1.0 - reliability
    
    def _simulate_reliability_event(self, lap: int, engine_temp: float) -> ReliabilityEvent:
        """
        Simulate potential reliability events based on conditions.
        
        Args:
            lap: Current lap number
            engine_temp: Current engine temperature
        
        Returns:
            ReliabilityEvent enum indicating what occurred
        """
        # Calculate probability of an issue
        failure_probability = self._calculate_reliability_probability(lap, engine_temp)
        
        # Check if an issue occurs
        if random.random() < failure_probability:
            # Determine type of issue
            if engine_temp > self.critical_engine_temp:
                # High probability of overheating if above critical temp
                if random.random() < 0.8:
                    return ReliabilityEvent.ENGINE_OVERHEATING
            
            # Otherwise, randomly select an issue based on component wear
            weighted_issues = [
                (ReliabilityEvent.ENGINE_OVERHEATING, self.component_wear['engine'] * 10),
                (ReliabilityEvent.COOLING_SYSTEM_FAILURE, self.component_wear['cooling'] * 10),
                (ReliabilityEvent.TRANSMISSION_ISSUE, self.component_wear['transmission'] * 10),
                (ReliabilityEvent.ELECTRONICS_FAILURE, self.component_wear['electronics'] * 10),
                (ReliabilityEvent.FUEL_SYSTEM_ISSUE, 0.1 * failure_probability),
                (ReliabilityEvent.MECHANICAL_FAILURE, 0.2 * failure_probability),
                (ReliabilityEvent.DRIVER_ERROR, 0.05)  # Small constant chance
            ]
            
            # Calculate total weight
            total_weight = sum(weight for _, weight in weighted_issues)
            
            # Normalize weights
            normalized_issues = [(issue, weight / total_weight) for issue, weight in weighted_issues]
            
            # Cumulative distribution
            cdf = []
            cumulative = 0.0
            for issue, weight in normalized_issues:
                cumulative += weight
                cdf.append((issue, cumulative))
            
            # Select issue
            r = random.random()
            for issue, threshold in cdf:
                if r <= threshold:
                    return issue
            
            # Fallback
            return ReliabilityEvent.MECHANICAL_FAILURE
        
        return ReliabilityEvent.NONE
    
    def _update_component_wear(self, lap_time: float, engine_temp: float, reliability_event: ReliabilityEvent):
        """
        Update component wear based on lap conditions.
        
        Args:
            lap_time: Lap time in seconds
            engine_temp: Engine temperature
            reliability_event: Any reliability event that occurred
        """
        # Base wear from completing a lap
        for component, rate in self.component_wear_rates.items():
            # Increase wear based on lap time (longer lap = more wear)
            time_factor = lap_time / 90.0  # Normalized to a 90-second lap
            
            # Increase wear based on temperature
            if engine_temp > self.optimal_engine_temp:
                temp_factor = 1.0 + (engine_temp - self.optimal_engine_temp) * 0.02
            else:
                temp_factor = 1.0
            
            # Base wear update
            self.component_wear[component] += rate * time_factor * temp_factor
        
        # Additional wear from reliability events
        if reliability_event == ReliabilityEvent.ENGINE_OVERHEATING:
            self.component_wear['engine'] += 0.05
            self.component_wear['cooling'] += 0.03
        elif reliability_event == ReliabilityEvent.COOLING_SYSTEM_FAILURE:
            self.component_wear['cooling'] += 0.08
        elif reliability_event == ReliabilityEvent.TRANSMISSION_ISSUE:
            self.component_wear['transmission'] += 0.07
        elif reliability_event == ReliabilityEvent.ELECTRONICS_FAILURE:
            self.component_wear['electronics'] += 0.06
        elif reliability_event == ReliabilityEvent.FUEL_SYSTEM_ISSUE:
            # No direct component for fuel system in our model
            self.component_wear['engine'] += 0.02
    
    def _calculate_fuel_consumption(self, lap_result):
        """
        Calculate fuel consumption for a lap.
        
        Args:
            lap_result: Results from lap_simulator.simulate_lap()
            
        Returns:
            Fuel consumption in liters
        """
        # Initialize fuel consumption calculator if not present
        if not hasattr(self, '_base_fuel_consumption'):
            # Calculate a reasonable base consumption
            engine_displacement = getattr(self.vehicle.engine, 'displacement', 0.6)  # Default 600cc
            self._base_fuel_consumption = 0.3  # Base consumption in L/km
        
        # Get track length
        track_length = 0
        if hasattr(self.lap_simulator, 'track_profile') and hasattr(self.lap_simulator.track_profile, 'length'):
            track_length = self.lap_simulator.track_profile.length / 1000.0  # Convert to km
        elif 'distance' in lap_result and len(lap_result['distance']) > 0:
            track_length = lap_result['distance'][-1] / 1000.0  # Last distance point, convert to km
        else:
            track_length = 1.0  # Default 1km if unknown
        
        # Base consumption for this lap
        base_consumption = track_length * self._base_fuel_consumption
        
        # ---------- Apply variability factors ----------
        
        # 1. Engine temperature effect (hotter engine = more fuel)
        temp_factor = 1.0
        if hasattr(self.vehicle.engine, 'engine_temperature'):
            temp = self.vehicle.engine.engine_temperature
            # Efficiency decreases as temp increases above optimal (around 90°C)
            if temp > 90:
                temp_factor = 1.0 + (temp - 90) * 0.002  # +2% per 10°C above optimal
            elif temp < 80:
                # Cold engine is less efficient
                temp_factor = 1.0 + (80 - temp) * 0.001  # +1% per 10°C below optimal
        
        # 2. Driving aggression factor - varies by lap and previous events
        aggression_factor = 1.0
        if hasattr(self, 'reliability_events') and len(self.reliability_events) > 0:
            # Check if there was an event in the previous lap
            last_event = self.reliability_events[-1]
            if last_event != ReliabilityEvent.NONE:
                # Driver becomes more cautious after an event
                aggression_factor = 0.95  # 5% better fuel economy
        
        # 3. Lap variation - accounts for different lines, tire wear
        lap_count = len(getattr(self, 'lap_times', []))
        # Tire degradation increases consumption slightly each lap
        tire_factor = 1.0 + min(0.05, lap_count * 0.01)  # Up to +5% over time
        
        # 4. Random variation (±3%)
        random_factor = 0.97 + random.random() * 0.06
        
        # Calculate final consumption with all factors
        fuel_consumption = base_consumption * temp_factor * aggression_factor * tire_factor * random_factor
        
        return fuel_consumption

        
        # Fallback method with rough estimate based on track length and vehicle efficiency
        track_length = lap_results.get('distance', [])[-1] / 1000  # km
        
        # Typical consumption for a Formula Student car
        # Around 0.5-0.7 L/km for E85
        base_consumption = 0.6  # L/km
        
        # Adjust based on average speed (higher speed = higher consumption)
        avg_speed = track_length / (lap_results['time'][-1] / 3600)  # km/h
        speed_factor = 0.8 + 0.2 * (avg_speed / 60)  # Normalized to 60 km/h
        
        # Calculate consumption
        fuel_volume = track_length * base_consumption * speed_factor
        
        return fuel_volume
    
    def _update_thermal_state(self, lap_result, recovery_time):
        """Update thermal state after a lap and recovery period."""
        
        # Extract final temperatures from lap results if available
        if 'engine_temp' in lap_result and len(lap_result['engine_temp']) > 0:
            engine_temp = lap_result['engine_temp'][-1]
            coolant_temp = lap_result.get('coolant_temp', [90.0])[-1]
            oil_temp = lap_result.get('oil_temp', [85.0])[-1]
        else:
            # Estimate temperatures based on previous state
            if hasattr(self.vehicle.engine, 'engine_temperature'):
                engine_temp = self.vehicle.engine.engine_temperature
                # Some natural heating for longer simulations
                engine_temp += 5.0  # Each lap adds some heat
                engine_temp = min(130.0, engine_temp)  # Cap at reasonable max
            else:
                engine_temp = 90.0 + len(self.lap_times) * 2.0  # Progressive heating
                
            coolant_temp = engine_temp - 5.0  # Typically slightly lower than engine
            oil_temp = engine_temp - 10.0  # Oil typically runs cooler than block
        
        # Apply cooling during recovery
        ambient_temp = 25.0
        cooling_rate = 0.01  # Per second
        
        # Exponential cooling formula: T(t) = Tambient + (Tinitial - Tambient) * e^(-kt)
        engine_temp_after = ambient_temp + (engine_temp - ambient_temp) * np.exp(-cooling_rate * recovery_time)
        coolant_temp_after = ambient_temp + (coolant_temp - ambient_temp) * np.exp(-cooling_rate * recovery_time)
        oil_temp_after = ambient_temp + (oil_temp - ambient_temp) * np.exp(-cooling_rate * recovery_time)
        
        # Apply new temperature to engine
        if hasattr(self.vehicle.engine, 'engine_temperature'):
            self.vehicle.engine.engine_temperature = engine_temp_after
        if hasattr(self.vehicle.engine, 'coolant_temperature'):
            self.vehicle.engine.coolant_temperature = coolant_temp_after
        if hasattr(self.vehicle.engine, 'oil_temperature'):
            self.vehicle.engine.oil_temperature = oil_temp_after
        
        return {
            'engine_temp': engine_temp_after,
            'coolant_temp': coolant_temp_after,
            'oil_temp': oil_temp_after
        }
    
    def _apply_thermal_state(self, thermal_state: Dict):
        """
        Apply a thermal state to the vehicle before starting a lap.
        
        Args:
            thermal_state: Thermal state dictionary
        """
        # Apply thermal state to engine model if it supports it
        if hasattr(self.vehicle.engine, 'set_thermal_state'):
            self.vehicle.engine.set_thermal_state(
                thermal_state['engine_temp'],
                thermal_state['coolant_temp'],
                thermal_state['oil_temp']
            )
        elif hasattr(self.vehicle, 'engine_thermal_model'):
            # Try setting temperatures directly if thermal model is available
            thermal_model = self.vehicle.engine_thermal_model
            if hasattr(thermal_model, 'engine_temp'):
                thermal_model.engine_temp = thermal_state['engine_temp']
            if hasattr(thermal_model, 'coolant_temp'):
                thermal_model.coolant_temp = thermal_state['coolant_temp']
            if hasattr(thermal_model, 'oil_temp'):
                thermal_model.oil_temp = thermal_state['oil_temp']
    
    def _calculate_thermal_performance_impact(self, thermal_state: Dict) -> float:
        """
        Calculate performance impact from thermal conditions.
        
        Args:
            thermal_state: Thermal state dictionary
            
        Returns:
            Performance factor (0-1, where 1 is optimal)
        """
        engine_temp = thermal_state['engine_temp']
        
        # Optimal range for engine temperature
        if engine_temp < self.optimal_engine_temp:
            # Cold engine - some performance loss
            temp_delta = self.optimal_engine_temp - engine_temp
            cold_factor = 1.0 - 0.005 * temp_delta
            return max(0.90, cold_factor)
        elif engine_temp > self.optimal_engine_temp:
            # Hot engine - performance degradation
            temp_delta = engine_temp - self.optimal_engine_temp
            hot_factor = 1.0 - self.thermal_degradation_factor * temp_delta
            return max(0.70, hot_factor)
        else:
            # Optimal temperature
            return 1.0
    
    def simulate_endurance(self, include_thermal: bool = True) -> Dict:
        """
        Simulate a complete endurance event.
        
        Args:
            include_thermal: Whether to include thermal effects
            
        Returns:
            Dictionary with complete simulation results
        """
        
        if not self.lap_simulator.track_profile:
            raise ValueError("No track loaded for simulation")
        
        # Reset simulation results
        self.lap_results = []
        self.lap_times = []
        self.thermal_states = []
        self.fuel_consumption = []
        self.reliability_events = []
        self.component_wear = {component: 0.0 for component in self.component_wear_rates}
        
        # Initial thermal state (ambient temperature)

        thermal_state = {
            'engine_temp': 25.0,  # Starting from cold
            'coolant_temp': 25.0,
            'oil_temp': 25.0
        }

        # Apply initial thermal state to engine if it has temperature properties
        if hasattr(self.vehicle.engine, 'engine_temperature'):
            self.vehicle.engine.engine_temperature = thermal_state['engine_temp']
        if hasattr(self.vehicle.engine, 'coolant_temperature'):
            self.vehicle.engine.coolant_temperature = thermal_state['coolant_temp']
        if hasattr(self.vehicle.engine, 'oil_temperature'):
            self.vehicle.engine.oil_temperature = thermal_state['oil_temp']
        
        # Initial fuel level
        remaining_fuel = self.current_fuel_level
        
        # Track total event time
        total_time = 0.0
        current_driver = 'driver1'
        
        # Perform driver briefing
        logger.info(f"Starting endurance simulation with {self.num_laps} laps")
        logger.info(f"Initial fuel: {remaining_fuel:.2f}L of {self.fuel_capacity:.2f}L capacity")
        
        # Store whether the simulation completed successfully
        completed = True
        dnf_reason = None
        
        # Simulate each lap
        for lap in range(1, self.num_laps + 1):
            logger.info(f"Simulating lap {lap}/{self.num_laps}...")
            
            # Driver change if needed
            if lap == self.driver_change_lap + 1 and self.driver_change_lap > 0:
                logger.info(f"Driver change after lap {lap-1}, taking {self.driver_change_time:.1f} seconds")
                total_time += self.driver_change_time
                current_driver = 'driver2'
            
            # Apply thermal state to vehicle
            if include_thermal:
                self._apply_thermal_state(thermal_state)
            
            # Calculate driver performance factor
            driver_factor = self.driver_performance[current_driver]
            
            # Calculate thermal performance impact
            thermal_factor = self._calculate_thermal_performance_impact(thermal_state)
            
            # Apply performance factors to vehicle
            original_mass = self.vehicle.mass
            original_power = None
            
            # Apply driver and thermal factors
            # For simplicity, we'll just adjust vehicle mass to simulate performance differences
            # (Increasing mass = slower lap times)
            performance_multiplier = 1.0 / (driver_factor * thermal_factor)
            self.vehicle.mass *= performance_multiplier
            
            # If engine has a power method that we can modify
            if hasattr(self.vehicle.engine, 'get_power'):
                self.vehicle.engine.thermal_factor = thermal_factor
                
            # Simulate the lap        
            try:
                lap_start_time = time.time()
                lap_result = self.lap_simulator.simulate_lap(include_thermal=include_thermal)
                lap_end_time = time.time()
                
                # Record lap time
                lap_time = lap_result['lap_time']
                self.lap_times.append(lap_time)
                total_time += lap_time
                
                logger.info(f"Lap {lap} completed in {lap_time:.2f}s (sim time: {lap_end_time-lap_start_time:.2f}s)")
                
                # Calculate fuel consumption
                fuel_used = self._calculate_fuel_consumption(lap_result)
                self.fuel_consumption.append(fuel_used)
                remaining_fuel -= fuel_used
                
                logger.info(f"Fuel used: {fuel_used:.2f}L, remaining: {remaining_fuel:.2f}L")
                
                # Check if out of fuel
                if remaining_fuel <= 0:
                    logger.warning(f"Vehicle out of fuel on lap {lap}")
                    completed = False
                    dnf_reason = "Out of fuel"
                    break
                
                # Update thermal state considering recovery between laps
                # Recovery time will be proportional to lap time (typically ~15-30% of lap time)
                recovery_factor = 0.2  # 20% of lap time
                recovery_time = lap_time * recovery_factor + 5.0  # Additional 5s minimum
                
                thermal_state = self._update_thermal_state(lap_result, recovery_time)
                self.thermal_states.append(thermal_state)
                
                logger.info(f"Engine temp after cooling: {thermal_state['engine_temp']:.1f}°C")
                
                # Check for reliability issues
                reliability_event = self._simulate_reliability_event(lap, thermal_state['engine_temp'])
                self.reliability_events.append(reliability_event)
                
                # Update component wear
                self._update_component_wear(lap_time, thermal_state['engine_temp'], reliability_event)
                
                # Check if reliability issue caused DNF
                if reliability_event != ReliabilityEvent.NONE:
                    logger.warning(f"Reliability issue on lap {lap}: {reliability_event.name}")
                    
                    # Severe issues cause DNF
                    if reliability_event in [
                        ReliabilityEvent.ENGINE_OVERHEATING,
                        ReliabilityEvent.COOLING_SYSTEM_FAILURE,
                        ReliabilityEvent.MECHANICAL_FAILURE
                    ]:
                        logger.error(f"Vehicle DNF on lap {lap} due to {reliability_event.name}")
                        completed = False
                        dnf_reason = reliability_event.name
                        break
                
                # Store lap results
                self.lap_results.append(lap_result)
                
            except Exception as e:
                logger.error(f"Error simulating lap {lap}: {str(e)}")
                completed = False
                dnf_reason = f"Simulation error: {str(e)}"
                break
            
            # Restore original vehicle parameters
            self.vehicle.mass = original_mass
            # Clean up thermal factor attribute
            if hasattr(self.vehicle.engine, 'thermal_factor'):
                delattr(self.vehicle.engine, 'thermal_factor')
        
        # Calculate event score
        if completed:
            logger.info(f"Endurance completed successfully in {total_time:.2f}s")
            logger.info(f"Total fuel used: {sum(self.fuel_consumption):.2f}L")
        else:
            logger.warning(f"Endurance not completed. Reason: {dnf_reason}")
        
        # Compile results
        results = {
            'completed': completed,
            'dnf_reason': dnf_reason,
            'total_time': total_time,
            'lap_times': self.lap_times,
            'average_lap': np.mean(self.lap_times) if self.lap_times else 0.0,
            'fastest_lap': min(self.lap_times) if self.lap_times else 0.0,
            'slowest_lap': max(self.lap_times) if self.lap_times else 0.0,
            'lap_count': len(self.lap_times),
            'total_fuel': sum(self.fuel_consumption),
            'fuel_efficiency': sum(self.fuel_consumption) / len(self.lap_times) if self.lap_times else 0.0,
            'final_thermal_state': thermal_state if self.thermal_states else None,
            'component_wear': self.component_wear,
            'reliability_events': [event.name for event in self.reliability_events],
            'detailed_results': {
                'lap_results': self.lap_results,
                'thermal_states': self.thermal_states,
                'fuel_consumption': self.fuel_consumption
            }
        }
        
        return results
    
    def calculate_score(self, results: Dict, fastest_time: Optional[float] = None) -> Dict:
        """
        Calculate Formula Student endurance and efficiency scores.
        
        Args:
            results: Results from simulate_endurance
            fastest_time: Optional fastest time from all teams (for scoring)
            
        Returns:
            Dictionary with calculated scores
        """
        # If the vehicle didn't finish, score is 0
        if not results['completed']:
            return {
                'endurance_score': 0.0,
                'efficiency_score': 0.0,
                'total_score': 0.0,
                'status': 'DNF',
                'reason': results['dnf_reason']
            }
        
        # Endurance scoring formula (actual formula may vary by competition)
        max_endurance_score = 275.0
        
        if fastest_time is None:
            # No reference time, assume this is the fastest
            endurance_score = max_endurance_score
        else:
            # Calculate score based on time relative to fastest
            # Typically: Score = max_score * (T_min / T_your) * (5.8 - 4.8 * (T_your / T_min))
            # Simplified version
            time_ratio = fastest_time / results['total_time']
            endurance_score = max_endurance_score * min(1.0, time_ratio**0.8)
        
        # Efficiency scoring formula
        max_efficiency_score = 75.0
        
        # Calculate efficiency factor (lower is better)
        # Typical formula uses: CO2 equivalent = fuel_used * CO2_factor / (T_min / T_your)
        # Simplified version
        if fastest_time is None:
            efficiency_score = max_efficiency_score
        else:
            fuel_used = results['total_fuel']
            time_factor = fastest_time / results['total_time']
            
            # Lower fuel use and faster time = better efficiency
            efficiency_factor = fuel_used / time_factor
            
            # Assume a baseline of 7L for a typical car
            baseline_efficiency = 7.0
            
            efficiency_score = max_efficiency_score * min(1.0, (baseline_efficiency / efficiency_factor)**0.5)
        
        # Cap scores within valid range
        endurance_score = max(0.0, min(max_endurance_score, endurance_score))
        efficiency_score = max(0.0, min(max_efficiency_score, efficiency_score))
        
        # Calculate total score
        total_score = endurance_score + efficiency_score
        
        return {
            'endurance_score': endurance_score,
            'efficiency_score': efficiency_score,
            'total_score': total_score,
            'max_endurance_score': max_endurance_score,
            'max_efficiency_score': max_efficiency_score,
            'status': 'Finished',
            'fastest_time_factor': 1.0 if fastest_time is None else (fastest_time / results['total_time'])
        }
    
    def plot_lap_times(self, results: Dict, save_path: Optional[str] = None):
        """
        Plot lap times for the endurance event.
        
        Args:
            results: Results from simulate_endurance
            save_path: Optional path to save the plot
        """
        lap_times = results['lap_times']
        if not lap_times:
            logger.warning("No lap times to plot")
            return
        
        plt.figure(figsize=(12, 6))
        
        # Plot lap times
        laps = range(1, len(lap_times) + 1)
        plt.bar(laps, lap_times, color='blue', alpha=0.7)
        plt.axhline(y=results['average_lap'], color='red', linestyle='--', label=f"Average: {results['average_lap']:.2f}s")
        
        # Add driver change indicator if applicable
        if self.driver_change_lap > 0 and self.driver_change_lap < len(lap_times):
            plt.axvline(x=self.driver_change_lap + 0.5, color='green', linestyle='-', linewidth=2, 
                       label='Driver Change')
            
            # Annotate driver sections
            plt.annotate('Driver 1', 
                       xy=(self.driver_change_lap/2, min(lap_times)),
                       xytext=(self.driver_change_lap/2, min(lap_times) - 5),
                       ha='center', fontsize=12)
            
            plt.annotate('Driver 2', 
                       xy=(self.driver_change_lap + (len(lap_times) - self.driver_change_lap)/2, min(lap_times)),
                       xytext=(self.driver_change_lap + (len(lap_times) - self.driver_change_lap)/2, min(lap_times) - 5),
                       ha='center', fontsize=12)
        
        # Add reliability events
        for i, event in enumerate(self.reliability_events):
            if event != ReliabilityEvent.NONE:
                plt.annotate(event.name.replace('_', ' '), 
                           xy=(i+1, lap_times[i]),
                           xytext=(i+1, lap_times[i] + 2),
                           ha='center', va='bottom', 
                           arrowprops=dict(arrowstyle='->', lw=1.5, color='red'),
                           fontsize=9, color='red')
        
        # Add labels and title
        plt.xlabel('Lap Number')
        plt.ylabel('Lap Time (s)')
        plt.title('Endurance Lap Times')
        plt.grid(True, alpha=0.3)
        plt.xticks(laps)
        plt.legend()
        
        if results['completed']:
            status_text = f"Completed {len(lap_times)} laps in {results['total_time']:.1f}s"
        else:
            status_text = f"DNF after {len(lap_times)} laps. Reason: {results['dnf_reason']}"
        
        plt.figtext(0.5, 0.01, status_text, ha='center', fontsize=12, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_thermal_profile(self, results: Dict, save_path: Optional[str] = None):
        """
        Plot thermal profile throughout the endurance event.
        
        Args:
            results: Results from simulate_endurance
            save_path: Optional path to save the plot
        """
        thermal_states = results['detailed_results']['thermal_states']
        if not thermal_states:
            logger.warning("No thermal data to plot")
            return
        
        plt.figure(figsize=(12, 6))
        
        # Extract temperatures
        laps = range(1, len(thermal_states) + 1)
        engine_temps = [state['engine_temp'] for state in thermal_states]
        coolant_temps = [state['coolant_temp'] for state in thermal_states]
        oil_temps = [state['oil_temp'] for state in thermal_states]
        
        # Plot temperatures
        plt.plot(laps, engine_temps, 'r-', marker='o', label='Engine Temperature')
        plt.plot(laps, coolant_temps, 'b-', marker='s', label='Coolant Temperature')
        plt.plot(laps, oil_temps, 'g-', marker='^', label='Oil Temperature')
        
        # Add warning thresholds
        plt.axhline(y=self.optimal_engine_temp, color='gray', linestyle='--', alpha=0.7, 
                   label=f'Optimal ({self.optimal_engine_temp}°C)')
        plt.axhline(y=self.critical_engine_temp, color='red', linestyle='--', alpha=0.7, 
                   label=f'Critical ({self.critical_engine_temp}°C)')
        
        # Add driver change indicator if applicable
        if self.driver_change_lap > 0 and self.driver_change_lap < len(thermal_states):
            plt.axvline(x=self.driver_change_lap + 0.5, color='green', linestyle='-', linewidth=2, 
                       label='Driver Change')
        
        # Add labels and title
        plt.xlabel('Lap Number')
        plt.ylabel('Temperature (°C)')
        plt.title('Thermal Profile During Endurance')
        plt.grid(True, alpha=0.3)
        plt.xticks(laps)
        plt.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_fuel_consumption(self, results: Dict, save_path: Optional[str] = None):
        """
        Plot fuel consumption throughout the endurance event.
        
        Args:
            results: Results from simulate_endurance
            save_path: Optional path to save the plot
        """
        fuel_consumption = results['detailed_results']['fuel_consumption']
        if not fuel_consumption:
            logger.warning("No fuel data to plot")
            return
        
        plt.figure(figsize=(12, 6))
        
        # Calculate cumulative consumption
        laps = range(1, len(fuel_consumption) + 1)
        cumulative_consumption = np.cumsum(fuel_consumption)
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot per-lap consumption
        ax1.bar(laps, fuel_consumption, color='green', alpha=0.7)
        ax1.set_xlabel('Lap Number')
        ax1.set_ylabel('Fuel Consumption (L)')
        ax1.set_title('Per-Lap Fuel Consumption')
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(laps)
        
        # Add average line
        avg_consumption = np.mean(fuel_consumption)
        ax1.axhline(y=avg_consumption, color='red', linestyle='--', 
                   label=f'Average: {avg_consumption:.2f}L/lap')
        ax1.legend()
        
        # Plot cumulative consumption
        ax2.plot(laps, cumulative_consumption, 'b-', marker='o', linewidth=2)
        ax2.set_xlabel('Lap Number')
        ax2.set_ylabel('Cumulative Fuel Consumption (L)')
        ax2.set_title('Cumulative Fuel Consumption')
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(laps)
        
        # Add capacity line
        ax2.axhline(y=self.fuel_capacity, color='red', linestyle='--', 
                   label=f'Capacity: {self.fuel_capacity:.2f}L')
        ax2.legend()
        
        # Add driver change indicator if applicable
        if self.driver_change_lap > 0 and self.driver_change_lap < len(fuel_consumption):
            ax1.axvline(x=self.driver_change_lap + 0.5, color='green', linestyle='-', linewidth=2, 
                       label='Driver Change')
            ax2.axvline(x=self.driver_change_lap + 0.5, color='green', linestyle='-', linewidth=2, 
                       label='Driver Change')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_component_wear(self, results: Dict, save_path: Optional[str] = None):
        """
        Plot component wear at the end of the endurance event.
        
        Args:
            results: Results from simulate_endurance
            save_path: Optional path to save the plot
        """
        component_wear = results['component_wear']
        if not component_wear:
            logger.warning("No component wear data to plot")
            return
        
        plt.figure(figsize=(10, 6))
        
        # Extract components and wear values
        components = list(component_wear.keys())
        wear_values = list(component_wear.values())
        
        # Convert wear to percentage
        wear_percentage = [w * 100 for w in wear_values]
        
        # Create color map based on wear level
        colors = []
        for w in wear_percentage:
            if w < 20:
                colors.append('green')
            elif w < 50:
                colors.append('yellow')
            else:
                colors.append('red')
        
        # Plot component wear
        bars = plt.bar(components, wear_percentage, color=colors)
        
        # Add wear values as text
        for bar, value in zip(bars, wear_percentage):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{value:.1f}%', ha='center', va='bottom')
        
        # Add failure threshold
        plt.axhline(y=100, color='red', linestyle='--', label='Failure Threshold')
        
        # Add labels and title
        plt.xlabel('Component')
        plt.ylabel('Wear Percentage (%)')
        plt.title('Component Wear After Endurance')
        plt.grid(True, alpha=0.3, axis='y')
        plt.ylim(0, 110)
        plt.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def generate_endurance_report(self, results: Dict, score: Dict, save_dir: Optional[str] = None):
        """
        Generate a comprehensive endurance report.
        
        Args:
            results: Results from simulate_endurance
            score: Scoring results from calculate_score
            save_dir: Directory to save report files
        """
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        # Generate plots
        if save_dir:
            self.plot_lap_times(results, os.path.join(save_dir, 'lap_times.png'))
            self.plot_thermal_profile(results, os.path.join(save_dir, 'thermal_profile.png'))
            self.plot_fuel_consumption(results, os.path.join(save_dir, 'fuel_consumption.png'))
            self.plot_component_wear(results, os.path.join(save_dir, 'component_wear.png'))
        else:
            self.plot_lap_times(results)
            self.plot_thermal_profile(results)
            self.plot_fuel_consumption(results)
            self.plot_component_wear(results)
        
        # Create summary report
        summary = {
            'event_summary': {
                'status': score['status'],
                'laps_completed': results['lap_count'],
                'total_time': results['total_time'],
                'total_fuel': results['total_fuel'],
                'fuel_efficiency': results['fuel_efficiency']
            },
            'lap_statistics': {
                'average_lap': results['average_lap'],
                'fastest_lap': results['fastest_lap'],
                'slowest_lap': results['slowest_lap'],
                'lap_times': results['lap_times']
            },
            'scores': {
                'endurance_score': score.get('endurance_score', 0.0),
                'efficiency_score': score.get('efficiency_score', 0.0),
                'total_score': score.get('total_score', 0.0),
                'max_possible': score.get('max_endurance_score', 0.0) + score.get('max_efficiency_score', 0.0)
            },

            'reliability': {
                'events': results['reliability_events'],
                'component_wear': results['component_wear']
            }
        }
        
        # Save summary as JSON
        if save_dir:
            import json
            with open(os.path.join(save_dir, 'endurance_summary.json'), 'w') as f:
                json.dump(summary, f, indent=2)
            
            # Save detailed results as CSV files
            if results['lap_times']:
                lap_data = {
                    'lap_number': list(range(1, len(results['lap_times']) + 1)),
                    'lap_time': results['lap_times'],
                    'fuel_used': results['detailed_results']['fuel_consumption'] if results['detailed_results']['fuel_consumption'] else [0] * len(results['lap_times'])
                }
                
                # Add thermal data if available
                if results['detailed_results']['thermal_states']:
                    lap_data['engine_temp'] = [state['engine_temp'] for state in results['detailed_results']['thermal_states']]
                    lap_data['coolant_temp'] = [state['coolant_temp'] for state in results['detailed_results']['thermal_states']]
                    lap_data['oil_temp'] = [state['oil_temp'] for state in results['detailed_results']['thermal_states']]
                
                # Add reliability events
                lap_data['reliability_event'] = results['reliability_events'] if results['reliability_events'] else ['NONE'] * len(results['lap_times'])
                
                # Create DataFrame and save
                df = pd.DataFrame(lap_data)
                df.to_csv(os.path.join(save_dir, 'lap_data.csv'), index=False)
        
        # Return summary
        return summary


class EnduranceAnalysis:
    """
    Analysis toolset for Formula Student endurance performance.
    
    This class provides tools for analyzing endurance performance data,
    comparing different vehicle configurations, and optimizing setup.
    """
    
    def __init__(self, simulator: EnduranceSimulator):
        """
        Initialize the endurance analysis tool.
        
        Args:
            simulator: EnduranceSimulator instance
        """
        self.simulator = simulator
        
        # Storage for comparison data
        self.comparison_data = []
        
        logger.info("Endurance analysis tool initialized")
    
    def add_vehicle_configuration(self, vehicle: Vehicle, config_name: str,
                                include_thermal: bool = True) -> Dict:
        """
        Add a vehicle configuration to the analysis set.
        
        Args:
            vehicle: Vehicle model
            config_name: Name of the configuration
            include_thermal: Whether to include thermal effects
            
        Returns:
            Results for this configuration
        """
        # Store original vehicle
        original_vehicle = self.simulator.vehicle
        
        # Set new vehicle
        self.simulator.vehicle = vehicle
        
        # Run simulation
        logger.info(f"Simulating configuration: {config_name}")
        results = self.simulator.simulate_endurance(include_thermal=include_thermal)
        
        # Calculate score
        score = self.simulator.calculate_score(results)
        
        # Store results
        config_data = {
            'name': config_name,
            'vehicle': vehicle,
            'results': results,
            'score': score
        }
        
        self.comparison_data.append(config_data)
        
        # Restore original vehicle
        self.simulator.vehicle = original_vehicle
        
        logger.info(f"Configuration {config_name} simulated: {score['total_score']:.1f} points")
        
        return results
    
    def compare_configurations(self, save_path: Optional[str] = None) -> Dict:
        """
        Compare all vehicle configurations that have been simulated.
        
        Args:
            save_path: Optional path to save the comparison plot
            
        Returns:
            Dictionary with comparison results
        """
        if not self.comparison_data:
            logger.warning("No configurations to compare")
            return {}
        
        # Extract key metrics
        names = [data['name'] for data in self.comparison_data]
        total_scores = [data['score']['total_score'] for data in self.comparison_data]
        endurance_scores = [data['score']['endurance_score'] for data in self.comparison_data]
        efficiency_scores = [data['score']['efficiency_score'] for data in self.comparison_data]
        
        # Sort configurations by total score
        sorted_indices = np.argsort(total_scores)[::-1]  # Descending order
        
        sorted_names = [names[i] for i in sorted_indices]
        sorted_total = [total_scores[i] for i in sorted_indices]
        sorted_endurance = [endurance_scores[i] for i in sorted_indices]
        sorted_efficiency = [efficiency_scores[i] for i in sorted_indices]
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        # Set width of bars
        bar_width = 0.35
        
        # Set positions of bars on x-axis
        r1 = np.arange(len(sorted_names))
        r2 = [x + bar_width for x in r1]
        
        # Create bars
        plt.bar(r1, sorted_endurance, width=bar_width, color='blue', alpha=0.7, label='Endurance Score')
        plt.bar(r2, sorted_efficiency, width=bar_width, color='green', alpha=0.7, label='Efficiency Score')
        
        # Add total score as text
        for i, score in enumerate(sorted_total):
            plt.text(r1[i] + bar_width/2, sorted_endurance[i] + sorted_efficiency[i] + 5, 
                   f'{score:.1f}', ha='center', va='bottom')
        
        # Add labels and legend
        plt.xlabel('Vehicle Configuration')
        plt.ylabel('Score')
        plt.title('Endurance Event Score Comparison')
        plt.xticks([r + bar_width/2 for r in range(len(sorted_names))], sorted_names, rotation=45, ha='right')
        plt.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        # Create comparison table
        comparison_table = []
        for i, idx in enumerate(sorted_indices):
            data = self.comparison_data[idx]
            
            # Extract key metrics
            total_time = data['results']['total_time'] if data['results']['completed'] else float('inf')
            lap_count = data['results']['lap_count']
            avg_lap = data['results']['average_lap']
            total_fuel = data['results']['total_fuel']
            status = data['score']['status']
            
            comparison_table.append({
                'rank': i + 1,
                'name': data['name'],
                'total_score': data['score']['total_score'],
                'endurance_score': data['score']['endurance_score'],
                'efficiency_score': data['score']['efficiency_score'],
                'total_time': total_time if total_time != float('inf') else 'DNF',
                'laps_completed': lap_count,
                'average_lap': avg_lap,
                'total_fuel': total_fuel,
                'fuel_per_lap': total_fuel / lap_count if lap_count > 0 else 0,
                'status': status
            })
        
        # Convert to DataFrame for easier handling
        if save_path:
            df = pd.DataFrame(comparison_table)
            csv_path = save_path.replace('.png', '.csv')
            df.to_csv(csv_path, index=False)
        
        # Return comparison data
        return {
            'table': comparison_table,
            'best_config': sorted_names[0] if sorted_names else None,
            'best_score': sorted_total[0] if sorted_total else None
        }
    
    def optimize_vehicle_setup(self, param_ranges: Dict[str, Tuple[float, float]],
                             iterations: int = 10, include_thermal: bool = True) -> Dict:
        """
        Optimize vehicle setup for best endurance performance.
        
        Args:
            param_ranges: Dictionary of parameter names to (min, max) ranges
            iterations: Number of optimization iterations to perform
            include_thermal: Whether to include thermal effects
            
        Returns:
            Dictionary with optimization results
        """
        # Store original vehicle
        original_vehicle = self.simulator.vehicle
        vehicle = original_vehicle
        
        # Extract baseline parameters
        baseline_params = {}
        for param_name in param_ranges:
            if hasattr(vehicle, param_name):
                baseline_params[param_name] = getattr(vehicle, param_name)
            elif '.' in param_name:
                # Handle nested attributes like 'engine.thermal_efficiency'
                parts = param_name.split('.')
                obj = vehicle
                for part in parts[:-1]:
                    if hasattr(obj, part):
                        obj = getattr(obj, part)
                    else:
                        logger.warning(f"Attribute path {param_name} not found")
                        break
                else:
                    if hasattr(obj, parts[-1]):
                        baseline_params[param_name] = getattr(obj, parts[-1])
        
        logger.info(f"Optimizing {len(param_ranges)} parameters over {iterations} iterations")
        
        # Run baseline simulation
        self.simulator.vehicle = vehicle
        baseline_results = self.simulator.simulate_endurance(include_thermal=include_thermal)
        baseline_score = self.simulator.calculate_score(baseline_results)
        best_score = baseline_score['total_score']
        best_params = baseline_params.copy()
        
        logger.info(f"Baseline score: {best_score:.1f}")
        
        # Store all iterations for analysis
        all_iterations = [{
            'params': baseline_params.copy(),
            'score': best_score,
            'results': baseline_results
        }]
        
        # Helper function to set parameters
        def set_params(params):
            for param_name, value in params.items():
                if hasattr(vehicle, param_name):
                    setattr(vehicle, param_name, value)
                elif '.' in param_name:
                    # Handle nested attributes
                    parts = param_name.split('.')
                    obj = vehicle
                    for part in parts[:-1]:
                        if hasattr(obj, part):
                            obj = getattr(obj, part)
                        else:
                            break
                    else:
                        if hasattr(obj, parts[-1]):
                            setattr(obj, parts[-1], value)
        
        # Simple coordinate descent optimization
        for iteration in range(iterations):
            improved = False
            
            # Try modifying each parameter
            for param_name, (min_val, max_val) in param_ranges.items():
                # Skip parameters not found in the vehicle
                if param_name not in baseline_params:
                    continue
                
                # Try both increasing and decreasing
                for direction in [-1, 1]:
                    # Copy current best parameters
                    test_params = best_params.copy()
                    
                    # Modify parameter
                    param_range = max_val - min_val
                    step_size = param_range * 0.1  # 10% step
                    
                    new_value = test_params[param_name] + direction * step_size
                    
                    # Clamp to valid range
                    new_value = max(min_val, min(max_val, new_value))
                    
                    # Skip if unchanged
                    if new_value == test_params[param_name]:
                        continue
                    
                    test_params[param_name] = new_value
                    
                    # Apply parameters
                    set_params(test_params)
                    
                    # Run simulation
                    results = self.simulator.simulate_endurance(include_thermal=include_thermal)
                    score = self.simulator.calculate_score(results)
                    
                    # Store iteration
                    all_iterations.append({
                        'params': test_params.copy(),
                        'score': score['total_score'],
                        'results': results
                    })
                    
                    logger.info(f"Iteration {iteration+1}, {param_name}={new_value:.4f}: score={score['total_score']:.1f}")
                    
                    # Update best if improved
                    if score['total_score'] > best_score:
                        best_score = score['total_score']
                        best_params = test_params.copy()
                        improved = True
                        
                        logger.info(f"Found improvement! New best score: {best_score:.1f}")
            
            # Early stop if no improvement
            if not improved:
                logger.info(f"No improvement in iteration {iteration+1}, stopping")
                break
        
        # Reset vehicle to original state
        self.simulator.vehicle = original_vehicle
        
        # Sort iterations by score
        all_iterations.sort(key=lambda x: x['score'], reverse=True)
        
        # Return optimization results
        optimization_results = {
            'baseline_params': baseline_params,
            'baseline_score': baseline_score['total_score'],
            'best_params': best_params,
            'best_score': best_score,
            'improvement': best_score - baseline_score['total_score'],
            'improvement_percentage': (best_score - baseline_score['total_score']) / baseline_score['total_score'] * 100 if baseline_score['total_score'] > 0 else 0,
            'iterations': all_iterations,
            'param_ranges': param_ranges
        }
        
        return optimization_results
    
    def plot_optimization_results(self, optimization_results: Dict, save_path: Optional[str] = None):
        """
        Plot results of parameter optimization.
        
        Args:
            optimization_results: Results from optimize_vehicle_setup
            save_path: Optional path to save the plot
        """
        iterations = optimization_results['iterations']
        if not iterations:
            logger.warning("No optimization iterations to plot")
            return
        
        # Extract data
        iteration_numbers = list(range(len(iterations)))
        scores = [iter_data['score'] for iter_data in iterations]
        best_scores = [max(scores[:i+1]) for i in range(len(scores))]
        
        # Create figure with multiple plots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot score progression
        ax1.plot(iteration_numbers, scores, 'b-', marker='o', alpha=0.5, label='Iteration Score')
        ax1.plot(iteration_numbers, best_scores, 'r-', linewidth=2, label='Best Score')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Total Score')
        ax1.set_title('Optimization Progress')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot parameter changes for top N parameters
        param_names = list(optimization_results['baseline_params'].keys())
        num_params = min(5, len(param_names))  # Show at most 5 parameters
        
        # Calculate parameter importance (correlation with score)
        param_importance = {}
        for param_name in param_names:
            param_values = [iter_data['params'].get(param_name, 0) for iter_data in iterations]
            
            # Normalize parameter values
            if max(param_values) > min(param_values):
                normalized_values = [(v - min(param_values)) / (max(param_values) - min(param_values)) 
                                   for v in param_values]
                
                # Calculate correlation with score
                if len(set(normalized_values)) > 1:  # Skip if all values are the same
                    correlation = np.corrcoef(normalized_values, scores)[0, 1]
                    param_importance[param_name] = abs(correlation)
        
        # Sort parameters by importance
        sorted_params = sorted(param_importance.items(), key=lambda x: x[1], reverse=True)
        top_params = [param for param, _ in sorted_params[:num_params]]
        
        # Plot parameter changes
        for param_name in top_params:
            param_values = [iter_data['params'].get(param_name, 0) for iter_data in iterations]
            
            # Normalize for plotting
            min_val = min(param_values)
            max_val = max(param_values)
            
            if max_val > min_val:
                normalized_values = [(v - min_val) / (max_val - min_val) for v in param_values]
                ax2.plot(iteration_numbers, normalized_values, marker='s', linewidth=2, 
                       label=f"{param_name} [{min_val:.2f}-{max_val:.2f}]")
        
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Normalized Parameter Value')
        ax2.set_title('Parameter Evolution')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Add summary text
        summary_text = (
            f"Baseline score: {optimization_results['baseline_score']:.1f}\n"
            f"Best score: {optimization_results['best_score']:.1f}\n"
            f"Improvement: {optimization_results['improvement']:.1f} "
            f"({optimization_results['improvement_percentage']:.1f}%)"
        )
        
        plt.figtext(0.5, 0.01, summary_text, ha='center', fontsize=12, 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


def create_endurance_simulator(vehicle: Vehicle, track_file: str, num_laps: int = 22) -> EnduranceSimulator:
    """
    Create and configure an endurance simulator.
    
    Args:
        vehicle: Vehicle model to simulate
        track_file: Path to track file
        num_laps: Number of laps in the endurance event
        
    Returns:
        Configured EnduranceSimulator
    """
    # Create lap simulator
    from .lap_time import create_lap_time_simulator
    lap_simulator = create_lap_time_simulator(vehicle, track_file)
    
    # Create endurance simulator
    simulator = EnduranceSimulator(vehicle, lap_simulator=lap_simulator)
    
    # Configure for specified number of laps
    simulator.configure_event(num_laps=num_laps)
    
    # Configure fuel capacity based on vehicle if available
    if hasattr(vehicle, 'fuel_system') and hasattr(vehicle.fuel_system, 'tank_capacity'):
        simulator.configure_vehicle(fuel_capacity=vehicle.fuel_system.tank_capacity)
    
    return simulator


def run_endurance_simulation(vehicle: Vehicle, track_file: str, 
                           output_dir: Optional[str] = None, 
                           include_thermal: bool = True) -> Dict:
    """
    Run a complete endurance simulation and generate a report.
    
    Args:
        vehicle: Vehicle model to simulate
        track_file: Path to track file
        output_dir: Optional directory to save results
        include_thermal: Whether to include thermal effects
        
    Returns:
        Dictionary with simulation results
    """
    # Create simulator
    simulator = create_endurance_simulator(vehicle, track_file)
    
    # Run simulation
    logger.info("Starting endurance simulation...")
    results = simulator.simulate_endurance(include_thermal=include_thermal)
    
    # Calculate score
    score = simulator.calculate_score(results)
    
    # Generate report if output directory specified
    if output_dir:
        logger.info(f"Generating endurance report in {output_dir}")
        simulator.generate_endurance_report(results, score, output_dir)
    else:
        # Just show plots
        simulator.plot_lap_times(results)
        simulator.plot_thermal_profile(results)
        simulator.plot_fuel_consumption(results)
        simulator.plot_component_wear(results)
    
    # Print summary
    logger.info("\nEndurance Simulation Results:")
    logger.info(f"  Status: {score['status']}")
    
    if results['completed']:
        logger.info(f"  Total Time: {results['total_time']:.2f}s")
        logger.info(f"  Average Lap: {results['average_lap']:.2f}s")
        logger.info(f"  Fastest Lap: {results['fastest_lap']:.2f}s")
        logger.info(f"  Total Fuel: {results['total_fuel']:.2f}L")
        logger.info(f"  Fuel Efficiency: {results['fuel_efficiency']:.2f}L/lap")
        logger.info(f"  Endurance Score: {score['endurance_score']:.1f}/{score['max_endurance_score']}")
        logger.info(f"  Efficiency Score: {score['efficiency_score']:.1f}/{score['max_efficiency_score']}")
        logger.info(f"  Total Score: {score['total_score']:.1f}/{score['max_endurance_score'] + score['max_efficiency_score']}")
    else:
        logger.info(f"  DNF Reason: {results['dnf_reason']}")
        logger.info(f"  Completed Laps: {len(results['lap_times'])}")
    
    # Return combined results
    return {
        'results': results,
        'score': score,
        'simulator': simulator
    }


def optimize_endurance_setup(vehicle: Vehicle, track_file: str, 
                           param_ranges: Dict[str, Tuple[float, float]], 
                           iterations: int = 10,
                           output_dir: Optional[str] = None) -> Dict:
    """
    Optimize vehicle setup for endurance performance.
    
    Args:
        vehicle: Vehicle model to optimize
        track_file: Path to track file
        param_ranges: Dictionary of parameter names to (min, max) ranges
        iterations: Number of optimization iterations
        output_dir: Optional directory to save results
        
    Returns:
        Dictionary with optimization results
    """
    # Create simulator
    simulator = create_endurance_simulator(vehicle, track_file)
    
    # Create analyzer
    analyzer = EnduranceAnalysis(simulator)
    
    # Run optimization
    logger.info(f"Optimizing vehicle setup with {len(param_ranges)} parameters over {iterations} iterations...")
    optimization_results = analyzer.optimize_vehicle_setup(param_ranges, iterations)
    
    # Plot results
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        analyzer.plot_optimization_results(optimization_results, 
                                         os.path.join(output_dir, "optimization_results.png"))
        
        # Save optimization results as JSON
        import json
        with open(os.path.join(output_dir, "optimization_results.json"), 'w') as f:
            # Convert some types for JSON compatibility
            json_results = {
                'baseline_params': {k: float(v) for k, v in optimization_results['baseline_params'].items()},
                'baseline_score': float(optimization_results['baseline_score']),
                'best_params': {k: float(v) for k, v in optimization_results['best_params'].items()},
                'best_score': float(optimization_results['best_score']),
                'improvement': float(optimization_results['improvement']),
                'improvement_percentage': float(optimization_results['improvement_percentage']),
                'param_ranges': {k: [float(v[0]), float(v[1])] for k, v in optimization_results['param_ranges'].items()}
            }
            json.dump(json_results, f, indent=2)
    else:
        analyzer.plot_optimization_results(optimization_results)
    
    # Print summary
    logger.info("\nOptimization Results:")
    logger.info(f"  Baseline Score: {optimization_results['baseline_score']:.1f}")
    logger.info(f"  Best Score: {optimization_results['best_score']:.1f}")
    logger.info(f"  Improvement: {optimization_results['improvement']:.1f} ({optimization_results['improvement_percentage']:.1f}%)")
    logger.info("\nParameter Changes:")
    
    # Print parameter changes
    for param, (baseline, best) in zip(
        optimization_results['baseline_params'].keys(),
        zip(optimization_results['baseline_params'].values(), optimization_results['best_params'].values())
    ):
        if baseline != best:
            change_pct = (best - baseline) / baseline * 100 if baseline != 0 else float('inf')
            logger.info(f"  {param}: {baseline:.4f} -> {best:.4f} ({change_pct:+.1f}%)")
    
    return optimization_results


def compare_endurance_configurations(vehicle_configs: Dict[str, Vehicle], track_file: str, 
                                   output_dir: Optional[str] = None) -> Dict:
    """
    Compare different vehicle configurations in endurance performance.
    
    Args:
        vehicle_configs: Dictionary mapping configuration names to vehicle models
        track_file: Path to track file
        output_dir: Optional directory to save results
        
    Returns:
        Dictionary with comparison results
    """
    if not vehicle_configs:
        logger.warning("No vehicle configurations provided")
        return {}
    
    # Use the first vehicle to create the simulator
    first_vehicle = next(iter(vehicle_configs.values()))
    simulator = create_endurance_simulator(first_vehicle, track_file)
    
    # Create analyzer
    analyzer = EnduranceAnalysis(simulator)
    
    # Run simulations for each configuration
    logger.info(f"Comparing {len(vehicle_configs)} vehicle configurations...")
    
    for name, vehicle in vehicle_configs.items():
        analyzer.add_vehicle_configuration(vehicle, name)
    
    # Run comparison
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        comparison_results = analyzer.compare_configurations(
            save_path=os.path.join(output_dir, "configuration_comparison.png")
        )
    else:
        comparison_results = analyzer.compare_configurations()
    
    # Print summary
    logger.info("\nConfiguration Comparison:")
    logger.info(f"  Best Configuration: {comparison_results['best_config']}")
    logger.info(f"  Best Score: {comparison_results['best_score']:.1f}")
    logger.info("\nRanking:")
    
    for i, config in enumerate(comparison_results['table']):
        logger.info(f"  {i+1}. {config['name']}: {config['total_score']:.1f} points")
    
    return comparison_results


# Example usage
if __name__ == "__main__":
    from ..core.vehicle import create_formula_student_vehicle
    import tempfile
    import os.path
    
    print("Formula Student Endurance Simulation Example")
    print("-------------------------------------------")
    
    # Create a Formula Student vehicle
    vehicle = create_formula_student_vehicle()
    
    # Create a temporary directory for outputs
    output_dir = tempfile.mkdtemp()
    print(f"Creating output directory: {output_dir}")
    
    # Create example track
    track_file = os.path.join(output_dir, "endurance_track.yaml")
    print("Using example track...")
    
    from .lap_time import create_example_track
    create_example_track(track_file, difficulty='medium')
    
    # Run a simple endurance simulation
    print("\nRunning endurance simulation...")
    results = run_endurance_simulation(
        vehicle, 
        track_file,
        output_dir=os.path.join(output_dir, "simulation_results")
    )
    
    # Create a modified vehicle for comparison
    modified_vehicle = create_formula_student_vehicle()
    modified_vehicle.mass -= 15  # 15kg lighter
    
    # Improve cooling performance
    if hasattr(modified_vehicle, 'cooling_system'):
        modified_vehicle.cooling_system.radiator.core_area *= 1.2  # 20% larger radiator
    
    # Create another variant with different fuel efficiency
    efficient_vehicle = create_formula_student_vehicle()
    if hasattr(efficient_vehicle.engine, 'thermal_efficiency'):
        efficient_vehicle.engine.thermal_efficiency *= 1.1  # 10% better efficiency
    
    # Compare configurations
    print("\nComparing vehicle configurations...")
    
    vehicle_configs = {
        "Baseline": vehicle,
        "Lightweight": modified_vehicle,
        "Efficient": efficient_vehicle
    }
    
    compare_endurance_configurations(
        vehicle_configs,
        track_file,
        output_dir=os.path.join(output_dir, "comparison_results")
    )
    
    # Run optimization (with reduced iterations for example)
    print("\nRunning setup optimization...")
    
    param_ranges = {
        'mass': (140.0, 230.0),  # Vehicle mass range (kg)
        'drag_coefficient': (0.7, 1.2),  # Aero drag coefficient range
    }
    
    # Add engine thermal efficiency if available
    if hasattr(vehicle.engine, 'thermal_efficiency'):
        param_ranges['engine.thermal_efficiency'] = (0.25, 0.35)
    
    # Add cooling system parameters if available
    if hasattr(vehicle, 'cooling_system') and hasattr(vehicle.cooling_system, 'radiator'):
        param_ranges['cooling_system.radiator.core_area'] = (
            vehicle.cooling_system.radiator.core_area * 0.8,
            vehicle.cooling_system.radiator.core_area * 1.5
        )
    
    optimize_endurance_setup(
        vehicle,
        track_file,
        param_ranges,
        iterations=3,  # Reduced for example
        output_dir=os.path.join(output_dir, "optimization_results")
    )
    
    print(f"\nResults saved to: {output_dir}")