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
from scipy.interpolate import interp1d


class MotorcycleEngine:
    """
    Models a motorcycle engine for Formula Student competitions.
    
    This class simulates the behavior of a Honda CBR600F4i engine, including
    torque/power generation, thermal characteristics, and fuel consumption.
    It can be parameterized from a YAML configuration file.
    """
    
    def __init__(self, config_path: Optional[str] = None, engine_params: Optional[Dict] = None):
        """
        Initialize the MotorcycleEngine with either a config file path or direct parameters.
        
        Args:
            config_path: Path to YAML configuration file
            engine_params: Dictionary of engine parameters (used if config_path is None)
        """
        # Engine specification defaults (Honda CBR600F4i)
        self.make = "Honda"
        self.model = "CBR600F4i"
        self.displacement = 599  # cc
        self.cylinders = 4
        self.configuration = "Inline"
        self.compression_ratio = 12.0
        self.bore = 67.0  # mm
        self.stroke = 42.5  # mm
        self.valves_per_cylinder = 4
        self.max_power = 110  # hp
        self.max_power_rpm = 12500
        self.max_torque = 65  # Nm
        self.max_torque_rpm = 10500
        self.redline = 14000  # rpm
        self.idle_rpm = 1300
        self.weight = 57  # kg (dry)
        
        # Engine state variables
        self.current_rpm = self.idle_rpm
        self.throttle_position = 0.0  # 0.0 to 1.0
        self.engine_temperature = 20.0  # °C
        self.oil_temperature = 20.0  # °C
        self.coolant_temperature = 20.0  # °C
        self.target_coolant_temp = 85.0  # °C
        self.fuel_consumption_rate = 0.0  # g/s
        
        # Performance maps
        self.rpm_range = np.arange(1000, self.redline + 1, 100)
        self.torque_curve = None
        self.power_curve = None
        self.volumetric_efficiency_map = None
        self.thermal_efficiency_map = None
        self.torque_function = None
        
        # Load configuration if provided
        if config_path:
            self.load_config(config_path)
        elif engine_params:
            self.set_parameters(engine_params)
        
        # Generate performance curves
        self.generate_performance_curves()
        
    def load_config(self, config_path: str):
        """
        Load engine configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Engine configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.set_parameters(config)
        
    def set_parameters(self, params: Dict):
        """
        Set engine parameters from dictionary.
        
        Args:
            params: Dictionary of engine parameters
        """
        # Map configuration parameters to class attributes
        for key, value in params.items():
            if key == 'displacement_cc':
                self.displacement = value
            elif key == 'max_power_hp':
                self.max_power = value
            elif key == 'max_power_rpm':
                self.max_power_rpm = value
            elif key == 'max_torque_nm':
                self.max_torque = value
            elif key == 'max_torque_rpm':
                self.max_torque_rpm = value
            elif key == 'redline_rpm':
                self.redline = value
            elif key == 'idle_rpm':
                self.idle_rpm = value
            elif key == 'dry_weight_kg':
                self.weight = value
            elif key == 'bore_mm':
                self.bore = value
            elif key == 'stroke_mm':
                self.stroke = value
            elif key == 'compression_ratio':
                self.compression_ratio = value
            elif key == 'valves_per_cylinder':
                self.valves_per_cylinder = value
            elif key == 'make':
                self.make = value
            elif key == 'model':
                self.model = value
            elif key == 'cylinders':
                self.cylinders = value
            elif key == 'configuration':
                self.configuration = value
            # Additional FS-specific modifications could be handled here
            elif key == 'fs_modifications' and isinstance(value, dict):
                # Apply Formula Student modifications if specified
                if 'weight_reduction_kg' in value:
                    self.weight -= value['weight_reduction_kg']
        
        # After setting parameters, regenerate the performance curves
        self.generate_performance_curves()
    
    def generate_performance_curves(self):
        """
        Generate torque and power curves based on engine parameters.
        
        This uses a physics-based approach to model the torque curve of the CBR600F4i
        engine, considering volumetric efficiency, mechanical losses, and thermal effects.
        """
        # Create RPM points
        self.rpm_range = np.arange(1000, self.redline + 1, 100)
        
        # Generate realistic torque curve for CBR600F4i
        # Base torque curve shape for inline-4 motorcycle engines
        # The curve is modeled as a combination of physical factors
        
        # Volumetric efficiency model (varies with RPM)
        vol_eff = self._volumetric_efficiency_model(self.rpm_range)
        
        # Mechanical efficiency decreases with RPM due to friction
        mech_eff = self._mechanical_efficiency_model(self.rpm_range)
        
        # Thermal efficiency varies with load and RPM
        thermal_eff = self._thermal_efficiency_model(self.rpm_range)
        
        # Theoretical torque calculation
        theoretical_torque = (self.displacement / 1000) * self.compression_ratio * 20  # Baseline
        
        # Calculate torque considering all efficiencies
        torque = theoretical_torque * vol_eff * mech_eff * thermal_eff
        
        # Scale to match specified max torque
        torque = torque * (self.max_torque / np.max(torque))
        
        # Store the torque curve
        self.torque_curve = torque
        
        # Calculate power from torque (P = T * ω)
        # Power in kW = Torque in Nm * RPM * 2π / 60 / 1000
        power_kw = torque * self.rpm_range * 2 * np.pi / 60 / 1000
        
        # Convert to hp (1 kW = 1.34102 hp)
        power_hp = power_kw * 1.34102
        
        # Store the power curve
        self.power_curve = power_hp
        
        # Create interpolation functions for torque and power
        self.torque_function = interp1d(
            self.rpm_range, torque, 
            kind='cubic', 
            bounds_error=False, 
            fill_value=(torque[0], torque[-1])
        )
    
    def _volumetric_efficiency_model(self, rpm: np.ndarray) -> np.ndarray:
        """
        Model the volumetric efficiency as a function of RPM.
        
        Args:
            rpm: Array of RPM points
        
        Returns:
            Array of volumetric efficiency values (0.0 to 1.0)
        """
        # For CBR600F4i with DOHC, 16 valves, and factory tuning:
        # Volumetric efficiency increases with RPM, peaks around max torque RPM,
        # then gradually decreases
        
        # Normalize RPM to 0-1 range
        norm_rpm = (rpm - self.idle_rpm) / (self.redline - self.idle_rpm)
        
        # Create base curve with peak at max torque RPM
        max_torque_norm = (self.max_torque_rpm - self.idle_rpm) / (self.redline - self.idle_rpm)
        
        # Use a combination of polynomial and exponential decay for realistic curve
        vol_eff = 0.75 + 0.2 * np.exp(-10 * ((norm_rpm - max_torque_norm) ** 2))
        
        # Add resonance effects for more realistic engine behavior (intake resonance)
        resonance1 = 0.05 * np.sin(norm_rpm * 12 * np.pi) * np.exp(-4 * (norm_rpm - 0.4) ** 2)
        resonance2 = 0.03 * np.sin(norm_rpm * 8 * np.pi) * np.exp(-4 * (norm_rpm - 0.7) ** 2)
        
        vol_eff = vol_eff + resonance1 + resonance2
        
        # Ensure values stay in reasonable range
        vol_eff = np.clip(vol_eff, 0.65, 0.95)
        
        return vol_eff
    
    def _mechanical_efficiency_model(self, rpm: np.ndarray) -> np.ndarray:
        """
        Model the mechanical efficiency (friction losses) as a function of RPM.
        
        Args:
            rpm: Array of RPM points
        
        Returns:
            Array of mechanical efficiency values (0.0 to 1.0)
        """
        # Mechanical efficiency decreases with RPM due to friction
        # High at low RPM, decreases gradually, drops faster at high RPM
        
        # Normalize RPM to 0-1 range
        norm_rpm = (rpm - self.idle_rpm) / (self.redline - self.idle_rpm)
        
        # Calculate mechanical efficiency
        # Formula Student modifications typically reduce parasitic losses
        mech_eff = 0.92 - 0.12 * norm_rpm - 0.05 * norm_rpm ** 2
        
        return mech_eff
    
    def _thermal_efficiency_model(self, rpm: np.ndarray) -> np.ndarray:
        """
        Model the thermal efficiency as a function of RPM.
        
        Args:
            rpm: Array of RPM points
        
        Returns:
            Array of thermal efficiency values (0.0 to 1.0)
        """
        # Thermal efficiency tends to be best at moderate to high load
        # For CBR600F4i with 12:1 compression ratio
        
        # Normalize RPM to 0-1 range
        norm_rpm = (rpm - self.idle_rpm) / (self.redline - self.idle_rpm)
        
        # Thermal efficiency increases with load up to a point
        # Using Gaussian-like shape for peak efficiency
        thermal_eff = 0.30 + 0.05 * np.exp(-15 * (norm_rpm - 0.65) ** 2)
        
        return thermal_eff
    
    def get_torque(self, rpm: float, throttle: float = 1.0, engine_temp: float = 90.0) -> float:
        """
        Calculate engine torque at specified RPM and throttle position.
        
        Args:
            rpm: Engine speed in RPM
            throttle: Throttle position (0.0 to 1.0)
            engine_temp: Engine temperature in °C
            
        Returns:
            Torque in Nm
        """
        # Ensure RPM is within operating range
        rpm = max(self.idle_rpm, min(rpm, self.redline))
        
        # Get base torque from the curve
        base_torque = float(self.torque_function(rpm))
        
        # Apply throttle position (non-linear relationship)
        # At partial throttle, torque doesn't scale linearly
        throttle_factor = throttle ** 0.8  # Slight nonlinearity
        
        # Temperature effects
        # Too cold = reduced performance, optimal around 90°C, too hot = reduced
        temp_factor = 0.5 + 0.5 * np.exp(-0.001 * (engine_temp - 90) ** 2)
        temp_factor = min(1.0, temp_factor)  # Cap at 1.0
        
        # Apply all factors
        actual_torque = base_torque * throttle_factor * temp_factor
        
        return actual_torque
    
    def get_power(self, rpm: float, throttle: float = 1.0, engine_temp: float = 90.0) -> float:
        """
        Calculate engine power at specified RPM and throttle position.
        
        Args:
            rpm: Engine speed in RPM
            throttle: Throttle position (0.0 to 1.0)
            engine_temp: Engine temperature in °C
            
        Returns:
            Power in kW
        """
        # Get torque and convert to power
        torque = self.get_torque(rpm, throttle, engine_temp)
        
        # Power (kW) = Torque (Nm) * Angular velocity (rad/s) / 1000
        # Angular velocity = 2π * rpm / 60
        power_kw = torque * rpm * 2 * np.pi / 60 / 1000
        
        # Apply thermal factor if it exists
        if hasattr(self, 'thermal_factor'):
            power_kw *= self.thermal_factor
            
        return power_kw
    
    def get_fuel_consumption(self, rpm: float, throttle: float = 1.0) -> float:
        """
        Calculate fuel consumption rate at the given operating point.
        
        Args:
            rpm: Engine speed in RPM
            throttle: Throttle position (0.0 to 1.0)
            
        Returns:
            Fuel consumption in g/s
        """
        # Basic model for fuel consumption
        # Modified for E85 fuel as mentioned in the configuration
        
        # Get power output
        power_kw = self.get_power(rpm, throttle)
        
        # Base specific fuel consumption (g/kWh)
        # E85 fuel has approximately 30% higher consumption than gasoline
        base_bsfc = 320  # g/kWh for a well-tuned motorcycle engine on E85
        
        # BSFC varies with load and RPM
        # Best efficiency typically around 75-80% of max torque RPM
        normalized_rpm = (rpm - self.idle_rpm) / (self.redline - self.idle_rpm)
        rpm_factor = 1.0 + 0.2 * abs(normalized_rpm - 0.6)
        
        # Throttle (load) effect on BSFC
        # Partial throttle typically has higher BSFC
        load_factor = 1.0 + 0.3 * (1.0 - throttle) ** 0.5
        
        # Calculate actual BSFC
        actual_bsfc = base_bsfc * rpm_factor * load_factor
        
        # Convert g/kWh to g/s
        # g/s = g/kWh * kW / 3600
        fuel_consumption = actual_bsfc * power_kw / 3600
        
        return fuel_consumption
    
    def update_thermal_state(self, rpm: float, throttle: float, ambient_temp: float, 
                           cooling_effectiveness: float, dt: float) -> Dict:
        """
        Update engine thermal state based on current operation and cooling.
        
        Args:
            rpm: Engine speed in RPM
            throttle: Throttle position (0.0 to 1.0)
            ambient_temp: Ambient temperature in °C
            cooling_effectiveness: Cooling system effectiveness (0.0 to 1.0)
            dt: Time step in seconds
            
        Returns:
            Dictionary with updated temperatures
        """
        # Calculate heat generation
        # Heat is roughly proportional to fuel energy that doesn't become work
        power_kw = self.get_power(rpm, throttle, self.engine_temperature)
        fuel_power_kw = self.get_fuel_consumption(rpm, throttle) * 42.5 / 1000  # E85 has ~42.5 MJ/kg energy content
        
        # Thermal efficiency is power / fuel energy
        thermal_efficiency = power_kw / fuel_power_kw if fuel_power_kw > 0 else 0.3
        
        # Heat generated is the remaining energy
        heat_generated_kw = fuel_power_kw * (1 - thermal_efficiency)
        
        # Engine temperature rises with heat generated
        # And falls based on cooling system effectiveness
        heat_capacity_engine = 10.0  # kJ/°C (simplified model)
        
        # Temperature rise from generated heat
        temp_rise = heat_generated_kw * dt * 1000 / heat_capacity_engine  # in °C
        
        # Cooling effect (proportional to temperature difference and cooling effectiveness)
        cooling_power = 5.0 * cooling_effectiveness * (self.engine_temperature - ambient_temp) / 80
        temp_fall = cooling_power * dt * 1000 / heat_capacity_engine  # in °C
        
        # Net temperature change
        self.engine_temperature += temp_rise - temp_fall
        
        # Oil temperature follows engine temperature with some lag
        oil_lag_factor = 0.1
        self.oil_temperature += (self.engine_temperature - self.oil_temperature) * oil_lag_factor * dt
        
        # Coolant temperature is controlled by thermostat
        coolant_lag_factor = 0.2
        target_coolant = min(self.target_coolant_temp, self.engine_temperature - 5)
        self.coolant_temperature += (target_coolant - self.coolant_temperature) * coolant_lag_factor * dt
        
        return {
            'engine_temp': self.engine_temperature,
            'oil_temp': self.oil_temperature,
            'coolant_temp': self.coolant_temperature
        }
    
    def plot_performance_curves(self, save_path: Optional[str] = None):
        """
        Plot torque and power curves.
        
        Args:
            save_path: If provided, save the plot to this file path
        """
        plt.figure(figsize=(10, 6))
        
        # Plot torque
        ax1 = plt.gca()
        ax1.plot(self.rpm_range, self.torque_curve, 'b-', linewidth=2, label='Torque (Nm)')
        ax1.set_xlabel('Engine Speed (RPM)')
        ax1.set_ylabel('Torque (Nm)', color='b')
        ax1.tick_params(axis='y', colors='b')
        
        # Plot power on secondary y-axis
        ax2 = ax1.twinx()
        ax2.plot(self.rpm_range, self.power_curve, 'r-', linewidth=2, label='Power (hp)')
        ax2.set_ylabel('Power (hp)', color='r')
        ax2.tick_params(axis='y', colors='r')
        
        # Add vertical lines for key points
        plt.axvline(x=self.max_torque_rpm, color='b', linestyle='--', alpha=0.5, 
                   label=f'Max Torque: {self.max_torque} Nm @ {self.max_torque_rpm} RPM')
        plt.axvline(x=self.max_power_rpm, color='r', linestyle='--', alpha=0.5,
                   label=f'Max Power: {self.max_power} hp @ {self.max_power_rpm} RPM')
        
        # Add title and legend
        plt.title(f'{self.make} {self.model} Engine Performance')
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        # Set x-axis limits
        plt.xlim(1000, self.redline)
        
        # Grid for better readability
        plt.grid(True, alpha=0.3)
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        plt.tight_layout()
        plt.show()

    def get_engine_specs(self) -> Dict:
        """
        Get a dictionary of engine specifications.
        
        Returns:
            Dictionary with engine specifications
        """
        return {
            'make': self.make,
            'model': self.model,
            'displacement_cc': self.displacement,
            'cylinders': self.cylinders,
            'configuration': self.configuration,
            'compression_ratio': self.compression_ratio,
            'bore_mm': self.bore,
            'stroke_mm': self.stroke,
            'valves_per_cylinder': self.valves_per_cylinder,
            'max_power_hp': self.max_power,
            'max_power_rpm': self.max_power_rpm,
            'max_torque_nm': self.max_torque,
            'max_torque_rpm': self.max_torque_rpm,
            'redline_rpm': self.redline,
            'idle_rpm': self.idle_rpm,
            'weight_kg': self.weight
        }


# Example usage
if __name__ == "__main__":
    # Path to config file
    config_path = os.path.join("configs", "engine", "cbr600f4i.yaml")
    
    # Create engine
    engine = MotorcycleEngine(config_path=config_path)
    
    # Print engine specs
    print(f"Engine: {engine.make} {engine.model}")
    print(f"Displacement: {engine.displacement} cc")
    print(f"Max Power: {engine.max_power} hp @ {engine.max_power_rpm} RPM")
    print(f"Max Torque: {engine.max_torque} Nm @ {engine.max_torque_rpm} RPM")
    
    # Plot performance curves
    engine.plot_performance_curves()
    
    # Calculate torque and power at various RPMs
    test_rpms = [3000, 6000, 9000, 12000]
    for rpm in test_rpms:
        torque = engine.get_torque(rpm)
        power_kw = engine.get_power(rpm)
        power_hp = power_kw * 1.34102
        fuel_consumption = engine.get_fuel_consumption(rpm)
        print(f"At {rpm} RPM: Torque = {torque:.1f} Nm, Power = {power_hp:.1f} hp, Fuel = {fuel_consumption:.2f} g/s")