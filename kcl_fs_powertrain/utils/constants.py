"""
Constants module for Formula Student powertrain simulation.

This module provides physical constants, unit conversion factors, and reference values
used throughout the Formula Student powertrain simulation.
"""

import numpy as np
from enum import Enum, auto
from typing import Optional, List

# Physical constants
GRAVITY = 9.80665  # m/s², standard gravity
AIR_DENSITY_SEA_LEVEL = 1.225  # kg/m³, air density at sea level (15°C, 1013.25 hPa)
WATER_DENSITY = 999.97  # kg/m³, density of water at 4°C
AIR_VISCOSITY = 1.825e-5  # Pa·s or kg/(m·s), dynamic viscosity of air at 20°C
WATER_SPECIFIC_HEAT = 4186.0  # J/(kg·K), specific heat capacity of water at 20°C
AIR_SPECIFIC_HEAT_CP = 1005.0  # J/(kg·K), specific heat capacity of air at constant pressure
AIR_SPECIFIC_HEAT_CV = 718.0   # J/(kg·K), specific heat capacity of air at constant volume
AIR_GAS_CONSTANT = 287.058  # J/(kg·K), specific gas constant for dry air
STANDARD_PRESSURE = 101325.0  # Pa, standard atmospheric pressure
STEFAN_BOLTZMANN = 5.670374419e-8  # W/(m²·K⁴), Stefan-Boltzmann constant
ABSOLUTE_ZERO_C = -273.15  # Absolute zero in Celsius

# Unit conversion factors
KMH_TO_MS = 1.0 / 3.6  # Convert km/h to m/s
MS_TO_KMH = 3.6  # Convert m/s to km/h
MPH_TO_MS = 0.44704  # Convert mph to m/s
MS_TO_MPH = 2.23694  # Convert m/s to mph
KW_TO_HP = 1.34102209  # Convert kilowatts to horsepower (mechanical)
HP_TO_KW = 1 / KW_TO_HP  # Convert horsepower to kilowatts
NM_TO_LBFT = 0.73756215  # Convert N·m to lb·ft
LBFT_TO_NM = 1 / NM_TO_LBFT  # Convert lb·ft to N·m
KG_TO_LBS = 2.20462262  # Convert kg to pounds
LBS_TO_KG = 1 / KG_TO_LBS  # Convert pounds to kg
PSI_TO_PA = 6894.75729  # Convert psi to Pascal
PA_TO_PSI = 1 / PSI_TO_PA  # Convert Pascal to psi
BAR_TO_PA = 100000.0  # Convert bar to Pascal
PA_TO_BAR = 0.00001   # Convert Pascal to bar
INCH_TO_M = 0.0254  # Convert inches to meters
M_TO_INCH = 1 / INCH_TO_M  # Convert meters to inches
MM_TO_M = 0.001  # Convert mm to meters
M_TO_MM = 1000.0  # Convert meters to mm
M_TO_KM = 0.001  # Convert meters to kilometers
DEG_TO_RAD = np.pi / 180.0  # Convert degrees to radians
RAD_TO_DEG = 180.0 / np.pi  # Convert radians to degrees
LITERS_TO_M3 = 0.001  # Convert liters to cubic meters
M3_TO_LITERS = 1000.0  # Convert cubic meters to liters
GAL_TO_LITERS = 3.78541178  # Convert US gallons to liters
LITERS_TO_GAL = 1 / GAL_TO_LITERS  # Convert liters to US gallons
BAR_TO_PA = 100000.0  # Convert bar to Pascal
PA_TO_BAR = 1 / BAR_TO_PA   # Convert Pascal to bar

# Temperature conversions
def celsius_to_kelvin(temp_c: float) -> float:
    """Convert temperature from Celsius to Kelvin."""
    return temp_c - ABSOLUTE_ZERO_C # Use constant for precision

def kelvin_to_celsius(temp_k: float) -> float:
    """Convert temperature from Kelvin to Celsius."""
    return temp_k + ABSOLUTE_ZERO_C

def celsius_to_fahrenheit(temp_c: float) -> float:
    """Convert temperature from Celsius to Fahrenheit."""
    return temp_c * 9.0/5.0 + 32.0

def fahrenheit_to_celsius(temp_f: float) -> float:
    """Convert temperature from Fahrenheit to Celsius."""
    return (temp_f - 32.0) * 5.0/9.0

# Formula Student reference values (Based on typical rules/events)
FS_MAX_TRACK_WIDTH = 1.5  # m, maximum width constraint for some event elements
FS_ACCELERATION_LENGTH = 75.0  # m, standard acceleration event distance
FS_SKIDPAD_RADIUS = 15.25 / 2.0  # m, center-to-center radius of the skidpad circles
FS_ENDURANCE_TYPICAL_LENGTH = 22000.0  # m, typical endurance event total distance
FS_AUTOCROSS_TYPICAL_LENGTH = 1000.0  # m, typical autocross track length (can vary)

# Vehicle reference values (Typical ranges/defaults for FS cars)
DEFAULT_TIRE_RADIUS = 0.2286  # m, default 18" Overall Diameter tire (common on 10" rim)
DEFAULT_WEIGHT_DISTRIBUTION = 0.48  # 48% front, 52% rear (example)
DEFAULT_CG_HEIGHT = 0.28  # m, typical center of gravity height for FS car
DEFAULT_FRONTAL_AREA = 1.1  # m², typical frontal area for FS car
DEFAULT_DRAG_COEFFICIENT = 1.0  # typical drag coefficient for FS car with basic aero
DEFAULT_LIFT_COEFFICIENT = -2.0  # typical lift coefficient (downforce) for FS car with moderate aero

# Engine reference values
DEFAULT_REDLINE = 14000  # RPM, typical redline for motorcycle engines used in FS
DEFAULT_IDLE_RPM = 1300  # RPM, typical idle speed for motorcycle engines
DEFAULT_POWER_TO_WEIGHT = 0.25  # kW/kg, example power-to-weight ratio

# Thermal reference values
DEFAULT_AMBIENT_TEMP = 25.0  # °C, standard ambient temperature for simulations
DEFAULT_ENGINE_OPERATING_TEMP = 95.0  # °C, typical engine operating temperature
DEFAULT_COOLANT_OPERATING_TEMP = 90.0  # °C, typical coolant operating temperature
DEFAULT_OIL_OPERATING_TEMP = 105.0  # °C, typical oil operating temperature
DEFAULT_RADIATOR_EFFICIENCY = 0.75  # typical radiator effectiveness
DEFAULT_THERMOSTAT_OPENING_TEMP = 82.0  # °C, typical thermostat opening temperature
DEFAULT_THERMOSTAT_FULLY_OPEN_TEMP = 92.0  # °C, typical temperature when thermostat is fully open

# Transmission reference values
DEFAULT_SHIFT_TIME = 0.040  # s, typical shift time for CAS system (40ms)
DEFAULT_WHEEL_SLIP_RATIO = 0.15  # typical target wheel slip ratio for launch

# Enumerations for use throughout the project
class EventType(Enum):
    """Types of Formula Student events."""
    ACCELERATION = auto()
    SKIDPAD = auto()
    AUTOCROSS = auto()
    ENDURANCE = auto()
    EFFICIENCY = auto()

class TireType(Enum):
    """Types of tires used in Formula Student."""
    DRY_SLICK = auto()
    WET_TREADED = auto()
    INTERMEDIATE = auto() # Less common

class EngineType(Enum):
    """Types of engines used in Formula Student."""
    MOTORCYCLE_FOUR_CYLINDER = auto()
    MOTORCYCLE_SINGLE_CYLINDER = auto()
    MOTORCYCLE_TWIN_CYLINDER = auto()
    CUSTOM = auto()
    ELECTRIC = auto()
    HYBRID = auto()

class ThermalWarningLevel(Enum):
    """Thermal warning levels for the cooling system."""
    NORMAL = auto()
    WARNING = auto()
    CRITICAL = auto()
    SHUTDOWN = auto()

# Formula Student scoring references (Max points per event)
FS_MAX_ACCELERATION_POINTS = 75.0
FS_MAX_SKIDPAD_POINTS = 75.0
FS_MAX_AUTOCROSS_POINTS = 100.0
FS_MAX_ENDURANCE_POINTS = 275.0 # Reduced from 325 in recent rules
FS_MAX_EFFICIENCY_POINTS = 100.0
FS_MAX_TOTAL_DYNAMIC_POINTS = 625.0 # Acceleration + Skidpad + Autocross + Endurance

# FS scoring formulas (Simplified versions - consult official rules for accuracy)
def calculate_acceleration_score(time_your: float, time_min: float, time_max: float = None) -> float:
    """Calculate FS acceleration score."""
    if time_your <= 0: return 0.0
    if time_max is None: time_max = 1.5 * time_min # FS Rules: Max time is 1.5 * Tmin
    if time_your > time_max: return 4.5 # Min points for completing
    score = 70.5 * ((time_max / time_your) - 1) / ((time_max / time_min) - 1) + 4.5
    return max(4.5, min(FS_MAX_ACCELERATION_POINTS, score))

def calculate_skidpad_score(time_your: float, time_min: float, time_max: float = None) -> float:
    """Calculate FS skidpad score."""
    if time_your <= 0: return 0.0
    if time_max is None: time_max = 1.25 * time_min # FS Rules: Max time is 1.25 * Tmin
    if time_your > time_max: return 4.5 # Min points for completing
    score = 70.5 * (((time_max / time_your)**2) - 1) / (((time_max / time_min)**2) - 1) + 4.5
    return max(4.5, min(FS_MAX_SKIDPAD_POINTS, score))

def calculate_autocross_score(time_your: float, time_min: float, time_max: float = None) -> float:
    """Calculate FS autocross score."""
    if time_your <= 0: return 0.0
    if time_max is None: time_max = 1.45 * time_min # FS Rules: Max time is 1.45 * Tmin
    if time_your > time_max: return 6.5 # Min points for completing
    score = 93.5 * ((time_max / time_your) - 1) / ((time_max / time_min) - 1) + 6.5
    return max(6.5, min(FS_MAX_AUTOCROSS_POINTS, score))

def calculate_endurance_score(time_your: float, time_min: float, time_max: float = None) -> float:
    """Calculate FS endurance score."""
    if time_your <= 0: return 0.0
    if time_max is None: time_max = 1.45 * time_min # FS Rules: Max time is 1.45 * Tmin
    if time_your > time_max: return 25.0 # Min points for completing
    score = 250.0 * ((time_max / time_your) - 1) / ((time_max / time_min) - 1) + 25.0
    return max(25.0, min(FS_MAX_ENDURANCE_POINTS, score))

def calculate_efficiency_score(eff_factor_your: float, eff_factor_min: float, eff_factor_max: float = None) -> float:
    """Calculate FS efficiency score based on efficiency factor (lower is better)."""
    if eff_factor_your <= 0 or eff_factor_min <= 0: return 0.0
    if eff_factor_max is None: eff_factor_max = 1.5 * eff_factor_min # Estimate if max not given
    if eff_factor_your >= eff_factor_max: return 0.0 # No points if too inefficient
    score = FS_MAX_EFFICIENCY_POINTS * (eff_factor_max - eff_factor_your) / (eff_factor_max - eff_factor_min)
    return max(0.0, min(FS_MAX_EFFICIENCY_POINTS, score))

# Standard track properties
DEFAULT_TRACK_WIDTH = 3.0  # m, typical track width for Formula Student tracks

# Fuel properties Class (moved here from fuel_systems.py to avoid circular imports if needed later)
class FuelPropertiesConstants:
    """Properties of different fuel types."""
    # Values: [density (kg/L), energy density (MJ/kg), stoich_afr, latent_heat (kJ/kg), octane_ron]
    _PROPERTIES = {
        'GASOLINE_98RON': [0.75, 44.4, 14.7, 350, 98],
        'E85': [0.78, 29.2, 9.8, 850, 105],
        'E100': [0.79, 26.8, 9.0, 920, 108],
        'METHANOL': [0.79, 19.9, 6.5, 1100, 109]
    }

class FuelPropertiesData: # Renamed from FuelProperties
    """Class holding properties for a specific fuel type, based on constants."""
    def __init__(self, fuel_type: str = 'E85'):
        lookup_key = fuel_type.upper() # Allow case-insensitive lookup
        if lookup_key not in FuelPropertiesConstants._PROPERTIES_LOOKUP:
            # Try matching without RON if applicable
            if 'GASOLINE' in lookup_key: lookup_key = 'GASOLINE_98RON'
            if lookup_key not in FuelPropertiesConstants._PROPERTIES_LOOKUP:
                raise ValueError(f"Unknown fuel type: {fuel_type}. Available: {list(FuelPropertiesConstants._PROPERTIES_LOOKUP.keys())}")

        props = FuelPropertiesConstants._PROPERTIES_LOOKUP[lookup_key]
        self.name = fuel_type
        self.density_kg_per_L = props[0]
        self.energy_density_MJ_per_kg = props[1]
        self.stoichiometric_afr = props[2]
        self.latent_heat_kJ_per_kg = props[3]
        self.octane_ron = props[4]

        # Derived properties
        self.density_kg_per_m3 = self.density_kg_per_L * 1000
        self.energy_density_J_per_kg = self.energy_density_MJ_per_kg * 1e6
        self.latent_heat_J_per_kg = self.latent_heat_kJ_per_kg * 1000

    def get_volumetric_energy_density_MJ_per_L(self) -> float:
        """Calculate volumetric energy density in MJ/L."""
        return self.density_kg_per_L * self.energy_density_MJ_per_kg

    def to_dict(self) -> dict:
        """Convert properties to a dictionary."""
        return {
            'name': self.name,
            'density_kg_per_L': self.density_kg_per_L,
            'energy_density_MJ_per_kg': self.energy_density_MJ_per_kg,
            'stoichiometric_afr': self.stoichiometric_afr,
            'latent_heat_kJ_per_kg': self.latent_heat_kJ_per_kg,
            'octane_ron': self.octane_ron,
            'volumetric_energy_density_MJ_per_L': self.get_volumetric_energy_density_MJ_per_L()
        }
    def __init__(self, fuel_type: str = 'E85'):
        """
        Initialize fuel properties.

        Args:
            fuel_type (str): Name of the fuel type (e.g., 'E85', 'GASOLINE_98RON').
                             Must match a key in _PROPERTIES.
        """
        if fuel_type not in self._PROPERTIES:
            raise ValueError(f"Unknown fuel type: {fuel_type}. Available: {list(self._PROPERTIES.keys())}")

        props = self._PROPERTIES[fuel_type]
        self.name = fuel_type
        self.density_kg_per_L = props[0]
        self.energy_density_MJ_per_kg = props[1]
        self.stoichiometric_afr = props[2]
        self.latent_heat_kJ_per_kg = props[3]
        self.octane_ron = props[4]

        # Derived properties
        self.density_kg_per_m3 = self.density_kg_per_L * 1000
        self.energy_density_J_per_kg = self.energy_density_MJ_per_kg * 1e6
        self.latent_heat_J_per_kg = self.latent_heat_kJ_per_kg * 1000