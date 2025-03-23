"""
Constants module for Formula Student powertrain simulation.

This module provides physical constants, unit conversion factors, and reference values
used throughout the Formula Student powertrain simulation.
"""

import numpy as np
from enum import Enum, auto

# Physical constants
GRAVITY = 9.81  # m/s², standard gravity
AIR_DENSITY_SEA_LEVEL = 1.225  # kg/m³, air density at sea level (15°C, 1013.25 hPa)
WATER_DENSITY = 1000.0  # kg/m³, density of water at 4°C
AIR_VISCOSITY = 1.81e-5  # Pa·s, dynamic viscosity of air at 15°C
WATER_SPECIFIC_HEAT = 4.186  # kJ/(kg·K), specific heat capacity of water
AIR_SPECIFIC_HEAT = 1.005  # kJ/(kg·K), specific heat capacity of air
AIR_GAS_CONSTANT = 287.05  # J/(kg·K), specific gas constant for dry air
STANDARD_PRESSURE = 101325.0  # Pa, standard atmospheric pressure
STEFAN_BOLTZMANN = 5.67e-8  # W/(m²·K⁴), Stefan-Boltzmann constant
ABSOLUTE_ZERO_C = -273.15  # Absolute zero in Celsius

# Unit conversion factors
KMH_TO_MS = 1.0 / 3.6  # Convert km/h to m/s
MS_TO_KMH = 3.6  # Convert m/s to km/h
MPH_TO_MS = 0.44704  # Convert mph to m/s
MS_TO_MPH = 2.23694  # Convert m/s to mph
KW_TO_HP = 1.34102  # Convert kilowatts to horsepower
HP_TO_KW = 0.7457  # Convert horsepower to kilowatts
NM_TO_LBFT = 0.7376  # Convert N·m to lb·ft
LBFT_TO_NM = 1.3558  # Convert lb·ft to N·m
KG_TO_LBS = 2.20462  # Convert kg to pounds
LBS_TO_KG = 0.45359  # Convert pounds to kg
PSI_TO_PA = 6894.76  # Convert psi to Pascal
PA_TO_PSI = 1.45038e-4  # Convert Pascal to psi
INCH_TO_M = 0.0254  # Convert inches to meters
M_TO_INCH = 39.3701  # Convert meters to inches
MM_TO_M = 0.001  # Convert mm to meters
M_TO_MM = 1000.0  # Convert meters to mm
DEG_TO_RAD = np.pi / 180.0  # Convert degrees to radians
RAD_TO_DEG = 180.0 / np.pi  # Convert radians to degrees
LITERS_TO_M3 = 0.001  # Convert liters to cubic meters
M3_TO_LITERS = 1000.0  # Convert cubic meters to liters
GAL_TO_LITERS = 3.78541  # Convert US gallons to liters
LITERS_TO_GAL = 0.264172  # Convert liters to US gallons

# Temperature conversions
def celsius_to_kelvin(temp_c):
    """Convert temperature from Celsius to Kelvin."""
    return temp_c + 273.15

def kelvin_to_celsius(temp_k):
    """Convert temperature from Kelvin to Celsius."""
    return temp_k - 273.15

def celsius_to_fahrenheit(temp_c):
    """Convert temperature from Celsius to Fahrenheit."""
    return temp_c * 9.0/5.0 + 32.0

def fahrenheit_to_celsius(temp_f):
    """Convert temperature from Fahrenheit to Celsius."""
    return (temp_f - 32.0) * 5.0/9.0

# Formula Student reference values
FS_MAX_TRACK_WIDTH = 1.5  # m, maximum track width constraint
FS_ACCELERATION_LENGTH = 75.0  # m, standard acceleration event distance
FS_SKIDPAD_RADIUS = 15.25 / 2  # m, center-to-center radius of the skidpad circles
FS_ENDURANCE_TYPICAL_LENGTH = 22000.0  # m, typical endurance event total distance
FS_AUTOCROSS_TYPICAL_LENGTH = 800.0  # m, typical autocross track length

# Vehicle reference values
DEFAULT_TIRE_RADIUS = 0.2286  # m, default 18" wheel (9" rim + 9" tire height)
DEFAULT_WEIGHT_DISTRIBUTION = 0.45  # 45% front, 55% rear (typical FS car)
DEFAULT_CG_HEIGHT = 0.3  # m, typical center of gravity height for FS car
DEFAULT_FRONTAL_AREA = 1.2  # m², typical frontal area for FS car
DEFAULT_DRAG_COEFFICIENT = 0.9  # typical drag coefficient for FS car without aero
DEFAULT_LIFT_COEFFICIENT = -1.5  # typical lift coefficient for FS car with aero

# Engine reference values
DEFAULT_REDLINE = 14000  # RPM, typical redline for motorcycle engines used in FS
DEFAULT_IDLE_RPM = 1300  # RPM, typical idle speed for motorcycle engines
DEFAULT_POWER_TO_WEIGHT = 0.2  # kW/kg, typical power-to-weight ratio for FS cars

# Thermal reference values
DEFAULT_AMBIENT_TEMP = 25.0  # °C, standard ambient temperature for simulations
DEFAULT_ENGINE_OPERATING_TEMP = 90.0  # °C, typical engine operating temperature
DEFAULT_COOLANT_OPERATING_TEMP = 85.0  # °C, typical coolant operating temperature
DEFAULT_OIL_OPERATING_TEMP = 95.0  # °C, typical oil operating temperature
DEFAULT_RADIATOR_EFFICIENCY = 0.8  # typical radiator effectiveness
DEFAULT_THERMOSTAT_OPENING_TEMP = 82.0  # °C, typical thermostat opening temperature
DEFAULT_THERMOSTAT_FULLY_OPEN_TEMP = 92.0  # °C, typical temperature when thermostat is fully open

# Transmission reference values
DEFAULT_SHIFT_TIME = 0.2  # s, typical shift time for CAS system
DEFAULT_WHEEL_SLIP_RATIO = 0.1  # typical optimal wheel slip ratio for acceleration

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
    DRY = auto()
    WET = auto()
    INTERMEDIATE = auto()

class EngineType(Enum):
    """Types of engines used in Formula Student."""
    MOTORCYCLE = auto()  # Converted motorcycle engine (most common)
    CUSTOM = auto()      # Custom-built engine
    ELECTRIC = auto()    # Electric motor
    HYBRID = auto()      # Hybrid system

class ThermalWarningLevel(Enum):
    """Thermal warning levels for the cooling system."""
    NORMAL = auto()
    WARNING = auto()
    CRITICAL = auto()
    SHUTDOWN = auto()

# Formula Student scoring references
FS_MAX_ACCELERATION_POINTS = 75.0
FS_MAX_SKIDPAD_POINTS = 75.0
FS_MAX_AUTOCROSS_POINTS = 100.0
FS_MAX_ENDURANCE_POINTS = 325.0
FS_MAX_EFFICIENCY_POINTS = 100.0
FS_MAX_TOTAL_DYNAMIC_POINTS = 675.0  # Sum of all dynamic event points

# FS scoring formulas (simplified)
def calculate_acceleration_score(time, best_time):
    """Calculate FS acceleration score based on time."""
    if time <= 0:
        return 0.0
    return FS_MAX_ACCELERATION_POINTS * min(1.0, (best_time / time)**2)

def calculate_skidpad_score(time, best_time):
    """Calculate FS skidpad score based on time."""
    if time <= 0:
        return 0.0
    return FS_MAX_SKIDPAD_POINTS * min(1.0, (best_time / time)**2)

def calculate_autocross_score(time, best_time):
    """Calculate FS autocross score based on time."""
    if time <= 0:
        return 0.0
    return FS_MAX_AUTOCROSS_POINTS * min(1.0, (best_time / time)**2)

def calculate_endurance_score(time, best_time):
    """Calculate FS endurance score based on time."""
    if time <= 0:
        return 0.0
    return FS_MAX_ENDURANCE_POINTS * min(1.0, (best_time / time)**2)

def calculate_efficiency_score(energy, min_energy, time, best_time):
    """Calculate FS efficiency score based on energy consumption and time."""
    if time <= 0 or energy <= 0:
        return 0.0
    
    t_factor = best_time / time
    energy_factor = min_energy / energy
    
    return FS_MAX_EFFICIENCY_POINTS * min(1.0, t_factor * energy_factor)

# Standard track properties
DEFAULT_TRACK_WIDTH = 3.0  # m, typical track width for Formula Student tracks

# Fuel properties
class FuelProperties:
    """Properties of different fuel types."""
    GASOLINE = {
        'density': 750.0,  # kg/m³
        'energy_density': 44.0,  # MJ/kg
        'specific_heat': 2.22,  # kJ/(kg·K)
        'latent_heat': 305.0,  # kJ/kg
        'stoich_afr': 14.7,  # stoichiometric air-fuel ratio
        'octane_rating': 95.0  # RON
    }
    
    E85 = {
        'density': 781.0,  # kg/m³
        'energy_density': 29.2,  # MJ/kg
        'specific_heat': 2.44,  # kJ/(kg·K)
        'latent_heat': 850.0,  # kJ/kg
        'stoich_afr': 9.8,  # stoichiometric air-fuel ratio
        'octane_rating': 105.0  # RON
    }
    
    METHANOL = {
        'density': 791.0,  # kg/m³
        'energy_density': 19.7,  # MJ/kg
        'specific_heat': 2.51,  # kJ/(kg·K)
        'latent_heat': 1100.0,  # kJ/kg
        'stoich_afr': 6.5,  # stoichiometric air-fuel ratio
        'octane_rating': 108.7  # RON
    }