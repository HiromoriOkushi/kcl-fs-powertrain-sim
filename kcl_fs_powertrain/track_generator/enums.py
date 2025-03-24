"""Enumerations for track generation."""

from enum import Enum

class TrackMode(Enum):
    """Possible modes for how Voronoi regions are selected"""
    EXPAND = 1  # Results in roundish track shapes
    EXTEND = 2   # Results in elongated track shapes
    RANDOM = 3  # Select regions randomly

class SimType(Enum):
    """Selection between output format for different simulators"""
    FSSIM = 1    # FSSIM compatible .yaml file
    FSDS = 2       # FSDS compatible .csv file
    GPX = 3         # GPX track format
