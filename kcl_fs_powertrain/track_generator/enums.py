"""Enumerations for track generation."""

from enum import Enum

class TrackMode(Enum):
    """Possible modes for how Voronoi regions are selected"""
    EXPAND = "expand"   # Results in roundish track shapes
    EXTEND = "extend"   # Results in elongated track shapes
    RANDOM = "random"   # Select regions randomly

class SimType(Enum):
    """Selection between output format for different simulators"""
    FSSIM = "fssim"     # FSSIM compatible .yaml file
    FSDS = "fsds"       # FSDS compatible .csv file
    GPX = "gpx"         # GPX track format
