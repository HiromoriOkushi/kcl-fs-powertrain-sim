"""Enumerations for track generation modes and output formats."""

from enum import Enum, auto

class TrackMode(Enum):
    """
    Defines how Voronoi regions are selected to shape the track.

    Attributes:
        EXPAND: Selects adjacent regions, resulting in roundish track shapes.
        EXTEND: Selects regions along a line, resulting in elongated shapes.
        RANDOM: Selects regions randomly.
    """
    EXPAND = auto()
    EXTEND = auto()
    RANDOM = auto()

class SimType(Enum):
    """
    Defines the output format for the generated track data.

    Attributes:
        FSSIM: Formula Student Simulator (FSSIM) YAML format (.yaml).
        FSDS: Formula Student Driverless Simulator (FSDS) CSV format (.csv).
        GPX: GPS Exchange Format (.gpx).
    """
    FSSIM = 'yaml' # Use extension as value for clarity
    FSDS = 'csv'
    GPX = 'gpx'