"""Virtual sensors of the Nibelungen bridge digital twin (BAM)."""

from .transform import DEFAULT_INPUT_CHANNELS, transform
from .weather import add_wind_speed

__all__ = ["transform", "add_wind_speed", "DEFAULT_INPUT_CHANNELS"]
__version__ = "0.1.0"
