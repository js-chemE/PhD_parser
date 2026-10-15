"""Spectroscopy: one generic core class for spectra over a spectral axis, one subpackage per technique.

``SpectroscopyData`` (``spectroscopy/core.py``) holds what infrared and Raman
share; ``IRData`` and ``RamanData`` add what only they have. ``xps`` is a file
reader only for now, without a core class.
"""

from phd_parser.spectroscopy.core import SpectroscopyData
import phd_parser.spectroscopy.infrared as infrared
import phd_parser.spectroscopy.raman as raman
import phd_parser.spectroscopy.xps as xps
from phd_parser.spectroscopy.infrared import IRData
from phd_parser.spectroscopy.raman import RamanData

__all__ = ["SpectroscopyData", "IRData", "RamanData", "infrared", "raman", "xps"]
