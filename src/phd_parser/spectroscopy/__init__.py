"""Spectroscopy: one data class for spectra over a wavenumber axis, one reader namespace per technique.

``SpectroscopyData`` (``spectroscopy/core.py``) is the whole data model for
infrared and Raman: the axis, the processing steps, NetCDF and the library
bridges. ``IRData`` and ``RamanData`` are not classes of their own but the
readers of each technique: ``IRData.from_omnic_spa(...)`` and
``RamanData.from_renishaw_wdf(...)`` both return a ``SpectroscopyData``.
``XPSData`` (``xps/core.py``) is the class for X-ray photoelectron
spectra, with the readers of the Thermo instrument's VAMAS and Avantage files.
"""

from phd_parser.spectroscopy.core import SpectroscopyData
import phd_parser.spectroscopy.infrared as infrared
import phd_parser.spectroscopy.raman as raman
import phd_parser.spectroscopy.xps as xps
from phd_parser.spectroscopy.infrared import IRData
from phd_parser.spectroscopy.raman import RamanData
from phd_parser.spectroscopy.xps import XPSData
import phd_parser.spectroscopy.units as units

__all__ = ["SpectroscopyData", "IRData", "RamanData", "XPSData", "infrared", "raman", "xps", "units"]
