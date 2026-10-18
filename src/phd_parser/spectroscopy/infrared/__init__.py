"""Infrared readers: OMNIC ``.spa`` files, the SpectroChemPy bridge and the modulation (MES) helpers.

``IRData`` is :class:`~phd_parser.spectroscopy.SpectroscopyData` under its
former name; the readers are its ``from_omnic_spa`` and ``from_spectrochempy``.
"""

import phd_parser.spectroscopy.infrared.omnic as omnic
import phd_parser.spectroscopy.infrared.spectrochempy as spectrochempy
import phd_parser.spectroscopy.infrared.modulation as modulation
from phd_parser.spectroscopy.infrared.readers import IRData

__all__ = ["IRData", "omnic", "spectrochempy", "modulation"]
