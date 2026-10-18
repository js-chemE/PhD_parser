"""X-ray photoelectron spectroscopy: :class:`XPSData` and the readers of the Thermo instrument's files.

``vamas`` reads the ISO 14976 ``.vms`` export (the raw counts, what CasaXPS
reads), ``avantage`` the Thermo Avantage ``.avg`` text dump, ``casaxps`` the
tab-separated export of a CasaXPS fit. ``XPSData.from_vamas``,
``from_avantage_avg`` and ``from_casaxps_export`` build the data class from
them.
"""

import phd_parser.spectroscopy.xps.avantage as avantage
import phd_parser.spectroscopy.xps.casaxps as casaxps
import phd_parser.spectroscopy.xps.vamas as vamas
from phd_parser.spectroscopy.xps.core import XPSData

__all__ = ["fitting", "PeakSpec", "doublet", "DOUBLETS", "parse_lineshape", "fit_peaks", "XPSData", "avantage", "casaxps", "vamas"]
from phd_parser.spectroscopy.xps import fitting
from phd_parser.spectroscopy.xps.fitting import PeakSpec, doublet, DOUBLETS, parse_lineshape, fit_peaks
