"""Temperature-programmed reduction: :class:`TPRData` and the reader of the LabVIEW TPR rig's export.

``labview_tcd`` reads the tab-separated ``.txt`` export of the Urakawa
group's in-house rig (quartz reactor, TCD detectors, optional FID, bed
thermocouple ``TC 1``), both column variants; ``TPRData.from_labview_tcd``
builds the data class from it.
"""

import phd_parser.tpr.labview_tcd as labview_tcd
from phd_parser.tpr.core import TPRData

__all__ = ["TPRData", "labview_tcd"]
