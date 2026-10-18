"""phd_parser: read and process the data of the Urakawa group's instruments into one consistent form.

One subpackage per technique, one core data class per technique, and the
core classes reachable from the package root: ``pp.GCData`` is
``pp.chromatography.GCData``. See the README for the design.
"""

import phd_parser.kernels as kernels
import phd_parser.spectroscopy as spectroscopy
import phd_parser.xrd as xrd
import phd_parser.tga as tga
import phd_parser.tpr as tpr
import phd_parser.massspec as massspec
import phd_parser.physisorption as physisorption
import phd_parser.logging as logging
import phd_parser.labview as labview
import phd_parser.chromatography as chromatography
import phd_parser.naming as naming
import phd_parser.pipeline as pipeline

# The core data containers, also reachable without their subpackage:
# pp.GCData is pp.chromatography.GCData. IRData and RamanData are the reader
# namespaces of SpectroscopyData, the one spectroscopy data class.
from phd_parser.chromatography import GCData
from phd_parser.spectroscopy import IRData, RamanData, SpectroscopyData, XPSData
from phd_parser.labview import LVData
from phd_parser.massspec import MSData
from phd_parser.physisorption import PhysisorptionData
from phd_parser.tga import TGAData
from phd_parser.tpr import TPRData
from phd_parser.xrd import XRDData
from phd_parser.naming import SpeciesConvention, rename_species
from phd_parser.pipeline import Pipeline, Step

__all__ = [
    # subpackages
    "kernels", "logging", "spectroscopy", "xrd", "tga", "tpr", "massspec", "physisorption",
    "labview", "chromatography", "naming", "pipeline",
    # core data containers and the spectroscopy readers
    "GCData", "IRData", "LVData", "MSData", "PhysisorptionData", "RamanData", "SpectroscopyData",
    "TGAData", "TPRData", "XPSData", "XRDData",
    # naming and pipelines
    "SpeciesConvention", "rename_species", "Pipeline", "Step",
]
