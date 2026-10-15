import sys as _sys

import phd_parser.spectroscopy as spectroscopy
import phd_parser.xrd as xrd
import phd_parser.tga as tga
import phd_parser.massspec as massspec
import phd_parser.physisorption as physisorption
import phd_parser.logging as logging
import phd_parser.labview as labview
import phd_parser.chromatography as chromatography
import phd_parser.naming as naming

# Infrared, Raman and XPS moved under spectroscopy. The old paths stay importable,
# as the same module objects: pp.infrared is pp.spectroscopy.infrared,
# and "import phd_parser.infrared.omnic" still works.
from phd_parser.spectroscopy import infrared, raman, xps

import importlib as _importlib
import importlib.abc as _importlib_abc
import importlib.util as _importlib_util

_MOVED = ("infrared", "raman", "xps")


def _moved_target(fullname: str):
    for technique in _MOVED:
        old = f"phd_parser.{technique}"
        if fullname == old or fullname.startswith(old + "."):
            return "phd_parser.spectroscopy." + fullname[len("phd_parser."):]
    return None


class _AliasLoader(_importlib_abc.Loader):
    def __init__(self, module):
        self.module = module

    def create_module(self, spec):
        return self.module

    def exec_module(self, module):
        pass


class _MovedFinder(_importlib_abc.MetaPathFinder):
    """Resolve phd_parser.infrared[.x] (and raman, xps) to the module under phd_parser.spectroscopy."""

    def find_spec(self, fullname, path, target=None):
        target_name = _moved_target(fullname)
        if target_name is None:
            return None
        return _importlib_util.spec_from_loader(fullname, _AliasLoader(_importlib.import_module(target_name)))


if not any(isinstance(finder, _MovedFinder) for finder in _sys.meta_path):
    _sys.meta_path.insert(0, _MovedFinder())

# modules already imported: register them under the old name right away
for _name, _module in list(_sys.modules.items()):
    if _name.startswith("phd_parser.spectroscopy."):
        _old = "phd_parser." + _name[len("phd_parser.spectroscopy."):]
        if _moved_target(_old) is not None:
            _sys.modules.setdefault(_old, _module)

# The core data containers, also reachable without their subpackage:
# pp.GCData is pp.chromatography.GCData.
from phd_parser.chromatography import GCData
from phd_parser.spectroscopy import IRData, RamanData, SpectroscopyData
from phd_parser.labview import LVData
from phd_parser.massspec import MSData
from phd_parser.physisorption import PhysisorptionData
from phd_parser.tga import TGAData
from phd_parser.xrd import XRDData
from phd_parser.naming import SpeciesConvention, rename_species

__all__ = [
    # subpackages
    "logging", "spectroscopy", "infrared", "raman", "xps", "xrd", "tga", "massspec", "physisorption",
    "labview", "chromatography", "naming",
    # core data containers
    "GCData", "IRData", "LVData", "MSData", "PhysisorptionData", "RamanData", "SpectroscopyData",
    "TGAData", "XRDData",
    # naming
    "SpeciesConvention", "rename_species",
]