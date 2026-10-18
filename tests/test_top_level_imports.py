"""The core data containers are reachable from the package root, as the same objects as in their subpackages."""
import phd_parser as pp


def test_core_classes_at_the_top_level():
    assert pp.GCData is pp.chromatography.GCData
    assert pp.IRData is pp.spectroscopy.infrared.IRData
    assert pp.LVData is pp.labview.LVData
    assert pp.MSData is pp.massspec.MSData
    assert pp.PhysisorptionData is pp.physisorption.PhysisorptionData
    assert pp.RamanData is pp.spectroscopy.raman.RamanData
    assert pp.TGAData is pp.tga.TGAData
    assert pp.XRDData is pp.xrd.XRDData
    assert pp.XPSData is pp.spectroscopy.xps.XPSData
    assert pp.rename_species is pp.naming.rename_species


def test_from_import():
    from phd_parser import GCData, LVData, rename_species  # noqa: F401
