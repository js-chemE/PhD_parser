import os

import numpy as np
import pytest
import xarray as xr

from phd_parser.physisorption.core import PhysisorptionData

MOCK_FILE_PATH = os.path.join(os.path.dirname(__file__), "tristar", "2026-N-198.XLS")


# ----------------------------------------------------------------
# from_tristar_xls
# ----------------------------------------------------------------

def test_from_tristar_xls_branches():
    pdata = PhysisorptionData.from_tristar_xls(MOCK_FILE_PATH)
    assert pdata.branches == ["ads", "des"]
    assert pdata.n_points("ads") == pdata.p_rel("ads").size == pdata.q("ads").size
    assert pdata.n_points("des") == pdata.p_rel("des").size == pdata.q("des").size


def test_from_tristar_xls_isotherm_values():
    pdata = PhysisorptionData.from_tristar_xls(MOCK_FILE_PATH)
    assert np.all(np.diff(pdata.p_rel("ads")) > 0)
    assert np.all(np.diff(pdata.p_rel("des")) > 0)
    assert pdata.q("ads").min() > 0


def test_from_tristar_xls_bet():
    pdata = PhysisorptionData.from_tristar_xls(MOCK_FILE_PATH)
    assert pdata.bet is not None
    assert pdata.surface_area_bet == pytest.approx(105.1138)
    assert pdata.bet["surface_area_unit"] == "m\xb2/g"


def test_from_tristar_xls_report_preserved():
    pdata = PhysisorptionData.from_tristar_xls(MOCK_FILE_PATH)
    assert set(pdata.report.keys()) == {"header", "summary", "analyses", "sample_log"}
    assert "t_plot" in pdata.report["analyses"]
    assert "bjh_adsorption" in pdata.report["analyses"]


def test_from_tristar_xls_repr_includes_bet():
    pdata = PhysisorptionData.from_tristar_xls(MOCK_FILE_PATH)
    assert "BET=" in repr(pdata)


# ----------------------------------------------------------------
# get_quantity
# ----------------------------------------------------------------

def test_get_quantity_scalar():
    pdata = PhysisorptionData.from_tristar_xls(MOCK_FILE_PATH)
    q = pdata.get_quantity(0.5, branch="ads")
    assert isinstance(q, float)
    assert q == pytest.approx(44.242365739011824)


def test_get_quantity_array():
    pdata = PhysisorptionData.from_tristar_xls(MOCK_FILE_PATH)
    q = pdata.get_quantity([0.1, 0.5, 0.9], branch="ads")
    assert isinstance(q, np.ndarray)
    assert q.shape == (3,)


def test_get_quantity_tolerance_raises():
    pdata = PhysisorptionData.from_tristar_xls(MOCK_FILE_PATH)
    with pytest.raises(ValueError):
        pdata.get_quantity(0.5, branch="ads", tolerance=0.001)


def test_get_quantity_unknown_branch_raises():
    pdata = PhysisorptionData.from_tristar_xls(MOCK_FILE_PATH)
    with pytest.raises(KeyError):
        pdata.q("nonexistent")


# ----------------------------------------------------------------
# from_arrays — minimal, instrument-agnostic construction
# ----------------------------------------------------------------

def test_from_arrays_single_branch():
    pdata = PhysisorptionData.from_arrays(
        p_rel_ads=np.linspace(0.01, 0.9, 5),
        q_ads=np.array([10.0, 20.0, 30.0, 40.0, 50.0]),
    )
    assert pdata.branches == ["ads"]
    with pytest.raises(KeyError):
        pdata.q("des")


def test_from_arrays_requires_at_least_one_branch():
    with pytest.raises(ValueError):
        PhysisorptionData.from_arrays()


def test_from_arrays_with_bet():
    pdata = PhysisorptionData.from_arrays(
        p_rel_ads=np.linspace(0.01, 0.9, 5),
        q_ads=np.array([10.0, 20.0, 30.0, 40.0, 50.0]),
        bet={"surface_area": 123.4},
    )
    assert pdata.surface_area_bet == 123.4


def test_from_arrays_mismatched_dims_rejected():
    ds = xr.Dataset(
        {"q_ads": (("wrong_dim",), [1.0, 2.0])},
        coords={"wrong_dim": [0.1, 0.2]},
    )
    with pytest.raises(ValueError):
        PhysisorptionData(ds=ds)


def test_from_arrays_wrong_type_rejected():
    with pytest.raises(TypeError):
        PhysisorptionData(ds="not a dataset")


# ----------------------------------------------------------------
# NetCDF round-trip
# ----------------------------------------------------------------

def test_netcdf_roundtrip(tmp_path):
    pdata = PhysisorptionData.from_tristar_xls(MOCK_FILE_PATH)
    path = tmp_path / "isotherm.nc"
    pdata.to_netcdf(path)
    reloaded = PhysisorptionData.from_netcdf(path)
    assert reloaded.branches == pdata.branches
    np.testing.assert_allclose(reloaded.p_rel("ads"), pdata.p_rel("ads"))
    np.testing.assert_allclose(reloaded.q("ads"), pdata.q("ads"))
    reloaded.ds.close()
