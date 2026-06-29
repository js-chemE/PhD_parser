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
    branches = PhysisorptionData.from_tristar_xls(MOCK_FILE_PATH)
    assert set(branches.keys()) == {"adsorption", "desorption"}
    for pdata in branches.values():
        assert isinstance(pdata, PhysisorptionData)
        assert pdata.n_points == pdata.relative_pressure.size == pdata.values.size


def test_from_tristar_xls_isotherm_values():
    branches = PhysisorptionData.from_tristar_xls(MOCK_FILE_PATH)
    adsorption = branches["adsorption"]
    assert np.all(np.diff(adsorption.relative_pressure) > 0)
    assert adsorption.values.min() > 0


def test_from_tristar_xls_bet_on_adsorption_only():
    branches = PhysisorptionData.from_tristar_xls(MOCK_FILE_PATH)
    adsorption, desorption = branches["adsorption"], branches["desorption"]
    assert adsorption.bet is not None
    assert adsorption.surface_area_bet == pytest.approx(105.1138)
    assert adsorption.bet["surface_area_unit"] == "m\xb2/g"
    # BET is conventionally fit on the adsorption branch only
    assert desorption.bet is None
    assert desorption.surface_area_bet is None


def test_from_tristar_xls_report_preserved_on_both_branches():
    branches = PhysisorptionData.from_tristar_xls(MOCK_FILE_PATH)
    for pdata in branches.values():
        assert set(pdata.report.keys()) == {"header", "summary", "analyses", "sample_log"}
        assert "t_plot" in pdata.report["analyses"]
        assert "bjh_adsorption" in pdata.report["analyses"]


def test_from_tristar_xls_repr_includes_bet():
    branches = PhysisorptionData.from_tristar_xls(MOCK_FILE_PATH)
    assert "BET=" in repr(branches["adsorption"])
    assert "BET=" not in repr(branches["desorption"])


# ----------------------------------------------------------------
# get_quantity_adsorbed
# ----------------------------------------------------------------

def test_get_quantity_adsorbed_scalar():
    adsorption = PhysisorptionData.from_tristar_xls(MOCK_FILE_PATH)["adsorption"]
    q = adsorption.get_quantity_adsorbed(0.5)
    assert isinstance(q, float)
    assert q == pytest.approx(44.242365739011824)


def test_get_quantity_adsorbed_array():
    adsorption = PhysisorptionData.from_tristar_xls(MOCK_FILE_PATH)["adsorption"]
    q = adsorption.get_quantity_adsorbed([0.1, 0.5, 0.9])
    assert isinstance(q, np.ndarray)
    assert q.shape == (3,)


def test_get_quantity_adsorbed_tolerance_raises():
    adsorption = PhysisorptionData.from_tristar_xls(MOCK_FILE_PATH)["adsorption"]
    with pytest.raises(ValueError):
        adsorption.get_quantity_adsorbed(0.5, tolerance=0.001)


# ----------------------------------------------------------------
# from_arrays — minimal, instrument-agnostic construction
# ----------------------------------------------------------------

def test_from_arrays_basic():
    pdata = PhysisorptionData.from_arrays(
        relative_pressure=np.linspace(0.01, 0.9, 5),
        quantity_adsorbed=np.array([10.0, 20.0, 30.0, 40.0, 50.0]),
    )
    assert pdata.n_points == 5
    assert pdata.bet is None


def test_from_arrays_with_bet():
    pdata = PhysisorptionData.from_arrays(
        relative_pressure=np.linspace(0.01, 0.9, 5),
        quantity_adsorbed=np.array([10.0, 20.0, 30.0, 40.0, 50.0]),
        bet={"surface_area": 123.4},
    )
    assert pdata.surface_area_bet == 123.4


def test_from_arrays_wrong_dims_rejected():
    da = xr.DataArray([1.0, 2.0], coords={"wrong_dim": [0.1, 0.2]}, dims=["wrong_dim"])
    with pytest.raises(ValueError):
        PhysisorptionData(quantity_adsorbed=da)


def test_from_arrays_wrong_type_rejected():
    with pytest.raises(TypeError):
        PhysisorptionData(quantity_adsorbed="not a dataarray")


# ----------------------------------------------------------------
# NetCDF round-trip
# ----------------------------------------------------------------

def test_netcdf_roundtrip(tmp_path):
    adsorption = PhysisorptionData.from_tristar_xls(MOCK_FILE_PATH)["adsorption"]
    path = tmp_path / "isotherm.nc"
    adsorption.to_netcdf(path)
    reloaded = PhysisorptionData.from_netcdf(path)
    np.testing.assert_allclose(reloaded.relative_pressure, adsorption.relative_pressure)
    np.testing.assert_allclose(reloaded.values, adsorption.values)
    assert reloaded.surface_area_bet == pytest.approx(adsorption.surface_area_bet)
    reloaded.quantity_adsorbed.close()
