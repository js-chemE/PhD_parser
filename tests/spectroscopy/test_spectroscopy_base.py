"""SpectroscopyData: one data model for IR and Raman, the old import paths, SpectroChemPy, smooth_gaussian."""
import importlib

import numpy as np
import pytest

import phd_parser as pp
from phd_parser.spectroscopy import IRData, RamanData, SpectroscopyData

AXIS_PER_CM = np.linspace(1000.0, 3000.0, 101)


@pytest.fixture
def ir():
    values = np.outer(np.arange(1.0, 5.0), np.ones(AXIS_PER_CM.size)) + np.sin(AXIS_PER_CM / 100.0)
    return IRData.from_arrays(AXIS_PER_CM, values, tos=np.arange(4) * 30.0, tos_start="2025-05-15 10:00",
                              data_type="absorbance")


def test_both_techniques_are_spectroscopy_data(ir):
    assert issubclass(IRData, SpectroscopyData) and issubclass(RamanData, SpectroscopyData)
    assert pp.SpectroscopyData is SpectroscopyData
    assert ir.technique == "infrared"
    assert IRData.from_arrays(AXIS_PER_CM, np.ones(AXIS_PER_CM.size)).data_type == "single_beam"


def test_the_base_is_usable_on_its_own(ir):
    generic = ir.as_type(SpectroscopyData)
    assert type(generic) is SpectroscopyData
    assert type(generic.smooth_moving(3)) is SpectroscopyData
    np.testing.assert_allclose(generic.to_transmittance().values, 10 ** -ir.values)


@pytest.mark.parametrize("old, new", [
    ("phd_parser.infrared", "phd_parser.spectroscopy.infrared"),
    ("phd_parser.infrared.core", "phd_parser.spectroscopy.infrared.core"),
    ("phd_parser.raman", "phd_parser.spectroscopy.raman"),
    ("phd_parser.xps", "phd_parser.spectroscopy.xps"),
    ("phd_parser.xps.casaxps", "phd_parser.spectroscopy.xps.casaxps"),
    ("phd_parser.raman.renishaw_wdf", "phd_parser.spectroscopy.raman.renishaw_wdf"),
])
def test_old_import_paths_still_work(old, new):
    assert importlib.import_module(old) is importlib.import_module(new)


def test_from_netcdf_on_the_base_returns_the_recorded_class(ir, tmp_path):
    ir.to_netcdf(tmp_path / "ir.nc")
    assert type(SpectroscopyData.from_netcdf(tmp_path / "ir.nc")) is IRData


def test_smooth_gaussian_sigma_is_in_per_cm():
    axis = np.arange(1000.0, 3000.0, 2.0)  # 2 cm⁻¹ spacing
    spike = np.zeros(axis.size)
    spike[500] = 1.0
    smoothed = IRData.from_arrays(axis, spike).smooth_gaussian(10.0)  # 10 cm⁻¹ = 5 points
    values = smoothed.values
    # second moment of the smoothed spike: sigma in cm⁻¹
    sigma_cm = np.sqrt(np.sum(values * (axis - axis[500]) ** 2) / np.sum(values))
    assert sigma_cm == pytest.approx(10.0, rel=1e-3)


def test_smooth_gaussian_warns_on_a_non_uniform_grid(caplog):
    axis = np.concatenate([np.arange(1000.0, 1100.0, 1.0), np.arange(1100.0, 1300.0, 4.0)])
    IRData.from_arrays(axis, np.ones(axis.size)).smooth_gaussian(5.0)
    assert "not uniform" in caplog.text


def test_to_spectrochempy_round_trips(ir):
    pytest.importorskip("spectrochempy")
    nd = ir.smooth_moving(3).to_spectrochempy()
    assert nd.shape == ir.shape
    np.testing.assert_allclose(np.asarray(nd.x.data), AXIS_PER_CM)
    assert any("smooth_moving" in line for line in nd.history)

    for back in (IRData.from_scp(nd), IRData.from_spectrochempy(nd), SpectroscopyData.from_spectrochempy(nd)):
        assert type(back) is IRData
        assert back.data_type == "absorbance"
        np.testing.assert_allclose(back.values, ir.smooth_moving(3).values)
        # the origin moves to the first scan, the absolute timestamps stay
        assert list(back.timestamps) == list(ir.timestamps)


def test_raman_through_spectrochempy_keeps_excitation():
    pytest.importorskip("spectrochempy")
    raman = RamanData.from_arrays(AXIS_PER_CM, np.ones((2, AXIS_PER_CM.size)), excitation_wavelength_nm=785.0,
                                  values_label="counts", tos=[0.0, 30.0])
    back = SpectroscopyData.from_spectrochempy(raman.to_spectrochempy())
    assert type(back) is RamanData
    assert back.excitation_wavelength_nm == 785.0
    assert back.data_type == "counts"


def test_to_spectrochempy_of_a_single_spectrum_is_one_row():
    pytest.importorskip("spectrochempy")
    single = IRData.from_arrays(AXIS_PER_CM, np.ones(AXIS_PER_CM.size), data_type="transmittance")
    nd = single.to_spectrochempy()
    assert nd.shape == (1, AXIS_PER_CM.size)
    assert nd.title == "transmittance"
