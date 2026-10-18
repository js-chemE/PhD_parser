"""SpectroscopyData: one class for IR and Raman, the old names and import paths, SpectroChemPy, smooth_gaussian."""
import numpy as np
import pytest

import phd_parser as pp
from phd_parser.spectroscopy import IRData, RamanData, SpectroscopyData

AXIS_PER_CM = np.linspace(1000.0, 3000.0, 101)


@pytest.fixture
def ir():
    values = np.outer(np.arange(1.0, 5.0), np.ones(AXIS_PER_CM.size)) + np.sin(AXIS_PER_CM / 100.0)
    return SpectroscopyData.from_arrays(AXIS_PER_CM, values, tos=np.arange(4) * 30.0, tos_start="2025-05-15 10:00",
                                        data_type="absorbance", technique="infrared")


def test_the_technique_names_are_readers_of_the_one_class():
    assert pp.SpectroscopyData is SpectroscopyData
    assert pp.IRData is pp.spectroscopy.infrared.IRData and pp.RamanData is pp.spectroscopy.raman.RamanData
    ir = IRData.from_arrays(AXIS_PER_CM, np.ones(AXIS_PER_CM.size))
    raman = RamanData.from_arrays(AXIS_PER_CM, np.ones(AXIS_PER_CM.size), excitation_wavelength_nm=532.0)
    assert type(ir) is type(raman) is SpectroscopyData
    assert (ir.technique, ir.data_type) == ("infrared", "single_beam")
    assert (raman.technique, raman.data_type) == ("raman", "intensity")
    assert ir.history[-1]["step"] == "from_arrays" and "technique" not in ir.history[-1]["parameters"]
    with pytest.raises(TypeError, match="reader namespace"):
        IRData()


def test_technique_is_metadata_and_sets_the_default_data_type(ir):
    assert ir.technique == "infrared"
    assert SpectroscopyData.from_arrays(AXIS_PER_CM, np.ones(AXIS_PER_CM.size), technique="infrared").data_type == "single_beam"
    raman = SpectroscopyData.from_arrays(AXIS_PER_CM, np.ones(AXIS_PER_CM.size), excitation_wavelength_nm=532.0)
    assert raman.technique == "raman" and raman.data_type == "intensity"
    plain = SpectroscopyData.from_arrays(AXIS_PER_CM, np.ones(AXIS_PER_CM.size))
    assert plain.technique is None and plain.data_type is None
    assert "technique" not in plain.ds.attrs


def test_processing_needs_no_technique():
    values = np.outer(np.arange(1.0, 5.0), np.ones(AXIS_PER_CM.size))
    generic = SpectroscopyData.from_arrays(AXIS_PER_CM, values, tos=np.arange(4) * 30.0, data_type="absorbance")
    assert generic.technique is None
    assert type(generic.smooth_moving(3)) is SpectroscopyData
    np.testing.assert_allclose(generic.to_transmittance().values, 10 ** -generic.values)


def test_ir_and_raman_data_combine(ir):
    raman = SpectroscopyData.from_arrays(AXIS_PER_CM, np.ones((4, AXIS_PER_CM.size)), tos=ir.tos,
                                         excitation_wavelength_nm=532.0, data_type="absorbance")
    assert (ir - raman).technique == "infrared"  # the left operand wins on attributes


def test_netcdf_round_trip_keeps_the_technique(ir, tmp_path):
    ir.to_netcdf(tmp_path / "ir.nc")
    back = SpectroscopyData.from_netcdf(tmp_path / "ir.nc")
    assert type(back) is SpectroscopyData and back.technique == "infrared"
    np.testing.assert_allclose(back.values, ir.values)


def test_smooth_gaussian_sigma_is_in_per_cm():
    axis = np.arange(1000.0, 3000.0, 2.0)  # 2 cm^-1 spacing
    spike = np.zeros(axis.size)
    spike[500] = 1.0
    smoothed = SpectroscopyData.from_arrays(axis, spike).smooth_gaussian(10.0)  # 10 cm^-1 = 5 points
    values = smoothed.values
    # second moment of the smoothed spike: sigma in cm^-1
    sigma_cm = np.sqrt(np.sum(values * (axis - axis[500]) ** 2) / np.sum(values))
    assert sigma_cm == pytest.approx(10.0, rel=1e-3)


def test_smooth_gaussian_warns_on_a_non_uniform_grid(caplog):
    axis = np.concatenate([np.arange(1000.0, 1100.0, 1.0), np.arange(1100.0, 1300.0, 4.0)])
    SpectroscopyData.from_arrays(axis, np.ones(axis.size)).smooth_gaussian(5.0)
    assert "not uniform" in caplog.text


def test_to_spectrochempy_round_trips(ir):
    pytest.importorskip("spectrochempy")
    nd = ir.smooth_moving(3).to_spectrochempy()
    assert nd.shape == ir.shape
    np.testing.assert_allclose(np.asarray(nd.x.data), AXIS_PER_CM)
    assert any("smooth_moving" in line for line in nd.history)

    back = SpectroscopyData.from_spectrochempy(nd)
    assert back.technique == "infrared"
    assert back.data_type == "absorbance"
    np.testing.assert_allclose(back.values, ir.smooth_moving(3).values)
    # the origin moves to the first scan, the absolute timestamps stay
    assert list(back.timestamps) == list(ir.timestamps)
    assert back.history[-1]["step"] == "from_spectrochempy"


def test_raman_through_spectrochempy_keeps_excitation():
    pytest.importorskip("spectrochempy")
    raman = SpectroscopyData.from_arrays(AXIS_PER_CM, np.ones((2, AXIS_PER_CM.size)), excitation_wavelength_nm=785.0,
                                         data_type="counts", tos=[0.0, 30.0])
    back = SpectroscopyData.from_spectrochempy(raman.to_spectrochempy())
    assert back.technique == "raman"
    assert back.excitation_wavelength_nm == 785.0
    assert back.data_type == "counts"


def test_spectrochempy_dataset_without_our_meta_gets_the_technique_of_its_title():
    scp = pytest.importorskip("spectrochempy")
    nd = scp.NDDataset(np.ones((1, AXIS_PER_CM.size)), title="absorbance")
    nd.set_coordset(y=scp.Coord([0.0], title="scan"), x=scp.Coord(AXIS_PER_CM, title="wavenumber", units="1/cm"))
    back = SpectroscopyData.from_spectrochempy(nd)
    assert back.technique == "infrared" and back.data_type == "absorbance"
    nd.title = "whatever"
    with pytest.raises(ValueError, match="data_type"):
        SpectroscopyData.from_spectrochempy(nd)


def test_to_spectrochempy_of_a_single_spectrum_is_one_row():
    pytest.importorskip("spectrochempy")
    single = SpectroscopyData.from_arrays(AXIS_PER_CM, np.ones(AXIS_PER_CM.size), data_type="transmittance")
    nd = single.to_spectrochempy()
    assert nd.shape == (1, AXIS_PER_CM.size)
    assert nd.title == "transmittance"
