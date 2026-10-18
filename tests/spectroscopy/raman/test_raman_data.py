"""The Raman face of SpectroscopyData: Raman names, one wavenumber axis with IR, maps, NetCDF, RamanSPy, CSV."""
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from phd_parser.spectroscopy import RamanData, SpectroscopyData

SHIFT_PER_CM = np.linspace(200.0, 3200.0, 151)


@pytest.fixture
def series():
    values = np.outer(np.arange(1.0, 5.0), np.exp(-((SHIFT_PER_CM - 1600.0) / 30.0) ** 2))
    return SpectroscopyData.from_arrays(SHIFT_PER_CM, values, excitation_wavelength_nm=532.0,
                                        data_type="counts", tos=np.arange(4) * 10.0, tos_start="2025-05-15 10:00")


@pytest.fixture
def raman_map():
    data = np.random.default_rng(1).random((3, 4, SHIFT_PER_CM.size))
    ds = SpectroscopyData._build_ds(
        SHIFT_PER_CM * 100.0, data,
        attrs={"technique": "raman", "data_type": "counts", "excitation_wavelength_nm": 785.0},
    )
    return SpectroscopyData(ds=ds.assign_coords(y=[0, 1, 2], x=[10, 20, 30, 40]))


def test_raman_shift_is_the_wavenumber_axis(series):
    assert series.ds["data"].dims == ("scan", "wavenumber")
    np.testing.assert_allclose(series.shift_per_cm, SHIFT_PER_CM)
    np.testing.assert_array_equal(series.shift, series.wavenumber)
    assert series.technique == "raman" and series.ds.attrs["technique"] == "raman"
    assert series.n_spectral == SHIFT_PER_CM.size


def test_excitation_and_scattered_light(series):
    assert series.excitation_wavelength_nm == 532.0
    assert series.values_label == "counts" == series.data_type
    np.testing.assert_allclose(series.scattered_wavenumber_per_cm, 1e7 / 532.0 - SHIFT_PER_CM)
    np.testing.assert_allclose(series.scattered_wavelength_nm, 1e9 / (series.excitation_wavenumber - series.shift))
    assert series.da.name == "counts"
    assert series.timestamps is not None
    assert "excitation=532.0 nm" in repr(series)


def test_scattered_axes_need_an_excitation_wavelength():
    plain = SpectroscopyData.from_arrays(SHIFT_PER_CM, np.ones(SHIFT_PER_CM.size))
    with pytest.raises(ValueError, match="excitation_wavelength_nm"):
        plain.scattered_wavenumber


def test_excitation_must_be_positive():
    with pytest.raises(ValueError):
        SpectroscopyData.from_arrays(SHIFT_PER_CM, np.ones(SHIFT_PER_CM.size), excitation_wavelength_nm=-1)


def test_raman_names_and_shared_steps(series):
    cropped = series.select_shift_range(1000, 2000)
    np.testing.assert_array_equal(cropped.values, series.select_wavenumber_range(1000, 2000).values)
    assert cropped.history[-1]["step"] == "select_shift_range"
    assert series.get_evolution(1600).dims == ("scan",)
    assert series.sort(ascending=False).shift_per_cm[0] == SHIFT_PER_CM.max()
    corrected = series.correct_offset((2800, 3200)).normalise_max()
    assert corrected.has_baseline
    assert corrected.excitation_wavelength_nm == 532.0
    assert [entry["step"] for entry in corrected.history] == ["from_arrays", "correct_offset", "normalise_max"]


def test_merge_of_two_raman_series_needs_no_background(series, caplog):
    later = SpectroscopyData.from_arrays(SHIFT_PER_CM, np.ones((2, SHIFT_PER_CM.size)), excitation_wavelength_nm=532.0,
                                         data_type="counts", tos=[0.0, 10.0], tos_start="2025-05-15 11:00")
    merged = series.merge(later)
    assert merged.shape == (6, SHIFT_PER_CM.size)
    np.testing.assert_allclose(merged.tos[-2:], [3600.0, 3610.0])
    assert "background" not in caplog.text


def test_maps_keep_their_layout_and_coordinates(raman_map):
    assert raman_map.is_map and raman_map.shape == (3, 4, SHIFT_PER_CM.size)
    smoothed = raman_map.smooth_moving(3).select_shift_range(500, 2500).smooth_gaussian(20)
    assert smoothed.ds["data"].dims == ("y", "x", "wavenumber")
    assert smoothed.ds.coords["x"].values.tolist() == [10, 20, 30, 40]
    np.testing.assert_allclose(smoothed.get_map_spectrum(x=1, y=2), smoothed.values[2, 1])
    with pytest.raises(ValueError, match="requires 2-D data"):
        raman_map.average_scans(2)
    with pytest.raises(ValueError, match="map"):
        raman_map.merge(raman_map)


def test_netcdf_round_trip_keeps_technique_and_metadata(series, tmp_path):
    series = SpectroscopyData(ds=series.ds, metadata={"instrument": "test", "power": [10, "mW"]})
    series.to_netcdf(tmp_path / "raman.nc")
    back = SpectroscopyData.from_netcdf(tmp_path / "raman.nc")
    np.testing.assert_allclose(back.values, series.values)
    np.testing.assert_allclose(back.tos, series.tos)
    assert back.tos_start == series.tos_start
    assert back.values_label == "counts"
    assert back.technique == "raman" and back.excitation_wavelength_nm == 532.0
    assert back.metadata == {"instrument": "test", "power": [10, "mW"]}


def test_reads_files_of_the_former_dataarray_class(tmp_path):
    da = xr.DataArray(
        np.ones((2, SHIFT_PER_CM.size)), dims=["scan", "shift"],
        coords={"shift": SHIFT_PER_CM * 100.0, "scan": [0, 1], "tos": ("scan", [0.0, 5.0])},
        attrs={"values_label": "counts", "shift_units": "m^-1", "excitation_wavelength_nm": 532.0},
        name="counts",
    )
    da.to_netcdf(tmp_path / "old.nc")
    back = SpectroscopyData.from_netcdf(tmp_path / "old.nc")
    assert back.values_label == "counts"
    assert back.excitation_wavelength_nm == 532.0
    np.testing.assert_allclose(back.tos, [0.0, 5.0])


def test_to_csv_labels_the_axis_by_technique(series, tmp_path):
    series.to_csv(tmp_path / "raman.csv")
    frame = pd.read_csv(tmp_path / "raman.csv", index_col=0)
    assert frame.index.name == "raman_shift [cm-1]"
    assert list(frame.columns) == ["tos_0.00s", "tos_10.00s", "tos_20.00s", "tos_30.00s"]
    np.testing.assert_allclose(frame.values, series.values.T)
    ir = SpectroscopyData.from_arrays(SHIFT_PER_CM, np.ones(SHIFT_PER_CM.size), technique="infrared")
    ir.to_csv(tmp_path / "ir.csv", wavenumber_units="m-1")
    frame = pd.read_csv(tmp_path / "ir.csv", index_col=0)
    assert frame.index.name == "wavenumber [m-1]" and list(frame.columns) == ["single_beam"]


def test_ramanspy_round_trip(series, raman_map):
    ramanspy = pytest.importorskip("ramanspy")
    single = series.select_by_idx(0)
    assert isinstance(single.to_ramanspy(), ramanspy.Spectrum)
    container = series.to_ramanspy()
    assert isinstance(container, ramanspy.SpectralContainer)
    np.testing.assert_allclose(container.spectral_axis, SHIFT_PER_CM)
    np.testing.assert_allclose(container.spectral_data, series.values)
    assert isinstance(raman_map.to_ramanspy(), ramanspy.SpectralImage)

    back = SpectroscopyData.from_ramanspy(container, excitation_wavelength_nm=532.0, data_type="counts", tos=series.tos)
    np.testing.assert_allclose(back.values, series.values)
    np.testing.assert_allclose(back.tos, series.tos)
    assert back.technique == "raman"
    assert back.history[-1]["step"] == "from_ramanspy"
    image = SpectroscopyData.from_ramanspy(raman_map.to_ramanspy())
    assert image.is_map
    np.testing.assert_allclose(image.values, raman_map.values)
