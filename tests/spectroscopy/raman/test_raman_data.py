"""RamanData as a thin SpectroscopyData: Raman names, one wavenumber axis with IR, maps, NetCDF, RamanSPy."""
import numpy as np
import pytest
import xarray as xr

from phd_parser.spectroscopy import IRData, RamanData, SpectroscopyData

SHIFT_PER_CM = np.linspace(200.0, 3200.0, 151)


@pytest.fixture
def series():
    values = np.outer(np.arange(1.0, 5.0), np.exp(-((SHIFT_PER_CM - 1600.0) / 30.0) ** 2))
    return RamanData.from_arrays(SHIFT_PER_CM, values, excitation_wavelength_nm=532.0,
                                 values_label="counts", tos=np.arange(4) * 10.0, tos_start="2025-05-15 10:00")


@pytest.fixture
def raman_map():
    data = np.random.default_rng(1).random((3, 4, SHIFT_PER_CM.size))
    ds = RamanData._build_ds(SHIFT_PER_CM * 100.0, data, attrs={"data_type": "counts", "excitation_wavelength_nm": 785.0})
    return RamanData(ds=ds.assign_coords(y=[0, 1, 2], x=[10, 20, 30, 40]))


def test_raman_shift_is_the_wavenumber_axis(series):
    assert series.ds["data"].dims == ("scan", "wavenumber")
    np.testing.assert_allclose(series.shift_per_cm, SHIFT_PER_CM)
    np.testing.assert_array_equal(series.shift, series.wavenumber)
    assert series.technique == "raman" and series.ds.attrs["technique"] == "raman"


def test_excitation_and_scattered_light(series):
    assert series.excitation_wavelength_nm == 532.0
    assert series.values_label == "counts" == series.data_type
    np.testing.assert_allclose(series.scattered_wavenumber_per_cm, 1e7 / 532.0 - SHIFT_PER_CM)
    np.testing.assert_allclose(series.scattered_wavelength_nm, 1e9 / (series.excitation_wavenumber - series.shift))
    assert series.da.name == "counts"
    assert series.timestamps is not None


def test_legacy_dataarray_constructor_still_works():
    da = xr.DataArray(np.ones(SHIFT_PER_CM.size), dims=["shift"], coords={"shift": SHIFT_PER_CM * 100.0})
    raman = RamanData(da=da, excitation_wavelength_nm=633.0, values_label="arbitrary")
    assert raman.excitation_wavelength_nm == 633.0
    assert raman.values_label == "arbitrary"
    assert raman.ds["data"].dims == ("wavenumber",)


def test_excitation_must_be_positive():
    with pytest.raises(ValueError):
        RamanData.from_arrays(SHIFT_PER_CM, np.ones(SHIFT_PER_CM.size), excitation_wavelength_nm=-1)


def test_raman_names_and_shared_steps(series):
    cropped = series.select_shift_range(1000, 2000)
    np.testing.assert_array_equal(cropped.values, series.select_wavenumber_range(1000, 2000).values)
    assert series.get_evolution(1600).dims == ("scan",)
    assert series.sort(False).shift_per_cm[0] == SHIFT_PER_CM.max()
    corrected = series.correct_offset((2800, 3200)).normalise_max()
    assert corrected.has_baseline
    assert corrected.excitation_wavelength_nm == 532.0
    assert [entry["step"] for entry in corrected.history] == ["from_arrays", "correct_offset", "normalise_max"]


def test_merge_of_two_raman_series_needs_no_background(series, caplog):
    later = RamanData.from_arrays(SHIFT_PER_CM, np.ones((2, SHIFT_PER_CM.size)), excitation_wavelength_nm=532.0,
                                  values_label="counts", tos=[0.0, 10.0], tos_start="2025-05-15 11:00")
    merged = series.merge(later)
    assert isinstance(merged, RamanData)
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


def test_netcdf_round_trip_and_class_from_technique(series, tmp_path):
    series.to_netcdf(tmp_path / "raman.nc")
    back = RamanData.from_netcdf(tmp_path / "raman.nc")
    np.testing.assert_allclose(back.values, series.values)
    np.testing.assert_allclose(back.tos, series.tos)
    assert back.tos_start == series.tos_start
    assert back.values_label == "counts"
    generic = SpectroscopyData.from_netcdf(tmp_path / "raman.nc")
    assert type(generic) is RamanData and generic.excitation_wavelength_nm == 532.0


def test_reads_files_of_the_former_dataarray_class(tmp_path):
    da = xr.DataArray(
        np.ones((2, SHIFT_PER_CM.size)), dims=["scan", "shift"],
        coords={"shift": SHIFT_PER_CM * 100.0, "scan": [0, 1], "tos": ("scan", [0.0, 5.0])},
        attrs={"values_label": "counts", "shift_units": "m^-1", "excitation_wavelength_nm": 532.0},
        name="counts",
    )
    da.to_netcdf(tmp_path / "old.nc")
    back = RamanData.from_netcdf(tmp_path / "old.nc")
    assert back.values_label == "counts"
    assert back.excitation_wavelength_nm == 532.0
    np.testing.assert_allclose(back.tos, [0.0, 5.0])


def test_ir_and_raman_convert_into_each_other(series):
    ir = series.as_type(IRData)
    assert isinstance(ir, IRData) and ir.technique == "raman"  # what recorded it stays
    np.testing.assert_array_equal(ir.wavenumber, series.wavenumber)
    assert isinstance(ir.as_type(RamanData), RamanData)


def test_ramanspy_round_trip(series, raman_map):
    ramanspy = pytest.importorskip("ramanspy")
    single = series.select_by_idx(0)
    assert isinstance(single.to_ramanspy(), ramanspy.Spectrum)
    container = series.to_ramanspy()
    assert isinstance(container, ramanspy.SpectralContainer)
    np.testing.assert_allclose(container.spectral_axis, SHIFT_PER_CM)
    np.testing.assert_allclose(container.spectral_data, series.values)
    assert isinstance(raman_map.to_ramanspy(), ramanspy.SpectralImage)

    back = RamanData.from_ramanspy(container, excitation_wavelength_nm=532.0, values_label="counts", tos=series.tos)
    np.testing.assert_allclose(back.values, series.values)
    np.testing.assert_allclose(back.tos, series.tos)
    assert back.history[-1]["step"] == "from_ramanspy"
    image = RamanData.from_ramanspy(raman_map.to_ramanspy())
    assert image.is_map
    np.testing.assert_allclose(image.values, raman_map.values)
