"""Processing steps of SpectroscopyData that had no test: normalisation, averaging windows, baselines, ids, maps."""
import numpy as np
import pytest

from phd_parser.spectroscopy import SpectroscopyData

AXIS_PER_CM = np.linspace(1000.0, 3000.0, 201)


@pytest.fixture
def series():
    peak = np.exp(-((AXIS_PER_CM - 2000.0) / 50.0) ** 2)
    values = np.outer(np.arange(1.0, 7.0), peak) + 0.1
    return SpectroscopyData.from_arrays(AXIS_PER_CM, values, tos=np.arange(6) * 10.0, data_type="absorbance",
                                        technique="infrared")


def test_normalise_integral_uses_the_trapezoid_rule(series):
    normalised = series.normalise_integral()
    integrals = np.trapezoid(normalised.values, x=normalised.wavenumber, axis=-1)
    np.testing.assert_allclose(integrals, 1.0)


def test_snr_windows_integral_metric_runs(series):
    snr = series.snr_windows(signal_range_cm=(1900, 2100), noise_range_cm=(1000, 1200), signal_metric="integral")
    assert snr.shape == (6,) and np.all(snr > 0)


def test_get_scan_by_tos_average_rejects_both_window_arguments(series):
    with pytest.raises(ValueError, match="exactly one"):
        series.get_scan_by_tos_average(20.0, number_of_scans=2, time_window=15.0)
    np.testing.assert_allclose(series.get_scan_by_tos_average(20.0), series.get_scan_by_tos(20.0))
    averaged = series.get_scan_by_tos_average(20.0, number_of_scans=3, direction="center")
    np.testing.assert_allclose(averaged, series.values[1:4].mean(axis=0))


def test_average_scans_by_tos_anchors_on_the_real_scan(series):
    averaged = series.average_scans_by_tos([19.0, 41.0], number_of_scans=1)
    np.testing.assert_allclose(averaged.tos, [20.0, 40.0])
    assert averaged.ds.attrs["averaged_anchor_tos"] == [20.0, 40.0]
    assert averaged.ds.attrs["averaged_target_tos"] == [19.0, 41.0]


def test_correct_pchip_treats_every_scan_like_a_single_spectrum(series):
    corrected = series.correct_pchip([1100.0, 1800.0, 2200.0, 2900.0], point_avg_half_width=2)
    for index in range(len(series)):
        single = series.select_by_idx(index).correct_pchip([1100.0, 1800.0, 2200.0, 2900.0], point_avg_half_width=2)
        np.testing.assert_allclose(corrected.values[index], single.values)
        np.testing.assert_allclose(corrected.baseline[index], single.baseline)


def test_select_scan_id_range_raises_on_an_empty_range(series):
    assert series.select_scan_id_range(2, 4).shape[0] == 3
    assert series.select_scan_id_range(min_id=100).shape[0] == 0
    with pytest.raises(ValueError, match="the data cover"):
        series.select_scan_id_range(min_id=100, empty="raise")


def test_get_gram_schmidt_scan_accepts_numpy_integers(series):
    np.testing.assert_allclose(
        series.get_gram_schmidt_scan(np.int64(0)).values, series.get_gram_schmidt_scan(0).values
    )


def test_len_counts_scans_or_map_pixels(series):
    assert len(series) == 6
    assert len(series.select_by_idx(0)) == 1
    data = np.ones((3, 4, AXIS_PER_CM.size))
    ds = SpectroscopyData._build_ds(AXIS_PER_CM * 100.0, data, attrs={"data_type": "counts"})
    assert len(SpectroscopyData(ds=ds)) == 12


def test_build_ds_does_not_edit_the_callers_attrs():
    attrs = {"data_type": None, "technique": "raman"}
    SpectroscopyData._build_ds(AXIS_PER_CM * 100.0, np.ones(AXIS_PER_CM.size), attrs=attrs)
    assert attrs == {"data_type": None, "technique": "raman"}


def test_timestamps_are_vectorised_and_keep_the_timezone():
    data = SpectroscopyData.from_arrays(AXIS_PER_CM, np.ones((3, AXIS_PER_CM.size)), tos=[0.0, 1.5, 3.0],
                                        tos_start="2026-04-21 14:00:00+02:00")
    assert str(data.timestamps.tz) == "UTC+02:00"
    assert data.timestamps[1] - data.timestamps[0] == np.timedelta64(1500, "ms")


def test_background_switch_recalculates_absorbance():
    wavenumber = AXIS_PER_CM
    single_beam = np.outer([2.0, 4.0], np.ones(wavenumber.size))
    data = SpectroscopyData.from_arrays(wavenumber, single_beam, tos=[0.0, 10.0], technique="infrared")
    with_first = data.with_background(np.full(wavenumber.size, 1.0)).to_absorbance()
    np.testing.assert_allclose(with_first.values[0], -np.log10(2.0))
    switched = with_first.with_background(np.full(wavenumber.size, 2.0))
    np.testing.assert_allclose(switched.values[0], 0.0, atol=1e-12)
    np.testing.assert_allclose(switched.to_single_beam().values, single_beam)


def test_a_series_is_always_stored_in_time_order():
    values = np.arange(4, dtype=float)[:, None] * np.ones(AXIS_PER_CM.size)
    shuffled = SpectroscopyData.from_arrays(AXIS_PER_CM, values, tos=[30.0, 0.0, 20.0, 10.0])
    np.testing.assert_allclose(shuffled.tos, [0.0, 10.0, 20.0, 30.0])
    np.testing.assert_allclose(shuffled.values[:, 0], [1.0, 3.0, 2.0, 0.0])  # each scan kept its own time


def test_average_scans_by_tos_with_unsorted_targets_stays_in_time_order(series):
    corrected = series.correct_offset((2800, 3000))
    averaged = corrected.average_scans_by_tos([40.0, 0.0, 20.0], number_of_scans=1)
    np.testing.assert_allclose(averaged.tos, [0.0, 20.0, 40.0])
    for position, target in enumerate([0.0, 20.0, 40.0]):
        np.testing.assert_allclose(averaged.values[position], corrected.get_scan_by_tos(target))
        np.testing.assert_allclose(averaged.baseline[position], corrected.get_baseline_by_tos(target))
