import json

import numpy as np
import pandas as pd
import pytest

from phd_parser.infrared import IRData

WAVENUMBER_PER_CM = np.linspace(1000.0, 4000.0, 61)


def make_ir(n_scans, tos_start, level, background_level, delta_seconds=10.0, tos_first=0.0):
    """Build a synthetic single_beam IRData with a flat background."""
    values = float(level) + np.arange(n_scans, dtype=float)[:, None] * np.ones(WAVENUMBER_PER_CM.size)
    tos = tos_first + delta_seconds * np.arange(n_scans, dtype=float)
    ir = IRData.from_arrays(
        WAVENUMBER_PER_CM, values, tos=tos, tos_start=tos_start, data_type="single_beam"
    )
    return ir.set_background(np.full(WAVENUMBER_PER_CM.size, float(background_level)))


@pytest.fixture
def first():
    # 5 scans, 10:00:00 → 10:00:40
    return make_ir(5, "2024-05-01 10:00:00", level=100.0, background_level=1.0)


@pytest.fixture
def second():
    # 4 scans starting two minutes later, own background recorded after a restart
    return make_ir(4, "2024-05-01 10:02:00", level=200.0, background_level=2.0)


def test_merge_concatenates_scans_and_rebases_tos(first, second):
    merged = first.merge(second)

    assert merged.ndim == 2
    assert merged.shape == (9, WAVENUMBER_PER_CM.size)
    assert merged.tos_start == pd.Timestamp("2024-05-01 10:00:00")
    np.testing.assert_allclose(merged.tos, [0, 10, 20, 30, 40, 120, 130, 140, 150])
    np.testing.assert_allclose(merged.values[:5], first.values)
    np.testing.assert_allclose(merged.values[5:], second.values)
    np.testing.assert_array_equal(merged.ds.coords["scan"].values, np.arange(9))


def test_merge_keeps_only_the_first_background(first, second):
    merged = first.merge(second)

    assert merged.has_background
    assert "background" in merged.ds
    np.testing.assert_allclose(merged.background, 1.0)
    assert merged.background.ndim == 1


def test_merge_keep_background_last(first, second):
    np.testing.assert_allclose(first.merge(second, keep_background="last").background, 2.0)


def test_merge_keep_background_none_and_explicit(first, second):
    assert not first.merge(second, keep_background="none").has_background

    explicit = np.full(WAVENUMBER_PER_CM.size, 7.0)
    np.testing.assert_allclose(first.merge(second, keep_background=explicit).background, 7.0)


def test_merge_background_falls_back_when_chosen_segment_has_none(first, second):
    merged = first.del_background().merge(second)
    np.testing.assert_allclose(merged.background, 2.0)


def test_merge_is_chronological_regardless_of_call_order(first, second):
    forward = first.merge(second)
    backward = second.merge(first)

    np.testing.assert_allclose(backward.values, forward.values)
    np.testing.assert_allclose(backward.tos, forward.tos)
    np.testing.assert_allclose(backward.background, forward.background)
    assert backward.tos_start == forward.tos_start


def test_merge_order_given_sets_origin_to_self(first, second):
    merged = second.merge(first, order="given")

    # 'second' defines the origin and keeps its background, but sort=True still
    # orders the scans in time, so 'first' lands at negative tos.
    assert merged.tos_start == pd.Timestamp("2024-05-01 10:02:00")
    np.testing.assert_allclose(merged.background, 2.0)
    np.testing.assert_allclose(merged.tos, [-120, -110, -100, -90, -80, 0, 10, 20, 30])
    np.testing.assert_allclose(merged.values[:5], first.values)


def test_merge_unsorted_keeps_segments_as_blocks(first, second):
    merged = second.merge(first, order="given", sort=False)

    np.testing.assert_allclose(merged.values[:4], second.values)
    np.testing.assert_allclose(merged.values[4:], first.values)
    assert merged.tos[0] > merged.tos[-1]


def test_merge_timestamps_are_continuous(first, second):
    timestamps = first.merge(second).timestamps

    assert timestamps[0] == pd.Timestamp("2024-05-01 10:00:00")
    assert timestamps[-1] == pd.Timestamp("2024-05-01 10:02:30")
    assert timestamps.is_monotonic_increasing


def test_merge_promotes_single_spectra(first, second):
    merged = first.select_by_idx(2).merge(second.select_by_idx(0))

    assert merged.shape == (2, WAVENUMBER_PER_CM.size)
    np.testing.assert_allclose(merged.tos, [20.0, 120.0])


def test_merge_one_dimensional_into_time_series(first, second):
    merged = first.select_by_idx(2).merge(second)

    assert merged.shape == (5, WAVENUMBER_PER_CM.size)
    np.testing.assert_allclose(merged.tos, [20.0, 120.0, 130.0, 140.0, 150.0])


def test_merge_overlap_raises(first):
    overlapping = make_ir(4, "2024-05-01 10:00:20", level=300.0, background_level=3.0)

    with pytest.raises(ValueError, match="overlap"):
        first.merge(overlapping, on_overlap="raise")


def test_merge_overlap_trim_drops_duplicated_scans(first):
    overlapping = make_ir(4, "2024-05-01 10:00:20", level=300.0, background_level=3.0)
    merged = first.merge(overlapping, on_overlap="trim")

    np.testing.assert_allclose(merged.tos, [0, 10, 20, 30, 40, 50])
    assert merged.shape[0] == 6


def test_merge_without_timestamps_requires_explicit_offset():
    values = np.ones((3, WAVENUMBER_PER_CM.size))
    tos = np.array([0.0, 10.0, 20.0])
    a = IRData.from_arrays(WAVENUMBER_PER_CM, values, tos=tos, data_type="single_beam")
    b = IRData.from_arrays(WAVENUMBER_PER_CM, 2 * values, tos=tos, data_type="single_beam")

    merged = a.merge(b, tos_offset_seconds=120.0)
    np.testing.assert_allclose(merged.tos, [0, 10, 20, 120, 130, 140])


def test_merge_raises_when_only_one_segment_has_tos_start(first):
    no_start = IRData.from_arrays(
        WAVENUMBER_PER_CM,
        np.ones((2, WAVENUMBER_PER_CM.size)),
        tos=[0.0, 10.0],
        data_type="single_beam",
    )
    with pytest.raises(ValueError, match="tos_start"):
        first.merge(no_start)


def test_merge_rejects_different_data_types(first, second):
    with pytest.raises(ValueError, match="data_type"):
        first.merge(second.to_absorbance())


def test_merge_rejects_absorbance_with_different_backgrounds(first, second):
    with pytest.raises(ValueError, match="single_beam"):
        first.to_absorbance().merge(second.to_absorbance())


def test_merge_allows_absorbance_with_identical_backgrounds(first, second):
    second_same_background = second.set_background(np.full(WAVENUMBER_PER_CM.size, 1.0))
    merged = first.to_absorbance().merge(second_same_background.to_absorbance())

    assert merged.data_type == "absorbance"
    assert merged.shape == (9, WAVENUMBER_PER_CM.size)


def test_merge_rejects_different_wavenumber_axes_by_default(first):
    other_grid = np.linspace(1500.0, 3500.0, 41)
    other = IRData.from_arrays(
        other_grid,
        np.ones((3, other_grid.size)),
        tos=[0.0, 10.0, 20.0],
        tos_start="2024-05-01 10:05:00",
        data_type="single_beam",
    )
    with pytest.raises(ValueError, match="wavenumber axes"):
        first.merge(other)


def test_merge_interpolates_onto_common_wavenumber_range(first):
    other_grid = np.linspace(1500.0, 3500.0, 41)
    other = IRData.from_arrays(
        other_grid,
        np.full((3, other_grid.size), 5.0),
        tos=[0.0, 10.0, 20.0],
        tos_start="2024-05-01 10:05:00",
        data_type="single_beam",
    )
    merged = first.merge(other, wavenumber="interp")

    assert merged.shape[0] == 8
    assert merged.wavenumber_per_cm.min() >= 1500.0
    assert merged.wavenumber_per_cm.max() <= 3500.0
    assert merged.background.size == merged.shape[1]
    np.testing.assert_allclose(merged.values[-3:], 5.0)


def test_merge_drops_a_lonely_baseline(first, second):
    assert not first.correct_offset((1000, 1200)).merge(second).has_baseline


def test_merge_keeps_baseline_when_both_segments_have_one(first, second):
    merged = first.correct_offset((1000, 1200)).merge(second.correct_offset((1000, 1200)))

    assert merged.has_baseline
    assert merged.baseline.shape == merged.shape


def test_merge_records_provenance(first, second):
    log = json.loads(first.merge(second).ds.attrs["merge_log"])

    assert len(log) == 1
    assert log[0]["n_scans_first"] == 5
    assert log[0]["n_scans_second"] == 4
    assert log[0]["tos_offset_seconds"] == 120.0
    assert log[0]["background_kept"] == "first"


def test_merge_survives_netcdf_roundtrip(first, second, tmp_path):
    merged = first.merge(second)
    path = tmp_path / "merged.nc"
    merged.to_netcdf(path)
    loaded = IRData.from_netcdf(path)

    np.testing.assert_allclose(loaded.values, merged.values)
    np.testing.assert_allclose(loaded.tos, merged.tos)
    np.testing.assert_allclose(loaded.background, merged.background)
    assert loaded.tos_start == merged.tos_start


def test_merge_all_folds_in_chronological_order(first, second):
    third = make_ir(3, "2024-05-01 10:05:00", level=400.0, background_level=4.0)
    merged = IRData.merge_all([second, third, first])

    assert merged.shape[0] == 12
    assert merged.tos_start == pd.Timestamp("2024-05-01 10:00:00")
    np.testing.assert_allclose(merged.background, 1.0)
    assert np.all(np.diff(merged.tos) > 0)


def test_merge_all_requires_items():
    with pytest.raises(ValueError):
        IRData.merge_all([])


def test_merge_all_single_item_returns_it(first):
    assert IRData.merge_all([first]) is first


def test_to_single_beam_round_trips(first):
    np.testing.assert_allclose(first.to_absorbance().to_single_beam().values, first.values)
    np.testing.assert_allclose(first.to_transmittance().to_single_beam().values, first.values)
    assert first.to_single_beam() is first


def test_to_single_beam_requires_background(first):
    absorbance = first.to_absorbance().del_background()
    with pytest.raises(ValueError, match="background"):
        absorbance.to_single_beam()
