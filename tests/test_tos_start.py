"""The with_/set_/del_/move_ tos_start family, shared by every class that has a tos origin."""

import numpy as np
import pandas as pd
import pytest

from phd_parser.infrared import IRData
from phd_parser.labview import LVData
from phd_parser.massspec import MSData

TOS_START = pd.Timestamp("2026-04-21 14:00:00", tz="Europe/Amsterdam")
TOS = np.array([0.0, 60.0, 120.0])


def make_ir():
    return IRData.from_arrays(
        np.linspace(1000.0, 4000.0, 11),
        np.ones((3, 11)),
        tos=TOS,
        tos_start=TOS_START,
        data_type="single_beam",
    )


def make_lv():
    df = pd.DataFrame({
        "timestamp": [TOS_START + pd.Timedelta(seconds=float(t)) for t in TOS],
        "Reactor T PV": [100.0, 200.0, 300.0],
    })
    return LVData.from_dataframe(df, tos_start=TOS_START)


def make_ms():
    return MSData.from_arrays(
        cycle=np.arange(3),
        channels={0: np.array([28.0, 44.0])},
        values={0: np.ones((3, 2))},
        tos=TOS,
        tos_start=TOS_START,
    )


BUILDERS = [pytest.param(make_ir, id="IRData"),
            pytest.param(make_lv, id="LVData"),
            pytest.param(make_ms, id="MSData")]


@pytest.mark.parametrize("build", BUILDERS)
def test_with_tos_start_preserves_absolute_timestamps(build):
    data = build()
    before = list(data.timestamps)

    moved = data.with_tos_start(TOS_START + pd.Timedelta(seconds=30))

    assert moved.tos_start == TOS_START + pd.Timedelta(seconds=30)
    # Origin moved 30 s later -> every tos is 30 s smaller...
    np.testing.assert_allclose(moved.tos, TOS - 30.0)
    # ...and the wall-clock of every sample is untouched.
    assert list(moved.timestamps) == before


@pytest.mark.parametrize("build", BUILDERS)
def test_with_tos_start_earlier_origin_grows_tos(build):
    data = build()
    before = list(data.timestamps)

    moved = data.with_tos_start(TOS_START - pd.Timedelta(minutes=5))

    np.testing.assert_allclose(moved.tos, TOS + 300.0)
    assert list(moved.timestamps) == before


@pytest.mark.parametrize("build", BUILDERS)
def test_set_tos_start_keeps_tos_and_moves_timestamps(build):
    data = build()

    stamped = data.set_tos_start(TOS_START + pd.Timedelta(hours=1))

    np.testing.assert_allclose(stamped.tos, TOS)
    assert stamped.tos_start == TOS_START + pd.Timedelta(hours=1)
    assert list(stamped.timestamps) == [t + pd.Timedelta(hours=1) for t in data.timestamps]


@pytest.mark.parametrize("build", BUILDERS)
def test_del_tos_start(build):
    data = build()

    dropped = data.del_tos_start()

    assert dropped.tos_start is None
    assert dropped.timestamps is None
    np.testing.assert_allclose(dropped.tos, TOS)
    # Dropping twice is not an error.
    assert dropped.del_tos_start().tos_start is None


@pytest.mark.parametrize("build", BUILDERS)
def test_move_tos_start_by_seconds(build):
    data = build()

    moved = data.move_tos_start_by(90)

    assert moved.tos_start == TOS_START + pd.Timedelta(seconds=90)
    np.testing.assert_allclose(moved.tos, TOS - 90.0)
    assert list(moved.timestamps) == list(data.timestamps)


@pytest.mark.parametrize("build", BUILDERS)
def test_move_tos_start_by_accepts_timedelta_and_string(build):
    data = build()

    by_timedelta = data.move_tos_start_by(pd.Timedelta(minutes=2))
    by_string = data.move_tos_start_by("2min")
    by_seconds = data.move_tos_start_by(120)

    assert by_timedelta.tos_start == by_seconds.tos_start == by_string.tos_start
    np.testing.assert_allclose(by_timedelta.tos, by_seconds.tos)
    np.testing.assert_allclose(by_string.tos, by_seconds.tos)


@pytest.mark.parametrize("build", BUILDERS)
def test_move_tos_start_by_negative_grows_tos(build):
    data = build()

    moved = data.move_tos_start_by(-45)

    np.testing.assert_allclose(moved.tos, TOS + 45.0)


@pytest.mark.parametrize("build", BUILDERS)
def test_move_tos_start_by_requires_an_origin(build):
    data = build().del_tos_start()

    with pytest.raises(ValueError, match="No tos_start to move"):
        data.move_tos_start_by(30)


@pytest.mark.parametrize("build", BUILDERS)
def test_all_methods_are_immutable(build):
    data = build()
    tos_before = data.tos.copy()

    data.with_tos_start(TOS_START + pd.Timedelta(seconds=30))
    data.set_tos_start(TOS_START + pd.Timedelta(seconds=30))
    data.move_tos_start_by(30)
    data.del_tos_start()

    np.testing.assert_allclose(data.tos, tos_before)
    assert data.tos_start == TOS_START


@pytest.mark.parametrize("build", BUILDERS)
def test_with_tos_start_on_unanchored_data_just_anchors_it(build):
    data = build().del_tos_start()

    anchored = data.with_tos_start(TOS_START)

    assert anchored.tos_start == TOS_START
    np.testing.assert_allclose(anchored.tos, TOS)


@pytest.mark.parametrize("build", BUILDERS)
def test_round_trip_returns_to_the_original(build):
    data = build()

    same = data.move_tos_start_by(120).move_tos_start_by(-120)

    assert same.tos_start == data.tos_start
    np.testing.assert_allclose(same.tos, data.tos)


def test_ir_keeps_background_and_baseline_across_reanchoring():
    ir = make_ir().set_background(np.full(11, 2.0)).correct_offset((1000, 1400))

    moved = ir.move_tos_start_by(30)

    assert moved.has_background
    assert moved.has_baseline
    np.testing.assert_allclose(moved.background, ir.background)
    np.testing.assert_allclose(moved.baseline, ir.baseline)


def test_lv_tos_dimension_coordinate_stays_usable():
    lv = make_lv().move_tos_start_by(60)

    # 'tos' is LVData's dimension coord: selection must still work after the shift.
    assert lv.select_tos_range(-60, 0).n_samples == 2
    np.testing.assert_allclose(lv.tos, TOS - 60.0)


def test_ms_survives_netcdf_round_trip(tmp_path):
    ms = make_ms().move_tos_start_by(30)
    path = tmp_path / "ms.nc"
    ms.to_netcdf(path)
    loaded = MSData.from_netcdf(path)

    assert loaded.tos_start == ms.tos_start
    np.testing.assert_allclose(loaded.tos, ms.tos)
