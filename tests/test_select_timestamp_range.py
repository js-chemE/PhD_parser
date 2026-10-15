"""``select_timestamp_range``, shared by every class that has a tos origin."""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from phd_parser.spectroscopy.infrared import IRData
from phd_parser.labview import LVData
from phd_parser.massspec import MSData

TOS_START = pd.Timestamp("2026-04-21 14:00:00", tz="Europe/Amsterdam")
TOS = np.array([0.0, 60.0, 120.0, 180.0])


def make_ir():
    return IRData.from_arrays(
        np.linspace(1000.0, 4000.0, 11),
        np.ones((4, 11)),
        tos=TOS,
        tos_start=TOS_START,
        data_type="single_beam",
    )


def make_lv():
    df = pd.DataFrame({
        "timestamp": [TOS_START + pd.Timedelta(seconds=float(t)) for t in TOS],
        "Reactor T PV": [100.0, 200.0, 300.0, 400.0],
    })
    return LVData.from_dataframe(df, tos_start=TOS_START)


def make_ms():
    return MSData.from_arrays(
        cycle=np.arange(4),
        channels={0: np.array([28.0, 44.0])},
        values={0: np.ones((4, 2))},
        tos=TOS,
        tos_start=TOS_START,
    )


BUILDERS = [pytest.param(make_ir, id="IRData"),
            pytest.param(make_lv, id="LVData"),
            pytest.param(make_ms, id="MSData")]


@pytest.mark.parametrize("build", BUILDERS)
def test_matches_the_equivalent_tos_selection(build):
    data = build()

    by_timestamp = data.select_timestamp_range(
        TOS_START + pd.Timedelta(seconds=60),
        TOS_START + pd.Timedelta(seconds=120),
    )

    np.testing.assert_allclose(by_timestamp.tos, data.select_tos_range(60, 120).tos)
    np.testing.assert_allclose(by_timestamp.tos, [60.0, 120.0])


@pytest.mark.parametrize("build", BUILDERS)
@pytest.mark.parametrize("start, end", [
    pytest.param("2026-04-21 14:01:00+02:00", "2026-04-21 14:02:00+02:00", id="str-aware"),
    pytest.param("2026-04-21 14:01:00", "2026-04-21 14:02:00", id="str-naive"),
    pytest.param(datetime(2026, 4, 21, 14, 1), datetime(2026, 4, 21, 14, 2), id="datetime-naive"),
    pytest.param(np.datetime64("2026-04-21T14:01:00"), np.datetime64("2026-04-21T14:02:00"),
                 id="datetime64-naive"),
])
def test_accepts_timestamp_string_datetime_and_datetime64(build, start, end):
    # Every form denotes the same two moments: 14:01 and 14:02 Europe/Amsterdam,
    # the naive ones because a naive bound is read in the origin's timezone.
    selected = build().select_timestamp_range(start, end)

    np.testing.assert_allclose(selected.tos, [60.0, 120.0])


@pytest.mark.parametrize("build", BUILDERS)
def test_open_ended_bounds(build):
    data = build()

    np.testing.assert_allclose(
        data.select_timestamp_range(end=TOS_START + pd.Timedelta(seconds=60)).tos,
        [0.0, 60.0],
    )
    np.testing.assert_allclose(
        data.select_timestamp_range(start=TOS_START + pd.Timedelta(seconds=120)).tos,
        [120.0, 180.0],
    )
    np.testing.assert_allclose(data.select_timestamp_range().tos, TOS)


@pytest.mark.parametrize("build", BUILDERS)
def test_survives_reanchoring_the_origin(build):
    data = build()
    window = (TOS_START + pd.Timedelta(seconds=60), TOS_START + pd.Timedelta(seconds=120))

    # with_tos_start keeps absolute timestamps, so an absolute window must too.
    before = list(data.select_timestamp_range(*window).timestamps)
    after = list(data.move_tos_start_by(90).select_timestamp_range(*window).timestamps)

    assert before == after


@pytest.mark.parametrize("build", BUILDERS)
def test_requires_an_origin(build):
    data = build().del_tos_start()

    with pytest.raises(ValueError, match="requires a tos_start"):
        data.select_timestamp_range(TOS_START, TOS_START + pd.Timedelta(seconds=60))


@pytest.mark.parametrize("build", BUILDERS)
def test_is_immutable(build):
    data = build()

    data.select_timestamp_range(TOS_START, TOS_START + pd.Timedelta(seconds=60))

    np.testing.assert_allclose(data.tos, TOS)
