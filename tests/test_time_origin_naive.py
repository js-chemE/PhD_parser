"""A timezone-naive origin or bound means wall-clock time in the data's own timezone, in every class.

This is what the log book gives: "14:00:30", no offset. Before this test the
three classes raised ``TypeError`` on ``with_tos_start`` and silently
dropped the timezone on ``set_tos_start``.
"""
import numpy as np
import pandas as pd
import pytest

from phd_parser.chromatography import GCData
from phd_parser.labview import LVData
from phd_parser.massspec import MSData
from phd_parser.spectroscopy import IRData

TOS = np.array([0.0, 60.0, 120.0])
TOS_START = pd.Timestamp("2026-04-21 14:00:00", tz="Europe/Amsterdam")


def make_ir():
    return IRData.from_arrays(np.linspace(1000.0, 4000.0, 11), np.ones((3, 11)), tos=TOS, tos_start=TOS_START)


def make_ms():
    return MSData.from_arrays(
        cycle=np.arange(3), channels={0: np.array([28.0, 44.0])}, values={0: np.ones((3, 2))},
        tos=TOS, tos_start=TOS_START,
    )


def make_lv():
    frame = pd.DataFrame({"timestamp": TOS_START + pd.to_timedelta(TOS, unit="s"), "Reactor T PV": [1.0, 2.0, 3.0]})
    return LVData.from_dataframe(frame, tos_start=TOS_START)


def make_gc():
    retention_time = np.arange(0, 10, 0.5)
    return GCData.from_arrays(retention_time, np.ones(retention_time.size), detector="fid", unit="pA",
                              injection_time=TOS_START)


@pytest.mark.parametrize("make", [make_ir, make_ms, make_lv, make_gc], ids=["ir", "ms", "lv", "gc"])
def test_naive_origin_is_read_in_the_data_timezone(make):
    data = make()
    rebased = data.with_tos_start("2026-04-21 14:00:30")
    assert rebased.tos_start == TOS_START + pd.Timedelta(seconds=30)
    assert rebased.tos_start.tzinfo is not None
    np.testing.assert_allclose(rebased.tos, data.tos - 30.0)
    assert list(rebased.timestamps) == list(data.timestamps)

    stamped = data.set_tos_start("2026-04-21 15:00:00")
    assert stamped.tos_start == pd.Timestamp("2026-04-21 15:00:00", tz="Europe/Amsterdam")
    np.testing.assert_allclose(stamped.tos, data.tos)


@pytest.mark.parametrize("make", [make_ir, make_ms, make_lv, make_gc], ids=["ir", "ms", "lv", "gc"])
def test_an_empty_tos_window_is_empty_not_everything(make, caplog):
    data = make()
    empty = data.select_tos_range(min_s=1e6)
    assert len(empty.tos) == 0
    assert "the data cover" in caplog.text
    with pytest.raises(ValueError, match="the data cover"):
        data.select_tos_range(max_s=-1e6, empty="raise")
    with pytest.raises(ValueError, match="the data cover"):
        data.select_timestamp_range(end="2000-01-01 00:00", empty="raise")
    # a window reaching past the data is fine
    assert len(data.select_tos_range(min_s=-1e6, max_s=1e6).tos) == len(data.tos)
