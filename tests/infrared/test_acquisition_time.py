"""tos must come from the time recorded inside each .spa, not from the filename.

OMNIC filenames encode elapsed hours rounded to two decimals (0.01 h = 36 s) and
restart at zero for every new series, so a filename-derived axis cannot place two
measurements on a common timeline — which is exactly what merging needs.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from phd_parser.infrared import IRData, omnic

MOCK_DIR_PATH = Path(os.path.dirname(__file__)) / "omnic-test-data"
FILES = sorted(MOCK_DIR_PATH.glob("*.spa"), key=omnic.extract_spectrum_id)
TOS_START = pd.Timestamp("2026-03-26 12:00:00", tz="UTC")


def test_read_spa_datetime_matches_the_full_read():
    raw = omnic.read_spa(FILES[:3])

    for path, expected in zip(FILES[:3], raw["meta"]["datetime"]):
        assert pd.Timestamp(omnic.read_spa_datetime(path)) == pd.Timestamp(expected)


def test_recorded_times_are_finer_than_the_filename_hours():
    recorded = [pd.Timestamp(omnic.read_spa_datetime(f)) for f in FILES]
    from_filenames = [omnic.extract_spectrum_tos(f) for f in FILES]

    # The filenames cannot separate these scans; the recorded times can.
    assert len(set(from_filenames)) < len(from_filenames)
    assert len(set(recorded)) == len(recorded)


@pytest.mark.parametrize("backend", ["omnic", "spectrochempy"])
def test_tos_is_derived_from_the_recorded_times(backend):
    pytest.importorskip("spectrochempy") if backend == "spectrochempy" else None

    ir = IRData.from_omnic_spa(MOCK_DIR_PATH, tos_start=TOS_START, backend=backend)
    recorded = [pd.Timestamp(omnic.read_spa_datetime(f)) for f in FILES]
    expected = [(t - TOS_START).total_seconds() for t in recorded]

    np.testing.assert_allclose(np.sort(ir.tos), np.sort(expected))
    # Per-scan resolution, not a 36 s staircase.
    assert len(np.unique(ir.tos)) == len(ir.tos)


def test_both_backends_agree_on_tos():
    pytest.importorskip("spectrochempy")

    from_omnic = IRData.from_omnic_spa(MOCK_DIR_PATH, tos_start=TOS_START, backend="omnic")
    from_scp = IRData.from_omnic_spa(MOCK_DIR_PATH, tos_start=TOS_START, backend="spectrochempy")

    np.testing.assert_allclose(from_omnic.tos, from_scp.tos)
    assert from_omnic.tos_start == from_scp.tos_start


@pytest.mark.parametrize("backend", ["omnic", "spectrochempy"])
def test_two_series_read_with_the_same_tos_start_merge_cleanly(backend):
    pytest.importorskip("spectrochempy") if backend == "spectrochempy" else None

    # The restart case: two disjoint chunks, the SAME tos_start for both, and no
    # hand-recorded time for the second one anywhere.
    first = IRData.from_omnic_spa(FILES[:4], tos_start=TOS_START, backend=backend)
    second = IRData.from_omnic_spa(FILES[7:], tos_start=TOS_START, backend=backend)

    merged = first.merge(second)

    assert merged.shape[0] == len(FILES[:4]) + len(FILES[7:])
    assert np.all(np.diff(merged.tos) > 0), "merged tos must be strictly increasing"
    assert len(np.unique(merged.tos)) == len(merged.tos), "no duplicated scan times"
    # The junction sits between the two chunks, at the real recorded times.
    assert merged.tos[3] < merged.tos[4]


def test_merge_records_the_gap_between_the_segments():
    import json

    first = IRData.from_omnic_spa(FILES[:4], tos_start=TOS_START, backend="omnic")
    second = IRData.from_omnic_spa(FILES[7:], tos_start=TOS_START, backend="omnic")

    merged = first.merge(second)
    entry = json.loads(merged.ds.attrs["merge_log"])[-1]

    expected_gap = second.tos.min() - first.tos.max()
    assert entry["gap_seconds"] == pytest.approx(expected_gap)
    assert entry["rebasing"] == "none"  # these files carry no background


@pytest.mark.parametrize("backend", ["omnic", "spectrochempy"])
def test_naive_tos_start_is_rejected_clearly(backend):
    pytest.importorskip("spectrochempy") if backend == "spectrochempy" else None

    with pytest.raises(ValueError, match="timezone-aware"):
        IRData.from_omnic_spa(
            MOCK_DIR_PATH, tos_start=pd.Timestamp("2026-03-26 12:00:00"), backend=backend
        )
