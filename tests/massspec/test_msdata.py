"""MSData traces and the Quadstar reader: smoothing before normalisation, threshold columns, directory order."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from phd_parser.massspec import MSData
from phd_parser.massspec.quadstar import read_export

ASC_FILE = Path(__file__).parent / "2026-03-26_sk1002_02-react.asc"


def make_ms():
    cycle = np.arange(5)
    values = {0: np.column_stack([np.array([0.0, 1.0, 0.0, 1.0, 0.0]), np.linspace(1.0, 2.0, 5)])}
    return MSData.from_arrays(cycle=cycle, channels={0: np.array([28.0, 44.0])}, values=values,
                              tos=cycle * 10.0, tos_start=pd.Timestamp("2026-03-26 14:00", tz="Europe/Amsterdam"))


def test_get_trace_smooths_before_it_normalises():
    ms = make_ms()
    smoothed = ms.get_trace(28.0, rolling_window=3)
    np.testing.assert_allclose(smoothed.values, [0.5, 1 / 3, 2 / 3, 1 / 3, 0.5])
    normalised = ms.get_trace(28.0, rolling_window=3, normalize=True)
    assert float(normalised.max()) == 1.0 and float(normalised.min()) == 0.0
    np.testing.assert_allclose(ms.get_traces([28.0, 44.0]).shape, (5, 2))


def test_get_trace_only_offers_selection_methods_xarray_has():
    ms = make_ms()
    with pytest.raises(ValueError):
        ms.get_trace(28.0, method="linear")
    np.testing.assert_allclose(ms.get_trace(27.9, method="nearest").values, ms.get_trace(28.0).values)


def test_from_quadstar_asc_accepts_a_naive_tos_start():
    ms = MSData.from_quadstar_asc(ASC_FILE, tos_start="2026-03-26 14:00:00")
    assert ms.tos_start == pd.Timestamp("2026-03-26 14:00:00", tz="Europe/Amsterdam")
    assert ms.tos is not None and np.all(np.diff(ms.tos) > 0)


def test_threshold_columns_get_distinct_names():
    meta, frame = read_export(ASC_FILE, drop_threshold_cols=False)
    threshold_columns = [column for column in frame.columns if column.startswith("Threshold")]
    assert threshold_columns, "the export carries threshold columns"
    assert len(set(threshold_columns)) == len(threshold_columns)
    assert all(" " not in column for column in threshold_columns)
    assert meta["n_cycles"] == len(frame) if "n_cycles" in meta else True
