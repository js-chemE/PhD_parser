"""NetCDF round trip of `MSData`, including the dict-valued attributes.

`to_netcdf` used to fail on any real export: the per-block ``channels`` table
and the ``trace_corrections`` record are dicts, which NetCDF attributes cannot
hold. They are stored as JSON strings and decoded on load.
"""
import os

import numpy as np

from phd_parser.massspec import MSData

MOCK_FILE_PATH = os.path.join(os.path.dirname(__file__), "2026-03-26_sk1002_02-react.asc")


def test_roundtrip_keeps_data_and_dict_attrs(tmp_path):
    ms = MSData.from_quadstar_asc(MOCK_FILE_PATH).correct_traces().mask_overloaded()
    target = tmp_path / "ms.nc"
    ms.to_netcdf(target)
    back = MSData.from_netcdf(target)

    mz = float(ms.ds.coords["mz"].values[0])
    np.testing.assert_allclose(back.get_trace(mz).values, ms.get_trace(mz).values)
    np.testing.assert_array_equal(back.ds.coords["mz"].values, ms.ds.coords["mz"].values)
    assert isinstance(back.ds["block_0"].attrs["channels"], dict)
    corrections = back.ds.attrs["trace_corrections"]
    assert [entry["method"] for entry in corrections] == ["min_shift", "mask_overloaded"]


def test_roundtrip_leaves_the_instance_untouched(tmp_path):
    ms = MSData.from_quadstar_asc(MOCK_FILE_PATH)
    ms.to_netcdf(tmp_path / "ms.nc")
    assert isinstance(ms.ds["block_0"].attrs["channels"], dict)
