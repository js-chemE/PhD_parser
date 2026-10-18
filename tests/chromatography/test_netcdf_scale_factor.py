"""A chromatogram with a scale factor survives NetCDF whether or not it can be packed as integer counts."""
import numpy as np

from phd_parser.chromatography import GCData


def make_gc(values):
    retention_time = np.arange(0, 5, 0.05)
    gc = GCData.from_arrays(retention_time, values, detector="fid", unit="pA", injection_time="2025-05-15 16:10:51")
    ds = gc.ds.copy()
    ds["fid"].attrs["scale_factor"] = 1.0 / 7680.0
    return GCData(ds=ds)


def test_integer_counts_are_packed_and_restored(tmp_path):
    counts = np.round(np.exp(-((np.arange(0, 5, 0.05) - 2.5) ** 2)) * 1e5)
    gc = make_gc(counts / 7680.0)
    gc.to_netcdf(tmp_path / "gc.nc")
    back = GCData.from_netcdf(tmp_path / "gc.nc")
    np.testing.assert_allclose(back.get_chromatogram("fid").values, gc.get_chromatogram("fid").values, rtol=1e-9)
    assert back.ds["fid"].attrs["scale_factor"] == 1.0 / 7680.0


def test_processed_values_with_a_scale_factor_are_not_rescaled_on_reload(tmp_path):
    processed = np.exp(-((np.arange(0, 5, 0.05) - 2.5) ** 2)) * 0.013  # no longer integer counts
    gc = make_gc(processed)
    gc.to_netcdf(tmp_path / "gc.nc")
    back = GCData.from_netcdf(tmp_path / "gc.nc")
    np.testing.assert_allclose(back.get_chromatogram("fid").values[0], processed)
    assert back.ds["fid"].attrs["scale_factor"] == 1.0 / 7680.0
    assert "counts_scale_factor" not in back.ds["fid"].attrs


def test_boolean_and_none_attrs_survive(tmp_path):
    gc = make_gc(np.ones(100))
    ds = gc.ds.copy()
    ds.attrs.update({"checked": True, "note": None, "unit_text": "25 µV"})
    gc = GCData(ds=ds)
    gc.to_netcdf(tmp_path / "gc.nc")
    back = GCData.from_netcdf(tmp_path / "gc.nc")
    assert back.ds.attrs["checked"] is True
    assert back.ds.attrs["note"] is None
    assert back.ds.attrs["unit_text"] == "25 µV"
